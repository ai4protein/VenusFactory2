"""Bind training-only program retrieval to ordinary Scientist campaign plans."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from agent.harness.contracts import Profile, digest, file_hash

from agent.rsi import context_feedback_dispatch as feedback
from agent.rsi import library_experience, library_program_screen, model_tasks, training_attempt_memory
from agent.rsi.engine.model_access import authorize_profile
from agent.rsi.engine.model_coverage import PROGRAM_ENTRIES
from .research_batches import project_batch
from .research_project_catalog import build_catalog, campaign_plan
from agent.rsi.campaigns.scientist_model_campaign import ALL_ALLOCATIONS, admit_allocations
from agent.rsi.engine.storage import atomic_json

SCHEMA = "screened_research_batch.v1"
FIELDS = {
    "schema",
    "parent_manifest",
    "parent_catalog",
    "candidate_ids",
    "selection",
    "keep",
    "profile",
    "goal_budget",
    "generations",
    feedback.FIELD,
    training_attempt_memory.FIELD,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def binding(path):
    return {"path": str(path), "sha256": file_hash(path)}


def read(reference):
    return json.loads(model_tasks.bound_file(reference).read_text())


def export(request, output):
    """Prepare a bounded executable plan without starting goals or publishing fits.

    The declared catalog task must contain the screen evidence before the
    executed task adds its coverage-manifest reference. This preserves the
    existing exact-task coverage check, including when candidate and task IDs
    differ. Every selected recipient remains allocated, even with no usable donor.
    """
    require(isinstance(request, dict) and set(request) == FIELDS, "Invalid screened-batch request fields")
    require(request["schema"] == SCHEMA, "Unsupported screened-batch schema")
    request = copy.deepcopy(request)
    ids = request["candidate_ids"]
    require(isinstance(ids, list) and 1 <= len(ids) <= library_program_screen.MAX_TASKS, "Select 1..32 candidates")
    output = Path(output)
    require(output.is_absolute() and ".." not in output.parts, "Absolute output path required")
    require(not any(p.is_symlink() for p in [output, *output.parents]), "Output cannot traverse symlinks")
    require(not output.exists(), "Output already exists")
    manifest, prior, parent = project_batch(request["parent_manifest"], request["parent_catalog"], ids)
    screen_request = library_program_screen.declaration(
        {
            "schema": library_program_screen.SCHEMA,
            "selection": request["selection"],
            "keep": request["keep"],
            "tasks": [x["task"] for x in manifest["candidates"]],
        }
    )
    training_attempt_memory.policy_for(request)
    authorize_profile(Profile.from_dict(request["profile"]))
    # Validate task-specific policies before any screen computation or output.
    for candidate in manifest["candidates"]:
        task, _ = model_tasks.read_task(model_tasks.bound_file(candidate["task"]))
        feedback.policy_for(request, plugin_id=task["plugin_id"])
        admit_allocations(
            [
                {
                    "id": candidate["id"],
                    "task": task,
                    "profile": request["profile"],
                    "generations": request["generations"],
                    "goal_budget": request["goal_budget"],
                    feedback.FIELD: request[feedback.FIELD],
                    training_attempt_memory.FIELD: request[training_attempt_memory.FIELD],
                }
            ],
            policy=PROGRAM_ENTRIES,
        )
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "request.json", request)
    atomic_json(output / "parent_projection.json", {"manifest": manifest, "catalog": prior, "provenance": parent})
    try:
        screen = library_program_screen.export(screen_request, output / "screen")
        handoff = json.loads((output / "screen/handoff.json").read_text())
        by_task = {item["task_id"]: item for item in handoff["tasks"]}
        require(len(by_task) == len(manifest["candidates"]), "Incomplete program-screen handoff")
        selections = {}
        for candidate in manifest["candidates"]:
            original, _ = model_tasks.read_task(model_tasks.bound_file(candidate["task"]))
            adopted = by_task[original["id"]]
            screened, _ = model_tasks.read_task(model_tasks.bound_file(adopted["task"]))
            expected = copy.deepcopy(original)
            expected["references"].append(handoff["screen"])
            require(screened == expected, "Screen changed task beyond its evidence reference")
            candidate["task"] = adopted["task"]
            selections[candidate["id"]] = read(adopted["selection"])
        catalog = build_catalog(manifest)

        # All scientific descriptors and admission decisions must survive. Only
        # the hash-bound task reference changes to include retrieval evidence.
        def projection(entries):
            return [{k: v for k, v in item.items() if k != "task"} for item in entries]

        require(projection(catalog["entries"]) == projection(prior["entries"]), "Screen changed scientific admission")
        atomic_json(output / "manifest.json", manifest)
        atomic_json(output / "catalog.json", catalog)
        manifest_reference = binding(output / "manifest.json")
        plan = campaign_plan(
            manifest, catalog, request["profile"], request["goal_budget"], manifest_reference, request["generations"]
        )
        plan["execution_policy"] = ALL_ALLOCATIONS
        admissions = []
        screened_by_task = {item["task_id"]: item for item in screen["outcomes"]}
        with library_experience.audit_session():
            for allocation in plan["allocations"]:
                selection = selections[allocation["coverage"]["candidate_id"]]
                allocation["experience"] = {
                    "condition": selection["condition"],
                    "selection": "explicit",
                    "entry_ids": selection["entry_ids"],
                    **{k: selection[k] for k in ("overlap_policy", "adaptation_policy") if k in selection},
                }
                allocation[feedback.FIELD] = request[feedback.FIELD]
                allocation[training_attempt_memory.FIELD] = request[training_attempt_memory.FIELD]
                snapshot = library_experience.snapshot(selection, allocation["task"])
                source_hashes = {
                    x["entry_id"]: x["source_sha256"]
                    for x in screened_by_task[allocation["task"]["id"]]["candidates"]
                    if x["status"] == "screened"
                }
                require(
                    {x["entry_id"]: x["source_sha256"] for x in snapshot["manifest"]["donors"]}
                    == {identity: source_hashes[identity] for identity in selection["entry_ids"]},
                    "Selected source changed after screening",
                )
                admissions.append(
                    {
                        "allocation_id": allocation["id"],
                        "selected_donors": len(selection["entry_ids"]),
                        "projection_sha256": digest(snapshot["projection"]),
                        "manifest_sha256": digest(snapshot["manifest"]),
                    }
                )
        admit_allocations(plan["allocations"], policy=plan["counting_policy"])
        # Recheck the original hash-bound parent and task evidence after work.
        replay_manifest, replay_catalog, replay_parent = project_batch(
            request["parent_manifest"], request["parent_catalog"], ids
        )
        require(replay_catalog == prior and replay_parent == parent, "Parent admission changed during preparation")
        require(len(replay_manifest["candidates"]) == len(plan["allocations"]) == len(ids), "Selected task was lost")
        atomic_json(output / "source_admission.json", {"allocations": admissions})
        # A runnable plan appears only after all binding and admission checks.
        atomic_json(output / "campaign_plan.json", plan)
        receipt = {
            "schema": SCHEMA,
            "status": "ready",
            "request_sha256": digest(request),
            "implementation_sha256": file_hash(Path(__file__)),
            "library_root": request["selection"]["library_root"],
            "selected_candidates": ids,
            "allocated_tasks": len(plan["allocations"]),
            "zero_donor_tasks": sum(x["selected_donors"] == 0 for x in admissions),
            "candidate_source_tests": screen["candidate_source_tests"],
            "resources": plan["resources"],
            "outputs": {
                name: binding(output / name)
                for name in [
                    "request.json",
                    "parent_projection.json",
                    "screen/screen.json",
                    "screen/handoff.json",
                    "manifest.json",
                    "catalog.json",
                    "source_admission.json",
                    "campaign_plan.json",
                ]
            },
            "external_model_calls": 0,
            "new_acquired_models": 0,
            "new_published_models": 0,
            "validation_scored": False,
            "future_scored": False,
            "note": "Prepared retrieval and allocations; each recipient still requires native Scientist development, independent acceptance, publication and fresh reuse. Screening and archive costs are separate from the declared model-call budget.",
        }
        atomic_json(output / "preparation.json", receipt)
        return receipt
    except Exception as exc:
        atomic_json(
            output / "failure.json",
            {"schema": SCHEMA, "status": "preparation_failed", "error_type": type(exc).__name__},
        )
        raise
