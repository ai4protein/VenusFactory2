"""Training-only screening of verified programs before new Scientist goals.

This is a bounded retrieval computation, not autonomous acquisition or model
publication. It exports ordinary source selections for the existing goal driver.
The recipient must still develop, qualify and freshly reuse its own model.
"""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path

from agent.harness.contracts import digest, file_hash

from agent.rsi import context_interfaces, library_experience, model_tasks
from .code_sandbox import CandidateCodeError, ModelOutputError
from agent.rsi.datasets.research_project_catalog import scope_key
from .storage import atomic_json

SCHEMA = "library_program_screen.v1"
MAX_TASKS = 32
SEED = 314159
RANKING = "descending_training_holdout_skill_then_entry_id"


def require(value, message):
    if not value:
        raise ValueError(message)


def declaration(request):
    require(
        isinstance(request, dict) and set(request) == {"schema", "selection", "tasks", "keep"},
        "Invalid program-screen request fields",
    )
    require(request["schema"] == SCHEMA, "Unsupported program-screen schema")
    selection = library_experience.selection(request["selection"])
    require(selection["condition"] == "source" and selection["entry_ids"], "Screening requires explicit sources")
    require(type(request["keep"]) is int and 1 <= request["keep"] <= len(selection["entry_ids"]), "Invalid donor limit")
    require(
        isinstance(request["tasks"], list) and 1 <= len(request["tasks"]) <= MAX_TASKS, "Declare 1..32 target tasks"
    )
    return copy.deepcopy(request)


def _targets(request):
    result, ids, scopes = [], set(), set()
    for binding in request["tasks"]:
        task, data = model_tasks.read_task(model_tasks.bound_file(binding))
        require(
            task["plugin_id"] in context_interfaces.PLUGINS, "Program screening currently requires contextual tasks"
        )
        # Different names and repeated fits of the same measured scope do not
        # create additional recipients in a screen.
        scope = task["scope"]
        key = scope_key(
            {
                "protein_scope": {"kind": "protein", "identifier": scope["protein"]},
                "property": scope["target"],
                "scenario": scope["context"],
            },
            task,
        )
        require(task["id"] not in ids and key not in scopes, "Repeated target task or scientific scope")
        ids.add(task["id"])
        scopes.add(key)
        api = context_interfaces.module_for(task["plugin_id"])
        fit_rows, feedback_rows = api.development_split(data["training"])
        result.append((binding, task, data["training"], fit_rows, feedback_rows))
    return result


def _evaluate(task, source, fit_rows, feedback_rows):
    api = context_interfaces.module_for(task["plugin_id"])
    model = api.fit_candidate(fit_rows, task["scope"], {"source_code": source, "seed": SEED})
    metrics = api.evaluate(model, feedback_rows, task["scope"])
    skill = metrics[api.ACCEPTANCE_METRIC]
    require(type(skill) in (float, int) and math.isfinite(skill), "Nonfinite program-screen score")
    # Only summaries cross this boundary. Discard fitted parameters and raw
    # predictions; this result cannot be submitted as a trained candidate.
    return {
        "rows": metrics["rows"],
        "independent_groups": metrics["independent_groups"],
        "rmse": metrics["rmse"],
        "training_mean_rmse": metrics["training_mean_rmse"],
        "mse_skill_vs_training_mean": skill,
        "predictions_sha256": metrics["predictions_sha256"],
    }


def screen(request):
    started = time.monotonic()
    request = declaration(request)
    targets = _targets(request)
    outcomes, attempts = [], 0
    with library_experience.audit_session():
        for binding, task, training, fit_rows, feedback_rows in targets:
            candidates = []
            for identity in sorted(request["selection"]["entry_ids"]):
                selection = {**request["selection"], "entry_ids": [identity]}
                record = {"entry_id": identity}
                try:
                    packet = library_experience.snapshot(selection, task)
                except library_experience.IneligibleDonor as exc:
                    record.update(status="ineligible", reason=str(exc))
                    candidates.append(record)
                    continue
                entry = packet["projection"]["entries"][0]
                record.update(source_sha256=entry["source_sha256"], provenance_sha256=digest(packet["manifest"]))
                attempts += 1
                try:
                    feedback = _evaluate(task, entry["source_code"], fit_rows, feedback_rows)
                except (CandidateCodeError, ModelOutputError) as exc:
                    # Candidate failures are retained; corrupt evidence and
                    # invalid host declarations remain fatal rather than being
                    # silently treated as a low-performing program.
                    record.update(status="execution_failed", error_type=type(exc).__name__)
                else:
                    record.update(status="screened", training_feedback=feedback)
                candidates.append(record)
            ranked = sorted(
                (v for v in candidates if v["status"] == "screened"),
                key=lambda v: (-v["training_feedback"]["mse_skill_vs_training_mean"], v["entry_id"]),
            )
            selected = [v["entry_id"] for v in ranked[: request["keep"]]]
            outcomes.append(
                {
                    "task_id": task["id"],
                    "task": binding,
                    "admission": model_tasks.admission(task),
                    "plugin_id": task["plugin_id"],
                    "plugin_sha256": context_interfaces.module_for(task["plugin_id"]).implementation_hash(),
                    "training_sha256": digest(training),
                    "fit_sha256": digest(fit_rows),
                    "feedback_sha256": digest(feedback_rows),
                    "training_rows": len(training),
                    "fit_rows": len(fit_rows),
                    "feedback_rows": len(feedback_rows),
                    "fit_groups": len({r["group"] for r in fit_rows}),
                    "feedback_groups": len({r["group"] for r in feedback_rows}),
                    "candidates": candidates,
                    "selection": {**request["selection"], "entry_ids": selected},
                }
            )
    # Recheck target bindings after screening before allowing any handoff.
    for binding, task, *_ in targets:
        current, _ = model_tasks.read_task(model_tasks.bound_file(binding))
        require(current == task, "Target changed during program screening")
    return {
        "schema": SCHEMA,
        "request": request,
        "request_sha256": digest(request),
        "implementation_sha256": file_hash(Path(__file__)),
        "selection_data": "group_holdout_within_recipient_training_partition",
        "ranking": RANKING,
        "seed": SEED,
        "outcomes": outcomes,
        "candidate_source_tests": attempts,
        "maximum_candidate_source_tests": len(targets) * len(request["selection"]["entry_ids"]),
        "wall_seconds": time.monotonic() - started,
        "external_model_calls": 0,
        "new_acquired_models": 0,
        "new_published_models": 0,
        "validation_scored": False,
        "future_scored": False,
        "limitation": "Retrieval feedback reuses an internal training holdout and can overfit. Successful execution or rank does not imply target acceptance, biological transfer or autonomous improvement. Include screening and archive costs in comparisons.",
    }


def export(request, output):
    output = Path(output)
    require(output.is_absolute() and ".." not in output.parts, "Absolute output path required")
    require(not any(p.is_symlink() for p in [output, *output.parents]), "Output cannot traverse symlinks")
    require(not output.exists(), "Output already exists")
    report = screen(request)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "screen.json", report)
    evidence = {"path": str(output / "screen.json"), "sha256": file_hash(output / "screen.json")}
    handoffs = []
    for result in report["outcomes"]:
        task, _ = model_tasks.read_task(model_tasks.bound_file(result["task"]))
        require(model_tasks.admission(task) == result["admission"], "Screened target admission changed")
        task["references"].append(evidence)
        folder = output / "tasks" / task["id"]
        atomic_json(folder / "task.json", task)
        atomic_json(folder / "selection.json", result["selection"])
        # Exercise the same input parser used by run_science_goal.prepare.
        model_tasks.read_task(folder / "task.json")
        library_experience.selection(result["selection"])
        handoffs.append(
            {
                "task_id": task["id"],
                "task": {"path": str(folder / "task.json"), "sha256": file_hash(folder / "task.json")},
                "selection": {"path": str(folder / "selection.json"), "sha256": file_hash(folder / "selection.json")},
                "selected_donors": len(result["selection"]["entry_ids"]),
            }
        )
    atomic_json(output / "handoff.json", {"schema": SCHEMA, "screen": evidence, "tasks": handoffs})
    return report
