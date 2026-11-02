"""Run an append-only multi-task Scientist campaign through existing goal drivers."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

from run_harness_probe import ROOT, credential_from_toml, default_runtimes
from run_science_evolution import EvolvingModelGoal
from run_science_goal import GeneratedModelGoal, prepare

from agent.harness import Profile
from agent.harness.contracts import file_hash
from agent.harness.registry import builtin_registry
from agent.rsi import training_attempt_memory
from agent.rsi.library_experience import select_for_campaign
from agent.rsi.lineage_promotion import new_evolution
from agent.rsi.model_access import authorize_profile
from agent.rsi.model_coverage import PROGRAM_ENTRIES, SCIENTIFIC_SCOPES, counting_policy
from agent.rsi.scientist_model_campaign import (
    ALL_ALLOCATIONS,
    COVERAGE_TARGET,
    ModelCampaignStore,
    execution_policy,
    prior_publications,
    process_live,
    run_campaign,
)
from agent.rsi.source_snapshot import research_implementation_paths
from agent.rsi.storage import atomic_json
from agent.science_goal import GoalStore, run_goal


def access_failure(execution):
    path = execution / "model_requests.jsonl"
    if not path.is_file():
        return None
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("event") == "response" and row.get("status") in {401, 403}:
            return {"kind": "model_access", "http_status": row["status"], "requests_sha256": file_hash(path)}
    return None


async def execute_allocation(allocation, root, credential_toml=None):
    profile = Profile.from_dict(allocation["profile"])
    authorize_profile(profile)
    check = builtin_registry().preflight(profile, default_runtimes())
    if not check["ready"]:
        raise ValueError("Campaign harness preflight failed: " + ", ".join(check["reasons"]))
    if not (root / "goal.json").exists():
        inputs = root.parent.parent / "admitted_inputs" / allocation["id"]
        atomic_json(inputs / "task.json", allocation["task"])
        atomic_json(inputs / "profile.json", profile.public())
        selection_path = None
        if "experience" in allocation:
            campaign = ModelCampaignStore(root.parent.parent)
            request, review = select_for_campaign(
                allocation["experience"],
                campaign.read()["definition"]["library_root"],
                prior_publications(campaign, allocation["id"]),
                allocation["task"],
            )
            selection_path = inputs / "experience_selection.json"
            atomic_json(selection_path, request)
            atomic_json(inputs / "experience_selection_review.json", review)
        args = SimpleNamespace(
            output=root,
            profile=inputs / "profile.json",
            task_definition=inputs / "task.json",
            objective="Develop and iteratively improve a verified model for " + allocation["task"]["id"],
            case=allocation["task"]["id"],
            source_cache=None,
            experience_selection=selection_path,
            training_baseline_feedback=allocation.get("training_baseline_feedback", "none"),
            attempt_memory=allocation.get(training_attempt_memory.FIELD, training_attempt_memory.POLICY),
            **allocation["goal_budget"],
        )
        store = prepare(args)
        if allocation["generations"] > 1:
            # Use the existing evolving-goal criterion and independently replayed
            # strict improvement policy; no alternate training path is introduced.
            from agent.harness.contracts import digest

            def configure(state):
                state["definition"]["goal"]["criteria"].append("verified_lineage")
                state["definition_sha256"] = digest(state["definition"])

            store.mutate(configure)
            path = root / "protocol.json"
            protocol = json.loads(path.read_text())
            name = "scripts/rsi/run_science_evolution.py"
            protocol["implementation"][name] = file_hash(ROOT / name)
            archived = root / "implementation" / name
            archived.parent.mkdir(parents=True, exist_ok=True)
            archived.write_bytes((ROOT / name).read_bytes())
            protocol["evolution"] = new_evolution(allocation["generations"])
            atomic_json(path, protocol)
    else:
        store = GoalStore(root)
    if training_attempt_memory.FIELD in allocation:
        protocol = json.loads((root / "protocol.json").read_text())
        if protocol.get(training_attempt_memory.FIELD) != allocation[training_attempt_memory.FIELD]:
            raise ValueError("Campaign training-attempt memory policy changed")
    state = store.read()
    campaign = ModelCampaignStore(root.parent.parent)
    if campaign.read()["status"] != "active":
        store.pause()
        return store.read()
    if state["status"] != "active":
        return state
    key = (
        credential_from_toml(credential_toml, profile.model)
        if credential_toml
        else os.environ.get(profile.model.credential_env)
    )
    if not key:
        raise ValueError("Campaign model credential is unresolved")
    driver = (GeneratedModelGoal if allocation["generations"] == 1 else EvolvingModelGoal)(store, key)

    async def cycle(index, episode_profile, state, context):
        if campaign.read()["status"] != "active":
            store.pause()
            return {"status": "interrupted", "summary": "Campaign pause observed before the next episode"}
        outcome = await driver.cycle(index, episode_profile, state, context)
        folder = store.root / "episodes" / f"cycle_{index:03d}"
        failure = access_failure(folder / "execution")
        if failure:
            outcome.update(needs_input=True, external_blocker=failure)
            atomic_json(folder / "outcome.json", outcome)
        return outcome

    options = {}
    if allocation["generations"] > 1:
        from agent.rsi.generation_publication import publisher
        from agent.rsi.scientist_model_library import ScientistModelLibrary

        library = ScientistModelLibrary(campaign.read()["definition"]["library_root"])
        options["on_checkpoint"] = publisher(
            library, root, lambda report: campaign.record_generation(allocation["id"], report)
        )
    return await run_goal(store, cycle, driver.evaluate, **options)


async def main(args):
    requested_execution = getattr(args, "execution_policy", None)
    if args.action == "create":
        plan = json.loads(args.plan.read_text())
        selected_counting = args.counting_policy or plan.get("counting_policy", SCIENTIFIC_SCOPES)
        if "counting_policy" in plan and plan["counting_policy"] != selected_counting:
            raise ValueError("Requested counting policy differs from frozen plan")
        selected_execution = requested_execution or plan.get("execution_policy", COVERAGE_TARGET)
        if "execution_policy" in plan and plan["execution_policy"] != selected_execution:
            raise ValueError("Requested execution policy differs from frozen plan")
        store = ModelCampaignStore.create(
            args.output,
            objective=args.objective,
            target_models=args.target_models,
            library_root=args.model_library,
            repository=ROOT,
            implementation={str(p.relative_to(ROOT)): file_hash(p) for p in research_implementation_paths(ROOT)},
            allocations=plan["allocations"],
            resources=plan["resources"],
            required_stages=args.require_stage,
            counting=selected_counting,
            execution=selected_execution,
        )
    else:
        store = ModelCampaignStore(args.output)
        if requested_execution is not None and requested_execution != execution_policy(store.read()["definition"]):
            raise ValueError("Requested execution policy differs from existing campaign")
    if args.action == "extend":
        plan = json.loads(args.plan.read_text())
        if "counting_policy" in plan and plan["counting_policy"] != counting_policy(store.read()["definition"]):
            raise ValueError("Extension counting policy differs from campaign")
        if "execution_policy" in plan and plan["execution_policy"] != execution_policy(store.read()["definition"]):
            raise ValueError("Extension execution policy differs from campaign")
        store.extend(plan["allocations"], resources=plan["resources"])
    elif args.action == "pause":
        store.pause()
    elif args.action == "recover":
        store.recover()
    elif args.action in {"run", "resume"}:
        for allocation in store.specifications().values():
            profile = Profile.from_dict(allocation["profile"])
            authorize_profile(profile)
            key = (
                credential_from_toml(args.credential_toml, profile.model)
                if args.credential_toml
                else os.environ.get(profile.model.credential_env)
            )
            if not key:
                raise ValueError("Campaign model credential is unresolved; no resources reserved")
            check = builtin_registry().preflight(profile, default_runtimes())
            if not check["ready"]:
                raise ValueError("Campaign harness is unavailable; no resources reserved")
        if args.action == "resume":
            store.resume()
        await run_campaign(store, lambda allocation, root: execute_allocation(allocation, root, args.credential_toml))
    elif args.action == "audit":
        print(json.dumps(store.audit_publications(), indent=2))
    state = store.read()
    counts = {}
    for item in state["allocations"].values():
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    print(
        json.dumps(
            {
                "status": state["status"],
                "allocations": counts,
                "limits": state["limits"],
                "usage": state["usage"],
                "counting_policy": counting_policy(state["definition"]),
                "execution_policy": execution_policy(state["definition"]),
                "last_verified_models": state.get("verified_scope_count", 0)
                if counting_policy(state["definition"]) == SCIENTIFIC_SCOPES
                else len(state["verified_entry_ids"]),
                "last_verified_program_entries": len(state["verified_entry_ids"]),
                "target_models": state["definition"]["target_models"],
                "runner_live": process_live(state["owner"] or (state["inflight"] or {}).get("owner")),
                "inflight": state["inflight"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["create", "extend", "run", "pause", "resume", "recover", "status", "audit"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--objective")
    parser.add_argument("--target-models", type=int)
    parser.add_argument("--require-stage", action="append", default=[])
    parser.add_argument("--model-library", type=Path)
    parser.add_argument("--credential-toml", type=Path)
    parser.add_argument(
        "--counting-policy",
        choices=[SCIENTIFIC_SCOPES, PROGRAM_ENTRIES],
        help="New campaigns default to scientific scopes; program-entry counting is for explicit legacy reproduction",
    )
    parser.add_argument(
        "--execution-policy",
        choices=[COVERAGE_TARGET, ALL_ALLOCATIONS],
        help="Use all_allocations.v1 for prospective comparisons; acquisition defaults to coverage_target.v1",
    )
    asyncio.run(main(parser.parse_args()))
