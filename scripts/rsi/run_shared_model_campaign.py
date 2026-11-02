"""Run persistent shared-program acquisition through existing Scientist goals."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from run_harness_probe import ROOT
from run_shared_science_goal import SharedScientistGoal, prepare

from agent.harness import Profile
from agent.harness.contracts import file_hash
from agent.rsi import shared_campaign_experience
from agent.rsi.model_access import authorize_profile
from agent.rsi.shared_model_campaign import CRITERIA, SharedCampaignStore, bind_goal, read, require, run_campaign
from agent.rsi.source_snapshot import research_implementation_paths
from agent.science_goal import GoalSpec, GoalStore, run_goal


def prepare_campaign(output, plan, library_root):
    require(
        set(plan) == {"objective", "target_scopes", "required_stages", "excluded_scopes", "resources", "batches"},
        "Invalid shared campaign plan",
    )
    for batch in plan["batches"]:
        authorize_profile(Profile.from_dict(batch["profile"]))
    implementation = {
        path.relative_to(ROOT).as_posix(): file_hash(path) for path in research_implementation_paths(ROOT)
    }
    return SharedCampaignStore.create(
        output, library_root=library_root, repository=ROOT, implementation=implementation, **plan
    )


async def execute_campaign(store):
    state = store.read()
    require(
        Path(state["definition"]["repository"]).resolve() == ROOT.resolve(), "Run from the frozen campaign repository"
    )

    for name, batch in store.specifications(state).items():
        if state["batches"][name]["status"] == "queued":
            profile = Profile.from_dict(batch["profile"])
            authorize_profile(profile)
            require(
                bool(os.environ.get(profile.model.credential_env)),
                "Declared model credential unavailable before reservation; no fallback",
            )

    async def execute(batch, folder):
        profile = Profile.from_dict(batch["profile"])
        authorize_profile(profile)
        credential = os.environ.get(profile.model.credential_env)
        require(bool(credential), "Declared model credential unavailable; no fallback")
        if (folder / "goal.json").exists():
            child = GoalStore(folder)
        else:
            require(not folder.exists(), "Partial shared goal preparation requires review")
            goal = GoalSpec(state["definition"]["objective"], CRITERIA, **batch["goal_budget"])
            selection = (
                shared_campaign_experience.select_for_batch(store, batch)
                if shared_campaign_experience.FIELD in batch
                else None
            )
            child = prepare(
                folder,
                batch["tasks"],
                profile,
                goal,
                training_baseline_feedback=batch["training_baseline_feedback"],
                shared_training_history=batch["shared_training_history"],
                library_selection=selection,
            )
        if shared_campaign_experience.FIELD in batch:
            bind_goal(store, batch, folder)
        driver = SharedScientistGoal(child, credential)
        if store.read()["status"] == "paused":
            child.pause()
        return await run_goal(child, driver.cycle, driver.evaluate, stop_on_uncheckpointed_failure=True)

    await run_campaign(store, execute)
    return store.read()


async def main(args):
    if args.action == "create":
        require(args.plan is not None and args.model_library is not None, "Create requires a plan and model library")
        store = prepare_campaign(args.output, read(args.plan), args.model_library)
    else:
        store = SharedCampaignStore(args.output)
    if args.action == "extend":
        require(args.plan is not None, "Extend requires a plan")
        plan = read(args.plan)
        require(set(plan) == {"batches", "resources"}, "An extension adds batches and resource budgets only")
        for batch in plan["batches"]:
            authorize_profile(Profile.from_dict(batch["profile"]))
        store.extend(**plan)
    elif args.action == "pause":
        store.pause()
    elif args.action == "recover":
        store.recover()
    elif args.action == "audit":
        print(json.dumps(store.audit_publications()), flush=True)
        return
    elif args.action in {"run", "resume"}:
        if args.action == "resume":
            store.resume()
        await execute_campaign(store)
    state = store.read(verify_source=args.action != "pause")
    print(
        json.dumps(
            {
                "status": state["status"],
                "batches": {name: item["status"] for name, item in state["batches"].items()},
                "usage": state["usage"],
                "limits": state["limits"],
                "last_audit": state["last_audit"],
                "target_scopes": state["definition"]["target_scopes"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["create", "extend", "run", "pause", "resume", "recover", "status", "audit"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--model-library", type=Path)
    asyncio.run(main(parser.parse_args()))
