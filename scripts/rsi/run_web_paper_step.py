"""Session-goal adapter: host catalog admission and automatic source publication."""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from run_harness_probe import credential_from_toml, default_runtimes

from agent.harness import Profile
from agent.rsi.generated_step_library import GeneratedStepLibrary
from agent.rsi.paper_step_catalog import task
from agent.rsi.paper_step_goal import CRITERIA, PaperStepGoal, create_step_goal
from agent.rsi.paper_step_sessions import read
from agent.rsi.paper_step_transfer import CRITERIA as TRANSFER_CRITERIA
from agent.rsi.paper_step_transfer import TransferredStepGoal, create_transfer_goal
from agent.rsi.storage import atomic_json, sha256_file
from agent.science_goal import GoalSpec, GoalStore, run_goal


def publish(store):
    state = store.read()
    library = GeneratedStepLibrary(store.root.parent / ".generated_steps")
    entry = library.publish(store.root)
    atomic_json(
        store.root / "step_publication.json",
        {
            "entry_id": entry,
            "goal_sha256": sha256_file(store.root / "goal.json"),
            "capability_id": state["progress"]["capability_id"],
            "new_scientific_result_claim": False,
        },
    )
    return entry


def prepare(args):
    item = task(ROOT, args.case)
    if item["definition_sha256"] != args.definition_sha256:
        raise ValueError("Selected host task changed before creation")
    profile = Profile.from_dict(read(args.profile))
    criteria = TRANSFER_CRITERIA if args.entry_id else CRITERIA
    spec = GoalSpec(
        args.objective,
        criteria,
        max_cycles=args.max_cycles,
        model_calls=args.model_calls,
        tool_calls=args.tool_calls,
        wall_seconds=args.wall_seconds,
    )
    if args.entry_id:
        return create_transfer_goal(
            args.output,
            item["definition"],
            profile,
            spec,
            GeneratedStepLibrary(args.output.parent / ".generated_steps"),
            args.entry_id,
        )
    return create_step_goal(args.output, item["definition"], profile, spec)


async def main(args):
    store = prepare(args) if args.action == "create" else GoalStore(args.output)
    if args.action == "publish":
        publish(store)
        state = store.read()
    elif args.action in {"create", "status"}:
        state = store.read()
    else:
        kind = store.read()["definition"].get("admission", {}).get("kind")
        cls = {
            "generated_paper_step_goal.v1": PaperStepGoal,
            "generated_paper_step_transfer.v1": TransferredStepGoal,
        }.get(kind)
        if cls is None:
            raise ValueError("Not a generated computation goal")
        driver = cls(store, None, default_runtimes())
        if args.action == "resume":
            store.resume()
        profile = Profile.from_dict(store.read()["definition"]["profile"])
        driver.credential = (
            credential_from_toml(args.credential_toml, profile.model)
            if args.credential_toml
            else os.environ.get(profile.model.credential_env)
        )
        if not driver.credential:
            raise ValueError("Selected model credential is unresolved")
        state = await run_goal(
            store, driver.cycle, driver.evaluate, cycle_wall_seconds=driver.protocol["cycle_wall_seconds"]
        )
        if state["status"] == "completed":
            publish(store)
    print(
        json.dumps(
            {
                "status": state["status"],
                "cycles": len(state["cycles"]),
                "usage": state["usage"],
                "criteria": state["assessment"],
            }
        ),
        flush=True,
    )


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["create", "run", "resume", "status", "publish"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profile", type=Path)
    p.add_argument("--objective")
    p.add_argument("--case")
    p.add_argument("--definition-sha256")
    p.add_argument("--entry-id")
    p.add_argument("--max-cycles", type=int, default=6)
    p.add_argument("--model-calls", type=int, default=144)
    p.add_argument("--tool-calls", type=int, default=72)
    p.add_argument("--wall-seconds", type=float, default=7200)
    p.add_argument("--credential-toml", type=Path)
    return p


if __name__ == "__main__":
    asyncio.run(main(parser().parse_args()))
