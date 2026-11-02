"""Persistent paper-step development and verified reuse through interchangeable harnesses."""

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
from agent.rsi.campaigns.paper_step_goal import CRITERIA, PaperStepGoal, create_step_goal
from agent.rsi.engine.paper_step_sessions import read
from agent.science_goal import GoalSpec, GoalStore, run_goal


async def main(args):
    if args.action == "create":
        if args.definition is None or args.profile is None or not args.objective:
            raise ValueError("create requires definition, profile and objective")
        spec = GoalSpec(
            args.objective,
            CRITERIA,
            max_cycles=args.max_cycles,
            model_calls=args.model_calls,
            tool_calls=args.tool_calls,
            wall_seconds=args.wall_seconds,
        )
        store = create_step_goal(
            args.output, read(args.definition), Profile.from_dict(read(args.profile)), spec, condition=args.condition
        )
    else:
        store = GoalStore(args.output)
    if args.action == "pause":
        state = store.pause()
    elif args.action in {"create", "status"}:
        state = store.read()
    else:
        if args.action == "resume":
            store.resume()
        profile = Profile.from_dict(store.read()["definition"]["profile"])
        plugin = PaperStepGoal(store, None, default_runtimes())
        key = (
            credential_from_toml(args.credential_toml, profile.model)
            if args.credential_toml
            else os.environ.get(profile.model.credential_env)
        )
        if not key:
            raise ValueError("Required model credential is unresolved")
        plugin.credential = key
        state = await run_goal(
            store, plugin.cycle, plugin.evaluate, cycle_wall_seconds=plugin.protocol["cycle_wall_seconds"]
        )
    print(
        json.dumps(
            {
                "status": state["status"],
                "cycles": len(state["cycles"]),
                "usage": state["usage"],
                "criteria": state["assessment"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["create", "run", "resume", "pause", "status"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--definition", type=Path)
    p.add_argument("--profile", type=Path)
    p.add_argument("--objective")
    p.add_argument(
        "--condition",
        choices=["science_goal", "persistent_agent_ablation", "scripted_agent_turns"],
        default="science_goal",
    )
    p.add_argument("--max-cycles", type=int, default=6)
    p.add_argument("--model-calls", type=int, default=144)
    p.add_argument("--tool-calls", type=int, default=72)
    p.add_argument("--wall-seconds", type=float, default=7200)
    p.add_argument("--credential-toml", type=Path)
    asyncio.run(main(p.parse_args()))
