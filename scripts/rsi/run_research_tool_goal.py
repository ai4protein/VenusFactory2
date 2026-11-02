"""Create/run a typed, library-first Scientist tool goal; DMX free models only."""

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
from agent.rsi.model_access import authorize_profile
from agent.rsi.paper_step_sessions import read
from agent.rsi.research_tool_goal import create_research_tool_goal, research_tool_driver
from agent.science_goal import GoalStore, run_goal


async def main(args):
    if args.action == "create":
        if not all((args.definition, args.contract, args.profile, args.objective, args.library)):
            raise ValueError("create requires definition, contract, profile, objective and library")
        profile = Profile.from_dict(read(args.profile))
        authorize_profile(profile)
        store = create_research_tool_goal(
            args.output,
            read(args.definition),
            read(args.contract),
            read(args.policy),
            profile,
            args.objective,
            args.library,
            entry_id=args.entry_id,
            max_cycles=args.max_cycles,
            model_calls=args.model_calls,
            tool_calls=args.tool_calls,
            wall_seconds=args.wall_seconds,
        )
    else:
        store = GoalStore(args.output)
    if args.action == "publish":
        if args.library is None:
            raise ValueError("publish requires library")
        driver = research_tool_driver(store, None, {})
        state = store.read()
        if driver.evaluate(state) != state["assessment"] or state["status"] != "completed":
            raise ValueError("Only independently completed research tools may publish")
        origin_profile = Profile.from_dict(state["definition"]["profile"])
        if origin_profile.model.provider.lower() in {"test", "test_fixture", "fixture", "mock"}:
            raise ValueError("Simulated records cannot publish a research tool")
        entry = GeneratedStepLibrary(args.library).publish(store.root)
        print(json.dumps({"entry_id": entry, "library": str(args.library)}))
        return
    if args.action == "pause":
        state = store.pause()
    elif args.action in {"create", "status"}:
        state = store.read()
    else:
        profile = Profile.from_dict(store.read()["definition"]["profile"])
        authorize_profile(profile)  # Before credentials, resume or budget reservation.
        driver = research_tool_driver(store, None, default_runtimes())
        key = (
            credential_from_toml(args.credential_toml, profile.model)
            if args.credential_toml
            else os.environ.get(profile.model.credential_env)
        )
        if not key:
            raise ValueError("Required model credential is unresolved")
        if args.action == "resume":
            store.resume()
        driver.credential = key
        state = await run_goal(
            store, driver.cycle, driver.evaluate, cycle_wall_seconds=driver.protocol["cycle_wall_seconds"]
        )
    protocol = read(store.root / "protocol.json")
    print(
        json.dumps(
            {
                "status": state["status"],
                "cycles": len(state["cycles"]),
                "usage": state["usage"],
                "discovery": protocol["scientist_tool_context"]["discovery"],
                "criteria": state["assessment"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["create", "run", "resume", "pause", "status", "publish"])
    p.add_argument("--output", required=True, type=Path)
    for name in ("definition", "contract", "profile", "library", "credential-toml"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--policy", type=Path, default=ROOT / "configs/rsi/scientist_tool_policy_v1.json")
    p.add_argument("--objective")
    p.add_argument("--entry-id")
    p.add_argument("--max-cycles", type=int, default=6)
    p.add_argument("--model-calls", type=int, default=144)
    p.add_argument("--tool-calls", type=int, default=72)
    p.add_argument("--wall-seconds", type=float, default=7200)
    asyncio.run(main(p.parse_args()))
