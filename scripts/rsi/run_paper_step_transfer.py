"""Publish, discover and independently revalidate generated code across goals."""

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
from agent.rsi.paper_step_sessions import read
from agent.rsi.paper_step_transfer import CRITERIA, TransferredStepGoal, create_transfer_goal
from agent.science_goal import GoalSpec, GoalStore, run_goal


async def main(a):
    if a.action in {"publish", "find", "create"}:
        if not a.library:
            raise ValueError("This action requires an owner-scoped --library")
        library = GeneratedStepLibrary(a.library)
        if a.action == "publish":
            print(json.dumps({"entry_id": library.publish(a.output)}))
            return
        if not a.definition:
            raise ValueError("This action requires --definition")
        definition = read(a.definition)
        if a.action == "find":
            print(
                json.dumps(
                    [
                        {
                            "entry_id": x["entry_id"],
                            "interface": x["packet"]["interface"],
                            "origin_record_kind": x["origin_record_kind"],
                        }
                        for x in library.find(definition)
                    ],
                    indent=2,
                )
            )
            return
        if not a.profile or not a.objective:
            raise ValueError("Create requires --profile and --objective")
        store = create_transfer_goal(
            a.output,
            definition,
            Profile.from_dict(read(a.profile)),
            GoalSpec(
                a.objective,
                CRITERIA,
                max_cycles=a.max_cycles,
                model_calls=a.model_calls,
                tool_calls=a.tool_calls,
                wall_seconds=a.wall_seconds,
            ),
            library,
            a.entry_id,
            condition=a.condition,
        )
    else:
        store = GoalStore(a.output)
    if a.action == "pause":
        state = store.pause()
    elif a.action in {"create", "status"}:
        state = store.read()
    else:
        driver = TransferredStepGoal(store, None, default_runtimes())
        if a.action == "resume":
            store.resume()
        profile = Profile.from_dict(store.read()["definition"]["profile"])
        driver.credential = (
            credential_from_toml(a.credential_toml, profile.model)
            if a.credential_toml
            else os.environ.get(profile.model.credential_env)
        )
        if not driver.credential:
            raise ValueError("Model credential is unresolved")
        state = await run_goal(
            store, driver.cycle, driver.evaluate, cycle_wall_seconds=driver.protocol["cycle_wall_seconds"]
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
    p.add_argument("action", choices=["publish", "find", "create", "run", "resume", "pause", "status"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--library", type=Path)
    p.add_argument("--entry-id")
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
