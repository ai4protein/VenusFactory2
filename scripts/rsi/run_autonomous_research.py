#!/usr/bin/env python3
"""Detached, resumable natural-language Scientist goal runner."""
import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from agent.harness import Profile
from agent.harness.autonomous_research import AutonomousResearch, create
from agent.harness.catalog import resolve_credential
from agent.harness.installation import default_runtimes
from agent.science_goal import GoalStore


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=[
        "create", "run", "resume", "submit-pbs", "status-pbs", "cancel-pbs",
    ])
    parser.add_argument("--output", required=True, help="Scientist goal directory")
    parser.add_argument("--profile", help="Profile JSON used when creating a goal")
    parser.add_argument("--objective")
    parser.add_argument("--input", action="append", default=[], metavar="PATH",
                        help="Local input file to snapshot when creating a goal; repeat for multiple files")
    parser.add_argument("--credential-toml")
    parser.add_argument("--pbs-config", help="Administrator-owned PBS configuration")
    parser.add_argument("--resource-profile", default="gpu")
    parser.add_argument("--new-attempt", action="store_true")
    parser.add_argument("--resume-goal", action="store_true")
    for name, default in [
        ("max-cycles", 8), ("model-calls", 192),
        ("tool-calls", 96), ("wall-seconds", 7200),
    ]:
        parser.add_argument("--" + name, type=int, default=default)
    args = parser.parse_args(argv)
    if args.input and args.action != "create":
        parser.error("--input applies only to create; later attachments use research revision")
    pbs_action = args.action.endswith("-pbs")
    if pbs_action and not args.pbs_config:
        parser.error("--pbs-config is required for PBS actions")
    if args.action != "submit-pbs" and (args.new_attempt or args.resume_goal):
        parser.error("--new-attempt and --resume-goal apply only to submit-pbs")
    if pbs_action:
        from backend.compute.access import require_local_execution

        require_local_execution()
        from agent.harness.pbs_goal import Controller

        controller = Controller.from_file(args.pbs_config)
        if args.action == "submit-pbs":
            result = controller.submit(
                args.output, profile=args.resource_profile,
                resume=args.resume_goal, new_attempt=args.new_attempt,
            )
        elif args.action == "status-pbs":
            result = controller.status(args.output)
        else:
            result = controller.cancel(args.output)
        print(json.dumps(result))
        return
    if args.action == "create":
        if not args.profile or not args.objective:
            parser.error("create requires --profile and --objective")
        from agent.harness.research_cli_inputs import prepare_inputs, stage_inputs

        profile = Profile.from_dict(json.loads(Path(args.profile).read_text()))
        with prepare_inputs(args.input) as inputs:
            store = create(
                args.output, profile, args.objective,
                {key: getattr(args, key) for key in [
                    "max_cycles", "model_calls", "tool_calls", "wall_seconds",
                ]},
            )
            if inputs:
                stage_inputs(store, inputs)
        return
    from agent.harness.pbs_goal import assert_execution_allowed

    assert_execution_allowed(args.output)
    store = GoalStore(args.output)
    profile = Profile.from_dict(store.read()["definition"]["profile"])
    if args.action == "resume":
        store.resume()
    runtime = AutonomousResearch(
        store, resolve_credential(profile.model, args.credential_toml),
        default_runtimes(ROOT),
    )
    result = asyncio.run(runtime.run())
    print(json.dumps({"status": result["status"], "cycles": len(result["cycles"])}))


if __name__ == "__main__":
    main()
