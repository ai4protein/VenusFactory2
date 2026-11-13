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
        "submit-cluster", "status-cluster", "cancel-cluster",
    ])
    parser.add_argument("--output", required=True, help="Scientist goal directory")
    parser.add_argument("--profile", help="Profile JSON used when creating a goal")
    parser.add_argument("--objective")
    parser.add_argument("--input", action="append", default=[], metavar="PATH",
                        help="Local input file to snapshot when creating a goal; repeat for multiple files")
    parser.add_argument("--credential-toml")
    configs = parser.add_mutually_exclusive_group()
    configs.add_argument("--pbs-config", help="Legacy administrator-owned PBS configuration")
    configs.add_argument("--cluster-config", help="Administrator-owned local PBS or Slurm configuration")
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
    cluster_action = args.action.endswith(("-pbs", "-cluster"))
    config = args.cluster_config or args.pbs_config
    if cluster_action and not config:
        parser.error("--cluster-config (or legacy --pbs-config) is required for cluster actions")
    if args.action not in {"submit-pbs", "submit-cluster"} and (args.new_attempt or args.resume_goal):
        parser.error("--new-attempt and --resume-goal apply only to cluster submission")
    if cluster_action:
        from backend.compute.access import require_local_execution

        require_local_execution()
        from agent.harness.pbs_goal import Controller

        controller = Controller.from_file(config)
        if args.action in {"submit-pbs", "submit-cluster"}:
            result = controller.submit(
                args.output, profile=args.resource_profile,
                resume=args.resume_goal, new_attempt=args.new_attempt,
            )
        elif args.action in {"status-pbs", "status-cluster"}:
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
