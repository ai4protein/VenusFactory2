"""Publish, inspect and use independently verified Scientist models."""

import argparse
import json
from pathlib import Path

from agent.rsi.scientist_model_library import ScientistModelLibrary
from agent.rsi.storage import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    commands = parser.add_subparsers(dest="action", required=True)
    publish = commands.add_parser("publish")
    publish.add_argument("--goal", type=Path, required=True)
    publish.add_argument("--model-id", required=True)
    whole_goal = commands.add_parser("publish-goal")
    whole_goal.add_argument("--goal", type=Path, required=True)
    migration = commands.add_parser("migrate")
    migration.add_argument("--goal", type=Path, required=True)
    migration.add_argument("--model-id", required=True)
    listing = commands.add_parser("list")
    listing.add_argument("--verify", action="store_true")
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--entry-id", required=True)
    infer = commands.add_parser("predict")
    infer.add_argument("--entry-id", required=True)
    infer.add_argument("--inputs", type=Path, required=True)
    infer.add_argument("--scope", type=Path, required=True)
    infer.add_argument("--session-id", required=True)
    infer.add_argument("--mode", choices=("science_agent", "science_expert"), default="science_agent")
    infer.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    library = ScientistModelLibrary(args.library)
    if args.action == "publish":
        result = library.publish(args.goal, args.model_id)
    elif args.action == "publish-goal":
        result = library.publish_goal(args.goal)
    elif args.action == "migrate":
        result = library.migrate(args.goal, args.model_id)
    elif args.action == "list":
        result = library.catalog(verify=args.verify)
    elif args.action == "inspect":
        verified = library.resolve(args.entry_id)
        result = {k: verified[k] for k in ("entry_id", "key", "origin", "skill")}
    else:
        if args.output.exists():
            raise ValueError("Prediction output already exists")
        result = library.predict(
            args.entry_id,
            json.loads(args.inputs.read_text()),
            json.loads(args.scope.read_text()),
            session_id=args.session_id,
            mode=args.mode,
        )
        atomic_json(args.output, result)
        result = {"output": str(args.output), "rows": result["receipt"]["rows"], "receipt_id": result["receipt_id"]}
    print(json.dumps(result, indent=2))
    if args.action == "publish-goal" and not result["all_published"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
