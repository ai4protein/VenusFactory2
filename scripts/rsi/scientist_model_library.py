"""Publish, inspect and use independently verified Scientist models."""

import argparse
import json
from pathlib import Path

from agent.rsi.engine.scientist_model_library import ScientistModelLibrary
from agent.rsi.engine.storage import atomic_json, file_lock


def _write_new_output(path, value, label):
    # Concurrent audits must not replace another invocation's completed report.
    with file_lock(path.with_name(f".{path.name}.lock")):
        if path.exists() or path.is_symlink():
            raise ValueError(f"{label} output already exists")
        atomic_json(path, value)


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
    verify = commands.add_parser("verify", help="Replay current-runtime training and provenance for 1..256 model IDs")
    verify.add_argument("--entry-ids", type=Path, required=True, help="JSON array of registered entry IDs")
    verify.add_argument("--output", type=Path, required=True)
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
    elif args.action == "verify":
        if args.output.exists():
            raise ValueError("Verification output already exists")
        values = library.resolve_many(json.loads(args.entry_ids.read_text()), replay_training=True)
        report = {
            "schema": "scientist_library_verification.v1",
            "verified_entries": len(values),
            "models": [{k: row[k] for k in ("entry_id", "key", "origin", "skill")} for row in values.values()],
            "training_replayed": True,
            "external_model_calls": 0,
            "new_autonomous_models": 0,
            "final_labels_scored": False,
        }
        _write_new_output(args.output, report, "Verification")
        result = {"output": str(args.output), "verified_entries": len(values), "training_replayed": True}
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
        _write_new_output(args.output, result, "Prediction")
        result = {"output": str(args.output), "rows": result["receipt"]["rows"], "receipt_id": result["receipt_id"]}
    print(json.dumps(result, indent=2))
    if args.action == "publish-goal" and not result["all_published"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
