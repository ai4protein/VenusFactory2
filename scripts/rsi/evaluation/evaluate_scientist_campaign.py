"""Admit a prospective study or evaluate a terminal Scientist model campaign."""

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import json
from pathlib import Path

from agent.rsi.engine.campaign_evaluation import admit, admit_descriptive, audit, evaluate


def main(args):
    if args.action == "admit":
        pairs = json.loads(args.pairs.read_text()) if args.pairs else []
        result = admit(args.campaign, args.output, pairs=pairs, allocation_ids=args.allocation_id)
        print(
            json.dumps(
                {
                    "status": "admitted",
                    "allocations": len(result["allocations"]),
                    "comparisons": len(result["pairs"]),
                    "model_calls": 0,
                }
            )
        )
    elif args.action == "admit-descriptive":
        result = admit_descriptive(args.campaign, args.output)
        print(
            json.dumps(
                {
                    "status": "admitted",
                    "protocol": result["admission_protocol"],
                    "allocations": len(result["allocations"]),
                    "model_calls": 0,
                }
            )
        )
    elif args.action == "evaluate":
        result = evaluate(args.study, args.output)
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "status",
                        "allocated",
                        "publication_and_reuse_successes",
                        "future_scored",
                        "external_model_calls",
                        "new_autonomous_models",
                    )
                }
            )
        )
        if result["status"] != "completed":
            raise SystemExit(1)
    else:
        print(json.dumps(audit(args.study, args.evaluation, args.output)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    admission = commands.add_parser("admit")
    admission.add_argument("--campaign", type=Path, required=True)
    admission.add_argument("--pairs", type=Path)
    admission.add_argument("--allocation-id", action="append")
    admission.add_argument("--output", type=Path, required=True)
    descriptive = commands.add_parser(
        "admit-descriptive", help="Freeze every terminal acquisition for descriptive scoring"
    )
    descriptive.add_argument("--campaign", type=Path, required=True)
    descriptive.add_argument("--output", type=Path, required=True)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--study", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    auditing = commands.add_parser("audit")
    auditing.add_argument("--study", type=Path, required=True)
    auditing.add_argument("--evaluation", type=Path, required=True)
    auditing.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
