#!/usr/bin/env python3
"""Freeze, assess or replay a terminal shared Scientist campaign."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from agent.rsi.engine import shared_campaign_evaluation as evaluation
from agent.rsi.engine import shared_campaign_study as study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    admission = commands.add_parser("admit")
    admission.add_argument("--campaign", required=True)
    admission.add_argument("--library", required=True)
    admission.add_argument("--output", required=True)
    for name in ("evaluate", "audit"):
        command = commands.add_parser(name)
        command.add_argument("--study", required=True)
        command.add_argument("--output", required=True)
        if name == "evaluate":
            command.add_argument("--resume", action="store_true")
        else:
            command.add_argument("--evaluation", required=True)
    args = parser.parse_args()
    if args.command == "admit":
        value = study.admit(args.campaign, args.library, args.output)
        result = {
            "status": "admitted",
            "allocated": len(value["snapshot"]["tasks"]),
            "indexed_models": sum(len(row["models"]) for row in value["snapshot"]["publications"].values()),
            "final_labels_decoded": False,
        }
    elif args.command == "evaluate":
        value = evaluation.evaluate(
            args.study, args.output, resume=args.resume, progress=lambda row: print(json.dumps(row), flush=True)
        )
        result = {
            key: value[key]
            for key in ("status", "allocated", "future_scored", "verified_callable", "controls_completed")
        }
    else:
        result = evaluation.audit(args.study, args.evaluation, args.output)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
