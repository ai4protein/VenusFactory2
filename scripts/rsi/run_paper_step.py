#!/usr/bin/env python3
"""Manage an extensible paper calculation under a private host admission.

This command does not call an LLM and never labels controller-authored source
as autonomous code generation. Reference outputs must stay outside agent mounts.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from agent.rsi.engine.paper_step import PaperStepLifecycle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--definition", type=Path, required=True)
    parser.add_argument(
        "--action",
        choices=["inspect", "propose", "assess", "register", "capability", "reuse", "verify-reuse"],
        required=True,
    )
    parser.add_argument("--source", type=Path)
    parser.add_argument("--declared-origin")
    parser.add_argument("--identity")
    parser.add_argument("--case")
    args = parser.parse_args()
    lifecycle = PaperStepLifecycle(args.root, json.loads(args.definition.read_text()))
    if args.action == "inspect":
        result = lifecycle.public_task()
    elif args.action == "propose":
        if args.source is None or args.declared_origin is None:
            parser.error("propose requires --source and --declared-origin")
        result = {"candidate_id": lifecycle.propose(args.source.read_text(), declared_origin=args.declared_origin)}
    elif args.action == "assess":
        result = lifecycle.assess(args.identity)
    elif args.action == "register":
        result = {"capability_id": lifecycle.register(args.identity)}
    elif args.action == "capability":
        result = lifecycle.capability(args.identity)
    elif args.action == "reuse":
        result = lifecycle.reuse(args.identity, args.case)
    else:
        result = lifecycle.verify_reuse(args.identity, args.case)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 2 if result.get("passed") is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
