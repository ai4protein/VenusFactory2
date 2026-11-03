#!/usr/bin/env python3
"""Publish an operator-selected, independently reviewed Scientist deliverable."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="Reviewed candidate directory containing manifest.json and admission.json")
    parser.add_argument("--goal-id", required=True)
    parser.add_argument("--author-email", required=True, help="Author email displayed in the platform catalog")
    parser.add_argument("--curator", required=True, help="Identity of the operator approving publication")
    args = parser.parse_args()
    from agent.harness.curated_publication import publish_candidate
    result = publish_candidate(args.bundle, args.goal_id, author_email=args.author_email, curator=args.curator)
    print(json.dumps({k: result[k] for k in ("id", "name", "version", "author_email", "visibility")}, indent=2))


if __name__ == "__main__":
    main()
