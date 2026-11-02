"""Materialize audited Y3K scientific tasks without fitting or publishing models."""

import argparse
import json
from pathlib import Path

from agent.rsi.y3k_tasks import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(json.loads(args.source.read_text()), args.output)
    print(json.dumps({k: v for k, v in report.items() if k != "excluded"}, indent=2))


if __name__ == "__main__":
    main()
