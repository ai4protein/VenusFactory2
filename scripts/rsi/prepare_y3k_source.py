"""Reconstruct the source-bound Y3K numerical matrices without model execution."""

import argparse
import json
from pathlib import Path

from agent.rsi.y3k_source import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(json.loads(args.source.read_text()), args.output)
    print(json.dumps({"status": report["status"], "conditions": report["conditions"], "models_trained": 0}, indent=2))


if __name__ == "__main__":
    main()
