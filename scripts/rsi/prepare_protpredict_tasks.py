"""Prepare source-bound laboratory yeast tasks without fitting or model calls."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from agent.rsi.protpredict_tasks import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="JSON declaration of bound matrix, partition plan and license"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(json.loads(args.source.read_text()), args.output), indent=2))


if __name__ == "__main__":
    main()
