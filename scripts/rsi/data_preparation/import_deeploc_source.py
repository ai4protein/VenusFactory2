"""Normalize a frozen DeepLoc snapshot without training or admitting models."""

import argparse
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(_root), str(_root / "src")]

from agent.rsi.datasets.deeploc_source import export_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = export_source(args.manifest, args.sha256, args.output)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
