"""Freeze a sequence-only localization task with original DeepLoc partitions."""

import argparse
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(_root), str(_root / "src")]

from agent.rsi.datasets.deeploc_protocol import export_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Curated source.json")
    parser.add_argument("--sha256", required=True, help="Frozen curated source.json SHA-256")
    parser.add_argument("--output", required=True, help="New host-only staging directory")
    args = parser.parse_args()
    protocol = export_protocol(args.source, args.sha256, args.output)
    print(json.dumps({key: protocol[key] for key in ("schema", "cohorts", "external_hpa", "trained_models")}, indent=2))


if __name__ == "__main__":
    main()
