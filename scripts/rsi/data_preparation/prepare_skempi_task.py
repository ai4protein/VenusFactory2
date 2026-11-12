"""Prepare one grouped binding-ddG task without training, publishing or model calls."""
import argparse
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(_root), str(_root / "src")]
from agent.rsi.datasets.skempi_task import export_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    protocol = export_protocol(args.audit, args.sha256, args.output)
    print(json.dumps({key: protocol[key] for key in ("schema", "state", "cohorts", "excluded_rows", "trained_models")}, indent=2))


if __name__ == "__main__":
    main()
