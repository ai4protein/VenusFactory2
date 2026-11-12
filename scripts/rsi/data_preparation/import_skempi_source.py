"""Stage frozen SKEMPI2 measurements without admitting tasks or fitting models."""
import argparse
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(_root), str(_root / "src")]
from agent.rsi.datasets.skempi_source import export_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--structure-manifest")
    parser.add_argument("--structure-sha256")
    args = parser.parse_args()
    result = export_source(args.manifest, args.sha256, args.output,
                           structure_manifest=args.structure_manifest, structure_sha256=args.structure_sha256)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
