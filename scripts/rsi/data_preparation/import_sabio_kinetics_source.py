"""Import reviewed plant kinetics into observations, without launching models."""

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import json

from agent.rsi.datasets.sabio_kinetics_source import export_source


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
