"""Extract source-bound maize observations; this does not launch model tasks."""

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import json

from agent.rsi.datasets.maize_population_source import export_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = export_source(args.source, args.output)
    print(json.dumps({"summary": manifest["summary"], "policy": manifest["policy"]}, indent=2))


if __name__ == "__main__":
    main()
