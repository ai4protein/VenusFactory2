#!/usr/bin/env python3
"""Freeze the yeast food-population source protocol without reading outcomes."""

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import json

from agent.rsi.datasets.yeast_population import freeze_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(freeze_source(args.raw_root, args.output), indent=2))


if __name__ == "__main__":
    main()
