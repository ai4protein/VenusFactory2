#!/usr/bin/env python3
"""Reconstruct the frozen food-population protein-response dataset locally."""

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import json

from agent.rsi.engine.yeast_population_quant import quantify_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--frozen-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    quantify_source(
        args.raw_root,
        args.frozen_root,
        args.output,
        progress=lambda status: print(json.dumps(status, sort_keys=True), flush=True),
    )


if __name__ == "__main__":
    main()
