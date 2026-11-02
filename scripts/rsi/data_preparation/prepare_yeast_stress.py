#!/usr/bin/env python3
"""Prepare the public yeast stress paper computation without any LLM requests."""

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from agent.rsi.datasets.yeast_stress import admit_task, export_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--admit', action='store_true', help='Add the frozen definition to this research repository catalog')
    args = parser.parse_args()
    result = export_task(args.table, args.output)
    if args.admit:
        result['catalog'] = admit_task(ROOT, result['definition'])
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
