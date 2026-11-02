#!/usr/bin/env python3
"""Prepare the public yeast stress paper computation without any LLM requests."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from agent.rsi.yeast_stress import admit_task, export_task


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
