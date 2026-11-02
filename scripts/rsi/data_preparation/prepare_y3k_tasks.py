"""Materialize audited Y3K scientific tasks without fitting or publishing models."""

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import json
from pathlib import Path

from agent.rsi.datasets.y3k_tasks import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(json.loads(args.source.read_text()), args.output)
    print(json.dumps({k: v for k, v in report.items() if k != "excluded"}, indent=2))


if __name__ == "__main__":
    main()
