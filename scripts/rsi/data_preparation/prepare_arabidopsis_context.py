"""Export source-verified Arabidopsis condition tasks without model calls."""

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import json
from pathlib import Path

from agent.rsi.datasets.arabidopsis_context import export_tasks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--methods-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = export_tasks(args.raw_root, args.methods_root, args.output)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in {"excluded_training_candidates", "per_replicate"}}, indent=2
        )
    )
