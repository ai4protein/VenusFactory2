"""Prepare pinned, nonimputed author phosphosite measurements."""

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import json
from pathlib import Path

from agent.rsi.datasets.phospho_source import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = prepare(json.loads(args.declaration.read_text()), args.output)
    print(
        json.dumps(
            {
                key: report[key]
                for key in [
                    "status",
                    "samples",
                    "site_scopes_before_task_admission",
                    "distinct_proteins",
                    "finite_observations",
                    "external_model_calls",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
