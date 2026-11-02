"""Prepare phosphosite tasks with globally held-out condition groups."""

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import json
from pathlib import Path

from agent.rsi.datasets.phospho_tasks import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261027)
    args = parser.parse_args()
    report = prepare(args.source, args.output, seed=args.seed)
    print(
        json.dumps(
            {
                key: report[key]
                for key in [
                    "status",
                    "scientific_scopes",
                    "distinct_proteins",
                    "models_trained",
                    "models_published",
                    "external_model_calls",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
