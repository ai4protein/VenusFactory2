"""Reconstruct public T-MAP source measurements; make no model requests."""

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import json
from pathlib import Path

from agent.rsi.datasets.tmap_source import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = prepare(json.loads(args.source.read_text()), args.output)
    print(
        json.dumps(
            {
                key: report[key]
                for key in [
                    "status",
                    "samples",
                    "independent_genes",
                    "resolved_proteins",
                    "finite_observations",
                    "models_trained",
                    "external_model_calls",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
