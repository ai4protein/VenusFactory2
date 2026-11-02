"""Prepare independently grouped plant kinetics tasks from frozen public sources."""

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import json

from agent.rsi.datasets.sabio_assay_tasks import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--sequence-snapshots", required=True, nargs="+")
    parser.add_argument("--taxonomy-snapshots", nargs="+", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = prepare(args.observations, args.sequence_snapshots, args.taxonomy_snapshots, args.output)
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "source_observations",
                    "retained_observations",
                    "proteins",
                    "publications",
                    "independent_components",
                    "descriptor_dimensions",
                    "new_admitted_tasks",
                    "new_autonomous_models",
                    "external_model_calls",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
