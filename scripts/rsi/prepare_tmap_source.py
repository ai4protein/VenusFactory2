"""Reconstruct public T-MAP source measurements; make no model requests."""

import argparse
import json
from pathlib import Path

from agent.rsi.tmap_source import prepare


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
