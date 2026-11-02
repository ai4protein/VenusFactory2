"""Prepare phosphosite tasks with globally held-out condition groups."""

import argparse
import json
from pathlib import Path

from agent.rsi.phospho_tasks import prepare


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
