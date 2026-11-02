"""Prepare pinned, nonimputed author phosphosite measurements."""

import argparse
import json
from pathlib import Path

from agent.rsi.phospho_source import prepare


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
