#!/usr/bin/env python3
"""Prepare outcome-blind shared allocations; never launch an external model."""

import argparse
import json
from pathlib import Path

from agent.rsi.campaigns.shared_acquisition_plan import plan_acquisition
from agent.rsi.engine.storage import atomic_json, sha256_file


def reference(path):
    path = Path(path).absolute()
    return {"path": str(path), "sha256": sha256_file(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--project", action="append", required=True)
    parser.add_argument("--targets", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--previous-panel", action="append", default=[])
    parser.add_argument("--registered-catalog", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("Output already exists")
    plan = plan_acquisition(
        reference(args.manifest),
        reference(args.catalog),
        projects=args.project,
        target_count=args.targets,
        batch_size=args.batch_size,
        previous_panels=[reference(p) for p in args.previous_panel],
        registered_catalogs=[reference(p) for p in args.registered_catalog],
    )
    atomic_json(output, plan)
    print(
        json.dumps(
            {
                k: plan[k]
                for k in [
                    "status",
                    "target_count",
                    "batch_count",
                    "available_scopes",
                    "selection_counts",
                    "external_model_calls",
                    "new_autonomous_models",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
