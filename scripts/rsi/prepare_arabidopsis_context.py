"""Export source-verified Arabidopsis condition tasks without model calls."""

import argparse
import json
from pathlib import Path

from agent.rsi.arabidopsis_context import export_tasks

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
