"""Screen verified source programs using recipient training data only."""

import argparse
import json
from pathlib import Path

from agent.rsi.library_program_screen import export


def main(args):
    report = export(json.loads(args.request.read_text()), args.output)
    print(
        json.dumps(
            {
                "tasks": len(report["outcomes"]),
                "candidate_source_tests": report["candidate_source_tests"],
                "selected_donors": sum(len(v["selection"]["entry_ids"]) for v in report["outcomes"]),
                "external_model_calls": 0,
                "new_published_models": 0,
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    main(parser.parse_args())
