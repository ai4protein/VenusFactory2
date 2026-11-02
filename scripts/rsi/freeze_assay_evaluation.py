"""Freeze assay scoring and source bindings without scoring final labels."""

import argparse
import datetime
import json
from pathlib import Path

from agent.rsi import shared_assay_evaluation as evaluation
from agent.rsi.storage import sha256_file

ROOT = Path(__file__).resolve().parents[2]


def implementation():
    paths = sorted((ROOT / "src/agent/rsi").glob("*.py"))
    paths += [
        ROOT / "scripts/rsi/freeze_assay_evaluation.py",
        ROOT / "scripts/rsi/evaluate_shared_assay_campaign.py",
        ROOT / "scripts/rsi/evaluate_shared_context_campaign.py",
        ROOT / "scripts/rsi/replay_assay_scores.py",
    ]
    return {str(p): sha256_file(p) for p in paths}


def main(args):
    tasks = json.loads(args.tasks.read_text())
    protocol = evaluation.freeze_protocol(tasks, implementation())
    protocol["utc"] = datetime.datetime.now(datetime.UTC).isoformat()
    protocol["timing_claim"] = "Freeze precedes scoring; pre-acquisition timing must be bound by the study launcher"
    with args.output.open("x") as stream:
        json.dump(protocol, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "protocol": str(args.output),
                "sha256": sha256_file(args.output),
                "tasks": len(tasks),
                "final_labels_scored": False,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
