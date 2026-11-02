"""Verify and summarize a globally terminal matched three-arm history study."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path

from agent.rsi import (
    context_baseline_refit,
    matched_history_analysis,
    matched_history_evidence,
    matched_history_operations,
)
from agent.rsi.matched_history_analysis import require
from agent.rsi.matched_history_evidence import Evidence, analyze, terminal_study


def main(args):
    repository = Path(__file__).resolve().parents[2]
    evidence = Evidence(repository)
    study = evidence.path(args.study)
    if args.check_ready:
        require(args.launch_basis_sha256, "Launch-basis hash is required")
        design, plans, _ = terminal_study(evidence, study, args.launch_basis_sha256)
        print(
            json.dumps(
                {
                    "status": "global_terminal_barrier_passed",
                    "units": len(plans),
                    "scientific_scopes": design["scientific_scopes"],
                    "final_labels_read": False,
                    "external_model_calls": 0,
                }
            )
        )
        return
    require(args.manifest and args.manifest_sha256 and args.output, "Manifest, its SHA256 and output are required")
    manifest = evidence.read(args.manifest, args.manifest_sha256, inside=study)
    output = evidence.path(args.output, inside=study)
    require(output != study and not output.exists(), "A new output directory inside the study is required")
    implementation = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [
            Path(__file__),
            *[
                Path(m.__file__)
                for m in (
                    matched_history_analysis,
                    matched_history_evidence,
                    matched_history_operations,
                    context_baseline_refit,
                )
            ],
        ]
    }
    report = analyze(study, manifest, repository=repository)
    evidence.check_bindings(implementation)
    evidence.unchanged()
    report.update(
        {
            "utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "analysis_manifest_sha256": args.manifest_sha256,
            "implementation": implementation,
        }
    )
    output.mkdir()
    with (output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "scientific_scopes",
                    "allocated_model_attempts",
                    "distinct_registered_scopes",
                    "arms",
                )
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-ready", action="store_true")
    parser.add_argument("--launch-basis-sha256")
    main(parser.parse_args())
