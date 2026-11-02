"""Independently refit all declared controls in an existing final assay evaluation."""

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
import sklearn

from agent.rsi import assay_baseline_refit as refit


def main(args):
    root = args.evaluation.absolute()
    report = refit.read_bound({"path": str(root / "report.json"), "sha256": args.report_sha256})
    refit.require(
        report["schema"] == "shared_assay_final_evaluation.v1" and report["final_labels_scored"] is True,
        "Completed final score report required",
    )
    protocol = refit.read_bound({"path": str(root / "protocol.json"), "sha256": report["protocol_sha256"]})
    frozen = refit.read_bound({"path": protocol["frozen_protocol"], "sha256": protocol["frozen_protocol_sha256"]})
    versions = {"numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__}
    refit.require(frozen["versions"] == versions, "Numerical dependency versions differ")
    bindings = {**frozen["source_bindings"], **frozen["implementation"], **protocol["terminal_and_reuse_bindings"]}
    for path, expected in bindings.items():
        refit.require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, "Frozen evidence changed")
    tasks = {task["id"]: task for task in frozen["tasks"]}
    refit.require(
        len(tasks) == len(frozen["tasks"]) and set(tasks) == set(report["task_reports"]), "Task coverage differs"
    )
    results = []
    for task_id, expected in report["task_reports"].items():
        row = refit.read_bound({"path": str(root / (task_id + ".json")), "sha256": expected})
        refit.require(
            row["task_sha256"] == refit.canonical(tasks[task_id])
            and row["task_id"] == task_id
            and row["data_bindings"] == tasks[task_id]["data"]
            and row["fit_partition"] == "training"
            and row["score_partition"] == "future",
            "Task or partition binding differs",
        )
        results.append(refit.verify_controls(tasks[task_id], row["controls"]))
    for path, expected in bindings.items():
        refit.require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, "Evidence changed during refit")
    refit.require(
        report["totals"]["completed_control_fits"] == sum(len(r["methods"]) for r in results)
        and report["totals"]["failed_control_fits"] == 0,
        "Control-fit totals differ",
    )
    output = {
        "status": "independent_assay_baseline_refit_passed",
        "report_sha256": args.report_sha256,
        "tasks": results,
        "versions": versions,
        "completed_refits": sum(len(r["methods"]) for r in results),
        "maximum_absolute_prediction_error": max(r["maximum_absolute_prediction_error"] for r in results),
        "implementation_sha256": hashlib.sha256(Path(refit.__file__).read_bytes()).hexdigest(),
        "descriptor_encoding_reimplemented": False,
        "baseline_refit_performed": True,
        "new_autonomous_models": 0,
        "external_model_calls": 0,
    }
    with args.output.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in output.items() if k != "tasks"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
