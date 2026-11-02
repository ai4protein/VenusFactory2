"""Independently refit fixed controls in a completed contextual evaluation."""

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import scipy
import sklearn

from agent.rsi import context_baseline_refit as refit


def evidence_bindings(protocol, archive=None):
    sources, implementation = protocol["source_bindings"], protocol["implementation"]
    refit.require(
        all(sources[p] == implementation[p] for p in set(sources) & set(implementation)),
        "Conflicting evidence bindings",
    )
    bindings = {p: {"path": p, "sha256": pin} for p, pin in {**sources, **implementation}.items()}
    if archive is not None:
        manifest = refit.read_bound(archive)
        refit.require(
            set(manifest) == {"schema", "files"} and manifest["schema"] == "context_implementation_archive.v1",
            "Invalid implementation archive",
        )
        for original, reference in manifest["files"].items():
            refit.require(
                original in implementation and original not in sources,
                "Only implementation-only bindings may use an archive",
            )
            refit.require(
                set(reference) == {"path", "sha256"} and reference["sha256"] == implementation[original],
                "Archived implementation must match the original digest",
            )
            bindings[original] = reference
    return bindings


def main(args):
    root = args.evaluation.absolute()
    refit.require(not args.output.exists(), "Refit output must be new")
    report_ref = {"path": str(root / "report.json"), "sha256": args.report_sha256}
    report = refit.read_bound(report_ref)
    refit.require(
        report["schema"] == "shared_context_final_evaluation.v1" and report["final_labels_scored"] is True,
        "Completed contextual score report required",
    )
    protocol_ref = {"path": str(root / "protocol.json"), "sha256": report["protocol_sha256"]}
    protocol = refit.read_bound(protocol_ref)
    refit.require(
        protocol["fit_partition"] == "training"
        and protocol["score_partition"] == "future"
        and protocol["hyperparameter_search"] is False,
        "Frozen training-only protocol required",
    )
    archive_path = getattr(args, "implementation_archive", None)
    archive_pin = getattr(args, "implementation_archive_sha256", None)
    refit.require(bool(archive_path) == bool(archive_pin), "Archive path and digest must be supplied together")
    archive = {"path": str(archive_path.absolute()), "sha256": archive_pin} if archive_path else None
    bindings = evidence_bindings(protocol, archive)
    refit.require(bool(bindings), "Missing original evidence bindings")

    def check_bindings():
        if archive is not None:
            refit.read_bound(archive)
        for name, reference in bindings.items():
            path, expected = Path(reference["path"]), reference["sha256"]
            refit.require(
                path.is_absolute() and not any(p.is_symlink() for p in [path, *path.parents]), "Symlinked evidence"
            )
            refit.require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, "Frozen evidence changed: " + name)

    check_bindings()
    tasks = {task["id"]: task for task in protocol["tasks"]}
    refit.require(
        len(tasks) == len(protocol["tasks"]) and set(tasks) == set(report["task_reports"]), "Task coverage differs"
    )
    refit.require(
        protocol["task_bindings"] == {key: refit.canonical(value) for key, value in tasks.items()},
        "Task binding differs",
    )
    versions = {"numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__}
    results, task_report_refs = [], []
    for number, (task_id, expected) in enumerate(report["task_reports"].items(), 1):
        refit.require(re.fullmatch(r"[a-z][a-z0-9_]{0,159}", task_id), "Invalid task identity")
        task = tasks[task_id]
        reference = {"path": str(root / (task_id + ".json")), "sha256": expected}
        row = refit.read_bound(reference)
        task_report_refs.append(reference)
        refit.require(
            row["task_id"] == task_id
            and row["task_sha256"] == refit.canonical(task)
            and row["data_bindings"] == task["data"]
            and row["fit_partition"] == "training"
            and row["score_partition"] == "future"
            and row["future_scored"] is True,
            "Task or partition differs",
        )
        provenance = row["control_provenance"]
        refit.require(provenance["versions"] == versions, "Numerical dependency versions differ")
        refit.require(
            provenance["fit_partition"] == "training"
            and provenance["score_partition"] == "future"
            and provenance["task_sha256"] == refit.canonical(task),
            "Control provenance differs",
        )
        result = refit.verify_controls(task, row["controls"])
        refit.require(result["query_sha256"] == row["query_sha256"], "Query binding differs")
        results.append(result)
        if number % 16 == 0:
            print(json.dumps({"refitted_tasks": number, "allocated_tasks": len(tasks)}), flush=True)
    check_bindings()
    for reference in [report_ref, protocol_ref, *task_report_refs]:
        refit.read_bound(reference)
    completed = sum(len(r["completed_methods"]) for r in results)
    failed = sum(len(r["reproduced_failed_methods"]) for r in results)
    refit.require(
        report["totals"]["allocated"] == len(results)
        and report["totals"]["completed_control_fits"] == completed
        and report["totals"]["failed_control_fits"] == failed,
        "Control-fit totals differ",
    )
    output = {
        "status": "independent_context_baseline_refit_passed",
        "report_sha256": args.report_sha256,
        "tasks": results,
        "versions": versions,
        "completed_refits": completed,
        "reproduced_failed_refits": failed,
        "maximum_absolute_prediction_error": max(r["maximum_absolute_prediction_error"] for r in results),
        "implementation_sha256": hashlib.sha256(Path(refit.__file__).read_bytes()).hexdigest(),
        "cli_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "historical_implementation_archive": archive,
        "implementation_archive_substitutions": {p: v for p, v in bindings.items() if p != v["path"]},
        "prediction_tolerance": {
            "sigmoid": "1e-7 * max(1,abs(independent_prediction))",
            "other": "1e-9 * max(1,abs(independent_prediction))",
        },
        "source_preprocessing_reimplemented": False,
        "baseline_refit_performed": True,
        "new_autonomous_models": 0,
        "external_model_calls": 0,
    }
    with args.output.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in output.items() if k != "tasks"}), flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation-archive", type=Path)
    parser.add_argument("--implementation-archive-sha256")
    main(parser.parse_args())
