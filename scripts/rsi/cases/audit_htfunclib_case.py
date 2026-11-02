"""Read-only terminal audit with fresh native inference for both GFP classifiers."""

from __future__ import annotations

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from agent.rsi import htfunclib_case as case  # noqa: E402


def native_predictions(model, features, classes, expected):
    import numpy as np

    if model.current_iteration() != 10000:
        raise ValueError("Incomplete native boosting schedule")
    values = np.asarray(model.predict(features, num_threads=4), dtype=float)
    if len(classes) == 2:
        if values.shape != (len(features),):
            raise ValueError("Binary probability shape mismatch")
        values = np.column_stack([1 - values, values])
    if (
        values.shape != (len(features), len(classes))
        or not np.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
        or not np.allclose(values.sum(axis=1), 1, rtol=0, atol=1e-7)
    ):
        raise ValueError("Invalid native probability matrix")
    predictions = np.asarray(classes)[values.argmax(axis=1)]
    if not np.array_equal(predictions, expected):
        raise ValueError("Native predictions disagree with archived per-row predictions")
    return predictions


def audit(run, output):
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    from sklearn.metrics import confusion_matrix

    run, output = run.resolve(), output.resolve()
    base = (ROOT / "research_runs/rsi").resolve()
    run.relative_to(base)
    output.relative_to(base)
    if run == base or output == base or output == run or run in output.parents:
        raise ValueError("Audit must use a separate research directory")
    if output.exists():
        raise FileExistsError(output)
    versions = {k: importlib.metadata.version(k) for k in case.ENVIRONMENT}
    if versions != case.ENVIRONMENT or sys.version_info[:2] != (3, 8):
        raise ValueError("Pinned author environment required for native model replay")
    output.mkdir(parents=True)
    started = time.monotonic()
    record = {
        "status": "running",
        "external_model_calls": 0,
        "read_only_source_run": str(run),
        "audit_environment": versions,
        "audit_source_sha256": case.sha256(__file__),
        "audit_helper_sha256": case.sha256(case.__file__),
        "full_paper_reproduced": False,
        "agent_generated_model": False,
        "new_training_performed": False,
    }
    case.atomic_json(output / "report.json", record)
    try:
        report_path = run / "report.json"
        before = case.sha256(report_path)
        report = json.loads(report_path.read_text())
        source = ROOT / "research_runs/rsi/case_assets/htFuncLib/inspection_api/random_forest"
        evidence = case.verify_artifacts(report, run, source)
        if report.get("evidence") != evidence:
            raise ValueError("Stored scientific evidence differs from reconstructed evidence")
        protocol, features, targets, identities = case.admit(source)
        saved = json.loads((run / "protocol.json").read_text())
        if any(saved.get(k) != v for k, v in protocol.items()):
            raise ValueError("Full admission protocol reconstruction mismatch")
        if case.sha256(case.__file__) != report["artifact_sha256"]["htfunclib_case.py"] or saved[
            "implementation_sha256"
        ] != {name: report["artifact_sha256"][name] for name in ["htfunclib_case.py", "run_htfunclib_case.py"]}:
            raise ValueError("Training implementation binding mismatch")
        stages = {}
        predictions = {name: {} for name in case.SPLIT_ROWS}
        for stage, expected_classes in [("functionality", [False, True]), ("colour", ["C", "G", "G/C", "U"])]:
            classes = json.loads((run / (stage + ".classes.json")).read_text())
            if classes != expected_classes:
                raise ValueError("Class order mismatch")
            histories = json.loads((run / (stage + ".evaluations.json")).read_text())
            lengths = {
                group + "/" + metric: len(values)
                for group, metrics in histories.items()
                for metric, values in metrics.items()
            }
            if not lengths or set(lengths.values()) != {10000}:
                raise ValueError("Incomplete evaluation history")
            model = lgb.Booster(model_file=str(run / (stage + ".txt")))
            if model.num_feature() != len(protocol["features"]) or model.feature_name() != protocol["features"]:
                raise ValueError("Native feature contract mismatch")
            stage_result = {
                "boosting_iterations": model.current_iteration(),
                "history_lengths": lengths,
                "native_model_sha256": case.sha256(run / (stage + ".txt")),
                "partitions": {},
            }
            for split, n in case.SPLIT_ROWS.items():
                archived = pd.read_csv(run / (split + ".predictions.csv"))
                if len(archived) != n or not archived[["pseudo_prot", "assign"]].equals(identities[split]):
                    raise ValueError("Row binding mismatch")
                predicted = native_predictions(model, features[split], classes, archived[stage].to_numpy())
                truth = targets[split] != "U" if stage == "functionality" else targets[split]
                matrix = confusion_matrix(truth, predicted, labels=classes)
                if matrix.tolist() != report["metrics"][split][stage]["confusion_matrix"]:
                    raise ValueError("Independent confusion matrix mismatch")
                predictions[split][stage] = predicted
                stage_result["partitions"][split] = {
                    "rows": n,
                    "prediction_mismatches": 0,
                    "confusion_matrix": matrix.tolist(),
                    "balanced_accuracy": float(np.mean(matrix.diagonal() / matrix.sum(axis=1))),
                }
            stages[stage] = stage_result
            record["completed_native_stages"] = stages
            case.atomic_json(output / "report.json", record)
        for split in case.SPLIT_ROWS:
            chained = np.where(predictions[split]["functionality"], predictions[split]["colour"], "U")
            archived = pd.read_csv(run / (split + ".predictions.csv"))
            if not np.array_equal(chained, archived["chained"].to_numpy()):
                raise ValueError("Independent chained inference mismatch")
        case.verify_artifacts(report, run, source)
        if case.sha256(report_path) != before:
            raise ValueError("Source report changed during audit")
        record.update(
            status="passed",
            source_report_sha256=before,
            all_rows_native_replayed=True,
            chained_inference_verified=True,
            evidence=evidence,
        )
    except Exception as exc:
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        record["wall_seconds"] = time.monotonic() - started
        case.atomic_json(output / "report.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        os.environ[key] = "4" if key == "OMP_NUM_THREADS" else "1"
    result = audit(args.run, args.output)
    print(json.dumps({"status": result["status"], "wall_seconds": result["wall_seconds"]}))
