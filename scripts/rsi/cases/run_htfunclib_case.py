"""Replay the pinned GFP two-stage classifier in an isolated Python 3.8 environment."""

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
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from agent.rsi import htfunclib_case as case  # noqa: E402


def run(output, admit_only=False):
    output = output.resolve()
    output.relative_to((ROOT / "research_runs/rsi").resolve())
    if output.exists() or output == (ROOT / "research_runs/rsi").resolve():
        raise FileExistsError(output)
    versions = {k: importlib.metadata.version(k) for k in case.ENVIRONMENT}
    if versions != case.ENVIRONMENT or sys.version_info[:2] != (3, 8):
        raise ValueError("Pinned GFP environment is required; no version fallback")
    for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        os.environ[key] = "4" if key == "OMP_NUM_THREADS" else "1"
    import lightgbm as lgb
    import numpy as np

    source = ROOT / "research_runs/rsi/case_assets/htFuncLib/inspection_api/random_forest"
    output.mkdir(parents=True)
    start = time.monotonic()
    report = {
        "case": case.CASE_ID,
        "status": "admitting",
        "source_commit": case.COMMIT,
        "execution_origin": "new_processed_input_training",
        "full_paper_reproduced": False,
        "agent_generated_model": False,
        "completed_stages": [],
        "artifact_sha256": {},
        "metrics": {},
        "split_rows": case.SPLIT_ROWS,
        "second_stage_retained_rows": {},
        "completed_boosting_iterations": {},
        "native_prediction_replay_max_abs_error": {},
    }
    case.atomic_json(output / "report.json", report)
    try:
        module_path, entry_path = Path(case.__file__), Path(__file__)
        for path in [module_path, entry_path]:
            shutil.copy2(path, output / path.name)
            report["artifact_sha256"][path.name] = case.sha256(output / path.name)
        protocol, features, targets, identities = case.admit(source)
        protocol.update(
            environment=versions,
            python=sys.version,
            cpu_threads=4,
            implementation_sha256={p.name: case.sha256(p) for p in [module_path, entry_path]},
            admission_only=admit_only,
        )
        for split, ids in identities.items():
            path = output / (split + ".identities.csv")
            ids.to_csv(path, index=False)
            report["artifact_sha256"][path.name] = case.sha256(path)
        case.atomic_json(output / "protocol.json", protocol)
        report["protocol_sha256"] = case.sha256(output / "protocol.json")
        report["status"] = "admitted" if admit_only else "running"
        case.atomic_json(output / "report.json", report)
        if admit_only:
            return report
        models, predictions = {}, {}
        for stage in ["functionality", "colour"]:
            report["current_stage"] = stage
            case.atomic_json(output / "report.json", report)
            if case.verify_sources(source) != protocol["source_files_sha256"]:
                raise ValueError("Source changed during training")
            if stage == "functionality":
                labels = {k: v != "U" for k, v in targets.items()}
                xx = features
            else:
                xx, labels = {}, {}
                report["filter_warnings"] = {}
                for split in case.SPLIT_ROWS:
                    kept, warning = case.legacy_second_stage_rows(predictions["functionality"][split])
                    xx[split], labels[split] = features[split].iloc[kept], targets[split][kept]
                    report["second_stage_retained_rows"][split] = len(kept)
                    report["filter_warnings"][split] = warning
                if report["second_stage_retained_rows"] != case.SPLIT_ROWS:
                    raise ValueError("Author legacy filter behaviour changed")
            parameters = dict(protocol["author_parameters"][stage], n_jobs=4)
            model = lgb.LGBMClassifier(**parameters)
            stage_start = time.monotonic()
            model.fit(
                xx["train"],
                labels["train"],
                eval_set=[(xx["train"], labels["train"]), (xx["valid"], labels["valid"])],
                verbose=500,
            )
            report.setdefault("stage_seconds", {})[stage] = time.monotonic() - stage_start
            if model.booster_.current_iteration() != 10000:
                raise ValueError("Full author boosting schedule did not complete")
            models[stage] = model
            path = output / (stage + ".txt")
            model.booster_.save_model(str(path))
            case.atomic_json(output / (stage + ".classes.json"), model.classes_.tolist())
            case.atomic_json(output / (stage + ".evaluations.json"), model.evals_result_)
            fresh = lgb.Booster(model_file=str(path))
            predictions[stage] = {}
            max_error = 0.0
            for split in case.SPLIT_ROWS:
                expected = model.predict_proba(features[split])
                native = fresh.predict(features[split], num_threads=4)
                observed = np.column_stack([1 - native, native]) if stage == "functionality" else native
                max_error = max(max_error, float(np.abs(expected - observed).max()))
                predictions[stage][split] = model.classes_[np.argmax(observed, axis=1)]
                if not np.array_equal(model.predict(features[split]), predictions[stage][split]):
                    raise ValueError("Native model class mapping differs")
            if max_error != 0.0:
                raise ValueError("Native model probability replay differs")
            report["native_prediction_replay_max_abs_error"][stage] = max_error
            report["completed_boosting_iterations"][stage] = model.booster_.current_iteration()
            report["completed_stages"].append(stage)
            for name in [stage + ".txt", stage + ".classes.json", stage + ".evaluations.json"]:
                report["artifact_sha256"][name] = case.sha256(output / name)
            case.atomic_json(output / "report.json", report)
        for split in case.SPLIT_ROWS:
            frame = identities[split].copy()
            frame["functionality"] = predictions["functionality"][split]
            frame["colour"] = predictions["colour"][split]
            frame["chained"] = case.chained_predictions(frame["functionality"].to_numpy(), frame["colour"].to_numpy())
            path = output / (split + ".predictions.csv")
            frame.to_csv(path, index=False)
            report["artifact_sha256"][path.name] = case.sha256(path)
            report["metrics"][split] = {
                "functionality": case.classification_metrics(
                    frame["assign"] != "U", frame["functionality"], [False, True]
                ),
                "colour": case.classification_metrics(frame["assign"], frame["colour"], ["C", "G", "G/C", "U"]),
                "chained": case.classification_metrics(frame["assign"], frame["chained"], ["C", "G", "G/C", "U"]),
            }
        actual = report["metrics"]["test"]["functionality"]["balanced_accuracy"]
        ref = protocol["published_binary_test_balanced_accuracy"]
        report["binary_reference_comparison"] = {
            "published": ref,
            "actual": actual,
            "absolute_difference": abs(actual - ref),
            "tolerance": 1e-6,
            "within_tolerance": abs(actual - ref) <= 1e-6,
        }
        case.verify_sources(source)
        if {p.name: case.sha256(p) for p in [module_path, entry_path]} != protocol["implementation_sha256"]:
            raise ValueError("Implementation changed during reproduction")
        report.update(status="completed", current_stage=None, wall_seconds=time.monotonic() - start)
        report["evidence"] = case.verify_artifacts(report, output, source)
    except Exception as exc:
        report.update(
            status="failed", error_type=type(exc).__name__, error=str(exc), wall_seconds=time.monotonic() - start
        )
        raise
    finally:
        case.atomic_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--admit-only", action="store_true")
    args = parser.parse_args()
    result = run(args.output, args.admit_only)
    print(json.dumps({"status": result["status"], "output": str(args.output)}, indent=2))
