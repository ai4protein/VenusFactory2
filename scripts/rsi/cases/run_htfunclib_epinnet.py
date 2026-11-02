"""Train the published EpiNNet and reproduce its fixed top-40 library choice."""

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
from agent.rsi import htfunclib_case as base  # noqa: E402
from agent.rsi import htfunclib_epinnet as case  # noqa: E402


def run(output):
    output = output.resolve()
    output.relative_to((ROOT / "research_runs/rsi").resolve())
    if output.exists():
        raise FileExistsError(output)
    if sys.version_info[:2] != (3, 8):
        raise ValueError("Isolated Python 3.8 required")
    versions = {k: importlib.metadata.version(k) for k in case.ENVIRONMENT}
    if versions != case.ENVIRONMENT:
        raise ValueError("Pinned author-compatible environment required")
    for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        os.environ[key] = "1"
    import numpy as np
    import pandas as pd
    from sklearn.neural_network import MLPClassifier
    from threadpoolctl import threadpool_limits

    source = ROOT / "research_runs/rsi/case_assets/htFuncLib/inspection_api/htFuncLib_notebook"
    output.mkdir(parents=True)
    report = {
        "status": "running",
        "case": case.CASE_ID,
        "source_commit": case.COMMIT,
        "execution_origin": "new_author_epinnet_training",
        "full_paper_reproduced": False,
        "agent_generated_model": False,
        "new_rosetta_jobs_executed": 0,
        "completed_stages": [],
        "artifact_sha256": {},
    }
    started = time.monotonic()
    base.atomic_json(output / "report.json", report)
    try:
        code_files = [Path(base.__file__), Path(case.__file__), Path(__file__)]
        for path in code_files:
            shutil.copy2(path, output / path.name)
            report["artifact_sha256"][path.name] = base.sha256(path)
        protocol, data, x = case.prepare(source)
        protocol.update(
            cpu_threads=1, python=sys.version, implementation_sha256={p.name: base.sha256(p) for p in code_files}
        )
        base.atomic_json(output / "protocol.json", protocol)
        report["protocol_sha256"] = base.sha256(output / "protocol.json")
        report["overlap"] = protocol["overlap"]
        report["completed_stages"].append("data_reconstruction")
        base.atomic_json(output / "report.json", report)
        model = MLPClassifier(**case.PARAMETERS)
        train_ids = protocol["partitions"]["train"]
        with threadpool_limits(limits=1):
            model.fit(x.loc[train_ids], data.loc[train_ids, ["y"]])
        if model.n_iter_ >= model.max_iter:
            raise ValueError("Author stopping criterion did not converge within the complete budget")
        if model.classes_.tolist() != [0, 1] or len(model.coefs_) != 2:
            raise ValueError("Unexpected trained network")
        weights = {"w0": model.coefs_[0], "b0": model.intercepts_[0], "w1": model.coefs_[1], "b1": model.intercepts_[1]}
        np.savez_compressed(output / "weights.npz", **weights)
        base.atomic_json(output / "loss_curve.json", model.loss_curve_)
        report["training_iterations"] = model.n_iter_
        report["completed_stages"].append("training")
        report["metrics"] = {}
        max_error = 0.0
        for name, ids in protocol["partitions"].items():
            with threadpool_limits(limits=1):
                probabilities = model.predict_proba(x.loc[ids])
                replayed = case.forward(weights, x.loc[ids])
            max_error = max(max_error, float(np.abs(probabilities - replayed).max()))
            prediction = (probabilities[:, 1] > 0.5).astype(int)
            frame = pd.DataFrame(
                {
                    "source_row": ids,
                    "sequence": data.loc[ids, "sequence"].values,
                    "label": data.loc[ids, "y"].values,
                    "probability": probabilities[:, 1],
                    "prediction": prediction,
                }
            )
            frame.to_csv(output / (name + ".predictions.csv"), index=False)
            report["metrics"][name] = base.classification_metrics(frame.label, frame.prediction, [0, 1])
        report["native_probability_max_abs_error"] = max_error
        with threadpool_limits(limits=1):
            probabilities = model.predict_proba(np.eye(91))[:, 1]
        ranking, library_path, chosen = case.rank_and_select(protocol, data, probabilities)
        ranking.to_csv(output / "ranking.csv", index=False)
        base.atomic_json(output / "library_path.json", library_path)
        base.atomic_json(output / "chosen_library.json", chosen)
        report["completed_stages"].extend(["mutation_ranking", "library_selection"])
        report["reference_comparison"] = case.reference_comparison(
            protocol, report["metrics"], ranking, chosen, model.n_iter_
        )
        report["published_score_panel"] = case.archived_rosetta_panel(source, protocol["wt_after_restore_energy"])
        report["completed_stages"].append("published_score_reanalysis")
        for name in [
            "weights.npz",
            "loss_curve.json",
            "ranking.csv",
            "library_path.json",
            "chosen_library.json",
            "all.predictions.csv",
            "train.predictions.csv",
            "test.predictions.csv",
        ]:
            report["artifact_sha256"][name] = base.sha256(output / name)
        if {p.name: base.sha256(p) for p in code_files} != protocol["implementation_sha256"]:
            raise ValueError("EpiNNet implementation changed")
        report["status"] = "completed"
        with threadpool_limits(limits=1):
            report["evidence"] = case.verify_artifacts(report, output, source)
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        base.atomic_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "reference_comparison": result["reference_comparison"],
                "overlap": result["overlap"],
            },
            indent=2,
        )
    )
