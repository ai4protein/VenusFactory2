"""Six fixed-seed EpiNNet fits comparing record and sequence-group partitions."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from agent.rsi import htfunclib_ablation as design  # noqa: E402
from agent.rsi import htfunclib_case as base  # noqa: E402
from agent.rsi import htfunclib_epinnet as case  # noqa: E402


def run(output):
    output = output.resolve()
    output.relative_to((ROOT / "research_runs/rsi").resolve())
    if output.exists():
        raise FileExistsError(output)
    versions = {k: importlib.metadata.version(k) for k in case.ENVIRONMENT}
    if versions != case.ENVIRONMENT or sys.version_info[:2] != (3, 8):
        raise ValueError("Pinned Python 3.8 author environment required")
    for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        os.environ[key] = "1"
    import numpy as np
    import pandas as pd
    from sklearn.neural_network import MLPClassifier
    from threadpoolctl import threadpool_limits

    source = ROOT / "research_runs/rsi/case_assets/htFuncLib/inspection_api/htFuncLib_notebook"
    author, data, x = case.prepare(source)
    plan = {
        "design": "two_split_modes_three_prespecified_seeds",
        "schedule": [],
        "environment": versions,
        "model_parameters": case.PARAMETERS,
        "model_seed": 42,
        "split_seeds": list(design.SEEDS),
        "test_fraction": 0.2,
        "group_unit": "complete 26-position sequence identity",
        "model_selection_from_test": False,
        "primary_endpoint": "test balanced accuracy",
        "secondary_endpoint": "positive recall and unseen-sequence-only test performance",
        "upstream_inputs": author["source_files_sha256"],
        "population": "all author positive and worst-half negative energy-labeled records; conflicts retained",
        "compute_budget": "six fits; each uses original max_iter=20000, tol and no-improvement stopping; one CPU thread",
        "scope_limitations": [
            "Different test populations by split mode; not a paired same-example causal experiment.",
            "Sequence-grouping changes train/test record counts; all counts and class supports reported.",
            "Encodings use the same published feature vocabulary for both arms.",
            "A retrospective single-scaffold computational-score experiment, not agent-generated code or cross-protein efficacy.",
        ],
    }
    for allocation in design.schedule():
        ids, overlap = design.partition(data, allocation["split_seed"], allocation["mode"])
        plan["schedule"].append(
            {
                **allocation,
                "partitions": ids,
                "overlap": overlap,
                "partition_sha256": {k: case.order_digest(v) for k, v in ids.items()},
            }
        )
    output.mkdir(parents=True)
    files = [Path(base.__file__), Path(case.__file__), Path(design.__file__), Path(__file__)]
    for f in files:
        shutil.copy2(f, output / f.name)
    plan["implementation_sha256"] = {f.name: base.sha256(f) for f in files}
    base.atomic_json(output / "protocol.json", plan)
    report = {
        "status": "running",
        "protocol_sha256": base.sha256(output / "protocol.json"),
        "allocations": [],
        "external_model_calls": 0,
        "agent_generated_model": False,
        "cross_protein_generalization": False,
    }
    base.atomic_json(output / "report.json", report)
    for allocation in plan["schedule"]:
        folder = output / allocation["id"]
        folder.mkdir()
        row = {k: allocation[k] for k in ["id", "split_seed", "mode", "overlap"]}
        row["status"] = "running"
        report["allocations"].append(row)
        base.atomic_json(output / "report.json", report)
        start = time.monotonic()
        try:
            ids = allocation["partitions"]
            row.update(train_rows=len(ids["train"]), test_rows=len(ids["test"]))
            model = MLPClassifier(**case.PARAMETERS)
            with threadpool_limits(limits=1):
                model.fit(x.loc[ids["train"]], data.loc[ids["train"], ["y"]])
            if model.n_iter_ >= model.max_iter:
                raise ValueError("Fit exhausted the original maximum without convergence")
            weights = {
                "w0": model.coefs_[0],
                "b0": model.intercepts_[0],
                "w1": model.coefs_[1],
                "b1": model.intercepts_[1],
            }
            np.savez_compressed(folder / "weights.npz", **weights)
            base.atomic_json(folder / "loss_curve.json", model.loss_curve_)
            with threadpool_limits(limits=1):
                probability = model.predict_proba(x.loc[ids["test"]])[:, 1]
                native = case.forward(weights, x.loc[ids["test"]])[:, 1]
            if not np.allclose(probability, native, rtol=0, atol=1e-10):
                raise ValueError("Independent weight inference mismatch")
            test = data.loc[ids["test"]]
            frame = pd.DataFrame(
                {
                    "source_row": ids["test"],
                    "sequence": test.sequence.values,
                    "label": test.y.values,
                    "probability": probability,
                    "prediction": (probability > 0.5).astype(int),
                    "sequence_seen_in_training": test.sequence.isin(set(data.loc[ids["train"], "sequence"])).values,
                }
            )
            frame.to_csv(folder / "predictions.csv", index=False)
            row["metrics"] = base.classification_metrics(frame.label, frame.prediction, [0, 1])
            unseen = frame[~frame.sequence_seen_in_training]
            row["unseen_sequence_test_metrics"] = base.classification_metrics(unseen.label, unseen.prediction, [0, 1])
            row.update(
                status="completed",
                training_iterations=model.n_iter_,
                native_probability_max_abs_error=float(np.abs(probability - native).max()),
                artifact_sha256={
                    n: base.sha256(folder / n) for n in ["weights.npz", "loss_curve.json", "predictions.csv"]
                },
            )
        except Exception as exc:
            row.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        row["seconds"] = time.monotonic() - start
        base.atomic_json(output / "report.json", report)
    if {f.name: base.sha256(f) for f in files} != plan["implementation_sha256"]:
        report["status"] = "source_changed"
    elif all(r["status"] == "completed" for r in report["allocations"]):
        report["status"] = "completed"
        report["summary"] = design.aggregate(report["allocations"])
    else:
        report["status"] = "completed_with_failures"
    base.atomic_json(output / "report.json", report)
    return report


def audit(output, destination):
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits

    output, destination = Path(output).resolve(), Path(destination).resolve()
    output.relative_to((ROOT / "research_runs/rsi").resolve())
    destination.relative_to((ROOT / "research_runs/rsi").resolve())
    if destination.exists():
        raise FileExistsError(destination)
    source = ROOT / "research_runs/rsi/case_assets/htFuncLib/inspection_api/htFuncLib_notebook"
    report = json.loads((output / "report.json").read_text())
    plan = json.loads((output / "protocol.json").read_text())
    if base.sha256(output / "protocol.json") != report["protocol_sha256"]:
        raise ValueError("Ablation protocol changed")
    if (
        plan["model_parameters"] != case.PARAMETERS
        or plan["environment"] != case.ENVIRONMENT
        or plan["model_seed"] != 42
        or plan["split_seeds"] != list(design.SEEDS)
        or plan["test_fraction"] != 0.2
        or plan["model_selection_from_test"] is not False
        or report["status"] != "completed"
        or report["agent_generated_model"] is not False
        or report["cross_protein_generalization"] is not False
    ):
        raise ValueError("Ablation allocation or scope changed")
    _, data, x = case.prepare(source)
    if [r["id"] for r in plan["schedule"]] != [r["id"] for r in design.schedule()]:
        raise ValueError("Ablation schedule changed")
    for path, digest in plan["implementation_sha256"].items():
        if Path(path).name != path or base.sha256(output / path) != digest:
            raise ValueError("Ablation implementation archive changed")
    for wanted, allocation, row in zip(design.schedule(), plan["schedule"], report["allocations"]):
        if {k: row[k] for k in wanted} != wanted or {k: allocation[k] for k in wanted} != wanted:
            raise ValueError("Ablation identity changed")
        ids, overlap = design.partition(data, wanted["split_seed"], wanted["mode"])
        if (
            allocation["partitions"] != ids
            or row["overlap"] != overlap
            or allocation["overlap"] != overlap
            or row["status"] != "completed"
            or row["train_rows"] != len(ids["train"])
            or row["test_rows"] != len(ids["test"])
        ):
            raise ValueError("Ablation split changed")
        folder = output / row["id"]
        if set(row["artifact_sha256"]) != {"weights.npz", "loss_curve.json", "predictions.csv"}:
            raise ValueError("Incomplete ablation artifacts")
        for name, digest in row["artifact_sha256"].items():
            if (folder / name).is_symlink() or base.sha256(folder / name) != digest:
                raise ValueError("Ablation artifact changed")
        frame = pd.read_csv(folder / "predictions.csv")
        with np.load(folder / "weights.npz", allow_pickle=False) as weights, threadpool_limits(limits=1):
            probability = case.forward(weights, x.loc[ids["test"]])[:, 1]
        test = data.loc[ids["test"]]
        seen = test.sequence.isin(set(data.loc[ids["train"], "sequence"])).to_numpy()
        if (
            frame.source_row.tolist() != ids["test"]
            or frame.sequence.tolist() != test.sequence.tolist()
            or not np.array_equal(frame.label, test.y)
            or not np.array_equal(frame.sequence_seen_in_training, seen)
            or not np.isfinite(frame.probability).all()
            or not np.allclose(frame.probability, probability, rtol=0, atol=1e-10)
            or not np.array_equal(frame.prediction, (probability > 0.5).astype(int))
        ):
            raise ValueError("Ablation source binding or native inference changed")
        if base.classification_metrics(frame.label, frame.prediction, [0, 1]) != row["metrics"]:
            raise ValueError("Ablation metrics changed")
        unseen = frame[~frame.sequence_seen_in_training]
        if base.classification_metrics(unseen.label, unseen.prediction, [0, 1]) != row["unseen_sequence_test_metrics"]:
            raise ValueError("Ablation unseen-sequence metric changed")
        losses = json.loads((folder / "loss_curve.json").read_text())
        if not np.isfinite(losses).all() or len(losses) != row["training_iterations"] or not 1 <= len(losses) < 20000:
            raise ValueError("Ablation convergence record changed")
    if design.aggregate(report["allocations"]) != report["summary"]:
        raise ValueError("Ablation summary changed")
    destination.mkdir(parents=True)
    result = {
        "status": "passed",
        "source_report_sha256": base.sha256(output / "report.json"),
        "verifier_sha256": base.sha256(Path(__file__)),
        "allocations": 6,
        "summary": report["summary"],
        "external_model_calls": 0,
    }
    base.atomic_json(destination / "report.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-to", type=Path)
    args = parser.parse_args()
    result = audit(args.output, args.audit_to) if args.audit_to else run(args.output)
    print(json.dumps({"status": result["status"], "summary": result.get("summary")}, indent=2))
