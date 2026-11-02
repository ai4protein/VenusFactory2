#!/usr/bin/env python3
"""Fixed manual controls through the same isolated model ABI used by Scientist."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from agent.rsi import generated_sequence as api  # noqa: E402
from agent.rsi.code_sandbox import capture_code_runs  # noqa: E402
from agent.rsi.model_tasks import inference_inputs, read_task  # noqa: E402
from agent.rsi.published_sequence_tasks import _write, audit_catalog, bound  # noqa: E402

SOURCE = """import numpy as np
DEGREE = 1
def kernel(left, right):
    matrix = np.zeros((len(left), len(right)))
    for position in range(left.shape[1]):
        matrix += left[:, position:position+1] == right[:, position].reshape(1, -1)
    matrix = matrix / left.shape[1]
    return matrix if DEGREE == 1 else (1.0 + matrix)**2
def fit(sequences, targets, seed):
    full = np.array([list(sequence) for sequence in sequences])
    positions = np.flatnonzero(np.any(full != full[0], axis=0))
    if not len(positions):
        raise ValueError('No variable training positions')
    train = full[:, positions]
    target = np.asarray(targets)
    mean = float(np.mean(target))
    weights = np.linalg.solve(kernel(train, train) + np.eye(len(train)), target - mean)
    return {'positions': positions.tolist(), 'train': train.tolist(), 'mean': mean, 'weights': weights.tolist()}
def predict(model, sequences):
    query = np.array([list(sequence) for sequence in sequences])[:, model['positions']]
    train = np.asarray(model['train'])
    return (kernel(query, train) @ np.asarray(model['weights']) + model['mean']).tolist()
"""


def independent_prediction(training, validation, degree):
    """Independent categorical encoding and sklearn kernel-ridge solver."""
    positions = [i for i in range(len(training[0]["sequence"])) if len({row["sequence"][i] for row in training}) > 1]
    alphabet = list("ACDEFGHIKLMNPQRSTVWY")
    encoder = OneHotEncoder(categories=[alphabet] * len(positions), handle_unknown="error")
    train = encoder.fit_transform([[row["sequence"][i] for i in positions] for row in training])
    valid = encoder.transform([[row["sequence"][i] for i in positions] for row in validation])
    target = np.asarray([row["target"] for row in training])
    predictor = KernelRidge(
        alpha=1.0, kernel="polynomial", degree=degree, gamma=1.0 / len(positions), coef0=float(degree == 2)
    )
    predictor.fit(train, target - target.mean())
    return predictor.predict(valid) + target.mean()


def run(catalog, output):
    catalog, output = Path(catalog).absolute(), Path(output).absolute()
    entries = json.loads(catalog.read_text())["entries"]
    audit_catalog(entries)
    output.mkdir(parents=True)
    sources = {degree: SOURCE.replace("DEGREE = 1", f"DEGREE = {degree}") for degree in (1, 2)}
    for degree, source in sources.items():
        (output / f"degree_{degree}.py").write_text(source)
    _write(
        output / "protocol.json",
        {
            "origin": "manually_specified_fixed_controls",
            "autonomous_model": False,
            "registered": False,
            "external_model_calls": 0,
            "future_metrics_evaluated": False,
            "controls": ["additive categorical kernel ridge", "degree-2 categorical kernel ridge with pairwise terms"],
            "features": "Variable positions identified from training sequences only",
            "kernels": ["K=matching positions/number of variable positions", "(1+K)^2"],
            "alpha": 1.0,
            "seed": 0,
            "center_targets": "training mean",
            "hyperparameter_search": False,
            "independent_replay": "sklearn one-hot encoding and KernelRidge polynomial kernel",
            "numeric_rtol": 1e-9,
            "numeric_atol": 1e-9,
            "catalog": bound(catalog),
            "runner": bound(Path(__file__)),
            "sources": [bound(output / f"degree_{degree}.py") for degree in sources],
            "plugin_sha256": api.implementation_hash(),
        },
    )
    reports = []
    for entry in entries:
        task, data = read_task(entry["task"]["path"])
        for degree, source in sources.items():
            folder = output / f"{task['id']}_degree{degree}"
            folder.mkdir()
            report = {
                "id": task["id"],
                "degree": degree,
                "autonomous_model": False,
                "registered": False,
                "future_metrics_evaluated": False,
            }
            try:
                with capture_code_runs() as calls:
                    model = api.fit_candidate(data["training"], task["scope"], {"source_code": source, "seed": 0})
                    model["development_type"] = "manually_specified_fixed_control"
                    _write(folder / "manual_control.json", model)
                    metrics, predictions = api.evaluate_with_predictions(model, data["validation"], task["scope"])
                    _write(folder / "validation_predictions.json", predictions)
                    expected = independent_prediction(data["training"], data["validation"], degree)
                    actual = np.asarray([row["prediction"] for row in predictions])
                    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)
                    truth = np.asarray([row["target"] for row in data["validation"]])
                    train_mean = np.mean([row["target"] for row in data["training"]])
                    skill = 1 - np.mean((truth - actual) ** 2) / np.mean((truth - train_mean) ** 2)
                    np.testing.assert_allclose(metrics["mse_skill_vs_training_mean"], skill, rtol=1e-9, atol=1e-9)
                    replay = api.fit_candidate(data["training"], task["scope"], {"source_code": source, "seed": 0})
                    second = api.predict(replay, inference_inputs(data["validation"], api.PLUGIN_ID), task["scope"])
                    np.testing.assert_allclose(actual, [row["prediction"] for row in second], rtol=1e-9, atol=1e-9)
                _write(folder / "numeric_execution.json", calls)
                report.update(
                    status="verified",
                    metrics=metrics,
                    independent_max_abs_error=float(np.max(np.abs(actual - expected))),
                    fresh_sandbox_fit_replay=True,
                )
            except Exception as error:
                report.update(status="failed", error=type(error).__name__, detail=str(error))
            _write(folder / "report.json", report)
            reports.append(report)
            print(json.dumps(report), flush=True)
    result = {
        "tasks": len(entries),
        "controls": reports,
        "verified": sum(row["status"] == "verified" for row in reports),
        "total": len(reports),
        "new_autonomous_models": 0,
        "model_publications": 0,
        "external_model_calls": 0,
        "future_metrics_evaluated": False,
    }
    _write(output / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.catalog, args.output)
    if result["verified"] != result["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
