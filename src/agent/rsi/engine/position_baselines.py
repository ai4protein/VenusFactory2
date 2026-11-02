"""Fixed positional controls for aligned, single-protein regression tasks."""

from __future__ import annotations

import math
import time

import numpy as np
from scipy.stats import spearmanr
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import OneHotEncoder

from agent.rsi.datasets.model_tasks import load_task
from .protein_model import digest

VERSION = "positional_categorical_regression.v1"
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


class IncompatibleAlignment(ValueError):
    """The predeclared representation cannot handle this task."""


def matrices(training, query):
    rows = [*training, *query]
    if not training or not query:
        raise ValueError("Nonempty training and query sets required")
    if any(not r["sequence"] or set(r["sequence"]) - set(ALPHABET) for r in rows):
        raise ValueError("Canonical amino-acid sequences required")
    lengths = {len(r["sequence"]) for r in rows}
    if len(lengths) != 1:
        raise IncompatibleAlignment("Equal-length aligned sequences required; no automatic alignment")
    train = np.array([list(r["sequence"]) for r in training])
    test = np.array([list(r["sequence"]) for r in query])
    positions = np.flatnonzero(np.any(train != train[0], axis=0))
    invariant = np.ones(train.shape[1], dtype=bool)
    invariant[positions] = False
    outside = np.any(test[:, invariant] != train[0, invariant], axis=1)
    return (
        train[:, positions],
        test[:, positions],
        {
            "sequence_length": train.shape[1],
            "training_variable_positions": positions.tolist(),
            "validation_rows_with_changes_at_training_invariant_positions": int(outside.sum()),
            "validation_rows": len(query),
            "interpretation": "Training-invariant positions carry no learned mutation effect in these controls",
        },
    )


def kernel(left, right, degree):
    if degree not in (1, 2):
        raise ValueError("Only the two predeclared degrees are supported")
    result = np.zeros((len(left), len(right)))
    for position in range(left.shape[1]):
        result += left[:, position : position + 1] == right[:, position].reshape(1, -1)
    result /= left.shape[1]
    return result if degree == 1 else (1.0 + result) ** 2


def predict_control(training, query, degree):
    if degree not in (1, 2):
        raise ValueError("Only the two predeclared degrees are supported")
    train, test, support = matrices(training, query)
    target = np.asarray([r["target"] for r in training], dtype=float)
    if not np.all(np.isfinite(target)):
        raise ValueError("Finite training targets required")
    mean = float(target.mean())
    if not train.shape[1]:
        return np.full(len(query), mean), np.full(len(query), mean), support
    weights = np.linalg.solve(kernel(train, train, degree) + np.eye(len(train)), target - mean)
    predictions = kernel(test, train, degree) @ weights + mean
    # Independent categorical representation and solver; neither sees query labels.
    encoder = OneHotEncoder(categories=[list(ALPHABET)] * train.shape[1], handle_unknown="error")
    encoded = encoder.fit_transform(train)
    estimator = KernelRidge(
        alpha=1.0,
        kernel="polynomial",
        degree=degree,
        gamma=1.0 / train.shape[1],
        coef0=float(degree == 2),
    )
    estimator.fit(encoded, target - mean)
    independent = estimator.predict(encoder.transform(test)) + mean
    return predictions, independent, support


def run_controls(task):
    if task["plugin_id"] != "generated_sequence.v1":
        raise ValueError("This suite requires the sequence-regression interface")
    data = load_task(task)
    training, validation = data["training"], data["validation"]
    report = {
        "schema": VERSION,
        "task_id": task["id"],
        "task_sha256": digest(task),
        "fit_partition": "training",
        "score_partition": "validation",
        "future_scored": False,
        "external_model_calls": 0,
        "new_autonomous_models": 0,
        "hyperparameters": {"alpha": 1.0, "degrees": [1, 2], "search": False},
        "representation": "Categorical identities at variable training positions; no alignment or external predictor",
        "outcomes": [],
    }
    try:
        matrices(training, validation)
    except IncompatibleAlignment as exc:
        report.update(status="unsupported", reason=str(exc))
        return report
    truth = np.asarray([r["target"] for r in validation])
    mean = float(np.mean([r["target"] for r in training]))
    null_mse = float(np.mean((truth - mean) ** 2))
    for degree in (1, 2):
        started = time.monotonic()
        predictions, independent, support = predict_control(training, validation, degree)
        np.testing.assert_allclose(predictions, independent, rtol=1e-9, atol=1e-9)
        if not np.all(np.isfinite(predictions)):
            raise ValueError("Nonfinite positional predictions")
        mse = float(np.mean((truth - predictions) ** 2))
        rho = None
        if np.ptp(truth) > 0 and np.ptp(predictions) > 1e-12:
            rho = float(spearmanr(truth, predictions).statistic)
        report["outcomes"].append(
            {
                "method": "positional_additive_ridge" if degree == 1 else "positional_pairwise_ridge",
                "status": "verified",
                "seconds": time.monotonic() - started,
                "independent_max_abs_error": float(np.max(np.abs(predictions - independent))),
                "metrics": {
                    "rmse": math.sqrt(mse),
                    "mae": float(np.mean(np.abs(truth - predictions))),
                    "spearman": rho,
                    "mse_skill_vs_training_mean": 1 - mse / null_mse if null_mse else None,
                },
                "predictions": [
                    {"id": r["id"], "prediction": float(p)} for r, p in zip(validation, predictions, strict=True)
                ],
            }
        )
    report.update(status="completed", support=support)
    return report
