"""Final sequence-property assessment using predeclared, training-only controls."""

from __future__ import annotations

import math
import time

import numpy as np
from scipy.stats import spearmanr

from agent.rsi.datasets.model_tasks import inference_inputs, load_task
from .position_baselines import IncompatibleAlignment, predict_control
from .protein_model import digest
from .traditional_baselines import estimators, features

SCHEMA = "shared_sequence_final_evaluation.v1"
POSITIONAL_METHODS = {1: "positional_additive_ridge", 2: "positional_pairwise_ridge"}


def require_terminal(state, *, runner_live):
    if (
        runner_live
        or state["status"] not in {"completed", "needs_tasks", "budget_exhausted"}
        or state["owner"] is not None
        or state["inflight"] is not None
        or not state["batches"]
        or any(row["status"] not in {"completed", "budget_exhausted"} for row in state["batches"].values())
    ):
        raise ValueError("Final scoring requires every acquisition batch terminal and its runner exited")


def predict_controls(training, query):
    """Query labels are forbidden; every estimator is fitted only to training."""
    if not training or not query or any("target" in row for row in query):
        raise ValueError("Nonempty training and unlabeled query rows required")
    target = np.asarray([row["target"] for row in training], dtype=float)
    if not np.all(np.isfinite(target)):
        raise ValueError("Finite training labels required")
    location, scale = float(target.mean()), float(target.std())
    scale = scale if scale > 0 else 1.0
    x_train, x_query = features(training), features(query)
    outcomes = []
    for name, estimator in estimators().items():
        started = time.monotonic()
        estimator.fit(x_train, (target - location) / scale)
        predicted = estimator.predict(x_query) * scale + location
        if predicted.shape != (len(query),) or not np.all(np.isfinite(predicted)):
            raise ValueError("Invalid classical predictions")
        outcomes.append(
            {
                "method": name,
                "status": "completed",
                "seconds": time.monotonic() - started,
                "predictions": [
                    {"id": row["id"], "prediction": float(value)} for row, value in zip(query, predicted, strict=True)
                ],
            }
        )
    for degree, name in POSITIONAL_METHODS.items():
        started = time.monotonic()
        try:
            predicted, independent, support = predict_control(training, query, degree)
        except IncompatibleAlignment as exc:
            outcomes.append({"method": name, "status": "unsupported", "reason": str(exc), "predictions": None})
            continue
        np.testing.assert_allclose(predicted, independent, rtol=1e-9, atol=1e-9)
        if predicted.shape != (len(query),) or not np.all(np.isfinite(predicted)):
            raise ValueError("Invalid positional predictions")
        # The legacy helper named these query diagnostics after its validation caller.
        support = {key.replace("validation_rows", "query_rows"): value for key, value in support.items()}
        outcomes.append(
            {
                "method": name,
                "status": "completed",
                "seconds": time.monotonic() - started,
                "independent_max_abs_error": float(np.max(np.abs(predicted - independent))),
                "support": support,
                "predictions": [
                    {"id": row["id"], "prediction": float(value)} for row, value in zip(query, predicted, strict=True)
                ],
            }
        )
    return outcomes


def score_predictions(truth_rows, predictions, training_mean):
    ids = [row["id"] for row in truth_rows]
    if len(set(ids)) != len(ids) or [row["id"] for row in predictions] != ids:
        raise ValueError("Prediction identities must match every final row in order")
    truth = np.asarray([row["target"] for row in truth_rows], dtype=float)
    values = np.asarray([row["prediction"] for row in predictions], dtype=float)
    if (
        not len(truth)
        or not np.all(np.isfinite(truth))
        or not np.all(np.isfinite(values))
        or not math.isfinite(training_mean)
    ):
        raise ValueError("Finite nonempty final scores required")
    mse = float(np.mean((truth - values) ** 2))
    null_mse = float(np.mean((truth - training_mean) ** 2))
    rho = None
    if np.ptp(truth) > 0 and np.ptp(values) > 1e-12:
        rho = float(spearmanr(truth, values).statistic)
        if not math.isfinite(rho):
            raise ValueError("Nonfinite rank correlation")
    return {
        "rmse": math.sqrt(mse),
        "mae": float(np.mean(np.abs(truth - values))),
        "spearman": rho,
        "mse_skill_vs_training_mean": 1 - mse / null_mse if null_mse else None,
    }


def evaluate_task(task, scientist_predictions):
    if task["plugin_id"] != "generated_sequence.v1":
        raise ValueError("This evaluation requires sequence-regression tasks")
    data = load_task(task)
    training, future = data["training"], data["future"]
    query = inference_inputs(future, task["plugin_id"])
    controls = predict_controls(training, query)
    mean = float(np.mean([row["target"] for row in training]))
    for control in controls:
        control["metrics"] = (
            score_predictions(future, control["predictions"], mean) if control["status"] == "completed" else None
        )
    scientist = {"status": "not_acquired", "metrics": None, "predictions": None}
    comparison = None
    if scientist_predictions is not None:
        metrics = score_predictions(future, scientist_predictions, mean)
        scientist = {"status": "scored", "metrics": metrics, "predictions": scientist_predictions}
        best = min(row["metrics"]["rmse"] for row in controls if row["status"] == "completed")
        tolerance = 1e-12 * max(1.0, best)
        comparison = {
            "best_completed_control_rmse": best,
            "best_completed_control_methods": [
                row["method"]
                for row in controls
                if row["status"] == "completed" and abs(row["metrics"]["rmse"] - best) <= tolerance
            ],
            "scientist_minus_best_rmse": metrics["rmse"] - best,
            "scientist_beats_best_control": metrics["rmse"] < best - tolerance,
            "scientist_ties_best_control": abs(metrics["rmse"] - best) <= tolerance,
            "tie_tolerance": tolerance,
        }
    return {
        "schema": SCHEMA,
        "task_id": task["id"],
        "task_sha256": digest(task),
        "stage": task["stage"],
        "scope": task["scope"],
        "source_paper": task["paper"],
        "data_bindings": task["data"],
        "rows": {key: len(rows) for key, rows in data.items()},
        "fit_partition": "training",
        "score_partition": "future",
        "future_scored": True,
        "training_mean": mean,
        "query_sha256": digest(query),
        "controls": controls,
        "scientist": scientist,
        "comparison": comparison,
        "external_model_calls": 0,
        "new_autonomous_models": 0,
        "domain_baseline_comparison_complete": False,
    }
