"""Frozen final scoring for grouped, source-bound assay models."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import scipy
import sklearn

from agent.rsi import assay_baselines, generated_assay
from agent.rsi.datasets.model_tasks import load_task
from .prediction_metrics import regression_scores
from .protein_model import digest
from .storage import sha256_file

SCHEMA = "shared_assay_final_evaluation.v1"
METRICS = {
    "primary_quality": "group_balanced_mse_skill_vs_training_mean",
    "secondary_quality": ["rmse", "mae", "spearman", "mse_skill_vs_training_mean"],
    "group_weighting": "Equal weight per source-defined independent component, equal rows within component",
    "reference": "Mean of training observations; identical reference for every method",
    "acquisition": "Registered and independently reused scopes / all allocated scopes",
    "missing_models": "Retained in acquisition denominator; prediction quality is missing, not imputed",
    "comparison": "Paired task and replicate; repeated runs and overlapping endpoints are not independent scopes",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def versions():
    return {"numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__}


def measure(training, scoring, predictions, scope):
    """Score supplied predictions without calling any model-plugin evaluator."""
    generated_assay.validate_rows(training, scope, labels=True)
    generated_assay.validate_rows(scoring, scope, labels=True)
    generated_assay.require_disjoint_parts({"training": training, "scoring": scoring}, scope)
    require(isinstance(predictions, list), "Prediction rows required")
    expected = [(r["id"], r["protein"], r["group"], generated_assay.input_digest(r)) for r in scoring]
    require(
        all(isinstance(p, dict) for p in predictions)
        and [(p.get("id"), p.get("protein"), p.get("group"), p.get("inputs_sha256")) for p in predictions] == expected,
        "Prediction identities, complete assay inputs or order differ",
    )
    require(
        all(type(p.get("prediction")) in (float, int) and math.isfinite(p["prediction"]) for p in predictions),
        "Finite numeric predictions required",
    )
    train = np.asarray([r["target"] for r in training], dtype=float)
    truth = np.asarray([r["target"] for r in scoring], dtype=float)
    predicted = np.asarray([p["prediction"] for p in predictions], dtype=float)
    result = regression_scores(train, truth, predicted)
    mean = result["training_mean"]
    groups = sorted({r["group"] for r in scoring})
    grouped = []
    for group in groups:
        indices = [i for i, r in enumerate(scoring) if r["group"] == group]
        grouped.append(
            {
                "group": group,
                "rows": len(indices),
                "mse": math.fsum((float(truth[i]) - float(predicted[i])) ** 2 for i in indices) / len(indices),
                "mae": math.fsum(abs(float(truth[i]) - float(predicted[i])) for i in indices) / len(indices),
                "training_mean_mse": math.fsum((float(truth[i]) - mean) ** 2 for i in indices) / len(indices),
            }
        )
    mse = math.fsum(g["mse"] for g in grouped) / len(groups)
    null = math.fsum(g["training_mean_mse"] for g in grouped) / len(groups)
    result.update(
        rows=len(scoring),
        independent_groups=len(groups),
        training_groups=len({r["group"] for r in training}),
        predictions_sha256=digest(predictions),
        group_balanced_mse=mse,
        group_balanced_rmse=math.sqrt(mse),
        group_balanced_mae=math.fsum(g["mae"] for g in grouped) / len(groups),
        group_balanced_training_mean_mse=null,
        group_balanced_mse_skill_vs_training_mean=None if null == 0 else 1 - mse / null,
        per_group=grouped,
    )
    require(all(not isinstance(v, float) or math.isfinite(v) for v in result.values()), "Nonfinite assay scores")
    return result


def evaluate_task(task, scientist_predictions):
    """Caller enforces terminal acquisition and independent inference barriers."""
    require(task["plugin_id"] == generated_assay.PLUGIN_ID, "Assay interface required")
    data = load_task(task)
    training, future = data["training"], data["future"]
    baseline = assay_baselines.run_controls(task, score_partition="future")
    controls = baseline["outcomes"]
    for control in controls:
        if control["status"] != "completed":
            continue
        predictions = control["predictions"]
        require(
            [(p["id"], p["inputs_sha256"]) for p in predictions]
            == [(r["id"], generated_assay.input_digest(r)) for r in future],
            "Control predictions differ from bound assay inputs",
        )
        control["predictions"] = [
            {**p, "protein": r["protein"], "group": r["group"]} for p, r in zip(predictions, future, strict=True)
        ]
        control["metrics"] = measure(training, future, control["predictions"], task["scope"])
    scientist = {"status": "not_registered", "metrics": None, "predictions": None}
    comparison = None
    if scientist_predictions is not None:
        metrics = measure(training, future, scientist_predictions, task["scope"])
        scientist = {"status": "scored", "metrics": metrics, "predictions": scientist_predictions}
        completed = [c for c in controls if c["status"] == "completed"]
        require(bool(completed), "At least one completed control required")
        key = "group_balanced_rmse"
        best = min(c["metrics"][key] for c in completed)
        tolerance = 1e-12 * max(1.0, best)
        comparison = {
            "metric": key,
            "best_control": best,
            "best_control_methods": [c["method"] for c in completed if abs(c["metrics"][key] - best) <= tolerance],
            "scientist_minus_best": metrics[key] - best,
            "scientist_beats_best_control": metrics[key] < best - tolerance,
            "scientist_ties_best_control": abs(metrics[key] - best) <= tolerance,
            "tie_tolerance": tolerance,
        }
    return {
        "schema": SCHEMA,
        "task_id": task["id"],
        "task_sha256": digest(task),
        "scope": task["scope"],
        "source_paper": task["paper"],
        "data_bindings": task["data"],
        "rows": {p: len(r) for p, r in data.items()},
        "groups": {p: len({x["group"] for x in r}) for p, r in data.items()},
        "fit_partition": "training",
        "score_partition": "future",
        "future_scored": True,
        "query_sha256": digest(generated_assay.inference_inputs(future)),
        "scientist": scientist,
        "controls": controls,
        "comparison": comparison,
        "control_provenance": {k: v for k, v in baseline.items() if k != "outcomes"},
        "external_model_calls": 0,
        "new_autonomous_models": 0,
        "statistical_independence_assumed": False,
        "domain_baseline_comparison_complete": False,
    }


def summarize(rows):
    acquired = [r for r in rows if r["scientist"]["status"] == "scored"]
    return {
        "allocated": len(rows),
        "scored_scientist_models": len(acquired),
        "not_registered": len(rows) - len(acquired),
        "acquisition_rate": len(acquired) / len(rows) if rows else None,
        "positive_group_balanced_final_skill": sum(
            (r["scientist"]["metrics"][METRICS["primary_quality"]] or 0) > 0 for r in acquired
        ),
        "beats_best_control": sum(r["comparison"]["scientist_beats_best_control"] for r in acquired),
        "ties_best_control": sum(r["comparison"]["scientist_ties_best_control"] for r in acquired),
        "completed_control_fits": sum(c["status"] == "completed" for r in rows for c in r["controls"]),
        "failed_control_fits": sum(c["status"] != "completed" for r in rows for c in r["controls"]),
    }


def freeze_protocol(tasks, implementation):
    require(bool(tasks) and len({t["id"] for t in tasks}) == len(tasks), "Distinct assay tasks required")
    require(bool(implementation), "Evaluator implementation must be frozen")
    bindings = {}
    for task in tasks:
        require(task["plugin_id"] == generated_assay.PLUGIN_ID, "Assay interface required")
        load_task(task)
        for ref in [*task["data"].values(), *task["references"]]:
            require(sha256_file(Path(ref["path"])) == ref["sha256"], "Task source changed")
            bindings[ref["path"]] = ref["sha256"]
    result = {
        "schema": SCHEMA,
        "tasks": tasks,
        "task_bindings": {t["id"]: digest(t) for t in tasks},
        "source_bindings": bindings,
        "implementation": implementation,
        "versions": versions(),
        "metrics": METRICS,
        "fit_partition": "training",
        "score_partition": "future",
        "score_partition_meaning": "Source-group held-out observations, not a future calendar period",
        "final_selection": "First independently accepted, freshly reused and published model; no final-label selection",
        "hyperparameter_search": False,
        "final_labels_scored": False,
        "external_model_calls": 0,
        "new_autonomous_models": 0,
    }
    verify_protocol(result, tasks)
    return result


def verify_protocol(protocol, tasks):
    require(protocol["schema"] == SCHEMA and protocol["metrics"] == METRICS, "Final scoring policy changed")
    require(protocol["versions"] == versions(), "Numerical dependency versions changed")
    require(
        protocol["task_bindings"] == {t["id"]: digest(t) for t in tasks}
        and len(tasks) == len(protocol["task_bindings"])
        and protocol["tasks"] == tasks,
        "Final task allocation changed",
    )
    require(protocol["fit_partition"] == "training" and protocol["score_partition"] == "future", "Partition changed")
    require(protocol["hyperparameter_search"] is False and protocol["final_labels_scored"] is False, "Invalid freeze")
    for path, expected in {**protocol["source_bindings"], **protocol["implementation"]}.items():
        p = Path(path)
        require(p.is_file() and not any(x.is_symlink() for x in [p, *p.parents]), "Missing or symlinked frozen file")
        require(sha256_file(p) == expected, "Frozen source or evaluator changed")
