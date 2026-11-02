"""Independent standard-library replay of saved assay prediction scores.

This module does not call the assay evaluator or refit baseline estimators.
"""

import hashlib
import json
import math
from pathlib import Path

INPUTS = ("id", "protein", "sequence", "substrate", "study", "group", "values")


def canonical(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        for k in order[i:j]:
            result[k] = (i + 1 + j) / 2
        i = j
    return result


def recompute(training, scoring, predictions):
    require(len(training) > 0 and len(scoring) > 0, "Empty score cohort")
    require(len(predictions) == len(scoring), "Prediction count differs")
    for row, pred in zip(scoring, predictions, strict=True):
        require(
            (pred.get("id"), pred.get("protein"), pred.get("group"), pred.get("inputs_sha256"))
            == (row["id"], row["protein"], row["group"], canonical({k: row[k] for k in INPUTS})),
            "Assay prediction witness differs",
        )
        require(
            type(pred.get("prediction")) in (int, float) and math.isfinite(pred["prediction"]), "Invalid prediction"
        )
    mean = math.fsum(r["target"] for r in training) / len(training)
    errors = [float(r["target"]) - float(p["prediction"]) for r, p in zip(scoring, predictions, strict=True)]
    null_errors = [float(r["target"]) - mean for r in scoring]
    mse = math.fsum(e * e for e in errors) / len(errors)
    null = math.fsum(e * e for e in null_errors) / len(errors)
    grouped = []
    for group in sorted({r["group"] for r in scoring}):
        ii = [i for i, r in enumerate(scoring) if r["group"] == group]
        grouped.append(
            {
                "group": group,
                "rows": len(ii),
                "mse": math.fsum(errors[i] ** 2 for i in ii) / len(ii),
                "mae": math.fsum(abs(errors[i]) for i in ii) / len(ii),
                "training_mean_mse": math.fsum(null_errors[i] ** 2 for i in ii) / len(ii),
            }
        )
    gmse = math.fsum(g["mse"] for g in grouped) / len(grouped)
    gnull = math.fsum(g["training_mean_mse"] for g in grouped) / len(grouped)
    ry = ranks([r["target"] for r in scoring])
    rp = ranks([p["prediction"] for p in predictions])
    ry = [r - (len(ry) + 1) / 2 for r in ry]
    rp = [r - (len(rp) + 1) / 2 for r in rp]
    denom = math.sqrt(math.fsum(r * r for r in ry) * math.fsum(r * r for r in rp))
    return {
        "rows": len(scoring),
        "independent_groups": len(grouped),
        "training_groups": len({r["group"] for r in training}),
        "predictions_sha256": canonical(predictions),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": math.fsum(abs(e) for e in errors) / len(errors),
        "training_mean": mean,
        "training_mean_mse": null,
        "mse_skill_vs_training_mean": None if null == 0 else 1 - mse / null,
        "spearman": math.fsum(a * b for a, b in zip(ry, rp, strict=True)) / denom if denom else None,
        "undefined_spearman": not bool(denom),
        "degenerate_baseline": null == 0,
        "group_balanced_mse": gmse,
        "group_balanced_rmse": math.sqrt(gmse),
        "group_balanced_mae": math.fsum(g["mae"] for g in grouped) / len(grouped),
        "group_balanced_training_mean_mse": gnull,
        "group_balanced_mse_skill_vs_training_mean": None if gnull == 0 else 1 - gmse / gnull,
        "per_group": grouped,
    }


def compare(actual, expected, *, path="metrics"):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), "Score fields differ: " + path)
        return max([compare(actual[k], v, path=path + "." + k) for k, v in expected.items()] or [0.0])
    if isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), "Score rows differ: " + path)
        return max([compare(a, b, path=path) for a, b in zip(actual, expected, strict=True)] or [0.0])
    if type(expected) in (int, float):
        require(type(actual) in (int, float) and math.isfinite(actual), "Invalid score: " + path)
        error = abs(actual - expected)
        require(error <= 1e-10 * max(1.0, abs(expected)), "Numerical replay differs: " + path)
        return error
    require(actual == expected, "Score value differs: " + path)
    return 0.0


def replay_task(task, result):
    require(result["schema"] == "shared_assay_final_evaluation.v1", "Unsupported score report")
    require(result["task_sha256"] == canonical(task) and result["task_id"] == task["id"], "Task binding differs")
    data = {}
    for phase, ref in task["data"].items():
        payload = Path(ref["path"]).read_bytes()
        require(hashlib.sha256(payload).hexdigest() == ref["sha256"], "Scoring data changed")
        data[phase] = json.loads(payload)
    future = data["future"]
    require(result["query_sha256"] == canonical([{k: r[k] for k in INPUTS} for r in future]), "Query binding differs")
    require(result["rows"] == {k: len(v) for k, v in data.items()}, "Row counts differ")
    require(result["groups"] == {k: len({r["group"] for r in v}) for k, v in data.items()}, "Group counts differ")
    errors = []
    methods = []
    for control in result["controls"]:
        if control["status"] == "completed":
            errors.append(compare(control["metrics"], recompute(data["training"], future, control["predictions"])))
            methods.append(control["method"])
    scientist = result["scientist"]
    if scientist["status"] == "scored":
        scores = recompute(data["training"], future, scientist["predictions"])
        errors.append(compare(scientist["metrics"], scores))
        completed = [c for c in result["controls"] if c["status"] == "completed"]
        best = min(c["metrics"]["group_balanced_rmse"] for c in completed)
        tolerance = 1e-12 * max(1.0, best)
        expected = {
            "metric": "group_balanced_rmse",
            "best_control": best,
            "best_control_methods": [
                c["method"] for c in completed if abs(c["metrics"]["group_balanced_rmse"] - best) <= tolerance
            ],
            "scientist_minus_best": scores["group_balanced_rmse"] - best,
            "scientist_beats_best_control": scores["group_balanced_rmse"] < best - tolerance,
            "scientist_ties_best_control": abs(scores["group_balanced_rmse"] - best) <= tolerance,
            "tie_tolerance": tolerance,
        }
        errors.append(compare(result["comparison"], expected))
    else:
        require(
            scientist == {"status": "not_registered", "metrics": None, "predictions": None}, "Invalid missing model"
        )
        require(result["comparison"] is None, "Missing model has comparison")
    return {
        "task_id": task["id"],
        "status": "prediction_scores_replayed",
        "control_methods": methods,
        "maximum_absolute_error": max(errors or [0]),
        "baseline_refit_performed": False,
    }
