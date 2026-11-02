"""Compact training-only feedback, retaining candidate/control numeric values."""

import copy


def compact_feedback(feedback):
    result = copy.deepcopy(feedback)
    # The enclosing attempt already binds the common source; old remaining-call
    # counters are not the budget for the current development episode.
    result.pop("source_sha256", None)
    result.pop("attempts_remaining", None)
    metrics = result.get("training_only_feedback")
    if not isinstance(metrics, dict):
        return result
    metrics.pop("predictions_sha256", None)
    comparison = metrics.get("baseline_comparison")
    if not isinstance(comparison, dict):
        return result
    controls = comparison["controls"]
    names = sorted({key for control in controls for key in control.get("metrics", {})})
    rows = []
    for control in controls:
        details = {
            key: value
            for key, value in control.items()
            if key not in {"method", "status", "metrics", "predictions_sha256"}
        }
        rows.append(
            [
                control["method"],
                control["status"],
                *[control.get("metrics", {}).get(key) for key in names],
                details or None,
            ]
        )
    split = comparison["split"]
    metrics["baseline_comparison"] = {
        "schema": "compact_training_control_metrics.v1",
        "source_schema": comparison["schema"],
        "data_boundary": comparison["data_boundary"],
        "columns": ["method", "status", *names, "diagnostic"],
        "rows": rows,
        "split_counts": {
            key: len(value) if isinstance(value, list) else value
            for key, value in split.items()
            if key in {"seed", "fit_rows", "score_rows", "fit_groups", "score_groups"}
        },
        **{
            key: comparison[key]
            for key in [
                "completed_controls",
                "attempted_controls",
                "candidate_rmse",
                "best_completed_control",
                "candidate_minus_best_rmse",
            ]
        },
    }
    return result
