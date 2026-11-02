"""Final contextual-task comparison with strict prediction and group identities."""

import numpy as np

from .context_baselines import run_controls
from .model_tasks import inference_inputs, load_task
from .prediction_metrics import measure_context
from .protein_model import digest
from .shared_sequence_evaluation import score_predictions

SCHEMA = "shared_context_final_evaluation.v1"


def evaluate_task(task, scientist_predictions):
    """Score a sealed task; the caller must enforce the campaign terminal barrier."""
    if task["plugin_id"] not in {"generated_context.v1", "generated_context.v2", "generated_context.v3"}:
        raise ValueError("Context response interface required")
    data = load_task(task)
    training, future = data["training"], data["future"]
    query = inference_inputs(future, task["plugin_id"])
    mean = float(np.mean([row["target"] for row in training]))
    baseline = run_controls(task, score_partition="future")
    controls = baseline["outcomes"]
    for row in controls:
        if row["status"] == "completed":
            row["metrics"] = score_predictions(future, row["predictions"], mean)
    scientist = {"status": "not_registered", "metrics": None, "predictions": None}
    comparison = None
    if scientist_predictions is not None:
        metrics = measure_context(training, future, scientist_predictions, scope=task["scope"])
        scientist = {"status": "scored", "metrics": metrics, "predictions": scientist_predictions}
        completed = [row for row in controls if row["status"] == "completed"]
        if not completed:
            raise ValueError("At least one completed control required for comparison")
        best = min(row["metrics"]["rmse"] for row in completed)
        tolerance = 1e-12 * max(1.0, best)
        comparison = {
            "best_control_rmse": best,
            "best_control_methods": [r["method"] for r in completed if abs(r["metrics"]["rmse"] - best) <= tolerance],
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
        "groups": {key: len({r["group"] for r in rows}) for key, rows in data.items()},
        "fit_partition": "training",
        "score_partition": "future",
        "future_scored": True,
        "training_mean": mean,
        "query_sha256": digest(query),
        "scientist": scientist,
        "controls": controls,
        "comparison": comparison,
        "control_provenance": {k: v for k, v in baseline.items() if k != "outcomes"},
        "external_model_calls": 0,
        "new_autonomous_models": 0,
        "statistical_independence_assumed": False,
        "domain_baseline_comparison_complete": False,
    }
