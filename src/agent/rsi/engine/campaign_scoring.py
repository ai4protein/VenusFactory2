"""Numerical scoring shared by single-goal and shared-program assessments."""

from agent.harness.contracts import digest, file_hash
from agent.rsi.datasets import model_tasks

from .campaign_evidence import require
from .model_plugins import get_model_plugin
from .prediction_metrics import measure, measure_context
from .storage import atomic_json


def score_model(task, data, model, output, artifact):
    """Score a frozen model with label-free inputs; never turn a failure into zero."""
    require(
        model["scope"] == task["scope"] and model["training_sha256"] == digest(data["training"]),
        "Scoring model differs from the admitted task",
    )
    plugin = get_model_plugin(task["plugin_id"])
    try:
        predictions = plugin.predict(model, model_tasks.inference_inputs(data["future"], plugin.id), model["scope"])
        if plugin.id in {"generated_context.v1", "generated_context.v2", "generated_context.v3"}:
            metrics = measure_context(data["training"], data["future"], predictions, scope=model["scope"])
        else:
            metrics = measure(
                data["training"],
                data["future"],
                predictions,
                classes=len(model["scope"]["class_labels"]) if "class_labels" in model["scope"] else None,
            )
    except Exception as exc:
        return {"status": "prediction_failed", "error_type": type(exc).__name__, "metrics": None}
    atomic_json(artifact, predictions)
    return {
        "status": "scored",
        "metrics": metrics,
        "predictions": {"path": str(artifact.relative_to(output)), "sha256": file_hash(artifact)},
    }


def context_controls(task, data, output, identity, *, independent_refit=False):
    """Refit fixed training-only controls and score their predictions independently."""
    from .context_baselines import run_controls

    controls = run_controls(task, score_partition="future")
    result = {
        k: controls[k]
        for k in (
            "schema",
            "implementation_sha256",
            "numeric_implementation_sha256",
            "shared_implementation_sha256",
            "reference_implementation_sha256",
            "versions",
            "fit_partition",
            "score_partition",
            "future_scored",
            "new_autonomous_models",
        )
    }
    if independent_refit:
        from .context_baseline_refit import verify_controls

        result["independent_refit"] = verify_controls(task, controls["outcomes"], partition="future")
    result["outcomes"] = []
    for item in controls["outcomes"]:
        row = {"method": item["method"], "status": item["status"]}
        if item["status"] != "completed":
            row["error"] = item["error"]
        else:
            require(
                [p["id"] for p in item["predictions"]] == [r["id"] for r in data["future"]],
                "Fixed-control prediction identities differ",
            )
            predictions = [
                {
                    "id": r["id"],
                    "protein": r["protein"],
                    "group": r["group"],
                    "values_sha256": digest(r["values"]),
                    "prediction": p["prediction"],
                }
                for r, p in zip(data["future"], item["predictions"], strict=True)
            ]
            row["metrics"] = measure_context(data["training"], data["future"], predictions, scope=task["scope"])
            path = output / "control_predictions" / identity / (item["method"] + ".json")
            atomic_json(path, predictions)
            row["predictions"] = {"path": str(path.relative_to(output)), "sha256": file_hash(path)}
        result["outcomes"].append(row)
    result["status"] = "completed" if all(r["status"] == "completed" for r in result["outcomes"]) else "partial"
    return result


SUPPORTED_SHARED_PLUGINS = {
    "generated_context.v1",
    "generated_context.v2",
    "generated_context.v3",
    "generated_sequence.v1",
}


def fixed_controls(task, data, output, identity):
    """Dispatch predeclared controls without dropping unsupported representations."""
    if task["plugin_id"].startswith("generated_context."):
        return context_controls(task, data, output, identity, independent_refit=True)
    require(task["plugin_id"] == "generated_sequence.v1", "Unsupported shared control interface")
    from .shared_sequence_evaluation import predict_controls

    controls = predict_controls(data["training"], model_tasks.inference_inputs(data["future"], task["plugin_id"]))
    result = {
        "schema": "fixed_sequence_controls.v1",
        "fit_partition": "training",
        "score_partition": "future",
        "independent_refit_methods": [
            row["method"] for row in controls if row["status"] == "completed" and "independent_max_abs_error" in row
        ],
        "aac_refit": "Fixed versioned implementations; independently rerun by the full cohort audit",
        "outcomes": [],
    }
    for control in controls:
        row = {key: value for key, value in control.items() if key not in {"seconds", "predictions"}}
        row["metrics"] = None
        if control["status"] == "completed":
            require(
                [r["id"] for r in control["predictions"]] == [r["id"] for r in data["future"]],
                "Sequence control prediction identities differ",
            )
            predictions = [
                {**predicted, "sequence": expected["sequence"]}
                for expected, predicted in zip(data["future"], control["predictions"], strict=True)
            ]
            row["metrics"] = measure(data["training"], data["future"], predictions)
            path = output / "control_predictions" / identity / (control["method"] + ".json")
            atomic_json(path, predictions)
            row["predictions"] = {"path": str(path.relative_to(output)), "sha256": file_hash(path)}
        result["outcomes"].append(row)
    result["status"] = "completed" if all(row["status"] == "completed" for row in result["outcomes"]) else "partial"
    return result
