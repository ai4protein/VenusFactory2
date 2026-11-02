"""Declared comparisons within the admitted training cohort, never outer labels."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

try:
    from agent.rsi import context_controls, generated_context, reference_thermal_baselines
    from .protein_model import digest
except ImportError:  # Frozen standalone MCP capability.
    import context_controls
    import generated_context
    import reference_thermal_baselines
    from protein_model import digest

FIELD = "training_baseline_feedback"
POLICY = "group_baselines.v1"


def policy_for(declaration, *, plugin_id=None):
    value = declaration.get(FIELD, "none")
    if value not in ("none", POLICY):
        raise ValueError("Unsupported training-baseline feedback policy")
    plugin = plugin_id or declaration.get("model_plugin_id", declaration.get("plugin_id"))
    if value != "none" and plugin != generated_context.PLUGIN_ID:
        raise ValueError("Group-baseline feedback requires a contextual model task")
    return value


def verify_goal(state, protocol):
    policy = policy_for(protocol)
    if state["definition"].get(FIELD, "none") != policy:
        raise ValueError("Goal training-baseline feedback binding changed")
    return policy


def verify_bundle(task, protocol):
    expected = policy_for(protocol) if task["phase"] == "develop" else "none"
    if policy_for(task) != expected:
        raise ValueError("Episode training-baseline feedback policy differs from goal")


def description(scope):
    return {
        "policy": POLICY,
        "data_boundary": "training_only",
        "split": "same whole-group split as candidate feedback",
        "methods": list(context_controls.METHODS + context_controls.thermal_methods(scope)),
        "hyperparameters": "fixed; fit and standardize on the fitting groups only",
        "purpose": "Compare candidate predictions with fixed methods before revising or submitting source.",
    }


def comparison(rows, scope, candidate_metrics):
    """Compute from explicit training rows; this function accepts no task/file paths."""
    generated_context.validate_rows(rows, scope, labels=True)
    training, scoring = generated_context.development_split(rows)
    if (
        candidate_metrics.get("rows") != len(scoring)
        or candidate_metrics.get("independent_groups") != len({r["group"] for r in scoring})
        or type(candidate_metrics.get("rmse")) not in (int, float)
        or not math.isfinite(candidate_metrics["rmse"])
        or candidate_metrics["rmse"] < 0
    ):
        raise ValueError("Candidate feedback does not match the internal group split")
    outcomes = context_controls.score_rows(training, scoring, scope, timings=False)
    controls = []
    for outcome in outcomes:
        item = {key: value for key, value in outcome.items() if key != "predictions"}
        if "predictions" in outcome:
            item["predictions_sha256"] = digest(outcome["predictions"])
        controls.append(item)
    complete = [x for x in controls if x["status"] == "completed"]
    best = min(complete, key=lambda x: (x["metrics"]["rmse"], x["method"])) if complete else None
    modules = [
        Path(__file__),
        Path(context_controls.__file__),
        Path(generated_context.__file__),
        Path(reference_thermal_baselines.__file__),
    ]
    return {
        "schema": POLICY,
        "data_boundary": "training_only",
        "training_sha256": digest(sorted(rows, key=lambda r: r["id"])),
        "split": {
            "seed": 314159,
            "fit_rows": len(training),
            "score_rows": len(scoring),
            "fit_groups": sorted({r["group"] for r in training}),
            "score_groups": sorted({r["group"] for r in scoring}),
            "fit_sha256": digest(training),
            "score_sha256": digest(scoring),
        },
        "implementation": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in modules},
        "controls": controls,
        "completed_controls": len(complete),
        "attempted_controls": len(controls),
        "candidate_rmse": candidate_metrics["rmse"],
        "best_completed_control": best["method"] if best else None,
        "candidate_minus_best_rmse": candidate_metrics["rmse"] - best["metrics"]["rmse"] if best else None,
        "new_autonomous_models": 0,
    }


def verify_feedback(task, rows, feedback):
    policy = policy_for(task)
    packet = feedback.get("baseline_comparison")
    if policy == "none":
        if packet is not None:
            raise ValueError("Undeclared baseline feedback")
    elif packet != comparison(rows, task["scope"], feedback):
        raise ValueError("Training-baseline feedback differs from numerical replay")
