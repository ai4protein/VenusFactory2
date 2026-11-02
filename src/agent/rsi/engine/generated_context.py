"""Generated numeric response models for measured protein assay conditions.

Observation and independent-sample identities stay on the host. Only numeric
covariates and permitted targets enter the isolated model worker.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import numpy as np

try:
    from .code_sandbox import ModelOutputError, guard_source, run_code
    from .protein_model import digest
except ImportError:  # Standalone capability bundle.
    from code_sandbox import ModelOutputError, guard_source, run_code
    from protein_model import digest

PLUGIN_ID = "generated_context.v1"
ACCEPTANCE_METRIC = "mse_skill_vs_training_mean"
GROUP_LEVELS = {"biological_replicate", "workflow_replicate", "independent_batch", "genetic_population"}


def implementation_hash():
    root = Path(__file__).resolve().parent
    return hashlib.sha256(
        b"".join(
            (root / name).read_bytes()
            for name in [
                "generated_context.py",
                "code_sandbox.py",
                "code_worker.py",
                "protein_references.py",
                "numeric_runtime.py",
                "protein_model.py",
            ]
        )
    ).hexdigest()


def text(value):
    return isinstance(value, str) and bool(value.strip()) and len(value) <= 256


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_scope(scope):
    if not isinstance(scope, dict) or set(scope) != {
        "protein",
        "assay",
        "target",
        "unit",
        "context",
        "covariates",
        "grouping",
    }:
        raise ValueError("Context model requires protein, endpoint, context, covariates and grouping")
    if any(not text(scope[key]) for key in ("protein", "assay", "target", "unit")):
        raise ValueError("Nonempty scientific identity and units required")
    context = scope["context"]
    if (
        not isinstance(context, dict)
        or not 1 <= len(context) <= 16
        or any(not text(k) or not text(v) for k, v in context.items())
    ):
        raise ValueError("Recorded assay context required")
    grouping = scope["grouping"]
    if (
        not isinstance(grouping, dict)
        or set(grouping) != {"level", "evidence_sha256"}
        or not isinstance(grouping["level"], str)
        or grouping["level"] not in GROUP_LEVELS
        or not isinstance(grouping["evidence_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", grouping["evidence_sha256"]) is None
    ):
        raise ValueError("Independent-group level and source evidence digest required")
    columns = scope["covariates"]
    if not isinstance(columns, list) or not 1 <= len(columns) <= 16:
        raise ValueError("Declare 1..16 physical covariates")
    seen = set()
    for column in columns:
        if (
            not isinstance(column, dict)
            or set(column) != {"name", "unit", "minimum", "maximum"}
            or not text(column["name"])
            or column["name"] in seen
            or not text(column["unit"])
            or not finite(column["minimum"])
            or not finite(column["maximum"])
            or column["minimum"] >= column["maximum"]
        ):
            raise ValueError("Unique covariates with units and finite increasing domains required")
        seen.add(column["name"])


def validate_rows(rows, scope, *, labels):
    validate_scope(scope)
    if not isinstance(rows, list) or not 1 <= len(rows) <= 10000:
        raise ValueError("Declare 1..10000 contextual observations")
    seen = set()
    required = {"id", "protein", "group", "values"} | ({"target"} if labels else set())
    for row in rows:
        if not isinstance(row, dict) or not required <= set(row) <= required | {"target"}:
            raise ValueError("Invalid contextual observation fields")
        if not text(row["id"]) or row["id"] in seen or not text(row["group"]):
            raise ValueError("Unique observation IDs and independent group identifiers required")
        seen.add(row["id"])
        if row["protein"] != scope["protein"]:
            raise ValueError("Observation protein differs from model scope")
        values = row["values"]
        if not isinstance(values, list) or len(values) != len(scope["covariates"]):
            raise ValueError("Covariates must match the declared column order")
        for value, column in zip(values, scope["covariates"], strict=True):
            if not finite(value) or not column["minimum"] <= value <= column["maximum"]:
                raise ValueError("Finite covariates inside their declared domains required")
            if column["unit"] == "binary_indicator" and value not in (0, 1):
                raise ValueError("Binary condition covariates require zero or one")
        if "target" in row and not finite(row["target"]):
            raise ValueError("Finite observed targets required")


def inference_inputs(rows):
    return [{key: row[key] for key in ("id", "protein", "group", "values")} for row in rows]


def require_independent(model, rows):
    if set(model["training_ids"]) & {row["id"] for row in rows}:
        raise ValueError("Training/assessment observation overlap")
    if set(model["training_groups"]) & {row["group"] for row in rows}:
        raise ValueError("Training/assessment independent-group overlap")


def development_split(rows):
    """Hold out whole source-defined groups within the permitted training cohort."""
    groups = sorted({r["group"] for r in rows}, key=lambda g: digest({"seed": 314159, "group": g}))
    if len(groups) < 2:
        raise ValueError("Training feedback requires at least two independent groups")
    selected = set(groups[: max(1, min(len(groups) - 1, int(len(groups) * 0.8)))])
    training = sorted([r for r in rows if r["group"] in selected], key=lambda r: r["id"])
    feedback = sorted([r for r in rows if r["group"] not in selected], key=lambda r: r["id"])
    if len(training) < 8 or len(feedback) < 2:
        raise ValueError("Insufficient observations for group-held-out training feedback")
    return training, feedback


def prediction_rows_match(predictions, rows):
    expected = [(r["id"], r["protein"], r["group"], digest(r["values"])) for r in rows]
    observed = [(p.get("id"), p.get("protein"), p.get("group"), p.get("values_sha256")) for p in predictions]
    return observed == expected


def fit_candidate(rows, scope, recipe):
    validate_rows(rows, scope, labels=True)
    if not 8 <= len(rows) <= 600 or set(recipe) != {"source_code", "seed"} or type(recipe["seed"]) is not int:
        raise ValueError("Context fitting requires source, integer seed and 8..600 observations")
    reviewed = guard_source(recipe["source_code"])
    result = run_code(
        recipe["source_code"],
        {
            "operation": "fit",
            "input_kind": "features",
            "features": [r["values"] for r in rows],
            "targets": [r["target"] for r in rows],
            "seed": recipe["seed"],
        },
    )
    return {
        "schema": 1,
        "plugin_id": PLUGIN_ID,
        "scope": scope,
        "development_type": "agent_generated_model_code",
        "input_kind": "features",
        "recipe": recipe,
        "source_sha256": reviewed["source_sha256"],
        "training_sha256": digest(rows),
        "training_rows": len(rows),
        "training_ids": [r["id"] for r in rows],
        "training_groups": sorted({r["group"] for r in rows}),
        "training_mean": float(np.mean([r["target"] for r in rows])),
        "model_state": result["model"],
    }


def predict(model, rows, scope):
    if model.get("schema") != 1 or model.get("plugin_id") != PLUGIN_ID or model.get("scope") != scope:
        raise ValueError("Context model applicability, covariates or units mismatch")
    validate_rows(rows, scope, labels=False)
    code = model["recipe"]["source_code"]
    if hashlib.sha256(code.encode()).hexdigest() != model["source_sha256"]:
        raise ValueError("Context model source changed")
    result = run_code(
        code,
        {
            "operation": "predict",
            "input_kind": "features",
            "model": model["model_state"],
            "features": [r["values"] for r in rows],
            "seed": model["recipe"]["seed"],
        },
    )
    values = result.get("predictions")
    if not isinstance(values, list) or len(values) != len(rows) or any(not finite(v) for v in values):
        raise ModelOutputError("One finite numeric prediction required per contextual observation")
    return [
        {
            "id": row["id"],
            "protein": row["protein"],
            "group": row["group"],
            "values_sha256": digest(row["values"]),
            "prediction": float(value),
        }
        for row, value in zip(rows, values, strict=True)
    ]


def evaluate_with_predictions(model, rows, scope):
    from scipy.stats import spearmanr

    validate_rows(rows, scope, labels=True)
    require_independent(model, rows)
    predictions = predict(model, rows, scope)
    truth = np.asarray([r["target"] for r in rows], dtype=float)
    values = np.asarray([r["prediction"] for r in predictions], dtype=float)
    mse = float(np.square(truth - values).mean())
    baseline = float(np.square(truth - model["training_mean"]).mean())
    if baseline <= 0:
        raise ValueError("Degenerate assessment baseline")
    correlation = float(spearmanr(truth, values).statistic) if np.ptp(truth) > 0 and np.ptp(values) > 0 else 0.0
    metrics = {
        "rows": len(rows),
        "independent_groups": len({r["group"] for r in rows}),
        "rmse": math.sqrt(mse),
        "mae": float(np.abs(truth - values).mean()),
        "training_mean_rmse": math.sqrt(baseline),
        ACCEPTANCE_METRIC: 1 - mse / baseline,
        "spearman": correlation,
        "predictions_sha256": digest(predictions),
    }
    columns = scope["covariates"]
    temperature = [i for i, c in enumerate(columns) if c["name"] == "temperature" and c["unit"] == "degC"]
    reference = [i for i, c in enumerate(columns) if c["name"] == "reference_temperature" and c["unit"] == "degC"]
    if scope["unit"] == "ratio_to_within_replicate_reference" and len(temperature) == len(reference) == 1:
        ti, ri = temperature[0], reference[0]
        mask = np.asarray([not math.isclose(r["values"][ti], r["values"][ri], rel_tol=0, abs_tol=1e-12) for r in rows])
        if mask.any():
            part_mse = float(np.square(truth[mask] - values[mask]).mean())
            part_null = float(np.square(truth[mask] - model["training_mean"]).mean())
            metrics["non_reference_metrics"] = {
                "rows": int(mask.sum()),
                "rmse": math.sqrt(part_mse),
                "mae": float(np.abs(truth[mask] - values[mask]).mean()),
                ACCEPTANCE_METRIC: 1 - part_mse / part_null if part_null else None,
            }
    return metrics, predictions


def evaluate(model, rows, scope):
    return evaluate_with_predictions(model, rows, scope)[0]
