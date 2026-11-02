"""Agent-generated numeric models over versioned, host-verified protein features.

Only the feature matrix and permitted labels enter isolated fit/predict calls.
The frozen encoder identity and full protein sequences remain in host evidence.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

try:
    from .code_sandbox import ModelOutputError, guard_source, run_code
    from .generated_model import validate_rows as validate_sequences
    from .protein_model import digest
except ImportError:  # Read-only standalone MCP bundle.
    from code_sandbox import ModelOutputError, guard_source, run_code
    from generated_model import validate_rows as validate_sequences
    from protein_model import digest

PLUGIN_ID = "generated_features.v1"
ACCEPTANCE_METRIC = "mse_skill_vs_training_mean"


def implementation_hash():
    root = Path(__file__).resolve().parent
    return hashlib.sha256(
        b"".join(
            (root / n).read_bytes()
            for n in [
                "generated_features.py",
                "generated_model.py",
                "code_sandbox.py",
                "code_worker.py",
                "protein_references.py",
                "numeric_runtime.py",
            ]
        )
    ).hexdigest()


def validate_scope(scope):
    if set(scope) != {"protein", "assay", "target", "unit", "sequence_length", "representation"}:
        raise ValueError("Feature model requires scientific scope and frozen representation")
    rep = scope["representation"]
    if not isinstance(rep, dict) or set(rep) != {"name", "dimensions", "encoder_sha256"}:
        raise ValueError("Invalid representation specification")
    if (
        not isinstance(rep["name"], str)
        or not rep["name"].strip()
        or type(rep["dimensions"]) is not int
        or not 1 <= rep["dimensions"] <= 4096
    ):
        raise ValueError("Invalid representation dimensions or name")
    sha = rep["encoder_sha256"]
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("Frozen encoder digest required")


def validate_rows(rows, scope, *, labels):
    validate_scope(scope)
    rep = scope["representation"]
    validate_sequences(rows, {k: v for k, v in scope.items() if k != "representation"}, labels=labels)
    for row in rows:
        values = row.get("features")
        if (
            not isinstance(values, list)
            or len(values) != rep["dimensions"]
            or any(type(x) not in (int, float) or not np.isfinite(x) for x in values)
        ):
            raise ValueError("Features must be a finite vector in the declared representation")


def training_examples(rows):
    return [
        {
            "id": r["id"],
            "sequence_prefix": r["sequence"][:24],
            "target": r["target"],
            "feature_head": r["features"][:8],
            "feature_l2": sum(x * x for x in r["features"]) ** 0.5,
        }
        for r in rows[:3]
    ]


def fit_candidate(rows, scope, recipe):
    validate_rows(rows, scope, labels=True)
    if not 10 <= len(rows) <= 600 or set(recipe) != {"source_code", "seed"} or type(recipe["seed"]) is not int:
        raise ValueError("Feature model requires source_code, integer seed and 10..600 training rows")
    reviewed = guard_source(recipe["source_code"])
    result = run_code(
        recipe["source_code"],
        {
            "operation": "fit",
            "input_kind": "features",
            "features": [r["features"] for r in rows],
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
        "training_sequences": [r["sequence"] for r in rows],
        "training_mean": float(np.mean([r["target"] for r in rows])),
        "model_state": result["model"],
    }


def predict(model, rows, scope):
    if model.get("schema") != 1 or model.get("plugin_id") != PLUGIN_ID or model.get("scope") != scope:
        raise ValueError("Generated feature model applicability or encoder mismatch")
    validate_rows(rows, scope, labels=False)
    code = model["recipe"]["source_code"]
    if hashlib.sha256(code.encode()).hexdigest() != model["source_sha256"]:
        raise ValueError("Generated feature model source changed")
    result = run_code(
        code,
        {
            "operation": "predict",
            "input_kind": "features",
            "model": model["model_state"],
            "features": [r["features"] for r in rows],
            "seed": model["recipe"]["seed"],
        },
    )
    values = result.get("predictions")
    if (
        not isinstance(values, list)
        or len(values) != len(rows)
        or any(type(v) not in (float, int) or not np.isfinite(v) for v in values)
    ):
        raise ModelOutputError("One finite numeric prediction is required for each input feature row")
    return [
        {"id": r["id"], "sequence": r["sequence"], "features_sha256": digest(r["features"]), "prediction": float(v)}
        for r, v in zip(rows, values, strict=True)
    ]


def evaluate_with_predictions(model, rows, scope):
    from scipy.stats import spearmanr

    validate_rows(rows, scope, labels=True)
    if set(model["training_ids"]) & {r["id"] for r in rows} or set(model["training_sequences"]) & {
        r["sequence"] for r in rows
    }:
        raise ValueError("Training/assessment sample overlap")
    predictions = predict(model, rows, scope)
    y = np.asarray([r["target"] for r in rows], dtype=float)
    p = np.asarray([r["prediction"] for r in predictions], dtype=float)
    mse = float(np.square(y - p).mean())
    baseline = float(np.square(y - model["training_mean"]).mean())
    if baseline <= 0:
        raise ValueError("Degenerate assessment baseline")
    rho = float(spearmanr(y, p).statistic) if np.std(y) > 0 and np.std(p) > 0 else 0.0
    metrics = {
        "rows": len(rows),
        "rmse": mse**0.5,
        "training_mean_rmse": baseline**0.5,
        ACCEPTANCE_METRIC: 1 - mse / baseline,
        "spearman": rho,
        "predictions_sha256": digest(predictions),
    }
    return metrics, predictions


def evaluate(model, rows, scope):
    return evaluate_with_predictions(model, rows, scope)[0]
