"""Plugin for actual agent-authored fit/predict Python, with isolated execution."""

from __future__ import annotations

import hashlib

import numpy as np

try:
    from .code_sandbox import ModelOutputError, guard_source, run_code
    from .protein_model import digest
except ImportError:  # read-only MCP bundle
    from code_sandbox import ModelOutputError, guard_source, run_code
    from protein_model import digest

PLUGIN_ID = "generated_python.v1"
AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


def validate_rows(rows, scope, *, labels):
    if set(scope) != {"protein", "assay", "target", "unit", "sequence_length"}:
        raise ValueError("missing scientific scope")
    length = scope["sequence_length"]
    if type(length) is not int or not 1 <= length <= 2048:
        raise ValueError("unsupported sequence length")
    if any(not isinstance(scope[k], str) or not scope[k].strip() for k in ["protein", "assay", "target", "unit"]):
        raise ValueError("empty scientific scope")
    if not 1 <= len(rows) <= 5000:
        raise ValueError("expected 1..5000 protein rows")
    ids, sequences = [], []
    for row in rows:
        if not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError("invalid sample id")
        seq = row.get("sequence")
        if not isinstance(seq, str) or len(seq) != length or set(seq) - AA:
            raise ValueError("invalid protein sequence or alignment")
        if labels and (type(row.get("target")) not in (int, float) or not np.isfinite(row["target"])):
            raise ValueError("labels must be finite measured numbers")
        ids.append(row["id"])
        sequences.append(seq)
    if len(set(ids)) != len(ids) or len(set(sequences)) != len(sequences):
        raise ValueError("duplicate protein samples")


def fit_candidate(rows: list[dict], scope: dict, recipe: dict) -> dict:
    validate_rows(rows, scope, labels=True)
    if not 10 <= len(rows) <= 600 or set(recipe) != {"source_code", "seed"} or type(recipe["seed"]) is not int:
        raise ValueError("generated model requires source_code, integer seed, 10..600 training rows")
    review = guard_source(recipe["source_code"])
    result = run_code(
        recipe["source_code"],
        {
            "operation": "fit",
            "sequences": [r["sequence"] for r in rows],
            "targets": [r["target"] for r in rows],
            "seed": recipe["seed"],
        },
    )
    return {
        "schema": 1,
        "plugin_id": PLUGIN_ID,
        "scope": scope,
        "development_type": "agent_generated_model_code",
        "recipe": recipe,
        "source_sha256": review["source_sha256"],
        "training_sha256": digest(rows),
        "training_rows": len(rows),
        "training_ids": [r["id"] for r in rows],
        "training_sequences": [r["sequence"] for r in rows],
        "training_mean": float(np.mean([r["target"] for r in rows])),
        "model_state": result["model"],
    }


def predict(model: dict, rows: list[dict], scope: dict) -> list[dict]:
    if model.get("schema") != 1 or model.get("plugin_id") != PLUGIN_ID or model.get("scope") != scope:
        raise ValueError("generated model applicability mismatch")
    validate_rows(rows, scope, labels=False)
    code = model["recipe"]["source_code"]
    if hashlib.sha256(code.encode()).hexdigest() != model["source_sha256"]:
        raise ValueError("generated model source changed")
    result = run_code(
        code,
        {
            "operation": "predict",
            "model": model["model_state"],
            "sequences": [r["sequence"] for r in rows],
            "seed": model["recipe"]["seed"],
        },
    )
    values = result.get("predictions")
    if (
        not isinstance(values, list)
        or len(values) != len(rows)
        or any(type(v) not in (float, int) or not np.isfinite(v) for v in values)
    ):
        raise ModelOutputError("predictions must be one finite number per input sequence")
    return [
        {"id": r["id"], "sequence": r["sequence"], "prediction": float(v)} for r, v in zip(rows, values, strict=True)
    ]


def evaluate_with_predictions(model, rows, scope):
    validate_rows(rows, scope, labels=True)
    if set(model["training_ids"]) & {r["id"] for r in rows} or set(model["training_sequences"]) & {
        r["sequence"] for r in rows
    }:
        raise ValueError("training/assessment sample overlap")
    predictions = predict(model, rows, scope)
    y = np.array([r["target"] for r in rows])
    p = np.array([r["prediction"] for r in predictions])
    mse = float(np.mean((y - p) ** 2))
    baseline = float(np.mean((y - model["training_mean"]) ** 2))
    if baseline <= 0:
        raise ValueError("degenerate assessment baseline")
    metrics = {
        "rows": len(rows),
        "rmse": mse**0.5,
        "training_mean_rmse": baseline**0.5,
        "mse_skill_vs_training_mean": 1 - mse / baseline,
        "predictions_sha256": digest(predictions),
    }
    return metrics, predictions


def evaluate(model, rows, scope):
    return evaluate_with_predictions(model, rows, scope)[0]
