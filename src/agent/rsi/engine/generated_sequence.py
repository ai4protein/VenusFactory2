"""Isolated agent-authored regression over variable-length protein sequences.

This separate interface preserves the frozen aligned-sequence ABI. Input
sequences remain intact: there is no implicit padding, truncation or alignment.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

try:
    from .code_sandbox import ModelOutputError, guard_source, run_code
    from .protein_model import digest
except ImportError:  # Read-only MCP bundle.
    from code_sandbox import ModelOutputError, guard_source, run_code
    from protein_model import digest

PLUGIN_ID = "generated_sequence.v1"
ACCEPTANCE_METRIC = "mse_skill_vs_training_mean"
AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
SCOPE_KEYS = {"protein", "assay", "target", "unit", "minimum_sequence_length", "maximum_sequence_length"}


def implementation_hash():
    root = Path(__file__).resolve().parent
    return hashlib.sha256(
        b"".join(
            (root / name).read_bytes()
            for name in (
                "generated_sequence.py",
                "code_sandbox.py",
                "code_worker.py",
                "protein_references.py",
                "numeric_runtime.py",
            )
        )
    ).hexdigest()


def validate_scope(scope):
    if not isinstance(scope, dict) or set(scope) != SCOPE_KEYS:
        raise ValueError("Variable-length regression requires explicit sequence length bounds and scientific scope")
    for key in ("protein", "assay", "target", "unit"):
        if not isinstance(scope[key], str) or not scope[key].strip():
            raise ValueError("Empty scientific scope")
    low, high = scope["minimum_sequence_length"], scope["maximum_sequence_length"]
    if type(low) is not int or type(high) is not int or not 1 <= low <= high <= 2048:
        raise ValueError("Sequence length bounds must be integers within 1..2048")


def validate_rows(rows, scope, *, labels):
    validate_scope(scope)
    if not isinstance(rows, list) or not 1 <= len(rows) <= 5000:
        raise ValueError("Expected 1..5000 protein records")
    ids, sequences = set(), set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Protein records must be mappings")
        identity, sequence = row.get("id"), row.get("sequence")
        if not isinstance(identity, str) or not identity.strip() or identity in ids:
            raise ValueError("Invalid or duplicate sample identity")
        if (
            not isinstance(sequence, str)
            or set(sequence) - AA
            or sequence in sequences
            or not scope["minimum_sequence_length"] <= len(sequence) <= scope["maximum_sequence_length"]
        ):
            raise ValueError("Invalid, duplicate or out-of-scope protein sequence")
        if labels and (type(row.get("target")) not in (int, float) or not np.isfinite(row["target"])):
            raise ValueError("Targets must be finite measured numbers")
        ids.add(identity)
        sequences.add(sequence)


def fit_candidate(rows, scope, recipe):
    validate_rows(rows, scope, labels=True)
    if (
        not isinstance(recipe, dict)
        or set(recipe) != {"source_code", "seed"}
        or type(recipe["seed"]) is not int
        or not 10 <= len(rows) <= 600
    ):
        raise ValueError("Sequence regression requires source_code, integer seed and 10..600 training rows")
    reviewed = guard_source(recipe["source_code"])
    result = run_code(
        recipe["source_code"],
        {
            "operation": "fit",
            "sequences": [row["sequence"] for row in rows],
            "targets": [row["target"] for row in rows],
            "seed": recipe["seed"],
        },
    )
    return {
        "schema": 1,
        "plugin_id": PLUGIN_ID,
        "scope": scope,
        "development_type": "agent_generated_model_code",
        "recipe": recipe,
        "source_sha256": reviewed["source_sha256"],
        "training_sha256": digest(rows),
        "training_rows": len(rows),
        "training_ids": [row["id"] for row in rows],
        "training_sequences": [row["sequence"] for row in rows],
        "training_mean": float(np.mean([row["target"] for row in rows])),
        "model_state": result["model"],
    }


def predict(model, rows, scope):
    if model.get("schema") != 1 or model.get("plugin_id") != PLUGIN_ID or model.get("scope") != scope:
        raise ValueError("Variable-length model applicability mismatch")
    validate_rows(rows, scope, labels=False)
    code = model["recipe"]["source_code"]
    if hashlib.sha256(code.encode()).hexdigest() != model["source_sha256"]:
        raise ValueError("Generated sequence model source changed")
    result = run_code(
        code,
        {
            "operation": "predict",
            "model": model["model_state"],
            "sequences": [row["sequence"] for row in rows],
            "seed": model["recipe"]["seed"],
        },
    )
    values = result.get("predictions")
    if (
        not isinstance(values, list)
        or len(values) != len(rows)
        or any(type(value) not in (int, float) or not np.isfinite(value) for value in values)
    ):
        raise ModelOutputError("Predict must return one finite numeric value per input sequence")
    return [
        {"id": row["id"], "sequence": row["sequence"], "prediction": float(value)}
        for row, value in zip(rows, values, strict=True)
    ]


def evaluate_with_predictions(model, rows, scope):
    validate_rows(rows, scope, labels=True)
    if set(model["training_ids"]) & {row["id"] for row in rows} or set(model["training_sequences"]) & {
        row["sequence"] for row in rows
    }:
        raise ValueError("Training/assessment sample overlap")
    predictions = predict(model, rows, scope)
    y = np.asarray([row["target"] for row in rows], dtype=float)
    predicted = np.asarray([row["prediction"] for row in predictions], dtype=float)
    mse = float(np.square(y - predicted).mean())
    baseline = float(np.square(y - model["training_mean"]).mean())
    if baseline <= 0:
        raise ValueError("Degenerate assessment baseline")
    return {
        "rows": len(rows),
        "rmse": mse**0.5,
        "training_mean_rmse": baseline**0.5,
        ACCEPTANCE_METRIC: 1 - mse / baseline,
        "predictions_sha256": digest(predictions),
    }, predictions


def evaluate(model, rows, scope):
    return evaluate_with_predictions(model, rows, scope)[0]
