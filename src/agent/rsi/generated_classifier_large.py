"""Versioned large-training classifier, retaining the original probability ABI.

Training admits up to 150,000 unique aligned sequences and four million amino
acids. Predictions remain bounded at 60,000 rows per operation. Numeric workers
retain their original 90-CPU-second / 120-wall-second / 4-GiB ceilings.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

try:
    from . import generated_classifier as classifier
    from .code_sandbox import guard_source, run_code
    from .protein_model import digest
except ImportError:  # Read-only standalone MCP bundle.
    import generated_classifier as classifier
    from code_sandbox import guard_source, run_code
    from protein_model import digest

PLUGIN_ID = "generated_classifier.large.v1"
ACCEPTANCE_METRIC = classifier.ACCEPTANCE_METRIC
MAX_TRAIN_ROWS = 150000
MAX_TOTAL_RESIDUES = 4000000
MAX_PREDICT_ROWS = 60000
validate_scope = classifier.validate_scope


def implementation_hash():
    root = Path(__file__).resolve().parent
    return hashlib.sha256(
        b"".join(
            (root / name).read_bytes()
            for name in [
                "generated_classifier_large.py",
                "generated_classifier.py",
                "code_sandbox.py",
                "code_worker.py",
                "protein_references.py",
                "numeric_runtime.py",
            ]
        )
    ).hexdigest()


def validate_rows(rows, scope, *, labels):
    validate_scope(scope)
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_TRAIN_ROWS:
        raise ValueError("Large classifier requires 1..150000 rows")
    if len(rows) * scope["sequence_length"] > MAX_TOTAL_RESIDUES:
        raise ValueError("Large classifier total-residue budget exceeded")
    ids, sequences = set(), set()
    for start in range(0, len(rows), MAX_PREDICT_ROWS):
        chunk = rows[start : start + MAX_PREDICT_ROWS]
        classifier.validate_rows(chunk, scope, labels=labels)
        chunk_ids = {r["id"] for r in chunk}
        chunk_sequences = {r["sequence"] for r in chunk}
        if ids & chunk_ids or sequences & chunk_sequences:
            raise ValueError("Duplicate identity or sequence across validation chunks")
        ids.update(chunk_ids)
        sequences.update(chunk_sequences)


def fit_candidate(rows, scope, recipe):
    validate_rows(rows, scope, labels=True)
    if len(rows) < 20 or set(recipe) != {"source_code", "seed"} or type(recipe["seed"]) is not int:
        raise ValueError("Large classifier needs source_code, integer seed and at least 20 rows")
    classes = len(scope["class_labels"])
    if {r["target"] for r in rows} != set(range(classes)):
        raise ValueError("Training must represent every declared class")
    reviewed = guard_source(recipe["source_code"])
    state = run_code(
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
        "source_sha256": reviewed["source_sha256"],
        "training_sha256": digest(rows),
        "training_rows": len(rows),
        "training_ids": [r["id"] for r in rows],
        "training_sequences": [r["sequence"] for r in rows],
        "training_class_prior": (np.bincount([r["target"] for r in rows], minlength=classes) / len(rows)).tolist(),
        "model_state": state["model"],
    }


def _probability_view(model, rows, scope):
    if model.get("plugin_id") != PLUGIN_ID or model.get("scope") != scope:
        raise ValueError("Large classifier applicability mismatch")
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_PREDICT_ROWS:
        raise ValueError("Prediction is limited to 60000 rows per worker operation")
    # A temporary ABI view does not alter the registered artifact or its identity.
    return {**model, "plugin_id": classifier.PLUGIN_ID}


def predict(model, rows, scope):
    return classifier.predict(_probability_view(model, rows, scope), rows, scope)


def evaluate_with_predictions(model, rows, scope):
    return classifier.evaluate_with_predictions(_probability_view(model, rows, scope), rows, scope)


def evaluate(model, rows, scope):
    return evaluate_with_predictions(model, rows, scope)[0]
