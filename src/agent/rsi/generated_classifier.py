"""Agent-authored sequence classification with host-validated probabilities.

The generated fit/predict ABI is unchanged; this plugin explicitly returns
class probabilities instead of silently treating classes as regression labels.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

try:
    from .code_sandbox import ModelOutputError, guard_source, run_code
    from .protein_model import digest
except ImportError:
    from code_sandbox import ModelOutputError, guard_source, run_code
    from protein_model import digest

PLUGIN_ID = "generated_classifier.v1"
ACCEPTANCE_METRIC = "brier_skill_vs_training_prior"
AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
SCOPE_KEYS = {"protein", "assay", "target", "unit", "sequence_length", "class_labels"}


def implementation_hash():
    root = Path(__file__).resolve().parent
    return hashlib.sha256(
        b"".join(
            (root / n).read_bytes()
            for n in [
                "generated_classifier.py",
                "code_sandbox.py",
                "code_worker.py",
                "protein_references.py",
                "numeric_runtime.py",
            ]
        )
    ).hexdigest()


def validate_scope(scope):
    if set(scope) != SCOPE_KEYS:
        raise ValueError("Classification scope requires explicit ordered class labels")
    if any(
        not isinstance(scope[k], str) or not scope[k].strip() for k in SCOPE_KEYS - {"sequence_length", "class_labels"}
    ):
        raise ValueError("Empty scientific scope")
    if type(scope["sequence_length"]) is not int or not 1 <= scope["sequence_length"] <= 2048:
        raise ValueError("Unsupported aligned sequence length")
    labels = scope["class_labels"]
    if (
        not isinstance(labels, list)
        or not 2 <= len(labels) <= 20
        or any(not isinstance(x, str) or not x.strip() for x in labels)
        or len(set(labels)) != len(labels)
    ):
        raise ValueError("Expected 2..20 distinct ordered class labels")


def validate_rows(rows, scope, *, labels):
    validate_scope(scope)
    if not isinstance(rows, list) or not 1 <= len(rows) <= 60000:
        raise ValueError("Expected 1..60000 protein rows")
    ids, sequences = set(), set()
    for row in rows:
        sid, seq = row.get("id"), row.get("sequence")
        if not isinstance(sid, str) or not sid or sid in ids:
            raise ValueError("Invalid or duplicate sample identity")
        if not isinstance(seq, str) or len(seq) != scope["sequence_length"] or set(seq) - AA or seq in sequences:
            raise ValueError("Invalid or duplicate protein sequence")
        if labels and (type(row.get("target")) is not int or not 0 <= row["target"] < len(scope["class_labels"])):
            raise ValueError("Classification labels must be declared integer class indices")
        ids.add(sid)
        sequences.add(seq)


def validate_probabilities(values, n, classes):
    if (
        not isinstance(values, list)
        or len(values) != n
        or any(
            not isinstance(row, list) or len(row) != classes or any(type(v) not in (int, float) for v in row)
            for row in values
        )
    ):
        raise ValueError("Predict must return one ordered probability vector per sequence")
    array = np.asarray(values, dtype=float)
    if (
        not np.isfinite(array).all()
        or (array < 0).any()
        or (array > 1).any()
        or not np.allclose(array.sum(axis=1), 1, rtol=0, atol=1e-7)
    ):
        raise ValueError("Probabilities must be finite, in [0,1], and sum to one per sequence")
    return array


def fit_candidate(rows, scope, recipe):
    validate_rows(rows, scope, labels=True)
    if not 20 <= len(rows) <= 50000 or set(recipe) != {"source_code", "seed"} or type(recipe["seed"]) is not int:
        raise ValueError("Classification requires source_code, integer seed, 20..50000 training rows")
    k = len(scope["class_labels"])
    if {r["target"] for r in rows} != set(range(k)):
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
        "training_class_prior": (np.bincount([r["target"] for r in rows], minlength=k) / len(rows)).tolist(),
        "model_state": state["model"],
    }


def predict(model, rows, scope):
    if model.get("schema") != 1 or model.get("plugin_id") != PLUGIN_ID or model.get("scope") != scope:
        raise ValueError("Classifier applicability mismatch")
    validate_rows(rows, scope, labels=False)
    code = model["recipe"]["source_code"]
    if hashlib.sha256(code.encode()).hexdigest() != model["source_sha256"]:
        raise ValueError("Generated classifier source changed")
    out = run_code(
        code,
        {
            "operation": "predict",
            "model": model["model_state"],
            "sequences": [r["sequence"] for r in rows],
            "seed": model["recipe"]["seed"],
        },
    )
    try:
        values = validate_probabilities(out.get("predictions"), len(rows), len(scope["class_labels"]))
    except ValueError as exc:
        raise ModelOutputError(str(exc)) from exc
    return [
        {"id": r["id"], "sequence": r["sequence"], "probabilities": p.tolist(), "prediction": int(np.argmax(p))}
        for r, p in zip(rows, values, strict=True)
    ]


def evaluate_with_predictions(model, rows, scope):
    from sklearn.metrics import average_precision_score, roc_auc_score

    validate_rows(rows, scope, labels=True)
    if set(model["training_ids"]) & {r["id"] for r in rows} or set(model["training_sequences"]) & {
        r["sequence"] for r in rows
    }:
        raise ValueError("Training/assessment sample overlap")
    predictions = predict(model, rows, scope)
    y = np.array([r["target"] for r in rows])
    p = np.array([r["probabilities"] for r in predictions])
    k = len(scope["class_labels"])
    truth = np.eye(k)[y]
    prior = np.array(model["training_class_prior"])
    validate_probabilities([prior.tolist()], 1, k)
    brier = float(np.mean(np.sum((p - truth) ** 2, axis=1)))
    baseline = float(np.mean(np.sum((prior - truth) ** 2, axis=1)))
    if baseline <= 0:
        raise ValueError("Degenerate classification baseline")
    recalls = [float(np.mean(p.argmax(axis=1)[y == c] == c)) if np.any(y == c) else None for c in range(k)]
    metrics = {
        "rows": len(rows),
        "class_counts": np.bincount(y, minlength=k).tolist(),
        "brier": brier,
        "training_prior_brier": baseline,
        ACCEPTANCE_METRIC: 1 - brier / baseline,
        "log_loss": float(-np.mean(np.log(np.maximum(p[np.arange(len(y)), y], 1e-15)))),
        "accuracy": float(np.mean(p.argmax(axis=1) == y)),
        "class_recall": recalls,
        "balanced_accuracy": float(np.mean([x for x in recalls if x is not None])),
        "predictions_sha256": digest(predictions),
    }
    if k == 2:
        metrics["positive_prevalence"] = float(np.mean(y))
        metrics["average_precision"] = float(average_precision_score(y, p[:, 1])) if 0 < int(y.sum()) < len(y) else None
        metrics["roc_auc"] = float(roc_auc_score(y, p[:, 1])) if 0 < int(y.sum()) < len(y) else None
    return metrics, predictions


def evaluate(model, rows, scope):
    return evaluate_with_predictions(model, rows, scope)[0]
