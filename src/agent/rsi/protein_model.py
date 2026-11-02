"""Auditable CPU protein models with a portable JSON inference contract.

These are task-specific configured ridge models, not novel neural architectures.
Residue strings encode aligned mutable sites, not arbitrary full proteins.
No pickle, generated Python or provider credentials are needed for inference.
This module is also bundled into the read-only research MCP capability package.
"""

from __future__ import annotations

import hashlib
import json
from itertools import combinations

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
PLUGIN_ID = "aligned_site_ridge.v1"
SCOPE_KEYS = {"protein", "assay", "target", "unit", "sequence_length"}


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def validate_scope(scope: dict) -> None:
    if set(scope) != SCOPE_KEYS or any(
        not isinstance(scope[k], str) or not scope[k].strip() for k in SCOPE_KEYS - {"sequence_length"}
    ):
        raise ValueError("scope requires protein, assay, target, unit and sequence_length")
    if type(scope["sequence_length"]) is not int or not 1 <= scope["sequence_length"] <= 8:
        raise ValueError("this plugin supports 1..8 aligned mutable sites")


def validate_rows(rows: list[dict], scope: dict, *, labels: bool) -> None:
    validate_scope(scope)
    if not rows or len(rows) > 5000:
        raise ValueError("expected 1..5000 rows")
    ids, seqs = [], []
    for row in rows:
        if not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError("sample IDs must be nonempty strings")
        seq = row.get("sequence")
        if not isinstance(seq, str) or len(seq) != scope["sequence_length"] or set(seq) - set(AA):
            raise ValueError("noncanonical or incompatible aligned-site sequence")
        if labels and (type(row.get("target")) not in (int, float) or not np.isfinite(row["target"])):
            raise ValueError("targets must be finite measured numbers")
        ids.append(row["id"])
        seqs.append(seq)
    if len(set(ids)) != len(ids) or len(set(seqs)) != len(seqs):
        raise ValueError("duplicate sample IDs or sequences require an explicit replicate policy")


def features(sequences: list[str], encoding: str) -> np.ndarray:
    if encoding not in {"onehot", "onehot_pairwise"}:
        raise ValueError("unknown residue encoding")
    size = len(sequences[0])
    indices = np.array([[AA.index(aa) for aa in seq] for seq in sequences])
    base = np.eye(20)[indices].reshape(len(sequences), -1)
    if encoding == "onehot":
        return base
    pairs = [np.eye(400)[indices[:, a] * 20 + indices[:, b]] for a, b in combinations(range(size), 2)]
    return np.concatenate([base, *pairs], axis=1)


def fit_candidate(rows: list[dict], scope: dict, recipe: dict) -> dict:
    """Fit using training-only 5-fold CV; never consumes controller validation rows."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV, KFold

    validate_rows(rows, scope, labels=True)
    if not 10 <= len(rows) <= 600:
        raise ValueError("this CPU pilot requires 10..600 training rows")
    if set(recipe) != {"encoding", "alphas"}:
        raise ValueError("recipe must declare encoding and alphas only")
    alphas = recipe["alphas"]
    if (
        not isinstance(alphas, list)
        or not 1 <= len(alphas) <= 8
        or any(type(a) not in (int, float) or not np.isfinite(a) or not 0.001 <= a <= 10000 for a in alphas)
    ):
        raise ValueError("bounded positive ridge alphas required")
    x = features([r["sequence"] for r in rows], recipe["encoding"])
    y = np.array([r["target"] for r in rows])
    search = GridSearchCV(
        Ridge(),
        {"alpha": alphas},
        scoring="neg_mean_squared_error",
        cv=KFold(5, shuffle=True, random_state=314159),
        n_jobs=1,
        error_score="raise",
    )
    search.fit(x, y)
    estimator = search.best_estimator_
    return {
        "schema": 1,
        "plugin_id": PLUGIN_ID,
        "scope": scope,
        "development_type": "configured_task_specific_model",
        "recipe": recipe,
        "training_sha256": digest(rows),
        "training_rows": len(rows),
        "training_ids": [r["id"] for r in rows],
        "training_sequences": [r["sequence"] for r in rows],
        "training_mean": float(y.mean()),
        "selected_alpha": float(search.best_params_["alpha"]),
        "cv_folds": 5,
        "cv_seed": 314159,
        "cv_rmse": float(np.sqrt(-search.best_score_)),
        "coefficients": estimator.coef_.tolist(),
        "intercept": float(estimator.intercept_),
    }


def predict(model: dict, rows: list[dict], scope: dict) -> list[dict]:
    if model.get("schema") != 1 or model.get("plugin_id") != PLUGIN_ID or model.get("scope") != scope:
        raise ValueError("model plugin or scientific applicability does not match")
    validate_rows(rows, scope, labels=False)
    x = features([r["sequence"] for r in rows], model["recipe"]["encoding"])
    coef = np.asarray(model["coefficients"], dtype=float)
    if coef.shape != (x.shape[1],) or not np.isfinite(coef).all() or not np.isfinite(model["intercept"]):
        raise ValueError("invalid model parameters")
    values = x @ coef + model["intercept"]
    if not np.isfinite(values).all():
        raise ValueError("nonfinite predictions")
    return [
        {"id": row["id"], "sequence": row["sequence"], "prediction": float(value)}
        for row, value in zip(rows, values, strict=True)
    ]


def evaluate_with_predictions(model: dict, rows: list[dict], scope: dict) -> tuple[dict, list[dict]]:
    validate_rows(rows, scope, labels=True)
    if set(model["training_ids"]) & {r["id"] for r in rows} or set(model["training_sequences"]) & {
        r["sequence"] for r in rows
    }:
        raise ValueError("training/assessment sample overlap")
    preds = predict(model, rows, scope)
    y = np.array([r["target"] for r in rows])
    yhat = np.array([r["prediction"] for r in preds])
    mse = float(np.mean((y - yhat) ** 2))
    baseline = float(np.mean((y - model["training_mean"]) ** 2))
    if baseline <= 0:
        raise ValueError("degenerate constant-baseline evaluation")
    metrics = {
        "rows": len(rows),
        "rmse": mse**0.5,
        "training_mean_rmse": baseline**0.5,
        "mse_skill_vs_training_mean": 1 - mse / baseline,
        "predictions_sha256": digest(preds),
    }
    return metrics, preds


def evaluate(model, rows, scope):
    return evaluate_with_predictions(model, rows, scope)[0]
