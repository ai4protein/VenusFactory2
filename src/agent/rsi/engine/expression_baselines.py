"""Portable expression controls fitted only to explicitly supplied fitting rows."""

import math

import numpy as np
from scipy.stats import spearmanr

try:
    from .generated_context import require_independent, validate_rows
except ImportError:  # Frozen standalone MCP capability.
    from generated_context import require_independent, validate_rows

METHODS = ("transcript_only_ols", "culture_metadata_ols")
COVARIATES = ("author_log2_tpm", "od_rna", "od_protein", "ploidy")


def require_scope(scope):
    if scope.get("assay") != "population_protein_expression" or tuple(
        c["name"] for c in scope["covariates"]
    ) != COVARIATES:
        raise ValueError("Declared yeast expression covariates required")


def score_rows(training, scoring, scope):
    require_scope(scope)
    validate_rows(training, scope, labels=True)
    validate_rows(scoring, scope, labels=True)
    require_independent(
        {"training_ids": [r["id"] for r in training], "training_groups": [r["group"] for r in training]},
        scoring,
    )
    x = np.asarray([r["values"] for r in training], dtype=float)
    query = np.asarray([r["values"] for r in scoring], dtype=float)
    y = np.asarray([r["target"] for r in training], dtype=float)
    truth = np.asarray([r["target"] for r in scoring], dtype=float)
    null_mse = float(np.mean((truth - y.mean()) ** 2))
    outcomes = []
    for method, columns in zip(METHODS, ([0], [1, 2, 3]), strict=True):
        design = np.column_stack([np.ones(len(x)), x[:, columns]])
        coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
        prediction = np.column_stack([np.ones(len(query)), query[:, columns]]) @ coefficients
        if not np.all(np.isfinite(prediction)):
            raise ValueError("Nonfinite ordinary-least-squares control")
        mse = float(np.mean((truth - prediction) ** 2))
        correlation = float(spearmanr(truth, prediction).statistic) if np.ptp(truth) and np.ptp(prediction) else None
        outcomes.append(
            {
                "method": method,
                "status": "completed",
                "metrics": {
                    "rmse": math.sqrt(mse),
                    "mae": float(np.mean(np.abs(truth - prediction))),
                    "spearman": correlation,
                    "mse_skill_vs_training_mean": 1 - mse / null_mse if null_mse else None,
                },
                "predictions": [
                    {"id": row["id"], "prediction": float(p)} for row, p in zip(scoring, prediction, strict=True)
                ],
            }
        )
    return outcomes
