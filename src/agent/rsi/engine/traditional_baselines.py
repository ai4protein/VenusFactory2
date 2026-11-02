"""Fixed, training-only classical regression controls for admitted model tasks."""

from __future__ import annotations

import math
import time

import numpy as np
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from agent.rsi.datasets.model_tasks import load_task
from .protein_model import digest

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
VERSION = "aac_classical_regression.v1"


def features(rows):
    # A transparent common representation; not a published domain predictor.
    vectors = []
    for row in rows:
        sequence = row["sequence"]
        if not sequence or set(sequence) - set(AMINO_ACIDS):
            raise ValueError("Canonical amino-acid sequence required")
        vectors.append([sequence.count(a) / len(sequence) for a in AMINO_ACIDS] + [math.log1p(len(sequence))])
    return np.asarray(vectors, dtype=float)


def estimators():
    return {
        "training_mean": DummyRegressor(strategy="mean"),
        "aac_ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "aac_random_forest": RandomForestRegressor(
            n_estimators=100, max_depth=10, min_samples_leaf=3, random_state=0, n_jobs=1
        ),
        "aac_svr": make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.1, gamma="scale")),
    }


def run_controls(task):
    if task["plugin_id"] == "generated_antibody.v1":
        from .antibody_baselines import run_controls as antibody_controls

        return antibody_controls(task)
    if task["plugin_id"] == "generated_assay.v1":
        from .assay_baselines import run_controls as assay_controls

        return assay_controls(task)
    if task["plugin_id"] in {"generated_context.v1", "generated_context.v2", "generated_context.v3"}:
        from .context_baselines import run_controls as context_controls

        return context_controls(task)
    if "classifier" in task["plugin_id"]:
        raise ValueError("This suite is regression-only")
    data = load_task(task)
    training, validation = data["training"], data["validation"]
    x_train, x_valid = features(training), features(validation)
    y_train = np.asarray([r["target"] for r in training], dtype=float)
    y_valid = np.asarray([r["target"] for r in validation], dtype=float)
    # Scale labels using training statistics, including for SVR's fixed C/epsilon.
    location, scale = float(y_train.mean()), float(y_train.std())
    scale = scale if scale > 0 else 1.0
    null_mse = float(np.mean((y_valid - location) ** 2))
    outcomes = []
    for name, estimator in estimators().items():
        started = time.monotonic()
        estimator.fit(x_train, (y_train - location) / scale)
        predictions = estimator.predict(x_valid) * scale + location
        if not np.all(np.isfinite(predictions)):
            raise ValueError("Nonfinite baseline predictions")
        mse = float(np.mean((y_valid - predictions) ** 2))
        correlation = None
        if np.ptp(y_valid) > 0 and np.ptp(predictions) > 0:
            correlation = float(spearmanr(y_valid, predictions).statistic)
        outcomes.append(
            {
                "method": name,
                "seconds": time.monotonic() - started,
                "metrics": {
                    "rmse": math.sqrt(mse),
                    "mae": float(np.mean(np.abs(y_valid - predictions))),
                    "spearman": correlation,
                    "mse_skill_vs_training_mean": 1 - mse / null_mse if null_mse else None,
                },
                "predictions": [
                    {"id": row["id"], "prediction": float(pred)}
                    for row, pred in zip(validation, predictions, strict=True)
                ],
            }
        )
    return {
        "schema": VERSION,
        "task_id": task["id"],
        "task_sha256": digest(task),
        "fit_partition": "training",
        "score_partition": "validation",
        "future_scored": False,
        "hyperparameters": "fixed in versioned implementation; no validation selection",
        "representation": "20 amino-acid fractions and log(1 + sequence length)",
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "rows": {"training": len(training), "validation": len(validation)},
        "outcomes": outcomes,
        "domain_baseline_comparison_complete": False,
        "new_autonomous_models": 0,
    }
