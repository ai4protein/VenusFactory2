"""Fixed classical controls for patent-grouped paired-antibody measurements."""

from pathlib import Path

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

from . import generated_antibody
from .model_tasks import load_task
from .protein_model import digest
from .storage import sha256_file

METHODS = ("training_mean", "antibody_ridge", "antibody_random_forest", "antibody_svr", "paired_edit_nearest_neighbor")


def score_rows(training, scoring, scope):
    generated_antibody.validate_rows(training, scope, labels=True)
    generated_antibody.validate_rows(scoring, scope, labels=True)
    generated_antibody.require_disjoint_parts({"training": training, "scoring": scoring}, scope)
    x_train = np.asarray([r["values"] for r in training], dtype=float)
    x_score = np.asarray([r["values"] for r in scoring], dtype=float)
    y_train = np.asarray([r["target"] for r in training], dtype=float)
    y_score = np.asarray([r["target"] for r in scoring], dtype=float)
    location, scale = float(y_train.mean()), float(y_train.std())
    scale = scale or 1.0
    null_mse = float(np.square(y_score - location).mean())
    estimators = {
        "training_mean": DummyRegressor(strategy="mean"),
        "antibody_ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "antibody_random_forest": RandomForestRegressor(
            n_estimators=100, max_depth=10, min_samples_leaf=3, random_state=0, n_jobs=1
        ),
        "antibody_svr": make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.1, gamma="scale")),
    }
    outcomes = []
    for name, estimator in estimators.items():
        estimator.fit(x_train, (y_train - location) / scale)
        prediction = estimator.predict(x_score) * scale + location
        if not np.isfinite(prediction).all():
            raise ValueError("Nonfinite assay control predictions")
        mse = float(np.square(y_score - prediction).mean())
        outcomes.append(
            {
                "method": name,
                "status": "completed",
                "metrics": {
                    "rmse": mse**0.5,
                    "mae": float(np.abs(y_score - prediction).mean()),
                    "spearman": float(spearmanr(y_score, prediction).statistic)
                    if np.ptp(y_score) > 0 and np.ptp(prediction) > 0
                    else None,
                    "mse_skill_vs_training_mean": 1 - mse / null_mse if null_mse else None,
                },
                "predictions": [
                    {"id": row["id"], "inputs_sha256": generated_antibody.input_digest(row), "prediction": float(value)}
                    for row, value in zip(scoring, prediction, strict=True)
                ],
            }
        )
    from Bio.Align import PairwiseAligner

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score, aligner.mismatch_score = 0, -1
    aligner.open_gap_score = aligner.extend_gap_score = -1

    def distance(left, right):
        total = 0.0
        for chain in ("heavy", "light"):
            a, b = left[chain], right[chain]
            total += -aligner.score(a, b) / max(len(a), len(b)) if a and b else float(a != b)
        return total

    values = []
    for row in scoring:
        distances = np.asarray([distance(row, ref) for ref in training])
        values.append(float(y_train[distances == distances.min()].mean()))
    prediction = np.asarray(values)
    mse = float(np.square(y_score - prediction).mean())
    outcomes.append(
        {
            "method": "paired_edit_nearest_neighbor",
            "status": "completed",
            "metrics": {
                "rmse": mse**0.5,
                "mae": float(np.abs(y_score - prediction).mean()),
                "spearman": float(spearmanr(y_score, prediction).statistic)
                if np.ptp(y_score) > 0 and np.ptp(prediction) > 0
                else None,
                "mse_skill_vs_training_mean": 1 - mse / null_mse if null_mse else None,
            },
            "predictions": [
                {"id": row["id"], "inputs_sha256": generated_antibody.input_digest(row), "prediction": float(value)}
                for row, value in zip(scoring, prediction, strict=True)
            ],
        }
    )
    return outcomes


def run_controls(task, *, score_partition="validation"):
    if task["plugin_id"] != generated_antibody.PLUGIN_ID or score_partition not in {"validation", "future"}:
        raise ValueError("Explicit assay task and assessment partition required")
    data = load_task(task)
    outcomes = score_rows(data["training"], data[score_partition], task["scope"])
    domain_completed = 1
    return {
        "schema": "antibody_classical_regression.v1",
        "task_id": task["id"],
        "task_sha256": digest(task),
        "implementation_sha256": sha256_file(Path(__file__)),
        "fit_partition": "training",
        "score_partition": score_partition,
        "future_scored": score_partition == "future",
        "hyperparameters": "fixed; no assessment selection or fitting",
        "representation": task["scope"]["representation"],
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "outcomes": outcomes,
        "domain_baseline_comparison_complete": False,
        "declared_domain_controls_completed": domain_completed,
        "new_autonomous_models": 0,
    }
