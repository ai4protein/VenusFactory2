"""Portable fixed numerical controls; accepts explicit fitting and scoring rows only."""

from __future__ import annotations

import math
import time

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    from agent.rsi import reference_thermal_baselines as reference_thermal
    from .generated_context import validate_rows
except ImportError:
    import reference_thermal_baselines as reference_thermal
    from generated_context import validate_rows

METHODS = ("training_mean", "condition_ridge", "condition_random_forest", "condition_svr")
THERMAL_METHODS = ("thermal_isotonic", "thermal_sigmoid")


def thermal_methods(scope):
    if reference_thermal.reference_columns(scope) is not None:
        return reference_thermal.METHODS
    return (
        THERMAL_METHODS if any(c["name"] == "temperature" and c["unit"] == "degC" for c in scope["covariates"]) else ()
    )


def estimators():
    return {
        "training_mean": DummyRegressor(strategy="mean"),
        "condition_ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "condition_random_forest": RandomForestRegressor(
            n_estimators=100, max_depth=10, min_samples_leaf=3, random_state=0, n_jobs=1
        ),
        "condition_svr": make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.1, gamma="scale")),
    }


def sigmoid(temperature, lower, amplitude, midpoint, width):
    return lower + amplitude * expit((midpoint - temperature) / width)


def thermal_predictions(name, x_train, targets, x_query, temperature_column):
    """Fit each observed non-temperature condition separately, using training only."""
    other = [i for i in range(x_train.shape[1]) if i != temperature_column]
    training_groups = {tuple(row[other]) for row in x_train}
    query_groups = {tuple(row[other]) for row in x_query}
    if not query_groups <= training_groups:
        raise ValueError("Thermal reference has no training observations for a query condition")
    predictions = np.empty(len(x_query), dtype=float)
    for group in sorted(query_groups):
        train_mask = np.asarray([tuple(row[other]) == group for row in x_train])
        query_mask = np.asarray([tuple(row[other]) == group for row in x_query])
        temperature = x_train[train_mask, temperature_column]
        response = targets[train_mask]
        query = x_query[query_mask, temperature_column]
        if len(np.unique(temperature)) < 4 or len(temperature) < 8:
            raise ValueError("Thermal reference requires eight observations at four distinct temperatures")
        if name == "thermal_isotonic":
            estimator = IsotonicRegression(increasing=False, out_of_bounds="clip")
            predictions[query_mask] = estimator.fit(temperature, response).predict(query)
        else:
            # Fixed decreasing four-parameter response, no acceptance-set tuning.
            if np.any(response < 0) or float(np.max(response)) <= 0:
                raise ValueError("Thermal sigmoid requires nonnegative response with nonzero signal")
            ceiling = 2 * float(np.max(response))
            lower = [0.0, 0.0, float(temperature.min()) - 10.0, 0.1]
            upper = [ceiling, ceiling, float(temperature.max()) + 10.0, 100.0]
            initial = [float(response.min()) / 2, float(response.max()), float(np.median(temperature)), 5.0]
            parameters, _ = curve_fit(sigmoid, temperature, response, p0=initial, bounds=(lower, upper), maxfev=20000)
            predictions[query_mask] = sigmoid(query, *parameters)
    return predictions


def score_rows(training, validation, scope, *, timings=True):
    validate_rows(training, scope, labels=True)
    validate_rows(validation, scope, labels=True)
    if {r["id"] for r in training} & {r["id"] for r in validation} or {r["group"] for r in training} & {
        r["group"] for r in validation
    }:
        raise ValueError("Control fitting and scoring groups must be independent")
    x_train = np.asarray([r["values"] for r in training], dtype=float)
    x_valid = np.asarray([r["values"] for r in validation], dtype=float)
    y_train = np.asarray([r["target"] for r in training], dtype=float)
    y_valid = np.asarray([r["target"] for r in validation], dtype=float)
    location, scale = float(y_train.mean()), float(y_train.std())
    scale = scale if scale > 0 else 1.0
    null_mse = float(np.mean((y_valid - location) ** 2))
    thermal = [
        i
        for i, column in enumerate(scope["covariates"])
        if column["name"] == "temperature" and column["unit"] == "degC"
    ]
    reference_columns = reference_thermal.reference_columns(scope)
    methods = estimators()
    methods.update(dict.fromkeys(thermal_methods(scope)))
    outcomes = []
    for name, estimator in methods.items():
        started = time.monotonic()
        try:
            if estimator is None:
                if reference_columns is not None:
                    predictions = reference_thermal.predict(name, x_train, y_train, x_valid, reference_columns)
                else:
                    predictions = thermal_predictions(name, x_train, y_train, x_valid, thermal[0])
            else:
                estimator.fit(x_train, (y_train - location) / scale)
                predictions = estimator.predict(x_valid) * scale + location
            if not np.all(np.isfinite(predictions)):
                raise ValueError("Nonfinite baseline predictions")
        except (ValueError, RuntimeError) as exc:
            outcomes.append(
                {
                    "method": name,
                    "status": "failed",
                    "error": str(exc),
                    **({"seconds": time.monotonic() - started} if timings else {}),
                }
            )
            continue
        mse = float(np.mean((y_valid - predictions) ** 2))
        correlation = (
            float(spearmanr(y_valid, predictions).statistic)
            if np.ptp(y_valid) > 0 and np.ptp(predictions) > 0
            else None
        )
        outcomes.append(
            {
                "method": name,
                "status": "completed",
                **({"seconds": time.monotonic() - started} if timings else {}),
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
        if reference_columns is not None:
            ti, ri = reference_columns
            mask = ~np.isclose(x_valid[:, ti], x_valid[:, ri], rtol=0, atol=1e-12)
            if mask.any():
                non_reference_mse = float(np.mean((y_valid[mask] - predictions[mask]) ** 2))
                non_reference_null = float(np.mean((y_valid[mask] - location) ** 2))
                outcomes[-1]["non_reference_metrics"] = {
                    "rows": int(mask.sum()),
                    "rmse": math.sqrt(non_reference_mse),
                    "mae": float(np.mean(np.abs(y_valid[mask] - predictions[mask]))),
                    "mse_skill_vs_training_mean": 1 - non_reference_mse / non_reference_null
                    if non_reference_null
                    else None,
                }
    return outcomes
