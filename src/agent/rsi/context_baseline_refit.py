"""Independent refits of fixed context controls; no production evaluator imports.

Uses the declared sklearn/scipy estimators and versions, with separately written
fitting, conditioning and verification. Raw assay preprocessing is not repeated.
Only training targets are accepted by the numerical refitter.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import expit
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

INPUTS = {"id", "protein", "group", "values"}
NUMERICAL = ("training_mean", "condition_ridge", "condition_random_forest", "condition_svr")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def read_bound(reference):
    path = Path(reference["path"])
    require(
        path.is_absolute() and not any(p.is_symlink() for p in [path, *path.parents]),
        "Ordinary absolute input required",
    )
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == reference["sha256"], "Bound input changed")
    return json.loads(data)


def query_rows(rows):
    return [{field: row[field] for field in INPUTS} for row in rows]


def thermal_spec(scope):
    columns = scope["covariates"]
    temperatures = [i for i, c in enumerate(columns) if (c["name"], c["unit"]) == ("temperature", "degC")]
    references = [i for i, c in enumerate(columns) if (c["name"], c["unit"]) == ("reference_temperature", "degC")]
    require(len(temperatures) <= 1 and len(references) <= 1, "Ambiguous thermal columns")
    if not temperatures:
        return (), None, None
    ti = temperatures[0]
    if scope["unit"] == "ratio_to_within_replicate_reference" and references:
        return ("thermal_reference_isotonic", "thermal_reference_sigmoid"), ti, references[0]
    return ("thermal_isotonic", "thermal_sigmoid"), ti, None


def _ordinary_curve(t, floor, amplitude, midpoint, width):
    return floor + amplitude * expit((midpoint - t) / width)


def _ratio_curve(pair, floor, midpoint, width):
    numerator = floor + (1.0 - floor) * expit((midpoint - pair[0]) / width)
    denominator = floor + (1.0 - floor) * expit((midpoint - pair[1]) / width)
    return numerator / np.maximum(denominator, 1e-12)


def _thermal(name, x, y, q, ti, ri):
    condition_columns = [i for i in range(x.shape[1]) if i not in {ti, ri}]
    fit_conditions = [tuple(row[condition_columns]) for row in x]
    query_conditions = [tuple(row[condition_columns]) for row in q]
    require(set(query_conditions) <= set(fit_conditions), "Unseen preparation condition")
    predictions = np.empty(len(q))
    for condition in sorted(set(query_conditions)):
        fitted = np.asarray([i for i, value in enumerate(fit_conditions) if value == condition])
        requested = np.asarray([i for i, value in enumerate(query_conditions) if value == condition])
        temperature, target = x[fitted, ti], y[fitted]
        require(len(fitted) >= 8 and len(set(temperature)) >= 4, "Insufficient thermal support")
        if ri is not None:
            require(np.all(target >= 0), "Negative reference ratio")
        if name.endswith("isotonic"):
            if ri is not None:
                require(len(set(x[fitted, ri])) == 1, "Inconsistent training reference")
            model = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(temperature, target)
            values = model.predict(q[requested, ti])
            if ri is not None:
                denominator = model.predict(q[requested, ri])
                require(np.all(denominator > 0), "Zero predicted reference")
                values = values / denominator
        elif ri is None:
            require(np.all(target >= 0) and max(target) > 0, "Invalid sigmoid signal")
            ceiling = 2 * float(max(target))
            bounds = (
                [0.0, 0.0, float(min(temperature)) - 10, 0.1],
                [ceiling, ceiling, float(max(temperature)) + 10, 100.0],
            )
            initial = [float(min(target)) / 2, float(max(target)), float(np.median(temperature)), 5.0]
            parameters, _ = curve_fit(_ordinary_curve, temperature, target, p0=initial, bounds=bounds, maxfev=20000)
            values = _ordinary_curve(q[requested, ti], *parameters)
        else:
            bounds = ([0.0, float(min(temperature)) - 10, 0.1], [0.999, float(max(temperature)) + 10, 100.0])
            parameters, _ = curve_fit(
                _ratio_curve,
                np.array([temperature, x[fitted, ri]]),
                target,
                p0=[0.05, float(np.median(temperature)), 5.0],
                bounds=bounds,
                maxfev=20000,
            )
            values = _ratio_curve(np.array([q[requested, ti], q[requested, ri]]), *parameters)
        predictions[requested] = values
    return predictions


def refit_predictions(training, inputs, scope):
    require(bool(training) and bool(inputs), "Empty fitting or query rows")
    for rows, keys in [(training, INPUTS | {"target"}), (inputs, INPUTS)]:
        require(all(set(row) == keys for row in rows), "Queries must be unlabelled and rows have exact fields")
        require(len({row["id"] for row in rows}) == len(rows), "Duplicate observation ID")
        require(all(row["protein"] == scope["protein"] for row in rows), "Protein scope mismatch")
    for key in ["id", "group"]:
        require(not {r[key] for r in training} & {r[key] for r in inputs}, "Partition overlap: " + key)
    columns = scope["covariates"]
    require(bool(columns), "Empty covariates")
    for row in training + inputs:
        require(len(row["values"]) == len(columns), "Feature dimension mismatch")
        for value, column in zip(row["values"], columns, strict=True):
            require(type(value) in (float, int) and math.isfinite(value), "Nonfinite or nonnumeric covariate")
            require(column["minimum"] <= value <= column["maximum"], "Covariate outside domain")
            require(column["unit"] != "binary_indicator" or value in (0, 1), "Invalid binary covariate")
    require(
        all(type(r["target"]) in (float, int) and math.isfinite(r["target"]) for r in training),
        "Invalid training target",
    )
    x = np.asarray([r["values"] for r in training], dtype=float)
    q = np.asarray([r["values"] for r in inputs], dtype=float)
    y = np.asarray([r["target"] for r in training], dtype=float)
    centre, scale = float(y.mean()), float(y.std()) or 1.0
    standardized = (y - centre) / scale
    scaler = StandardScaler().fit(x)
    estimators = {
        "condition_ridge": Ridge(alpha=1.0),
        "condition_random_forest": RandomForestRegressor(
            n_estimators=100, max_depth=10, min_samples_leaf=3, random_state=0, n_jobs=1
        ),
        "condition_svr": SVR(C=1.0, epsilon=0.1, gamma="scale"),
    }
    thermal, ti, ri = thermal_spec(scope)
    outcomes = {}
    for name in NUMERICAL + thermal:
        try:
            if name == "training_mean":
                # The declared dummy estimator operates on standardized targets.
                values = np.full(len(q), float(standardized.mean()) * scale + centre)
            elif name in estimators:
                raw = name == "condition_random_forest"
                model = estimators[name].fit(x if raw else scaler.transform(x), standardized)
                values = model.predict(q if raw else scaler.transform(q)) * scale + centre
            else:
                values = _thermal(name, x, y, q, ti, ri)
            require(np.isfinite(values).all(), "Nonfinite refitted prediction")
            outcomes[name] = {
                "status": "completed",
                "predictions": [
                    {"id": row["id"], "prediction": float(v)} for row, v in zip(inputs, values, strict=True)
                ],
            }
        except (ValueError, RuntimeError) as exc:
            outcomes[name] = {"status": "failed", "error": str(exc)}
    return outcomes


def verify_controls(task, controls, *, partition="future"):
    require(
        task["plugin_id"] in {"generated_context.v1", "generated_context.v2", "generated_context.v3"},
        "Context task required",
    )
    require(partition in {"validation", "future"}, "Assessment partition required")
    data = {name: read_bound(ref) for name, ref in task["data"].items()}
    require(set(data) == {"training", "validation", "future"}, "Three fixed partitions required")
    for left, right in [("training", "validation"), ("training", "future"), ("validation", "future")]:
        for field in ["id", "group"]:
            require(not {r[field] for r in data[left]} & {r[field] for r in data[right]}, "Partition overlap: " + field)
    inputs = query_rows(data[partition])
    independent = refit_predictions(data["training"], inputs, task["scope"])
    require(
        len(controls) == len(independent) and {c["method"] for c in controls} == set(independent),
        "Control coverage differs",
    )
    errors, failed = {}, []
    for saved in controls:
        name = saved["method"]
        expected = independent[name]
        require(saved["status"] == expected["status"], "Independent control status differs: " + name)
        if expected["status"] == "failed":
            require(not saved.get("predictions"), "Failed control contains predictions")
            failed.append(name)
            continue
        rows = saved["predictions"]
        require(len(rows) == len(inputs), "Prediction count differs")
        maximum = 0.0
        tolerance = 1e-7 if name.endswith("sigmoid") else 1e-9
        for actual, predicted in zip(rows, expected["predictions"], strict=True):
            require(
                set(actual) == {"id", "prediction"} and actual["id"] == predicted["id"], "Prediction identity differs"
            )
            value = actual["prediction"]
            require(type(value) in (float, int) and math.isfinite(value), "Invalid saved prediction")
            error = abs(value - predicted["prediction"])
            require(
                error <= tolerance * max(1.0, abs(predicted["prediction"])),
                "Independent baseline prediction differs: " + name,
            )
            maximum = max(maximum, error)
        errors[name] = maximum
    return {
        "task_id": task["id"],
        "task_sha256": canonical(task),
        "status": "independent_context_controls_refitted",
        "fit_partition": "training",
        "score_partition": partition,
        "query_sha256": canonical(inputs),
        "completed_methods": list(errors),
        "reproduced_failed_methods": failed,
        "maximum_absolute_prediction_error": max(errors.values(), default=0.0),
        "per_method_maximum_error": errors,
        "prediction_sha256": {
            n: canonical(v["predictions"]) for n, v in independent.items() if v["status"] == "completed"
        },
        "baseline_refit_performed": True,
        "source_preprocessing_reimplemented": False,
        "external_model_calls": 0,
        "new_autonomous_models": 0,
    }
