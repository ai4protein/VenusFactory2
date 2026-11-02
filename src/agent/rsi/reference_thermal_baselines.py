"""Thermal controls that explicitly account for a per-sample reference temperature."""

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression

METHODS = ("thermal_reference_isotonic", "thermal_reference_sigmoid")


def reference_columns(scope):
    columns = scope["covariates"]
    temperature = [i for i, c in enumerate(columns) if c["name"] == "temperature" and c["unit"] == "degC"]
    reference = [i for i, c in enumerate(columns) if c["name"] == "reference_temperature" and c["unit"] == "degC"]
    if scope["unit"] == "ratio_to_within_replicate_reference" and len(temperature) == len(reference) == 1:
        return temperature[0], reference[0]
    return None


def ratio_curve(inputs, lower, midpoint, width):
    temperature, reference = inputs
    numerator = lower + (1 - lower) * expit((midpoint - temperature) / width)
    denominator = lower + (1 - lower) * expit((midpoint - reference) / width)
    return numerator / np.maximum(denominator, 1e-12)


def predict(name, x_train, targets, x_query, columns):
    """Fit only training observations; query reference temperatures are covariates."""
    if name not in METHODS:
        raise ValueError("Unknown reference-aware thermal control")
    temperature_column, reference_column = columns
    other = [i for i in range(x_train.shape[1]) if i not in columns]
    training_groups = {tuple(row[other]) for row in x_train}
    query_groups = {tuple(row[other]) for row in x_query}
    if not query_groups <= training_groups:
        raise ValueError("No training observations for the query preparation condition")
    result = np.empty(len(x_query), dtype=float)
    for group in sorted(query_groups):
        train_mask = np.asarray([tuple(row[other]) == group for row in x_train])
        query_mask = np.asarray([tuple(row[other]) == group for row in x_query])
        temperature = x_train[train_mask, temperature_column]
        reference = x_train[train_mask, reference_column]
        response = targets[train_mask]
        query = x_query[query_mask, temperature_column]
        query_reference = x_query[query_mask, reference_column]
        if len(np.unique(temperature)) < 4 or len(temperature) < 8:
            raise ValueError("Eight training measurements at four temperatures required")
        if np.any(response < 0):
            raise ValueError("Reference-ratio controls require nonnegative responses")
        if name == "thermal_reference_isotonic":
            if len(np.unique(reference)) != 1:
                raise ValueError(
                    "Isotonic reference transfer requires a common training reference within each condition"
                )
            fitted = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(temperature, response)
            denominator = fitted.predict(query_reference)
            if np.any(denominator <= 0):
                raise ValueError("Cannot normalize to a zero predicted reference response")
            result[query_mask] = fitted.predict(query) / denominator
        else:
            # Scale cancels in a within-sample ratio: fix the upper asymptote to one.
            parameters, _ = curve_fit(
                ratio_curve,
                np.stack([temperature, reference]),
                response,
                p0=[0.05, float(np.median(temperature)), 5.0],
                bounds=([0.0, float(temperature.min()) - 10, 0.1], [0.999, float(temperature.max()) + 10, 100.0]),
                maxfev=20000,
            )
            result[query_mask] = ratio_curve(np.stack([query, query_reference]), *parameters)
    return result
