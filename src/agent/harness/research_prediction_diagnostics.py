"""Compact prediction checks computed from host-observed model outputs."""
from collections import Counter
import math

REL_TOL = 1e-9
ABS_TOL = 1e-12


def prediction_diagnostics(task, predictions, training_targets, *, split):
    """Describe behavior without query labels, per-row outputs or an admission verdict."""
    if task not in {"regression", "classification"} or split not in {"validation", "test"}:
        raise ValueError("Declare the task and evaluated split")
    if not predictions or not training_targets:
        raise ValueError("Prediction diagnostics require predictions and training targets")
    allowed = (int, float) if task == "regression" else (int, float, str)
    for values in (predictions, training_targets):
        if not all(type(value) in allowed and
                   (isinstance(value, str) or math.isfinite(value)) for value in values):
            raise ValueError("Prediction diagnostics require finite scalar values")
    if task == "regression":
        reference = sum(training_targets) / len(training_targets)
        if not math.isfinite(reference):
            raise ValueError("Training reference is not finite")
        constant = math.isclose(min(predictions), max(predictions),
                                rel_tol=REL_TOL, abs_tol=ABS_TOL)
        matches_reference = all(math.isclose(value, reference,
                                rel_tol=REL_TOL, abs_tol=ABS_TOL) for value in predictions)
        method = "training_mean"
    else:
        reference = Counter(training_targets).most_common(1)[0][0]
        constant = len(set(predictions)) == 1
        matches_reference = all(value == reference for value in predictions)
        method = "training_majority"
    warnings = []
    if constant:
        warnings.append("constant_predictions")
    if matches_reference:
        warnings.append("matches_training_reference")
    result = {
        "schema": "prediction_diagnostics.v1", "split": split,
        "prediction_count": len(predictions), "distinct_predictions": len(set(predictions)),
        "constant_predictions": constant, "matches_training_reference": matches_reference,
        "reference_method": method, "warnings": warnings,
    }
    if task == "regression":
        result["comparison_tolerance"] = {"relative": REL_TOL, "absolute": ABS_TOL}
    return result
