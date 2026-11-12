"""Describe saved target space without inventing a physical-unit conversion.

Legacy factory checkpoints save the normalization name, not a verified fitted
transform. Recording training-only scaler statistics is a separate migration.
"""

import hashlib
import json
from pathlib import Path

SCHEMA = "venusfactory.prediction-scale.v1"
_NORMALIZED = {
    "standard": "standardized_target",
    "min_max": "min_max_normalized_target",
    "robust": "robust_scaled_target",
    "log": "log_transformed_target",
    "quantile": "quantile_transformed_target",
}
_CLASSIFICATION = {
    "single_label_classification",
    "multi_label_classification",
    "residue_single_label_classification",
    "residue_multi_label_classification",
}


def output_scale(config=None):
    config = config if isinstance(config, dict) else {}
    problem = config.get("problem_type")
    result = {
        "schema": SCHEMA,
        "value_space": "unknown",
        "normalization": "unknown",
        "unit": "unknown",
        "inverse_transform_available": False,
        "inverse_transform_applied": False,
    }
    if problem in _CLASSIFICATION:
        result.update(value_space="class_index_and_probability", normalization="not_applicable", unit="not_applicable")
    elif problem == "regression":
        method = config.get("normalize", "unknown")
        if method is None:
            result.update(value_space="original_target", normalization="none")
            # A task name such as Optimal Temperature does not establish units.
            unit = config.get("target_unit")
            if isinstance(unit, str) and 0 < len(unit) <= 64 and "\n" not in unit:
                result["unit"] = unit
        elif isinstance(method, str) and method in _NORMALIZED:
            result.update(
                value_space=_NORMALIZED[method],
                normalization=method,
                unit="dimensionless" if method in {"standard", "min_max", "robust", "quantile"} else "unknown",
            )
    # Bind interpretation to this exact saved configuration, not a task-name map.
    if config:
        raw = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
        result["checkpoint_config_sha256"] = hashlib.sha256(raw).hexdigest()
    return result


def checkpoint_output_scale(adapter):
    try:
        path = Path(adapter) / "lr5e-4_bt12k_ga8.json"
        if path.stat().st_size > 1_000_000:
            return output_scale()
        return output_scale(json.loads(path.read_text()))
    except (OSError, ValueError, TypeError):
        return output_scale()


def result_output_scale(result, expected=None):
    metadata = result.get("biological_metadata") or {}
    declared = metadata.get("output_scale") if isinstance(metadata, dict) else None
    if expected is not None and declared != expected:
        raise ValueError("Prediction output scale does not match the saved checkpoint configuration")
    if declared is None:
        return output_scale()
    if not isinstance(declared, dict) or declared.get("schema") != SCHEMA:
        raise ValueError("Prediction output scale metadata is invalid")
    return dict(declared)
