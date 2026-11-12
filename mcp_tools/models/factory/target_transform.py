"""Training-only scalar target transforms and checkpoint-bound, JSON-only state."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

SCHEMA = "venusfactory.target-transform.v1"
CHECKPOINT_SCHEMA = "venusfactory.checkpoint-target-transform.v1"
METHODS = {"none", "standard", "min_max", "robust", "log", "quantile"}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(value):
    return hashlib.sha256(_json(value)).hexdigest()


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _values(values):
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Target values must be finite numbers") from exc
    if not np.isfinite(result).all():
        raise ValueError("Target values must be finite numbers")
    return result


def _unit(value):
    if value is None:
        return "unknown"
    if not isinstance(value, str) or not value.strip() or len(value) > 64 or any(c in value for c in "\r\n\x00"):
        raise ValueError("target_unit must be a short explicit unit label")
    return value.strip()


def _inverse_metadata(method):
    if method == "quantile":
        return {"kind": "empirical_quantile", "approximate": True, "bounds": "clamp_to_training_range",
                "limitations": "Empirical quantile interpolation; predictions outside [0,1] clamp to training extrema. Out-of-range targets cannot be recovered exactly."}
    return {"kind": "identity" if method == "none" else "analytic", "approximate": False,
            "bounds": "log_domain" if method == "log" else "unbounded"}


def validate_target_transform(transform):
    """Reject malformed/unknown state rather than interpreting legacy guesses."""
    if not isinstance(transform, dict) or transform.get("schema") != SCHEMA:
        raise ValueError("Unsupported target transform schema")
    method = transform.get("method")
    if not isinstance(method, str) or method not in METHODS or transform.get("fit_scope") != "train" or transform.get("target_shape") != "scalar":
        raise ValueError("Invalid target transform contract")
    if type(transform.get("training_count")) is not int or transform["training_count"] < 1:
        raise ValueError("Invalid target transform training count")
    fingerprint = transform.get("training_targets_sha256")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint):
        raise ValueError("Invalid target transform training fingerprint")
    if not isinstance(transform.get("label_column"), str) or not transform["label_column"]:
        raise ValueError("Target transform requires a label column")
    if not isinstance(transform.get("target_unit"), str) or type(transform.get("constant_training_targets")) is not bool:
        raise ValueError("Target transform requires explicit unit and constant-target metadata")
    _unit(transform["target_unit"])
    if transform.get("inverse") != _inverse_metadata(method):
        raise ValueError("Target transform inverse semantics changed")
    params = transform.get("parameters")
    if not isinstance(params, dict):
        raise ValueError("Invalid target transform parameters")
    expected = {"none": set(), "standard": {"mean", "scale", "ddof"},
                "min_max": {"minimum", "maximum", "scale"}, "robust": {"center", "scale"},
                "log": {"offset"}, "quantile": {"quantiles", "references"}}[method]
    if set(params) != expected:
        raise ValueError("Invalid target transform parameter fields")
    if method == "quantile":
        q, ref = _values(params["quantiles"]), _values(params["references"])
        if q.ndim != 1 or ref.shape != q.shape or not 1 <= q.size <= 10000:
            raise ValueError("Invalid quantile target transform shape")
        if np.any(np.diff(q) < 0) or np.any(np.diff(ref) <= 0) or not np.array_equal(ref, np.linspace(0, 1, len(ref))):
            raise ValueError("Invalid quantile target transform ordering")
    elif params:
        if any(type(v) not in (int, float) for v in params.values()) or not np.isfinite(list(params.values())).all():
            raise ValueError("Target transform parameters must be finite numbers")
        if "scale" in params and params["scale"] <= 0:
            raise ValueError("Target transform scale must be positive")
        if method == "standard" and params["ddof"] != 0:
            raise ValueError("Unsupported standardization ddof")
        if method == "min_max" and params["maximum"] - params["minimum"] != params["scale"]:
            raise ValueError("Target transform range mismatch")
    _json(transform)
    return transform


def fit_target_transform(values, method, *, label_column="label", target_unit=None, seed=0, offset=1.0, n_quantiles=1000):
    """Fit scalar training labels only. standard/min_max reject zero variance.

    robust retains sklearn's effective scale (including 1 for zero IQR).
    Quantile state persists the fitted empirical mapping, not the estimator.
    """
    method = "none" if method is None else method
    if not isinstance(method, str) or method not in METHODS:
        raise ValueError(f"Unsupported target normalization: {method}")
    train = _values(values)
    if train.ndim != 1 or not train.size:
        raise ValueError("Target normalization requires nonempty scalar training labels")
    params = {}
    if method == "standard":
        params = {"mean": float(train.mean()), "scale": float(train.std(ddof=0)), "ddof": 0}
    elif method == "min_max":
        params = {"minimum": float(train.min()), "maximum": float(train.max()), "scale": float(train.max() - train.min())}
    elif method == "robust":
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler().fit(train.reshape(-1, 1))
        params = {"center": float(scaler.center_[0]), "scale": float(scaler.scale_[0])}
    elif method == "log":
        params = {"offset": float(offset)}
        if not np.isfinite(params["offset"]) or np.any(train + params["offset"] <= 0):
            raise ValueError("Log normalization requires target + offset > 0")
    elif method == "quantile":
        from sklearn.preprocessing import QuantileTransformer
        if type(n_quantiles) is not int or not 1 <= n_quantiles <= 10000:
            raise ValueError("n_quantiles must be an integer in 1..10000")
        estimator = QuantileTransformer(n_quantiles=min(n_quantiles, len(train)), output_distribution="uniform", random_state=seed)
        estimator.fit(train.reshape(-1, 1))
        params = {"quantiles": estimator.quantiles_[:, 0].tolist(), "references": estimator.references_.tolist()}
    if method in {"standard", "min_max"} and params["scale"] == 0:
        raise ValueError(f"{method} normalization is undefined for constant training targets")
    result = {"schema": SCHEMA, "method": method, "parameters": params, "fit_scope": "train", "target_shape": "scalar",
              "label_column": label_column, "target_unit": _unit(target_unit), "training_count": len(train),
              "training_targets_sha256": _sha(train.tolist()), "constant_training_targets": bool(np.ptp(train) == 0),
              "inverse": _inverse_metadata(method)}
    return validate_target_transform(result)


def transform_targets(values, transform):
    """Transform with existing state; never fit on validation/test observations."""
    state = validate_target_transform(transform)
    method, p = state["method"], state["parameters"]
    x = _values(values)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if method == "standard":
            result = (x - p["mean"]) / p["scale"]
        elif method == "min_max":
            result = (x - p["minimum"]) / p["scale"]
        elif method == "robust":
            result = (x - p["center"]) / p["scale"]
        elif method == "log":
            if np.any(x + p["offset"] <= 0):
                raise ValueError("Log normalization requires target + offset > 0")
            result = np.log(x + p["offset"])
        elif method == "quantile":
            q, ref = np.asarray(p["quantiles"]), np.asarray(p["references"])
            # Match sklearn's symmetric interpolation at duplicated quantiles.
            result = .5 * (np.interp(x, q, ref) - np.interp(-x, -q[::-1], -ref[::-1]))
            result = np.where(x == q[-1], 1., result)
            result = np.where(x == q[0], 0., result)
        else:
            result = x.copy()
    return _values(result)


def inverse_targets(values, transform):
    """Return an ndarray with input shape; callers explicitly handle tensor devices."""
    state = validate_target_transform(transform)
    method, p = state["method"], state["parameters"]
    x = _values(values)
    with np.errstate(over="ignore", invalid="ignore"):
        if method == "standard":
            result = x * p["scale"] + p["mean"]
        elif method == "min_max":
            result = x * p["scale"] + p["minimum"]
        elif method == "robust":
            result = x * p["scale"] + p["center"]
        elif method == "log":
            result = np.exp(x) - p["offset"]
        elif method == "quantile":
            result = np.interp(x, p["references"], p["quantiles"])
        else:
            result = x.copy()
    return _values(result)


def checkpoint_transform_path(model_path):
    return Path(model_path).with_suffix(".target-transform.json")


def _check_config(transform, config):
    config = config or {}
    if config.get("target_transform_schema") not in (None, SCHEMA):
        raise ValueError("Unsupported declared target transform schema")
    if config.get("problem_type", "regression") != "regression" or config.get("num_labels", 1) != 1:
        raise ValueError("Checkpoint target transform requires scalar regression")
    if "normalize" in config and (config["normalize"] or "none") != transform["method"]:
        raise ValueError("Checkpoint normalization differs from its target transform")
    if "label_column_name" in config and config["label_column_name"] != transform["label_column"]:
        raise ValueError("Checkpoint target column differs from its target transform")
    if config.get("target_unit") is not None and _unit(config["target_unit"]) != transform["target_unit"]:
        raise ValueError("Checkpoint target unit differs from its target transform")
    if config.get("target_transform") is not None and config["target_transform"] != transform:
        raise ValueError("Checkpoint configuration target transform differs from its bound state")


def save_checkpoint_transform(model_path, transform, *, config=None):
    """Atomically publish state after checkpoint save, without changing .pt format."""
    validate_target_transform(transform)
    _check_config(transform, config)
    path = checkpoint_transform_path(model_path)
    receipt = {"schema": CHECKPOINT_SCHEMA, "checkpoint_sha256": _file_sha(model_path),
               "transform_sha256": _sha(transform), "transform": transform}
    data = _json(receipt) + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return receipt


def load_checkpoint_transform(model_path, config=None):
    """Legacy absence is explicit None; declared, corrupt or mismatched state fails."""
    config = config or {}
    path = checkpoint_transform_path(model_path)
    if not path.exists():
        if config.get("target_transform_schema") is not None or config.get("target_transform") is not None:
            raise ValueError("Declared checkpoint target transform is missing")
        return None
    try:
        if path.is_symlink() or path.stat().st_size > 2_000_000:
            raise ValueError("Invalid checkpoint target transform file")
        receipt = json.loads(path.read_text())
        if not isinstance(receipt, dict) or receipt.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("Unsupported checkpoint target transform schema")
        transform = validate_target_transform(receipt.get("transform"))
        if receipt.get("checkpoint_sha256") != _file_sha(model_path) or receipt.get("transform_sha256") != _sha(transform):
            raise ValueError("Checkpoint target transform hash mismatch")
        _check_config(transform, config)
        return transform
    except (OSError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid checkpoint target transform") from exc


def target_output_scale(config=None, transform=None, *, inverse_applied=False):
    """Extend the existing interpretation; units come only from saved metadata."""
    from mcp_tools.models.factory.prediction.output_scale import output_scale
    result = output_scale(config)
    if transform is None:
        if inverse_applied:
            raise ValueError("Cannot inverse-transform without verified target state")
        return result
    validate_target_transform(transform)
    _check_config(transform, config)
    result.update(inverse_transform_available=True, inverse_transform_applied=bool(inverse_applied),
                  transform_schema=SCHEMA, transform_sha256=_sha(transform), inverse=transform["inverse"])
    if inverse_applied:
        result.update(value_space="original_target", normalization=transform["method"], unit=transform["target_unit"])
    return result
