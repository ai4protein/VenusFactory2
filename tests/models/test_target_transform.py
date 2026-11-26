"""Checkpoint-bound regression transforms: no heldout fitting or guessed units."""
import copy
import hashlib
import json

import numpy as np
import pytest
from sklearn.preprocessing import QuantileTransformer, RobustScaler

from mcp_tools.models.factory import target_transform as tt
from mcp_tools.models.factory.training.data.norm import normalize_dataset


@pytest.mark.parametrize("method", [None, "standard", "min_max", "robust", "log", "quantile"])
def test_roundtrip_serialized_transform_uses_only_training_split(method):
    train = [{"label": x, "aa_seq": "AA"} for x in [1., 3., 5., 9., 17.]]
    val = [{"label": 7.}]
    test = [{"label": 11.}]
    originals = copy.deepcopy((train, val, test))
    a, b, c, state = normalize_dataset(train, val, test, method, target_unit="nM", return_transform=True)
    assert (train, val, test) == originals
    state = json.loads(json.dumps(state, allow_nan=False))
    for source, transformed in zip(originals, (a, b, c)):
        restored = tt.inverse_targets([row["label"] for row in transformed], state)
        np.testing.assert_allclose(restored, [row["label"] for row in source])
    _, _, _, other = normalize_dataset(train, [{"label": 100000.}], [{"label": 1e7}], method,
                                        target_unit="nM", return_transform=True)
    assert state == other
    assert state["training_count"] == 5
    assert state["fit_scope"] == "train"
    scale = tt.target_output_scale({"problem_type": "regression", "normalize": method}, state, inverse_applied=True)
    assert scale["value_space"] == "original_target" and scale["unit"] == "nM"


def test_standard_uses_population_sd_and_preserves_shapes():
    state = tt.fit_target_transform([10., 14.], "standard")
    assert state["parameters"] == {"mean": 12., "scale": 2., "ddof": 0}
    shape = np.array([[10., 12.], [14., 18.]])
    np.testing.assert_allclose(tt.transform_targets(shape, state), [[-1., 0.], [1., 3.]])
    np.testing.assert_allclose(tt.inverse_targets(tt.transform_targets(shape, state), state), shape)
    assert tt.inverse_targets(np.array(0.), state).shape == ()


@pytest.mark.parametrize("train", [[1., 1., 1.], [1., 1., 1., 1., 30.], [1., 2., 5., 9., 30.]])
def test_robust_retains_sklearn_effective_scale(train):
    reference = RobustScaler().fit(np.asarray(train).reshape(-1, 1))
    state = tt.fit_target_transform(train, "robust")
    x = np.asarray([0., 1., 15., 100.]).reshape(-1, 1)
    np.testing.assert_allclose(tt.transform_targets(x, state), reference.transform(x))
    np.testing.assert_allclose(tt.inverse_targets(tt.transform_targets(x, state), state), x)


@pytest.mark.parametrize("train", [[1., 1., 1.], [1., 1., 2., 2., 4., 9.], [1., 2., 5., 9., 30.]])
def test_quantile_reference_and_explicit_clamped_inverse(train):
    reference = QuantileTransformer(n_quantiles=len(train), output_distribution="uniform", random_state=0)
    reference.fit(np.asarray(train).reshape(-1, 1))
    state = tt.fit_target_transform(train, "quantile")
    x = np.asarray([0., 1., 1.5, 2., 5., 15., 100.]).reshape(-1, 1)
    np.testing.assert_allclose(tt.transform_targets(x, state), reference.transform(x))
    y = np.asarray([-.1, 0., .1, .5, .9, 1., 1.1]).reshape(-1, 1)
    np.testing.assert_allclose(tt.inverse_targets(y, state), reference.inverse_transform(y))
    assert state["inverse"]["approximate"] is True
    assert state["inverse"]["bounds"] == "clamp_to_training_range"
    assert tt.inverse_targets([-9., 9.], state).tolist() == [min(train), max(train)]


@pytest.mark.parametrize("method", ["standard", "min_max"])
def test_undefined_constant_training_scale_fails_explicitly(method):
    with pytest.raises(ValueError, match="constant training"):
        tt.fit_target_transform([2., 2.], method)


@pytest.mark.parametrize("values", [[], [float("nan")], [1., float("inf")], [[1., 2.], [3., 4.]]])
def test_fit_rejects_nonfinite_empty_or_nonscalar_labels(values):
    with pytest.raises(ValueError):
        tt.fit_target_transform(values, "standard")


def test_log_domain_and_inverse_overflow_are_not_silent():
    with pytest.raises(ValueError, match="offset"):
        tt.fit_target_transform([-1., 1.], "log")
    state = tt.fit_target_transform([0., 1., 2.], "log")
    with pytest.raises(ValueError, match="offset"):
        tt.transform_targets([-2.], state)
    with pytest.raises(ValueError, match="finite"):
        tt.inverse_targets([1000.], state)


def saved_checkpoint(tmp_path, method="standard"):
    path = tmp_path / "weights.pt"
    path.write_bytes(b"synthetic CPU checkpoint bytes")
    state = tt.fit_target_transform([10., 14.], method, target_unit="nM")
    config = {"problem_type": "regression", "num_labels": 1, "normalize": method,
              "target_transform_schema": tt.SCHEMA, "target_transform": state, "target_unit": "nM"}
    tt.save_checkpoint_transform(path, state, config=config)
    return path, state, config


def test_checkpoint_binding_survives_relocation_and_config_paths(tmp_path):
    path, state, config = saved_checkpoint(tmp_path)
    moved = tmp_path / "published.pt"
    moved.write_bytes(path.read_bytes())
    tt.checkpoint_transform_path(moved).write_bytes(tt.checkpoint_transform_path(path).read_bytes())
    config.update(output_dir="different/relative/directory", model_path="published.pt")
    assert tt.load_checkpoint_transform(moved, config) == state
    receipt = json.loads(tt.checkpoint_transform_path(moved).read_text())
    assert receipt["checkpoint_sha256"] == hashlib.sha256(moved.read_bytes()).hexdigest()


@pytest.mark.parametrize("change", ["weights", "statistics", "receipt_schema", "transform_schema", "unit", "method", "classification"])
def test_checkpoint_or_semantic_tampering_fails_closed(tmp_path, change):
    path, _, config = saved_checkpoint(tmp_path)
    sidecar = tt.checkpoint_transform_path(path)
    receipt = json.loads(sidecar.read_text())
    if change == "weights":
        path.write_bytes(b"different weights")
    elif change == "statistics":
        receipt["transform"]["parameters"]["mean"] += 1
    elif change == "receipt_schema":
        receipt["schema"] = "future"
    elif change == "transform_schema":
        receipt["transform"]["schema"] = "future"
    elif change == "unit":
        config["target_unit"] = "Celsius"
    elif change == "method":
        config["normalize"] = "min_max"
    elif change == "classification":
        config["problem_type"] = "single_label_classification"
    sidecar.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        tt.load_checkpoint_transform(path, config)


def test_missing_legacy_stats_remain_unavailable_and_declared_stats_required(tmp_path):
    path = tmp_path / "legacy.pt"
    path.write_bytes(b"old checkpoint")
    config = {"problem_type": "regression", "normalize": "standard"}
    assert tt.load_checkpoint_transform(path, config) is None
    scale = tt.target_output_scale(config)
    assert scale["value_space"] == "standardized_target"
    assert scale["inverse_transform_available"] is False
    assert tt.target_output_scale()["value_space"] == "unknown"
    assert not tt.checkpoint_transform_path(path).exists()
    for declared in [dict(config, target_transform_schema=tt.SCHEMA), dict(config, target_transform_schema="unknown")]:
        with pytest.raises(ValueError, match="missing"):
            tt.load_checkpoint_transform(path, declared)
    with pytest.raises(ValueError, match="without verified"):
        tt.target_output_scale(config, inverse_applied=True)


def test_failed_atomic_publication_preserves_existing_manifest(tmp_path, monkeypatch):
    path, state, config = saved_checkpoint(tmp_path)
    sidecar = tt.checkpoint_transform_path(path)
    original = sidecar.read_bytes()
    def fail_replace(*_):
        raise OSError("simulated interrupted publication")
    monkeypatch.setattr(tt.os, "replace", fail_replace)
    with pytest.raises(OSError, match="interrupted"):
        tt.save_checkpoint_transform(path, state, config=config)
    assert sidecar.read_bytes() == original
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([path.name, sidecar.name])
    assert tt.load_checkpoint_transform(path, config) == state


@pytest.mark.parametrize("field,value", [("method", []), ("training_count", True), ("target_unit", None),
                                          ("constant_training_targets", "false"), ("fit_scope", "test")])
def test_invalid_state_contract_is_rejected(field, value):
    state = tt.fit_target_transform([1., 2.], "standard")
    state[field] = value
    with pytest.raises(ValueError):
        tt.validate_target_transform(state)
