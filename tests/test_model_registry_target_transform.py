"""Registration preserves original training transforms without refitting them."""
import json
from pathlib import Path

import numpy as np
import pytest

from mcp_tools.models.factory.target_transform import (
    SCHEMA,
    checkpoint_transform_path,
    fit_target_transform,
    inverse_targets,
    load_checkpoint_transform,
    save_checkpoint_transform,
)
from mcp_tools.models.factory.training import model_registry as registry


@pytest.fixture
def registration(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "get_resources", lambda: None)
    monkeypatch.setattr(registry, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(registry, "_stored_path", lambda p: str(Path(p).relative_to(tmp_path)))
    folder = tmp_path / "experiment"
    folder.mkdir()
    weights = folder / "best_model.pt"
    weights.write_bytes(b"checkpoint-trained-on-train-only")
    config = {"problem_type": "regression", "num_labels": 1, "normalize": "standard", "output_dir": str(folder)}
    config_path = folder / "config.json"
    config_path.write_text(json.dumps(config))
    return tmp_path, weights, config_path, config


def test_registered_weights_retain_exact_training_receipt_and_original_values(registration):
    root, weights, config_path, config = registration
    transform = fit_target_transform([10., 20., 30.], "standard", target_unit="nM")
    save_checkpoint_transform(weights, transform, config=config)
    original = checkpoint_transform_path(weights).read_bytes()
    result = registry.register_trained_model(str(config_path), model_id="affinity")
    assert result["success"]
    saved = root / result["model_path"]
    saved_config = json.loads((root / result["config_path"]).read_text())
    assert saved.read_bytes() == weights.read_bytes()
    assert checkpoint_transform_path(saved).read_bytes() == original
    assert saved_config["target_transform_schema"] == SCHEMA
    assert json.loads(saved.with_suffix(".json").read_text()) == saved_config
    state = load_checkpoint_transform(saved, saved_config)
    np.testing.assert_allclose(inverse_targets([0.], state), [20.])
    assert state["training_count"] == 3 and state["target_unit"] == "nM"


@pytest.mark.parametrize("damage", ["weights", "sidecar", "missing", "configuration"])
def test_invalid_declared_training_state_is_not_registered(registration, damage):
    root, weights, config_path, config = registration
    config["target_transform_schema"] = SCHEMA
    state = fit_target_transform([1., 2., 3.], "standard")
    save_checkpoint_transform(weights, state, config=config)
    sidecar = checkpoint_transform_path(weights)
    if damage == "weights":
        weights.write_bytes(b"different-checkpoint")
    elif damage == "sidecar":
        receipt = json.loads(sidecar.read_text())
        receipt["transform"]["parameters"]["mean"] = 99.
        sidecar.write_text(json.dumps(receipt))
    elif damage == "missing":
        sidecar.unlink()
    else:
        config["normalize"] = "min_max"
    config_path.write_text(json.dumps(config))
    result = registry.register_trained_model(str(config_path), model_id="damaged")
    assert not result["success"]
    assert not (root / "ckpt/user_trained/registry.json").exists()
    assert not (root / "ckpt/user_trained/damaged").exists()


def test_legacy_registration_does_not_invent_normalization_statistics(registration):
    root, weights, config_path, config = registration
    result = registry.register_trained_model(str(config_path), model_id="legacy")
    assert result["success"]
    saved = root / result["model_path"]
    saved_config = json.loads((root / result["config_path"]).read_text())
    assert not checkpoint_transform_path(saved).exists()
    assert "target_transform" not in saved_config
    assert load_checkpoint_transform(saved, saved_config) is None


def test_existing_model_id_cannot_be_overwritten(registration):
    root, weights, config_path, config = registration
    state = fit_target_transform([1., 2., 3.], "standard")
    save_checkpoint_transform(weights, state, config=config)
    first = registry.register_trained_model(str(config_path), model_id="version_one")
    assert first["success"]
    registered = root / first["model_path"]
    original = {path.name: path.read_bytes() for path in registered.parent.iterdir()}
    index = registry.index_path().read_bytes()
    weights.write_bytes(b"new model with same requested id")
    save_checkpoint_transform(weights, state, config=config)
    second = registry.register_trained_model(str(config_path), model_id="version_one")
    assert not second["success"] and "immutable" in second["error"]
    assert registry.index_path().read_bytes() == index
    assert {path.name: path.read_bytes() for path in registered.parent.iterdir()} == original


def test_corrupted_copy_does_not_publish_or_leave_partial_artifact(registration, monkeypatch):
    root, weights, config_path, config = registration
    state = fit_target_transform([1., 2., 3.], "standard")
    save_checkpoint_transform(weights, state, config=config)
    original_copy = registry.shutil.copy2

    def corrupt_copy(source, destination):
        result = original_copy(source, destination)
        if Path(destination).suffix == ".pt":
            Path(destination).write_bytes(b"interrupted copy")
        return result

    monkeypatch.setattr(registry.shutil, "copy2", corrupt_copy)
    result = registry.register_trained_model(str(config_path), model_id="incomplete")
    assert not result["success"]
    assert not (root / "ckpt/user_trained/incomplete").exists()
    assert not list(registry.registry_root().glob(".register-*"))
    assert not registry.index_path().exists()


@pytest.mark.parametrize("after_commit", [False, True])
def test_index_failure_reconciles_before_rolling_back_new_artifact(registration, monkeypatch, after_commit):
    root, _, config_path, _ = registration
    save_index = registry._save_index

    def interrupted_save(index):
        if after_commit:
            save_index(index)
        raise OSError("simulated index interruption")

    monkeypatch.setattr(registry, "_save_index", interrupted_save)
    result = registry.register_trained_model(str(config_path), model_id="index_failure")
    assert result["success"] is after_commit
    assert (root / "ckpt/user_trained/index_failure").exists() is after_commit
    if not after_commit:
        monkeypatch.setattr(registry, "_save_index", save_index)
        retried = registry.register_trained_model(str(config_path), model_id="index_failure")
        assert retried["success"]
