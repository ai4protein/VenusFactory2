"""Installed prediction assets may live outside a frozen source checkout."""

import hashlib
import json

import pytest

from backend.web_v2 import research_predictions as predictions
from mcp_tools.tools import capability_policy, runtime_tool_policy


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "src").mkdir(parents=True)
    config = {"web_ui": {
        "dataset_mapping_function": {"Solubility": ["sol"], "Stability": ["stable"]},
        "model_mapping_function": {"ESM2": "esm2"},
        "residue_mapping_function": {}, "model_residue_mapping_function": {},
        "model_adapter_mapping_function": {"esm2": "esm"},
    }}
    (source / "src/constant.json").write_text(json.dumps(config))
    monkeypatch.setattr(predictions, "ROOT", source)
    store = capability_policy.CapabilityStore(tmp_path / "capabilities.sqlite3")
    monkeypatch.setattr(capability_policy, "CapabilityStore", lambda: store)
    monkeypatch.setattr(runtime_tool_policy, "is_online_mode", lambda: True)
    return source, store


def install(root, dataset="sol", config=None, *, weights=True, settings=True):
    folder = root / dataset / "esm"
    folder.mkdir(parents=True)
    if weights:
        (folder / "lr5e-4_bt12k_ga8.pt").write_bytes(b"checkpoint fixture; never loaded")
    if settings:
        (folder / "lr5e-4_bt12k_ga8.json").write_text(json.dumps(config or {
            "problem_type": "single_label_classification", "normalize": "standard"}))
    return folder


def test_relocated_assets_preserve_catalog_identity_and_saved_scale(tmp_path, monkeypatch, catalog):
    source, _ = catalog
    first = tmp_path / "installation" / "weights"
    install(first)
    scale_config = {"problem_type": "regression", "normalize": "standard"}
    install(first, "stable", scale_config)
    monkeypatch.setenv("VENUS_CKPT_DIR", str(first))
    before = predictions.candidates({})
    assert [r["task"] for r in before] == ["Solubility", "Stability"]
    assert before[0]["output_scale"]["value_space"] == "class_index_and_probability"
    scale = before[1]["output_scale"]
    assert scale["value_space"] == "standardized_target"
    assert scale["checkpoint_config_sha256"] == hashlib.sha256(json.dumps(
        scale_config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    second = tmp_path / "relocated"
    first.rename(second)
    monkeypatch.setenv("VENUS_CKPT_DIR", str(second))
    after = predictions.candidates({})
    assert [{k: v for k, v in r.items() if k != "adapter_path"} for r in after] == [
        {k: v for k, v in r.items() if k != "adapter_path"} for r in before]
    assert all(r["adapter_path"].startswith(str(second) + "/") for r in after)
    assert not (source / "ckpt").exists()
    assert [p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file()] == ["src/constant.json"]


def test_explicit_source_fixture_does_not_consume_configured_host_assets(tmp_path, monkeypatch, catalog):
    source, _ = catalog
    host = tmp_path / "host"
    install(host, "stable")
    install(source / "ckpt", "sol")
    monkeypatch.setenv("VENUS_CKPT_DIR", str(host))
    assert [r["dataset"] for r in predictions.candidates({})] == ["stable"]
    explicit = predictions.candidates({}, root=source)
    assert [r["dataset"] for r in explicit] == ["sol"]
    assert explicit[0]["adapter_path"] == str(source / "ckpt/sol/esm")


@pytest.mark.parametrize("partial", [None, "weights", "settings"])
def test_missing_or_partial_installation_is_empty_without_download(tmp_path, monkeypatch, catalog, partial):
    weights = tmp_path / "not-installed"
    monkeypatch.setenv("VENUS_CKPT_DIR", str(weights))
    if partial:
        install(weights, weights=partial == "weights", settings=partial == "settings")
    assert predictions.candidates({}) == []
    assert predictions.options()["available"] is False
    if partial is None:
        assert not weights.exists()


def test_relocated_candidate_still_obeys_fresh_admin_switch(tmp_path, monkeypatch, catalog):
    _, store = catalog
    weights = tmp_path / "installation"
    install(weights)
    monkeypatch.setenv("VENUS_CKPT_DIR", str(weights))
    row = predictions.candidates({})[0]
    store.set("model:" + row["id"], False, 0, {
        "tools": [row["tool"]], "adapter": "function", "task": row["task"],
        "model": row["model"], "dataset": row["dataset"],
    }, "test-admin")
    assert predictions.candidates({}) == []
