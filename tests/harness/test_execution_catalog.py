import json
import sys
from dataclasses import replace

import pytest

from agent.harness import Budget, ModelBinding, Profile
from agent.harness.catalog import ExecutionCatalog, resolve_credential


def setup_catalog(tmp_path):
    first = ModelBinding("provider_a", "model-a", "https://a.example/v1", credential_env="TEST_MODEL_A_KEY")
    second = ModelBinding("provider_b", "model-b", "https://b.example/v1", credential_env="TEST_MODEL_B_KEY")
    base = Profile("science_agent", "langgraph", first, Budget(thinking=None))
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps(
            {
                "models": [
                    {"id": "a", "label": "Model A", "binding": base.public()["model"]},
                    {"id": "b", "label": "Model B", "binding": replace(base, model=second).public()["model"]},
                ]
            }
        )
    )
    return ExecutionCatalog(tmp_path, base, path=path), base


def test_harness_and_model_selection_are_orthogonal(tmp_path):
    catalog, base = setup_catalog(tmp_path)
    changed_harness, _ = catalog.select({"harness": "deepseek-harness", "model_id": "a"})
    assert changed_harness.model == base.model
    assert changed_harness.budget == base.budget and changed_harness.mode == base.mode
    changed_model, option = catalog.select({"harness": "langgraph", "model_id": "b"})
    assert changed_model.harness == base.harness and changed_model.budget == base.budget
    assert changed_model.model == option.binding
    assert changed_model.model.provider == "provider_b"


def test_catalog_rejects_incompatible_combinations_and_hides_secret_references(tmp_path, monkeypatch):
    catalog, _ = setup_catalog(tmp_path)
    monkeypatch.setenv("TEST_MODEL_A_KEY", "secret-a")
    monkeypatch.delenv("TEST_MODEL_B_KEY", raising=False)
    public = catalog.public({"python_bin": sys.executable, "kimi_bin": sys.executable, "dsh_bin": sys.executable})
    combos = {(x["harness"], x["model_id"]): x for x in public["combinations"]}
    assert combos["langgraph", "a"]["ready"]
    assert "credential_unavailable" in combos["langgraph", "b"]["reasons"]
    assert "adapter_not_implemented" not in combos["codex", "a"]["reasons"]
    assert not combos["codex", "a"]["ready"]
    assert "runtime_missing" in combos["codex", "a"]["reasons"]
    with_runtime = catalog.public({"python_bin": sys.executable, "codex_bin": sys.executable})
    codex = next(x for x in with_runtime["combinations"] if x["harness"] == "codex" and x["model_id"] == "a")
    assert codex["ready"]
    assert "mode_not_implemented_for_harness" in combos["role-graph", "a"]["reasons"]
    serialized = json.dumps(public)
    assert "secret-a" not in serialized and "TEST_MODEL_A_KEY" not in serialized
    assert "https://a.example" not in serialized


def test_credential_resolution_is_bound_to_selected_provider(tmp_path, monkeypatch):
    catalog, _ = setup_catalog(tmp_path)
    model = catalog.models["b"].binding
    monkeypatch.setenv("TEST_MODEL_A_KEY", "unrelated")
    monkeypatch.setenv("TEST_MODEL_B_KEY", "selected")
    assert resolve_credential(model) == "selected"
    monkeypatch.delenv("TEST_MODEL_B_KEY")
    cfg = tmp_path / "legacy.toml"
    cfg.write_text('[providers.provider_b]\nbase_url="https://wrong.example/v1"\napi_key="never-return"\n')
    with pytest.raises(ValueError, match="endpoint differs"):
        resolve_credential(model, cfg)


def test_untrusted_model_endpoints_and_unknown_selections_are_rejected(tmp_path):
    catalog, _ = setup_catalog(tmp_path)
    for selection in [
        {"harness": "langgraph", "model_id": "missing"},
        {"harness": "unknown", "model_id": "a"},
        {"harness": "langgraph", "model_id": "a", "base_url": "https://elsewhere.example"},
    ]:
        with pytest.raises(ValueError):
            catalog.select(selection)


@pytest.mark.parametrize("policy", ["empty_length_once_v1", "dmx_maas_once_v1"])
def test_incompatible_recovery_binding_disables_only_its_model_option(tmp_path, monkeypatch, policy):
    catalog, base = setup_catalog(tmp_path)
    if policy == "dmx_maas_once_v1":
        binding = replace(base.model, provider="dmx", base_url="https://www.dmxapi.cn/v1")
        base = replace(base, model=binding)
        catalog.models["a"] = replace(catalog.models["a"], binding=binding)
    else:
        option = catalog.models["b"]
        catalog.models["b"] = replace(option, binding=replace(option.binding, protocol="responses"))
    catalog.default_profile = replace(base, request_recovery=policy)
    monkeypatch.setenv("TEST_MODEL_A_KEY", "fixture-a")
    monkeypatch.setenv("TEST_MODEL_B_KEY", "fixture-b")
    public = catalog.public({"python_bin": sys.executable, "kimi_bin": sys.executable, "dsh_bin": sys.executable})
    combos = {(x["harness"], x["model_id"]): x for x in public["combinations"]}
    assert combos["langgraph", "a"]["ready"]
    assert not combos["langgraph", "b"]["ready"]
    assert "execution_policy_incompatible" in combos["langgraph", "b"]["reasons"]
    selected, _ = catalog.select({"harness": "langgraph", "model_id": "a"})
    assert selected.request_recovery == policy
    with pytest.raises(ValueError):
        catalog.select({"harness": "langgraph", "model_id": "b"})
