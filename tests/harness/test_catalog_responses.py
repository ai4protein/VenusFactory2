"""Host-declared controls enable Responses without changing existing selections."""

import json
from dataclasses import replace

import pytest

from agent.harness import Budget, ModelBinding, Profile
from agent.harness.catalog import ExecutionCatalog


def catalog(tmp_path, controls):
    model = ModelBinding("fixture", "model", "https://fixture.invalid/v1", credential_env="FIXTURE_CATALOG_KEY")
    base = Profile("science_expert", "expert-kimi-code", model, Budget(model_calls=12))
    row = {"id": "responses", "label": "Explicit Responses", "binding": replace(model, protocol="responses").__dict__}
    if controls is not None:
        row["inference_controls"] = controls
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"models": [row]}))
    return ExecutionCatalog(tmp_path, base, path=path), base


def test_catalog_composes_explicit_controls_and_preserves_resource_limits(tmp_path, monkeypatch):
    controls = {"thinking": None, "temperature": None, "reasoning_effort": "low"}
    value, base = catalog(tmp_path, controls)
    selected, option = value.select({"harness": "expert-codex", "model_id": "responses"})
    assert selected.model.protocol == "responses" and selected.budget.reasoning_effort == "low"
    assert selected.budget.thinking is None and selected.budget.temperature is None
    assert selected.budget.model_calls == base.budget.model_calls
    assert selected.budget.tool_calls == base.budget.tool_calls
    assert selected.budget.wall_seconds == base.budget.wall_seconds
    assert value.select()[0] == base
    monkeypatch.setenv("FIXTURE_CATALOG_KEY", "fixture-only")
    binary = tmp_path / "codex"
    binary.write_text("fixture")
    public = value.public({"codex_bin": str(binary)})
    entry = next(x for x in public["combinations"] if x["harness"] == "expert-codex" and x["model_id"] == "responses")
    assert entry["ready"] and not entry["reasons"]
    assert option.public()["inference_controls"] == controls
    assert "fixture-only" not in json.dumps(public) and "FIXTURE_CATALOG_KEY" not in json.dumps(public)


def test_protocol_selection_does_not_silently_rewrite_controls(tmp_path, monkeypatch):
    value, base = catalog(tmp_path, None)
    selected, _ = value.select({"harness": "expert-codex", "model_id": "responses"})
    assert selected.budget == base.budget
    monkeypatch.setenv("FIXTURE_CATALOG_KEY", "fixture-only")
    public = value.public({})
    entry = next(x for x in public["combinations"] if x["harness"] == "expert-codex" and x["model_id"] == "responses")
    assert "responses_controls_incompatible" in entry["reasons"]
    assert "gateway_protocol_not_implemented" not in entry["reasons"]


@pytest.mark.parametrize(
    "controls",
    [[], {"model_calls": 1000}, {"base_url": "http://other"}, {"temperature": 4}, {"reasoning_effort": "invalid"}],
)
def test_catalog_rejects_implicit_budget_or_endpoint_changes(tmp_path, controls):
    with pytest.raises((ValueError, TypeError)):
        catalog(tmp_path, controls)
