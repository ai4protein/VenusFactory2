"""Agent boundary tests use synthetic storage; no scientific model claims."""

import json

import pytest
from fastmcp import Client
from test_scientist_model_library import storage as model_storage  # noqa: F401

from mcp_tools.models.evolved import service
from mcp_tools.models.evolved.tools_agent import EVOLVED_MODEL_TOOLS, predict_with_evolved_model_tool
from mcp_tools.models.evolved.tools_mcp import mcp


@pytest.fixture
def published(request, monkeypatch):
    store, root, verified, calls = request.getfixturevalue("model_storage")
    verified["origin"].update(stage="activity", paper={"title": "Synthetic fixture", "doi": "fixture"})
    from agent.rsi.engine.model_plugins import get_model_plugin
    verified["packet"]["assessment"] = {"metrics": {"skill": 0.5},
                                          "plugin_sha256": get_model_plugin(verified["key"]["plugin_id"]).source_sha256}
    result = store.publish(root, "b" * 64)
    monkeypatch.setattr(service, "configured_library", lambda: store)
    seen = []

    class Plugin:
        def predict(self, model, rows, scope):
            seen.extend(rows)
            return [{"id": row["id"], "sequence": row["sequence"], "prediction": 0.25} for row in rows]

    monkeypatch.setattr("agent.rsi.engine.scientist_model_library.get_model_plugin", lambda name: Plugin())
    return store, root, verified, result["entry_id"], seen






@pytest.fixture(params=["generated_context.v1", "generated_context.v2", "generated_context.v3"])
def context_published(request, monkeypatch):
    from agent.rsi.engine.context_interfaces import module_for

    store, root, verified, _ = request.getfixturevalue("model_storage")
    interface = request.param
    columns = [
        {"name": f"measured_{i}", "unit": "arbitrary_units", "minimum": 0, "maximum": 100}
        for i in range(2 if interface.endswith("v1") else 19)
    ]
    scope = {
        "protein": "synthetic_fixture",
        "assay": "synthetic measured response",
        "target": "response",
        "unit": "arbitrary_units",
        "context": {"source": "synthetic_test_only"},
        "covariates": columns,
        "grouping": {"level": "biological_replicate", "evidence_sha256": "a" * 64},
    }
    if not interface.endswith("v1"):
        scope["interface_version"] = interface
        scope["grouping"]["level"] = "deleted_gene"
    verified["key"].update(plugin_id=interface, scope=scope)
    verified["packet"]["model"].update(plugin_id=interface, scope=scope)
    from agent.rsi.engine.model_plugins import get_model_plugin
    verified["packet"]["assessment"] = {"metrics": {"skill": 0.5},
                                          "plugin_sha256": get_model_plugin(verified["key"]["plugin_id"]).source_sha256}
    entry = store.publish(root, "b" * 64)["entry_id"]
    seen = []

    class Plugin:
        def predict(self, model, rows, model_scope):
            module_for(interface).validate_rows(rows, model_scope, labels=False)
            seen.extend(rows)
            return [{"id": row["id"], "prediction": sum(row["values"])} for row in rows]

    monkeypatch.setattr(service, "configured_library", lambda: store)
    monkeypatch.setattr("agent.rsi.engine.scientist_model_library.get_model_plugin", lambda name: Plugin())
    inputs = [
        {"id": "query", "protein": scope["protein"], "group": "recorded_group", "values": list(range(len(columns)))}
    ]
    return entry, scope, inputs, seen














































@pytest.mark.parametrize("online", [True, False])
@pytest.mark.parametrize("action", ["list", "describe", "predict"])
def test_retired_public_entry_rejected_before_library_access(monkeypatch, online, action):
    from mcp_tools.tools import runtime_tool_policy
    monkeypatch.setattr(runtime_tool_policy, "is_online_mode", lambda: online)
    monkeypatch.setattr(service, "configured_library", lambda: pytest.fail("Retired API opened its library"))
    result = service.dispatch(action, entry_id="a"*64)
    assert result["success"] is False and "RSI Plaza" in result["error"]


@pytest.mark.parametrize("online", [True, False])
def test_stale_langchain_prediction_is_rejected(monkeypatch, online):
    from mcp_tools.tools import runtime_tool_policy
    monkeypatch.setattr(runtime_tool_policy, "is_online_mode", lambda: online)
    monkeypatch.setattr(service, "configured_library", lambda: pytest.fail("Stale tool opened its library"))
    result = json.loads(predict_with_evolved_model_tool.invoke(
        {"entry_id": "a"*64, "scope": {"protein": "synthetic"}, "inputs": [{"id":"q", "sequence":"AA"}]},
        config={"configurable":{"thread_id":"old-conversation"}}))
    assert not result["success"] and "RSI Plaza" in result["error"]


@pytest.mark.asyncio
async def test_stale_mcp_transport_rejects_execution(monkeypatch):
    monkeypatch.setattr(service, "configured_library", lambda: pytest.fail("Stale MCP opened library"))
    async with Client(mcp) as client:
        result = (await client.call_tool("list_evolved_models", {})).data
        assert not result["success"] and "RSI Plaza" in result["error"]


@pytest.mark.parametrize("invalid", ["wrong_protein", "missing_value", "nonfinite", "out_of_range"])
def test_internal_scientific_input_validation_still_applies(context_published, invalid):
    from mcp_tools.models.evolved.inputs import validate_model_inputs, ModelInputError
    entry, scope, rows, seen = context_published
    row = {**rows[0], "values": list(rows[0]["values"])}
    if invalid == "wrong_protein": row["protein"] = "another-protein"
    elif invalid == "missing_value": row["values"] = row["values"][:-1]
    elif invalid == "nonfinite": row["values"][0] = float("nan")
    else: row["values"][0] = 101
    with pytest.raises(ModelInputError):
        validate_model_inputs(scope.get("interface_version", "generated_context.v1"), scope, [row])
    assert seen == []


def test_internal_scope_reference_preserves_original_binding(context_published):
    from mcp_tools.models.evolved.scope import scope_reference, bind_scope, ModelScopeError
    entry, scope, rows, seen = context_published
    reference = scope_reference(entry, scope)
    assert bind_scope(entry, scope, None, reference) == scope
    with pytest.raises(ModelScopeError): bind_scope("f"*64, scope, None, reference)
    assert seen == []
