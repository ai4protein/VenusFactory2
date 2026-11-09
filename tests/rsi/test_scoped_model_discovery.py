"""Model-level disclosure uses the existing verified library and prediction path."""
import json
from types import SimpleNamespace

import pytest

from test_evolved_model_tools import published, context_published, model_storage  # noqa: F401
from agent.harness.tool_disclosure import ToolDisclosure, ToolInspectionRequired, DisclosureError
from agent.harness.web_native_worker import mcp_input_schema
from mcp_tools.models.evolved import discovery, service
from mcp_tools.models.evolved.tools_agent import EVOLVED_MODEL_TOOLS


def index(allowed=lambda tool: True):
    mapping = {tool.name: tool for tool in EVOLVED_MODEL_TOOLS}
    models = discovery.ScopedModelCatalog(mapping, allowed)
    return ToolDisclosure(mapping, schema=mcp_input_schema, allowed=allowed, model_catalog=models)


def test_search_returns_one_model_card_then_inspection_binds_the_full_scope(context_published):
    entry, scope, rows, seen = context_published
    catalog = index()
    name = discovery.ScopedModelCatalog.prefix + entry
    assert name not in json.dumps(catalog.surface())
    page = catalog.metadata("search_tools", {"category": "rsi", "query": scope["protein"]})
    assert page["records_searched"] and page["total"] == 1
    assert page["tools"][0]["name"] == name
    assert set(page["tools"][0]) == {"name", "description"} and not seen
    assert "covariates" not in json.dumps(page)
    with pytest.raises(ToolInspectionRequired):
        catalog.resolve("call_tool", {"name": name, "arguments": {"inputs": rows}})
    card = catalog.metadata("describe_tool", {"name": name})
    assert card["scope"] == scope and not card["verified_now"] and not seen
    assert set(card["inputSchema"]["properties"]) == {"inputs"}
    assert card["input_contract"]["values"]["columns"] == scope["covariates"]
    key, tool, arguments = catalog.resolve("call_tool", {"name": name, "arguments": {"inputs": rows}})
    assert key == tool.name == "predict_with_evolved_model"
    assert arguments["inputs"] is rows
    assert arguments["entry_id"] == entry
    assert arguments["scope_ref"] == service.dispatch("describe", entry_id=entry)["scope_ref"]
    result = json.loads(tool.invoke(arguments, config={"configurable": {"session_id": "scoped-session"}}))
    assert result["success"] and result["scope"] == scope
    assert result["receipt"]["session_id"] == "scoped-session" and seen == rows


@pytest.mark.parametrize("key", ["entry_id", "scope_ref", "scope", "session_id", "library_root"])
def test_inspected_model_identity_and_host_fields_cannot_be_overridden(published, key):
    _, _, _, entry, seen = published
    name = discovery.ScopedModelCatalog.prefix + entry
    catalog = index()
    catalog.metadata("describe_tool", {"name": name})
    with pytest.raises(DisclosureError, match="inputs only") as error:
        catalog.resolve("call_tool", {"name": name, "arguments": {"inputs": [], key: "private-marker"}})
    assert "private-marker" not in str(error.value) and not seen


def test_inspection_is_session_scoped_and_invalid_input_still_fails_before_queue(published):
    _, _, _, entry, seen = published
    name = discovery.ScopedModelCatalog.prefix + entry
    catalog = index()
    catalog.metadata("describe_tool", {"name": name})
    with pytest.raises(ToolInspectionRequired):
        index().resolve("call_tool", {"name": name, "arguments": {"inputs": []}})
    with pytest.raises(DisclosureError, match="inputs"):
        catalog.resolve("call_tool", {"name": name, "arguments": {"inputs": []}})
    assert not seen


def test_live_enablement_and_changed_evidence_still_gate_bound_calls(published, monkeypatch):
    store, _, _, entry, seen = published
    name = discovery.ScopedModelCatalog.prefix + entry
    enabled = {tool.name for tool in EVOLVED_MODEL_TOOLS}
    catalog = index(lambda tool: tool.name in enabled)
    catalog.metadata("describe_tool", {"name": name})
    enabled.remove("predict_with_evolved_model")
    with pytest.raises(DisclosureError, match="unavailable"):
        catalog.resolve("call_tool", {"name": name, "arguments": {"inputs": [{"id": "x", "sequence": "AA"}]}})
    enabled.add("predict_with_evolved_model")
    def reject(*args, **kwargs):
        raise ValueError("private-evidence-path-marker")
    monkeypatch.setattr(store, "resolve", reject)
    _, tool, args = catalog.resolve("call_tool", {"name": name, "arguments": {"inputs": [{"id": "x", "sequence": "AA"}]}})
    result = json.loads(tool.invoke(args, config={"configurable": {"session_id": "fixture"}}))
    assert not result["success"] and "private-evidence-path-marker" not in json.dumps(result) and not seen


def test_per_model_policy_denies_inspection_and_use_without_library_access(published, monkeypatch):
    _, _, _, entry, seen = published
    name = discovery.ScopedModelCatalog.prefix + entry
    catalog = index()
    catalog.metadata("describe_tool", {"name": name})
    def deny(*args, **kwargs):
        raise ValueError("disabled")
    monkeypatch.setattr(discovery, "assert_tool_call_allowed", deny)
    monkeypatch.setattr(service, "dispatch", lambda *a, **k: pytest.fail("Disabled model accessed its library"))
    for operation in ("describe_tool", "call_tool"):
        with pytest.raises(DisclosureError, match="unavailable"):
            if operation == "describe_tool":
                catalog.metadata(operation, {"name": name})
            else:
                catalog.resolve(operation, {"name": name, "arguments": {"inputs": []}})
    assert not seen


def test_ten_thousand_models_do_not_load_before_category_selection_or_expand_pages(monkeypatch):
    calls = []
    monkeypatch.setattr(discovery, "assert_tool_call_allowed", lambda *a, **k: None)
    def read(action, **args):
        assert action == "list"
        calls.append(args)
        rows=[{"entry_id": format(i, "064x"), "name": f"Protein {i}",
               "description": "Predict a measured condition."} for i in range(args["offset"], min(10000, args["offset"]+args["limit"]))]
        return {"success": True, "models": rows, "matching_entries": 10000,
                "next_offset": args["offset"]+len(rows) if args["offset"]+len(rows)<10000 else None}
    monkeypatch.setattr(service, "dispatch", read)
    catalog = index()
    catalog.surface()
    catalog.metadata("search_tools", {})
    assert not calls and not catalog.model_catalog.bindings
    first=catalog.metadata("search_tools", {"category":"rsi", "limit":20})
    second=catalog.metadata("search_tools", {"category":"rsi", "limit":20, "offset":20})
    assert first["total"]==10000 and first["next_offset"]==20
    assert len(first["tools"])==20 and len(json.dumps(first))<6500
    assert not {r["name"] for r in first["tools"]} & {r["name"] for r in second["tools"]}
    assert not catalog.model_catalog.bindings


@pytest.mark.parametrize("harness", ["kimi-code", "deepseek-harness", "codex", "claude-code"])
@pytest.mark.asyncio
async def test_native_harness_uses_scoped_card_but_queues_the_original_tool(published, tmp_path, monkeypatch, harness):
    import httpx
    from agent.harness import web_native_worker as worker
    from agent.harness.contracts import Profile, Budget, ModelBinding
    from mcp_tools.scheduler import catalog as scheduler, client as queue_client
    _, _, verified, entry, seen = published
    name=discovery.ScopedModelCatalog.prefix+entry
    queued, events=[], []
    monkeypatch.setattr(scheduler, "native_tools", lambda: {t.name:t for t in EVOLVED_MODEL_TOOLS})
    async def invoke(tool_name, arguments, **host):
        queued.append((tool_name, arguments, host))
        return service.dispatch("predict", **arguments, session_id=host["session_id"])
    monkeypatch.setattr(queue_client,"invoke",invoke)
    workspace=tmp_path/"work";workspace.mkdir()
    control=tmp_path/"control";control.mkdir()
    async def adapter(ctx):
        connection=json.loads((ctx.task.capability/"connection.json").read_text())
        async with httpx.AsyncClient(trust_env=False) as client:
            async def call(tool,args,error=False):
                response=await client.post(connection["url"],headers={"Authorization":"Bearer "+connection["token"]},
                    json={"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":tool,"arguments":args}})
                result=response.json()["result"]
                assert result["isError"] is error, result
                return result["content"][0]["text"] if error else json.loads(result["content"][0]["text"])
            page=await call("search_tools",{"category":"rsi","query":entry})
            assert page["tools"][0]["name"]==name and not queued
            card=await call("describe_tool",{"name":name})
            assert card["scope"]==verified["key"]["scope"] and not queued
            rejected=await call("call_tool",{"name":name,"arguments":{"inputs":[{"id":"x","sequence":"INVALID!"}]}},error=True)
            assert "input contract" in rejected and not queued
            result=await call("call_tool",{"name":name,"arguments":{"inputs":[{"id":"x","sequence":"AA"}]}})
            assert result["success"] and result["predictions"][0]["prediction"]==0.25
            return {"status":"completed","text":"Prediction: 0.25."}
    monkeypatch.setattr(worker,"builtin_registry",lambda:SimpleNamespace(resolve=lambda *args:(SimpleNamespace(run=adapter),None)))
    profile=Profile("science_agent",harness,ModelBinding("fixture","model","http://127.0.0.1:1/v1"),Budget(tool_calls=8))
    await worker.run({"profile":profile.public(),"workspace":str(workspace),"control":str(control),
                     "session_id":"scoped-harness","runtimes":{},"base_url":"http://127.0.0.1:1/v1",
                     "token":"fixture","text":"Predict with the registered model"},tools=EVOLVED_MODEL_TOOLS,event_sink=events.append)
    assert len(queued)==1 and queued[0][0]=="predict_with_evolved_model"
    assert queued[0][1]["entry_id"]==entry and queued[0][2]["session_id"]=="scoped-harness"
    assert seen==[{"id":"x","sequence":"AA"}]
    assert any(e.get("kind")=="completed" for e in events)


def test_online_policy_cannot_be_bypassed_by_the_disclosure_callback(monkeypatch):
    monkeypatch.setenv("WEBUI_V2_MODE", "online")
    monkeypatch.setattr(service, "configured_library", lambda: pytest.fail("Online lookup reached private model storage"))
    catalog = index(lambda tool: True)
    name = discovery.ScopedModelCatalog.prefix + "a" * 64
    assert catalog.model_catalog.search("", 8, 0) is None
    with pytest.raises(DisclosureError, match="unavailable"):
        catalog.metadata("describe_tool", {"name": name})


def test_navigation_adapter_import_does_not_load_scientific_runtime():
    import os
    import subprocess
    import sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    code="import sys; from mcp_tools.models.evolved.discovery import ScopedModelCatalog; assert 'mcp_tools.models.evolved.service' not in sys.modules; assert 'agent.rsi.engine.scientist_model_library' not in sys.modules"
    subprocess.run([sys.executable,"-c",code],cwd=root,env={**os.environ,"PYTHON_DOTENV_DISABLED":"1"},check=True)


def test_wrong_protein_feedback_precedes_queue_and_inspected_scope_cannot_be_mutated(context_published):
    entry, scope, rows, seen = context_published
    catalog = index()
    name = discovery.ScopedModelCatalog.prefix + entry
    card = catalog.metadata("describe_tool", {"name": name})
    card["scope"]["protein"] = "unrelated-protein"
    with pytest.raises(DisclosureError, match="Search the RSI category"):
        catalog.resolve("call_tool", {"name": name, "arguments": {
            "inputs": [{**rows[0], "protein": "unrelated-protein"}]}})
    assert not seen
    _, _, args = catalog.resolve("call_tool", {"name": name, "arguments": {"inputs": rows}})
    assert args["inputs"] is rows and set(args) == {"entry_id", "scope_ref", "inputs"}
