"""Discovery and inference wiring with synthetic metadata, never benchmark evidence."""

import asyncio
import copy
import json
import os

import pytest
from fastmcp import Client
from langchain_core.tools import tool
from test_evolved_model_tools import published, model_storage  # noqa: F401

from mcp_tools.tools.discovery import service
from mcp_tools.tools.discovery.tools_agent import search_research_tools_tool
from mcp_tools.tools.discovery.tools_mcp import mcp
from mcp_tools.models.evolved.tools_agent import EVOLVED_MODEL_TOOLS


@pytest.fixture(autouse=True)
def isolated_discovery_runtime(monkeypatch):
    # These fixtures describe a synthetic local catalog, not the deployment's
    # mode or the real Scientist capabilities already published on the server.
    from agent.harness import research_workspace
    from mcp_tools.tools import runtime_tool_policy
    monkeypatch.setattr(research_workspace, "capability_cards", lambda: [])
    monkeypatch.setattr(runtime_tool_policy, "is_online_mode", lambda: False)


@tool
def stability_prediction(sequence: str) -> str:
    """Predict thermal stability of a protein sequence."""
    return "unused"


def card(i=1, **scope):
    return {
        "entry_id": f"{i:064x}", "interface": "generated_context.v3",
        "scope": {"protein": "AT5G51820", "target": "thermal stability", "unit": "C", **scope},
        "stage": "stability", "paper": {"title": "Synthetic catalog"}, "verified_now": False,
    }


@pytest.fixture
def catalog(monkeypatch):
    cards = [card(), {**card(2, protein="P38197", target="abundance"), "stage": "abundance"}]
    monkeypatch.setattr(service, "available_tools", lambda: [stability_prediction, search_research_tools_tool])
    monkeypatch.setattr(service, "model_cards", lambda: copy.deepcopy(cards))
    monkeypatch.setattr(service, "is_local_mode", lambda: True)
    return cards


@pytest.mark.parametrize("query", ["thermal stability", "热稳定性", "拟南芥热稳定性模型"])
def test_bilingual_property_search(catalog, query):
    # Species queries require species metadata; identifiers alone do not imply it.
    catalog[0]["scope"]["organism"] = "Arabidopsis"
    result = service.search(query, kind="models")
    assert result["success"] and result["matching_models"] == 1
    assert result["results"][0]["entry_id"] == card()["entry_id"]


def test_inventory_identifier_filter_and_pagination(catalog):
    overview = service.search("有哪些蛋白模型", kind="models", limit=1)
    assert overview["indexed_models"] == overview["matching_models"] == 2
    assert overview["next_offset"] == 1
    second = service.search(kind="models", offset=1, limit=1)
    assert second["next_offset"] is None and second["results"] != overview["results"]
    assert service.search("p38197", kind="models")["matching_models"] == 1
    assert service.search(kind="models", stage="activity")["matching_models"] == 0
    assert {r["kind"] for r in service.search(limit=2)["results"]} == {"tool", "scientist_model"}


def test_search_10000_models_returns_a_bounded_page(catalog):
    catalog[:] = [card(i, protein=f"protein_{i}") for i in range(10001)]
    result = service.search(kind="models", limit=7, offset=9996)
    assert result["matching_models"] == 10001
    assert len(result["results"]) == 5 and result["next_offset"] is None
    assert len(json.dumps(result)) < 10000


@pytest.mark.parametrize("kwargs", [{"limit": True}, {"limit": 31}, {"offset": -1}, {"kind": {}}, {"query": "a" * 201}])
def test_invalid_search_is_rejected(catalog, kwargs):
    assert not service.search(**kwargs)["success"]


def test_online_mode_never_reads_model_library(catalog, monkeypatch):
    monkeypatch.setattr(service, "is_local_mode", lambda: False)
    monkeypatch.setattr(service, "model_cards", lambda: pytest.fail("Disabled model catalog was accessed"))
    result = service.search()
    assert result["matching_tools"] == 1 and result["indexed_models"] == 0
    assert result["model_catalog_status"] == "local_only"


def test_catalog_failure_keeps_native_tools_available(catalog, monkeypatch):
    def unavailable():
        raise ValueError("Private path /private/secret")
    monkeypatch.setattr(service, "model_cards", unavailable)
    result = service.search()
    assert result["matching_tools"] == 1 and result["model_catalog_status"] == "unavailable"
    assert "/private" not in json.dumps(result)


def test_cache_refreshes_on_new_publication_and_does_not_share_mutable_cards(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from mcp_tools.models.evolved import service as evolved

    index = tmp_path / "index"
    index.mkdir()
    calls = []
    monkeypatch.setattr(evolved, "configured_library", lambda: SimpleNamespace(root=tmp_path))
    monkeypatch.setattr(evolved, "catalog_cards", lambda _: (calls.append(1) or [card()], {}))
    monkeypatch.setattr(service.time, "monotonic", lambda: 100)
    service._CACHE.clear()
    first = service.model_cards()
    first[0]["scope"]["protein"] = "modified"
    assert service.model_cards()[0]["scope"]["protein"] == "AT5G51820"
    assert len(calls) == 1
    prior = index.stat().st_mtime_ns
    (index / "new.json").write_text("{}")
    os.utime(index, ns=(prior + 1000000, prior + 1000000))
    service.model_cards()
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_langchain_and_mcp_search_the_same_catalog(catalog):
    arguments = {"query": "热稳定性", "kind": "models"}
    direct = json.loads(search_research_tools_tool.invoke(arguments))
    async with Client(mcp) as client:
        result = (await client.call_tool("search_research_tools", arguments)).data
    assert result == direct
    assert result["results"][0]["describe"]["tool"] == "describe_evolved_model"


@pytest.mark.parametrize("prefix", ["", "mcp__venusfactory__"])
def test_guidance_uses_actual_registered_names(prefix):
    from agent.harness.capability_guidance import discovery_instructions
    names = [prefix + n for n in ["search_research_tools", "describe_evolved_model", "predict_with_evolved_model"]]
    text = discovery_instructions(names)
    assert all(n in text for n in names)
    assert "before answering" in text and "kind='models'" in text
    assert discovery_instructions([]) == ""


def test_conversation_search_describe_predict_with_published_model(published, tmp_path, monkeypatch):
    from agent.harness import web_langgraph_worker as worker
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    store, root, verified, entry, seen = published
    service._CACHE.clear()
    monkeypatch.setattr(service, "is_local_mode", lambda: True)
    monkeypatch.setattr(service, "available_tools", lambda: EVOLVED_MODEL_TOOLS)
    events = []
    monkeypatch.setattr(worker, "emit", events.append)

    class ScriptedModel(GenericFakeChatModel):
        disable_streaming: bool = True
        def bind_tools(self, tools, **kwargs):
            return self

    steps = [
        ("search_research_tools", {"kind": "models", "query": ""}),
        ("describe_evolved_model", {"entry_id": entry}),
        ("predict_with_evolved_model", {"entry_id": entry, "inputs": [{"id": "query", "sequence": "AA"}], "scope": verified["key"]["scope"]}),
    ]
    messages = [AIMessage(content="", tool_calls=[{"name": "mcp__venusfactory__" + name, "args": args, "id": str(i)}]) for i, (name, args) in enumerate(steps)]
    messages.append(AIMessage(content="The synthetic prediction is 0.25."))
    asyncio.run(worker.run(
        {"text": "Find and apply the model to AA", "workspace": str(tmp_path), "session_id": "discovery-chat", "budget": {"tool_calls": 3}},
        tools=[search_research_tools_tool, *EVOLVED_MODEL_TOOLS], model=ScriptedModel(messages=iter(messages)),
    ))
    results = [json.loads(e["output"]) for e in events if e["kind"] == "tool_result"]
    assert len(results) == 3 and all(r["success"] for r in results)
    assert results[0]["results"][0]["entry_id"] == entry
    assert results[2]["receipt"]["session_id"] == "discovery-chat"
    assert seen == [{"id": "query", "sequence": "AA"}]
    assert events[-1]["kind"] == "completed"


@pytest.mark.asyncio
async def test_mcp_discovery_uses_its_actual_mounted_catalog(catalog):
    from fastmcp import FastMCP
    server = FastMCP("discovery_fixture")
    from mcp_tools.tools.discovery.tools_mcp import register_discovery_tools
    register_discovery_tools(server)

    @server.tool
    def mounted_only(sequence: str) -> str:
        """Inspect a sequence using this server's fixture tool."""
        return sequence

    async with Client(server) as client:
        result = (await client.call_tool("search_research_tools", {"kind": "tools"})).data
    expected = {"mounted_only", "query_platform_manual", "search_generated_capabilities", "describe_generated_capability", "run_generated_capability"}
    assert result["available_tools"] == len(expected)
    assert {r["name"] for r in result["results"]} == expected
    assert next(row for row in result["results"] if row["name"] == "mounted_only")["required_arguments"] == ["sequence"]
