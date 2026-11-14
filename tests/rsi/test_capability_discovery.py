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
from mcp_tools.models.evolved import catalog as metadata_cache


@pytest.fixture(autouse=True)
def isolated_discovery_runtime(tmp_path, monkeypatch):
    # These fixtures describe a synthetic local catalog, not the deployment's
    # mode or the real Scientist capabilities already published on the server.
    monkeypatch.setattr(service, "platform_models", lambda *a, **kw: [])
    from agent.harness import research_workspace
    from mcp_tools.tools import runtime_tool_policy
    monkeypatch.setattr(research_workspace, "LIBRARY", tmp_path / "published-capabilities")
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








@pytest.mark.parametrize("kwargs", [{"limit": True}, {"limit": 31}, {"offset": -1}, {"kind": {}}, {"query": "a" * 201}])
def test_invalid_search_is_rejected(catalog, kwargs):
    assert not service.search(**kwargs)["success"]






def test_cache_refreshes_on_new_publication_and_does_not_share_mutable_cards(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from mcp_tools.models.evolved import service as evolved

    index = tmp_path / "index"
    index.mkdir()
    calls = []
    record = card()
    metadata = {"models": [{"entry_id": record["entry_id"],
                            "key": {"plugin_id": record["interface"], "scope": record["scope"]},
                            "origin": {"stage": record["stage"], "paper": record["paper"]},
                            "plugin_sha256": "fixture"}],
                "indexed_entries": 1, "counting_rule": "fixture"}
    library = SimpleNamespace(root=tmp_path, discovery_metadata=lambda: (calls.append(1) or metadata))
    monkeypatch.setattr(evolved, "configured_library", lambda: library)
    monkeypatch.setattr(metadata_cache.time, "monotonic", lambda: 100)
    metadata_cache._CACHE.clear()
    first = service.model_cards()
    first[0]["scope"]["protein"] = "modified"
    assert service.model_cards()[0]["scope"]["protein"] == "AT5G51820"
    assert len(calls) == 1
    prior = index.stat().st_mtime_ns
    (index / "new.json").write_text("{}")
    os.utime(index, ns=(prior + 1000000, prior + 1000000))
    service.model_cards()
    assert len(calls) == 2









@pytest.mark.parametrize("online", [True, False])
def test_conversational_search_never_reads_rsi_registry(catalog, monkeypatch, online):
    from mcp_tools.tools import runtime_tool_policy
    monkeypatch.setattr(runtime_tool_policy, "is_online_mode", lambda: online)
    monkeypatch.setattr(service, "model_cards", lambda: pytest.fail("RSI catalog scanned by chat"))
    result = service.search()
    assert result["matching_tools"] == 1 and result["matching_models"] == 0
    assert result["model_stages"] == {} and result["model_catalog_status"] == "available"


@pytest.mark.asyncio
async def test_mcp_and_langchain_discover_only_enabled_native_tools(catalog):
    arguments = {"query":"thermal stability", "kind":"tools"}
    direct = json.loads(search_research_tools_tool.invoke(arguments))
    async with Client(mcp) as client:
        tools = await client.list_tools()
        from mcp_tools.models.evolved.policy import RSI_TOOL_NAMES
        assert not ({t.name for t in tools} & RSI_TOOL_NAMES)
        result = (await client.call_tool("search_research_tools", arguments)).data
    assert direct["matching_tools"] == 1
    assert all(item["kind"] != "scientist_model" for item in result["results"])


@pytest.mark.parametrize("prefix", ["", "mcp__venusfactory__"])
def test_guidance_points_rsi_to_case_plaza(prefix):
    from agent.harness.capability_guidance import discovery_instructions
    text = discovery_instructions([prefix + "search_research_tools", prefix + "predict_with_evolved_model"])
    assert "RSI Plaza" in text and "cannot be called" in text
    assert "To use a Scientist model" not in text
