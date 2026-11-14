"""Discovery navigation fixtures are metadata, never trained or admitted models."""

import json
import sqlite3

import pytest
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError
from test_private_research_capabilities import configure_access, lease, metadata_goal, session

from agent.harness import research_workspace as runtime
from agent.harness.tool_disclosure import DisclosureError, ToolDisclosure
from mcp_tools.tools.capability_policy import CapabilityDisabled
from mcp_tools.tools.discovery import generated_scope
from mcp_tools.tools.discovery.research_capabilities import RESEARCH_CAPABILITY_TOOLS
from mcp_tools.tools.discovery.scoped import ScopedScientistCatalog
from mcp_tools.tools.discovery.tools_mcp import register_discovery_tools


@pytest.fixture
def owned_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("WEBUI_V2_MODE", "local")
    monkeypatch.setenv("VENUS_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setenv("VENUS_SANDBOX_CONTROL_ROOT", str(tmp_path / "sandboxes"))
    monkeypatch.setenv("VENUS_WORKER_AUTOSTART", "0")
    state = configure_access(tmp_path, monkeypatch)
    for identity, owner in (("agent-a", "owner-a"), ("research-a", "owner-a"), ("research-b", "owner-b")):
        session(state, identity, owner)
    metadata_goal(state, "research-a", "a" * 64)
    metadata_goal(state, "research-b", "b" * 64)
    queue, active = lease(state, monkeypatch, "owner-a", "agent-a")
    state.mapping = {tool.name: tool for tool in RESEARCH_CAPABILITY_TOOLS}
    state.adapter = ScopedScientistCatalog(state.mapping, lambda tool: True)
    state.disclosure = ToolDisclosure(
        state.mapping, schema=lambda tool: tool.args_schema.model_json_schema(),
        allowed=lambda tool: True, model_catalog=state.adapter,
    )
    yield state
    queue.finish(active["id"], "completed", lease=active["lease"])


@pytest.mark.parametrize("mode", ["local", "online"])
@pytest.mark.parametrize("identity", ["a" * 64, "b" * 64, "c" * 64], ids=["owned", "foreign", "unknown"])
def test_dynamic_identifier_is_not_an_agent_tool(owned_metadata, monkeypatch, mode, identity):
    state = owned_metadata
    monkeypatch.setenv("WEBUI_V2_MODE", mode)
    overview = state.disclosure.metadata("search_tools", {})
    assert all(item["id"] != "rsi" for item in overview["categories"])
    name = state.adapter.prefix + identity
    assert state.adapter.generated_page("", 3, 0) is None
    with pytest.raises((DisclosureError, ValueError)):
        state.disclosure.metadata("describe_tool", {"name": name})
    with pytest.raises((DisclosureError, ValueError)):
        state.disclosure.resolve("call_tool", {"name": name, "arguments": {"inputs": {}}})
    assert state.adapter.bindings == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["local", "online"])
async def test_retired_tools_never_scan_registry_or_enter_mcp(owned_metadata, monkeypatch, mode):
    state = owned_metadata
    monkeypatch.setenv("WEBUI_V2_MODE", mode)
    def forbidden(*args, **kwargs):
        raise AssertionError("Retired tool must not scan, load or execute a research model")
    monkeypatch.setattr(generated_scope, "private_libraries", forbidden)
    monkeypatch.setattr(generated_scope.catalog, "ranked_references", forbidden)
    monkeypatch.setattr(runtime, "search_capabilities", forbidden)
    monkeypatch.setattr(runtime, "describe_capability", forbidden)
    monkeypatch.setattr(runtime, "run_capability", forbidden)
    assert generated_scope.search("synthetic")["total"] == 0
    assert not state.adapter.available()
    with pytest.raises(RuntimeError, match="RSI Plaza"):
        state.mapping["search_generated_capabilities"].invoke({})
    with pytest.raises(RuntimeError, match="RSI Plaza"):
        state.mapping["describe_generated_capability"].invoke({"capability_id": "a" * 64})
    with pytest.raises(RuntimeError, match="RSI Plaza"):
        await state.mapping["run_generated_capability"].ainvoke({"capability_id": "a" * 64, "inputs": {}})
    server = FastMCP("RSI retirement fixture")
    register_discovery_tools(server)
    async with Client(server) as client:
        names = {tool.name for tool in await client.list_tools()}
        assert not names.intersection(state.mapping)
        with pytest.raises(ToolError):
            await client.call_tool("run_generated_capability", {"capability_id": "a" * 64, "inputs": {}})
