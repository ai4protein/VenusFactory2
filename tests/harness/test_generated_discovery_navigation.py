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


@pytest.mark.parametrize("category", ["databases", "compute", "rsi"])
@pytest.mark.parametrize("identity", ["a" * 64, "b" * 64, "c" * 64], ids=["owned", "foreign", "unknown"])
def test_dynamic_identifier_navigation_never_becomes_a_static_tool(owned_metadata, category, identity):
    state = owned_metadata
    page = state.disclosure.metadata("search_tools", {"category": "rsi", "query": "synthetic"})
    assert [item["name"] for item in page["tools"]] == [state.adapter.prefix + "a" * 64]
    name = state.adapter.prefix + identity

    navigation = state.disclosure.metadata("search_tools", {"category": category, "query": name})

    assert navigation["level"] == "categories"
    assert [item["id"] for item in navigation["categories"]] == ["rsi"]
    assert navigation["tools"] == []
    assert len(json.dumps(navigation)) < 1500
    assert name not in json.dumps(navigation)
    assert "inputSchema" not in json.dumps(navigation)
    assert state.adapter.bindings == {} and state.disclosure.inspected == set()
    # Navigation neither proves that an arbitrary ID exists nor admits it. Even
    # the owner fixture has no real independent SC receipt and cannot be used.
    with pytest.raises((DisclosureError, ValueError, KeyError, FileNotFoundError)):
        state.disclosure.metadata("describe_tool", {"name": name})
    with pytest.raises(DisclosureError):
        state.disclosure.resolve("call_tool", {"name": name, "arguments": {"inputs": {}}})
    assert state.adapter.bindings == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["native", "mcp"])
async def test_execution_switch_hides_discovery_before_any_registry_scan(owned_metadata, monkeypatch, surface):
    state = owned_metadata
    identity = "a" * 64
    assert [item["id"] for item in generated_scope.search("synthetic")["items"]] == [identity]
    assert state.adapter.generated_page("synthetic", 3, 0)["total"] == 1
    revision = state.switches.set("tool:run_generated_capability", False, 0, {}, "test operator")

    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled discovery must not scan registries, read contracts or invoke a model")

    scans = []
    private_libraries = generated_scope.private_libraries

    def scan():
        scans.append(True)
        return private_libraries()

    with monkeypatch.context() as disabled:
        disabled.setattr(generated_scope, "private_libraries", scan)
        disabled.setattr(generated_scope.catalog, "ranked_references", forbidden)
        disabled.setattr(runtime, "search_capabilities", forbidden)
        disabled.setattr(runtime, "describe_capability", forbidden)
        disabled.setattr(runtime, "run_capability", forbidden)
        assert not state.adapter.available()
        if surface == "native":
            page = json.loads(state.mapping["search_generated_capabilities"].invoke({"query": "synthetic"}))
            assert scans == []
            with pytest.raises(CapabilityDisabled):
                state.mapping["describe_generated_capability"].invoke({"capability_id": identity})
            with pytest.raises(CapabilityDisabled):
                await state.mapping["run_generated_capability"].ainvoke({"capability_id": identity, "inputs": {}})
        else:
            server = FastMCP("disabled private discovery fixture")
            register_discovery_tools(server)
            async with Client(server) as client:
                page = (await client.call_tool("search_generated_capabilities", {"query": "synthetic"})).data
                assert scans == []
                with pytest.raises(ToolError):
                    await client.call_tool("describe_generated_capability", {"capability_id": identity})
                with pytest.raises(ToolError):
                    await client.call_tool("run_generated_capability", {"capability_id": identity, "inputs": {}})
        assert page["total"] == 0 and page["items"] == [] and page["next_offset"] is None

    # The request snapshot expires: enabling execution is visible immediately,
    # while deleting the source session still revokes private discoverability.
    state.switches.set("tool:run_generated_capability", True, revision, {}, "test operator")
    assert generated_scope.search("synthetic")["total"] == 1
    with sqlite3.connect(state.database) as connection:
        connection.execute("DELETE FROM sessions WHERE session_id='research-a'")
    assert generated_scope.search("synthetic")["total"] == 0
