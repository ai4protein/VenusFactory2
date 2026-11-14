"""SQLite scaling regression: 10,000 metadata records, not trained models.

Session ownership, private delivery provenance and the active queue lease are real.
Only metadata seeding and SQLite observation are test scaffolding.
"""
import json
import sqlite3

import pytest
from test_private_research_capabilities import configure_access, lease, metadata_goal, session

from agent.harness import capability_catalog as catalog
from agent.harness import research_workspace as runtime
from mcp_tools.tools import capability_policy
from mcp_tools.tools.discovery import generated_scope
from mcp_tools.tools.discovery.research_capabilities import RESEARCH_CAPABILITY_TOOLS, search_generated_capabilities
from mcp_tools.tools.discovery.scoped import ScopedScientistCatalog


@pytest.fixture
def large_owned_catalog(tmp_path, monkeypatch):
    state = configure_access(tmp_path, monkeypatch)
    for identity, owner in (("agent-a", "owner-a"), ("research-a", "owner-a"), ("research-b", "owner-b")):
        session(state, identity, owner)
    private = metadata_goal(state, "research-a", "a" * 64)
    metadata_goal(state, "research-b", "b" * 64)
    # 9,999 public metadata entries + one owned private delivery. The foreign
    # private record is a negative ownership control, not part of the inventory.
    rows = []
    for i in range(9999):
        identity = f"{i:064x}"
        card = {
            "id": identity, "name": f"Synthetic metadata {i}", "description": "Synthetic metadata only",
            "kind": "model", "version": "1", "visibility": "public", "review_status": "approved",
            "publication": {"policy": "operator_selected.v1"}, "scope": {},
            "test_schema_marker": True, "input_schema": {"type": "object", "properties": {"sequence": {"type": "string"}}},
            "output_schema": {"type": "number"}, "validation": {"fixture": "not trained"},
        }
        rows.append((identity, "fixture", card["name"], "1", "synthetic metadata model", json.dumps(card), "fixture", i))
    with catalog.database(runtime.LIBRARY) as connection:
        connection.executemany("INSERT INTO capabilities VALUES (?,?,?,?,?,?,?,?)", rows)
    # Ensure the switch DB exists; this unrelated disabled ID does not affect
    # the 10,000 authorized records.
    state.switches.set("model:capability:" + "f" * 64, False, 0, {}, "test")
    queue, active = lease(state, monkeypatch, "owner-a", "agent-a")
    state.private = private
    state.active_queue, state.active_lease = queue, active

    reads, materialized = [], []
    connect = state.switches.connect

    def traced_connect():
        connection = connect()
        connection.set_trace_callback(
            lambda statement: reads.append(statement)
            if statement.strip().casefold().startswith("select * from switches") else None
        )
        return connection

    monkeypatch.setattr(state.switches, "connect", traced_connect)
    decode = json.loads

    def counted_decode(value, *args, **kwargs):
        result = decode(value, *args, **kwargs)
        if isinstance(result, dict) and result.get("test_schema_marker"):
            materialized.append(result["id"])
        return result

    monkeypatch.setattr(catalog.json, "loads", counted_decode)

    def no_full_inventory(*args, **kwargs):
        raise AssertionError("Discovery must not load all full contracts or weights")

    monkeypatch.setattr(catalog, "cards", no_full_inventory)
    monkeypatch.setattr(runtime, "capability_cards", no_full_inventory)
    monkeypatch.setattr(runtime, "describe_capability", no_full_inventory)
    monkeypatch.setattr(runtime, "run_capability", no_full_inventory)
    state.reads, state.materialized = reads, materialized
    return state


@pytest.mark.parametrize("mode", ["online", "local"])
def test_retired_discovery_does_not_materialize_ten_thousand_models(large_owned_catalog, monkeypatch, mode):
    from mcp_tools.tools.capability_policy import CapabilityDisabled
    state = large_owned_catalog
    monkeypatch.setenv("WEBUI_V2_MODE", mode)
    state.reads.clear()
    state.materialized.clear()
    mapping = {tool.name: tool for tool in RESEARCH_CAPABILITY_TOOLS}
    adapter = ScopedScientistCatalog(mapping, lambda tool: True)
    assert adapter.generated_page("synthetic", 3, 0) is None
    assert generated_scope.search("synthetic")["items"] == []
    with pytest.raises(RuntimeError, match="RSI Plaza"):
        search_generated_capabilities.invoke({"query": "synthetic", "limit": 3})
    assert state.materialized == []
    # Retired discovery exits before reading the operator switches or registries.
    assert state.reads == []


@pytest.mark.parametrize("mode", ["online", "local"])
def test_enabling_legacy_switch_still_cannot_scan_registry(large_owned_catalog, monkeypatch, mode):
    from mcp_tools.tools.capability_policy import CapabilityDisabled
    state = large_owned_catalog
    monkeypatch.setenv("WEBUI_V2_MODE", mode)
    state.switches.set("tool:run_generated_capability", True, 0, {}, "operator")
    state.materialized.clear()
    assert generated_scope.search()["total"] == 0
    with pytest.raises(RuntimeError, match="RSI Plaza"):
        search_generated_capabilities.invoke({})
    assert state.materialized == []
