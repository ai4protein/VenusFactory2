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


def search_entrypoint(surface):
    if surface in {"native", "native-guarded"}:
        # The production hub guards shared tool objects in place. Isolate this
        # invocation so test order cannot silently add an authorization read.
        function = search_generated_capabilities.func
        while getattr(function, "_availability_guard", False):
            function = function.__wrapped__
        selected = search_generated_capabilities.model_copy(update={"func": function})
        if surface == "native-guarded":
            capability_policy.guard_tool(selected)

        def page(offset=0):
            value = json.loads(selected.invoke({"query": "synthetic", "limit": 3, "offset": offset}))
            return value["total"], [item["id"] for item in value["items"]], value
        return page
    mapping = {tool.name: tool for tool in RESEARCH_CAPABILITY_TOOLS}
    adapter = ScopedScientistCatalog(mapping, lambda tool: True)

    def page(offset=0):
        value = adapter.generated_page("synthetic", 3, offset)
        return value["total"], [item["name"].removeprefix(adapter.prefix) for item in value["tools"]], value
    return page


# Guarded execution adds one live entry authorization read. Inventory reads
# remain constant for 10,000 records; neither path may materialize full catalogs.
@pytest.mark.parametrize("surface,maximum_reads", [("native", 3), ("native-guarded", 4), ("scoped", 7)])
def test_one_policy_read_per_inventory_in_both_private_discovery_paths(large_owned_catalog, surface, maximum_reads):
    state = large_owned_catalog
    page = search_entrypoint(surface)
    for offset in (0, 3):
        state.reads.clear()
        state.materialized.clear()
        total, ids, response = page(offset)
        assert total == 10000
        assert ids == [f"{i:064x}" for i in range(offset, offset + 3)]
        assert len(state.materialized) == 3
        assert not any(key in json.dumps(response) for key in ("input_schema", "output_schema", "validation", "test_schema_marker"))
        assert len(state.reads) <= maximum_reads, f"{surface}: {len(state.reads)} switch-table SELECTs for 10,000 metadata entries"
    # The final private page uses actual owner/lease/goal provenance.
    total, ids, _ = page(9999)
    assert total == 10000 and ids == ["a" * 64]


@pytest.mark.parametrize("surface", ["native", "native-guarded", "scoped"])
def test_next_request_sees_admin_switch_changes_without_recreating_adapter(large_owned_catalog, surface):
    state = large_owned_catalog
    page = search_entrypoint(surface)
    first = "0" * 64
    assert page()[1][0] == first
    spec = {"id": "capability:" + first, "adapter": "capability", "tools": ["run_generated_capability"]}
    revision = state.switches.set("model:capability:" + first, False, 0, spec, "test")
    total, ids, _ = page()
    assert total == 9999 and first not in ids
    state.switches.set("model:capability:" + first, True, revision, spec, "test")
    assert page()[0] == 10000 and page()[1][0] == first
    # Calls never reuse discovery's temporary policy map.
    scope = generated_scope.GeneratedScope()
    assert scope.library_for(first) == runtime.LIBRARY
    state.switches.set("model:capability:" + first, False, revision + 1, spec, "test")
    with pytest.raises(capability_policy.CapabilityDisabled):
        scope.library_for(first)


@pytest.mark.parametrize("surface,revocation", [("native", "owner"), ("native-guarded", "owner"), ("scoped", "lease")])
def test_private_revocation_remains_live_after_large_search(large_owned_catalog, surface, revocation):
    state = large_owned_catalog
    page = search_entrypoint(surface)
    assert page(9999)[1] == ["a" * 64]
    scope = generated_scope.GeneratedScope()
    scope.references(lambda *args: 1)
    if revocation == "owner":
        with sqlite3.connect(state.database) as connection:
            connection.execute("UPDATE sessions SET owner_key='owner-b' WHERE session_id='research-a'")
    else:
        state.active_queue.cancel(state.active_lease["id"], "owner-a")
    assert scope.cards(["a" * 64]) == {}
    assert page(9999)[:2] == (9999, [])
    with pytest.raises(ValueError, match="Model unavailable"):
        scope.library_for("a" * 64)
