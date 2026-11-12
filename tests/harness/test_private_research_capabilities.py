"""Private discovery requires an actual leased task and live session ownership."""

import hashlib
import json
import sqlite3
from types import SimpleNamespace

import pytest
from test_autonomous_research import profile

from agent.harness import autonomous_research as ar
from agent.harness import capability_catalog as catalog
from agent.harness import research_workspace as rw
from backend.web_v2.job_queue import JobQueue
from backend.web_v2.research_capability_access import owned_sessions, private_libraries
from mcp_tools.tools import capability_policy
from mcp_tools.tools.discovery import generated_scope, service


def configure_access(tmp_path, monkeypatch):
    database = tmp_path / "sessions.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE sessions(session_id TEXT PRIMARY KEY, owner_key TEXT NOT NULL)")
    state = SimpleNamespace(database=database, root=tmp_path / "scientist", queue=tmp_path / "queue")
    monkeypatch.setenv("VENUSFACTORY_SESSION_DB_PATH", str(database))
    monkeypatch.setenv("VENUS_SCIENCE_GOAL_ROOT", str(state.root))
    monkeypatch.setenv("VENUS_JOB_QUEUE_ROOT", str(state.queue))
    monkeypatch.setenv("VENUS_QUEUE_LEASE_SECONDS", "600")
    monkeypatch.setenv("VENUS_USAGE_DB", str(tmp_path / "usage.sqlite3"))
    monkeypatch.setattr(rw, "LIBRARY", tmp_path / "public")
    switches = capability_policy.CapabilityStore(path=tmp_path / "switches.sqlite3")
    monkeypatch.setattr(capability_policy, "CapabilityStore", lambda: switches)
    state.switches = switches
    return state


def session(state, identity, owner):
    with sqlite3.connect(state.database) as connection:
        connection.execute("INSERT OR REPLACE INTO sessions VALUES (?,?)", (identity, owner))


def lease(state, monkeypatch, owner, session_id, *, kind="agent", parent=None):
    queue = JobQueue(state.queue)
    namespace = "private-capability-fixture"
    resource = "cpu" if parent else "api:agent"
    metadata = {"session_id": session_id}
    if parent:
        metadata["parent_job"] = parent
    identity, _ = queue.enqueue(owner, kind, resource, {"fixture": True}, namespace, metadata=metadata)
    row = queue.claim_next(namespace, resource.split(":")[0])
    assert row and row["id"] == identity
    monkeypatch.setenv("VENUS_QUEUE_JOB_ID", identity)
    monkeypatch.setenv("VENUS_QUEUE_OWNER", owner)
    monkeypatch.setenv("VENUS_QUEUE_LEASE_TOKEN", row["lease"])
    return queue, row


def attach_goal(state, goal, source_session):
    assert goal.root.parent == state.root / hashlib.sha256(source_session.encode()).hexdigest()
    (goal.root.parent / (goal.root.name + ".json")).write_text(
        json.dumps({"session_id": source_session, "spec": {"driver": "autonomous_research.v1"}})
    )


def metadata_goal(state, source_session, identity):
    folder = state.root / hashlib.sha256(source_session.encode()).hexdigest() / identity[:32]
    goal = ar.create(
        folder,
        profile(),
        "Develop a synthetic metadata fixture model",
        {"max_cycles": 1, "model_calls": 12, "tool_calls": 12, "wall_seconds": 120},
    )
    # Metadata only; the deliberately unverified SC binding cannot be invoked.
    goal.mutate(
        lambda value: value["progress"].update(
            deliveries=[{"candidate_id": identity, "kind": "model", "scientific_acceptance": {"fixture": "unverified"}}]
        )
    )
    attach_goal(state, goal, source_session)
    library = folder / "private_capabilities"
    card = {
        "id": identity,
        "name": "Private " + source_session,
        "description": "Synthetic metadata only",
        "kind": "model",
        "visibility": "private",
        "review_status": "approved",
        "scope": {},
    }
    with catalog.database(library) as connection:
        connection.execute(
            "INSERT INTO capabilities VALUES (?,?,?,?,?,?,?,?)",
            (identity, folder.name, card["name"], "1.0", "synthetic metadata model", json.dumps(card), "fixture", 0),
        )
    return goal


@pytest.fixture
def private_environment(tmp_path, monkeypatch):
    state = configure_access(tmp_path, monkeypatch)
    session(state, "agent-a", "owner-a")
    session(state, "research-a", "owner-a")
    session(state, "research-b", "owner-b")
    metadata_goal(state, "research-a", "a" * 64)
    metadata_goal(state, "research-b", "b" * 64)
    return state


def test_search_binds_current_owner_and_never_reveals_other_sessions(private_environment, monkeypatch):
    state = private_environment
    lease(state, monkeypatch, "owner-a", "agent-a")
    assert set(owned_sessions()) == {"agent-a", "research-a"}
    page = generated_scope.search("synthetic")
    assert [item["id"] for item in page["items"]] == ["a" * 64]
    assert page["items"][0]["visibility"] == "private"
    assert "research-b" not in json.dumps(page) and str(state.root) not in json.dumps(page)
    with pytest.raises(ValueError, match="Model unavailable"):
        generated_scope.describe("b" * 64)
    with pytest.raises((ValueError, KeyError, FileNotFoundError)):
        generated_scope.describe("a" * 64)  # Metadata cannot bypass actual SC evidence.
    monkeypatch.setattr(service, "model_cards", lambda: [])
    monkeypatch.setattr(service, "platform_models", lambda *args: [])
    assert service.search(kind="models", tool_catalog=[])["results"][0]["entry_id"] == "a" * 64


@pytest.mark.parametrize(
    "damage",
    [
        "missing_lease",
        "wrong_lease",
        "wrong_owner",
        "deleted_origin",
        "reassigned_origin",
        "cancelled",
        "stopped",
        "symlink_goal",
    ],
)
def test_revocation_is_rechecked_after_search(private_environment, monkeypatch, damage):
    state = private_environment
    queue, row = lease(state, monkeypatch, "owner-a", "agent-a")
    scope = generated_scope.GeneratedScope()
    assert scope.references(lambda *args: 1) == [("a" * 64, 1)]
    if damage == "missing_lease":
        monkeypatch.delenv("VENUS_QUEUE_LEASE_TOKEN")
    elif damage == "wrong_lease":
        monkeypatch.setenv("VENUS_QUEUE_LEASE_TOKEN", "not-current")
    elif damage == "wrong_owner":
        monkeypatch.setenv("VENUS_QUEUE_OWNER", "owner-b")
    elif damage in {"deleted_origin", "reassigned_origin"}:
        with sqlite3.connect(state.database) as connection:
            if damage == "deleted_origin":
                connection.execute("DELETE FROM sessions WHERE session_id='research-a'")
            else:
                connection.execute("UPDATE sessions SET owner_key='owner-b' WHERE session_id='research-a'")
    elif damage == "cancelled":
        queue.cancel(row["id"], "owner-a")
    elif damage == "stopped":
        queue.finish(row["id"], "completed", lease=row["lease"])
    else:
        folder = state.root / hashlib.sha256(b"research-a").hexdigest() / ("a" * 32)
        saved = folder.with_name("hidden-saved")
        folder.rename(saved)
        folder.symlink_to(saved, target_is_directory=True)
    assert scope.cards(["a" * 64]) == {}
    assert generated_scope.search()["total"] == 0
    with pytest.raises(ValueError, match="Model unavailable"):
        scope.library_for("a" * 64)


def test_queued_child_uses_parent_session_not_a_model_supplied_session(private_environment, monkeypatch):
    state = private_environment
    queue, parent = lease(state, monkeypatch, "owner-a", "agent-a")
    lease(state, monkeypatch, "owner-a", "research-b", kind="mcp:cpu", parent=parent["id"])
    assert [list(item.deliveries) for item in private_libraries()] == [["a" * 64]]
    queue.cancel(parent["id"], "owner-a")
    assert private_libraries() == []


@pytest.mark.parametrize("mode,allowed", [("local", True), ("online", False)])
def test_local_scientist_legacy_queue_owner_does_not_grant_online_access(
    private_environment, monkeypatch, mode, allowed
):
    state = private_environment
    monkeypatch.setenv("WEBUI_V2_MODE", mode)
    lease(state, monkeypatch, hashlib.sha256(b"research-a").hexdigest(), "research-a", kind="scientist")
    assert bool(private_libraries()) is allowed


async def verify_private_reuse(state, goal, source_session, monkeypatch):
    """Continue the actual trained/SC-reviewed worker-recovery fixture via Agent MCP."""
    from fastmcp import Client, FastMCP
    from fastmcp.exceptions import ToolError

    from agent.harness.tool_disclosure import DisclosureError, ToolDisclosure
    from mcp_tools.tools import runtime_tool_policy
    from mcp_tools.tools.discovery.research_capabilities import RESEARCH_CAPABILITY_TOOLS
    from mcp_tools.tools.discovery.scoped import ScopedScientistCatalog
    from mcp_tools.tools.discovery.tools_mcp import register_discovery_tools

    identity = goal.read()["progress"]["deliveries"][0]["candidate_id"]
    session(state, source_session, "fixture-owner")
    session(state, "agent-fixture", "fixture-owner")
    session(state, "agent-foreign", "foreign-owner")
    attach_goal(state, goal, source_session)
    queue, root = lease(state, monkeypatch, "fixture-owner", "agent-fixture")
    monkeypatch.setattr(service, "model_cards", lambda: [])
    monkeypatch.setattr(service, "platform_models", lambda *args: [])
    # The online policy is applied throughout. Private reuse never publishes a
    # model to the shared library or requires a permissive local policy.
    monkeypatch.setattr(runtime_tool_policy, "is_online_mode", lambda: True)
    mapping = {tool.name: tool for tool in RESEARCH_CAPABILITY_TOOLS}
    adapter = ScopedScientistCatalog(mapping, lambda tool: True)
    disclosure = ToolDisclosure(
        mapping,
        schema=lambda tool: tool.args_schema.model_json_schema(),
        allowed=lambda tool: True,
        model_catalog=adapter,
    )
    overview = disclosure.metadata("search_tools", {})
    assert any(row["id"] == "rsi" for row in overview["categories"])
    assert overview["tools"] == [] and identity not in json.dumps(disclosure.surface())
    page = disclosure.metadata("search_tools", {"category": "rsi", "query": "recovery"})
    name = adapter.prefix + identity
    assert page["tools"][0]["name"] == name
    assert set(page["tools"][0]) == {"name", "description"}
    assert "inputSchema" not in json.dumps(page) and str(goal.root) not in json.dumps(page)
    with pytest.raises(DisclosureError):
        disclosure.resolve("call_tool", {"name": name, "arguments": {"inputs": {"x": "3"}}})
    contract = disclosure.metadata("describe_tool", {"name": name})
    assert contract["inputSchema"]["properties"]["inputs"]["required"] == ["x"]
    assert contract["review_status"] == "approved" and contract["visibility"] == "private"
    assert "validation" not in contract and "review_receipt" not in contract
    with pytest.raises(DisclosureError):
        disclosure.resolve("call_tool", {"name": name, "arguments": {"inputs": {"x": "3"}, "owner": "foreign-owner"}})
    with pytest.raises(DisclosureError, match="contract"):
        disclosure.resolve("call_tool", {"name": name, "arguments": {"inputs": {"x": 3}}})
    key, tool, arguments = disclosure.resolve("call_tool", {"name": name, "arguments": {"inputs": {"x": "3"}}})
    assert key == "run_generated_capability" and tool is mapping[key]
    assert arguments == {"capability_id": identity, "inputs": {"x": "3"}}
    # Real scheduler child credentials preserve the parent browser identity;
    # the model-supplied child session ID is deliberately unrelated.
    _, child = lease(state, monkeypatch, "fixture-owner", "untrusted-child-session", kind="mcp:cpu", parent=root["id"])
    assert json.loads(await tool.ainvoke(arguments))["value"] == pytest.approx(9)
    server = FastMCP("owner-private trained capability fixture")
    register_discovery_tools(server)
    async with Client(server) as client:
        page = (await client.call_tool("search_generated_capabilities", {})).data
        assert [card["id"] for card in page["items"]] == [identity]
        native = json.loads(mapping["search_generated_capabilities"].invoke({}))
        assert native == page
        generic = (await client.call_tool("search_research_tools", {"kind": "models"})).data
        assert [card["entry_id"] for card in generic["results"]] == [identity]
        detail = (await client.call_tool("describe_generated_capability", {"capability_id": identity})).data
        assert detail["input_schema"]["required"] == ["x"]
        assert (await client.call_tool("run_generated_capability", arguments)).data["value"] == pytest.approx(9)
        with pytest.raises(ToolError):
            await client.call_tool("run_generated_capability", {"capability_id": identity, "inputs": {"x": "invalid"}})
        # Tampering with the independent SC record invalidates an already
        # inspected interface. Restore this test-owned fixture byte-for-byte.
        delivery = goal.read()["progress"]["deliveries"][0]
        receipt = goal.root / delivery["scientific_acceptance"]["receipt"]
        original = receipt.read_bytes()
        try:
            receipt.write_text("{}")
            with pytest.raises(DisclosureError):
                disclosure.resolve("call_tool", {"name": name, "arguments": {"inputs": {"x": "3"}}})
            with pytest.raises(ToolError):
                await client.call_tool("run_generated_capability", arguments)
        finally:
            receipt.write_bytes(original)
        # Operator disable must revoke a previously described binding as well
        # as both discovery surfaces, without restarting any process.
        state.switches.set(
            "model:capability:" + identity,
            False,
            0,
            {"id": "capability:" + identity, "adapter": "capability", "tools": ["run_generated_capability"]},
            "test",
        )
        assert (await client.call_tool("search_generated_capabilities", {})).data["total"] == 0
        with pytest.raises(DisclosureError):
            disclosure.resolve("call_tool", {"name": name, "arguments": {"inputs": {"x": "3"}}})
        with pytest.raises(ToolError):
            await client.call_tool("run_generated_capability", arguments)
        state.switches.set("model:capability:" + identity, True, 1, {}, "test")
        await verify_native_private_reuse(state, goal, identity, monkeypatch)
        # A different real owner cannot enumerate or invoke even a known ID.
        queue.finish(child["id"], "completed", lease=child["lease"])
        queue.finish(root["id"], "completed", lease=root["lease"])
        _, foreign = lease(state, monkeypatch, "foreign-owner", "agent-foreign")
        assert (await client.call_tool("search_generated_capabilities", {})).data["total"] == 0
        with pytest.raises(ToolError):
            await client.call_tool("describe_generated_capability", {"capability_id": identity})
        with pytest.raises(ToolError):
            await client.call_tool("run_generated_capability", arguments)
        queue.finish(foreign["id"], "completed", lease=foreign["lease"])
        lease(state, monkeypatch, "fixture-owner", "agent-fixture")
        assert (await client.call_tool("search_generated_capabilities", {})).data["total"] == 1
        with sqlite3.connect(state.database) as connection:
            connection.execute("DELETE FROM sessions WHERE session_id=?", (source_session,))
        assert (await client.call_tool("search_generated_capabilities", {})).data["total"] == 0
        with pytest.raises(DisclosureError):
            disclosure.resolve("call_tool", {"name": name, "arguments": {"inputs": {"x": "3"}}})
    # All reuse remained private; no metadata or weights were copied publicly.
    assert catalog.discovery_cards(rw.LIBRARY, [identity]) == {}


@pytest.mark.parametrize("limit", [1, 2, 3, 5])
def test_progressive_search_pages_across_new_and_older_model_registries(private_environment, monkeypatch, limit):
    from mcp_tools.tools.discovery.research_capabilities import RESEARCH_CAPABILITY_TOOLS
    from mcp_tools.tools.discovery.scoped import ScopedScientistCatalog

    state = private_environment
    metadata_goal(state, "research-a", "c" * 64)
    lease(state, monkeypatch, "owner-a", "agent-a")
    mapping = {tool.name: tool for tool in RESEARCH_CAPABILITY_TOOLS}
    adapter = ScopedScientistCatalog(mapping, lambda tool: True)
    old = [
        {"name": adapter.legacy.prefix + char * 64, "description": "Older reviewed metadata fixture"}
        for char in ("d", "e")
    ]

    def legacy(query, limit, offset):
        return {"total": len(old), "tools": old[offset : offset + limit], "next_offset": None}

    monkeypatch.setattr(adapter.legacy, "search", legacy)
    expected = [adapter.prefix + char * 64 for char in ("a", "c")] + [item["name"] for item in old]
    pages = [adapter.search("", limit, offset) for offset in range(0, 4, limit)]
    assert all(page["total"] == 4 for page in pages)
    assert [item["name"] for page in pages for item in page["tools"]] == expected
    assert pages[-1]["next_offset"] is None
    assert adapter.search("", limit, 4)["tools"] == []
    # Tool-level enablement still applies with a cached adapter.
    state.switches.set("tool:run_generated_capability", False, 0, {}, "test")
    assert adapter.search("", limit, 0) == legacy("", limit, 0)


def test_withdrawn_private_card_does_not_repeat_legacy_entries(private_environment, monkeypatch):
    from mcp_tools.tools.discovery.research_capabilities import RESEARCH_CAPABILITY_TOOLS
    from mcp_tools.tools.discovery.scoped import ScopedScientistCatalog

    state = private_environment
    metadata_goal(state, "research-a", "c" * 64)
    lease(state, monkeypatch, "owner-a", "agent-a")
    mapping = {tool.name: tool for tool in RESEARCH_CAPABILITY_TOOLS}
    adapter = ScopedScientistCatalog(mapping, lambda tool: True)
    old = [{"name": adapter.legacy.prefix + "d" * 64, "description": "Older fixture"}]
    monkeypatch.setattr(
        adapter.legacy, "search", lambda query, limit, offset: {"total": 1, "tools": old[offset : offset + limit]}
    )
    monkeypatch.setattr(generated_scope.GeneratedScope, "cards", lambda *args: {})
    first, second = adapter.search("", 2, 0), adapter.search("", 2, 2)
    assert first["tools"] == [] and first["next_offset"] == 2
    assert second["tools"] == old and second["next_offset"] is None


def test_model_navigation_does_not_import_training_or_review_runtime():
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    code = """import sys
from mcp_tools.tools.discovery.scoped import ScopedScientistCatalog
for name in ('agent.harness.capability_review', 'agent.harness.research_workspace', 'agent.harness.scientific_acceptance', 'mcp_tools.models.evolved.service'):
    assert name not in sys.modules, name
"""
    subprocess.run(
        [sys.executable, "-c", code], cwd=root, env={**os.environ, "PYTHON_DOTENV_DISABLED": "1"}, check=True
    )


async def verify_native_private_reuse(state, goal, identity, monkeypatch):
    """Real MCP navigation and model execution; controlled harness protocol only."""
    import httpx

    from agent.harness import web_native_worker as worker
    from agent.harness.contracts import Budget, ModelBinding, Profile
    from mcp_tools.scheduler import catalog as scheduler
    from mcp_tools.scheduler import client as queue_client
    from mcp_tools.tools.discovery.research_capabilities import RESEARCH_CAPABILITY_TOOLS
    from mcp_tools.tools.discovery.scoped import ScopedScientistCatalog

    mapping = {tool.name: tool for tool in RESEARCH_CAPABILITY_TOOLS}
    queued = []
    name = ScopedScientistCatalog.prefix + identity

    async def invoke(tool_name, arguments, **host):
        queued.append((tool_name, arguments, host))
        return json.loads(await mapping[tool_name].ainvoke(arguments))

    monkeypatch.setattr(scheduler, "native_tools", lambda: mapping)
    monkeypatch.setattr(queue_client, "invoke", invoke)

    async def adapter(context):
        connection = json.loads((context.task.capability / "connection.json").read_text())
        async with httpx.AsyncClient(trust_env=False) as client:

            async def call(tool, arguments):
                response = await client.post(
                    connection["url"],
                    headers={"Authorization": "Bearer " + connection["token"]},
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {"name": tool, "arguments": arguments},
                    },
                )
                result = response.json()["result"]
                assert not result["isError"], result
                return json.loads(result["content"][0]["text"])

            overview = await call("search_tools", {})
            assert overview["tools"] == []
            page = await call("search_tools", {"category": "rsi", "query": "recovery"})
            assert page["tools"][0]["name"] == name
            contract = await call("describe_tool", {"name": name})
            assert contract["inputSchema"]["properties"]["inputs"]["required"] == ["x"]
            output = await call("call_tool", {"name": name, "arguments": {"inputs": {"x": "3"}}})
            assert output["value"] == pytest.approx(9)
        return {"status": "completed", "text": "Synthetic prediction: 9."}

    monkeypatch.setattr(
        worker, "builtin_registry", lambda: SimpleNamespace(resolve=lambda *args: (SimpleNamespace(run=adapter), None))
    )
    for harness in ("kimi-code", "deepseek-harness", "codex", "claude-code"):
        workspace = state.root / ("native-" + harness)
        workspace.mkdir()
        control = workspace / "control"
        control.mkdir()
        profile = Profile(
            "science_agent", harness, ModelBinding("fixture", "model", "http://127.0.0.1:1/v1"), Budget(tool_calls=8)
        )
        events = []
        await worker.run(
            {
                "profile": profile.public(),
                "workspace": str(workspace),
                "control": str(control),
                "session_id": "agent-fixture",
                "runtimes": {},
                "base_url": "http://127.0.0.1:1/v1",
                "token": "fixture",
                "text": "Use my synthetic recovery model for x=3",
            },
            tools=RESEARCH_CAPABILITY_TOOLS,
            event_sink=events.append,
        )
        assert any(
            event.get("kind") == "completed" and event.get("text") == "Synthetic prediction: 9." for event in events
        )
    assert len(queued) == 4
    assert all(
        tool == "run_generated_capability" and args == {"capability_id": identity, "inputs": {"x": "3"}}
        for tool, args, host in queued
    )
