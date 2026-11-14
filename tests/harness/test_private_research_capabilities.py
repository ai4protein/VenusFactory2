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
from mcp_tools.tools.capability_policy import CapabilityDisabled
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
    assert page["items"] == [] and page["total"] == 0
    assert "research-b" not in json.dumps(page) and str(state.root) not in json.dumps(page)
    with pytest.raises(RuntimeError, match="RSI Plaza"):
        generated_scope.describe("b" * 64)
    with pytest.raises(RuntimeError, match="RSI Plaza"):
        generated_scope.describe("a" * 64)  # Metadata cannot bypass actual SC evidence.
    monkeypatch.setattr(service, "model_cards", lambda: [])
    monkeypatch.setattr(service, "platform_models", lambda *args: [])
    assert service.search(kind="models", tool_catalog=[])["results"] == []


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
    with pytest.raises(RuntimeError, match="RSI Plaza"):
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
    """A real SC-reviewed recovery fixture remains private and cannot enter Agent."""
    from fastmcp import Client, FastMCP
    from fastmcp.exceptions import ToolError
    from mcp_tools.tools.discovery.tools_mcp import register_discovery_tools

    before = goal.read()
    identity = before["progress"]["deliveries"][0]["candidate_id"]
    session(state, source_session, "fixture-owner")
    session(state, "agent-fixture", "fixture-owner")
    attach_goal(state, goal, source_session)
    queue, active = lease(state, monkeypatch, "fixture-owner", "agent-fixture")
    monkeypatch.setenv("WEBUI_V2_MODE", "online")
    server = FastMCP("RSI cases are not Agent tools")
    register_discovery_tools(server)
    async with Client(server) as client:
        names = {tool.name for tool in await client.list_tools()}
        retired = {
            "search_generated_capabilities": {},
            "describe_generated_capability": {"capability_id": identity},
            "run_generated_capability": {"capability_id": identity, "inputs": {"x": "3"}},
        }
        assert not (names & retired.keys())
        for name, arguments in retired.items():
            with pytest.raises(ToolError):
                await client.call_tool(name, arguments)
    assert goal.read() == before
    assert catalog.discovery_cards(rw.LIBRARY, [identity]) == {}
    queue.finish(active["id"], "completed", lease=active["lease"])


@pytest.mark.parametrize("mode", ["online", "local"])
@pytest.mark.parametrize("tool_name", ["search_generated_capabilities", "describe_generated_capability", "run_generated_capability"])
def test_old_bindings_and_admin_switch_cannot_restore_agent_execution(private_environment, monkeypatch, mode, tool_name):
    from mcp_tools.tools.discovery.research_capabilities import RESEARCH_CAPABILITY_TOOLS
    from mcp_tools.tools.discovery.scoped import ScopedScientistCatalog

    state = private_environment
    monkeypatch.setenv("WEBUI_V2_MODE", mode)
    lease(state, monkeypatch, "owner-a", "agent-a")
    mapping = {tool.name: tool for tool in RESEARCH_CAPABILITY_TOOLS}
    adapter = ScopedScientistCatalog(mapping, lambda tool: True)
    name = adapter.prefix + "a" * 64
    adapter.bindings[name] = {"type": "object"}
    state.switches.set("tool:" + tool_name, True, 0, {}, "test operator")
    assert not adapter.generated_available()
    assert adapter.generated_page("", 3, 0) is None
    with pytest.raises(ValueError):
        adapter.describe(name)
    with pytest.raises(ValueError):
        adapter.resolve(name, {"inputs": {}})
