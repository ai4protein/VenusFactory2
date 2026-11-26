"""The real SSE loop persists host result cards independently of native text."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from agent.harness.contracts import Budget, ModelBinding, Profile
from agent.session_store import SessionStore, SqliteBackend
from backend.web_v2.chat_api import _stream_agent as transport


@pytest.mark.asyncio
async def test_result_is_saved_before_answer_and_replayed_once(tmp_path, monkeypatch):
    profile = Profile("science_agent", "kimi-code",
                      ModelBinding("fixture", "model", "http://127.0.0.1:1/v1"),
                      Budget(wall_seconds=10))
    binding = {"profile": profile.public(), "sha256": "b" * 64}
    result = {"receipt_id": "a" * 64, "content": "Prediction 0.9942129558371191, ratio.",
              "scope_markdown": "Temperature 21.8–65.5 C.", "files": {}}
    rows = [{"kind": "catalog", "tools": ["mcp__venusfactory__search_tools"]},
            {"kind": "model_result", "result": result},
            {"kind": "model_result", "result": result},
            {"kind": "text", "text": "Original model answer."},
            {"kind": "completed", "text": "Original model answer.", "messages": []}]
    stdout = asyncio.StreamReader()
    stdout.feed_data("".join(json.dumps(row) + "\n" for row in rows).encode())
    stdout.feed_eof()

    async def nothing(*args):
        return None

    class Input:
        def write(self, value):
            pass
        drain = nothing

    proc = SimpleNamespace(stdout=stdout, stdin=Input(), returncode=0, wait=nothing)

    async def spawn(*args, **kwargs):
        return proc

    class Gateway:
        base_url = "http://127.0.0.1:1/v1"
        token = "fixture"
        def __init__(self, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass

    async def context(*args):
        return [], "", {}

    async def cancelled(*args):
        return False

    monkeypatch.setattr(transport, "admit", lambda state: (binding, "fixture"))
    monkeypatch.setattr(transport, "EvidenceGateway", Gateway)
    monkeypatch.setattr(transport, "prepare_context", context)
    monkeypatch.setattr(transport, "runtime_paths", lambda: {})
    monkeypatch.setattr(transport.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(transport, "stop_process", nothing)
    monkeypatch.setattr(transport, "_is_cancelled", cancelled)
    monkeypatch.setattr(transport, "audit_requests", lambda *args: {"passed": True})
    monkeypatch.setattr(transport, "audit_trace", lambda *args: {"passed": True})
    monkeypatch.setattr(transport, "EVIDENCE_ROOT", tmp_path/"evidence")
    store = SessionStore(backend=SqliteBackend(str(tmp_path/"sessions.db")),
                         ensure_runtime_fn=lambda state: state)
    monkeypatch.setattr(transport, "_session_store", store)
    state = {"session_id": "result-session", "history": [], "created_at": "fixture",
             "agent_session_dir": str(tmp_path/"workspace")}
    await store.create(state)
    seen = []
    async for frame in transport._stream_agent(state, "Predict", []):
        if frame.startswith("event: state\n"):
            snapshot = json.loads(frame.split("data: ", 1)[1])
            if any(row.get("kind") == "model_result" for row in snapshot["history"]):
                seen.append(snapshot)
                # This is an actual SQLite reload while the generator is suspended.
                restored = SessionStore(backend=SqliteBackend(str(tmp_path/"sessions.db")),
                                        ensure_runtime_fn=lambda state: state)
                disk = await restored.get("result-session")
                assert sum(row.get("kind") == "model_result" for row in disk["history"]) == 1
    assert seen and seen[0]["status"] == "chat_mode"
    assert not any(row.get("content") == "Original model answer." for row in seen[0]["history"])
    assert state["status"] == "completed"
    card = next(row for row in state["history"] if row.get("kind") == "model_result")
    assert card["content"] == result["content"]
    assert card["model_result"]["scope_markdown"] == result["scope_markdown"]
    assert state["agent_messages"][-1]["content"] == "Original model answer."
    assert state["conversation_log"][-1]["content"] == "Original model answer."
