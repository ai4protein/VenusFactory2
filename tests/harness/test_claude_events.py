import copy
import json

import pytest

from agent.harness.claude_events import ClaudeEvents
from agent.harness.events import audit_trace

TOOL = "mcp__venusfactory__protein_probe"


def init():
    return {"type": "system", "subtype": "init", "session_id": "s1"}


def call():
    return {
        "type": "assistant",
        "uuid": "a1",
        "session_id": "s1",
        "message": {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "c1", "name": TOOL, "input": {"sequence": "ACDEFGHIKC"}}],
        },
    }


def result():
    return {
        "type": "user",
        "uuid": "u1",
        "session_id": "s1",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "c1",
                    "is_error": False,
                    "content": [{"type": "text", "text": '{"length":10,"cysteines":2}'}],
                }
            ],
        },
    }


def terminal():
    return {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "session_id": "s1",
        "result": "10 residues and 2 cysteines",
    }


def test_partial_events_do_not_duplicate_complete_actions(tmp_path):
    validated = []
    state = ClaudeEvents({"protein_probe"}, 1, lambda name, data: validated.append((name, data)))
    rows = []
    for event in [
        init(),
        {"type": "system", "subtype": "status", "session_id": "s1", "status": None},
        {
            "type": "stream_event",
            "session_id": "s1",
            "event": {"type": "content_block_start", "content_block": call()["message"]["content"][0]},
        },
        call(),
        result(),
        terminal(),
    ]:
        rows.extend(state.consume(event))
    assert len(rows) == 2 and state.completed and not state.failed and state.tool_errors == 0
    assert validated == [("protein_probe", {"length": 10, "cysteines": 2})]
    path = tmp_path / "trace.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert audit_trace(path, {"protein_probe"}, 1)["passed"]
    with pytest.raises(ValueError):
        state.consume(terminal())


@pytest.mark.parametrize(
    "fault",
    [
        "missing_init",
        "unknown_tool",
        "duplicate_start",
        "missing_result",
        "duplicate_result",
        "result_first",
        "session_change",
        "nested",
        "budget",
        "malformed_result",
        "unknown_system",
        "hook",
    ],
)
def test_invalid_native_evidence_is_rejected(fault):
    state = ClaudeEvents({"protein_probe"}, 0 if fault == "budget" else 1)
    if fault == "missing_init":
        with pytest.raises(ValueError):
            state.consume(call())
        return
    state.consume(init())
    event = call()
    if fault == "unknown_tool":
        event["message"]["content"][0]["name"] = "Bash"
    elif fault == "session_change":
        event["session_id"] = "different"
    elif fault == "nested":
        event["parent_tool_use_id"] = "parent"
    elif fault == "result_first":
        event = result()
    elif fault in {"unknown_system", "hook"}:
        event = {"type": "system", "subtype": "hook_started" if fault == "hook" else "unknown", "session_id": "s1"}
    elif fault in {"duplicate_start", "missing_result", "duplicate_result", "malformed_result"}:
        state.consume(call())
        if fault == "duplicate_start":
            event = copy.deepcopy(call())
            event["uuid"] = "a2"
        elif fault == "missing_result":
            event = terminal()
        elif fault == "duplicate_result":
            state.consume(result())
            event = copy.deepcopy(result())
            event["uuid"] = "u2"
        else:
            event = result()
            event["message"]["content"][0]["is_error"] = "false"
    with pytest.raises(ValueError):
        state.consume(event)


def test_error_result_is_retained_and_never_scientifically_validated():
    def fail(*args):
        raise AssertionError("Failed result should not be validated")

    state = ClaudeEvents({"protein_probe"}, 1, fail)
    state.consume(init())
    state.consume(call())
    event = result()
    event["message"]["content"][0]["is_error"] = True
    row = state.consume(event)[0]
    assert row["is_error"] is True and state.tool_errors == 1


def test_native_terminal_failure_does_not_count_as_success():
    state = ClaudeEvents(set(), 1)
    state.consume(init())
    event = terminal()
    event["permission_denials"] = [{"tool_name": "Bash"}]
    state.consume(event)
    assert state.failed and not state.completed

def heartbeat():
    return {"type": "tool_progress", "session_id": "s1", "tool_use_id": "c1-heartbeat-0",
            "tool_name": TOOL, "parent_tool_use_id": "c1", "heartbeat": True, "elapsed_time_seconds": 30}


def test_long_tool_heartbeat_does_not_start_or_finish_another_call():
    state = ClaudeEvents({"protein_probe"}, 1)
    state.consume(init()); state.consume(call())
    assert state.consume(heartbeat()) == []
    assert len(state.started) == 1 and not state.finished
    state.consume(result()); state.consume(terminal())
    assert state.completed and state.tool_errors == 0


@pytest.mark.parametrize("change", [
    {"session_id": "other"}, {"parent_tool_use_id": "unknown"},
    {"tool_name": "mcp__venusfactory__unregistered"}, {"heartbeat": False},
    {"elapsed_time_seconds": -1}, {"elapsed_time_seconds": float("nan")},
])
def test_unbound_tool_progress_is_rejected(change):
    state = ClaudeEvents({"protein_probe"}, 1)
    state.consume(init()); state.consume(call())
    with pytest.raises(ValueError):
        state.consume({**heartbeat(), **change})


def compact():
    return {"type": "system", "subtype": "compact_boundary", "session_id": "s1",
            "compact_metadata": {"trigger": "auto", "pre_tokens": 180000}}


def test_native_compaction_keeps_tool_ledger_and_continues(tmp_path):
    state = ClaudeEvents({"protein_probe"}, 1)
    state.consume(init())
    start = state.consume({"type": "system", "subtype": "status",
                           "session_id": "s1", "status": "compacting"})[0]
    boundary = compact()
    before = copy.deepcopy(boundary)
    rows = state.consume(boundary) + state.consume(call()) + state.consume(result())
    state.consume(terminal())
    assert boundary == before
    assert start["phase"] == "started"
    assert rows[0]["kind"] == "context_compaction" and rows[0]["phase"] == "completed"
    assert all(row["session_id"] == "s1" for row in rows)
    assert state.completed and not state.failed
    path = tmp_path / "events.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert audit_trace(path, {"protein_probe"}, 1)["passed"]


@pytest.mark.parametrize("change", [
    {"session_id": "other"}, {"compact_metadata": {}},
    {"compact_metadata": {"trigger": "auto", "pre_tokens": -1}},
    {"compact_metadata": {"trigger": "auto", "pre_tokens": True}},
])
def test_compaction_boundary_requires_bound_native_metadata(change):
    state = ClaudeEvents({"protein_probe"}, 1)
    state.consume(init())
    with pytest.raises(ValueError):
        state.consume({**compact(), **change})


def test_compaction_does_not_erase_outstanding_tool_pairing():
    state = ClaudeEvents({"protein_probe"}, 1)
    state.consume(init())
    state.consume(call())
    state.consume(compact())
    with pytest.raises(ValueError, match="outstanding"):
        state.consume(terminal())
