import copy

import pytest

from agent.harness.events import normalize_dsh


def test_compaction_summary_is_not_completion_and_keeps_raw_event():
    event = {"type": "compaction/summary", "seq": 8,
             "data": {"compactionId": "c", "summary": [{"type": "text", "text": "native only"}]}}
    before = copy.deepcopy(event)
    row = normalize_dsh(event, session_id="s")[0]
    assert event == before and row["phase"] == "summary"
    assert row["session_id"] == "s" and row["compaction_id"] == "c"
    assert len(row["native_event_sha256"]) == 64 and "summary" not in row


def test_native_compaction_failure_does_not_report_success():
    row = normalize_dsh({"type": "compaction/end", "data": {"compactionId": "c", "error": "failed"}},
                        session_id="s")[0]
    assert row["phase"] == "failed"


def test_compaction_and_continuation_share_session():
    events = [
        {"type": "compaction/start", "data": {"compactionId": "c"}},
        {"type": "compaction/end", "data": {"compactionId": "c"}},
        {"type": "tool/call", "data": {"callId": "t", "name": "tool", "arguments": {}}},
        {"type": "tool/result", "data": {"message": {"content": [
            {"type": "tool-result", "toolCallId": "t", "content": [{"type": "text", "text": "ok"}]}
        ]}}},
    ]
    rows = [row for event in events for row in normalize_dsh(event, session_id="same")]
    assert [row.get("phase", row["kind"]) for row in rows] == [
        "started", "completed", "tool_call_start", "tool_result"]
    assert all(row["session_id"] == "same" for row in rows)


def test_terminal_only_comes_from_native_completed_turn_and_live_counter():
    from agent.harness.adapters import _dsh_observation

    event = {"type": "turn/end", "data": {"reason": {"kind": "completed"}}}
    row = _dsh_observation(event, "session", 7)[0]
    assert row["kind"] == "native_terminal" and row["phase"] == "completed"
    assert row["gateway_calls"] == 7 and row["session_id"] == "session"
    assert _dsh_observation({"type": "turn/end", "data": {"reason": {"kind": "error"}}}, "s", 8) == []


@pytest.mark.asyncio
async def test_worker_lines_preserve_large_native_event_without_line_limit():
    from agent.harness.adapters import _worker_lines

    class Stream:
        def __init__(self):
            self.chunks = iter([b"a" * 70000, b"b\nnext", b"\n"])
        async def read(self, _):
            return next(self.chunks, b"")

    assert [line async for line in _worker_lines(Stream())] == [b"a" * 70000 + b"b", b"next"]


def test_notification_requires_original_session_and_event():
    from agent.harness.adapters import _dsh_observation

    event = {"type": "turn/end", "data": {"reason": {"kind": "completed"}}}
    with pytest.raises(ValueError, match="mismatch"):
        _dsh_observation(event, "s", 7, {"sessionId": "other", "event": event})
    with pytest.raises(ValueError, match="mismatch"):
        _dsh_observation(event, "s", 7, {"sessionId": "s", "event": {}})
    row = _dsh_observation(event, "s", 7, {"sessionId": "s", "event": event})[0]
    assert row["native_notification_path"] == "runtime_notifications.jsonl"
    assert len(row["native_notification_sha256"]) == 64
