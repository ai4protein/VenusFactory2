"""Offline native recovery evidence: no model or summarization requests."""
import copy
import json
from pathlib import Path

import pytest

from agent.harness.contracts import digest
from agent.harness.events import audit_requests, native_compaction
from agent.harness.native_context_recovery import native_context_for_run, verify_native_continuation
from agent.harness.request_recovery import POLICY as OLD_POLICY, REJECTION_SHA256


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def evidence(tmp_path):
    sid = "fixture-session"
    raw = [
        {"type": "system", "subtype": "init", "session_id": sid},
        {"type": "system", "subtype": "compact_boundary", "session_id": sid,
         "compact_metadata": {"trigger": "auto", "pre_tokens": 100000}},
        {"type": "assistant", "session_id": sid, "message": {"content": [
            {"type": "tool_use", "id": "t1", "name": "mcp__venusfactory__probe", "input": {}}]}},
        {"type": "user", "session_id": sid, "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t1", "is_error": False, "content": "observed"}]}},
        {"type": "result", "subtype": "success", "session_id": sid, "is_error": False},
    ]
    rows = []
    for event, calls, kind in zip(raw[1:], [2, 3, 3, 4],
                                  ["context_compaction", "tool_call_start", "tool_result", "native_terminal"]):
        row = native_compaction(event, phase="completed", native_type=event["type"],
                                session_id=sid, path="native_events.jsonl")
        row.update(kind=kind, gateway_calls=calls)
        if kind == "tool_call_start":
            row.update(tool_call_id="t1", tool_name="mcp__venusfactory__probe", args={})
        elif kind == "tool_result":
            row.update(tool_call_id="t1", is_error=False, output="observed")
        rows.append(row)
    gateway = [{"event": "request", "call": 1},
               {"event": "upstream_rejection", "call": 1, "status": 400, "reported_signals": ["context_limit"]},
               {"event": "response", "call": 1, "status": 400, "completed": False, "error_type": None}]
    for call in [2, 3, 4]:
        gateway.extend([{"event": "request", "call": call},
                        {"event": "response", "call": call, "status": 200, "completed": True}])
    result = {"status": "completed", "session_id": sid, "sandbox_stopped": True}
    native = native_context_for_run(tmp_path, result, "claude-code", {"probe"}, 1, 4)
    write_rows(tmp_path / "native_events.jsonl", raw)
    write_rows(tmp_path / "events.jsonl", rows)
    write_rows(tmp_path / "requests.jsonl", gateway)
    return tmp_path, native, rows, raw, gateway


def test_recovered_failure_remains_charged_and_default_audit_is_strict(evidence):
    root, native, rows, raw, gateway = evidence
    path = root / "requests.jsonl"
    original = path.read_bytes()
    assert audit_requests(path)["passed"] is False
    result = audit_requests(path, native_context=native)
    assert result["passed"] is True
    assert result["model_calls"] == 4 and result["total_failed_attempts"] == 1
    assert result["uninterrupted_passed"] is False
    assert result["native_context_recovery"]["recovered_calls"][0]["failed_call"] == 1
    assert path.read_bytes() == original


@pytest.mark.parametrize("attack", ["raw_hash", "raw_session", "session", "no_anchor", "anchor_order",
    "boundary_after_success", "tool_error", "missing_terminal", "terminal_failed", "missing_tool",
    "phase", "scope", "wrong_status", "wrong_signal", "other_failure", "budget", "duplicate_call",
    "raw_tool_changed", "terminal_before_tool", "notice_only"])
def test_incomplete_or_changed_recovery_never_passes(evidence, attack):
    root, native, rows, raw, gateway = evidence
    if attack == "raw_hash": rows[0]["native_event_sha256"] = "0" * 64
    elif attack == "raw_session": raw[1]["session_id"] = "foreign"
    elif attack == "session": rows[0]["session_id"] = "foreign"
    elif attack == "no_anchor": rows[0].pop("gateway_calls")
    elif attack == "anchor_order": rows[2]["gateway_calls"] = 1
    elif attack == "boundary_after_success": rows[0]["gateway_calls"] = 3
    elif attack == "tool_error": rows[2]["is_error"] = True
    elif attack == "missing_terminal": rows.pop()
    elif attack == "terminal_failed": native["result"]["status"] = "runtime_error"
    elif attack == "missing_tool": rows[:] = [rows[0], rows[-1]]
    elif attack == "phase": rows[0]["phase"] = "summary"
    elif attack == "scope": native["calls_before"] = 1
    elif attack == "wrong_status": gateway[2]["status"] = 500
    elif attack == "wrong_signal": gateway[1]["reported_signals"] = ["model_unavailable"]
    elif attack == "other_failure": gateway[-1].update(status=500, completed=False)
    elif attack == "budget": gateway.append({"event": "budget_rejected"})
    elif attack == "duplicate_call": gateway.append(copy.deepcopy(gateway[0]))
    elif attack == "raw_tool_changed": raw[3]["message"]["content"][0]["is_error"] = True
    elif attack == "terminal_before_tool": raw[3], raw[4] = raw[4], raw[3]
    elif attack == "notice_only": rows[0]["phase"] = "notice"
    write_rows(root / "native_events.jsonl", raw)
    write_rows(root / "events.jsonl", rows)
    write_rows(root / "requests.jsonl", gateway)
    assert audit_requests(root / "requests.jsonl", native_context=native)["passed"] is False


def test_old_recovery_and_native_recovery_are_audited_independently(evidence):
    root, native, rows, raw, gateway = evidence
    lock = {"profile": {"request_recovery": OLD_POLICY,
            "model": {"provider": "dmx", "base_url": "https://www.dmxapi.cn"}, "budget": {"model_calls": 6}}}
    lock["sha256"] = digest(lock)
    (root / "profile.lock.json").write_text(json.dumps(lock))
    for row in gateway:
        if row["event"] == "request":
            row.update(recovery_policy=OLD_POLICY, logical_call=row["call"], attempt=1, retry_of=None)
    gateway += [
        {"event": "request", "call": 5, "recovery_policy": OLD_POLICY, "logical_call": 5, "attempt": 1,
         "retry_of": None, "request_sha256": "same"},
        {"event": "upstream_rejection", "call": 5, "status": 400, "body_prefix_sha256": REJECTION_SHA256,
         "truncated": False, "inspected_bytes": 229},
        {"event": "response", "call": 5, "status": 400, "completed": False, "error_type": None},
        {"event": "retry_scheduled", "failed_call": 5, "logical_call": 5, "delay_seconds": 0.5},
        {"event": "request", "call": 6, "recovery_policy": OLD_POLICY, "logical_call": 5, "attempt": 2,
         "retry_of": 5, "request_sha256": "same"},
        {"event": "response", "call": 6, "status": 200, "completed": True},
    ]
    write_rows(root / "requests.jsonl", gateway)
    result = audit_requests(root / "requests.jsonl", native_context=native)
    assert result["passed"] and result["recovery_valid"] and result["native_context_recovery"]["valid"]
    assert result["model_calls"] == 6 and result["total_failed_attempts"] == 2
    gateway[-1].update(status=500, completed=False)
    write_rows(root / "requests.jsonl", gateway)
    assert audit_requests(root / "requests.jsonl", native_context=native)["passed"] is False


def test_expert_binding_limits_recovery_to_mls_interval(evidence):
    root, native, rows, raw, gateway = evidence
    phase_root = root / "roles/MLS"
    phase_root.mkdir(parents=True)
    result = native["result"]
    (phase_root / "native_result.json").write_text(json.dumps(result))
    history = [{"role": "MLS", "status": "completed", "calls_before": 2, "calls_after": 4,
                "native_session_id": result["session_id"], "native_result_sha256": digest(result)}]
    (root / "role_history.json").write_text(json.dumps(history))
    bound = native_context_for_run(root, {"status": "completed"}, "expert-claude-code", {"probe"}, 1, 7)
    assert bound["root"] == phase_root and bound["calls_before"] == 2 and bound["calls_after"] == 4
    history[0]["native_result_sha256"] = "0" * 64
    (root / "role_history.json").write_text(json.dumps(history))
    with pytest.raises(ValueError, match="binding"):
        native_context_for_run(root, {}, "expert-claude-code", {"probe"}, 1, 7)


@pytest.mark.parametrize("mode", ["single", "overlap", "unbound"])
def test_followup_scopes_must_bind_each_failed_call_once(evidence, mode):
    root, native, rows, raw, gateway = evidence
    contexts = [native]
    if mode == "overlap": contexts.append(copy.deepcopy(native))
    if mode == "unbound": contexts = []
    audited = audit_requests(root / "requests.jsonl", native_context=contexts)
    assert audited["passed"] is (mode == "single")


@pytest.mark.parametrize("attack", [None, "no_start", "no_summary", "other_cycle", "summary_relabelled",
    "summary_after_boundary", "summary_also_late", "no_successful_summary", "no_continuation_request",
    "failure_after_start"])
def test_delayed_dsh_boundary_requires_same_cycle_and_actual_successful_requests(evidence, attack):
    root, native, _, _, gateway = evidence
    sid = native["result"]["session_id"]
    raw = [
        {"type": "compaction/start", "data": {"compactionId": "cycle"}},
        {"type": "compaction/summary", "data": {"compactionId": "cycle", "summary": []}},
        {"type": "compaction/end", "data": {"compactionId": "cycle"}},
        {"type": "tool/call", "data": {"callId": "t1", "name": "mcp__venusfactory__probe", "arguments": {}}},
        {"type": "tool/result", "data": {"message": {"content": [
            {"type": "tool-result", "toolCallId": "t1", "isError": False, "content": []}]}}},
        {"type": "turn/end", "data": {"reason": {"kind": "completed"}}},
    ]
    if attack == "other_cycle": raw[1]["data"]["compactionId"] = "unrelated"
    if attack == "summary_relabelled": raw[1]["type"] = "compaction/prune"
    rows, notifications = [], []
    for index, (event, calls, phase) in enumerate(zip(raw, [1, 2, 3, 3, 3, 4],
        ["started", "summary", "completed", None, None, "completed"])):
        event["seq"] = index
        row = native_compaction(event, phase=phase, native_type=event["type"],
                                session_id=sid, path="runtime_events.jsonl")
        row["gateway_calls"] = calls
        if index < 3:
            row["compaction_id"] = event["data"]["compactionId"]
        elif index == 3:
            row.update(kind="tool_call_start", tool_call_id="t1", tool_name="mcp__venusfactory__probe", args={})
        elif index == 4:
            row.update(kind="tool_result", tool_call_id="t1", is_error=False, output="observed")
        else:
            row["kind"] = "native_terminal"
        notification = {"sessionId": sid, "event": event}
        reference = native_compaction(notification, phase="observed", native_type="notification",
                                      session_id=sid, path="runtime_notifications.jsonl")
        row.update(native_notification_path="runtime_notifications.jsonl",
                   native_notification_sha256=reference["native_event_sha256"])
        rows.append(row)
        notifications.append(notification)
    if attack == "no_start": rows.pop(0)
    if attack == "no_summary": rows.pop(1)
    if attack == "summary_after_boundary": raw[1], raw[2] = raw[2], raw[1]
    if attack == "summary_also_late": rows[1]["gateway_calls"] = 3
    if attack == "no_successful_summary": gateway[4]["completed"] = False
    if attack == "no_continuation_request": gateway[:] = [row for row in gateway if row.get("call") != 3]
    if attack == "failure_after_start":
        gateway[1]["call"] = 3
        gateway[2].update(status=200, completed=True)
        gateway[6].update(status=400, completed=False, error_type=None)
    write_rows(root / "runtime_events.jsonl", raw)
    write_rows(root / "runtime_notifications.jsonl", notifications)
    write_rows(root / "events.jsonl", rows)
    write_rows(root / "requests.jsonl", gateway)
    result = audit_requests(root / "requests.jsonl", native_context=native)
    assert result["passed"] is (attack is None)
    assert result["native_context_recovery"]["valid"] is (attack is None)
    if attack is None:
        recovered = result["native_context_recovery"]["recovered_calls"][0]
        assert recovered["failed_call"] == recovered["compaction_start_call"] == 1
        assert recovered["boundary_call"] == recovered["summary_call"] == 2
        assert recovered["boundary_observed_call"] == recovered["successful_call"] == 3
        assert result["total_failed_attempts"] == 1 and result["model_calls"] == 4
