"""Worker event transport, tool-call identity mapping and execution audits."""

from __future__ import annotations

import hashlib
import json
import sys

KIMI_REPEAT_NOTICE = (
    "\n\n<system-reminder>\n"
    "You are repeating the exact same tool call with identical parameters. "
    "Please carefully analyze the previous result. If the task is not yet complete, "
    "try a different method or parameters instead of repeating the same call.\n"
    "</system-reminder>"
)


def emit(value):
    print(json.dumps(value, ensure_ascii=False), file=sys.__stdout__, flush=True)


def normalize_kimi_tool_result(output, *, is_error=False):
    """Separate one known display suffix while retaining every original byte.

    This is transport normalization, not scientific acceptance. Unknown suffixes,
    malformed JSON and error results remain intact for downstream validators.
    """
    result = {"output": str(output)}
    if is_error or not isinstance(output, str) or not output.endswith(KIMI_REPEAT_NOTICE):
        return result
    payload = output[: -len(KIMI_REPEAT_NOTICE)]
    try:
        decoded = json.loads(payload)
    except (ValueError, TypeError):
        return result
    if not isinstance(decoded, dict):
        return result
    return {
        "output": payload,
        "raw_output": output,
        "transport_annotations": [{"source": "kimi-code", "kind": "repeated_tool_call", "text": KIMI_REPEAT_NOTICE}],
    }


def valid_tool_result_annotations(row):
    """Optional annotations must reconstruct exactly from their retained raw text."""
    fields = {"output", "raw_output", "transport_annotations"}
    if not ({"raw_output", "transport_annotations"} & row.keys()):
        return True
    if not fields <= row.keys() or not isinstance(row["raw_output"], str):
        return False
    expected = normalize_kimi_tool_result(row["raw_output"], is_error=row.get("is_error") is not False)
    return set(expected) == fields and expected == {key: row[key] for key in fields}


def normalize_tool_arguments(arguments):
    """Observe malformed input without executing, rewriting or aborting the harness."""
    try:
        value = json.loads(arguments) if isinstance(arguments, str) else arguments
        if not isinstance(value, dict):
            return {"args": None, "arguments_error": {"code": "object_required"}}
        return {"args": value}
    except (ValueError, RecursionError) as exc:
        error = {"code": "invalid_json"}
        if isinstance(exc, json.JSONDecodeError):
            error.update(line=exc.lineno, column=exc.colno, offset=exc.pos)
        return {"args": None, "arguments_error": error}


def native_compaction(event, *, phase, native_type, session_id, path):
    """Audit metadata only; native summaries remain in the original event log."""
    encoded = json.dumps(event, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return {"kind": "context_compaction", "phase": phase, "native_type": native_type,
            "session_id": session_id, "native_event_path": path,
            "native_event_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest()}


def normalize_dsh(event, *, session_id=None):
    rows = _normalize_dsh(event, session_id=session_id)
    if session_id is not None:
        for row in rows:
            row["session_id"] = session_id
    return rows


def _normalize_dsh(event, *, session_id=None):
    data = event.get("data", {})
    kind = event.get("type")
    if kind in {"compaction/start", "compaction/summary", "compaction/end", "compaction/prune"}:
        phase = {"compaction/start": "started", "compaction/summary": "summary",
                 "compaction/prune": "pruned", "compaction/end": "completed"}[kind]
        if kind == "compaction/end" and data.get("error"):
            phase = "failed"
        row = native_compaction(event, phase=phase, native_type=kind,
                                session_id=session_id, path="runtime_events.jsonl")
        if data.get("compactionId") is not None:
            row["compaction_id"] = data["compactionId"]
        return [row]
    if event.get("type") == "tool/call":
        arguments = data["arguments"]
        return [
            {
                "kind": "tool_call_start",
                "tool_call_id": data["callId"],
                "tool_name": data["name"],
                **normalize_tool_arguments(arguments),
            }
        ]
    if event.get("type") == "tool/result":
        content = data["message"]["content"]
        return [
            {
                "kind": "tool_result",
                "tool_call_id": b["toolCallId"],
                "is_error": b.get("isError", False),
                "output": "\n".join(x.get("text", "") for x in b.get("content", []) if x.get("type") == "text"),
            }
            for b in content
            if b.get("type") == "tool-result"
        ]
    return []


def audit_trace(path, allowed_tools, limit):
    starts, ends, problems = {}, {}, []
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    allowed = {"mcp__venusfactory__" + name for name in allowed_tools}
    for row in rows:
        if row.get("kind") not in {"tool_call_start", "tool_result"}:
            continue
        if row.get("kind") == "tool_result" and not valid_tool_result_annotations(row):
            problems.append("invalid_transport_annotations")
        cid = row.get("tool_call_id")
        target = starts if row["kind"] == "tool_call_start" else ends
        if not cid or cid in target:
            problems.append("duplicate_or_missing_call_id")
        target[cid] = row
        if row["kind"] == "tool_call_start" and row.get("tool_name") not in allowed:
            problems.append("undeclared_tool")
        if row["kind"] == "tool_result" and cid not in starts:
            problems.append("result_before_call")
    for cid, start in starts.items():
        if start.get("arguments_error") and cid in ends and ends[cid].get("is_error") is not True:
            problems.append("invalid_arguments_not_rejected")
    if starts.keys() != ends.keys():
        problems.append("unpaired_tool_calls")
    if len(starts) > limit:
        problems.append("tool_budget_exceeded")
    if not path.exists():
        problems.append("missing_event_trace")
    return {
        "passed": not problems,
        "problems": problems,
        "tool_calls": len(starts),
        "tool_errors": sum(bool(x.get("is_error")) for x in ends.values()),
    }


def audit_requests(path, *, native_context=None):
    records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    requests = [x for x in records if x["event"] == "request"]
    responses = [x for x in records if x["event"] == "response"]
    ids = [x["call"] for x in requests]
    errors = [
        x["event"]
        for x in records
        if x["event"] in {"transport_error", "policy_rejected", "budget_rejected", "request_deadline_exhausted"}
    ]
    passed = (
        bool(ids)
        and len(ids) == len(set(ids))
        and sorted(ids) == sorted(x["call"] for x in responses)
        and not errors
        and all(x.get("status") == 200 and x.get("completed") is True for x in responses)
    )
    result = {
        "passed": bool(passed),
        "model_calls": len(requests),
        "errors": errors,
        "usage": [x["usage"] for x in responses if x.get("usage")],
    }
    from .request_recovery import audit_recovery

    from .native_context_recovery import audit_native_context_recovery

    native_recovery = audit_native_context_recovery(native_context, records, requests, responses)
    recovered = {row["failed_call"] for row in native_recovery["recovered_calls"]} if native_recovery and native_recovery["valid"] else set()
    # Keep the source log and physical accounting intact. Existing recovery rules
    # independently audit the remaining calls; neither mechanism can mask errors
    # or claim the same failed attempt twice.
    remaining_records = [row for row in records if row.get("call") not in recovered]
    remaining_requests = [row for row in requests if row["call"] not in recovered]
    remaining_responses = [row for row in responses if row["call"] not in recovered]
    recovery = audit_recovery(path, remaining_records, remaining_requests, remaining_responses)
    paired = bool(ids) and len(ids) == len(set(ids)) and sorted(ids) == sorted(x["call"] for x in responses)
    if recovery is not None:
        result.update(recovery, uninterrupted_passed=bool(passed and not recovery["failed_attempts"]))
        result["passed"] = bool(paired and not errors and recovery["recovery_valid"])
        if recovered:
            # Removing a proved recovered attempt from the old recovery view
            # must not reduce the original physical-budget accounting.
            lock = json.loads((path.parent / "profile.lock.json").read_text())
            limit = lock.get("profile", {}).get("budget", {}).get("model_calls")
            if type(limit) is not int or len(requests) > limit:
                result["passed"] = False
    elif recovered:
        result["passed"] = bool(paired and not errors and all(
            row.get("status") == 200 and row.get("completed") is True for row in remaining_responses))
    if native_recovery is not None:
        result["native_context_recovery"] = native_recovery
        result["uninterrupted_passed"] = False
        result["total_failed_attempts"] = sum(row.get("status") != 200 or row.get("completed") is not True for row in responses)
        result["passed"] = bool(result["passed"] and native_recovery["valid"])
    return result
