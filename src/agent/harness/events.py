"""Lossless tool-call identity mapping and backend-independent execution audits."""

from __future__ import annotations

import json

KIMI_REPEAT_NOTICE = (
    "\n\n<system-reminder>\n"
    "You are repeating the exact same tool call with identical parameters. "
    "Please carefully analyze the previous result. If the task is not yet complete, "
    "try a different method or parameters instead of repeating the same call.\n"
    "</system-reminder>"
)


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


def normalize_dsh(event):
    data = event.get("data", {})
    if event.get("type") == "tool/call":
        arguments = data["arguments"]
        return [
            {
                "kind": "tool_call_start",
                "tool_call_id": data["callId"],
                "tool_name": data["name"],
                "args": json.loads(arguments) if isinstance(arguments, str) else arguments,
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


def audit_requests(path):
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

    recovery = audit_recovery(path, records, requests, responses)
    if recovery is not None:
        paired = bool(ids) and len(ids) == len(set(ids)) and sorted(ids) == sorted(x["call"] for x in responses)
        result.update(recovery, uninterrupted_passed=bool(passed and not recovery["failed_attempts"]))
        result["passed"] = bool(paired and not errors and recovery["recovery_valid"])
    return result
