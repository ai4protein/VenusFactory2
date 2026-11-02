"""Lossless tool-call identity mapping and backend-independent execution audits."""

from __future__ import annotations

import json


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
