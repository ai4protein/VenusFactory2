"""Hand off a native Scientist turn at its audited checkpoint, not at research success."""

from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from pathlib import Path

from .contracts import digest
from .events import audit_trace


class ResearchHandoff:
    def __init__(self):
        self.checkpoint = None
        self.arguments = None

    @property
    def closed(self):
        return self.checkpoint is not None

    def save(self, arguments, result):
        if self.closed:
            raise ValueError("Research checkpoint handoff already saved")
        if isinstance(result, str):
            result = json.loads(result)
        if (
            not isinstance(result, dict)
            or result.get("checkpointed") is not True
            or not isinstance(result.get("artifacts"), list)
            or not isinstance(arguments.get("summary"), str)
            or not arguments["summary"].strip()
        ):
            raise ValueError("A successful research checkpoint receipt is required")
        self.arguments = copy.deepcopy(arguments)
        self.checkpoint = copy.deepcopy(result)

    @property
    def fingerprint(self):
        return digest({"arguments": self.arguments, "result": self.checkpoint}) if self.closed else None

    def audit(self, trace, allowed_tools, limit):
        trace = Path(trace)
        if not self.closed or not trace.is_file():
            return {"passed": False, "problems": ["checkpoint_trace_pending"]}
        try:
            audit = audit_trace(trace, allowed_tools, limit)
            if not audit["passed"]:
                return audit
            rows = [json.loads(line) for line in trace.read_text().splitlines()]
        except (ValueError, OSError):
            return {"passed": False, "problems": ["checkpoint_trace_pending"]}
        starts = {row.get("tool_call_id"): row for row in rows if row.get("kind") == "tool_call_start"}
        matched = []
        for row in rows:
            if row.get("kind") != "tool_result" or row.get("is_error"):
                continue
            start = starts.get(row.get("tool_call_id"), {})
            name = str(start.get("tool_name", "")).rsplit("__", 1)[-1]
            arguments = start.get("args")
            if name == "call_tool" and isinstance(arguments, dict):
                name = str(arguments.get("name", "")).rsplit("__", 1)[-1]
                arguments = arguments.get("arguments")
            if name != "submit_research_checkpoint" or arguments != self.arguments:
                continue
            output = row.get("output")
            if isinstance(output, str):
                try:
                    output, _ = json.JSONDecoder().raw_decode(output.lstrip())
                except ValueError:
                    continue
            if output == self.checkpoint:
                matched.append(row["tool_call_id"])
        if len(matched) != 1:
            return {**audit, "passed": False, "problems": ["checkpoint_receipt_not_matched"]}
        return {**audit, "checkpoint_call_id": matched[0], "checkpoint_sha256": self.fingerprint}

    def event(self, audit, *, executions, lookups, checkpoints):
        if not audit.get("passed") or audit.get("checkpoint_sha256") != self.fingerprint:
            raise ValueError("Research handoff requires a verified native checkpoint")
        return {
            "kind": "completed",
            "completion_source": "research_checkpoint_handoff",
            "text": self.arguments["summary"],
            "tool_calls": executions,
            "discovery_calls": lookups,
            "checkpoint_calls": checkpoints,
            "event_audit": audit,
            "checkpoint_sha256": self.fingerprint,
            "research_complete": False,
        }


@contextmanager
def native_phase(policy):
    if getattr(policy, "_research_handoff", None) is not None:
        raise ValueError("A native research phase is already active")
    boundary = ResearchHandoff()
    policy._research_handoff = boundary
    try:
        yield boundary
    finally:
        policy._research_handoff = None


def closed_response(policy, record):
    """Reject locally before charging or contacting a provider; never fake an LLM reply."""
    boundary = getattr(policy, "_research_handoff", None)
    if boundary is None or not boundary.closed:
        return None
    from aiohttp import web

    record({"event": "research_checkpoint_handoff", "checkpoint_sha256": boundary.fingerprint})
    return web.json_response(
        {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "code": "research_checkpoint_handoff",
                "message": "The research checkpoint is saved. This implementation turn has ended; the coordinator will review it.",
            },
        },
        status=400,
    )
