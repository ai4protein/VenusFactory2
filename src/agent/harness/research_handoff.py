"""Hand off a native Scientist turn at its audited checkpoint, not at research success."""

from __future__ import annotations

import copy
import json
import secrets
from contextlib import contextmanager
from pathlib import Path

from .contracts import digest
from .events import audit_trace


class NativeTurnFailure(RuntimeError):
    """Known native runtime termination; only the host may bind cleanup evidence."""

    def __init__(self):
        super().__init__("runtime_error")
        self.interruption = None


class ResearchHandoff:
    def __init__(self):
        self.checkpoint = None
        self.arguments = None
        self.relay_token = None
        self._request_scope = None

    @property
    def request_budget_exhausted(self):
        return self._request_scope is not None and self._request_scope.exhausted

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


class _NativeRequestScope:
    def __init__(self, ceiling):
        self.ceiling = ceiling
        self.active = True
        self.exhausted = False
        self.identity = secrets.token_hex(8)


@contextmanager
def native_phase(policy, *, relay=None, reserve_calls=3):
    if getattr(policy, "_research_handoff", None) is not None:
        raise ValueError("A native research phase is already active")
    boundary = ResearchHandoff()
    scope = None
    if relay is not None:
        bound_policy = relay.policy is policy or getattr(relay.policy, "policy", None) is policy
        if not bound_policy or type(reserve_calls) is not int or not 0 <= reserve_calls < policy.max_calls:
            raise ValueError("Invalid native phase request reserve")
        if getattr(relay.policy, "_research_handoff", None) is not None:
            raise ValueError("A native research phase is already active")
        ceiling = policy.max_calls - reserve_calls
        if relay.calls >= ceiling:
            raise ValueError("No model requests remain for the native phase")
        scope = _NativeRequestScope(ceiling)
        boundary._request_scope = scope
        boundary.relay_token = secrets.token_urlsafe(32)
        if not hasattr(relay, "_research_request_scopes"):
            relay._research_request_scopes = {}
        relay._research_request_scopes[boundary.relay_token] = scope
    policy._research_handoff = boundary
    try:
        yield boundary
    finally:
        # Keep a revoked scope for the relay lifetime: queued/late requests may
        # never become coordinator requests when a phase ends or is replaced.
        if scope is not None:
            scope.active = False
        policy._research_handoff = None


def request_scope(relay, authorization):
    """Capture a host-created credential's scope, never a model-supplied phase."""
    if not isinstance(authorization, str) or not authorization.startswith("Bearer "):
        raise PermissionError("Invalid relay credential")
    token = authorization[7:]
    if secrets.compare_digest(token.encode(), relay.token.encode()):
        return None
    scope = getattr(relay, "_research_request_scopes", {}).get(token)
    if scope is None:
        raise PermissionError("Invalid relay credential")
    return scope


def _request_error(relay, event, code, message, scope=None):
    from aiohttp import web

    relay.record({"event": event, "code": code,
                  **({"phase": "native", "phase_id": scope.identity} if scope is not None else {})})
    return web.json_response({"error": {"message": message, "type": "budget_exhausted"
                                       if event == "budget_rejected" else "invalid_request_error",
                                       "code": code}}, status=400)


def closed_phase_response(relay, scope):
    if scope is not None and not scope.active:
        return _request_error(relay, "phase_rejected", "research_phase_closed",
                              "This native research phase has ended.", scope)
    return None


def admit_model_request(relay, scope):
    """Atomically admit one physical dispatch on the relay's event loop.

    There must be no await between this function and recording the admitted call.
    A retry must pass this gate again; no limit is inferred from harness steps.
    """
    rejected = closed_phase_response(relay, scope)
    if rejected is not None:
        return rejected
    handoff = closed_response(relay.policy, relay.record)
    if handoff is not None:
        return handoff
    if scope is not None and relay.calls >= scope.ceiling:
        scope.exhausted = True
        return _request_error(relay, "budget_rejected", "research_phase_budget_exhausted",
                              "Native model request budget exhausted; coordinator requests are reserved.", scope)
    if relay.calls >= relay.policy.max_calls:
        return _request_error(relay, "budget_rejected", "model_call_budget_exhausted",
                              "model call budget exhausted")
    relay.calls += 1
    return None


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
