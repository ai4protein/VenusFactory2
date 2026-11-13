"""Canonical SDK overflow errors from host-recorded, sanitized rejection signals.

Never classify arbitrary prompts, exception text or upstream bodies here. This
module does not retry requests, compact context or claim recovery succeeded.
"""
from collections.abc import Mapping


def is_context_limit(evidence, status):
    """Accept only a matching HTTP rejection with a classified context signal."""
    return (
        status in (400, 413)
        and isinstance(evidence, Mapping)
        and evidence.get("event") == "upstream_rejection"
        and evidence.get("status") == status
        and isinstance(evidence.get("reported_signals"), list)
        and "context_limit" in evidence["reported_signals"]
    )


def context_limit_error(evidence, status, *, protocol="openai"):
    """Return a fixed public error body, or None to retain existing handling."""
    if not is_context_limit(evidence, status):
        return None
    if protocol == "messages":
        return {"type": "error", "error": {
            "type": "invalid_request_error", "message": "prompt is too long",
        }}
    if protocol != "openai":
        raise ValueError("Unsupported public error protocol")
    return {"error": {
        "type": "invalid_request_error", "code": "context_length_exceeded",
        "message": "Your input exceeds the context window of this model.",
        "param": None,
    }}
