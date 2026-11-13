"""Server-owned free reviewer selection; credentials never enter saved bindings."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re

SCHEMA = "research-reviewer.v1"


class ReviewerBindingUnavailable(ValueError):
    """Local admission failed before HTTP; content repair cannot fix configuration."""
    MESSAGES = {
        "selection_unavailable": "The selected research reviewer configuration or credential is unavailable.",
        "catalog_unavailable": "The configured free reviewer catalog is unavailable.",
        "free_route_unavailable": "Research reviewer must be an enabled free model.",
        "route_changed": "Research reviewer route changed; the saved selection cannot be replaced automatically.",
    }

    def __init__(self, code):
        message = self.MESSAGES[code]
        self.diagnostic = {"source": "reviewer_configuration", "code": code, "message": message,
                           "request_dispatched": False, "resume": "after_configuration_recovery"}
        super().__init__(message)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def binding_digest(binding):
    if (not isinstance(binding, dict) or set(binding) != {"schema", "route", "sha256", "catalog_sha256"}
            or binding.get("schema") != SCHEMA or not isinstance(binding.get("route"), dict)
            or binding.get("sha256") != _digest(binding["route"])):
        raise ValueError("Research reviewer binding is invalid")
    return binding["sha256"]


def resolve_reviewer(identity, expected=None):
    """Return the single effective option/key used by the subsequent HTTP request."""
    from agent.harness.free_catalog import free_routes

    from .research_report import model_option

    if not isinstance(identity, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", identity):
        raise ValueError("Choose a configured free reviewer")
    try:
        option, key = model_option(identity)
    except (ValueError, OSError, KeyError, TypeError):
        raise ReviewerBindingUnavailable("selection_unavailable") from None
    try:
        routes, catalog_sha = free_routes()
    except (ValueError, OSError, KeyError, TypeError):
        raise ReviewerBindingUnavailable("catalog_unavailable") from None
    b = option.binding
    if not any((row["model"], row["base_url"], row["protocol"]) ==
               (b.model, b.base_url.rstrip("/"), b.protocol) for row in routes):
        raise ReviewerBindingUnavailable("free_route_unavailable")
    route = {"option_id": option.id, "provider": b.provider, "model": b.model,
             "base_url": b.base_url.rstrip("/"), "protocol": b.protocol,
             "inference_controls": copy.deepcopy(option.inference_controls or {}),
             "max_output_tokens": option.max_output_tokens}
    binding = {"schema": SCHEMA, "route": route, "sha256": _digest(route), "catalog_sha256": catalog_sha}
    if expected is not None and binding_digest(expected) != binding["sha256"]:
        raise ReviewerBindingUnavailable("route_changed")
    return option, key, binding


def bind_payload(payload, *, new=False):
    """Resolve defaults only on creation; old requests retain their original writer ID."""
    value = dict(payload)
    saved = value.get("reviewer_binding")
    identity = value.get("reviewer_model_id")
    if saved is not None:
        binding_digest(saved)
        if identity != saved["route"].get("option_id"):
            raise ValueError("Research reviewer identity does not match its saved binding")
    else:
        identity = identity or (os.getenv("VENUS_REPORT_REVIEWER_MODEL_ID", "").strip() if new else "") or value["model_id"]
    _, _, current = resolve_reviewer(identity, saved)
    value.update(reviewer_model_id=identity, reviewer_binding=copy.deepcopy(saved or current))
    return value


def align_review_state(state, binding):
    """Re-audit changed/legacy reviewer generations without repeating data or predictions."""
    identity = binding_digest(binding)
    review = state.get("source_review")
    old = review.get("reviewer_binding_sha256") if isinstance(review, dict) else None
    changed = old != identity
    if changed:
        if review:
            state.setdefault("source_review_history", []).append(copy.deepcopy(review))
        state.pop("source_review", None)
        if state.get("review_source_review"):
            state.setdefault("review_source_review_history", []).append(copy.deepcopy(state["review_source_review"]))
        state.pop("review_source_review", None)
        if state.get("review"):
            state["review_status"] = "reviewing"
        for chapter in state.get("chapters", []):
            chapter.pop("source_review", None)
            if chapter.get("status") == "completed":
                chapter.update(status="reviewing", draft_status="completed")
    state["reviewer_binding"] = copy.deepcopy(binding)
    state["reviewer_model_id"] = binding["route"]["option_id"]
    for chapter in state.get("chapters", []):
        chapter["reviewer_binding_sha256"] = identity
    return changed
