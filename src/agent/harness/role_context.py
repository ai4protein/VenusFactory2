"""Deterministic, explicit handoffs between independent Scientist role sessions.

Callers supply a host-authored task projection, never a conversation transcript.
This module neither summarizes science nor promotes role opinions to evidence.
"""
from __future__ import annotations

import json

POLICY = "scientist_role_context.v2"
_FORBIDDEN_TASK_KEYS = frozenset({
    "messages", "memory", "history", "previous_episodes", "conversation",
    "PI", "CB", "CBL", "MLS", "SC", "role_outputs",
})
_REVISION_KEYS = frozenset({
    "facts", "unknowns", "constraints", "acceptance_criteria", "evidence_refs", "issues",
})


class RoleContextError(ValueError):
    """An explicit projection is missing, malformed or exceeds its declared bound."""


def json_snapshot(value, *, max_bytes=None):
    """Copy finite JSON without aliases or an implicit context-size admission limit.

    A caller may supply an explicit transport bound. Role isolation itself must
    not preempt native context management or truncate scientific requirements.
    """
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 0):
        raise RoleContextError("An explicit byte limit must be a nonnegative integer")
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        if max_bytes is not None and len(encoded.encode("utf-8")) > max_bytes:
            raise RoleContextError(
                "Role context exceeds its byte limit; preserve the original audit artifact "
                "and supply explicit evidence references or reduce scope, never truncate requirements"
            )
        return json.loads(encoded)
    except (TypeError, ValueError, RecursionError) as exc:
        if isinstance(exc, RoleContextError):
            raise
        raise RoleContextError("Role context must be finite JSON data") from exc


def canonical_role(role):
    role = "CB" if role == "CBL" else role
    if role not in {"PI", "CB", "MLS", "SC"}:
        raise RoleContextError("Unknown Scientist role")
    return role


def planning_handoff(role, response):
    """Validate a typed proposal; the full response stays in the audit log."""
    role = canonical_role(role)
    primary = {"PI": "direction", "CB": "steps"}.get(role)
    if primary is None:
        raise RoleContextError("Only planning roles produce planning handoffs")
    def unique_fields(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise RoleContextError("Repeated planning field could discard requirements")
            value[key] = item
        return value

    try:
        value = json.loads(response, object_pairs_hook=unique_fields) if isinstance(response, str) else json_snapshot(response)
    except ValueError as exc:
        raise RoleContextError("Planning role must return a structured JSON handoff") from exc
    if not isinstance(value, dict) or set(value) != {primary, "constraints", "unknowns"}:
        raise RoleContextError("Planning handoff requires " + primary + ", constraints and unknowns")
    for key, items in value.items():
        if not isinstance(items, list) or any(not isinstance(x, str) or not x.strip() for x in items):
            raise RoleContextError("Planning handoff fields must be lists of nonempty strings")
        if key == primary and not items:
            raise RoleContextError("Planning handoff cannot omit its primary proposal")
    return json_snapshot(value)


def build_role_context(role, task, *, handoffs=None, evidence=None, revision=None):
    """Build a new role input from explicit projections, never prior mutable input.

    task is a complete host projection of the user's goal and preset obligations.
    A caller must select objective evidence fields before passing evidence. SC never
    receives planning role outputs or the executor's narrative, even if handoffs are
    supplied. Revision issues are structured unresolved obligations, not prior votes.
    """
    role = canonical_role(role)
    if not isinstance(task, dict) or _FORBIDDEN_TASK_KEYS.intersection(task):
        raise RoleContextError("Task projection contains conversation or prior role context")
    context = json_snapshot(task)
    if any(key in context for key in ("role_context", "handoffs", "execution_evidence", "revision")):
        raise RoleContextError("Task projection contains reserved handoff fields")
    context["role_context"] = {"policy": POLICY, "role": role, "fresh_session": True}
    if role in {"CB", "MLS"} and handoffs:
        selected = {}
        for name in (("PI",) if role == "CB" else ("PI", "CB")):
            value = handoffs.get(name, handoffs.get("CBL") if name == "CB" else None)
            if value is not None:
                if not isinstance(value, dict):
                    raise RoleContextError("Handoffs must be explicit structured proposals, not transcripts")
                selected[name] = json_snapshot(value)
        if selected:
            context["handoffs"] = selected
    if evidence is not None and role in {"MLS", "SC"}:
        context["execution_evidence"] = json_snapshot(evidence)
    if revision is not None:
        if not isinstance(revision, dict) or set(revision) - _REVISION_KEYS:
            raise RoleContextError("Revision context requires objective structured state")
        context["revision"] = json_snapshot(revision)
    return json_snapshot(context)
