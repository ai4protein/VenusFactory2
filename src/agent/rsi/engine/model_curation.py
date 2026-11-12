"""Host-only reversible discovery curation; immutable publications remain intact."""

from __future__ import annotations

import json
import time
from copy import deepcopy

from agent.harness.contracts import digest

from .storage import atomic_json, file_lock, safe_child

SCHEMA = "model_curation.v1"


def _identity(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Invalid model curation identity")
    return value


def current(library):
    path = safe_child(library.root, "curation/current.json")
    if not path.exists():
        return {"schema": SCHEMA, "revision": None, "archived": {}}
    state = json.loads(path.read_text())
    if set(state) != {"schema", "revision", "archived"} or state["schema"] != SCHEMA:
        raise ValueError("Invalid model curation state")
    revision = _identity(state["revision"])
    event = json.loads(safe_child(library.root, f"curation/events/{revision}.json").read_text())
    if digest(event) != revision or event.get("state_sha256") != digest(state["archived"]):
        raise ValueError("Model curation state changed")
    if not isinstance(state["archived"], dict):
        raise ValueError("Invalid model archive")
    for entry, record in state["archived"].items():
        _identity(entry)
        if set(record) != {"replacement", "reason", "plan_sha256"}:
            raise ValueError("Invalid archive record")
        _identity(record["replacement"])
        _identity(record["plan_sha256"])
        if entry == record["replacement"] or record["replacement"] in state["archived"]:
            raise ValueError("Archive replacements must remain active")
    return state


def effective_archived(library):
    """A replaced/removed keeper must not silently hide the remaining model."""
    result = {}
    for entry, record in current(library)["archived"].items():
        replacement = library._object("entries", record["replacement"])
        path = safe_child(library.root, "index/" + digest(replacement["key"]) + ".json")
        if path.is_file() and json.loads(path.read_text()) == {"entry_id": record["replacement"]}:
            result[entry] = record
    return result


def _save(library, state, archived, event):
    event = {
        **event,
        "schema": SCHEMA,
        "previous_revision": state["revision"],
        "state_sha256": digest(archived),
        "at": time.time(),
    }
    revision = digest(event)
    path = safe_child(library.root, f"curation/events/{revision}.json")
    atomic_json(path, event, exclusive=True)
    result = {"schema": SCHEMA, "revision": revision, "archived": archived}
    atomic_json(safe_child(library.root, "curation/current.json"), result)
    return result


def apply(library, plan):
    """Recompute a metadata/evidence-bound plan before applying it atomically."""
    from .model_retention import plan_retention

    with file_lock(safe_child(library.root, ".library.lock")):
        state = current(library)
        if state["revision"]:
            event = json.loads(safe_child(library.root, f"curation/events/{state['revision']}.json").read_text())
            if event.get("plan_sha256") == digest(plan) and event.get("operation") == "archive":
                return state
        if state["revision"] != plan.get("base_revision"):
            raise ValueError("Model curation changed; prepare a fresh plan")
        checked = plan_retention(library, plan["policy"])
        if checked != plan:
            raise ValueError("Model evidence or retention plan changed")
        if not plan["archive"]:
            return state
        archived = deepcopy(state["archived"])
        for item in plan["archive"]:
            archived[item["entry_id"]] = {
                "replacement": item["replacement"],
                "reason": item["reason"],
                "plan_sha256": digest(plan),
            }
        # Existing archived models may point at a newly dominated keeper.
        for record in archived.values():
            seen = set()
            while record["replacement"] in archived:
                identity = record["replacement"]
                if identity in seen:
                    raise ValueError("Cyclic archive replacements")
                seen.add(identity)
                record["replacement"] = archived[identity]["replacement"]
        return _save(
            library,
            state,
            archived,
            {
                "operation": "archive",
                "plan_sha256": digest(plan),
                "entries": plan["archive"],
                "coverage": plan["coverage"],
                "historical_predictions_changed": False,
            },
        )


def restore(library, entry_id, *, reason):
    _identity(entry_id)
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 1000:
        raise ValueError("A bounded restoration reason is required")
    with file_lock(safe_child(library.root, ".library.lock")):
        state = current(library)
        if entry_id not in state["archived"]:
            return state
        library.inspect_publication(entry_id)
        archived = deepcopy(state["archived"])
        archived.pop(entry_id)
        return _save(library, state, archived, {"operation": "restore", "entry_id": entry_id, "reason": reason})
