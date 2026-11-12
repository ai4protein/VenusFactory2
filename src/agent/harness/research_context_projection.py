"""Pure native-execution view of already-public research context.

Never pass a transport request here: it includes relay credentials. The caller
must apply session_boundary.public_value first and keep its original context for
follow-up planning, scientific acceptance and host validation. This module does
not read files, write references, reconstruct evidence or change research state.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass

SCHEMA = "native-research-context.v1"
MAX_FRAGMENT_BYTES = 20_000


@dataclass(frozen=True)
class ResearchContextProjection:
    context: dict
    reference_payloads: dict[str, str]


def _encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _fragments(raw, limit):
    start = 0
    while start < len(raw):
        end = min(start + limit, len(raw))
        while end < len(raw) and raw[end] & 0xC0 == 0x80:
            end -= 1
        yield raw[start:end]
        start = end


def project_research_context(
    context: Mapping, *, current_step_id: str | None = None,
    reference_prefix: str = "context/research", max_fragment_bytes: int = 16_000,
) -> ResearchContextProjection:
    """Retain execution contracts; return optional lossless reference payloads.

    Payload keys are safe workspace-relative paths. Parts contain consecutive
    UTF-8 text from one canonical JSON document; concatenate in their listed
    order before JSON decoding. Their hashes bind information copies, never
    replace host-owned plan/evaluation hashes or certify scientific validity.

    Without an explicit unique current step, every approach remains inline.
    Unknown fields, scientific evidence and unresolved feedback are preserved.
    If references cost more context than they save, return an unchanged copy.
    """
    if not isinstance(context, Mapping) or not isinstance(context.get("objective"), str):
        raise ValueError("Expected public research context with an objective")
    if set(context) & {"token", "api_key", "authorization", "base_url", "runtimes", "context_references"}:
        raise ValueError("Use original public research context, never a transport request or an existing projection")
    if (type(max_fragment_bytes) is not int or not 256 <= max_fragment_bytes <= MAX_FRAGMENT_BYTES):
        raise ValueError("Reference fragments must be 256..20000 UTF-8 bytes")
    if (not isinstance(reference_prefix, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*(?:/[A-Za-z0-9][A-Za-z0-9_.-]*)*", reference_prefix) is None):
        raise ValueError("Reference prefix must be a safe workspace-relative path")
    original = copy.deepcopy(dict(context))
    original_bytes = _encode(original)  # Reject non-JSON and nonfinite input; never stringify objects.
    projected = copy.deepcopy(original)
    documents, payloads, duplicate_fields = [], {}, {}

    def reference(name, value, source_paths, *, inline_scope):
        raw = _encode(value)
        identity = _sha(raw)
        parts = []
        for index, fragment in enumerate(_fragments(raw, max_fragment_bytes), 1):
            path = f"{reference_prefix}/{name}-{identity}/{index:04d}.txt"
            payloads[path] = fragment.decode("utf-8")
            parts.append({"path": path, "sha256": _sha(fragment), "bytes": len(fragment)})
        document = {"name": name, "source_paths": source_paths, "sha256": identity,
                    "bytes": len(raw), "encoding": "concatenated-utf8-json", "parts": parts,
                    "inline_scope": inline_scope}
        documents.append(document)
        return name

    plan = projected.get("plan")
    steps = plan.get("steps") if isinstance(plan, dict) else None
    if current_step_id is not None:
        if (not isinstance(current_step_id, str) or not isinstance(steps, list)
                or sum(isinstance(step, dict) and step.get("id") == current_step_id for step in steps) != 1):
            raise ValueError("Current research step must match exactly one plan step")

    progress = projected.get("progress")
    if isinstance(plan, dict):
        paths = ["/plan"]
        duplicate = isinstance(progress, dict) and progress.get("plan") == original["plan"]
        if duplicate:
            # One complete set of current requirements remains at the top level.
            progress.pop("plan")
            duplicate_fields["/progress/plan"] = "/plan"
            paths.append("/progress/plan")
        deferred = []
        if current_step_id is not None:
            for index, step in enumerate(steps):
                if isinstance(step, dict) and step.get("id") != current_step_id and "approach" in step:
                    step.pop("approach")
                    deferred.append(f"/plan/steps/{index}/approach")
        reference("plan", original["plan"], paths,
                  inline_scope={"all_step_requirements_and_evidence": True,
                                "deferred_approaches": deferred})

    # These are explicit historical/inventory fields, not a generic pruning rule.
    # In particular revision_history, user_updates, training_progress, SC and
    # execution feedback (including superseded failures) are never shortened.
    for container, original_container, key, keep, direction, name, path in (
        (projected, original, "previous_episodes", 2, "last", "previous-episodes", "/previous_episodes"),
        (progress, original.get("progress"), "method_revisions", 2, "last", "method-revisions", "/progress/method_revisions"),
        (projected, original, "workspace_files", 20, "first", "workspace-files", "/workspace_files"),
    ):
        if not isinstance(container, dict) or not isinstance(original_container, dict):
            continue
        items = original_container.get(key)
        if not isinstance(items, list) or len(items) <= keep:
            continue
        known_fields = {"previous_episodes": {"summary", "status"},
                        "method_revisions": {"episode", "step_id", "previous", "approach", "acceptance"}}
        if key in known_fields and any(not isinstance(item, dict) or set(item) - known_fields[key] for item in items):
            continue  # A future history format may carry new unresolved requirements.
        if key == "workspace_files" and any(not isinstance(item, str) for item in items):
            continue
        reference(name, items, [path], inline_scope={"retained": direction, "count": keep, "total": len(items)})
        container[key] = copy.deepcopy(items[-keep:] if direction == "last" else items[:keep])

    if documents:
        projected["context_references"] = {
            "schema": SCHEMA, "documents": documents, "duplicate_fields": duplicate_fields,
            "instruction": "Read listed parts in order for complete details. These copies do not replace host-owned research evidence.",
        }
    if len(_encode(projected)) >= len(original_bytes):
        return ResearchContextProjection(original, {})
    return ResearchContextProjection(projected, payloads)
