"""Lossless handling of redundant empty plan metadata, before asking a model to repair."""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

from utils.scoped_files import write_bytes


def remove_empty_step_inputs(raw, protocol):
    """Remove only empty step needs_input when the root already owns that field."""
    from .scientist_role_output import _finite, _unique, specification, validate_output

    try:
        original = json.loads(raw, object_pairs_hook=_unique, parse_constant=_finite)
    except (ValueError, TypeError):
        return None
    if (not isinstance(original, dict) or not isinstance(original.get("needs_input"), list)
            or not isinstance(original.get("steps"), list)):
        return None
    candidate = {**original, "steps": []}
    removed = []
    for index, step in enumerate(original["steps"]):
        if not isinstance(step, dict):
            return None
        current = dict(step)
        if "needs_input" in step:
            if step["needs_input"] != []:
                return None
            del current["needs_input"]
            removed.append("/steps/" + str(index) + "/needs_input")
        candidate["steps"].append(current)
    if not removed:
        return None
    text = json.dumps(candidate, ensure_ascii=False)
    try:
        validated = validate_output(text, specification("CB", protocol))
    except ValueError:
        return None
    # Do not bundle other normalizations or scientific edits into this operation.
    if json.loads(validated) != candidate:
        return None
    return {"text": text, "removed_empty_fields": removed}


def record_adjustment(folder, original_receipt, adjustment):
    """Keep the exact rejected response and its distinct canonical representation."""
    text = adjustment["text"]
    value = {"schema": "scientist.plan-format-adjustment.v1",
             "original_response": original_receipt,
             "rule": "remove_empty_step_needs_input_with_existing_root_field",
             "removed_empty_fields": adjustment["removed_empty_fields"],
             "canonical_text": text, "canonical_sha256": hashlib.sha256(text.encode()).hexdigest(),
             "scientific_fields_changed": False}
    control = Path(os.path.abspath(folder))
    relative = control.relative_to(control.anchor).as_posix() + "/role_normalizations/" + uuid.uuid4().hex + ".json"
    write_bytes(control.anchor, relative, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())
