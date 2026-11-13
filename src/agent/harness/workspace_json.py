"""Bounded structural navigation of JSON artifacts through anchored session reads."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import deque
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from utils.scoped_files import open_file

JSON_INPUT_LIMIT = 8 * 1024**2
OVERVIEW_ENTRIES = 48
OVERVIEW_DEPTH = 4
PathPart = Annotated[str, Field(strict=True, max_length=256)] | Annotated[int, Field(strict=True, ge=0)]
RecordValue = Annotated[str, Field(strict=True, max_length=512)]


class JsonSelection(BaseModel):
    """Select a JSON field or exact string-valued records; paths are data keys, not filenames."""

    model_config = ConfigDict(extra="forbid", strict=True, revalidate_instances="always")
    path: list[PathPart] = Field(
        default_factory=list,
        max_length=16,
        description="Object keys and array indexes, e.g. ['result', 'predictions']. [] selects the root.",
    )
    match_field: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        description="For an array of objects, a field to match exactly, e.g. id.",
    )
    match_values: list[RecordValue] | None = Field(
        default=None,
        min_length=1,
        max_length=20,
        description="Exact string values for match_field. No substring or numeric coercion.",
    )
    offset: int = Field(default=0, ge=0, description="Offset within the selected or matched array.")
    limit: int = Field(default=20, ge=1, le=100, description="Maximum array records returned in this page.")

    @model_validator(mode="after")
    def paired_match(self):
        if (self.match_field is None) != (self.match_values is None):
            raise ValueError("Provide match_field and match_values together")
        return self


def _encoded(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _bounded_nesting(raw):
    # Bound parser/serializer recursion independently of other libraries' settings.
    depth = 0
    quoted = escaped = False
    for byte in raw:
        if quoted:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                quoted = False
        elif byte == 34:
            quoted = True
        elif byte in (91, 123):
            depth += 1
            if depth > 64:
                raise ValueError("JSON nesting exceeds 64 containers")
        elif byte in (93, 125):
            depth -= 1


def _finite_float(text):
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("Non-finite JSON number")
    return value


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON key")
        value[key] = item
    return value


def _invalid_constant(_):
    raise ValueError("Non-finite JSON constant")


def _kind(value):
    if value is None:
        return "null"
    return {dict: "object", list: "array", str: "string", bool: "boolean", int: "integer", float: "number"}[type(value)]


def _overview(value, maximum):
    """Index named fields without showing partial input/output rows as results."""
    pending = deque([([], value)])
    entries = []
    omitted = False
    while pending and len(entries) < OVERVIEW_ENTRIES:
        path, item = pending.popleft()
        entry = {"path": path, "type": _kind(item)}
        if isinstance(item, (dict, list)):
            entry["length"] = len(item)
        elif isinstance(item, str) and len(item) > 256:
            entry.update(value_excerpt=item[:256], value_truncated=True)
        else:
            entry["value"] = item
        if len(_encoded(entries + [entry])) > maximum:
            omitted = True
            break
        entries.append(entry)
        if isinstance(item, dict) and len(path) < OVERVIEW_DEPTH:
            # Only queue as many entries as can ever be returned.
            room = OVERVIEW_ENTRIES - len(entries) - len(pending)
            for index, (key, child) in enumerate(item.items()):
                if index >= room:
                    omitted = True
                    break
                pending.append((path + [key], child))
        elif isinstance(item, dict) and item:
            omitted = True
    return {
        "structure": entries,
        "structure_incomplete": bool(pending or omitted),
        "truncated": True,
        "next_step": "This is a field index, not complete records. Use select.path to read a named value. "
        "For an array, select.match_field and select.match_values retrieve exact string IDs; "
        "offset and limit page the matches. Use isolated Python for full-data analysis.",
    }


def _selected(value, selection, maximum):
    for part in selection.path:
        if isinstance(value, dict) and type(part) is str and part in value:
            value = value[part]
        elif isinstance(value, list) and type(part) is int and part < len(value):
            value = value[part]
        else:
            raise ValueError("JSON selection path does not exist or has the wrong key/index type")
    result = {"json_path": selection.path, "value_type": _kind(value)}
    if not isinstance(value, list):
        if selection.match_field is not None or selection.offset:
            raise ValueError("Record matching and offset require a JSON array")
        if len(_encoded(value)) > maximum:
            raise ValueError("Selected JSON value exceeds max_bytes; choose a narrower path or use isolated Python")
        return {**result, "value": value, "truncated": False}

    wanted = set(selection.match_values) if selection.match_values is not None else None
    found = set()
    page, indices = [], []
    matched = 0
    page_full = False
    for index, item in enumerate(value):
        if wanted is not None:
            if not isinstance(item, dict) or type(item.get(selection.match_field)) is not str:
                continue
            if item[selection.match_field] not in wanted:
                continue
            found.add(item[selection.match_field])
        ordinal = matched
        matched += 1
        if ordinal < selection.offset or page_full or len(page) >= selection.limit:
            continue
        if len(_encoded(page + [item])) > maximum:
            if not page:
                raise ValueError(
                    "Selected JSON record exceeds max_bytes; choose a narrower path or use isolated Python"
                )
            page_full = True
            continue
        page.append(item)
        indices.append(index)
    if selection.offset > matched:
        raise ValueError("Selection offset is beyond the matched array")
    if wanted is not None:
        result["missing_values"] = sorted(wanted - found)
    end = selection.offset + len(page)
    return {
        **result,
        "value": page,
        "source_items": len(value),
        "matched_items": matched,
        "indices": indices,
        "offset": selection.offset,
        "returned_items": len(page),
        "next_offset": end if end < matched else None,
        "truncated": end < matched,
    }


def json_preview(workspace, path, *, max_bytes, select=None):
    """Return None for unchanged small-text previews; otherwise read one bounded snapshot."""
    selection = JsonSelection.model_validate(select) if select is not None else None
    with open_file(workspace, path) as stream:
        size = os.fstat(stream.fileno()).st_size
        if selection is None and size <= max_bytes:
            return None
        if size > JSON_INPUT_LIMIT:
            raise ValueError("JSON exceeds the 8 MiB structured-read limit; use isolated Python or view='text'")
        raw = stream.read(JSON_INPUT_LIMIT + 1)
    if len(raw) > JSON_INPUT_LIMIT:
        raise ValueError("JSON exceeds the 8 MiB structured-read limit; use isolated Python or view='text'")
    try:
        _bounded_nesting(raw)
        value = json.loads(
            raw.decode("utf-8-sig"),
            parse_float=_finite_float,
            parse_constant=_invalid_constant,
            object_pairs_hook=_unique_object,
        )
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ValueError(
            "JSON must be valid UTF-8 with unique keys, finite numbers and at most 64 nested containers; inspect raw text if needed"
        ) from exc
    result = {
        "path": path,
        "preview_kind": "json_selection" if selection is not None else "json_structure",
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if selection is None:
        return {**result, **_overview(value, max_bytes)}
    return {**result, **_selected(value, selection, max_bytes)}
