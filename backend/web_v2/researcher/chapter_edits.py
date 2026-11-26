"""Apply author edits only at independently flagged Markdown spans."""
import json
import re

from .source_review import is_review_line


def _unique_fields(pairs):
    value = dict(pairs)
    if len(value) != len(pairs):
        raise ValueError("Duplicate edit field")
    return value


def unit_locations(draft, units):
    """Locate review text in the raw Markdown while retaining its original offsets."""
    lines, offsets = [], []
    start = 0
    for raw in draft.splitlines(keepends=True):
        line = raw.rstrip("\r\n")
        if is_review_line(line):
            if lines:
                offsets.append(start - 1)
            lines.append(line)
            offsets.extend(range(start, start + len(line)))
        start += len(raw)
    normalized = "\n".join(lines)
    locations = {}
    for unit in units:
        text = unit["text"]
        index = normalized.find(text)
        if not text or index < 0 or normalized.find(text, index + 1) >= 0:
            raise ValueError("Ambiguous source paragraph")
        locations[unit["id"]] = (offsets[index], offsets[index + len(text) - 1] + 1)
    return locations


def checked_full_revision(draft, spans, response):
    """Legacy writers may return Markdown. Only the requested spans may change."""
    gaps, last = [], 0
    for start, end in sorted(spans):
        if start < last:
            raise ValueError("Overlapping replacement targets")
        gaps.append(draft[last:start])
        last = end
    gaps.append(draft[last:])
    if response == "".join(gaps).strip():
        return response
    prefix, suffix = gaps[0], gaps[-1]
    stop = len(response) - len(suffix)
    cursor = len(prefix)
    if not response.startswith(prefix) or not response.endswith(suffix) or cursor > stop:
        raise ValueError("Revision changed a protected paragraph")
    for gap in gaps[1:-1]:
        index = response.find(gap, cursor, stop)
        if index < 0:
            raise ValueError("Revision changed a protected paragraph")
        cursor = index + len(gap)
    return response


def apply_unit_edits(draft, units, response, *, locations=None):
    """Apply one checked replacement per flagged unit against the original draft."""
    locations = dict(locations) if locations is not None else unit_locations(draft, units)
    if not locations:
        raise ValueError("No requested replacement targets")
    fence = re.fullmatch(r"\s*```(?:json)?\s*([\s\S]*?)\s*```\s*", response, re.IGNORECASE)
    if fence:
        response = fence[1].strip()
    if not response.lstrip().startswith(("{", "[")):
        return checked_full_revision(draft, locations.values(), response)
    edits = json.loads(response, object_pairs_hook=_unique_fields)
    if isinstance(edits, dict) and set(edits) == {"response_format"}:
        edits = edits["response_format"]
    if isinstance(edits, dict) and set(edits) == {"replacements"}:
        edits = edits["replacements"]
    if isinstance(edits, dict) and set(edits) == set(locations):
        edits = [{"unit_id": key, "text": value} for key, value in edits.items()]
    if len(locations) == 1 and isinstance(edits, dict) and set(edits) == {"unit_id", "text"}:
        edits = [edits]
    if not isinstance(edits, list):
        raise ValueError("Expected chapter replacements")
    patches = []
    for row in edits:
        if not isinstance(row, dict) or set(row) != {"unit_id", "text"}:
            raise ValueError("Invalid chapter replacement")
        span = locations.pop(row["unit_id"], None)
        if not span or not isinstance(row["text"], str):
            raise ValueError("Missing text or unknown replacement target")
        patches.append((*span, row["text"].strip()))
    if locations:
        raise ValueError("Missing replacement target")
    boundary = len(draft)
    for start, end, text in sorted(patches, reverse=True):
        if end > boundary:
            raise ValueError("Overlapping replacement targets")
        draft = draft[:start] + text + draft[end:]
        boundary = start
    return draft

def scoped_replacement(unit, answer):
    """Keep a one-unit author reply in its original slot during independent re-review."""
    value = answer.strip()
    if value.startswith(("{", "[{", chr(96) * 3)):
        value = apply_unit_edits(unit["text"], [unit], value)
    if len(value) > 8000:
        raise ValueError("Paragraph revision exceeds the review-unit limit")
    # Extra paragraph separators would renumber unrelated, already reviewed units.
    return re.sub(r"\n(?:[ \t]*\n)+", "\n", value.replace("\r\n", "\n")).strip()
