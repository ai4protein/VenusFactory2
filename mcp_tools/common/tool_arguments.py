"""Validate declared tool inputs without returning caller data or validator traces."""

from __future__ import annotations

import re

from pydantic import ValidationError

_FIELD_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
_RETRY = "Follow the tool input contract and retry."


class ToolArgumentError(ValueError):
    """A bounded public input-contract error, safe for MCP and harness feedback."""


def _safe_name(name):
    return isinstance(name, str) and _FIELD_NAME.fullmatch(name) is not None


def _field_hint(label, names):
    names = sorted(name for name in names if _safe_name(name))
    qualifier = " include" if len(names) > 12 else ""
    return f"{label}{qualifier}: " + ", ".join(names[:12]) + "." if names else ""


def _feedback(details):
    # Keep whole schema-owned clauses. Never truncate into submitted input data.
    message = ""
    for detail in details:
        if detail and len(message) + len(detail) + len(_RETRY) + 2 <= 1500:
            message += detail + " "
    return message + _RETRY


def validate_argument_fields(contract, arguments):
    """Reject undeclared top-level fields in an explicitly closed tool surface."""
    if not isinstance(arguments, dict):
        raise ToolArgumentError("Use a JSON object with the declared tool arguments")
    declared = contract.get("properties", {})
    if any(name not in declared for name in arguments):
        raise ToolArgumentError(
            _feedback(
                [
                    "Unexpected fields. Remove arguments that are not declared.",
                    _field_hint("Allowed fields", declared),
                ]
            )
        )


def _object_fields(node, contract):
    """Read declared object fields through local references and union branches.

    This supplies hints, not another schema validator. Pydantic still decides
    validity; no submitted keys, values, descriptions or remote references enter
    the feedback. Cyclic or unusually large schemas have a fixed traversal limit.
    """
    pending, visited, fields = [node], set(), set()
    for _ in range(32):
        if not pending:
            break
        current = pending.pop()
        if not isinstance(current, dict) or id(current) in visited:
            continue
        visited.add(id(current))
        fields.update(name for name in current.get("properties", {}) if _safe_name(name))
        ref = current.get("$ref", "")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            key = ref[len("#/$defs/") :]
            if "/" not in key:
                pending.append(contract.get("$defs", {}).get(key.replace("~1", "/").replace("~0", "~")))
        for keyword in ("anyOf", "oneOf", "allOf"):
            branches = current.get(keyword, [])
            if isinstance(branches, list):
                pending.extend(branches[:8])
    return fields


def validate_tool_arguments(tool, arguments):
    """Use the tool's own schema; preserve supplied values and host-bound paths."""
    validate_arguments(tool.get_input_schema(), arguments)


def validate_arguments(schema, arguments):
    """Validate an explicit public contract without normalizing the caller's data."""
    try:
        schema.model_validate(arguments)
    except ValidationError as exc:
        contract = schema.model_json_schema()
        declared = contract.get("properties", {})
        missing, invalid = [], []
        for error in exc.errors(include_input=False, include_url=False)[:8]:
            location = error.get("loc", ())
            name = location[0] if location else None
            field = name if _safe_name(name) and name in declared else "arguments"
            # A missing member of an existing nested object is not an absent
            # top-level argument. Its declared object fields are shown below.
            fields = missing if error.get("type") == "missing" and len(location) == 1 else invalid
            if field not in fields:
                fields.append(field)
        details = []
        if missing:
            details.append("Missing required arguments: " + ", ".join(missing) + ".")
        if invalid:
            details.append("Invalid arguments: " + ", ".join(invalid) + ".")
        for field in invalid:
            if field in declared:
                details.append(
                    _field_hint(
                        "Declared object fields for " + field,
                        _object_fields(declared[field], contract),
                    )
                )
        if "arguments" in invalid:
            # Root-level alternatives describe cross-field requirements. Only
            # schema-owned, declared names may be returned; never validator text.
            alternatives = []
            for branch in contract.get("anyOf", [])[:8]:
                required = branch.get("required", [])
                if required and all(_safe_name(name) and name in declared for name in required):
                    alternatives.append(" + ".join(required[:8]))
            if alternatives:
                details.append("Required alternatives: " + " or ".join(alternatives) + ".")
        raise ToolArgumentError(_feedback(details)) from None
