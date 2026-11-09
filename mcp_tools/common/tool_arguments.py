"""Validate declared tool inputs without returning caller data or validator traces."""
from __future__ import annotations

import re
from pydantic import ValidationError


class ToolArgumentError(ValueError):
    """A bounded public input-contract error, safe for MCP and harness feedback."""


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
            field = (name if isinstance(name, str) and name in declared
                     and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name)
                     else "arguments")
            fields = missing if error.get("type") == "missing" else invalid
            if field not in fields:
                fields.append(field)
        details = []
        if missing:
            details.append("Missing required arguments: " + ", ".join(missing) + ".")
        if invalid:
            details.append("Invalid arguments: " + ", ".join(invalid) + ".")
        if "arguments" in invalid:
            # Root-level alternatives describe cross-field requirements. Only
            # schema-owned, declared names may be returned; never validator text.
            alternatives = []
            for branch in contract.get("anyOf", [])[:8]:
                required = branch.get("required", [])
                if required and all(
                    isinstance(name, str) and name in declared
                    and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name)
                    for name in required
                ):
                    alternatives.append(" + ".join(required[:8]))
            if alternatives:
                details.append("Required alternatives: " + " or ".join(alternatives) + ".")
        raise ToolArgumentError(" ".join(details) + " Follow the tool input contract and retry.") from None
