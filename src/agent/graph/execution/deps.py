"""Dependency token resolution (``dependency:step_N[:field...]``).

Extracted verbatim from ``_execute_node_impl`` (the loop that scans the merged
tool input and pulls upstream step outputs in place). Logging messages and
failure semantics are preserved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from logger import get_logger

from agent.chat_agent_utils import (
    _get_output_file_path_from_raw,
    _tool_output_indicates_failure,
)
from agent.graph.execution.context import ExecutionContext

_logger = get_logger("agent.graph")


@dataclass
class DependencyResolveResult:
    ok: bool
    invoke_input: dict
    reason: str = ""


def resolve_dependencies(ctx: ExecutionContext) -> DependencyResolveResult:
    """Resolve ``dependency:step_N[:field...]`` tokens in ``ctx.merged_tool_input``.

    Returns a result with ``ok=False`` and a human-readable ``reason`` matching
    the original ``_execute_node_impl`` wording on the first failure encountered.
    On success, ``invoke_input`` is the mutated mapping ready for downstream
    sanitization.
    """
    from agent.graph.helpers.tool_io import _get_step_raw_output

    merged_tool_input = dict(ctx.merged_tool_input)
    step_results = ctx.step_results

    dependency_failure_reason = ""

    for key, value in list(merged_tool_input.items()):
        if not (isinstance(value, str) and value.startswith("dependency:")):
            continue

        parts = value.split(":")
        if len(parts) < 2:
            dependency_failure_reason = f"Invalid dependency token for `{key}`: {value}"
            _logger.info("Dependency resolve: %s", dependency_failure_reason)
            return DependencyResolveResult(False, merged_tool_input, dependency_failure_reason)

        dep_token = parts[1].replace("step_", "").replace("step", "").strip()
        try:
            dep_step = int(dep_token)
        except ValueError:
            dependency_failure_reason = f"Invalid dependency step token for `{key}`: {value}"
            _logger.info("Dependency resolve: %s", dependency_failure_reason)
            return DependencyResolveResult(False, merged_tool_input, dependency_failure_reason)

        dep_out = _get_step_raw_output(step_results, dep_step)
        if dep_out is None:
            dependency_failure_reason = f"Missing output for dependency step {dep_step} (key={key})"
            _logger.info("Dependency resolve: %s", dependency_failure_reason)
            return DependencyResolveResult(False, merged_tool_input, dependency_failure_reason)

        dep_failed, dep_reason = _tool_output_indicates_failure(dep_out)
        if dep_failed:
            dependency_failure_reason = (
                f"Dependency step {dep_step} failed"
                + (f": {dep_reason}" if dep_reason else "")
                + f" (needed for `{key}`)"
            )
            _logger.info("Dependency resolve: %s", dependency_failure_reason)
            return DependencyResolveResult(False, merged_tool_input, dependency_failure_reason)

        parsed: Any = dep_out
        if isinstance(dep_out, str):
            try:
                parsed = json.loads(dep_out)
            except Exception:
                parsed = dep_out

        if len(parts) > 2:
            field_path = [p for p in parts[2:] if p]
            cursor = parsed
            field_ok = True
            for field in field_path:
                if isinstance(cursor, dict) and field in cursor:
                    cursor = cursor[field]
                else:
                    field_ok = False
                    break
            if field_ok:
                val = cursor
            else:
                val = dep_out
                _logger.debug(
                    "Dependency resolve: field path %s not found in step %s output; using raw output",
                    "/".join(field_path),
                    dep_step,
                )
        else:
            val = dep_out

        # Heuristic auto-extraction for paths if the expected parameter is a file or path
        if any(k in key.lower() for k in ("path", "file")):
            if isinstance(val, dict):
                if "file_path" in val:
                    val = val["file_path"]
                elif (
                    "file_info" in val
                    and isinstance(val["file_info"], dict)
                    and "file_path" in val["file_info"]
                ):
                    val = val["file_info"]["file_path"]
            elif isinstance(val, str):
                extracted = _get_output_file_path_from_raw(val, "previous_step")
                if extracted:
                    val = extracted
            if val == dep_out and isinstance(dep_out, str):
                extracted = _get_output_file_path_from_raw(dep_out, "previous_step")
                if extracted:
                    val = extracted

        merged_tool_input[key] = val

    return DependencyResolveResult(True, merged_tool_input)
