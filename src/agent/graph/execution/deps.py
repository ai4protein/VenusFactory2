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


def _resolve_one_token(token: str, step_results, key_hint: str) -> tuple[bool, Any, str]:
    """Resolve a single ``dependency:step_N[:field...]`` string.

    Returns (ok, value, reason). ``key_hint`` is the surrounding parameter name
    (used by the file_path heuristic — e.g. for ``input_files`` we still want to
    auto-extract ``file_info.file_path``).
    """
    from agent.graph.helpers.tool_io import _get_step_raw_output

    parts = token.split(":")
    if len(parts) < 2:
        return False, token, f"Invalid dependency token for `{key_hint}`: {token}"

    dep_token = parts[1].replace("step_", "").replace("step", "").strip()
    try:
        dep_step = int(dep_token)
    except ValueError:
        return False, token, f"Invalid dependency step token for `{key_hint}`: {token}"

    dep_out = _get_step_raw_output(step_results, dep_step)
    if dep_out is None:
        return False, token, f"Missing output for dependency step {dep_step} (key={key_hint})"

    dep_failed, dep_reason = _tool_output_indicates_failure(dep_out)
    if dep_failed:
        return False, token, (
            f"Dependency step {dep_step} failed"
            + (f": {dep_reason}" if dep_reason else "")
            + f" (needed for `{key_hint}`)"
        )

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
            # Keep the parsed dict (if any) so the path-extraction heuristic
            # below can still find a usable field (e.g. requested ``file_path``
            # missing but ``config_path`` present). Falling back to ``dep_out``
            # (the raw JSON string) would hide the structured fields from the
            # heuristic and make it pass the whole JSON blob through as the
            # parameter value, which then fails Pydantic validation downstream.
            val = parsed if isinstance(parsed, dict) else dep_out
            _logger.debug(
                "Dependency resolve: field path %s not found in step %s output; "
                "passing parsed dict to path heuristic",
                "/".join(field_path),
                dep_step,
            )
    else:
        # No explicit field requested. Prefer the parsed dict so the heuristic
        # below can pick the right path field; fall back to the raw string.
        val = parsed if isinstance(parsed, dict) else dep_out

    # Heuristic auto-extraction for paths if the expected parameter is a file/path/input.
    # NB: list-valued params like ``input_files`` carry the 'file' substring so this
    # heuristic also applies per-element after recursion.
    if any(k in key_hint.lower() for k in ("path", "file", "input", "config")):
        # Ordered fallback for path-like fields, broadest first. Tools differ
        # in which field carries the canonical output path (``file_path`` is
        # most common, but ``config_path`` for generate_training_config,
        # ``model_path``/``output_dir`` for trainers, ``fasta_path`` for
        # proteinmpnn, etc.). Try the most specific to the caller's key first,
        # then broader fallbacks, so e.g. ``key_hint=config_path`` prefers
        # ``parsed['config_path']`` over ``parsed['file_path']``.
        if isinstance(val, dict):
            key_l = key_hint.lower()
            specific_candidates = []
            if "config" in key_l:
                specific_candidates.append("config_path")
            if "model" in key_l:
                specific_candidates.extend(["model_path"])
            if "fasta" in key_l:
                specific_candidates.append("fasta_path")
            generic_candidates = [
                "file_path", "config_path", "model_path", "fasta_path",
                "output_path", "out_path", "output_file",
            ]
            ordered_keys = []
            for k in specific_candidates + generic_candidates:
                if k not in ordered_keys:
                    ordered_keys.append(k)
            picked = None
            for k in ordered_keys:
                if k in val and isinstance(val[k], str) and val[k].strip():
                    picked = val[k]
                    break
            if picked is None and isinstance(val.get("file_info"), dict):
                fi = val["file_info"]
                for k in ordered_keys:
                    if k in fi and isinstance(fi[k], str) and fi[k].strip():
                        picked = fi[k]
                        break
            if picked is not None:
                val = picked
        elif isinstance(val, str):
            extracted = _get_output_file_path_from_raw(val, "previous_step")
            if extracted:
                val = extracted
        if val == dep_out and isinstance(dep_out, str):
            extracted = _get_output_file_path_from_raw(dep_out, "previous_step")
            if extracted:
                val = extracted

    return True, val, ""


def _resolve_nested(value: Any, step_results, key_hint: str) -> tuple[bool, Any, str]:
    """Recursively resolve dependency tokens inside nested str/list/dict structures.

    Plans frequently encode dependencies inside list parameters (e.g.
    ``input_files: ["dependency:step_3:file_path"]`` for ``agent_generated_code``)
    or nested dicts. Without recursion the token was passed through as a literal
    string path, and the tool then failed with FileNotFound.
    """
    if isinstance(value, str) and value.startswith("dependency:"):
        return _resolve_one_token(value, step_results, key_hint)
    if isinstance(value, list):
        new_list = []
        for item in value:
            ok, resolved, reason = _resolve_nested(item, step_results, key_hint)
            if not ok:
                return False, value, reason
            new_list.append(resolved)
        return True, new_list, ""
    if isinstance(value, dict):
        new_dict = {}
        for k, v in value.items():
            ok, resolved, reason = _resolve_nested(v, step_results, k)
            if not ok:
                return False, value, reason
            new_dict[k] = resolved
        return True, new_dict, ""
    return True, value, ""


def resolve_dependencies(ctx: ExecutionContext) -> DependencyResolveResult:
    """Resolve ``dependency:step_N[:field...]`` tokens in ``ctx.merged_tool_input``.

    Walks top-level values AND any nested str/list/dict structures so tokens
    embedded inside list parameters (e.g. ``input_files: [...]`` for
    ``agent_generated_code``) are resolved correctly.

    Returns a result with ``ok=False`` and a human-readable ``reason`` matching
    the original ``_execute_node_impl`` wording on the first failure encountered.
    On success, ``invoke_input`` is the mutated mapping ready for downstream
    sanitization.
    """
    merged_tool_input = dict(ctx.merged_tool_input)
    step_results = ctx.step_results

    for key, value in list(merged_tool_input.items()):
        ok, resolved, reason = _resolve_nested(value, step_results, key)
        if not ok:
            _logger.info("Dependency resolve: %s", reason)
            return DependencyResolveResult(False, merged_tool_input, reason)
        merged_tool_input[key] = resolved

    return DependencyResolveResult(True, merged_tool_input)
