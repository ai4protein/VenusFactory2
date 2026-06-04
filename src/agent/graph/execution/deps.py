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

    # Strip the leading "step_" / "step" alias and pull the first run of
    # digits — tolerates the LLM hallucinating suffixes like "step_5b",
    # "step5a", "step_5_a" (all interpreted as step 5).
    dep_token_raw = parts[1].replace("step_", "").replace("step", "").strip()
    import re as _re
    m = _re.match(r"\d+", dep_token_raw)
    if not m:
        return False, token, f"Invalid dependency step token for `{key_hint}`: {token}"
    dep_step = int(m.group(0))

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


def _looks_like_placeholder(value: str) -> tuple[bool, int]:
    """Detect placeholder-style strings the LLM sometimes emits in place
    of a real ``dependency:step_N:...`` token. Returns (matched, step_no).

    Examples we have seen in production CB output:
    - ``"PLACEHOLDER_FROM_STEP_3"``  →  step 3
    - ``"FROM_STEP_5"``              →  step 5
    - ``"<step_2 output>"``          →  step 2
    - ``"{{step_4_file_path}}"``     →  step 4
    - ``"step_7_output"``            →  step 7
    """
    import re as _re
    if not isinstance(value, str):
        return False, -1
    s = value.strip()
    if not s:
        return False, -1
    # Reject if it's already a real path or URL
    if "/" in s or s.startswith(("http://", "https://", ".", "~")):
        return False, -1
    # Reject if length suggests real content (paths, JSON snippets)
    if len(s) > 80:
        return False, -1
    # Two-stage match: (a) the whole string must look like a placeholder
    # (contains "step" + a number, optionally with placeholder/from/output
    # surrounding markers); (b) extract the step number.
    s_low = s.lower()
    placeholder_markers = ("placeholder", "from", "output", "<", "{{", "{", "[[")
    has_marker = any(m in s_low for m in placeholder_markers)
    # If no marker AND the string doesn't START with step_, reject — we
    # don't want to false-positive on free-form sentences containing "step 3".
    if not has_marker and not _re.match(r"(?i)^\s*step[_\s-]?\d+", s):
        return False, -1
    # Extract the step number.
    m = _re.search(r"(?i)step[_\s-]?(\d+)", s)
    if not m:
        return False, -1
    try:
        return True, int(m.group(1))
    except Exception:
        return False, -1


def _maybe_read_json_file_as_string(value: Any, key_hint: str) -> Any:
    """When a tool parameter expects a JSON-encoded string (heuristic: key
    name ends in ``_json``) but the resolved value is a path to an existing
    JSON file on disk, read the file content and use it as the string value.

    Also handles the dict-passthrough case: when ``_resolve_one_token``
    couldn't find a field path and returned the parsed dict, we look inside
    for a ``file_path`` or ``output_files[0]`` that points to a real JSON
    file and read THAT.

    Bridges the common CB pattern where step N writes a JSON file and step
    N+1 wants to pass the JSON content (not the path) to a tool like
    ``proteinmpnn_sequence_design_from_structure``'s
    ``fixed_residues_json`` parameter.
    """
    import os as _os
    import json as _json

    if not isinstance(key_hint, str) or not key_hint.lower().endswith("_json"):
        return value

    # First normalize the value to a candidate path. Accept three shapes:
    #   (a) string that looks like a path to a .json file
    #   (b) dict that carries file_path / output_files containing a .json
    candidate_path: str | None = None
    if isinstance(value, str):
        if value.endswith(".json") or "/" in value:
            candidate_path = value
    elif isinstance(value, dict):
        for k in ("file_path", "path", "out_path"):
            v = value.get(k)
            if isinstance(v, str) and v.endswith(".json"):
                candidate_path = v
                break
        if candidate_path is None:
            fi = value.get("file_info")
            if isinstance(fi, dict):
                fp = fi.get("file_path")
                if isinstance(fp, str) and fp.endswith(".json"):
                    candidate_path = fp
        if candidate_path is None:
            ofs = value.get("output_files")
            if isinstance(ofs, list):
                for f in ofs:
                    if isinstance(f, str) and f.endswith(".json"):
                        candidate_path = f
                        break

    if not candidate_path:
        return value

    try:
        if not _os.path.isabs(candidate_path):
            from pathlib import Path as _Path
            for root in (_Path.cwd(), _Path("/inspire/hdd/global_user/tanyang-253108120165/workspace/research/VenusFactory")):
                cand = root / candidate_path
                if cand.exists():
                    candidate_path = str(cand.resolve())
                    break
        if not _os.path.exists(candidate_path):
            return value
        with open(candidate_path, "r", encoding="utf-8") as f:
            content = f.read()
        # Validate it's parseable JSON before substituting
        _json.loads(content)
        _logger.info(
            "Dependency resolve: %s expects JSON string but got %s %s; "
            "reading file content (%d chars) as the string value.",
            key_hint,
            "dict pointing to" if isinstance(value, dict) else "file path",
            candidate_path,
            len(content),
        )
        return content
    except Exception:
        return value


def _resolve_nested(value: Any, step_results, key_hint: str) -> tuple[bool, Any, str]:
    """Recursively resolve dependency tokens inside nested str/list/dict structures.

    Plans frequently encode dependencies inside list parameters (e.g.
    ``input_files: ["dependency:step_3:file_path"]`` for ``agent_generated_code``)
    or nested dicts. Without recursion the token was passed through as a literal
    string path, and the tool then failed with FileNotFound.

    Also handles two LLM-output shapes that aren't proper dependency tokens:

    1. ``PLACEHOLDER_FROM_STEP_N`` / ``<step_N output>`` / ``step_N_output``
       — the LLM's informal way of saying "this comes from step N". We
       detect these and rewrite as if they were ``dependency:step_N``.
    2. Parameter expects JSON string (``_json`` suffix) but resolves to a
       file path on disk → read the file contents as the string value.
    """
    # Placeholder rewrite — happens BEFORE the dependency: prefix check
    if isinstance(value, str) and not value.startswith("dependency:"):
        is_placeholder, step_no = _looks_like_placeholder(value)
        if is_placeholder:
            _logger.info(
                "Dependency resolve: detected placeholder %r for key %s; "
                "treating as dependency:step_%d",
                value, key_hint, step_no,
            )
            value = f"dependency:step_{step_no}"

    if isinstance(value, str) and value.startswith("dependency:"):
        ok, resolved, reason = _resolve_one_token(value, step_results, key_hint)
        if ok:
            resolved = _maybe_read_json_file_as_string(resolved, key_hint)
        return ok, resolved, reason
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
