"""Path-resolution helpers used by :mod:`agent.graph.helpers.tool_io`.

Split out only to keep ``tool_io.py`` under the < 350 LoC ceiling. The
functions here are byte-for-byte ports of the inner closures from the
original ``_sanitize_tool_invoke_input`` body, parametrised so they can
live at module scope.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from agent.chat_agent_utils import _get_output_file_path_from_raw


def coerce_sequence(value: Any) -> str | None:
    """Best-effort extraction of an amino-acid sequence string from heterogeneous input."""
    if isinstance(value, str):
        seq = value.strip()
        return seq or None
    if isinstance(value, dict):
        for key in ("sequence", "aa_sequence", "seq"):
            v = value.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        sequences = value.get("sequences")
        if isinstance(sequences, dict) and sequences:
            first = next(iter(sequences.values()))
            if isinstance(first, str) and first.strip():
                return first.strip()
        if isinstance(sequences, list) and sequences:
            first = sequences[0]
            if isinstance(first, str) and first.strip():
                return first.strip()
            if isinstance(first, dict):
                for key in ("sequence", "aa_sequence", "seq"):
                    v = first.get(key)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
        headers = value.get("headers")
        if isinstance(headers, list):
            aa_re = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]{20,}", re.IGNORECASE)
            for h in headers:
                if isinstance(h, str):
                    m = aa_re.search(h)
                    if m:
                        return m.group(0).upper()
        for key in ("result", "data", "output"):
            nested = value.get(key)
            if nested is not None:
                nested_seq = coerce_sequence(nested)
                if nested_seq:
                    return nested_seq
    if isinstance(value, list) and value:
        for item in value:
            seq = coerce_sequence(item)
            if seq:
                return seq
    return None


def maybe_resolve_local_path(
    raw: Any,
    project_root: Path,
    session_root: Path | None,
) -> Any:
    """Resolve a path string against project/session/cwd, returning original on failure."""
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    if not text:
        return raw
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return str(candidate.resolve()) if candidate.exists() else raw

    for root in (project_root, session_root, Path.cwd().resolve()):
        if root is None:
            continue
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return str(resolved)

    # If only basename is provided, try locating it under current session output tree.
    if session_root is not None and len(candidate.parts) == 1:
        try:
            matched = next(session_root.rglob(candidate.name), None)
            if matched and matched.exists():
                return str(matched.resolve())
        except Exception:
            pass
    return raw


def rewrite_python_query_paths(
    query: Any,
    tool_name: str,
    merged_input: dict[str, Any],
    step_results: dict[Any, Any] | None,
    project_root: Path,
    session_root: Path | None,
    get_step_raw_output,
    normalize_step_number,
) -> Any:
    """Rewrite quoted basename references in python_repl query to absolute paths."""
    if tool_name != "python_repl" or not isinstance(query, str):
        return query

    def _resolve_dependency_token(token: str) -> str | None:
        if not isinstance(token, str) or not token.startswith("dependency:"):
            return None
        parts = token.split(":")
        if len(parts) < 2:
            return None
        dep_token = parts[1].replace("step_", "").replace("step", "").strip()
        try:
            dep_step = int(dep_token)
        except ValueError:
            return None
        dep_raw = get_step_raw_output(step_results, dep_step)
        if dep_raw is None:
            return None

        parsed: Any = dep_raw
        if isinstance(dep_raw, str):
            try:
                parsed = json.loads(dep_raw)
            except Exception:
                parsed = dep_raw

        if len(parts) > 2:
            cursor = parsed
            for field in [p for p in parts[2:] if p]:
                if isinstance(cursor, dict) and field in cursor:
                    cursor = cursor[field]
                else:
                    cursor = None
                    break
            val = cursor
        else:
            val = parsed

        if isinstance(val, str):
            resolved = maybe_resolve_local_path(val, project_root, session_root)
            return resolved if isinstance(resolved, str) else val
        if isinstance(val, dict):
            if "file_path" in val and isinstance(val.get("file_path"), str):
                resolved = maybe_resolve_local_path(val["file_path"], project_root, session_root)
                return resolved if isinstance(resolved, str) else val["file_path"]
            if isinstance(val.get("file_info"), dict) and isinstance(val["file_info"].get("file_path"), str):
                resolved = maybe_resolve_local_path(val["file_info"]["file_path"], project_root, session_root)
                return resolved if isinstance(resolved, str) else val["file_info"]["file_path"]

        extracted = _get_output_file_path_from_raw(dep_raw, "dependency_step")
        return extracted

    rewritten = query

    # Replace template-like placeholders: {{step_5.file_info.file_path}}
    for token in set(re.findall(r"\{\{step_?\d+(?:\.[A-Za-z0-9_]+)+\}\}", rewritten)):
        inner = token.strip("{}")
        token_as_dep = "dependency:" + inner.replace(".", ":")
        resolved = _resolve_dependency_token(token_as_dep)
        if resolved:
            rewritten = rewritten.replace(token, resolved)

    # Replace direct dependency tokens in code text.
    for token in set(re.findall(r"dependency:step_?\d+(?::[A-Za-z0-9_]+)*", rewritten)):
        resolved = _resolve_dependency_token(token)
        if resolved:
            rewritten = rewritten.replace(token, resolved)

    candidate_paths: list[str] = []
    for key, value in merged_input.items():
        key_l = str(key).lower()
        if isinstance(value, str) and any(tok in key_l for tok in ("path", "file", "dir")):
            candidate_paths.append(value)
        if key_l == "last_file" and isinstance(value, str):
            candidate_paths.append(value)
        if key_l == "files" and isinstance(value, list):
            candidate_paths.extend([v for v in value if isinstance(v, str)])
    if isinstance(step_results, dict) and step_results:
        for step_no in sorted(
            step_results.keys(), key=lambda x: normalize_step_number(x, 0), reverse=True
        ):
            raw_output = get_step_raw_output(step_results, step_no)
            if raw_output is None:
                continue
            extracted = _get_output_file_path_from_raw(raw_output, "dependency_step")
            if extracted:
                candidate_paths.append(extracted)

    basename_to_abs: dict[str, str] = {}
    for raw_path in candidate_paths:
        resolved = maybe_resolve_local_path(raw_path, project_root, session_root)
        if not isinstance(resolved, str):
            continue
        abs_path = Path(resolved).expanduser()
        if abs_path.exists() and abs_path.is_file():
            name = abs_path.name
            # Only rewrite when basename maps to one unique file.
            if name not in basename_to_abs:
                basename_to_abs[name] = str(abs_path.resolve())
            elif basename_to_abs[name] != str(abs_path.resolve()):
                basename_to_abs.pop(name, None)

    # Also replace plain basenames from known files.
    for name, abs_path in basename_to_abs.items():
        rewritten = rewritten.replace(f"'{name}'", f"'{abs_path}'")
        rewritten = rewritten.replace(f"\"{name}\"", f"\"{abs_path}\"")
    return rewritten
