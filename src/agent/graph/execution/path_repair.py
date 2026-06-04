"""Missing-file path repair.

Two distinct flows live here:

* :func:`repair_missing_file_paths` — pre-execution sweep, mirrors the original
  L1922-1977 block. Searches only the session dir, then step-results, then
  protein_ctx files.
* :func:`rebind_file_not_found` — post-failure rebinder for ``FileNotFoundError``
  output, mirrors L2056-2137. Returns a new invoke input (or ``None`` if no
  change) so the orchestrator can decide whether to spend a retry slot.
"""

from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from logger import get_logger

from agent.chat_agent_utils import (
    _get_output_file_path_from_raw,
    _tool_output_indicates_failure,
)
from agent.graph.execution.context import ExecutionContext
from agent.graph.execution.prepare import merge_retry_input

_logger = get_logger("agent.graph")


class PathRepairScope(str, Enum):
    SESSION_ONLY = "session_only"
    SESSION_AND_PREV_STEPS = "session_and_prev_steps"
    DISABLED = "disabled"


class AmbiguousFileRepairError(Exception):
    """Raised when multiple repair candidates are found for the same basename."""


def repair_missing_file_paths(
    ctx: ExecutionContext,
    scope: PathRepairScope = PathRepairScope.SESSION_AND_PREV_STEPS,
) -> dict[str, Any]:
    """Pre-execution path repair, behaviour matches the original block.

    ``scope`` is accepted for forward compatibility (see docs/dev/02 §8) but the
    legacy default behaviour searches session dir + step_results + protein_ctx,
    which is ``SESSION_AND_PREV_STEPS``. ``DISABLED`` skips repair entirely.
    """
    invoke_input = ctx.invoke_input
    if scope == PathRepairScope.DISABLED:
        return invoke_input
    if not isinstance(invoke_input, dict):
        return invoke_input

    session_root = (
        Path(ctx.agent_session_dir).expanduser().resolve() if ctx.agent_session_dir else None
    )
    step_results = ctx.step_results
    protein_ctx = ctx.protein_ctx

    repaired_input = dict(invoke_input)
    for _pk, _pv in list(repaired_input.items()):
        if not isinstance(_pv, str) or not _pv.strip():
            continue
        _pk_l = _pk.lower()
        if not any(tok in _pk_l for tok in ("path", "file", "input")):
            continue
        _pv_path = Path(_pv).expanduser()
        if _pv_path.exists():
            continue
        _repaired: Optional[str] = None
        _basename = _pv_path.name
        # Search ONLY within the current agent session directory to keep results
        # predictable and avoid leaking unrelated project/cwd matches.
        search_roots = [session_root] if session_root else []
        for _sr in search_roots:
            try:
                _matches = [p for p in _sr.rglob(_basename) if p.is_file()]
                if not _matches:
                    continue
                if len(_matches) > 1:
                    _logger.info(
                        "Pre-exec path repair: %d candidates for %s under %s; picking newest by mtime",
                        len(_matches),
                        _basename,
                        _sr,
                    )
                    _matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                _found = _matches[0]
                if _found.exists() and _found.is_file():
                    _repaired = str(_found.resolve())
                    break
            except Exception:
                pass
        if not _repaired and scope != PathRepairScope.SESSION_ONLY:
            for _dep_step_no in _sorted_step_keys(step_results):
                from agent.graph.helpers.tool_io import _get_step_raw_output

                _dep_raw = _get_step_raw_output(step_results, _dep_step_no)
                if _dep_raw is None:
                    continue
                _dep_file = _get_output_file_path_from_raw(_dep_raw, "dependency_step")
                if (
                    _dep_file
                    and Path(_dep_file).exists()
                    and Path(_dep_file).name == _basename
                ):
                    _repaired = _dep_file
                    break
        if not _repaired and scope != PathRepairScope.SESSION_ONLY:
            try:
                ctx_files = list(protein_ctx.files.values()) if hasattr(protein_ctx, "files") else []
                for _cf in ctx_files:
                    _cf_path = _cf.get("path") if isinstance(_cf, dict) else None
                    if (
                        isinstance(_cf_path, str)
                        and Path(_cf_path).exists()
                        and Path(_cf_path).name == _basename
                    ):
                        _repaired = str(Path(_cf_path).resolve())
                        break
            except Exception:
                pass
        if _repaired:
            _logger.info("Pre-exec path repair: %s=%s -> %s", _pk, _pv, _repaired)
            repaired_input[_pk] = _repaired
    return repaired_input


def rebind_file_not_found(
    ctx: ExecutionContext,
    raw_output: Any,
    failure_reason: str,
) -> Optional[dict[str, Any]]:
    """Detect FileNotFoundError-style failures, rebind paths, and return new input.

    Returns ``None`` when:

    * No FileNotFoundError signature was found, or
    * No candidate paths exist, or
    * No path could actually be substituted (so retrying would just repeat).

    Returning a dict means a substitution happened and the orchestrator should
    spend one ``file_not_found`` retry slot.
    """
    from agent.graph.helpers.plan_helpers import _normalize_step_number
    from agent.graph.helpers.tool_io import _get_step_raw_output

    is_failure, derived_reason = _tool_output_indicates_failure(raw_output)
    failure_reason = failure_reason or derived_reason or ""

    raw_text = raw_output if isinstance(raw_output, str) else (
        json.dumps(raw_output) if not isinstance(raw_output, (int, float, bool, type(None))) else str(raw_output)
    )
    raw_lower = raw_text.lower() if isinstance(raw_text, str) else ""
    _fnf_signatures = (
        "filenotfounderror",
        "no such file",
        "does not exist",
        "could not find",
        "cannot find",
        "search root directory does not exist",  # common agent_generated_code message
        # Schema-mismatch signatures — same recovery (re-prompt with all
        # upstream paths + inspect-before-assume hint) usually works because
        # the LLM regenerates the script after re-reading file_info.
        "keyerror",
        "could not identify",
        "column not found",
        "no column named",
        # Sandbox-grant violations: the LLM walked outside session_root
        # (e.g. os.walk('/'), reading project source). Retry with explicit
        # input_files pinned to the granted upstream paths.
        "outside granted directories",
        "sandbox validation failed",
        # Plot-task enforcement (train_operations): script didn't actually
        # save an image even though task wanted one. Re-prompting with the
        # input_files makes the LLM produce a real PNG.
        "plot-task enforcement",
        "task asked for a chart",
        # Static self-check failures (syntax errors, empty data assignment,
        # PALETTE multi-line key bug). H6 retry re-prompts with the input_files
        # populated, which usually fixes these.
        "static self-check failed",
        "syntaxerror at line",
        "empty ``data`` assignment",
    )
    _fnf_detected = any(sig in raw_lower for sig in _fnf_signatures) or (
        is_failure
        and isinstance(failure_reason, str)
        and ("file" in failure_reason.lower() and ("not found" in failure_reason.lower() or "does not exist" in failure_reason.lower()))
    )
    if not _fnf_detected:
        return None

    candidate_paths: list[str] = []
    try:
        candidate_paths.extend(
            [
                f.get("path")
                for f in ctx.protein_ctx.files.values()
                if isinstance(f, dict) and isinstance(f.get("path"), str)
            ]
        )
    except Exception:
        pass
    for dep_step in sorted(
        ctx.step_results.keys(), key=lambda x: _normalize_step_number(x, 0), reverse=True
    ):
        dep_raw = _get_step_raw_output(ctx.step_results, dep_step)
        if dep_raw is None:
            continue
        dep_path = _get_output_file_path_from_raw(dep_raw, "dependency_step")
        if dep_path:
            candidate_paths.append(dep_path)
    candidate_paths = sorted(
        list({os.path.abspath(p) for p in candidate_paths if isinstance(p, str) and p.strip()})
    )

    if not candidate_paths:
        return None

    invoke_input = ctx.invoke_input

    if ctx.tool_name == "python_repl":
        from agent.graph.helpers.tool_io import _sanitize_tool_invoke_input

        retry_seed = dict(invoke_input)
        retry_seed.setdefault("files", [])
        if isinstance(retry_seed.get("files"), list):
            retry_seed["files"] = sorted(list({*retry_seed["files"], *candidate_paths}))
        if not retry_seed.get("last_file"):
            retry_seed["last_file"] = candidate_paths[-1]
        rebound_input = _sanitize_tool_invoke_input(
            ctx.tool_name,
            ctx.tool,
            retry_seed,
            ctx.agent_session_dir,
            ctx.step_results,
        )
        new_query = rebound_input.get("query") if isinstance(rebound_input, dict) else None
        old_query = invoke_input.get("query") if isinstance(invoke_input, dict) else None
        if (
            isinstance(new_query, str)
            and isinstance(old_query, str)
            and new_query != old_query
        ):
            return merge_retry_input(ctx, invoke_input, rebound_input)
        return None

    # Special handling for agent_generated_code: the failure is INSIDE the
    # LLM-generated script (it hardcoded a path in code that doesn't exist on
    # disk). The script's input_files may be empty or omit the upstream output
    # the CB planner forgot to wire up. Auto-inject ALL upstream file_path
    # values into ``input_files`` and amend the task_description so the LLM
    # rewrites the script using the structured file list rather than the
    # original hardcoded names. This addresses the common CB pattern of
    # referencing files by name in task_description text instead of via
    # ``dependency:step_N:file_path`` tokens.
    if ctx.tool_name == "agent_generated_code":
        retry_seed = dict(invoke_input)
        existing_inputs = retry_seed.get("input_files") or []
        if not isinstance(existing_inputs, list):
            existing_inputs = []
        # Build the augmented input_files list by union with all upstream paths.
        # Coerce items to strings — agent_generated_code's CodeExecutionInput
        # rejects non-string entries, and dependency-token resolution
        # sometimes leaves a parsed dict/None in the list when the upstream
        # output doesn't have an obvious path field.
        def _as_path_str(v: Any) -> str | None:
            if isinstance(v, str):
                return v if v.strip() else None
            if isinstance(v, dict):
                for k in ("file_path", "path", "out_path", "config_path", "fasta_path", "model_path"):
                    if isinstance(v.get(k), str) and v[k].strip():
                        return v[k]
                fi = v.get("file_info")
                if isinstance(fi, dict) and isinstance(fi.get("file_path"), str):
                    return fi["file_path"]
            return None

        augmented: list[str] = []
        for src in (existing_inputs, candidate_paths):
            for item in src:
                s = _as_path_str(item)
                if s and s not in augmented and os.path.exists(s):
                    augmented.append(s)
        if not augmented or augmented == [s for s in (existing_inputs if all(isinstance(x, str) for x in existing_inputs) else []) ]:
            return None
        retry_seed["input_files"] = augmented
        # Amend task_description so the LLM is forced to use the structured list
        task_desc = retry_seed.get("task_description") or ""
        retry_hint = (
            "\n\n[AUTO-RETRY] Previous attempt failed with a file-not-found error. "
            "The harness has populated `input_files` with the actual absolute paths "
            "of upstream tool outputs. Use ONLY these paths via the file_info `path` "
            "field — do NOT reference any hardcoded filename from the original task "
            "text. If a required file is not in input_files, save a clear error and "
            "return success=false with details."
        )
        if "[AUTO-RETRY]" not in str(task_desc):
            retry_seed["task_description"] = str(task_desc) + retry_hint
        return merge_retry_input(ctx, invoke_input, retry_seed)

    # General tool path rebinding: replace broken file-like input values
    _rebind_changed = False
    session_root_rebind = (
        Path(ctx.agent_session_dir).expanduser().resolve() if ctx.agent_session_dir else None
    )
    retry_seed = dict(invoke_input)
    for _rk, _rv in list(retry_seed.items()):
        if not isinstance(_rv, str) or not _rv.strip():
            continue
        _rk_l = _rk.lower()
        if not any(tok in _rk_l for tok in ("path", "file", "input")):
            continue
        if Path(_rv).expanduser().exists():
            continue
        _target_name = Path(_rv).name
        # Try candidate paths first
        for _cp in candidate_paths:
            if Path(_cp).name == _target_name and Path(_cp).exists():
                retry_seed[_rk] = _cp
                _rebind_changed = True
                break
        if not _rebind_changed and session_root_rebind:
            try:
                _match = next(session_root_rebind.rglob(_target_name), None)
                if _match and _match.exists() and _match.is_file():
                    retry_seed[_rk] = str(_match.resolve())
                    _rebind_changed = True
            except Exception:
                pass
    if _rebind_changed:
        return merge_retry_input(ctx, invoke_input, retry_seed)
    return None


def _sorted_step_keys(step_results: dict) -> list:
    from agent.graph.helpers.plan_helpers import _normalize_step_number

    return sorted(step_results.keys(), key=lambda x: _normalize_step_number(x, 0), reverse=True)
