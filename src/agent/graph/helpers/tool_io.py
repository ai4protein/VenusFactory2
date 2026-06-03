"""Tool-input/output sanitization, path normalization, and output-field collection."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.graph.helpers._path_resolve import (
    coerce_sequence,
    maybe_resolve_local_path,
    rewrite_python_query_paths,
)
from agent.graph.helpers.plan_helpers import _normalize_step_number
from logger import get_logger
from web.utils.common_utils import get_project_root

_logger = get_logger("agent.graph")


def _normalize_tool_input(raw_input: Any) -> dict[str, Any]:
    if isinstance(raw_input, dict):
        return dict(raw_input)
    if raw_input is None:
        return {}
    if isinstance(raw_input, str):
        text = raw_input.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                return dict(parsed)
        return {"input": raw_input}
    if isinstance(raw_input, (list, tuple)):
        return {"items": list(raw_input)}
    return {"input": raw_input}


def _get_step_raw_output(step_results: dict[Any, Any] | None, step_no: Any) -> Any:
    if not isinstance(step_results, dict):
        return None
    normalized = _normalize_step_number(step_no, -1)
    for key in (normalized, str(normalized), f"step_{normalized}", f"step{normalized}"):
        item = step_results.get(key)
        if isinstance(item, dict) and "raw_output" in item:
            return item.get("raw_output")
    return None


def _get_tool_allowed_param_names(tool: Any) -> set | None:
    """Best-effort extraction of accepted parameter names from LangChain tools."""
    try:
        args_schema = getattr(tool, "args_schema", None)
        if args_schema is not None:
            model_fields = getattr(args_schema, "model_fields", None)  # pydantic v2
            if isinstance(model_fields, dict) and model_fields:
                return set(model_fields.keys())
            fields = getattr(args_schema, "__fields__", None)  # pydantic v1
            if isinstance(fields, dict) and fields:
                return set(fields.keys())
    except Exception:
        pass

    try:
        args = getattr(tool, "args", None)
        if isinstance(args, dict) and args:
            # JSON schema style: {"param": {...}} or {"properties": {...}}
            if isinstance(args.get("properties"), dict) and args["properties"]:
                return set(args["properties"].keys())
            return set(args.keys())
    except Exception:
        pass
    return None


def _map_single_arg_fallback(
    tool_name: str,
    only_key: str,
    merged_input: dict[str, Any],
) -> tuple[Any, str | None]:
    """Map planner output to a single-arg tool's only field. Returns (value, mapped_from)."""
    if "input" in merged_input:
        return merged_input.get("input"), "input"
    if "items" in merged_input:
        return merged_input.get("items"), "items"
    if tool_name == "python_repl" and only_key == "query":
        for alias in ("code", "script", "python", "source"):
            if alias in merged_input and merged_input.get(alias) not in (None, ""):
                return merged_input.get(alias), alias

    # Generic fallback for single-arg tools:
    # pick the first non-empty value from non-output/path-like keys.
    skip_keys = {
        "out_dir", "output_dir", "out_path", "output_file",
        "path", "file", "file_path", "filepath",
    }
    for key, value in merged_input.items():
        key_l = str(key).lower()
        if value in (None, ""):
            continue
        if key_l in skip_keys:
            continue
        if "path" in key_l or "file" in key_l:
            continue
        return value, str(key)
    return None, None


def _sanitize_tool_invoke_input(
    tool_name: str,
    tool: Any,
    merged_input: dict[str, Any],
    agent_session_dir: str = "",
    step_results: dict[int, Any] | None = None,
) -> dict[str, Any]:
    """Filter merged input by tool schema to avoid strict parameter-validation failures."""
    if not isinstance(merged_input, dict):
        return _normalize_tool_input(merged_input)

    allowed = _get_tool_allowed_param_names(tool)
    if not allowed:
        return merged_input

    # Tool-aware alias map: planners (especially DeepSeek) occasionally emit
    # param names that don't match the schema (e.g. ``code`` for
    # ``agent_generated_code`` which actually expects ``task_description``).
    # Translate well-known aliases before filtering so the tool doesn't fail
    # Pydantic validation with the cryptic ``Field required`` error.
    _alias_maps = {
        "agent_generated_code": {
            "code": "task_description",
            "script": "task_description",
            "source": "task_description",
            "task": "task_description",
            "prompt": "task_description",
            "language": None,  # silently dropped — sandbox is python-only
        },
        "python_repl": {
            "code": "query",
            "script": "query",
            "source": "query",
            "python": "query",
        },
        "predict_structure_esmfold": {
            "fasta": "sequence",
            "fasta_file": "sequence",
            "fasta_path": "sequence",
            "seq": "sequence",
        },
    }
    aliases = _alias_maps.get(tool_name) or {}
    if aliases:
        rewritten = {}
        for k, v in merged_input.items():
            if k in aliases:
                target = aliases[k]
                if target is None:
                    continue  # drop
                if target not in merged_input and target not in rewritten:
                    rewritten[target] = v
                    _logger.debug(
                        "Input sanitize: tool=%s, aliased %s -> %s",
                        tool_name,
                        k,
                        target,
                    )
                    continue
            rewritten[k] = v
        merged_input = rewritten

    filtered = {k: v for k, v in merged_input.items() if k in allowed}

    # If planner produced a scalar-like wrapper, map it to the only accepted field.
    if not filtered and len(allowed) == 1:
        only_key = next(iter(allowed))
        value, mapped_from = _map_single_arg_fallback(tool_name, only_key, merged_input)
        if mapped_from is not None:
            filtered[only_key] = value
            _logger.debug("Input sanitize: tool=%s, mapped %s -> %s", tool_name, mapped_from, only_key)

    if "sequence" in allowed:
        seq = coerce_sequence(filtered.get("sequence"))
        if not seq:
            seq = coerce_sequence(merged_input.get("sequence"))
        if not seq:
            seq = coerce_sequence(merged_input.get("last_sequence"))
        if seq:
            filtered["sequence"] = seq

    project_root = get_project_root().resolve()
    session_root = Path(agent_session_dir).expanduser().resolve() if agent_session_dir else None

    for key, value in list(filtered.items()):
        key_l = str(key).lower()
        if any(tok in key_l for tok in ("path", "file", "dir")):
            if isinstance(value, list):
                # Resolve each element; for ``input_files``-style params drop
                # directory entries so downstream tools (notably
                # ``agent_generated_code``) don't infer output_dir from a
                # whole directory blob and leak ``generated_scripts/`` into
                # the project root.
                drop_dirs = tool_name == "agent_generated_code" and "input" in key_l
                resolved_list = []
                for item in value:
                    if isinstance(item, str):
                        resolved_item = maybe_resolve_local_path(item, project_root, session_root)
                        if drop_dirs and isinstance(resolved_item, str):
                            try:
                                if os.path.isdir(resolved_item):
                                    _logger.warning(
                                        "Input sanitize: tool=%s dropping directory entry from %s: %s",
                                        tool_name,
                                        key,
                                        resolved_item,
                                    )
                                    continue
                            except Exception:
                                pass
                        resolved_list.append(resolved_item)
                    else:
                        resolved_list.append(item)
                filtered[key] = resolved_list
            else:
                filtered[key] = maybe_resolve_local_path(value, project_root, session_root)
    if tool_name == "python_repl" and "query" in filtered:
        filtered["query"] = rewrite_python_query_paths(
            filtered["query"],
            tool_name=tool_name,
            merged_input=merged_input,
            step_results=step_results,
            project_root=project_root,
            session_root=session_root,
            get_step_raw_output=_get_step_raw_output,
            normalize_step_number=_normalize_step_number,
        )

    if not filtered and merged_input:
        _logger.debug(
            "Input sanitize: tool=%s, kept none of %s by allowed=%s",
            tool_name,
            list(merged_input.keys()),
            sorted(list(allowed)),
        )
    return filtered


def _is_write_like_tool(tool_name: str) -> bool:
    if not tool_name:
        return False
    if tool_name.startswith("download_"):
        return True
    write_prefixes = (
        "predict_structure_",
        # Finetuned predict tools: write a CSV to out_dir; previously fell back to
        # a global agent/predict_finetuned dir, now session-scoped via out_dir.
        "predict_protein_function",
        "predict_residue_function",
        # Zero-shot mutation tools: write a heatmap CSV to out_dir; previously fell
        # back to a global Zero_shot/HeatMap dir, now session-scoped via out_dir.
        "zero_shot_mutation_",
        # ProteinMPNN design/score: write FASTAs to out_dir; previously fell back to
        # a global ProteinMPNN/Design|Score dir, now session-scoped via out_dir.
        "proteinmpnn_sequence_",
        "generate_training_config",
        "train_protein_model",
        "protein_model_predict",
        "agent_generated_code",
        "maxit_structure_convert",
        "uid_file_to_chunks",
        "pdb_dir_to_fasta",
        "unzip_archive",
        "ungzip_file",
    )
    return any(tool_name.startswith(p) for p in write_prefixes)


def _normalize_output_paths(
    tool_name: str,
    tool: Any,
    invoke_input: dict[str, Any],
    agent_session_dir: str,
) -> dict[str, Any]:
    """Rewrite output paths to session-scoped destinations (avoid repo-root writes)."""
    if not isinstance(invoke_input, dict):
        return invoke_input

    allowed = _get_tool_allowed_param_names(tool) or set(invoke_input.keys())
    out = dict(invoke_input)
    session_root = Path(agent_session_dir).expanduser().resolve()
    project_root = get_project_root().resolve()
    run_base = str(session_root / "tool_outputs" / tool_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _is_within_session(candidate: Path) -> bool:
        try:
            candidate.resolve().relative_to(session_root)
            return True
        except ValueError:
            return False

    def _as_session_abs(path_value: str, is_dir: bool) -> str:
        raw = (path_value or "").strip()
        if not raw or raw in {".", "./"}:
            return run_base if is_dir else os.path.join(run_base, f"{tool_name}_{timestamp}.json")
        if os.path.isabs(raw):
            abs_path = Path(raw).expanduser().resolve()
            if _is_write_like_tool(tool_name) and not _is_within_session(abs_path):
                fallback = Path(run_base) if is_dir else Path(run_base) / f"{tool_name}_{timestamp}.json"
                _logger.warning("Output normalize: unsafe absolute path %s; fallback to %s", raw, fallback)
                return str(fallback)
            return str(abs_path)

        # If caller already passed a project-relative temp_outputs path, anchor to project root
        # instead of nesting under the current session root again.
        if raw.startswith("temp_outputs/") or raw.startswith("temp_outputs\\"):
            resolved = (project_root / raw).resolve()
        else:
            resolved = (session_root / raw).resolve()
        if _is_write_like_tool(tool_name) and not _is_within_session(resolved):
            fallback = Path(run_base) if is_dir else Path(run_base) / f"{tool_name}_{timestamp}.json"
            _logger.warning("Output normalize: path escapes session %s; fallback to %s", raw, fallback)
            return str(fallback)
        return str(resolved)

    for key in ("out_dir", "output_dir"):
        if key in out and isinstance(out.get(key), str):
            out[key] = _as_session_abs(out[key], is_dir=True)
        elif key in out and out.get(key) is None:
            out[key] = run_base

    for key in ("out_path", "output_file"):
        if key in out and isinstance(out.get(key), str):
            out[key] = _as_session_abs(out[key], is_dir=False)
        elif key in out and out.get(key) is None:
            out[key] = os.path.join(run_base, f"{tool_name}_{timestamp}.json")

    # Missing-output fallback for write-like tools.
    if _is_write_like_tool(tool_name):
        has_output_key = any(k in out for k in ("out_dir", "output_dir", "out_path", "output_file"))
        if not has_output_key:
            if "out_dir" in allowed:
                out["out_dir"] = run_base
            elif "output_dir" in allowed:
                out["output_dir"] = run_base
            elif "out_path" in allowed:
                out["out_path"] = os.path.join(run_base, f"{tool_name}_{timestamp}.json")
            elif "output_file" in allowed:
                out["output_file"] = os.path.join(run_base, f"{tool_name}_{timestamp}.json")
    return out


def _collect_output_fields(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    keys = ("out_dir", "output_dir", "out_path", "output_file")
    return {k: data.get(k) for k in keys if k in data}
