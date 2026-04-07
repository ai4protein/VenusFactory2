"""
LangGraph orchestration for the VenusFactory2 Agent.
Decouples agent logic (PI -> CB -> MLS) from the UI (chat_tab.py).
"""
import json
import os
import time
import asyncio
import shutil
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, TypedDict, Annotated, Sequence

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from agent.chat_agent import (
    Chat_LLM, ProteinContextManager, 
    _merge_tool_parameters_with_context, 
    get_cached_tool_result, save_cached_tool_result
)
from agent.chat_agent_utils import (
    AGENT_CHAT_MAX_TOOL_CALLS, AGENT_CHAT_MAX_MESSAGES,
    PI_SEARCH_TOOL_NAMES, _run_section_search, _parse_cb_plan,
    _parse_sub_report_short_title,
    _tool_output_indicates_failure,
    _run_mls_debug_with_tools, _run_mls_post_step_verify,
    _get_output_file_path_from_raw, _read_output_file_preview,
    _cb_post_step_check, MAX_STEP_RETRIES, TOOL_EXECUTION_TIMEOUT,
    _extract_image_paths_from_tool_output
)
from web.utils.chat_format_utils import (
    _short_topic_title, _format_search_summary, _format_search_preview
)
from web.utils.common_utils import get_project_root, to_project_relative_path
from agent.skills import get_skills_metadata_string
from web.utils.file_oss import upload_file_to_cloud_async
from web_v2.analytics_store import analytics_store


class AgentState(TypedDict):
    # Core state
    messages: List[BaseMessage]
    protein_context: ProteinContextManager
    session_id: str
    agent_session_dir: str
    
    # Process management
    pi_report: str
    pi_suggest_steps: str
    plan: List[Dict[str, Any]]
    current_step_index: int
    step_results: Dict[int, Any]
    
    # UI compatibility (for yielding partial updates)
    history: List[Dict[str, Any]]
    conversation_log: List[Dict[str, Any]]
    dialogue_memory: List[Dict[str, Any]]
    tool_executions: List[Dict[str, Any]]
    tool_cache: Dict[str, Any]
    
    # Internal research state
    research_sections: List[Dict[str, Any]]
    research_idx: int
    search_idx: int
    current_search_results: List[str]
    research_sub_reports: List[str]
    
    # Control flags
    status: str
    error: Optional[str]
    execution_failed: bool
    failed_step: Optional[int]
    failed_reason: Optional[str]
    ui_lang: Optional[str]


def _detect_ui_lang(text: str) -> str:
    """Detect user-facing language from latest user text."""
    if isinstance(text, str) and re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    return "en"


def _extract_usage_from_output(raw_output: Any) -> tuple[int, int, int, bool]:
    data = None
    if isinstance(raw_output, dict):
        data = raw_output
    elif isinstance(raw_output, str):
        text = raw_output.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    data = parsed
            except Exception:
                data = None
    if not isinstance(data, dict):
        return 0, 0, 0, True
    usage = data.get("usage")
    if isinstance(usage, dict):
        prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        total = int(usage.get("total_tokens") or (prompt + completion))
        return prompt, completion, total, False
    prompt = int(data.get("prompt_tokens") or data.get("input_tokens") or 0)
    completion = int(data.get("completion_tokens") or data.get("output_tokens") or 0)
    total = int(data.get("total_tokens") or (prompt + completion))
    if prompt == 0 and completion == 0 and total == 0:
        return 0, 0, 0, True
    return prompt, completion, total, False


def _ui_text(lang: str, key: str, **kwargs) -> str:
    zh = {
        "pipeline_title": "📋 **Pipeline**\n\n以下是执行计划：\n\n{steps}",
        "executing_step": "⏳ **Machine Learning Specialist** 正在执行第 {step_num} 步：{task_desc} …",
        "step_failed": "❌ **第 {step_num} 步失败。**\n\n",
        "step_done": "✅ **第 {step_num} 步完成。**\n\n",
        "summary": "摘要：",
        "output_preview": "输出预览：",
        "raw_output": "**原始输出（调试/自检）：**",
        "cloud_download": "📎 **云端下载：** [{name}]({url})",
        "file_preview": "**文件预览（{name}）：**",
        "generated_image": "🖼️ **生成图片：** [{name}]({url})",
        "pipeline_paused": "⛔ **流水线已暂停：** 此步骤返回错误信号（如 `status:error` / `success:false`），已跳过后续步骤以避免级联失败。\n\n",
        "summarizing": "📝 **Principal Investigator** 正在汇总结果 …",
        "final_summary_title": "## 总结\n",
        "pipeline_paused_at": "流水线在第 {step} 步暂停：{reason}\n",
        "task_completed": "任务已完成，执行情况如下：\n",
        "tool_failed": "- **{tool}**：失败（{reason}）",
        "tool_executed": "- **{tool}**：已执行",
        "task_ended": "任务已结束。详见上方结果。",
        "sub_report_failed": "(小报告生成失败：{section}，错误：{error})",
    }
    en = {
        "pipeline_title": "📋 **Pipeline**\n\nHere's what we'll do:\n\n{steps}",
        "executing_step": "⏳ **Machine Learning Specialist** is executing Step {step_num}: {task_desc} …",
        "step_failed": "❌ **Step {step_num} failed.**\n\n",
        "step_done": "✅ **Step {step_num} Complete.**\n\n",
        "summary": "Summary: ",
        "output_preview": "Output Preview: ",
        "raw_output": "**Raw output (for debugging/self-check):**",
        "cloud_download": "📎 **Cloud Download:** [{name}]({url})",
        "file_preview": "**File Preview ({name}):**",
        "generated_image": "🖼️ **Generated Image:** [{name}]({url})",
        "pipeline_paused": "⛔ **Pipeline paused:** This step returned an error signal (e.g. `status:error` / `success:false`). Downstream steps were skipped to avoid cascading failures.\n\n",
        "summarizing": "📝 **Principal Investigator** is summarizing the results …",
        "final_summary_title": "## Summary\n",
        "pipeline_paused_at": "Pipeline paused at Step {step}: {reason}\n",
        "task_completed": "Task completed. Here's what was done:\n",
        "tool_failed": "- **{tool}**: failed ({reason})",
        "tool_executed": "- **{tool}**: executed",
        "task_ended": "Task ended. See results above.",
        "sub_report_failed": "(Sub-report failed for {section}: {error})",
    }
    bundle = zh if lang == "zh" else en
    template = bundle.get(key, key)
    return template.format(**kwargs)


def _parse_sections(raw: str) -> List[Dict[str, Any]]:
    try:
        raw = raw.strip()
        if "```" in raw:
            start = raw.find("```")
            if "json" in raw[: start + 10]:
                start = raw.find("```") + 7
            else:
                start = raw.find("```") + 3
            end = raw.find("```", start)
            raw = raw[start:end if end > 0 else None].strip()
        i = raw.find("[")
        if i >= 0:
            depth = 0
            for j in range(i, len(raw)):
                if raw[j] == "[":
                    depth += 1
                elif raw[j] == "]":
                    depth -= 1
                    if depth == 0:
                        raw = raw[i : j + 1]
                        break
        sections_list = json.loads(raw)
        if not isinstance(sections_list, list): return []
        parsed = []
        for s in sections_list[:5]:
            sec_name = (s.get("section_name") or "Section").strip() or "Section"
            focus = (s.get("focus") or "both").strip() or "both"
            queries = s.get("search_queries") or s.get("search_query") or []
            if isinstance(queries, str): queries = [queries]
            elif not isinstance(queries, list): queries = [""]
            parsed.append({"section_name": sec_name, "search_queries": queries, "focus": focus})
        return parsed
    except:
        return []


def _get_chat_history_messages(chains: Dict[str, Any], history: List[Dict[str, Any]], current_input: str = "") -> List[BaseMessage]:
    """Return dialogue context for prompts: previous user turns plus final model reports only."""
    dialogue_memory = chains.get("dialogue_memory") if isinstance(chains, dict) else None
    if isinstance(dialogue_memory, list) and dialogue_memory:
        out: List[BaseMessage] = []
        current = (current_input or "").strip()
        for item in dialogue_memory[-10:]:
            if not isinstance(item, dict):
                continue
            user = str(item.get("user") or item.get("input") or "").strip()
            assistant = str(item.get("assistant") or item.get("output") or "").strip()
            if user and not (current and user == current):
                out.append(HumanMessage(content=user))
            if assistant:
                out.append(AIMessage(content=assistant))
        if out:
            return out

    memory = chains.get("memory") if isinstance(chains, dict) else None
    try:
        messages = list(memory.chat_memory.messages) if memory is not None else []
    except Exception:
        messages = []
    if messages:
        return messages

    current = (current_input or "").strip()
    out: List[BaseMessage] = []
    for item in list(history or [])[-20:]:
        role = str(item.get("role") or "").strip().lower()
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        if role == "user" and current and content == current:
            continue
        if role == "user":
            out.append(HumanMessage(content=content))
    return out


def _format_conversation_history(chains: Dict[str, Any], history: List[Dict[str, Any]], current_input: str = "", limit: int = 10) -> str:
    rows = []
    for msg in _get_chat_history_messages(chains, history, current_input)[-limit:]:
        content = str(getattr(msg, "content", "") or "").strip()
        if not content:
            continue
        role = str(getattr(msg, "type", "") or getattr(msg, "role", "") or "").strip().lower()
        if not role:
            name = type(msg).__name__.lower()
            role = "user" if "human" in name else "assistant" if "ai" in name else "message"
        if role == "human":
            role = "user"
        elif role == "ai":
            role = "assistant"
        rows.append(f"{role}: {content}")
    return "\n".join(rows) if rows else "No previous conversation."


def _normalize_tool_input(raw_input: Any) -> Dict[str, Any]:
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


def _normalize_step_number(raw_step: Any, fallback: int) -> int:
    try:
        if isinstance(raw_step, str):
            match = re.search(r"\d+", raw_step)
            if match:
                return int(match.group(0))
        if raw_step is not None:
            return int(raw_step)
    except Exception:
        pass
    return fallback


def _get_step_raw_output(step_results: Optional[Dict[Any, Any]], step_no: Any) -> Any:
    if not isinstance(step_results, dict):
        return None
    normalized = _normalize_step_number(step_no, -1)
    for key in (normalized, str(normalized), f"step_{normalized}", f"step{normalized}"):
        item = step_results.get(key)
        if isinstance(item, dict) and "raw_output" in item:
            return item.get("raw_output")
    return None


def _get_tool_allowed_param_names(tool: Any) -> Optional[set]:
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


def _sanitize_tool_invoke_input(
    tool_name: str,
    tool: Any,
    merged_input: Dict[str, Any],
    agent_session_dir: str = "",
    step_results: Optional[Dict[int, Any]] = None,
) -> Dict[str, Any]:
    """Filter merged input by tool schema to avoid strict parameter-validation failures."""
    if not isinstance(merged_input, dict):
        return _normalize_tool_input(merged_input)

    allowed = _get_tool_allowed_param_names(tool)
    if not allowed:
        return merged_input

    filtered = {k: v for k, v in merged_input.items() if k in allowed}

    # If planner produced a scalar-like wrapper, map it to the only accepted field.
    if not filtered and len(allowed) == 1:
        only_key = next(iter(allowed))
        mapped_from = None
        if "input" in merged_input:
            filtered[only_key] = merged_input.get("input")
            mapped_from = "input"
        elif "items" in merged_input:
            filtered[only_key] = merged_input.get("items")
            mapped_from = "items"
        elif tool_name == "python_repl" and only_key == "query":
            for alias in ("code", "script", "python", "source"):
                if alias in merged_input and merged_input.get(alias) not in (None, ""):
                    filtered[only_key] = merged_input.get(alias)
                    mapped_from = alias
                    break

        # Generic fallback for single-arg tools:
        # pick the first non-empty value from non-output/path-like keys.
        if not filtered:
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
                filtered[only_key] = value
                mapped_from = str(key)
                break

        if filtered and mapped_from:
            print(f"[Input sanitize] tool={tool_name} | mapped `{mapped_from}` -> `{only_key}`")

    def _coerce_sequence(value: Any) -> Optional[str]:
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
                    nested_seq = _coerce_sequence(nested)
                    if nested_seq:
                        return nested_seq
        if isinstance(value, list) and value:
            for item in value:
                seq = _coerce_sequence(item)
                if seq:
                    return seq
        return None

    if "sequence" in allowed:
        seq = _coerce_sequence(filtered.get("sequence"))
        if not seq:
            seq = _coerce_sequence(merged_input.get("sequence"))
        if not seq:
            seq = _coerce_sequence(merged_input.get("last_sequence"))
        if seq:
            filtered["sequence"] = seq

    _project_root = get_project_root().resolve()
    _session_root = Path(agent_session_dir).expanduser().resolve() if agent_session_dir else None

    def _rewrite_python_query_paths(query: Any) -> Any:
        """Rewrite quoted basename references in python_repl query to absolute paths."""
        if tool_name != "python_repl" or not isinstance(query, str):
            return query

        def _resolve_dependency_token(token: str) -> Optional[str]:
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
            dep_raw = _get_step_raw_output(step_results, dep_step)
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
                resolved = _maybe_resolve_local_path(val)
                return resolved if isinstance(resolved, str) else val
            if isinstance(val, dict):
                if "file_path" in val and isinstance(val.get("file_path"), str):
                    resolved = _maybe_resolve_local_path(val["file_path"])
                    return resolved if isinstance(resolved, str) else val["file_path"]
                if isinstance(val.get("file_info"), dict) and isinstance(val["file_info"].get("file_path"), str):
                    resolved = _maybe_resolve_local_path(val["file_info"]["file_path"])
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

        candidate_paths: List[str] = []
        for key, value in merged_input.items():
            key_l = str(key).lower()
            if isinstance(value, str) and any(tok in key_l for tok in ("path", "file", "dir")):
                candidate_paths.append(value)
            if key_l == "last_file" and isinstance(value, str):
                candidate_paths.append(value)
            if key_l == "files" and isinstance(value, list):
                candidate_paths.extend([v for v in value if isinstance(v, str)])
        if isinstance(step_results, dict) and step_results:
            for step_no in sorted(step_results.keys(), key=lambda x: _normalize_step_number(x, 0), reverse=True):
                raw_output = _get_step_raw_output(step_results, step_no)
                if raw_output is None:
                    continue
                extracted = _get_output_file_path_from_raw(raw_output, "dependency_step")
                if extracted:
                    candidate_paths.append(extracted)

        basename_to_abs: Dict[str, str] = {}
        for raw_path in candidate_paths:
            resolved = _maybe_resolve_local_path(raw_path)
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

    def _maybe_resolve_local_path(raw: Any) -> Any:
        if not isinstance(raw, str):
            return raw
        text = raw.strip()
        if not text:
            return raw
        candidate = Path(text).expanduser()
        if candidate.is_absolute():
            return str(candidate.resolve()) if candidate.exists() else raw

        for root in (_project_root, _session_root, Path.cwd().resolve()):
            if root is None:
                continue
            resolved = (root / candidate).resolve()
            if resolved.exists():
                return str(resolved)

        # If only basename is provided, try locating it under current session output tree.
        if _session_root is not None and len(candidate.parts) == 1:
            try:
                matched = next(_session_root.rglob(candidate.name), None)
                if matched and matched.exists():
                    return str(matched.resolve())
            except Exception:
                pass
        return raw

    for key, value in list(filtered.items()):
        key_l = str(key).lower()
        if any(tok in key_l for tok in ("path", "file", "dir")):
            filtered[key] = _maybe_resolve_local_path(value)
    if tool_name == "python_repl" and "query" in filtered:
        filtered["query"] = _rewrite_python_query_paths(filtered["query"])

    if not filtered and merged_input:
        print(f"[Input sanitize] tool={tool_name} | kept none of {list(merged_input.keys())} by allowed={sorted(list(allowed))}")
    return filtered


def _is_write_like_tool(tool_name: str) -> bool:
    if not tool_name:
        return False
    if tool_name.startswith("download_"):
        return True
    write_prefixes = (
        "predict_structure_",
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
    invoke_input: Dict[str, Any],
    agent_session_dir: str,
) -> Dict[str, Any]:
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
                print(f"[Output normalize] unsafe absolute output path `{raw}`; fallback to `{fallback}`")
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
            print(f"[Output normalize] output path escapes session `{raw}`; fallback to `{fallback}`")
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


def _collect_output_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    keys = ("out_dir", "output_dir", "out_path", "output_file")
    return {k: data.get(k) for k in keys if k in data}


def _extract_skill_ids_from_metadata(skills_metadata: str) -> List[str]:
    if not isinstance(skills_metadata, str) or not skills_metadata.strip():
        return []
    ids = re.findall(r"skill_id:\s*`([^`]+)`", skills_metadata)
    seen = set()
    out: List[str] = []
    for s in ids:
        sid = str(s).strip()
        if not sid or sid in seen:
            continue
        seen.add(sid)
        out.append(sid)
    return out


def _pick_skill_for_code_step(task_desc: str, available_skill_ids: List[str]) -> Optional[str]:
    if not available_skill_ids:
        return None
    lower = (task_desc or "").lower()
    preferred: List[str] = []
    if any(k in lower for k in ("plot", "figure", "visual", "chart", "draw")):
        preferred = ["matplotlib", "seaborn", "biopython"]
    elif any(k in lower for k in ("fasta", "sequence", "pdb", "structure", "mutation")):
        preferred = ["biopython", "matplotlib", "seaborn"]
    else:
        preferred = ["biopython", "matplotlib", "seaborn"]
    for sid in preferred:
        if sid in available_skill_ids:
            return sid
    return available_skill_ids[0]


def _enforce_skill_first_plan(
    normalized_plan: List[Dict[str, Any]],
    available_tools_list: str,
    skills_metadata: str,
    ui_lang: str,
) -> List[Dict[str, Any]]:
    if not normalized_plan:
        return normalized_plan
    available_tools = {t.strip() for t in str(available_tools_list or "").split(",") if t.strip()}
    if "read_skill" not in available_tools:
        return normalized_plan

    skill_ids = _extract_skill_ids_from_metadata(skills_metadata)
    if not skill_ids:
        return normalized_plan

    code_tools = {"python_repl", "agent_generated_code"}
    existing_steps = [p.get("step") for p in normalized_plan if isinstance(p.get("step"), int)]
    next_aux_step = (max(existing_steps) + 1) if existing_steps else 1
    enforced: List[Dict[str, Any]] = []

    for p in normalized_plan:
        tname = str(p.get("tool_name") or "").strip()
        if tname in code_tools:
            prev_is_read_skill = bool(enforced) and str(enforced[-1].get("tool_name") or "").strip() == "read_skill"
            if not prev_is_read_skill:
                chosen_skill = _pick_skill_for_code_step(str(p.get("task_description") or ""), skill_ids)
                if chosen_skill:
                    enforced.append(
                        {
                            "step": next_aux_step,
                            "task_description": (
                                f"加载技能 `{chosen_skill}`，为后续代码执行提供规范接口与参数参考。"
                                if ui_lang == "zh"
                                else f"Load skill `{chosen_skill}` to provide API and parameter guidance for the next code step."
                            ),
                            "tool_name": "read_skill",
                            "tool_input": {"skill_id": chosen_skill},
                        }
                    )
                    next_aux_step += 1
        enforced.append(p)
    return enforced


async def research_plan_start_node(state: AgentState, config: RunnableConfig):
    """Initial node - passes through to research_plan_node for decision."""
    # Just pass through, research_plan_node will decide the path
    return {"status": "analyzing"}


async def research_plan_node(state: AgentState, config: RunnableConfig):
    """PI phase 1: Create a research plan with sections."""
    chains = config.get("configurable", {}).get("chains", {})
    protein_ctx = state["protein_context"]
    text = state["messages"][-1].content
    ui_lang = _detect_ui_lang(text)
    protein_context_summary = protein_ctx.get_context_summary()
    history = list(state.get("history", []))
    conversation_history = _format_conversation_history(chains, history, text)

    try:
        sections_out = await asyncio.to_thread(
            chains["pi_sections"].invoke,
            {
                "input": text,
                "protein_context_summary": protein_context_summary,
                "conversation_history": conversation_history,
            },
        )
        sections_list = _parse_sections(sections_out)
    except Exception as e:
        print(f"[PI sections] failed: {e}")
        sections_list = []

    # PI_SECTIONS_PROMPT is the routing authority here: [] means answer directly in chat mode,
    # without handing the turn to CB/MLS.
    if not sections_list:
        return {"status": "chat_mode", "pi_report": "", "pi_suggest_steps": "", "ui_lang": ui_lang}

    # This is a research request - show the analyzing message
    history.append({
        "role": "assistant",
        "content": "🤔 **Principal Investigator** 正在分析你的请求并制定研究计划..."
        if ui_lang == "zh" else
        "🤔 **Principal Investigator** is analyzing your request and creating a research plan...",
        "role_id": "principal_investigator",
    })

    return {
        "research_sections": sections_list,
        "research_idx": 0,
        "search_idx": 0,
        "current_search_results": [],
        "research_sub_reports": [],
        "history": history,
        "ui_lang": ui_lang,
        "status": "research_planning_done"
    }


async def chat_start_node(state: AgentState, config: RunnableConfig):
    """Show 'PI is responding' for simple chat/greeting inputs."""
    history = list(state.get("history", []))
    ui_lang = state.get("ui_lang") or _detect_ui_lang(state["messages"][-1].content)
    history.append({
        "role": "assistant",
        "content": "🤔 思考中..." if ui_lang == "zh" else "🤔 Thinking...",
        "role_id": "principal_investigator",
    })
    return {"history": history, "ui_lang": ui_lang}


async def chat_node(state: AgentState, config: RunnableConfig):
    """Chat mode: use pi_chat_chain for direct responses (greetings, simple questions)."""
    chains = config.get("configurable", {}).get("chains", {})
    text = state["messages"][-1].content
    ui_lang = state.get("ui_lang") or _detect_ui_lang(text)
    history = list(state.get("history", []))

    try:
        chat_history = _get_chat_history_messages(chains, history, text)
        # Use pi_chat chain for direct chat response (greetings, simple questions)
        response = await asyncio.to_thread(
            chains["pi_chat"].invoke,
            {"input": text, "chat_history": chat_history},
        )
        # Remove the "Thinking..." placeholder before adding the actual response
        if history and ("Thinking" in history[-1].get("content", "") or "思考中" in history[-1].get("content", "")):
            history.pop()
        history.append({"role": "assistant", "content": response, "role_id": "principal_investigator"})
    except Exception as e:
        # Remove the "Thinking..." placeholder before adding the error
        if history and ("Thinking" in history[-1].get("content", "") or "思考中" in history[-1].get("content", "")):
            history.pop()
        err_msg = f"抱歉，我遇到了错误：{str(e)}" if ui_lang == "zh" else f"I apologize, but I encountered an error: {str(e)}"
        history.append({"role": "assistant", "content": err_msg, "role_id": "principal_investigator"})

    return {"history": history, "status": "completed"}


async def research_search_start_node(state: AgentState, config: RunnableConfig):
    """Show 'PI is searching' so UI updates before search runs."""
    research_idx = state.get("research_idx", 0)
    search_idx = state.get("search_idx", 0)
    sections = state.get("research_sections", [])
    history = list(state.get("history", []))
    ui_lang = state.get("ui_lang") or _detect_ui_lang(state["messages"][-1].content)
    if research_idx >= len(sections):
        return {"status": "research_steps_done"}
    section = sections[research_idx]
    queries = section.get("search_queries", [])
    if search_idx >= len(queries):
        return {}
    sq = (queries[search_idx] or "").strip()[:80]
    if sq and len(sq) == 80:
        sq = sq + "…"
    history.append({
        "role": "assistant",
        "content": f"🔍 **Principal Investigator** 正在检索：**{sq or '…'}** …"
        if ui_lang == "zh" else
        f"🔍 **Principal Investigator** is searching: **{sq or '…'}** …",
        "role_id": "principal_investigator",
    })
    return {"history": history}


async def research_search_node(state: AgentState, config: RunnableConfig):
    """PI phase 2a: Process ONE search query from the current section."""
    chains = config.get("configurable", {}).get("chains", {})
    protein_ctx = state["protein_context"]
    research_idx = state.get("research_idx", 0)
    search_idx = state.get("search_idx", 0)
    sections = state.get("research_sections", [])
    history = list(state.get("history", []))
    ui_lang = state.get("ui_lang") or _detect_ui_lang(state["messages"][-1].content)
    executions = list(state.get("tool_executions", []))
    current_search_results = list(state.get("current_search_results", []))
    
    if research_idx >= len(sections) or len(protein_ctx.tool_history) >= AGENT_CHAT_MAX_TOOL_CALLS:
        return {"status": "research_steps_done"}

    section = sections[research_idx]
    queries = section.get("search_queries", [])
    
    if search_idx == 0:
        section_title = f"**第 {research_idx + 1} 节：** {section['section_name']}" if ui_lang == "zh" else f"**Section {research_idx + 1}:** {section['section_name']}"
        history.append({"role": "assistant", "content": section_title, "role_id": "principal_investigator"})

    if search_idx < len(queries):
        sq = queries[search_idx]
        search_results_list, search_logged = await asyncio.to_thread(_run_section_search, sq)
        current_search_results.extend(search_results_list)
        
        for tname, tinputs, toutputs in search_logged:
            step_off = len(protein_ctx.tool_history) + 1
            protein_ctx.add_tool_call(step_off, tname, tinputs, toutputs, cached=False)
            executions.append({
                "step": step_off, "tool_name": tname, "inputs": tinputs,
                "outputs": (str(toutputs)[:1000] + "..." if len(str(toutputs)) > 1000 else str(toutputs)),
                "timestamp": datetime.now().isoformat(),
            })
            summary_msg = _format_search_summary(tname, tinputs, str(toutputs))
            preview = _format_search_preview(tname, str(toutputs))
            history.append({"role": "assistant", "content": summary_msg + ("\n\n" + preview if preview else ""), "role_id": "principal_investigator"})
    
    return {
        "history": history,
        "tool_executions": executions,
        "current_search_results": current_search_results,
        "search_idx": search_idx + 1,
        "status": "research_search_done"
    }


async def research_sub_report_start_node(state: AgentState, config: RunnableConfig):
    """PI phase 2b (start): Append 'PI is writing sub-report' so UI updates before LLM runs."""
    research_idx = state.get("research_idx", 0)
    sections = state.get("research_sections", [])
    history = list(state.get("history", []))
    ui_lang = state.get("ui_lang") or _detect_ui_lang(state["messages"][-1].content)
    if research_idx >= len(sections):
        return {"status": "research_steps_done"}
    section = sections[research_idx]
    history.append({
        "role": "assistant",
        "content": f"✍️ **Principal Investigator** 正在撰写小报告：**{section['section_name']}** …"
        if ui_lang == "zh" else
        f"✍️ **Principal Investigator** is writing the sub-report for: **{section['section_name']}** …",
        "role_id": "principal_investigator",
    })
    return {"history": history}


async def research_sub_report_node(state: AgentState, config: RunnableConfig):
    """PI phase 2b: Generate sub-report for the current section after searches are done."""
    chains = config.get("configurable", {}).get("chains", {})
    research_idx = state.get("research_idx", 0)
    sections = state.get("research_sections", [])
    history = list(state.get("history", []))
    ui_lang = state.get("ui_lang") or _detect_ui_lang(state["messages"][-1].content)
    current_search_results = state.get("current_search_results", [])
    sub_reports = list(state.get("research_sub_reports", []))
    
    if research_idx >= len(sections):
        return {"status": "research_steps_done"}

    section = sections[research_idx]
    # Join all collected results from all queries in this section with sequential numbering [1], [2], ...
    formatted_refs = []
    for i, res_item in enumerate(current_search_results, 1):
        formatted_refs.append(f"[{i}] {res_item}")
    
    search_results_str = "\n\n".join(formatted_refs) if formatted_refs else ("该小节没有检索结果。" if ui_lang == "zh" else "No search results for this section.")

    try:
        sub_report = await asyncio.to_thread(
            chains["pi_sub_report"].invoke,
            {"section_name": section["section_name"], "focus": section["focus"], "search_results": search_results_str},
        )
        title, body = _parse_sub_report_short_title((sub_report or "").strip(), fallback_title=section["section_name"])
        history.append({"role": "assistant", "content": f"# {title}\n\n{body}", "role_id": "principal_investigator"})
        sub_reports.append(f"### {title}\n\n**References:**\n{search_results_str}\n\n**Sub-report:**\n{body}")
    except Exception as e:
        sub_reports.append(_ui_text(ui_lang, "sub_report_failed", section=section["section_name"], error=str(e)))

    return {
        "history": history,
        "research_sub_reports": sub_reports,
        "research_idx": research_idx + 1,
        "search_idx": 0,
        "current_search_results": [],
        "status": "research_step_done"
    }


async def research_report_start_node(state: AgentState, config: RunnableConfig):
    """Show 'PI is writing draft report' so UI updates before LLM runs."""
    history = list(state.get("history", []))
    ui_lang = state.get("ui_lang") or _detect_ui_lang(state["messages"][-1].content)
    history.append({
        "role": "assistant",
        "content": "✍️ **Principal Investigator** 正在撰写研究草案（摘要、引言、相关工作、参考文献）…"
        if ui_lang == "zh" else
        "✍️ **Principal Investigator** is writing the draft report (Abstract, Introduction, Related Work, References) …",
        "role_id": "principal_investigator",
    })
    return {"history": history}


async def research_report_node(state: AgentState, config: RunnableConfig):
    """PI phase 3: Aggregate sub-reports into final report and suggest steps."""
    chains = config.get("configurable", {}).get("chains", {})
    sub_reports_text = "\n\n".join(state.get("research_sub_reports", []))
    text = state["messages"][-1].content
    ui_lang = state.get("ui_lang") or _detect_ui_lang(text)
    history = list(state.get("history", []))

    try:
        final_report = await asyncio.to_thread(
            chains["pi_final_report"].invoke,
            {"input": text, "sub_reports": sub_reports_text},
        )
        history.append({"role": "assistant", "content": final_report, "role_id": "principal_investigator"})
    except Exception as e:
        final_report = (
            f"生成最终研究报告失败：{e}\n\n{sub_reports_text}"
            if ui_lang == "zh"
            else f"Failed to generate final report: {e}\n\n{sub_reports_text}"
        )
        history.append({"role": "assistant", "content": final_report, "role_id": "principal_investigator"})

    try:
        suggest_steps = await asyncio.to_thread(
            chains["pi_suggest_steps"].invoke,
            {"draft_report": final_report, "input": text},
        )
    except:
        suggest_steps = "执行基础分析。" if ui_lang == "zh" else "Execute basic analysis."

    return {
        "pi_report": final_report,
        "pi_suggest_steps": suggest_steps,
        "history": history,
        "status": "researched"
    }


async def plan_start_node(state: AgentState, config: RunnableConfig):
    """Show 'CB is designing pipeline' so UI updates before LLM runs."""
    history = list(state.get("history", []))
    ui_lang = state.get("ui_lang") or _detect_ui_lang(state["messages"][-1].content)
    history.append({
        "role": "assistant",
        "content": "📋 **Computational Biologist** 正在设计流程 …"
        if ui_lang == "zh" else
        "📋 **Computational Biologist** is designing the pipeline …",
        "role_id": "computational_biologist",
    })
    return {"history": history}


async def plan_node(state: AgentState, config: RunnableConfig):
    """CB planning node: PI report -> pipeline (JSON list)."""
    chains = config.get("configurable", {}).get("chains", {})
    pi_report = state.get("pi_report", "")
    pi_suggest_steps = state.get("pi_suggest_steps", "")
    protein_ctx = state["protein_context"]
    history = list(state.get("history", []))
    log_entries = list(state.get("conversation_log", []))
    ui_lang = state.get("ui_lang") or _detect_ui_lang(state["messages"][-1].content)

    # When research is skipped, use user's original input as the "PI report"
    if not pi_report:
        user_input = state["messages"][-1].content
        if ui_lang == "zh":
            pi_report = f"用户请求：{user_input}\n\n无需文献检索，直接进入工具执行。"
            pi_suggest_steps = pi_suggest_steps or "执行适当工具以完成用户请求。"
        else:
            pi_report = f"User request: {user_input}\n\nNo literature research needed. Proceed directly with tool execution."
            pi_suggest_steps = pi_suggest_steps or "Execute the appropriate tool to fulfill the user's request."

    context_parts = [f"蛋白上下文：{protein_ctx.get_context_summary()}" if ui_lang == "zh" else f"Protein context: {protein_ctx.get_context_summary()}"]
    if state.get("agent_session_dir"):
        context_parts.append(
            f"默认输出目录：{to_project_relative_path(state['agent_session_dir'])}"
            if ui_lang == "zh"
            else f"Default output directory: {to_project_relative_path(state['agent_session_dir'])}"
        )
    protein_context_summary = "; ".join(context_parts)

    recent_tool_calls = protein_ctx.get_tool_records(limit=10)
    tool_outputs_summary = json.dumps(recent_tool_calls, ensure_ascii=False)

    # Pass tools and skills explicitly so CB always sees the full list (session_state may be the only source)
    tools_description = chains.get("tools_description") or ""
    skills_metadata = chains.get("skills_metadata") or get_skills_metadata_string()
    available_tools_list = chains.get("available_tools_list") or ""
    if not available_tools_list and chains.get("workers"):
        available_tools_list = ", ".join(chains["workers"].keys())

    cb_planner_inputs = {
        "pi_report": pi_report,
        "pi_suggest_steps": pi_suggest_steps,
        "protein_context_summary": protein_context_summary,
        "tool_outputs": tool_outputs_summary,
        "tools_description": tools_description,
        "skills_metadata": skills_metadata,
        "available_tools_list": available_tools_list,
    }

    try:
        raw_msg = await asyncio.to_thread(chains["cb_planner_raw"].invoke, cb_planner_inputs)
        content = getattr(raw_msg, "content", None) or str(raw_msg) or ""
        plan = _parse_cb_plan(content)
    except Exception as e:
        print(f"[CB planner] failed: {e}")
        plan = []

    # Filter and normalize plan
    normalized_plan = []
    for i, p in enumerate(plan):
        if not isinstance(p, dict): continue
        tname = p.get("tool_name") or p.get("tool") or ""
        if not tname or tname in PI_SEARCH_TOOL_NAMES: continue
        normalized_plan.append({
            "step": _normalize_step_number(p.get("step"), i + 1),
            "task_description": p.get("task_description") or p.get("task") or "",
            "tool_name": tname.strip(),
            "tool_input": _normalize_tool_input(p.get("tool_input") or p.get("input"))
        })

    normalized_plan = _enforce_skill_first_plan(normalized_plan, available_tools_list, skills_metadata, ui_lang)

    if not normalized_plan:
        # No tools needed - this might be a greeting or simple question
        # Route to chat mode for a natural response
        return {"plan": [], "history": history, "status": "chat_mode"}

    step_lines = [
        (f"**第 {p['step']} 步。** {p['task_description']}" if ui_lang == "zh" else f"**Step {p['step']}.** {p['task_description']}")
        for p in normalized_plan
    ]
    plan_text = _ui_text(ui_lang, "pipeline_title", steps="\n\n".join(step_lines))
    history.append({"role": "assistant", "content": plan_text, "role_id": "computational_biologist"})
    log_entries.append({"role": "assistant", "content": plan_text, "role_id": "computational_biologist", "timestamp": datetime.now().isoformat()})

    return {
        "plan": normalized_plan,
        "current_step_index": 0,
        "step_results": {},
        "history": history,
        "conversation_log": log_entries,
        "execution_failed": False,
        "failed_step": None,
        "failed_reason": None,
        "ui_lang": ui_lang,
        "status": "planned"
    }


async def execute_start_node(state: AgentState, config: RunnableConfig):
    """Show 'MLS is executing step N' so UI updates before tool runs."""
    plan = state.get("plan", [])
    idx = state.get("current_step_index", 0)
    history = list(state.get("history", []))
    ui_lang = state.get("ui_lang") or _detect_ui_lang(state["messages"][-1].content)
    if idx >= len(plan):
        return {}
    step = plan[idx]
    step_num = _normalize_step_number(step.get("step"), idx + 1)
    task_desc = step.get("task_description", "…")
    history.append({
        "role": "assistant",
        "content": _ui_text(ui_lang, "executing_step", step_num=step_num, task_desc=task_desc),
        "role_id": "machine_learning_specialist",
    })
    return {"history": history}


async def execute_node(state: AgentState, config: RunnableConfig):
    """MLS execution node: executes current step in plan."""
    chains = config.get("configurable", {}).get("chains", {})
    plan = state["plan"]
    idx = state["current_step_index"]
    step = plan[idx]
    protein_ctx = state["protein_context"]
    history = list(state.get("history", []))
    log_entries = list(state.get("conversation_log", []))
    executions = list(state.get("tool_executions", []))
    step_results = dict(state.get("step_results", {}))
    ui_lang = state.get("ui_lang") or _detect_ui_lang(state["messages"][-1].content)

    step_num = _normalize_step_number(step.get("step"), idx + 1)
    task_desc = step["task_description"]
    tool_name = step["tool_name"]
    tool_input = step["tool_input"]
    disabled_tool_names = set(chains.get("disabled_tool_names") or [])

    # Resolve dependencies
    merged_tool_input = _merge_tool_parameters_with_context(protein_ctx, tool_input)
    dependency_resolution_failed = False
    dependency_failure_reason = ""
    for key, value in list(merged_tool_input.items()):
        if isinstance(value, str) and value.startswith("dependency:"):
            parts = value.split(":")
            if len(parts) < 2:
                dependency_failure_reason = f"Invalid dependency token for `{key}`: {value}"
                print(f"[Dependency resolve] {dependency_failure_reason}")
                dependency_resolution_failed = True
                continue
            dep_token = parts[1].replace("step_", "").replace("step", "").strip()
            try:
                dep_step = int(dep_token)
            except ValueError:
                dependency_failure_reason = f"Invalid dependency step token for `{key}`: {value}"
                print(f"[Dependency resolve] {dependency_failure_reason}")
                dependency_resolution_failed = True
                continue

            dep_out = _get_step_raw_output(step_results, dep_step)
            if dep_out is None:
                dependency_failure_reason = f"Missing output for dependency step {dep_step} (key={key})"
                print(f"[Dependency resolve] {dependency_failure_reason}")
                dependency_resolution_failed = True
                continue

            dep_failed, dep_reason = _tool_output_indicates_failure(dep_out)
            if dep_failed:
                dependency_failure_reason = (
                    f"Dependency step {dep_step} failed"
                    + (f": {dep_reason}" if dep_reason else "")
                    + f" (needed for `{key}`)"
                )
                print(f"[Dependency resolve] {dependency_failure_reason}")
                dependency_resolution_failed = True
                continue

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
                    print(
                        f"[Dependency resolve] field path `{'/'.join(field_path)}` not found in step {dep_step} output; using raw output"
                    )
            else:
                val = dep_out

            # Heuristic auto-extraction for paths if the expected parameter is a file or path
            if any(k in key.lower() for k in ("path", "file")):
                if isinstance(val, dict):
                    if "file_path" in val:
                        val = val["file_path"]
                    elif "file_info" in val and isinstance(val["file_info"], dict) and "file_path" in val["file_info"]:
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

    # Execution Loop with Retries
    execute_started = time.time()
    step_retry = 0
    step_done = False
    last_output = None
    cached_flag = False  # Default value
    failure_reason = ""

    if dependency_resolution_failed:
        failure_reason = dependency_failure_reason or "Dependency resolution failed."
        last_output = json.dumps(
            {
                "status": "error",
                "error": {"type": "DependencyResolutionError", "message": failure_reason},
                "file_info": None,
            },
            ensure_ascii=False,
        )
        step_done = True

    # Hard precondition: code tools require at least one successful read_skill step beforehand.
    if tool_name in {"python_repl", "agent_generated_code"}:
        has_successful_skill = False
        for prev in plan[:idx]:
            if str(prev.get("tool_name") or "").strip() != "read_skill":
                continue
            prev_step = _normalize_step_number(prev.get("step"), 0)
            prev_raw = _get_step_raw_output(step_results, prev_step)
            if prev_raw is None:
                continue
            prev_failed, _prev_reason = _tool_output_indicates_failure(prev_raw)
            if not prev_failed:
                has_successful_skill = True
                break
        if not has_successful_skill:
            failure_reason = (
                "代码执行步骤前缺少成功的 read_skill 步骤；请先由 CB 规划并执行 read_skill。"
                if ui_lang == "zh"
                else "Missing a successful read_skill step before code execution; CB must plan and run read_skill first."
            )
            last_output = json.dumps(
                {"status": "error", "error": {"type": "SkillPreconditionFailed", "message": failure_reason}},
                ensure_ascii=False,
            )
            step_done = True

    if tool_name in disabled_tool_names:
        last_output = json.dumps(
            {
                "success": False,
                "error": f"Tool `{tool_name}` is disabled in online mode.",
                "detail": "Training and protein-discovery tools are unavailable in online mode.",
            },
            ensure_ascii=False,
        )
        step_done = True

    # Get the actual tool for direct invocation
    tool = next((t for t in chains["all_tools"] if t.name == tool_name), None)
    agent_session_dir = state.get("agent_session_dir") or ""
    invoke_input = (
        _sanitize_tool_invoke_input(tool_name, tool, merged_tool_input, agent_session_dir, step_results)
        if tool else merged_tool_input
    )
    raw_output_fields = _collect_output_fields(invoke_input)
    if tool and agent_session_dir:
        invoke_input = _normalize_output_paths(tool_name, tool, invoke_input, agent_session_dir)
    normalized_output_fields = _collect_output_fields(invoke_input)
    if raw_output_fields or normalized_output_fields:
        print(
            f"[Output normalize] tool={tool_name} | raw={json.dumps(raw_output_fields, ensure_ascii=False)}"
            f" -> normalized={json.dumps(normalized_output_fields, ensure_ascii=False)}"
        )

    def _merge_retry_input(base_input: Dict[str, Any], retry_input: Dict[str, Any]) -> Dict[str, Any]:
        candidate = dict(base_input)
        candidate.update(retry_input)
        if tool:
            candidate = _sanitize_tool_invoke_input(tool_name, tool, candidate, agent_session_dir, step_results)
            if agent_session_dir:
                candidate = _normalize_output_paths(tool_name, tool, candidate, agent_session_dir)
        return candidate

    def _extract_output_artifact_paths(raw_output: Any) -> List[str]:
        paths: List[str] = []
        primary = _get_output_file_path_from_raw(raw_output, tool_name)
        if primary:
            paths.append(primary)
        try:
            parsed = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
            if isinstance(parsed, dict):
                direct = parsed.get("file_path")
                if isinstance(direct, str) and direct.strip():
                    resolved = _get_output_file_path_from_raw(json.dumps({"file_path": direct}, ensure_ascii=False), tool_name)
                    if resolved:
                        paths.append(resolved)
                info = parsed.get("file_info")
                if isinstance(info, dict):
                    fp = info.get("file_path")
                    if isinstance(fp, str) and fp.strip():
                        resolved = _get_output_file_path_from_raw(json.dumps({"file_info": {"file_path": fp}}, ensure_ascii=False), tool_name)
                        if resolved:
                            paths.append(resolved)
        except Exception:
            pass

        unique: List[str] = []
        seen = set()
        for p in paths:
            norm = os.path.abspath(p)
            if norm in seen:
                continue
            seen.add(norm)
            unique.append(norm)
        return unique

    def _register_artifacts_to_context(raw_output: Any):
        for file_path in _extract_output_artifact_paths(raw_output):
            try:
                ext = os.path.splitext(file_path)[1].lower()
                if ext in {".pdb", ".cif", ".mmcif"}:
                    protein_ctx.add_structure_file(file_path, source=tool_name, uniprot_id=protein_ctx.last_uniprot_id)
                protein_ctx.add_file(file_path)
            except Exception as e:
                print(f"[Artifact register] failed for {file_path}: {e}")

    while step_retry <= MAX_STEP_RETRIES and not step_done:
        # Cache check
        cached = get_cached_tool_result({"tool_cache": state.get("tool_cache", {})}, tool_name, invoke_input)
        if cached:
            raw_output = cached["outputs"]
            cached_flag = True
        elif tool:
            # Direct tool invocation (simpler and more reliable)
            try:
                inputs_str = json.dumps(invoke_input, ensure_ascii=False, sort_keys=True)
                if len(inputs_str) > 500:
                    inputs_str = inputs_str[:500] + "..."
                print(f"[Execute] tool={tool_name} | input={inputs_str}")
                out = await asyncio.wait_for(asyncio.to_thread(tool.invoke, invoke_input), timeout=TOOL_EXECUTION_TIMEOUT)
                raw_output = out if isinstance(out, (str, dict)) else str(out)
                out_preview = str(raw_output)[:300] + ("..." if len(str(raw_output)) > 300 else "")
                print(f"[Result] tool={tool_name} | output_preview={out_preview}")
                cached_flag = False
            except asyncio.TimeoutError:
                raw_output = json.dumps(
                    {"success": False, "error": f"Tool execution timed out ({TOOL_EXECUTION_TIMEOUT}s)"},
                    ensure_ascii=False,
                )
                print(f"[Result] tool={tool_name} | timeout after {TOOL_EXECUTION_TIMEOUT}s")
                cached_flag = False
            except Exception as e:
                raw_output = json.dumps({"success": False, "error": str(e)})
                print(f"[Result] tool={tool_name} | exception={e}")
                cached_flag = False
        else:
            # Tool not found
            last_output = json.dumps({"success": False, "error": f"Unknown tool: {tool_name}"})
            step_done = True
            cached_flag = False
            break

        is_failure, failure_reason = _tool_output_indicates_failure(raw_output)

        # Compat fallback: one-shot path rebinding retry for python_repl FileNotFoundError.
        if (
            tool_name == "python_repl"
            and isinstance(raw_output, str)
            and "FileNotFoundError" in raw_output
            and step_retry < MAX_STEP_RETRIES
        ):
            candidate_paths = []
            try:
                candidate_paths.extend(
                    [f.get("path") for f in protein_ctx.files.values() if isinstance(f, dict) and isinstance(f.get("path"), str)]
                )
            except Exception:
                pass
            for dep_step in sorted(step_results.keys(), key=lambda x: _normalize_step_number(x, 0), reverse=True):
                dep_raw = _get_step_raw_output(step_results, dep_step)
                if dep_raw is None:
                    continue
                dep_path = _get_output_file_path_from_raw(dep_raw, "dependency_step")
                if dep_path:
                    candidate_paths.append(dep_path)
            candidate_paths = sorted(list({os.path.abspath(p) for p in candidate_paths if isinstance(p, str) and p.strip()}))
            if candidate_paths:
                retry_seed = dict(invoke_input)
                retry_seed.setdefault("files", [])
                if isinstance(retry_seed.get("files"), list):
                    retry_seed["files"] = sorted(list({*retry_seed["files"], *candidate_paths}))
                if not retry_seed.get("last_file"):
                    retry_seed["last_file"] = candidate_paths[-1]
                rebound_input = _sanitize_tool_invoke_input(
                    tool_name,
                    tool,
                    retry_seed,
                    agent_session_dir,
                    step_results,
                )
                new_query = rebound_input.get("query") if isinstance(rebound_input, dict) else None
                old_query = invoke_input.get("query") if isinstance(invoke_input, dict) else None
                if isinstance(new_query, str) and isinstance(old_query, str) and new_query != old_query:
                    step_retry += 1
                    invoke_input = _merge_retry_input(invoke_input, rebound_input)
                    print(f"[FileNotFound retry] tool={tool_name} | step={step_num} | retry={step_retry}")
                    continue

        # MLS post-step verify for semantic/format errors (compat-first: try retry_input first)
        session_state_for_check = {
            "mls_debug_executor": chains.get("mls_debug_executor"),
            "llm": chains.get("llm"),
            "history": history,
            "conversation_log": log_entries,
        }
        status_ok, post_retry_input, post_report_for_cb = await _run_mls_post_step_verify(
            session_state_for_check,
            step_num,
            task_desc,
            tool_name,
            invoke_input,
            raw_output,
        )
        history = session_state_for_check.get("history", history)
        log_entries = session_state_for_check.get("conversation_log", log_entries)

        if not status_ok:
            post_reason = post_report_for_cb or ("步骤后置校验失败。" if ui_lang == "zh" else "Post-step verification failed.")
            if isinstance(post_retry_input, dict) and step_retry < MAX_STEP_RETRIES:
                step_retry += 1
                invoke_input = _merge_retry_input(invoke_input, post_retry_input)
                print(f"[Post-step retry] tool={tool_name} | step={step_num} | retry={step_retry}")
                continue
            is_failure = True
            failure_reason = post_reason
            raw_output = json.dumps(
                {"status": "error", "error": {"type": "PostStepCheckFailed", "message": post_reason}},
                ensure_ascii=False,
            )

        if not is_failure and chains.get("llm"):
            output_file_path = _get_output_file_path_from_raw(raw_output, tool_name)
            file_preview = _read_output_file_preview(output_file_path) if output_file_path else None
            cb_match, cb_note = await _cb_post_step_check(
                chains["llm"],
                step_num,
                task_desc,
                tool_name,
                raw_output,
                output_file_path=output_file_path,
                file_preview=file_preview,
            )
            if not cb_match:
                if step_retry < MAX_STEP_RETRIES:
                    debug_retry_input, _debug_report = await _run_mls_debug_with_tools(
                        session_state_for_check,
                        step_num,
                        task_desc,
                        tool_name,
                        invoke_input,
                        cb_note or ("CB 后置校验不一致。" if ui_lang == "zh" else "CB post-step check mismatch."),
                    )
                    history = session_state_for_check.get("history", history)
                    log_entries = session_state_for_check.get("conversation_log", log_entries)
                    if isinstance(debug_retry_input, dict):
                        step_retry += 1
                        invoke_input = _merge_retry_input(invoke_input, debug_retry_input)
                        print(f"[CB retry] tool={tool_name} | step={step_num} | retry={step_retry}")
                        continue
                is_failure = True
                failure_reason = cb_note or ("CB 后置校验不一致。" if ui_lang == "zh" else "CB post-step check mismatch.")
                raw_output = json.dumps(
                    {"status": "error", "error": {"type": "CBPostStepMismatch", "message": failure_reason}},
                    ensure_ascii=False,
                )

        # Runtime failures: attempt MLS debug retry before marking terminal failure.
        if is_failure and step_retry < MAX_STEP_RETRIES:
            debug_retry_input, debug_report_for_cb = await _run_mls_debug_with_tools(
                session_state_for_check,
                step_num,
                task_desc,
                tool_name,
                invoke_input,
                failure_reason or str(raw_output),
            )
            history = session_state_for_check.get("history", history)
            log_entries = session_state_for_check.get("conversation_log", log_entries)
            if isinstance(debug_retry_input, dict):
                step_retry += 1
                invoke_input = _merge_retry_input(invoke_input, debug_retry_input)
                print(f"[Failure retry] tool={tool_name} | step={step_num} | retry={step_retry}")
                continue
            if debug_report_for_cb:
                failure_reason = debug_report_for_cb

        if not is_failure and not cached_flag:
            save_cached_tool_result(state, tool_name, invoke_input, raw_output)

        last_output = raw_output
        step_done = True

    if last_output is not None:
        try:
            out_failed, _ = _tool_output_indicates_failure(last_output)
            if not out_failed:
                _register_artifacts_to_context(last_output)
        except Exception as e:
            print(f"[Artifact register] skipped due to error: {e}")

    # Record result
    protein_ctx.add_tool_call(step_num, tool_name, merged_tool_input, last_output, cached=cached_flag)
    step_results[step_num] = {"raw_output": last_output}
    
    # --- Generate detailed feedback for the UI (MLS needs full context for self-check) ---
    is_failure, parsed_failure_reason = _tool_output_indicates_failure(last_output)
    if not failure_reason:
        failure_reason = parsed_failure_reason
    try:
        out_data = json.loads(last_output) if isinstance(last_output, str) else last_output
    except Exception:
        out_data = None

    if is_failure:
        feedback_content = _ui_text(ui_lang, "step_failed", step_num=step_num)
    else:
        feedback_content = _ui_text(ui_lang, "step_done", step_num=step_num)

    # 1. Output summary (include error-related keys so MLS can self-check)
    try:
        if isinstance(out_data, dict):
            important_keys = [
                "success", "error", "message", "detail", "traceback", "tool_name",
                "protein_id", "pdb_id", "uniprot_id", "mutation", "delta_delta_g"
            ]
            summary_parts = []
            for k in important_keys:
                if k in out_data:
                    val = out_data[k]
                    summary_parts.append(f"**{k}:** {val}")
            if summary_parts:
                feedback_content += _ui_text(ui_lang, "summary") + ", ".join(summary_parts) + "\n\n"
            else:
                dump = json.dumps(out_data, ensure_ascii=False)
                output_label = "输出：" if ui_lang == "zh" else "Output: "
                feedback_content += f"{output_label}`{dump[:300]}...`\n\n"
        else:
            feedback_content += _ui_text(ui_lang, "output_preview") + f"`{str(last_output)[:300]}...`\n\n"
    except Exception:
        feedback_content += _ui_text(ui_lang, "output_preview") + f"`{str(last_output)[:300]}...`\n\n"

    # When failed or non-JSON: always show raw output for MLS self-check
    raw_str = last_output if isinstance(last_output, str) else json.dumps(last_output, ensure_ascii=False)
    if is_failure or not isinstance(out_data, dict):
        feedback_content += _ui_text(ui_lang, "raw_output") + "\n```\n"
        feedback_content += raw_str[:2000] + ("\n...(truncated)" if len(raw_str) > 2000 else "")
        feedback_content += "\n```\n\n"

    # 2. File Hosting (OSS)
    out_file = _get_output_file_path_from_raw(last_output, tool_name)
    oss_url = None
    if out_file:
        try:
            oss_url = await upload_file_to_cloud_async(out_file)
            if oss_url:
                feedback_content += _ui_text(ui_lang, "cloud_download", name=os.path.basename(out_file), url=oss_url) + "\n\n"
        except Exception as e:
            print(f"OSS upload failed for {out_file}: {e}")

        # 3. File Preview
        preview = _read_output_file_preview(out_file)
        if preview:
            feedback_content += _ui_text(ui_lang, "file_preview", name=os.path.basename(out_file)) + f"\n```\n{preview}\n```\n\n"

    executions.append({
        "step": step_num, "tool_name": tool_name, "inputs": merged_tool_input,
        "outputs": (str(last_output)[:1000] + "..." if len(str(last_output)) > 1000 else str(last_output)),
        "oss_url": oss_url,
        "timestamp": datetime.now().isoformat(),
    })

    try:
        input_tokens, output_tokens, total_tokens, usage_missing = _extract_usage_from_output(last_output)
        analytics_store.record_tool_call(
            ts=datetime.now().isoformat(),
            session_id=str(state.get("session_id", "")),
            tool_name=tool_name,
            status="failed" if is_failure else "success",
            latency_ms=int((time.time() - execute_started) * 1000),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            usage_missing=usage_missing,
            model=getattr(chains.get("llm"), "model_name", "") if chains.get("llm") else "",
            owner_key=str(config.get("configurable", {}).get("chains", {}).get("owner_key", state.get("owner_key", ""))),
            ip=str(state.get("client_ip", "")),
        )
    except Exception:
        pass

    # 4. Image Hosting (plots, images)
    try:
        img_paths = _extract_image_paths_from_tool_output(last_output, tool_name)
        for ip in img_paths:
            if ip != out_file: # Avoid duplicate link
                oss_img_url = await upload_file_to_cloud_async(ip)
                if oss_img_url:
                    feedback_content += _ui_text(ui_lang, "generated_image", name=os.path.basename(ip), url=oss_img_url) + "\n\n"
    except Exception:
        pass

    if is_failure:
        feedback_content += _ui_text(ui_lang, "pipeline_paused")

    history.append({"role": "assistant", "content": feedback_content, "role_id": "machine_learning_specialist"})

    status = "execution_failed" if is_failure else "executing"
    return {
        "current_step_index": idx + 1,
        "step_results": step_results,
        "history": history,
        "conversation_log": log_entries,
        "tool_executions": executions,
        "tool_cache": state.get("tool_cache", {}),
        "status": status,
        "execution_failed": is_failure,
        "failed_step": step_num if is_failure else None,
        "failed_reason": failure_reason if is_failure else None,
        "ui_lang": ui_lang,
    }


async def finalize_start_node(state: AgentState, config: RunnableConfig):
    """Show 'Summarizing' so UI updates before LLM runs."""
    history = list(state.get("history", []))
    ui_lang = state.get("ui_lang") or _detect_ui_lang(state["messages"][-1].content)
    history.append({
        "role": "assistant",
        "content": _ui_text(ui_lang, "summarizing"),
        "role_id": "principal_investigator",
    })
    return {"history": history, "ui_lang": ui_lang}


async def finalize_node(state: AgentState, config: RunnableConfig):
    """Finalizer node: generates final summary using tool execution history."""
    chains = config.get("configurable", {}).get("chains", {})
    history = list(state.get("history", []))
    tool_executions = state.get("tool_executions", [])
    protein_ctx = state["protein_context"]
    user_input = state["messages"][-1].content
    ui_lang = state.get("ui_lang") or _detect_ui_lang(user_input)

    # Build the full run record for the finalizer
    analysis_log = []
    for i, entry in enumerate(tool_executions, 1):
        step = entry.get("step", i)
        tool_name = entry.get("tool_name", "unknown")
        inputs = entry.get("inputs", {})
        outputs = entry.get("outputs", "")
        oss_url = entry.get("oss_url")
        analysis_log.append(
            f"Step {step}: {tool_name}\n"
            f"  Input: {json.dumps(inputs, ensure_ascii=False)}\n"
            f"  Output: {str(outputs)[:500]}\n"
            + (f"  Cloud Download: {oss_url}" if oss_url else "")
        )

    full_run_record = "\n\n".join([
        f"User request: {user_input}",
        f"Protein context: {protein_ctx.get_context_summary()}",
        f"Tool executions:\n" + "\n".join(analysis_log) if analysis_log else "No tools executed."
    ])

    try:
        summary = await asyncio.to_thread(
            chains["finalizer"].invoke,
            {
                "input": user_input,
                "full_run_record": full_run_record,
                "original_input": user_input,
                "analysis_log": "\n".join(analysis_log) if analysis_log else "No analysis log available.",
                "references": ""
            }
        )
        history.append({"role": "assistant", "content": summary, "role_id": "principal_investigator"})
    except Exception as e:
        print(f"[Finalizer] failed: {e}")
        # Fallback: derive status directly from recorded executions.
        if tool_executions:
            summary_parts = [_ui_text(ui_lang, "final_summary_title")]
            if state.get("execution_failed"):
                summary_parts.append(
                    _ui_text(
                        ui_lang,
                        "pipeline_paused_at",
                        step=state.get("failed_step"),
                        reason=state.get("failed_reason") or ("未知错误" if ui_lang == "zh" else "Unknown error"),
                    )
                )
            else:
                summary_parts.append(_ui_text(ui_lang, "task_completed"))
            for entry in tool_executions:
                output_text = entry.get("outputs", "")
                failed, reason = _tool_output_indicates_failure(output_text)
                if failed:
                    summary_parts.append(
                        _ui_text(
                            ui_lang,
                            "tool_failed",
                            tool=entry.get("tool_name", "Tool"),
                            reason=(reason or ("错误" if ui_lang == "zh" else "error"))[:120],
                        )
                    )
                else:
                    summary_parts.append(_ui_text(ui_lang, "tool_executed", tool=entry.get("tool_name", "Tool")))
            summary = "\n".join(summary_parts)
        else:
            summary = _ui_text(ui_lang, "task_ended")
        history.append({"role": "assistant", "content": summary, "role_id": "principal_investigator"})

    return {"history": history, "status": "completed"}


def should_continue_research(state: AgentState):
    if state.get("status") == "planning_failed" or state.get("error"):
        return END
    
    research_idx = state.get("research_idx", 0)
    search_idx = state.get("search_idx", 0)
    sections = state.get("research_sections", [])
    
    if research_idx < len(sections):
        section = sections[research_idx]
        queries = section.get("search_queries", [])
        if search_idx < len(queries):
            return "research_search_start_node"
        else:
            return "research_sub_report_start_node"
    return "research_report_start_node"


def should_continue(state: AgentState):
    if state.get("status") == "planning_failed" or state.get("error"):
        return END
    if state.get("execution_failed"):
        return "finalize_start_node"
    
    plan = state.get("plan", [])
    current_idx = state.get("current_step_index", 0)
    
    if current_idx < len(plan):
        return "execute_start_node"
    return "finalize_start_node"


def create_agent_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("research_plan_start_node", research_plan_start_node)
    workflow.add_node("research_plan_node", research_plan_node)
    workflow.add_node("chat_start_node", chat_start_node)
    workflow.add_node("chat_node", chat_node)
    workflow.add_node("research_search_start_node", research_search_start_node)
    workflow.add_node("research_search_node", research_search_node)
    workflow.add_node("research_sub_report_start_node", research_sub_report_start_node)
    workflow.add_node("research_sub_report_node", research_sub_report_node)
    workflow.add_node("research_report_start_node", research_report_start_node)
    workflow.add_node("research_report_node", research_report_node)
    workflow.add_node("plan_start_node", plan_start_node)
    workflow.add_node("plan_node", plan_node)
    workflow.add_node("execute_start_node", execute_start_node)
    workflow.add_node("execute_node", execute_node)
    workflow.add_node("finalize_start_node", finalize_start_node)
    workflow.add_node("finalize_node", finalize_node)

    workflow.add_edge(START, "research_plan_start_node")
    workflow.add_edge("research_plan_start_node", "research_plan_node")
    workflow.add_conditional_edges(
        "research_plan_node",
        lambda s: "chat_start_node" if s.get("status") == "chat_mode" else ("plan_start_node" if s.get("status") == "research_skipped" else "research_search_start_node"),
    )
    # Chat mode nodes for direct answers when PI sections returns [].
    workflow.add_edge("chat_start_node", "chat_node")
    workflow.add_edge("chat_node", END)
    workflow.add_edge("research_search_start_node", "research_search_node")
    workflow.add_conditional_edges("research_search_node", should_continue_research)
    workflow.add_edge("research_sub_report_start_node", "research_sub_report_node")
    workflow.add_edge("research_sub_report_node", "research_search_start_node")
    workflow.add_edge("research_report_start_node", "research_report_node")
    workflow.add_edge("research_report_node", "plan_start_node")
    workflow.add_edge("plan_start_node", "plan_node")
    workflow.add_conditional_edges(
        "plan_node",
        lambda s: "chat_start_node" if s.get("status") == "chat_mode" else ("finalize_start_node" if s.get("status") == "planning_failed" or not s.get("plan") else "execute_start_node"),
    )
    workflow.add_edge("execute_start_node", "execute_node")
    workflow.add_conditional_edges("execute_node", should_continue)
    workflow.add_edge("finalize_start_node", "finalize_node")
    workflow.add_edge("finalize_node", END)

    return workflow.compile()
