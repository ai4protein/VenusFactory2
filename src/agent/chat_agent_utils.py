from __future__ import annotations

import asyncio
import csv
import json
import os
import re
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage

from agent.graph.pi_retrieval import (
    PI_SEARCH_TOOL_NAMES,
    SEARCH_MAX_RESULTS,
    _extract_deepsearch_data,
    _fetch_literature_for_pi,
    _fetch_search_for_pi_report,
    _is_search_result_empty,
    _pi_database_fallback,
    _refine_query_for_search,
    _run_section_search,
    _translate_user_query_to_english,
)
from config import get_config
from mcp_tools.common.results import tool_output_indicates_failure as _tool_output_indicates_failure
from mcp_tools.common.common_utils import (
    get_project_root,
    get_temp_outputs_base_dir,
    get_web_v2_root_dir,
)

_cfg = get_config()

_PROJECT_ROOT = str(get_project_root().resolve())


def _find_existing_file_by_name(file_name: str) -> str | None:
    """Find a generated artifact by basename under controlled output roots."""
    if not file_name or not isinstance(file_name, str):
        return None
    raw = file_name.strip()
    if not raw or os.path.basename(raw) != raw:
        return None
    roots = []
    try:
        roots.append(str(get_temp_outputs_base_dir().resolve()))
    except Exception:
        pass
    try:
        roots.append(str(get_web_v2_root_dir().resolve()))
    except Exception:
        pass
    roots.append(os.path.join(_PROJECT_ROOT, "temp_outputs"))

    matches = []
    seen = set()
    for root in roots:
        if not root or root in seen or not os.path.isdir(root):
            continue
        seen.add(root)
        for dirpath, _, filenames in os.walk(root):
            if raw in filenames:
                candidate = os.path.abspath(os.path.join(dirpath, raw))
                try:
                    matches.append((os.path.getmtime(candidate), candidate))
                except OSError:
                    matches.append((0.0, candidate))
    if not matches:
        return None
    matches.sort(reverse=True)
    return matches[0][1]


def _resolve_existing_path(path: str) -> str | None:
    if not path or not isinstance(path, str):
        return None
    raw = path.strip()
    if not raw:
        return None
    if os.path.isfile(raw):
        return os.path.abspath(raw)
    candidate = os.path.join(_PROJECT_ROOT, raw)
    if os.path.isfile(candidate):
        return os.path.abspath(candidate)

    try:
        temp_root = str(get_temp_outputs_base_dir().resolve())
        candidate = os.path.join(temp_root, raw)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    except Exception:
        pass

    try:
        web_v2_root = str(get_web_v2_root_dir().resolve())
        candidate = os.path.join(web_v2_root, raw)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    except Exception:
        pass

    found = _find_existing_file_by_name(raw)
    if found:
        return found
    return None

def _is_online_mode() -> bool:
    return _cfg.server.is_online


def _resolve_agent_chat_limits() -> tuple[float, float]:
    """Return mode-based chat limits (online fixed, local unlimited)."""
    return _cfg.agent.max_messages, _cfg.agent.max_tool_calls


AGENT_CHAT_MAX_MESSAGES, AGENT_CHAT_MAX_TOOL_CALLS = _resolve_agent_chat_limits()
MLS_SELF_CHECK_MSG = "🔍 **MLS self-check:** Checking whether parameters can be adjusted and retried."
MLS_POST_STEP_SELF_CHECK_MSG = "🔍 **MLS self-check (post-step):** Step {step_num} executed; verifying output for technical errors before marking complete and proceeding."
MAX_STEP_RETRIES = _cfg.agent.max_step_retries
TOOL_EXECUTION_TIMEOUT = _cfg.agent.tool_execution_timeout


def _dedupe_references(refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for r in refs or []:
        if not isinstance(r, dict):
            continue
        title = (r.get('title') or '').strip().lower()
        doi = (r.get('doi') or '').strip().lower()
        url = (r.get('url') or '').strip().lower()
        key = (title, doi, url)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out

def extract_sequence_from_message(message: str) -> str | None:
    """Extract protein sequence from user message"""
    sequence_pattern = r'[ACDEFGHIKLMNPQRSTVWY]{20,}'
    matches = re.findall(sequence_pattern, message.upper())
    return matches[0] if matches else None

def extract_uniprot_id_from_message(message: str) -> str | None:
    """Extract UniProt ID from user message"""
    uniprot_pattern = r'\b[A-Z][A-Z0-9]{5}(?:[A-Z0-9]{4})?\b'
    matches = re.findall(uniprot_pattern, message.upper())
    return matches[0] if matches else None

def _extract_download_file_from_output(tool_name: str, output_data: dict) -> str | None:
    """Extract local file path from tool output for download. Returns path if file exists."""
    if not isinstance(output_data, dict):
        return None
    # Support both success: true and status: "success" (e.g. UniProt download tools)
    is_ok = output_data.get("success") or output_data.get("status") == "success"
    if not is_ok:
        return None
    path = (
        output_data.get("pdb_path") or output_data.get("pdb_file")
        or output_data.get("structure_file") or output_data.get("fasta_file")
        or output_data.get("file_path") or output_data.get("generated_code_path")
        or output_data.get("model_path")
    )
    if not path and isinstance(output_data.get("file_info"), dict):
        path = output_data["file_info"].get("file_path") or output_data["file_info"].get("file_name")
    if not path:
        path = output_data.get("file_name")
    if path and isinstance(path, str):
        resolved = _resolve_existing_path(path)
        if resolved:
            return resolved
    return None

def _get_output_file_path_from_raw(raw_output: Any, tool_name: str) -> str | None:
    """Get output file path from raw tool output for CB post-step verification (file existence + preview)."""
    try:
        data = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
        if not isinstance(data, dict):
            return None
        path = (
            data.get("pdb_path") or data.get("pdb_file") or data.get("structure_file")
            or data.get("fasta_file") or data.get("file_path") or data.get("generated_code_path")
            or data.get("model_path")
        )
        if not path and isinstance(data.get("file_info"), dict):
            path = data["file_info"].get("file_path") or data["file_info"].get("file_name")
        if not path:
            path = data.get("file_name")
        if path and isinstance(path, str):
            resolved = _resolve_existing_path(path)
            if resolved:
                return resolved
    except Exception:
        pass
    return None

_BINARY_EXTENSIONS = frozenset((
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif",
    ".webp", ".ico", ".svg",
    ".pdf", ".zip", ".gz", ".tar", ".bz2", ".xz", ".7z", ".rar",
    ".pkl", ".pickle", ".pt", ".pth", ".bin", ".npy", ".npz",
    ".h5", ".hdf5", ".ckpt", ".safetensors",
    ".mp3", ".mp4", ".wav", ".avi", ".mov",
    ".exe", ".dll", ".so", ".dylib",
))

def _read_output_file_preview(file_path: str, max_lines: int = 10, max_line_len: int = 200) -> str:
    """Read first max_lines of a file for CB verification. Returns preview string or empty on error. Skips binary files."""
    if not file_path or not os.path.isfile(file_path):
        return ""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in _BINARY_EXTENSIONS:
        return ""
    try:
        lines = []
        with open(file_path, encoding="utf-8", errors="replace") as f:
            for _ in range(max_lines):
                line = f.readline()
                if not line:
                    break
                lines.append(line.rstrip()[:max_line_len])
        return "\n".join(lines) if lines else ""
    except Exception:
        return ""

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".svg", ".webp")

def _extract_image_paths_from_tool_output(raw_output: Any, tool_name: str) -> list[str]:
    """Extract image file paths from tool output (e.g. plot_path, figure_path from agent_generated_code/python_repl). Returns list of absolute paths."""
    paths: list[str] = []
    try:
        data = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
        if not isinstance(data, dict):
            return paths
        # Common keys for plot/figure output
        for key in ("plot_path", "figure_path", "image_path", "plot_file", "figure_file", "output_path", "file_path", "saved_path"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                p = val.strip()
                if os.path.isfile(p) and p.lower().endswith(_IMAGE_EXTENSIONS):
                    paths.append(os.path.abspath(p))
        # Nested result (e.g. result as JSON string)
        for key in ("result", "data", "output"):
            val = data.get(key)
            if isinstance(val, str) and val.strip().startswith("{"):
                try:
                    inner = json.loads(val)
                    if isinstance(inner, dict):
                        for k in ("plot_path", "figure_path", "image_path", "file_path"):
                            v = inner.get(k)
                            if isinstance(v, str) and v.strip() and os.path.isfile(v.strip()) and v.strip().lower().endswith(_IMAGE_EXTENSIONS):
                                paths.append(os.path.abspath(v.strip()))
                except Exception:
                    pass
        # List of paths
        for key in ("plot_paths", "figure_paths", "image_paths", "files"):
            val = data.get(key)
            if isinstance(val, list):
                for v in val:
                    if isinstance(v, str) and v.strip() and os.path.isfile(v.strip()) and v.strip().lower().endswith(_IMAGE_EXTENSIONS):
                        paths.append(os.path.abspath(v.strip()))
    except Exception:
        pass
    return paths


def _mls_debug_step(llm, step_num: int, task_desc: str, tool_name: str, merged_tool_input: dict, error_str: str) -> tuple:
    """Ask MLS to analyze the error (no tools). Returns (retry_input_dict or None, report_for_cb or None). Fallback when mls_debug_executor is not used."""
    context = (
        f"Step {step_num} failed during tool execution.\n\n"
        f"**Task:** {task_desc}\n**Tool:** {tool_name}\n**Current input:** {json.dumps(merged_tool_input, ensure_ascii=False)}\n**Error:** {error_str}"
    )
    from agent.prompts import MLS_SELF_CHECK_TEMPLATE
    prompt = f"{context}\n\n{MLS_SELF_CHECK_TEMPLATE}"
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = (response.content if hasattr(response, "content") else str(response)).strip()
        return _parse_mls_debug_output(content)
    except Exception:
        return (None, None)

def _parse_mls_debug_output(content: str) -> tuple:
    """Parse MLS self-check final output for retry_input or report_for_cb. Returns (retry_input or None, report_for_cb or None)."""
    if content is None or not str(content).strip():
        return (None, None)
    # Strip markdown code block if present
    if "```" in content:
        start = content.find("```")
        if "json" in content[: start + 10]:
            start = content.find("```") + 7
        else:
            start = content.find("```") + 3
        end = content.find("```", start)
        content = content[start: end if end > 0 else None].strip()
    try:
        data = json.loads(content)
        retry = data.get("retry_input") if isinstance(data.get("retry_input"), dict) else None
        report = data.get("report_for_cb") if isinstance(data.get("report_for_cb"), str) else None
        return (retry, report)
    except Exception:
        return (None, None)

def _run_mls_debug_executor_sync(executor, context: str) -> tuple[Any, list]:
    """Run MLS debug executor synchronously. Returns (output_str, intermediate_steps)."""
    result = executor.invoke({"input": context})
    output = (result.get("output") or "").strip() if isinstance(result, dict) else ""
    steps = result.get("intermediate_steps") or []
    return (output, steps)

async def _run_mls_debug_with_tools(
    session_state: dict,
    step_num: int,
    task_desc: str,
    tool_name: str,
    merged_tool_input: dict,
    error_str: str,
) -> tuple:
    """Run MLS self-check; may use read_skill, python_repl, agent_generated_code, etc. Updates history with tool activity. Returns (retry_input or None, report_for_cb or None)."""
    context = (
        f"Step {step_num} failed during tool execution.\n\n"
        f"**Task:** {task_desc}\n**Tool:** {tool_name}\n**Current input:** {json.dumps(merged_tool_input, ensure_ascii=False)}\n**Error:** {error_str}\n\n"
        "Use available read_skill and session tools to diagnose or fix: write_session_file stores actual source; run_session_python executes that file in the sandbox. Then output exactly one JSON: {\"retry_input\": {...}} or {\"report_for_cb\": \"...\"}."
    )
    executor = session_state.get("mls_debug_executor")
    if not executor:
        return await asyncio.to_thread(
            _mls_debug_step,
            session_state["llm"],
            step_num,
            task_desc,
            tool_name,
            merged_tool_input,
            error_str,
        )
    try:
        output, intermediate_steps = await asyncio.to_thread(_run_mls_debug_executor_sync, executor, context)
        history = session_state.get("history") or []
        log_entries = session_state.get("conversation_log") or []
        for action, observation in intermediate_steps:
            tname = getattr(action, "tool", None) or (action.get("tool") if isinstance(action, dict) else None)
            tinputs = getattr(action, "tool_input", None) or (action.get("tool_input") if isinstance(action, dict) else {})
            if tname == "read_skill":
                skill_id = tinputs.get("skill_id", "") if isinstance(tinputs, dict) else ""
                msg = f"📖 **MLS self-check:** Loaded skill `{skill_id}`."
            elif tname == "python_repl":
                msg = "🔧 **MLS self-check:** Ran code (python_repl)."
            elif tname == "agent_generated_code":
                msg = "🔧 **MLS self-check:** Ran generated code."
            else:
                msg = f"🔧 **MLS self-check:** Called tool `{tname}`."
            history.append({"role": "assistant", "content": msg, "role_id": "machine_learning_specialist"})
            log_entries.append({
                "role": "assistant",
                "content": f"MLS self-check tool: {tname}",
                "role_id": "machine_learning_specialist",
                "timestamp": datetime.now().isoformat(),
            })
        session_state["history"] = history
        session_state["conversation_log"] = log_entries
        return _parse_mls_debug_output(output)
    except Exception:
        return await asyncio.to_thread(
            _mls_debug_step,
            session_state["llm"],
            step_num,
            task_desc,
            tool_name,
            merged_tool_input,
            error_str,
        )

def _parse_mls_post_step_output(content: str) -> tuple[bool, dict | None, str | None]:
    """Parse MLS post-step verify output. Returns (status_ok, retry_input, report_for_cb).

    Verifier *silence* (empty / unparseable / malformed-without-complaint output)
    is treated as a pass, not a failure: a verifier that fails to produce a
    structured verdict has not actually expressed a complaint, and the
    downstream CB post-step check still runs as a second line of defense.
    Only a parseable JSON that carries a concrete ``retry_input`` or
    ``report_for_cb`` is treated as a real complaint.
    """
    if content is None or not str(content).strip():
        return (True, None, None)
    raw = content
    if "```" in raw:
        start = raw.find("```")
        if "json" in raw[: start + 10]:
            start = raw.find("```") + 7
        else:
            start = raw.find("```") + 3
        end = raw.find("```", start)
        raw = raw[start: end if end > 0 else None].strip()
    if not raw:
        return (True, None, None)
    try:
        data = json.loads(raw)
    except Exception:
        # Try to extract a JSON object substring (the LLM may have prefixed
        # the verdict with prose like "Here is my analysis: { ... }").
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except Exception:
                return (True, None, None)
        else:
            return (True, None, None)
    if not isinstance(data, dict):
        return (True, None, None)
    if data.get("status") == "ok":
        return (True, None, None)
    retry = data.get("retry_input") if isinstance(data.get("retry_input"), dict) else None
    report = data.get("report_for_cb") if isinstance(data.get("report_for_cb"), str) else None
    if retry or (report and report.strip()):
        return (False, retry, report.strip() if report else None)
    # Parseable JSON but no actionable complaint — treat as pass.
    return (True, None, None)

async def _run_mls_post_step_verify(
    session_state: dict,
    step_num: int,
    task_desc: str,
    tool_name: str,
    merged_tool_input: dict,
    raw_output: Any,
) -> tuple[bool, dict | None, str | None]:
    """Run MLS post-step self-check: verify step output before proceeding. Shows new dialog. Returns (status_ok, retry_input, report_for_cb)."""
    out_preview = str(raw_output)
    if len(out_preview) > 2000:
        out_preview = out_preview[:2000] + "\n...(truncated)"
    context = (
        f"Step {step_num} has been executed. Verify the output for technical errors (bugs, failure, wrong format).\n\n"
        f"**Task:** {task_desc}\n**Tool:** {tool_name}\n**Input:** {json.dumps(merged_tool_input, ensure_ascii=False)}\n\n"
        f"**Output:**\n{out_preview}\n\n"
        "Check: Is the output successful and usable (e.g. file path exists, result not null/empty, no nested error)? "
        "If the output has success: true but results, references, or data is null or empty, or does not match the step goal, do NOT output status ok; output {\"retry_input\": {...}} or {\"report_for_cb\": \"...\"} so the step can be re-run with different parameters, another skill, or code. "
        "You may use read_skill, python_repl, or other tools to inspect. "
        "If OK, output exactly: {\"status\": \"ok\"}. "
        "If error, null/empty result, or wrong format, output {\"retry_input\": {...}} or {\"report_for_cb\": \"...\"}."
    )
    executor = session_state.get("mls_debug_executor")
    if not executor:
        return (True, None, None)
    try:
        output, intermediate_steps = await asyncio.to_thread(_run_mls_debug_executor_sync, executor, context)
        history = session_state.get("history") or []
        log_entries = session_state.get("conversation_log") or []
        for action, observation in intermediate_steps:
            tname = getattr(action, "tool", None) or (action.get("tool") if isinstance(action, dict) else None)
            tinputs = getattr(action, "tool_input", None) or (action.get("tool_input") if isinstance(action, dict) else {})
            if tname == "read_skill":
                skill_id = tinputs.get("skill_id", "") if isinstance(tinputs, dict) else ""
                msg = f"📖 **MLS self-check (post-step):** Loaded skill `{skill_id}`."
            elif tname == "python_repl":
                msg = "🔧 **MLS self-check (post-step):** Ran code (python_repl)."
            elif tname == "agent_generated_code":
                msg = "🔧 **MLS self-check (post-step):** Ran generated code."
            else:
                msg = f"🔧 **MLS self-check (post-step):** Called tool `{tname}`."
            history.append({"role": "assistant", "content": msg, "role_id": "machine_learning_specialist"})
            log_entries.append({
                "role": "assistant",
                "content": f"MLS post-step self-check tool: {tname}",
                "role_id": "machine_learning_specialist",
                "timestamp": datetime.now().isoformat(),
            })
        session_state["history"] = history
        session_state["conversation_log"] = log_entries
        return _parse_mls_post_step_output(output)
    except Exception:
        return (True, None, None)

def _output_looks_null_or_empty(raw_output: Any) -> bool:
    """Heuristic: True if a success-envelope payload has *explicitly* null or
    empty ``results``/``references``/``data``/``entries`` AND no other
    substantive content. Missing keys do NOT count as null — many tools
    return their data under a different key (``content``, ``sequence``,
    ``file_path``, ``file_info``, ``predictions``, ...), and treating an
    absent ``results`` field as null would mis-flag every one of them.
    """
    try:
        parsed = json.loads(str(raw_output)) if isinstance(raw_output, str) else raw_output
        if not isinstance(parsed, dict):
            return False
        is_success = parsed.get("success") is True or parsed.get("status") == "success"
        if not is_success:
            return False
        # Substantive content fields — if any of these has real content,
        # the payload is non-empty regardless of results/references/data.
        SUBSTANTIVE = (
            "sequence", "sequences", "file_path", "file_info", "file_name",
            "content", "content_preview", "predictions", "metrics", "model_info",
            "biological_metadata", "score", "scores", "config_path", "output_files",
            "skill_id", "uniprot_id",
        )
        for key in SUBSTANTIVE:
            val = parsed.get(key)
            if val is None:
                continue
            if isinstance(val, str) and not val.strip():
                continue
            if isinstance(val, (list, dict, tuple, set)) and len(val) == 0:
                continue
            return False
        # Only when none of the substantive keys carried real content do we
        # look at results/references/data/entries as a last signal. Missing
        # keys here mean "tool did not promise this field" — not "null".
        flagged = False
        for key in ("results", "references", "data", "entries"):
            if key not in parsed:
                continue
            val = parsed[key]
            if val is None:
                flagged = True
                break
            if isinstance(val, (list, dict)) and len(val) == 0:
                flagged = True
                break
        return flagged
    except Exception:
        return False

async def _cb_post_step_check(
    llm,
    step_num: int,
    task_desc: str,
    tool_name: str,
    raw_output: Any,
    output_file_path: str | None = None,
    file_preview: str | None = None,
    timeout_sec: float = 12.0,
) -> tuple[bool, str]:
    """CB verifies: (1) execution matches plan, (2) output not null/empty/weird, (3) if file produced, exists and preview correct. Returns (matches, note)."""
    try:
        # Deterministic guard for read_skill: when it returns success with a
        # non-empty ``content`` field, skip the LLM check entirely. The CB
        # prompt would otherwise summarize the output as ``success=True`` with
        # no content snippet, giving the LLM nothing to judge against and
        # often producing spurious MISMATCH verdicts.
        if tool_name == "read_skill":
            try:
                parsed_rs = json.loads(str(raw_output)) if isinstance(raw_output, str) else raw_output
                if isinstance(parsed_rs, dict):
                    is_ok = parsed_rs.get("success") is True or parsed_rs.get("status") == "success"
                    content_val = parsed_rs.get("content")
                    if is_ok and isinstance(content_val, str) and content_val.strip():
                        return (True, "")
            except Exception:
                pass

        # Deterministic guard for finetuned function prediction outputs.
        # Avoids LLM false negatives when CSV already contains a valid prediction value.
        if tool_name in {"predict_protein_function", "predict_residue_function"} and output_file_path and os.path.isfile(output_file_path):
            try:
                parsed = json.loads(str(raw_output)) if isinstance(raw_output, str) else raw_output
                status_ok = isinstance(parsed, dict) and (
                    parsed.get("success") is True or parsed.get("status") == "success"
                )
                if status_ok:
                    with open(output_file_path, encoding="utf-8", errors="replace", newline="") as f:
                        reader = csv.DictReader(f)
                        fieldnames = [str(x).strip().lower() for x in (reader.fieldnames or [])]
                        prediction_columns = [c for c in ("prediction", "predicted_class", "probabilities") if c in fieldnames]
                        if prediction_columns:
                            for row in reader:
                                values = [str(row.get(c, "")).strip() for c in prediction_columns]
                                if any(v and v.lower() not in {"nan", "none", "null", "[]", "{}"} for v in values):
                                    return (True, "")
            except Exception:
                pass

        out_str = str(raw_output) if raw_output is not None else ""
        if len(out_str) > 500:
            out_str = out_str[:500] + "..."
        null_or_empty = _output_looks_null_or_empty(raw_output)
        try:
            parsed = json.loads(str(raw_output)) if isinstance(raw_output, str) else raw_output
            if isinstance(parsed, dict):
                success = parsed.get("success", None)
                err = parsed.get("error", "")
                out_str = f"success={success}" + (f", error={err[:200]}" if err else "")
                if parsed.get("uniprot_id"):
                    out_str += f", uniprot_id={parsed.get('uniprot_id')}"
                if parsed.get("sequence") and isinstance(parsed["sequence"], str):
                    out_str += f", sequence length={len(parsed['sequence'])}"
                # Surface a content snippet so the LLM can actually judge the
                # payload. Without this, many tools collapse to
                # ``success=True`` and the LLM defaults to MISMATCH because
                # it has nothing to look at.
                for snippet_key in ("content_preview", "content", "data", "results", "biological_metadata", "predictions", "metrics"):
                    val = parsed.get(snippet_key)
                    if val in (None, "", [], {}):
                        continue
                    try:
                        val_str = val if isinstance(val, str) else json.dumps(val, ensure_ascii=False, default=str)
                    except Exception:
                        val_str = str(val)
                    val_str = val_str.strip()
                    if not val_str:
                        continue
                    if len(val_str) > 300:
                        val_str = val_str[:300] + "..."
                    out_str += f", {snippet_key}={val_str}"
                    break
                if null_or_empty:
                    out_str += "; results/references/data is null or empty"
        except Exception:
            pass
        prompt = (
            f"You are the Computational Biologist. Verify: (1) execution matches the plan, "
            f"(2) output is not null, empty, or useless for the step goal (e.g. if step was to find UniProt ID, output must contain IDs), "
            f"(3) if an output file was expected, it exists and preview looks correct.\n\n"
            f"**Planned step:** {task_desc}\n**Planned tool:** {tool_name}\n\n"
            f"**Actual tool used:** {tool_name}\n**Result summary:** {out_str}\n\n"
        )
        if null_or_empty:
            prompt += "**Note:** The result appears to have null or empty results/references/data. If the step goal required actual data (e.g. IDs, list of hits), reply MISMATCH and say output is null/empty or does not match plan; CB will ask MLS to re-execute with different parameters, another skill, or code.\n\n"
        if output_file_path:
            prompt += f"**Output file path:** {output_file_path}\n**File exists:** yes\n"
            if file_preview:
                prompt += f"**First 10 lines of file:**\n```\n{file_preview}\n```\n\nCheck that the preview is consistent with the step goal. If content looks wrong, reply MISMATCH.\n\n"
            else:
                prompt += "\n\n"
        prompt += "Reply with one line only: MATCH or MISMATCH: <brief reason>"
        response = await asyncio.wait_for(
            asyncio.to_thread(llm.invoke, [HumanMessage(content=prompt)]),
            timeout=timeout_sec,
        )
        if response is None:
            return (True, "")
        content = (response.content if hasattr(response, "content") else str(response)).strip().upper()
        if "MISMATCH" in content:
            note = content.split("MISMATCH", 1)[-1].strip(" :").strip()[:200]
            return (False, note or "Execution may deviate from plan.")
        return (True, "")
    except Exception:
        return (True, "")

def _try_parse_json_array(s: str) -> list | None:
    """Try to parse s as JSON array. Removes trailing commas before ] or } to tolerate LLM output. Returns list or None."""
    if not s or not s.strip():
        return None
    s = s.strip()
    # Remove trailing commas before ] or } (common in LLM-generated JSON)
    s = re.sub(r",\s*]", "]", s)
    s = re.sub(r",\s*}", "}", s)
    try:
        out = json.loads(s)
        return out if isinstance(out, list) else None
    except json.JSONDecodeError:
        return None

def _find_json_array_end(s: str, start: int) -> int:
    """Return index of the ']' that matches the '[' at start. Ignores [ ] inside double-quoted strings. Returns -1 if not found."""
    if start >= len(s) or s[start] != "[":
        return -1
    depth = 1
    i = start + 1
    in_string = False
    escape = False
    while i < len(s) and depth > 0:
        c = s[i]
        if escape:
            escape = False
            i += 1
            continue
        if c == "\\" and in_string:
            escape = True
            i += 1
            continue
        if c == '"':
            in_string = not in_string
            i += 1
            continue
        if in_string:
            i += 1
            continue
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1

def _parse_cb_plan(raw_content: str) -> list:
    """Extract pipeline (list of step dicts) from CB raw output. Tries JSON parse, then strip markdown, then find [...]. Returns [] on failure."""
    if not raw_content or not isinstance(raw_content, str):
        return []
    text = raw_content.strip()
    # 1) Direct JSON (with trailing-comma tolerance)
    out = _try_parse_json_array(text)
    if out is not None:
        return out
    # 2) Strip ```json ... ``` or ``` ... ```
    for marker in ("```json", "```"):
        if marker in text:
            start = text.find(marker) + len(marker)
            end = text.find("```", start)
            if end == -1:
                end = len(text)
            chunk = text[start:end].strip()
            # Remove trailing ``` if LLM put it inside the block
            if chunk.endswith("```"):
                chunk = chunk[:-3].strip()
            out = _try_parse_json_array(chunk)
            if out is not None:
                return out
            # Fallback: find first '[' and matching ']' (string-aware; handles [ ] inside task_description etc.)
            i = chunk.find("[")
            if i != -1:
                j = _find_json_array_end(chunk, i)
                if j != -1:
                    out = _try_parse_json_array(chunk[i : j + 1])
                    if out is not None:
                        return out
    # 3) Find first '[' and matching ']' in full text (string-aware)
    i = text.find("[")
    if i != -1:
        j = _find_json_array_end(text, i)
        if j != -1:
            out = _try_parse_json_array(text[i : j + 1])
            if out is not None:
                return out
    return []

def _parse_sub_report_short_title(sub_report: str, fallback_title: str = "Sub-report") -> tuple[str, str]:
    """Extract Short title: <phrase> from the first line of sub-report; return (title, body)."""
    if not sub_report or not isinstance(sub_report, str):
        return (fallback_title, sub_report or "")
    raw = sub_report.strip()
    if not raw:
        return (fallback_title, raw)
    first_line = raw.split("\n", 1)[0].strip()
    prefix = "**Short title:**"
    if first_line.startswith(prefix):
        title = first_line[len(prefix) :].strip()
        body = raw.split("\n", 1)[1].strip() if "\n" in raw else ""
        return (title or fallback_title, body)
    if first_line.lower().startswith("short title:"):
        title = first_line[12:].strip()
        body = raw.split("\n", 1)[1].strip() if "\n" in raw else ""
        return (title or fallback_title, body)
    return (fallback_title, raw)
