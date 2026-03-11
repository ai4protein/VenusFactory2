import json
import os
import re
import asyncio
from typing import Dict, Any, List, Optional
from langchain_classic.schema import HumanMessage
from agent.prompts import MLS_SELF_CHECK_TEMPLATE

# Free-use limits per chat (from .env)
AGENT_CHAT_MAX_MESSAGES = int(os.getenv("AGENT_CHAT_MAX_MESSAGES", "50"))
AGENT_CHAT_MAX_TOOL_CALLS = int(os.getenv("AGENT_CHAT_MAX_TOOL_CALLS", "40"))
# PI search: max results per literature/web/dataset call (from .env)
SEARCH_MAX_RESULTS = int(os.getenv("SEARCH_MAX_RESULTS", "3"))
# When a tool fails, MLS self-check bubble (new dialog)
MLS_SELF_CHECK_MSG = "🔍 **MLS self-check:** Checking whether parameters can be adjusted and retried."
# After every step execution, MLS post-step self-check (new dialog) — execution already happened, now verify output
MLS_POST_STEP_SELF_CHECK_MSG = "🔍 **MLS self-check (post-step):** Step {step_num} executed; verifying output for technical errors before marking complete and proceeding."
# Max step-level retries after MLS self-check (from .env; default 2)
MAX_STEP_RETRIES = int(os.getenv("MAX_STEP_RETRIES", "2"))
# Tool execution timeout in seconds (from .env; default 300 = 5 min); prevents indefinite hang
TOOL_EXECUTION_TIMEOUT = int(os.getenv("TOOL_EXECUTION_TIMEOUT", "300"))
PI_SEARCH_TOOL_NAMES = [
    "query_pubmed", "query_semantic_scholar", "query_arxiv",
    "query_tavily", "query_duckduckgo",
    "query_github", "query_hugging_face"
]

def _tool_output_indicates_failure(raw_output: Any) -> tuple[bool, str]:
    """Detect if tool output indicates failure: top-level success:false, or nested result/data with success:false or error (e.g. BLAST timeout in result string). Returns (is_failure, error_message)."""
    if raw_output is None:
        return (False, "")
    text = str(raw_output).strip()
    if not text:
        return (False, "")
    # Top-level parse
    try:
        parsed = json.loads(text) if isinstance(raw_output, str) else raw_output
    except Exception:
        parsed = None
    if not isinstance(parsed, dict):
        return (False, "")

    # 1) Top-level success is False
    if parsed.get("success") is False:
        err = parsed.get("error") or parsed.get("message") or str(parsed)
        return (True, err[:500] if isinstance(err, str) else str(err)[:500])

    # 1b) Top-level status is "error" (e.g. UniProt download tools)
    if parsed.get("status") == "error":
        err_obj = parsed.get("error")
        if isinstance(err_obj, dict):
            err = err_obj.get("message") or err_obj.get("type") or str(err_obj)
        else:
            err = err_obj or parsed.get("message") or str(parsed)
        return (True, (err[:500] if isinstance(err, str) else str(err)[:500]))

    # 2) Top-level success is True but nested payload indicates failure (e.g. result/data/output as JSON string or dict)
    for key in ("result", "data", "output", "response", "body"):
        val = parsed.get(key)
        if val is None:
            continue
        if isinstance(val, dict):
            if val.get("success") is False:
                err = val.get("error") or val.get("message") or str(val)
                return (True, (err[:500] if isinstance(err, str) else str(err)[:500]))
            if val.get("error"):
                return (True, str(val.get("error"))[:500])
        if isinstance(val, str):
            val_strip = val.strip()
            if not val_strip or val_strip[0] not in ("{", "["):
                if "timeout" in val_strip.lower() or "error" in val_strip.lower():
                    return (True, val_strip[:500])
                continue
            try:
                inner = json.loads(val_strip)
                if isinstance(inner, dict) and inner.get("success") is False:
                    err = inner.get("error") or inner.get("message") or val_strip
                    return (True, (err[:500] if isinstance(err, str) else str(err)[:500]))
                if isinstance(inner, dict) and inner.get("error"):
                    return (True, str(inner.get("error"))[:500])
            except Exception:
                if "success\": false" in val_strip or '"success":false' in val_strip or "Timeout" in val_strip:
                    return (True, val_strip[:500])
    return (False, "")

def _dedupe_references(refs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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

def extract_sequence_from_message(message: str) -> Optional[str]:
    """Extract protein sequence from user message"""
    sequence_pattern = r'[ACDEFGHIKLMNPQRSTVWY]{20,}'
    matches = re.findall(sequence_pattern, message.upper())
    return matches[0] if matches else None

def extract_uniprot_id_from_message(message: str) -> Optional[str]:
    """Extract UniProt ID from user message"""
    uniprot_pattern = r'\b[A-Z][A-Z0-9]{5}(?:[A-Z0-9]{4})?\b'
    matches = re.findall(uniprot_pattern, message.upper())
    return matches[0] if matches else None

def _extract_download_file_from_output(tool_name: str, output_data: dict) -> Optional[str]:
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
        path = output_data["file_info"].get("file_path")
    if path and isinstance(path, str) and os.path.isfile(path):
        return path
    return None

def _get_output_file_path_from_raw(raw_output: Any, tool_name: str) -> Optional[str]:
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
            path = data["file_info"].get("file_path")
        if path and isinstance(path, str) and os.path.isfile(path):
            return path
    except Exception:
        pass
    return None

def _read_output_file_preview(file_path: str, max_lines: int = 10, max_line_len: int = 200) -> str:
    """Read first max_lines of a file for CB verification. Returns preview string or empty on error."""
    if not file_path or not os.path.isfile(file_path):
        return ""
    try:
        lines = []
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for _ in range(max_lines):
                line = f.readline()
                if not line:
                    break
                lines.append(line.rstrip()[:max_line_len])
        return "\n".join(lines) if lines else ""
    except Exception:
        return ""

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".svg", ".webp")

def _extract_image_paths_from_tool_output(raw_output: Any, tool_name: str) -> List[str]:
    """Extract image file paths from tool output (e.g. plot_path, figure_path from agent_generated_code/python_repl). Returns list of absolute paths."""
    paths: List[str] = []
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

def _fetch_literature_for_pi(user_text: str, max_results: int = None) -> tuple:
    """Run literature_search tool. Returns (formatted_str_for_prompt, tool_input_dict, raw_output_str) for logging."""
    if max_results is None:
        max_results = SEARCH_MAX_RESULTS
    empty = ("", {}, "")
    try:
        from tools.tools_agent_hub import get_tools
        tools = get_tools()
        lit_tool = next((t for t in tools if getattr(t, "name", "") == "query_literature_by_keywords"), None)
        if not lit_tool:
            return empty

        query = (user_text or "").strip()[:120]
        
        if not query:
            return empty
        tool_input = {"query": query, "max_results": max_results, "source": "pubmed"}
        out = lit_tool.invoke(tool_input)
        raw_out = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
        
        if isinstance(out, str):
            data = json.loads(out) if out.strip().startswith("{") else {}
        else:
            data = out if isinstance(out, dict) else {}
        
        if not data.get("success"):
            return ("", tool_input, raw_out)
        refs_raw = data.get("references", [])
        
        if isinstance(refs_raw, str):
            try:
                refs = json.loads(refs_raw)
            except json.JSONDecodeError:
                refs = []
        else:
            refs = refs_raw if isinstance(refs_raw, list) else []

        lines = []
        for i, r in enumerate(refs[:5], 1):
            if not isinstance(r, dict):
                continue

            title = r.get("title") or r.get("citation") or "No title"
            authors = r.get("authors") or r.get("author") or ""
            if isinstance(authors, list):
                authors = ", ".join(str(a) for a in authors[:5])
            year = r.get("year") or r.get("published") or ""
            url = r.get("url") or r.get("link") or ""
            lines.append(f"[{i}] {title}. {authors}. {year}. {url}")
        return ("\n".join(lines) if lines else "", tool_input, raw_out)

    except Exception as e:
        print(f"[PI answer] literature_search failed: {e}")
        return empty

def _refine_query_for_search(user_text: str, max_len: int = 60) -> str:
    """Extract short search keywords: protein/gene ID plus up to 3 words (no long sentence)."""
    import re
    t = (user_text or "").strip()
    if not t:
        return ""
    # Protein/gene ID (e.g. P04040) then next 3 words only
    id_match = re.search(r"\b(P\d{5}|[A-Z0-9]{5,6})\b", t, re.IGNORECASE)
    if id_match:
        pid = id_match.group(1).strip()
        rest = re.sub(r"\s+", " ", t[id_match.end() :]).strip().split()[:3]
        return (pid + " " + " ".join(rest)).strip()[:max_len]
    words = re.sub(r"\s+", " ", t).strip().split()[:4]
    return " ".join(words)[:max_len]

def _mls_debug_step(llm, step_num: int, task_desc: str, tool_name: str, merged_tool_input: dict, error_str: str) -> tuple:
    """Ask MLS to analyze the error (no tools). Returns (retry_input_dict or None, report_for_cb or None). Fallback when mls_debug_executor is not used."""
    context = (
        f"Step {step_num} failed during tool execution.\n\n"
        f"**Task:** {task_desc}\n**Tool:** {tool_name}\n**Current input:** {json.dumps(merged_tool_input, ensure_ascii=False)}\n**Error:** {error_str}"
    )
    prompt = f"{context}\n\n{MLS_SELF_CHECK_TEMPLATE}"
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = (response.content if hasattr(response, "content") else str(response)).strip()
        return _parse_mls_debug_output(content)
    except Exception:
        return (None, None)

def _parse_mls_debug_output(content: str) -> tuple:
    """Parse MLS self-check final output for retry_input or report_for_cb. Returns (retry_input or None, report_for_cb or None)."""
    if not (content or content.strip()):
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
        "You may call read_skill, python_repl, agent_generated_code or other tools to diagnose or fix. Then output exactly one JSON: {\"retry_input\": {...}} or {\"report_for_cb\": \"...\"}."
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
            log_entries.append(f"MLS self-check tool: {tname}")
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

def _parse_mls_post_step_output(content: str) -> tuple[bool, Optional[dict], Optional[str]]:
    """Parse MLS post-step verify output. Returns (status_ok, retry_input or None, report_for_cb or None)."""
    if not (content or content.strip()):
        return (False, None, None)
    raw = content
    if "```" in raw:
        start = raw.find("```")
        if "json" in raw[: start + 10]:
            start = raw.find("```") + 7
        else:
            start = raw.find("```") + 3
        end = raw.find("```", start)
        raw = raw[start: end if end > 0 else None].strip()
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and data.get("status") == "ok":
            return (True, None, None)
        retry = data.get("retry_input") if isinstance(data.get("retry_input"), dict) else None
        report = data.get("report_for_cb") if isinstance(data.get("report_for_cb"), str) else None
        return (False, retry, report)
    except Exception:
        return (False, None, None)

async def _run_mls_post_step_verify(
    session_state: dict,
    step_num: int,
    task_desc: str,
    tool_name: str,
    merged_tool_input: dict,
    raw_output: Any,
) -> tuple[bool, Optional[dict], Optional[str]]:
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
            log_entries.append(f"MLS post-step self-check tool: {tname}")
        session_state["history"] = history
        session_state["conversation_log"] = log_entries
        return _parse_mls_post_step_output(output)
    except Exception:
        return (True, None, None)

def _output_looks_null_or_empty(raw_output: Any) -> bool:
    """Heuristic: True if output has success but results/references/data is null or empty. Sequence/download outputs (sequence, file_path, etc.) count as non-empty."""
    try:
        parsed = json.loads(str(raw_output)) if isinstance(raw_output, str) else raw_output
        if not isinstance(parsed, dict) or not parsed.get("success"):
            return False
        # Payload keys that indicate non-empty output (e.g. download_uniprot_sequence, download_ncbi_sequence, file downloads)
        if parsed.get("sequence") and isinstance(parsed["sequence"], str) and parsed["sequence"].strip():
            return False
        if parsed.get("file_path") and isinstance(parsed["file_path"], str) and parsed["file_path"].strip():
            return False
        if parsed.get("sequences") and isinstance(parsed["sequences"], (list, dict)) and len(parsed["sequences"]) > 0:
            return False
        for key in ("results", "references", "data", "entries"):
            val = parsed.get(key)
            if val is None:
                return True
            if isinstance(val, (list, dict)) and len(val) == 0:
                return True
        return False
    except Exception:
        return False

async def _cb_post_step_check(
    llm,
    step_num: int,
    task_desc: str,
    tool_name: str,
    raw_output: Any,
    output_file_path: Optional[str] = None,
    file_preview: Optional[str] = None,
    timeout_sec: float = 12.0,
) -> tuple[bool, str]:
    """CB verifies: (1) execution matches plan, (2) output not null/empty/weird, (3) if file produced, exists and preview correct. Returns (matches, note)."""
    try:
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

def _try_parse_json_array(s: str) -> Optional[list]:
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

def _extract_deepsearch_data(data: dict) -> dict:
    """Helper to extract inner data from deepsearch tool wrapped response."""
    if isinstance(data, dict):
        if data.get("status") == "error":
            return {"success": False, "error": data.get("error", {})}
        if data.get("status") == "success" and "content" in data:
            content_val = data["content"]
            if isinstance(content_val, str) and content_val.strip().startswith("{"):
                try:
                    return json.loads(content_val)
                except Exception:
                    pass
            elif isinstance(content_val, dict):
                return content_val
    return data

def _is_search_result_empty(tool_name: str, data: dict) -> bool:
    """True if the search returned success but no usable results (empty refs/results/datasets)."""
    if not data.get("success"):
        return True
    if tool_name == "query_literature_by_keywords":
        refs = data.get("references") or []
        if isinstance(refs, str):
            try:
                refs = json.loads(refs) if refs.strip().startswith("[") else []
            except Exception:
                refs = []
        return not (isinstance(refs, list) and len(refs) > 0)
    if tool_name == "query_web_by_keywords":
        res = data.get("results") or []
        return not (isinstance(res, list) and len(res) > 0) and not (isinstance(data.get("results"), str) and data.get("results"))
    if tool_name == "query_dataset_by_keywords":
        ds = data.get("datasets") or []
        if isinstance(ds, str):
            try:
                ds = json.loads(ds)
            except json.JSONDecodeError:
                return True
        return not (isinstance(ds, list) and len(ds) > 0)
    return True

def _translate_user_query_to_english(user_text: str, llm) -> str:
    """When user asks in non-English, use LLM to translate intent to 2-5 English search keywords."""
    if not llm or not (user_text or "").strip():
        return ""
    try:
        prompt = (
            "Translate this user question into 2-5 short English search keywords for PubMed/scientific literature. "
            "Output ONLY the keywords, nothing else. No quotes, no explanation."
            "\n\nUser question: " + (user_text or "").strip()[:500]
        )
        resp = llm.invoke([HumanMessage(content=prompt)])
        out = (resp.content if hasattr(resp, "content") else str(resp)).strip()[:120]
        # Ensure output is ASCII-safe (English); if LLM returned non-ASCII, fall through to empty
        if out and not any(ord(c) > 127 for c in out):
            return out
    except Exception:
        pass
    return ""


from web.utils.chat_format_utils import (
    _format_literature_for_reading, _format_web_for_reading,
    _format_literature_citations, _format_web_citations, _format_dataset_citations
)

def _run_section_search(query: str, max_results: int = None) -> tuple:
    """Run literature_search and web_search for one section query. Returns (formatted_citation_str, list of (tool_name, inputs, raw_output))."""
    if max_results is None:
        max_results = SEARCH_MAX_RESULTS
    query = (query or "").strip()[:80]
    if not query:
        return ("No query.", [])
    try:
        from tools.tools_agent_hub import get_tools
        tools = get_tools()
        tools_dict = {getattr(t, "name", ""): t for t in tools}
    except Exception:
        return ("Search tools unavailable.", [])
    sections = []
    logged = []
    
    # Try literature tools in order of preference
    for tool_name in ("query_pubmed", "query_semantic_scholar", "query_arxiv"):
        lit_tool = tools_dict.get(tool_name)
        if not lit_tool:
            continue
        lit_input = {"query": query, "max_results": max_results}
        try:
            print(f"\n[PI section] Invoking: `{tool_name}` with `{lit_input}`", flush=True)
            lit_out = lit_tool.invoke(lit_input)
            raw_lit = lit_out if isinstance(lit_out, str) else json.dumps(lit_out, ensure_ascii=False)
            logged.append((tool_name, lit_input, raw_lit))
            data = json.loads(raw_lit) if isinstance(raw_lit, str) and raw_lit.strip().startswith("{") else {}
            data = _extract_deepsearch_data(data)
            # Some tools return a list of results directly or wrap them in 'references', 'results', or 'papers'
            # We'll normalize to a list.
            if data.get("success") is not False:
                refs = data.get("references") or data.get("results") or data.get("papers") or data
                if isinstance(refs, str):
                    try:
                        refs = json.loads(refs) if refs.strip().startswith("[") else []
                    except Exception:
                        refs = []
                refs = refs if isinstance(refs, list) else []
                lines = _format_literature_for_reading(refs, max_n=5, abstract_max=400)
                if lines:
                    sections.append("**Literature**\n" + "\n".join(lines))
                    break
        except Exception as e:
            logged.append((tool_name, lit_input, json.dumps({"success": False, "error": str(e)})))
            
    # Try web tools in order of preference
    for tool_name in ("query_tavily", "query_duckduckgo"):
        web_tool = tools_dict.get(tool_name)
        if not web_tool:
            continue
        web_input = {"query": query, "max_results": max_results}
        try:
            print(f"\n[PI section] Invoking: `{tool_name}` with `{web_input}`", flush=True)
            web_out = web_tool.invoke(web_input)
            raw_web = web_out if isinstance(web_out, str) else json.dumps(web_out, ensure_ascii=False)
            logged.append((tool_name, web_input, raw_web))
            data = json.loads(raw_web) if isinstance(raw_web, str) and raw_web.strip().startswith("{") else {}
            data = _extract_deepsearch_data(data)
            if data.get("success") is not False:
                res = data.get("results") or data
                if isinstance(res, str):
                    try:
                        res = json.loads(res) if res.strip().startswith("[") else []
                    except Exception:
                        res = []
                if isinstance(res, list) and res:
                    lines = _format_web_for_reading(res, max_n=5, snippet_max=300)
                    if lines:
                        sections.append("**Web**\n" + "\n".join(lines))
                        break
        except Exception as e:
            logged.append((tool_name, web_input, json.dumps({"success": False, "error": str(e)})))
    if not sections:
        return ("No search results for this section.", logged)
    return ("References (cite as [1], [2], etc.):\n\n" + "\n\n".join(sections), logged)

def _fetch_search_for_pi_report(user_text: str, max_results: int = None, llm=None) -> tuple:
    """Run literature_search, web_search, dataset_search. Default sources: pubmed, tavily, github. When a search returns empty or fails, retry with another source (e.g. web: tavily → duckduckgo; literature: pubmed → semantic_scholar → arxiv). Returns (combined_str_with_citations, list of (tool_name, inputs, raw_output))."""
    if max_results is None:
        max_results = SEARCH_MAX_RESULTS
    # Use English keywords: if user_text has non-ASCII, translate via LLM; otherwise refine
    has_non_ascii = any(ord(c) > 127 for c in (user_text or ""))
    query = ""
    if has_non_ascii and llm:
        query = _translate_user_query_to_english(user_text, llm)
    if not query:
        query = _refine_query_for_search(user_text, 80) or (user_text or "").strip()[:80]
    if not query:
        return ("", [])
    try:
        from tools.tools_agent_hub import get_tools
        tools = get_tools()
        tools_dict = {getattr(t, "name", ""): t for t in tools}
        
        sections = []
        logged = []

        # Literature: default pubmed; if empty/fail try semantic_scholar, then arxiv
        for tool_name in ("query_pubmed", "query_semantic_scholar", "query_arxiv"):
            lit_tool = tools_dict.get(tool_name)
            if not lit_tool:
                continue
            lit_input = {"query": query, "max_results": max_results}
            try:
                print(f"\n[PI research] Invoking: `{tool_name}` with `{lit_input}`", flush=True)
                lit_out = lit_tool.invoke(lit_input)
                raw_lit = lit_out if isinstance(lit_out, str) else json.dumps(lit_out, ensure_ascii=False)
                logged.append((tool_name, lit_input, raw_lit))
                data = json.loads(raw_lit) if isinstance(raw_lit, str) and raw_lit.strip().startswith("{") else {}
                data = _extract_deepsearch_data(data)
                if data.get("success") is not False:
                    refs = data.get("references") or data.get("results") or data.get("papers") or data
                    if isinstance(refs, list) and refs:
                        lines = _format_literature_citations(refs, max_n=5)
                        if lines:
                            sections.append("**Literature (cite as [1], [2], ...)**\n" + "\n".join(lines))
                            break
            except Exception as e:
                print(f"[PI report] {tool_name} failed: {e}")
                logged.append((tool_name, lit_input, json.dumps({"success": False, "error": str(e)})))

        # Web: default tavily; if empty/fail try duckduckgo
        for tool_name in ("query_tavily", "query_duckduckgo"):
            web_tool = tools_dict.get(tool_name)
            if not web_tool:
                continue
            web_input = {"query": query, "max_results": max_results}
            try:
                print(f"\n[PI research] Invoking: `{tool_name}` with `{web_input}`", flush=True)
                web_out = web_tool.invoke(web_input)
                raw_web = web_out if isinstance(web_out, str) else json.dumps(web_out, ensure_ascii=False)
                logged.append((tool_name, web_input, raw_web))
                data = json.loads(raw_web) if isinstance(raw_web, str) and raw_web.strip().startswith("{") else {}
                data = _extract_deepsearch_data(data)
                if data.get("success") is not False:
                    res = data.get("results") or data
                    if isinstance(res, list) and res:
                        lines = _format_web_citations(res, max_n=5)
                        if lines:
                            sections.append("**Web (cite as [1], [2], ...)**\n" + "\n".join(lines))
                            break
                    elif isinstance(res, str) and res.strip():
                        sections.append("**Web**\n" + res[:1500])
                        break
            except Exception as e:
                print(f"[PI report] {tool_name} failed: {e}")
                logged.append((tool_name, web_input, json.dumps({"success": False, "error": str(e)})))

        # Dataset: default github; if empty/fail try hugging_face
        for tool_name in ("query_github", "query_hugging_face"):
            dataset_tool = tools_dict.get(tool_name)
            if not dataset_tool:
                continue
            ds_input = {"query": query, "max_results": max_results}
            try:
                print(f"\n[PI research] Invoking: `{tool_name}` with `{ds_input}`", flush=True)
                ds_out = dataset_tool.invoke(ds_input)
                raw_ds = ds_out if isinstance(ds_out, str) else json.dumps(ds_out, ensure_ascii=False)
                logged.append((tool_name, ds_input, raw_ds))
                data = json.loads(raw_ds) if isinstance(raw_ds, str) and raw_ds.strip().startswith("{") else {}
                data = _extract_deepsearch_data(data)
                if data.get("success") is not False:
                    ds_list = data.get("datasets") or data.get("results") or data
                    if isinstance(ds_list, str):
                        try:
                            ds_list = json.loads(ds_list)
                        except json.JSONDecodeError:
                            ds_list = []
                    if isinstance(ds_list, list) and ds_list:
                        lines = _format_dataset_citations(ds_list, max_n=5)
                        if lines:
                            sections.append("**Datasets (cite as [1], [2], ...)**\n" + "\n".join(lines))
                            break
            except Exception as e:
                print(f"[PI report] {tool_name} failed: {e}")
                logged.append((tool_name, ds_input, json.dumps({"success": False, "error": str(e)})))
        if not sections:
            return ("No search results.", logged)
        intro = "References from search (use [1], [2], etc. in your report):\n\n"
        return (intro + "\n\n".join(sections), logged)
    except Exception as e:
        print(f"[PI report] search failed: {e}")
        return ("", [])
