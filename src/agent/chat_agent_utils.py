from __future__ import annotations

import asyncio
import csv
import json
import os
import re
from datetime import datetime
from typing import Any

from langchain_classic.schema import HumanMessage

from agent.prompts import MLS_SELF_CHECK_TEMPLATE
from config import get_config
from logger import get_logger
from web.utils.common_utils import (
    get_project_root,
    get_temp_outputs_base_dir,
    get_web_v2_root_dir,
)

_logger = get_logger("agent.utils")
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
SEARCH_MAX_RESULTS = _cfg.agent.search_max_results
MLS_SELF_CHECK_MSG = "🔍 **MLS self-check:** Checking whether parameters can be adjusted and retried."
MLS_POST_STEP_SELF_CHECK_MSG = "🔍 **MLS self-check (post-step):** Step {step_num} executed; verifying output for technical errors before marking complete and proceeding."
MAX_STEP_RETRIES = _cfg.agent.max_step_retries
TOOL_EXECUTION_TIMEOUT = _cfg.agent.tool_execution_timeout
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
        _logger.warning("PI literature_search failed: %s", e)
        return empty

def _refine_query_for_search(user_text: str, max_len: int = 80) -> str:
    """Extract short, search-engine-friendly keywords from the user query.

    The previous version naively took the first 3–4 words, which for a
    request like "Search PubMed for recent PETase engineering studies,
    download PDB 5XJH, ..." kept the meta-instruction ("Search PubMed for
    recent") and dropped the real content keywords (PETase, 5XJH,
    ProteinMPNN). Now:

    1. Strip a small set of meta-verbs at the start ("search", "find",
       "look up", "query", "download", "fetch", "get") and the
       sources they reference ("PubMed for", "in PubMed", "from PDB",
       etc.).
    2. Keep biological identifiers (UniProt accessions, PDB IDs,
       all-caps gene-like tokens, EC numbers) verbatim.
    3. Drop stop-words and prepositions.
    4. Cap at max_len chars.
    """
    import re
    t = (user_text or "").strip()
    if not t:
        return ""

    # 1. Strip meta-instruction prefixes
    META_PREFIX = re.compile(
        r"^(please\s+)?(search|find|look\s*(up|for)|query|download|fetch|get|retrieve|use|"
        r"check|analyze|run|predict|examine|investigate)\s+(in\s+|for\s+|from\s+|the\s+)?"
        r"(pubmed|arxiv|semantic\s*scholar|biorxiv|uniprot|pdb|rcsb|alphafold|interpro|string|hpa|brenda|chembl|kegg|ncbi)"
        r"(\s+for\s+|\s+in\s+|\s+by\s+|\s+from\s+|\s+|\s*,\s*)?",
        re.IGNORECASE,
    )
    cleaned = META_PREFIX.sub("", t)
    # also strip ", download PDB ..." trailing meta clauses
    cleaned = re.sub(r",\s*(and\s+)?(use|download|fetch|run|then|use)\s+\S+\s+\S+", " ", cleaned, flags=re.IGNORECASE)

    # 2. Extract biological identifiers verbatim from the ORIGINAL text
    # so PDB IDs / UniProt accessions / EC numbers can never be lost to
    # the meta-prefix stripping above.
    bio_tokens: list[str] = []
    bio_tokens.extend(re.findall(r"\b([1-9][A-Za-z0-9]{3})\b", t))  # PDB IDs
    bio_tokens.extend(re.findall(r"\b([OPQ][0-9][A-Z0-9]{3}[0-9])\b", t))  # UniProt
    bio_tokens.extend(re.findall(r"\b(\d+\.\d+\.\d+\.\d+)\b", t))  # EC numbers
    # 3. Word-level cleanup
    STOP = {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "by", "with",
        "is", "are", "was", "were", "be", "this", "that", "these", "those",
        "studies", "study", "paper", "papers", "literature", "article", "articles",
        "recent", "latest", "new",
    }
    word_re = re.compile(r"\b[A-Za-z][A-Za-z0-9\-]{1,}\b")
    words = [w for w in word_re.findall(cleaned) if w.lower() not in STOP]
    # Dedup preserving order; prepend bio tokens so they're never dropped
    seen = set()
    out: list[str] = []
    for tok in bio_tokens + words:
        k = tok.upper()
        if k in seen:
            continue
        seen.add(k)
        out.append(tok)
        if sum(len(x) + 1 for x in out) > max_len:
            break
    return " ".join(out)[:max_len]

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
    _format_dataset_citations,
    _format_literature_citations,
    _format_literature_for_reading,
    _format_web_citations,
    _format_web_for_reading,
)


def _run_section_search(query: str, max_results: int = None, mode: str = "full") -> tuple:
    """Per-section deep-research: parallel-invoke literature + web sources,
    merge/dedupe, then fall back to structural-DB metadata if empty.

    ``mode``:
      - ``full``: PubMed + S2 + arXiv + bioRxiv + Tavily + DuckDuckGo
      - ``lite``: PubMed + Semantic Scholar + Tavily (Expert timeline-friendly)
    """
    import concurrent.futures as _cf

    if max_results is None:
        max_results = SEARCH_MAX_RESULTS
    query = (query or "").strip()[:80]
    if not query:
        return ([], [])
    try:
        from tools.tools_agent_hub import get_tools
        tools = get_tools()
        tools_dict = {getattr(t, "name", ""): t for t in tools}
    except Exception:
        return ([], [])

    sections: list = []
    logged: list = []

    if (mode or "full").strip().lower() == "lite":
        LITERATURE = ["query_pubmed", "query_semantic_scholar"]
        WEB = ["query_tavily"]
    else:
        LITERATURE = ["query_pubmed", "query_semantic_scholar", "query_arxiv", "query_biorxiv"]
        WEB = ["query_tavily", "query_duckduckgo"]
    plan = [(tn, {"query": query, "max_results": max_results})
            for tn in LITERATURE + WEB]

    def _invoke(tn, ti):
        tool = tools_dict.get(tn)
        if not tool:
            return (tn, ti, json.dumps({"success": False, "error": "tool not available"}))
        try:
            _logger.info("PI section invoking (parallel): %s with %s", tn, ti)
            out = tool.invoke(ti)
            raw = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
            return (tn, ti, raw)
        except Exception as e:
            return (tn, ti, json.dumps({"success": False, "error": str(e)}))

    results_by_tool: dict = {}
    with _cf.ThreadPoolExecutor(max_workers=min(6, len(plan))) as ex:
        futures = {ex.submit(_invoke, tn, ti): (tn, ti) for tn, ti in plan}
        for fut in _cf.as_completed(futures, timeout=30):
            try:
                tn, ti, raw = fut.result()
                results_by_tool[tn] = raw
                logged.append((tn, ti, raw))
            except Exception:
                pass

    def _parse_refs(raw_str: str) -> list:
        try:
            data = json.loads(raw_str) if isinstance(raw_str, str) and raw_str.strip().startswith("{") else {}
        except Exception:
            return []
        data = _extract_deepsearch_data(data) if isinstance(data, dict) else {}
        if not isinstance(data, dict) or data.get("success") is False:
            return []
        refs = (data.get("references") or data.get("results")
                or data.get("papers") or data.get("datasets") or [])
        if isinstance(refs, str):
            try:
                refs = json.loads(refs) if refs.strip().startswith("[") else []
            except Exception:
                refs = []
        return refs if isinstance(refs, list) else []

    # Merge literature across all 4 sources, dedupe by URL/DOI/PMID/title
    lit_refs: list = []
    for tn in LITERATURE:
        lit_refs.extend(_parse_refs(results_by_tool.get(tn, "")))
    seen = set()
    deduped_lit = []
    for r in lit_refs:
        if isinstance(r, dict):
            k = (r.get("url") or r.get("doi") or r.get("pmid") or r.get("title") or str(id(r)))[:120]
        else:
            k = str(r)[:120]
        if k in seen:
            continue
        seen.add(k)
        deduped_lit.append(r)
    if deduped_lit:
        lines = _format_literature_for_reading(deduped_lit, max_n=max_results * 2, abstract_max=400)
        if lines:
            sections.extend(lines)

    # Merge web across both sources
    web_refs: list = []
    for tn in WEB:
        web_refs.extend(_parse_refs(results_by_tool.get(tn, "")))
    seen = set()
    deduped_web = []
    for r in web_refs:
        if isinstance(r, dict):
            k = (r.get("url") or r.get("link") or r.get("title") or str(id(r)))[:120]
        else:
            k = str(r)[:120]
        if k in seen:
            continue
        seen.add(k)
        deduped_web.append(r)
    if deduped_web:
        lines = _format_web_for_reading(deduped_web, max_n=max_results * 2, snippet_max=300)
        if lines:
            sections.extend(lines)

    # Structural / sequence-database fallback when literature + web returned
    # nothing. Detect identifier patterns in the query (4-char PDB IDs,
    # UniProt accessions, common gene names) and call the corresponding
    # metadata endpoint directly. This is what a senior researcher would do:
    # if no paper exists for "5XJH", at least look up the RCSB metadata.
    if not sections:
        try:
            import re as _re
            db_sections = _pi_database_fallback(query, tools_dict)
            if db_sections:
                sections.extend(db_sections)
                logged.append(("__db_fallback__",
                              {"query": query, "via": "_pi_database_fallback"},
                              json.dumps({"sections_added": len(db_sections)})))
        except Exception as e:
            logged.append(("__db_fallback__", {"query": query},
                          json.dumps({"success": False, "error": str(e)})))

    if not sections:
        return ([], logged)
    return (sections, logged)


def _pi_database_fallback(query: str, tools_dict: dict) -> list:
    """Last-resort fallback: when literature + web search return nothing,
    parse the query for identifier patterns (PDB IDs, UniProt accessions,
    gene names) and call the corresponding structural / functional database
    endpoints directly. Returns a list of formatted text sections.

    Patterns recognized:
    - 4-character PDB ID: e.g. 5XJH, 1ABC → download_rcsb_entry_metadata_by_pdb_id
    - UniProt accession (1 letter + 5-10 alphanumeric): e.g. P04637, A0A0A0A → download_uniprot_meta_by_id
    - All-caps gene-like token (3-12 chars, ≥1 letter): TP53, EGFR, PETase →
      download_uniprot_search_by_query + download_string_map_ids
    """
    import re as _re

    sections: list = []
    seen_ids = set()

    def _try_tool(tool_name: str, tool_input: dict, label_prefix: str) -> str | None:
        tool = tools_dict.get(tool_name)
        if not tool:
            return None
        try:
            out = tool.invoke(tool_input)
            raw = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
            data = json.loads(raw) if isinstance(raw, str) and raw.strip().startswith("{") else {}
            if data.get("status") == "error" or data.get("success") is False:
                return None
            # Pull preview content / metadata so SC sees real data
            preview = (data.get("content_preview")
                       or data.get("biological_metadata")
                       or data.get("data")
                       or data)
            preview_str = (json.dumps(preview, ensure_ascii=False)[:2000]
                           if not isinstance(preview, str) else preview[:2000])
            return f"**{label_prefix}**\n{preview_str}"
        except Exception:
            return None

    # PDB IDs
    for pdb_id in _re.findall(r"\b([1-9][A-Za-z0-9]{3})\b", query):
        if pdb_id.upper() in seen_ids:
            continue
        seen_ids.add(pdb_id.upper())
        s = _try_tool(
            "download_rcsb_entry_metadata_by_pdb_id",
            {"pdb_id": pdb_id.upper(), "out_path": f"/tmp/pi_rcsb_{pdb_id.upper()}.json"},
            f"RCSB metadata for PDB {pdb_id.upper()}",
        )
        if s:
            sections.append(s)

    # UniProt accessions
    for up_id in _re.findall(r"\b([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})\b", query):
        # Above regex returns tuples for the second alternative; collapse
        if isinstance(up_id, tuple):
            up_id = up_id[0]
        if up_id in seen_ids:
            continue
        seen_ids.add(up_id)
        s = _try_tool(
            "download_uniprot_meta_by_id",
            {"uniprot_id": up_id, "out_path": f"/tmp/pi_uniprot_{up_id}.json"},
            f"UniProt metadata for {up_id}",
        )
        if s:
            sections.append(s)

    # Gene-like all-caps tokens (only try if we have NO sections yet — keeps
    # this cheap; gene-symbol lookups can be slow). Try HPA (per-gene record)
    # first because it's a fast REST endpoint that returns rich
    # functional+expression context in one call. STRING mapping is a cheap
    # backup if HPA doesn't have an entry.
    if not sections:
        STOP_TOKENS = {"PDB", "FASTA", "JSON", "API", "HTTP", "DNA", "RNA",
                       "PETase", "EGFR", "STRING", "UNIPROT", "INTERPRO",
                       "HPA", "KEGG", "MMSEQS", "BLAST", "NCBI", "REST"}
        # Note: PETase / EGFR ARE valid gene names — STOP_TOKENS is just to
        # skip obvious non-gene acronyms. We still try them because the
        # gene-lookup just returns empty for non-genes.
        STOP_TOKENS = {"PDB", "FASTA", "JSON", "API", "HTTP", "DNA", "RNA",
                       "STRING", "UNIPROT", "INTERPRO", "HPA", "KEGG",
                       "MMSEQS", "BLAST", "NCBI", "REST"}
        for tok in _re.findall(r"\b([A-Z][A-Z0-9]{2,11})\b", query):
            if tok in seen_ids or tok in STOP_TOKENS:
                continue
            seen_ids.add(tok)
            s = _try_tool(
                "download_hpa_protein_by_gene",
                {"gene_name": tok,
                 "out_path": f"/tmp/pi_hpa_{tok}.json"},
                f"Human Protein Atlas record for gene {tok}",
            )
            if s:
                sections.append(s)
                break  # one gene is enough for context
            # HPA empty → try STRING id mapping (still gives a hit list with
            # functional annotations)
            s2 = _try_tool(
                "download_string_map_ids",
                {"identifiers": tok,
                 "out_dir": "/tmp",
                 "species": 9606,
                 "limit": 3,
                 "echo_query": True,
                 "filename": f"pi_string_{tok}.tsv"},
                f"STRING ID mapping for gene {tok}",
            )
            if s2:
                sections.append(s2)
                break
    return sections


def _fetch_search_for_pi_report(user_text: str, max_results: int = None, llm=None) -> tuple:
    """Deep-research mode: query ALL configured sources in parallel for each
    of three search groups (literature, web, datasets), merge results across
    sources within each group (deduping by URL/DOI/PMID), then fall back to
    structural-database metadata when everything else is empty.

    Returns (combined_str_with_citations, list of (tool_name, inputs, raw_output)).

    Source coverage (was fallback-chain, now merge):
    - Literature: PubMed + Semantic Scholar + arXiv + bioRxiv (4)
    - Web:        Tavily + DuckDuckGo (2)
    - Datasets:   GitHub + HuggingFace (2)
    - FDA:        query_fda when query mentions a drug-related keyword (1)
    - DB metadata fallback: RCSB/UniProt/HPA/STRING when bio identifiers
      are present in the prompt
    """
    import concurrent.futures as _cf

    if max_results is None:
        max_results = SEARCH_MAX_RESULTS

    # Use English keywords: if user_text has non-ASCII, translate via LLM
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

        sections: list = []
        logged: list = []

        def _invoke_one(tool_name: str, tool_input: dict) -> tuple:
            """Invoke a single search tool; return (tool_name, input, raw_str)."""
            tool = tools_dict.get(tool_name)
            if not tool:
                return (tool_name, tool_input, json.dumps({"success": False, "error": "tool not available"}))
            try:
                _logger.info("PI deep-research invoking: %s with %s", tool_name, tool_input)
                out = tool.invoke(tool_input)
                raw = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
                return (tool_name, tool_input, raw)
            except Exception as e:
                _logger.warning("PI %s failed: %s", tool_name, e)
                return (tool_name, tool_input, json.dumps({"success": False, "error": str(e)}))

        def _parse_results(raw_str: str) -> list:
            try:
                data = json.loads(raw_str) if isinstance(raw_str, str) and raw_str.strip().startswith("{") else {}
            except Exception:
                return []
            data = _extract_deepsearch_data(data) if isinstance(data, dict) else {}
            if not isinstance(data, dict) or data.get("success") is False:
                return []
            refs = (data.get("references") or data.get("results")
                    or data.get("papers") or data.get("datasets") or [])
            if isinstance(refs, str):
                try:
                    refs = json.loads(refs) if refs.strip().startswith("[") else []
                except Exception:
                    refs = []
            return refs if isinstance(refs, list) else []

        def _dedupe(items: list, key_fn) -> list:
            seen = set()
            out = []
            for it in items:
                try:
                    k = key_fn(it)
                except Exception:
                    k = id(it)
                if k in seen:
                    continue
                seen.add(k)
                out.append(it)
            return out

        # Build the parallel invocation plan
        LITERATURE = ["query_pubmed", "query_semantic_scholar", "query_arxiv", "query_biorxiv"]
        WEB = ["query_tavily", "query_duckduckgo"]
        DATASETS = ["query_github", "query_hugging_face"]

        # FDA only triggered when query looks drug/clinical-related
        FDA_KEYWORDS = ("drug", "compound", "inhibitor", "small molecule", "clinical",
                        "FDA", "approved", "trial", "adverse", "antibody", "biologic")
        plan: list[tuple[str, dict]] = []
        for tn in LITERATURE + WEB + DATASETS:
            plan.append((tn, {"query": query, "max_results": max_results}))
        if any(kw.lower() in user_text.lower() for kw in FDA_KEYWORDS) and "query_fda" in tools_dict:
            plan.append(("query_fda", {"query": query, "max_results": max_results}))

        # Parallel execution — bounded concurrency to be polite to upstream
        # rate limits but not so slow that everything is serialized.
        results_by_tool: dict[str, str] = {}
        with _cf.ThreadPoolExecutor(max_workers=min(6, len(plan))) as ex:
            futures = {ex.submit(_invoke_one, tn, ti): (tn, ti) for tn, ti in plan}
            for fut in _cf.as_completed(futures, timeout=30):
                try:
                    tn, ti, raw = fut.result()
                    results_by_tool[tn] = raw
                    logged.append((tn, ti, raw))
                except _cf.TimeoutError:
                    pass
                except Exception:
                    pass

        # Merge literature across PubMed + S2 + arXiv + bioRxiv
        lit_refs: list = []
        for tn in LITERATURE:
            lit_refs.extend(_parse_results(results_by_tool.get(tn, "")))
        lit_refs = _dedupe(lit_refs, lambda r: (
            (r.get("url") or r.get("doi") or r.get("pmid") or r.get("title") or "")[:120]
            if isinstance(r, dict) else str(r)[:120]
        ))
        if lit_refs:
            lines = _format_literature_citations(lit_refs[: max_results * 2], max_n=max_results * 2)
            if lines:
                productive = [t for t in LITERATURE if _parse_results(results_by_tool.get(t, ""))]
                source_tag = ", ".join(t.replace("query_", "") for t in productive) or "literature"
                sections.append(f"**Literature ({source_tag}; cite as [1], [2], ...)**\n" + "\n".join(lines))

        # Merge web across Tavily + DuckDuckGo
        web_refs: list = []
        for tn in WEB:
            web_refs.extend(_parse_results(results_by_tool.get(tn, "")))
        web_refs = _dedupe(web_refs, lambda r: (
            (r.get("url") or r.get("link") or r.get("title") or "")[:120]
            if isinstance(r, dict) else str(r)[:120]
        ))
        if web_refs:
            lines = _format_web_citations(web_refs[: max_results * 2], max_n=max_results * 2)
            if lines:
                productive_web = [t for t in WEB if _parse_results(results_by_tool.get(t, ""))]
                source_tag = ", ".join(t.replace("query_", "") for t in productive_web) or "web"
                sections.append(f"**Web ({source_tag}; cite as [1], [2], ...)**\n" + "\n".join(lines))

        # Merge datasets across GitHub + HuggingFace
        ds_refs: list = []
        for tn in DATASETS:
            ds_refs.extend(_parse_results(results_by_tool.get(tn, "")))
        ds_refs = _dedupe(ds_refs, lambda r: (
            (r.get("url") or r.get("html_url") or r.get("id") or r.get("name") or "")[:120]
            if isinstance(r, dict) else str(r)[:120]
        ))
        if ds_refs:
            lines = _format_dataset_citations(ds_refs[: max_results * 2], max_n=max_results * 2)
            if lines:
                productive_ds = [t for t in DATASETS if _parse_results(results_by_tool.get(t, ""))]
                source_tag = ", ".join(t.replace("query_", "") for t in productive_ds) or "datasets"
                sections.append(f"**Datasets ({source_tag}; cite as [1], [2], ...)**\n" + "\n".join(lines))

        # FDA (when triggered)
        if "query_fda" in results_by_tool:
            fda_refs = _parse_results(results_by_tool["query_fda"])
            if fda_refs:
                fda_lines = []
                for r in fda_refs[:max_results]:
                    if isinstance(r, dict):
                        nm = r.get("title") or r.get("name") or r.get("brand_name") or "FDA record"
                        url = r.get("url") or r.get("link") or ""
                        fda_lines.append(f"- [{nm}]({url})" if url else f"- {nm}")
                if fda_lines:
                    sections.append("**FDA (drug/clinical context)**\n" + "\n".join(fda_lines))

        # Structural / functional database fallback: when literature + web
        # + dataset search all returned nothing useful, parse the ORIGINAL
        # user query (not the truncated keyword form) for identifier
        # patterns and call the corresponding metadata endpoint. This is
        # what gets the report substantive content for prompts like
        # "Search for PDB 5XJH" where no paper indexes the PDB ID directly.
        if not sections:
            try:
                db_sections = _pi_database_fallback(user_text, tools_dict)
                if db_sections:
                    sections.append(
                        "**Database metadata (cite as [1], [2], ...)**\n"
                        + "\n\n".join(db_sections)
                    )
                    logged.append((
                        "__db_fallback__",
                        {"query": user_text},
                        json.dumps({"sections_added": len(db_sections)}),
                    ))
            except Exception as e:
                _logger.warning("PI DB fallback failed: %s", e)
                logged.append((
                    "__db_fallback__",
                    {"query": user_text},
                    json.dumps({"success": False, "error": str(e)}),
                ))

        if not sections:
            return ("No search results.", logged)
        intro = "References from search (use [1], [2], etc. in your report):\n\n"
        return (intro + "\n\n".join(sections), logged)
    except Exception as e:
        _logger.warning("PI report search failed: %s", e)
        return ("", [])
