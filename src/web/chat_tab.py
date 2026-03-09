import json
import os
import re
import aiohttp
import asyncio
import base64
import hashlib
import tempfile
import shutil
import time
import uuid
import numpy as np
import pandas as pd
import gradio as gr
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Mapping
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langchain_classic.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage, LLMResult
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_classic.schema.runnable import RunnablePassthrough
from langchain_classic.callbacks.base import BaseCallbackHandler
from langchain_core.prompt_values import ChatPromptValue
from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks import CallbackManagerForLLMRun

from pathlib import Path
from dotenv import load_dotenv
from gradio_client import Client, handle_file
from tools.tools_agent_hub import *
from agent.prompts import get_role_avatar_path, ROLE_DISPLAY_NAMES, MLS_SELF_CHECK_TEMPLATE
from agent.skills import get_skills_metadata_string, list_skill_ids
from agent.chat_agent import (
    initialize_session_state,
    update_llm_model,
    get_cached_tool_result,
    save_cached_tool_result,
    _merge_tool_parameters_with_context,
    _build_tools_description,
)
from web.utils.css_loader import load_css_file
import threading

# Load chat tab CSS from assets (cached at module level)
_CHAT_DIALOG_CSS = load_css_file("chat_tab_dialog.css")
_CHAT_PANEL_CSS = load_css_file("chat_tab_panel.css")
_CHAT_LOG_CSS = load_css_file("chat_tab_log.css")
AGENT_TAB_CSS = load_css_file("chat_tab_layout.css")

try:
    import markdown
except ImportErorr:
    markdown = None
load_dotenv()


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


def _markdown_to_html(text: str) -> str:
    """Convert Markdown to HTML for chat bubble display. Falls back to escaped plain text if markdown not available."""
    if not (text or text.strip()):
        return ""
    if markdown is None:
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
    try:
        html = markdown.markdown(
            text,
            extensions=["fenced_code", "tables"],
        )
        return html if html else text.replace("\n", "<br>")
    except Exception:
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")


# Agent logic (LLM, chains, session state, cache) lives in agent.chat_agent and agent.prompts.
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


def _extract_download_file_from_output(tool_name: str, output_data: dict) -> Optional[str]:
    """Extract local file path from tool output for download. Returns path if file exists."""
    if not isinstance(output_data, dict) or not output_data.get("success"):
        return None
    path = (
        output_data.get("pdb_path") or output_data.get("pdb_file")
        or output_data.get("structure_file") or output_data.get("fasta_file")
        or output_data.get("file_path") or output_data.get("generated_code_path")
        or output_data.get("model_path")
    )
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


def _embed_image_paths_in_content(content: str, image_paths: List[str], max_size_mb: float = 2.0) -> str:
    """Append inline images (as base64 data URLs) to content so they show in the chat message. Skips files larger than max_size_mb."""
    if not image_paths:
        return content
    max_bytes = int(max_size_mb * 1024 * 1024)
    for path in image_paths:
        if not path or not os.path.isfile(path):
            continue
        try:
            size = os.path.getsize(path)
            if size > max_bytes:
                content += f"\n\n*[Image omitted: {os.path.basename(path)} exceeds {max_size_mb} MB]*"
                continue
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            ext = os.path.splitext(path)[1].lower() or ".png"
            mime = "image/png" if ext == ".png" else "image/jpeg" if ext in (".jpg", ".jpeg") else "image/svg+xml" if ext == ".svg" else "image/webp" if ext == ".webp" else "image/png"
            content += f'\n\n<img src="data:{mime};base64,{b64}" alt="Plot" style="max-width:100%; height:auto; border-radius:8px;" />'
        except Exception:
            content += f"\n\n*[Could not load image: {os.path.basename(path)}]*"
    return content


def _get_download_btn_update(session_state: dict):
    """Return gr.update for the tool output download button."""
    path = session_state.get("latest_tool_output_file")
    if path and os.path.isfile(path):
        return gr.update(value=path, visible=True)
    return gr.update(visible=False)

def extract_uniprot_id_from_message(message: str) -> Optional[str]:
    """Extract UniProt ID from user message"""
    uniprot_pattern = r'\b[A-Z][A-Z0-9]{5}(?:[A-Z0-9]{4})?\b'
    matches = re.findall(uniprot_pattern, message.upper())
    return matches[0] if matches else None

def _assistant_msg(content: str, role_id: str) -> Dict[str, Any]:
    """Build assistant message with role label for display (separate dialog + avatar per agent)."""
    label = ROLE_DISPLAY_NAMES.get(role_id, role_id)
    display_content = f"**{label}**\n\n{content}"

    return {"role": "assistant", "content": display_content, "role_id": role_id}


# Cache role avatar as base64 data URL so we can embed in HTML (per-role avatar in chat)

_AVATAR_B64_CACHE: Dict[str, str] = {}
_DEFAULT_BOT_AVATAR_URL = "https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/venus_logo.png"


def _avatar_data_url(role_id: str) -> str:
    """Return a data URL for the role avatar image (base64), or default URL if not found."""
    if role_id in _AVATAR_B64_CACHE:
        return _AVATAR_B64_CACHE[role_id]
    path = get_role_avatar_path(role_id, fallback_to_first=True)
    if path and os.path.isfile(path):
        try:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            ext = os.path.splitext(path)[1].lower() or ".png"
            mime = "image/png" if ext == ".png" else "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
            data_url = f"data:{mime};base64,{b64}"
            _AVATAR_B64_CACHE[role_id] = data_url
            return data_url
        except Exception:
            pass

    return _DEFAULT_BOT_AVATAR_URL





def _build_chat_html(history_list: List[Dict[str, Any]]) -> str:
    """Build HTML for the chat so each assistant message uses the correct role avatar (ROLE_AVATAR_FILES)."""
    style_tag = f"<style>{_CHAT_DIALOG_CSS}</style>" if _CHAT_DIALOG_CSS else ""

    if not history_list:
        return f"{style_tag}<div class='vf-chat-dialog-box'><div class='vf-chat-container'><div class='vf-chat-empty'>No messages yet.</div></div></div>"
    html_parts = []
    for msg in history_list:
        role = (msg.get("role") or "").lower()
        content = msg.get("content") or ""
        role_id = msg.get("role_id") or ""
        content_html = _markdown_to_html(content)
        if role == "user":
            html_parts.append(
                "<div class='vf-chat-row vf-chat-user'>"
                "<div class='vf-chat-bubble vf-chat-user-bubble'>" + content_html + "</div>"
                "</div>"
            )

        else:
            avatar_url = _avatar_data_url(role_id) if role_id else _DEFAULT_BOT_AVATAR_URL
            label = ROLE_DISPLAY_NAMES.get(role_id, role_id or "Assistant")
            label_esc = label.replace("&", "&amp;").replace("<", "&lt;").replace("'", "&#39;")
            # Strip leading "**RoleName**\n\n" from bubble so role is only shown above bubble
            prefix = f"**{label}**"
            content_for_bubble = content.strip()
            if content_for_bubble.startswith(prefix):
                content_for_bubble = content_for_bubble[len(prefix):].lstrip("\n\r ")
            bubble_html = _markdown_to_html(content_for_bubble)
            html_parts.append(
                "<div class='vf-chat-row vf-chat-assistant'>"
                "<div class='vf-chat-assistant-side'><img class='vf-chat-avatar' src='"
                + avatar_url + "' alt='" + label_esc + "' /></div>"
                "<div class='vf-chat-assistant-main'>"
                "<div class='vf-chat-role-label'>" + label_esc + "</div>"
                "<div class='vf-chat-bubble vf-chat-assistant-bubble'>" + bubble_html + "</div>"
                "</div></div>"
            )

    return f"{style_tag}<div class='vf-chat-dialog-box'><div class='vf-chat-container'>" + "".join(html_parts) + "</div></div>"


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

def _format_literature_citations(refs: list, max_n: int = 5) -> list:
    """Format literature references as [n] [title](url) for PI citations (Markdown link format)."""
    out = []
    for i, r in enumerate((refs or [])[:max_n], 1):
        if not isinstance(r, dict):
            continue
        title = (r.get("title") or r.get("citation") or "No title").strip()
        authors = r.get("authors") or r.get("author") or ""
        if isinstance(authors, list):
            authors = ", ".join(str(a) for a in authors[:5])
        year = r.get("year") or r.get("published") or ""
        url = r.get("url") or r.get("link") or ""
        if url:
            out.append(f"[{i}] [{title}]({url})" + (f" — {authors}, {year}" if authors or year else ""))
        else:
            out.append(f"[{i}] {title}" + (f" — {authors}, {year}" if authors or year else ""))
    return out


def _format_literature_for_reading(refs: list, max_n: int = 5, abstract_max: int = 400) -> list:
    """Format literature with abstract so PI can read and cite. Each item: [n] [title](url) + Abstract."""
    out = []
    for i, r in enumerate((refs or [])[:max_n], 1):
        if not isinstance(r, dict):
            continue
        title = (r.get("title") or r.get("citation") or "No title").strip()
        authors = r.get("authors") or r.get("author") or ""
        if isinstance(authors, list):
            authors = ", ".join(str(a) for a in authors[:5])
        year = r.get("year") or r.get("published") or ""
        url = r.get("url") or r.get("link") or ""
        if url:
            line = f"[{i}] [{title}]({url})" + (f" — {authors}, {year}" if authors or year else "")
        else:
            line = f"[{i}] {title}" + (f" — {authors}, {year}" if authors or year else "")
        abstract = (r.get("abstract") or "").strip()
        if abstract:
            ab = abstract[:abstract_max] + ("…" if len(abstract) > abstract_max else "")
            line += f"\n  **Abstract:** {ab}"
        out.append(line)
    return out


def _format_web_citations(results: list, max_n: int = 5) -> list:
    """Format web search results as [n] [title](url) for PI citations (Markdown link format)."""
    out = []
    for i, r in enumerate((results or [])[:max_n], 1):
        if isinstance(r, dict):
            title = (r.get("title") or r.get("snippet") or str(r)).strip()
            url = r.get("url") or ""
            if url:
                out.append(f"[{i}] [{title}]({url})")
            else:
                out.append(f"[{i}] {title}")
        else:
            out.append(f"[{i}] {str(r)}")
    return out


def _format_web_for_reading(results: list, max_n: int = 5, snippet_max: int = 300) -> list:
    """Format web results with snippet so PI can read and cite. Each item: [n] [title](url)."""
    out = []
    for i, r in enumerate((results or [])[:max_n], 1):
        if isinstance(r, dict):
            title = (r.get("title") or "").strip()
            snippet = (r.get("snippet") or "").strip()
            url = r.get("url") or ""
            if url:
                line = f"[{i}] [{title or 'Link'}]({url})"
            else:
                line = f"[{i}] {title or str(r)}"
            if snippet:
                sn = snippet[:snippet_max] + ("…" if len(snippet) > snippet_max else "")
                line += f"\n  **Snippet:** {sn}"
            out.append(line)
        else:
            out.append(f"[{i}] {str(r)}")
    return out


def _format_dataset_citations(datasets: list, max_n: int = 5) -> list:
    """Format dataset search results as [n] [title](url) for PI citations (Markdown link format)."""
    out = []
    for i, d in enumerate((datasets or [])[:max_n], 1):
        if not isinstance(d, dict):
            continue
        title = (d.get("title") or "Untitled").strip()
        url = d.get("url") or ""
        src = d.get("source") or ""
        if url:
            out.append(f"[{i}] [{title}]({url})" + (f" — Source: {src}" if src else ""))
        else:
            out.append(f"[{i}] {title}" + (f" — Source: {src}" if src else ""))
    return out


def _format_pi_steps_for_report(intermediate_steps: list) -> str:
    """Format PI agent intermediate_steps (list of (action, observation)) into a single string for pi_report_chain input."""
    if not intermediate_steps:
        return "No search results."
    sections = []
    for action, observation in intermediate_steps:
        tname = getattr(action, "tool", None) or (action.get("tool") if isinstance(action, dict) else None)
        tinputs = getattr(action, "tool_input", None) or (action.get("tool_input", {}) if isinstance(action, dict) else {})
        query = (tinputs.get("query") or "") if isinstance(tinputs, dict) else ""
        obs_str = observation if isinstance(observation, str) else json.dumps(observation, ensure_ascii=False)
        try:
            data = json.loads(obs_str) if obs_str.strip().startswith("{") else {}
            if data.get("references"):
                lines = _format_literature_citations(data["references"], max_n=5)
                if lines:
                    sections.append(f"**Literature ({tname})**\n" + "\n".join(lines))
            elif data.get("results"):
                lines = _format_web_citations(data["results"] if isinstance(data["results"], list) else [], max_n=5)
                if lines:
                    sections.append(f"**Web ({tname})**\n" + "\n".join(lines))
            elif data.get("datasets"):
                ds_list = data["datasets"]
                if isinstance(ds_list, str):
                    try:
                        ds_list = json.loads(ds_list)
                    except json.JSONDecodeError:
                        ds_list = []
                lines = _format_dataset_citations(ds_list, max_n=5)
                if lines:
                    sections.append(f"**Datasets ({tname})**\n" + "\n".join(lines))
            else:
                sections.append(f"**{tname}** (query: {query[:80]}…)\n{obs_str[:800]}…")
        except Exception:
            sections.append(f"**{tname}** (query: {query[:80]}…)\n{obs_str[:800]}…")
    if not sections:
        return "Search returned no structured results."
    return "References from search (cite as [1], [2], etc.):\n\n" + "\n\n".join(sections)


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
            # Fallback: find first '[' and matching ']' within the chunk (handles extra text after array)
            i = chunk.find("[")
            if i != -1:
                depth = 0
                for j in range(i, len(chunk)):
                    if chunk[j] == "[":
                        depth += 1
                    elif chunk[j] == "]":
                        depth -= 1
                        if depth == 0:
                            out = _try_parse_json_array(chunk[i : j + 1])
                            if out is not None:
                                return out
                            break
    # 3) Find first '[' and matching ']' in full text (no string-awareness; best effort)
    i = text.find("[")
    if i == -1:
        return []
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "[":
            depth += 1
        elif text[j] == "]":
            depth -= 1
            if depth == 0:
                out = _try_parse_json_array(text[i : j + 1])
                if out is not None:
                    return out
                break
    return []


def _format_mls_code_for_display(tool_name: str, merged_tool_input: dict, raw_output: Any, max_code_len: int = 3000) -> str:
    """Extract MLS-written code for display in message. Returns markdown block or empty string."""
    if tool_name == "python_repl":
        code = (merged_tool_input or {}).get("query") or (merged_tool_input or {}).get("code") or ""
        if code:
            code_str = code.strip() if isinstance(code, str) else str(code)
            if len(code_str) > max_code_len:
                code_str = code_str[:max_code_len] + "\n# ... (truncated)"
            return f"\n\n**Code:**\n```python\n{code_str}\n```"
    if tool_name == "agent_generated_code":
        try:
            parsed = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
            path = parsed.get("generated_code_path") if isinstance(parsed, dict) else None
            if path and isinstance(path, str) and os.path.isfile(path):
                code_str = Path(path).read_text(encoding="utf-8").strip()
                if len(code_str) > max_code_len:
                    code_str = code_str[:max_code_len] + "\n# ... (truncated)"
                return f"\n\n**Generated code:**\n```python\n{code_str}\n```"
        except Exception:
            pass
    return ""


def _clean_title(text: str, max_line: int = 120) -> str:
    """Strip trailing ad/junk and truncate to max_line."""
    if not text or not isinstance(text, str):
        return ""
    t = text.strip()
    for junk in ("\n\nAd", "\nAd", " Ad", "— Ad", "Ad"):
        if t.endswith(junk):
            t = t[: -len(junk)].strip()
    if len(t) > max_line:
        t = t[: max_line].rstrip() + "…"
    return t


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


def _short_topic_title(user_text: str, max_len: int = 56) -> str:
    """Condense user question into a short topic title for the research draft heading."""
    if not user_text or not isinstance(user_text, str):
        return "Research draft"
    t = user_text.strip().replace("\n", " ").strip()
    if not t:
        return "Research draft"
    if len(t) <= max_len:
        return t
    return t[: max_len].rstrip() + "…"


def _format_search_preview(tool_name: str, observation_str: str, max_items: int = 5, max_line: int = 120) -> str:
    """Format retrieved content for chat: one item per block, title + link on separate lines."""
    try:
        data = json.loads(observation_str) if isinstance(observation_str, str) and observation_str.strip().startswith("{") else {}
    except Exception:
        return ""
    if not data.get("success"):
        return ""
    blocks = []
    if tool_name == "query_literature_by_keywords":
        refs = data.get("references") or []
        if isinstance(refs, str):
            try:
                refs = json.loads(refs) if refs.strip().startswith("[") else []
            except Exception:
                refs = []
        for i, r in enumerate((refs or [])[:max_items], 1):
            if not isinstance(r, dict):
                continue
            title = _clean_title(r.get("title") or r.get("citation") or "No title", max_line)
            url = (r.get("url") or r.get("link") or "").strip()
            if url:
                blocks.append(f"[{i}] [{title}]({url})")
            else:
                blocks.append(f"[{i}] {title}")
    elif tool_name == "query_web_by_keywords":
        res = data.get("results") or []
        if isinstance(res, str):
            try:
                res = json.loads(res) if res.strip().startswith("[") else []
            except Exception:
                res = []
        for i, r in enumerate((res or [])[:max_items], 1):
            if isinstance(r, dict):
                title = _clean_title(r.get("title") or r.get("snippet") or str(r), max_line)
                url = (r.get("url") or r.get("link") or "").strip()
                if url:
                    blocks.append(f"[{i}] [{title or 'Link'}]({url})")
                else:
                    blocks.append(f"[{i}] {title}")
            else:
                blocks.append(f"[{i}] {_clean_title(str(r), max_line)}")
    elif tool_name == "query_dataset_by_keywords":
        ds = data.get("datasets") or []
        if isinstance(ds, str):
            try:
                ds = json.loads(ds) if ds.strip().startswith("[") else []
            except Exception:
                ds = []
        for i, d in enumerate((ds or [])[:max_items], 1):
            if isinstance(d, dict):
                title = _clean_title(d.get("title") or "Untitled", max_line)
                url = (d.get("url") or d.get("link") or "").strip()
                if url:
                    blocks.append(f"[{i}] [{title}]({url})")
                else:
                    blocks.append(f"[{i}] {title}")
    if not blocks:
        return ""
    return "**Retrieved:**\n\n" + "\n\n".join(blocks)


def _format_search_summary(tool_name: str, tool_inputs: dict, observation_str: str) -> str:
    """Human-like summary: what we searched (tool + source), what we searched for (query); no results vs brief reading."""
    query = (tool_inputs.get("query") or "") if isinstance(tool_inputs, dict) else ""
    source = (tool_inputs.get("source") or "") if isinstance(tool_inputs, dict) else ""
    q = (query[:80] + "…") if len(query) > 80 else query
    label = tool_name.replace("_", " ").title()
    if source and source.lower() != "all":
        label = f"{label} ({source})"
    intro = f"**{label}** — searched for \"{q}\"." if q else f"**{label}**."
    try:
        data = json.loads(observation_str) if isinstance(observation_str, str) and observation_str.strip().startswith("{") else {}
    except Exception:
        return intro + " Error."
    if not data.get("success"):
        return intro + " Error."
    if tool_name == "query_literature_by_keywords":
        refs = data.get("references") or []
        if isinstance(refs, str):
            try:
                refs = json.loads(refs) if refs.strip().startswith("[") else []
            except Exception:
                refs = []
        refs = refs if isinstance(refs, list) else []
        n = len(refs)
        if not n:
            return intro + " No results."
        first = refs[0] if refs and isinstance(refs[0], dict) else {}
        title = (first.get("title") or first.get("citation") or "")[:100]
        if title:
            return intro + f" Found {n} paper(s). e.g. {title}…"
        return intro + f" Found {n} paper(s)."
    if tool_name == "query_web_by_keywords":
        res = data.get("results") or []
        if isinstance(res, str):
            try:
                res = json.loads(res) if res.strip().startswith("[") else []
            except Exception:
                res = []
        n = len(res) if isinstance(res, list) else (1 if isinstance(data.get("results"), str) and data.get("results") else 0)
        if not n:
            return intro + " No results."
        if isinstance(res, list) and res and isinstance(res[0], dict):
            first = res[0]
            title = (first.get("title") or first.get("snippet") or str(first))[:100]
            return intro + f" Found {n} result(s). e.g. {title}…"
        return intro + f" Found {n} result(s)."
    if tool_name == "query_dataset_by_keywords":
        ds = data.get("datasets") or []
        if isinstance(ds, str):
            try:
                ds = json.loads(ds) if ds.strip().startswith("[") else []
            except Exception:
                ds = []
        n = len(ds) if isinstance(ds, list) else 0
        if not n:
            return intro + " No results."
        return intro + f" Found {n} dataset(s)."
    return intro + " Done."


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
        lit_tool = next((t for t in tools if getattr(t, "name", "") == "query_literature_by_keywords"), None)
        web_tool = next((t for t in tools if getattr(t, "name", "") == "query_web_by_keywords"), None)
    except Exception:
        return ("Search tools unavailable.", [])
    sections = []
    logged = []
    if lit_tool:
        for source in ("pubmed", "semantic_scholar", "arxiv"):
            lit_input = {"query": query, "max_results": max_results, "source": source}
            try:
                print(f"\n[PI section] Invoking: `literature_search` with `{lit_input}`", flush=True)
                lit_out = lit_tool.invoke(lit_input)
                raw_lit = lit_out if isinstance(lit_out, str) else json.dumps(lit_out, ensure_ascii=False)
                logged.append(("query_literature_by_keywords", lit_input, raw_lit))
                data = json.loads(raw_lit) if isinstance(raw_lit, str) and raw_lit.strip().startswith("{") else {}
                if data.get("success") and data.get("references"):
                    refs = data["references"]
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
                logged.append(("query_literature_by_keywords", lit_input, json.dumps({"success": False, "error": str(e)})))
    if web_tool:
        for source in ("tavily", "duckduckgo"):
            web_input = {"query": query, "max_results": max_results, "source": source}
            try:
                print(f"\n[PI section] Invoking: `web_search` with `{web_input}`", flush=True)
                web_out = web_tool.invoke(web_input)
                raw_web = web_out if isinstance(web_out, str) else json.dumps(web_out, ensure_ascii=False)
                logged.append(("query_web_by_keywords", web_input, raw_web))
                data = json.loads(raw_web) if isinstance(raw_web, str) and raw_web.strip().startswith("{") else {}
                if data.get("success") and data.get("results"):
                    res = data["results"]
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
                logged.append(("query_web_by_keywords", web_input, json.dumps({"success": False, "error": str(e)})))
    if not sections:
        return ("No search results for this section.", logged)
    return ("References (cite as [1], [2], etc.):\n\n" + "\n\n".join(sections), logged)


def _fetch_search_for_pi_report(user_text: str, max_results: int = None) -> tuple:
    """Run literature_search, web_search, dataset_search. Default sources: pubmed, tavily, github. When a search returns empty or fails, retry with another source (e.g. web: tavily → duckduckgo; literature: pubmed → semantic_scholar → arxiv). Returns (combined_str_with_citations, list of (tool_name, inputs, raw_output))."""
    if max_results is None:
        max_results = SEARCH_MAX_RESULTS
    query = _refine_query_for_search(user_text, 80) or (user_text or "").strip()[:80]
    if not query:
        return ("", [])
    try:
        from tools.tools_agent_hub import get_tools
        tools = get_tools()
        lit_tool = next((t for t in tools if getattr(t, "name", "") == "query_literature_by_keywords"), None)
        web_tool = next((t for t in tools if getattr(t, "name", "") == "query_web_by_keywords"), None)
        dataset_tool = next((t for t in tools if getattr(t, "name", "") == "query_dataset_by_keywords"), None)
        sections = []
        logged = []

        # Literature: default pubmed; if empty/fail try semantic_scholar, then arxiv
        if lit_tool:
            for source in ("pubmed", "semantic_scholar", "arxiv"):
                lit_input = {"query": query, "max_results": max_results, "source": source}
                try:
                    print(f"\n[PI research] Invoking: `literature_search` with `{lit_input}`", flush=True)
                    lit_out = lit_tool.invoke(lit_input)
                    raw_lit = lit_out if isinstance(lit_out, str) else json.dumps(lit_out, ensure_ascii=False)
                    logged.append(("query_literature_by_keywords", lit_input, raw_lit))
                    data = json.loads(raw_lit) if isinstance(raw_lit, str) and raw_lit.strip().startswith("{") else {}
                    if data.get("success") and data.get("references"):
                        refs = data["references"] if isinstance(data["references"], list) else []
                        lines = _format_literature_citations(refs, max_n=5)
                        if lines:
                            sections.append("**Literature (cite as [1], [2], ...)**\n" + "\n".join(lines))
                            break
                    if not _is_search_result_empty("query_literature_by_keywords", data):
                        break
                except Exception as e:
                    print(f"[PI report] literature_search source={source} failed: {e}")
                    logged.append(("query_literature_by_keywords", lit_input, json.dumps({"success": False, "error": str(e)})))

        # Web: default tavily; if empty/fail try duckduckgo
        if web_tool:
            for source in ("tavily", "duckduckgo"):
                web_input = {"query": query, "max_results": max_results, "source": source}
                try:
                    print(f"\n[PI research] Invoking: `web_search` with `{web_input}`", flush=True)
                    web_out = web_tool.invoke(web_input)
                    raw_web = web_out if isinstance(web_out, str) else json.dumps(web_out, ensure_ascii=False)
                    logged.append(("query_web_by_keywords", web_input, raw_web))
                    data = json.loads(raw_web) if isinstance(raw_web, str) and raw_web.strip().startswith("{") else {}
                    if data.get("success") and data.get("results"):
                        res = data["results"]
                        if isinstance(res, list):
                            lines = _format_web_citations(res, max_n=5)
                            if lines:
                                sections.append("**Web (cite as [1], [2], ...)**\n" + "\n".join(lines))
                                break
                        elif isinstance(res, str) and res.strip():
                            sections.append("**Web**\n" + res[:1500])
                            break
                    if not _is_search_result_empty("query_web_by_keywords", data):
                        break
                except Exception as e:
                    print(f"[PI report] web_search source={source} failed: {e}")
                    logged.append(("query_web_by_keywords", web_input, json.dumps({"success": False, "error": str(e)})))

        # Dataset: default github; if empty/fail try hugging_face
        if dataset_tool:
            for source in ("github", "hugging_face"):
                ds_input = {"query": query, "max_results": max_results, "source": source}
                try:
                    print(f"\n[PI research] Invoking: `dataset_search` with `{ds_input}`", flush=True)
                    ds_out = dataset_tool.invoke(ds_input)
                    raw_ds = ds_out if isinstance(ds_out, str) else json.dumps(ds_out, ensure_ascii=False)
                    logged.append(("query_dataset_by_keywords", ds_input, raw_ds))
                    data = json.loads(raw_ds) if isinstance(raw_ds, str) and raw_ds.strip().startswith("{") else {}
                    if data.get("success") and data.get("datasets"):
                        ds_list = data["datasets"]
                        if isinstance(ds_list, str):
                            try:
                                ds_list = json.loads(ds_list)
                            except json.JSONDecodeError:
                                ds_list = []
                        if isinstance(ds_list, list):
                            lines = _format_dataset_citations(ds_list, max_n=5)
                            if lines:
                                sections.append("**Datasets (cite as [1], [2], ...)**\n" + "\n".join(lines))
                                break
                    if not _is_search_result_empty("query_dataset_by_keywords", data):
                        break
                except Exception as e:
                    print(f"[PI report] dataset_search source={source} failed: {e}")
                    logged.append(("query_dataset_by_keywords", ds_input, json.dumps({"success": False, "error": str(e)})))
        if not sections:
            return ("No search results.", logged)
        intro = "References from search (use [1], [2], etc. in your report):\n\n"
        return (intro + "\n\n".join(sections), logged)
    except Exception as e:
        print(f"[PI report] search failed: {e}")
        return ("", [])



async def send_message(message, session_state):
    """Async message handler with Planner-Worker-Finalizer workflow"""
    if session_state is None:
        session_state = initialize_session_state()
    # All agent execution files in one session-specific temp folder
    agent_session_dir = session_state.get("agent_session_dir") or os.path.join(
        os.getenv("TEMP_OUTPUTS_DIR", "temp_outputs"), "agent_sessions", session_state.get("session_id", str(uuid.uuid4()))
    )
    os.makedirs(agent_session_dir, exist_ok=True)
    current_time = time.localtime()
    time_stamped_subdir = os.path.join(
        str(current_time.tm_year),
        f"{current_time.tm_mon:02d}",
        f"{current_time.tm_mday:02d}",
        f"{current_time.tm_hour:02d}_{current_time.tm_min:02d}_{current_time.tm_sec:02d}"
    )
    UPLOAD_DIR = os.path.join(agent_session_dir, time_stamped_subdir)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    if not message or not message.get("text"):
        yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None), session_state
        return

    text = message["text"]
    files = message.get("files", [])


    # Setup file paths (Gradio 6: file_obj may be dict with "path" or string)
    file_paths = []
    if files:
        for file_obj in files:
            try:
                original_temp_path = file_obj.get("path", file_obj) if isinstance(file_obj, dict) else file_obj
                if not original_temp_path or not os.path.isfile(original_temp_path):
                    continue  # Skip dirs and invalid paths - avoids IsADirectoryError

                if os.path.exists(original_temp_path):
                    original_filename = os.path.basename(original_temp_path)
                    unique_filename = f"{original_filename}"
                    destination_path = os.path.join(UPLOAD_DIR, unique_filename)
                    shutil.copy2(original_temp_path, destination_path)
                    normalized_path = destination_path.replace('\\', '/')
                    file_paths.append(normalized_path)
                    session_state['temp_files'].append(normalized_path)
                else:
                    print(f"Warning: Gradio temp file not found at {original_temp_path}")
            except Exception as e:
                print(f"Error processing file: {e}")

    display_text = text
    if file_paths:
        file_names = ", ".join([os.path.basename(f) for f in file_paths])
        display_text += f"\n📎 *Attached: {file_names}*"
   

    # Free-use limit: max conversation messages per chat
    history_len = len(session_state.get("history") or [])
    if history_len >= AGENT_CHAT_MAX_MESSAGES:
        limit_msg = f"**Limit reached.** This chat has reached the maximum of {AGENT_CHAT_MAX_MESSAGES} messages. Start a new chat to continue."
        session_state["history"].append({"role": "assistant", "content": limit_msg, "role_id": "principal_investigator"})
        yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=True, file_count="multiple"), session_state
        return

    session_state['history'].append({"role": "user", "content": display_text})
    # LangChain: keep conversation_log for backend (conversation + tool records)
    session_state.setdefault("conversation_log", []).append({
        "role": "user", "content": display_text, "timestamp": datetime.now().isoformat()
    })

    protein_ctx = session_state['protein_context']
    sequence = extract_sequence_from_message(text)
    uniprot_id = extract_uniprot_id_from_message(text)
    if sequence:
        protein_ctx.add_sequence(sequence)
    if uniprot_id:
        protein_ctx.add_uniprot_id(uniprot_id)
    for fp in file_paths:
        protein_ctx.add_file(fp)

    # Proceed with Planner + Advisor (short memory removed)

    # Call Planner (PI: research, feasibility, plan)
    session_state['history'].append({"role": "assistant", "content": "🤔 Thinking... Creating a plan...", "role_id": "principal_investigator"})
    yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state

    # Build comprehensive context
    context_parts = []

    all_known_files = []
    if protein_ctx.files:
        for _, file_data in protein_ctx.files.items():
            if file_data.get('path'):
                all_known_files.append(file_data['path'])
    if file_paths:
        all_known_files.extend(file_paths)
    all_known_files = sorted(list(set(all_known_files)))
    if all_known_files:
        context_parts.append(f"Available uploaded files: {', '.join(all_known_files)}")
    if protein_ctx.structure_files:
        struct_info = []
        for struct_id, struct_data in protein_ctx.structure_files.items():
            struct_info.append(f"{struct_data['source']} structure: {struct_data['name']} (path: {struct_data['path']})")
        context_parts.append(f"Available structure files: {'; '.join(struct_info)}")

    context_parts.append(f"Protein context: {protein_ctx.get_context_summary()}")

    protein_context_summary = "; ".join(context_parts)

    memory = session_state['memory']
    # Include current user message so LangChain chains see this turn
    chat_history = memory.chat_memory.messages + [HumanMessage(content=display_text)]

    # Aggregate recent tool outputs and build unified tools JSON for Planner
    recent_tool_calls = getattr(protein_ctx, "tool_history", [])[-10:]
    tool_outputs_summary = []
    for call in reversed(recent_tool_calls):  # most recent first
        tool_outputs_summary.append({
            "step": call.get("step"),
            "tool": call.get("tool_name"),
            "inputs": call.get("inputs"),
            "cache_key": call.get("cache_key"),
            "cached": call.get("cached", False),
            "timestamp": call.get("timestamp").isoformat() if call.get("timestamp") else None,
            "outputs": call.get("outputs")[:500]

        })

    tool_records = protein_ctx.get_tool_records(limit=10)

    # --- Phase 1: Retrieval then PI report ---
    # Architecture: (1) Retrieval phase: search tools run (manual loop or fallback); each result is summarized in chat (English, one line). (2) Report phase: research draft then Suggest steps (new dialog) for CB/MLS.
    session_state['history'][-1] = {"role": "assistant", "content": "**Principal Investigator** is thinking…", "role_id": "principal_investigator"}
    yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
    current_tool_count = len(getattr(protein_ctx, "tool_history", []))
    pi_suggest_steps = ""
    appended_draft_and_steps = False

    if current_tool_count >= AGENT_CHAT_MAX_TOOL_CALLS:
        report_instruction = (
            "Tool call limit reached. User question: " + text + "\n\n"
            "Output a short research report: ## Abstract, ## Introduction, ## Related Work, ## Tools, ## Methods (steps for CB/MLS), ## References (list all [1], [2], … at the end). No JSON."
        )
        pi_report_inputs = {
            "input": report_instruction,
            "chat_history": [],
            "protein_context_summary": protein_context_summary,
            "tool_outputs": json.dumps(tool_outputs_summary, ensure_ascii=False),
        }
        try:
            pi_report = await asyncio.to_thread(session_state["pi_report"].invoke, pi_report_inputs)
        except Exception as e:
            print(f"[PI report] fallback invoke failed: {e}")
            pi_report = "Unable to generate report (tool limit reached)."
        session_state['history'][-1] = {"role": "assistant", "content": pi_report, "role_id": "principal_investigator"}
    else:
        # PI 5-section flow: plan sections → search + sub-report → draft → Suggest steps (new dialog) for CB/MLS
        intermediate_steps = []
        pi_report = ""
        pi_suggest_steps = ""
        appended_draft_and_steps = False
        final_content_from_llm = ""
        sections_list = []
        try:
            sections_out = await asyncio.to_thread(
                session_state["pi_sections"].invoke,
                {"input": text, "protein_context_summary": protein_context_summary},
            )
            sections_raw = (sections_out or "").strip() if isinstance(sections_out, str) else ""
            if "```" in sections_raw:
                start = sections_raw.find("```")
                if "json" in sections_raw[: start + 10]:
                    start = sections_raw.find("```") + 7
                else:
                    start = sections_raw.find("```") + 3
                end = sections_raw.find("```", start)
                sections_raw = sections_raw[start:end if end > 0 else None].strip()
            i = sections_raw.find("[")
            if i >= 0:
                depth = 0
                for j in range(i, len(sections_raw)):
                    if sections_raw[j] == "[":
                        depth += 1
                    elif sections_raw[j] == "]":
                        depth -= 1
                        if depth == 0:
                            sections_raw = sections_raw[i : j + 1]
                            break
            sections_list = json.loads(sections_raw)
            if not isinstance(sections_list, list) or len(sections_list) == 0:
                sections_list = []
            else:
                sections_list = sections_list[:5]
                sections_list = [
                    {
                        "section_name": (s.get("section_name") or "Section").strip() or "Section",
                        "search_query": (s.get("search_query") or text[:60]).strip() or text[:60],
                        "focus": (s.get("focus") or "both").strip() or "both",
                    }
                    for s in sections_list
                ]
        except Exception as e:
            print(f"[PI sections] parse failed: {e}")
            sections_list = []

        if sections_list:
            session_state["history"].pop()
            sub_reports_parts = []
            for idx, section in enumerate(sections_list):
                if len(protein_ctx.tool_history) >= AGENT_CHAT_MAX_TOOL_CALLS:
                    break
                section_title = section["section_name"]
                session_state["history"].append({"role": "assistant", "content": f"**Section {idx + 1}:** {section_title}", "role_id": "principal_investigator"})
                yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                session_state["history"].append({"role": "assistant", "content": "🔍 **Searching…**", "role_id": "principal_investigator"})
                yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                search_results_str, search_logged = await asyncio.to_thread(_run_section_search, section["search_query"])
                if session_state["history"] and session_state["history"][-1].get("content") == "🔍 **Searching…**":
                    session_state["history"].pop()
                for step_off, (tname, tinputs, toutputs) in enumerate(search_logged, start=len(protein_ctx.tool_history) + 1):
                    if len(protein_ctx.tool_history) >= AGENT_CHAT_MAX_TOOL_CALLS:
                        break
                    protein_ctx.add_tool_call(step_off, tname, tinputs, toutputs, cached=False)
                    session_state.setdefault("tool_executions", []).append({
                        "step": step_off, "tool_name": tname, "inputs": tinputs,
                        "outputs": (str(toutputs)[:1000] + "..." if len(str(toutputs)) > 1000 else str(toutputs)),
                        "timestamp": datetime.now().isoformat(),
                    })
                    summary_msg = _format_search_summary(tname, tinputs, str(toutputs))
                    preview = _format_search_preview(tname, str(toutputs))
                    content = summary_msg + ("\n\n" + preview if preview else "")
                    session_state["history"].append({"role": "assistant", "content": content, "role_id": "principal_investigator"})
                    yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                session_state["history"].append({"role": "assistant", "content": "✍️ **Writing sub-report…**", "role_id": "principal_investigator"})
                yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                try:
                    sub_report = await asyncio.to_thread(
                        session_state["pi_sub_report"].invoke,
                        {"section_name": section_title, "focus": section["focus"], "search_results": search_results_str},
                    )
                except Exception as e2:
                    sub_report = f"(Sub-report failed: {e2})"
                if session_state["history"] and session_state["history"][-1].get("content") == "✍️ **Writing sub-report…**":
                    session_state["history"].pop()
                # Use short-summary title for this sub-report (not "Sub-report" or section name)
                sub_report_title, sub_report_body = _parse_sub_report_short_title((sub_report or "").strip(), fallback_title=section_title)
                # Include references (with [1], [2], titles, links) so final report PI can see and cite them
                ref_block = f"**References:**\n{search_results_str}" if search_results_str and search_results_str.strip() else ""
                sub_block = f"**Sub-report:**\n{sub_report_body}" if sub_report_body else f"**Sub-report:**\n{(sub_report or '').strip()}"
                part = f"### {sub_report_title}\n\n{ref_block}\n\n{sub_block}" if ref_block else f"### {sub_report_title}\n\n{sub_block}"
                sub_reports_parts.append(part)
                # Show full sub-report in chat with short-summary title as heading (no truncation)
                full_sub = f"# {sub_report_title}\n\n{sub_report_body}" if sub_report_body else (sub_report or "").strip()
                session_state["history"].append({"role": "assistant", "content": full_sub, "role_id": "principal_investigator"})
                yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
            sub_reports_text = "\n\n".join(sub_reports_parts)
            session_state["history"].append({"role": "assistant", "content": "📝 **Principal Investigator** is writing the research draft…", "role_id": "principal_investigator"})
            yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
            try:
                pi_report = await asyncio.to_thread(
                    session_state["pi_final_report"].invoke,
                    {"sub_reports": sub_reports_text, "input": text},
                )
                pi_report = (pi_report or "").strip()
            except Exception as e2:
                print(f"[PI final report] invoke failed: {e2}")
                pi_report = "Report generation failed."
            # Replace bubble with actual research draft
            draft_title = (sections_list[0]["section_name"].strip() if sections_list else "") or _short_topic_title(text)
            if not draft_title:
                draft_title = "Research draft"
            draft_content = f"# {draft_title}\n\n{pi_report}"
            session_state["history"][-1] = {"role": "assistant", "content": draft_content, "role_id": "principal_investigator"}
            session_state.setdefault("conversation_log", []).append({
                "role": "assistant", "content": draft_content, "role_id": "principal_investigator",
                "timestamp": datetime.now().isoformat(),
            })
            yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
            # Generate Suggest steps (new dialog) for CB/MLS — show bubble then replace with content
            pi_suggest_steps = ""
            if pi_report and "Report generation failed" not in pi_report:
                session_state["history"].append({"role": "assistant", "content": "📋 **Principal Investigator** is generating Suggest steps…", "role_id": "principal_investigator"})
                yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                try:
                    available_tools_list = ", ".join(sorted(session_state.get("workers", {}).keys())) or "(none)"
                    available_skills_list = ", ".join(sorted(list_skill_ids())) or "(none)"
                    pi_suggest_steps = await asyncio.to_thread(
                        session_state["pi_suggest_steps"].invoke,
                        {
                            "draft_report": pi_report,
                            "input": text,
                            "available_tools_list": available_tools_list,
                            "available_skills_list": available_skills_list,
                        },
                    )
                    pi_suggest_steps = (pi_suggest_steps or "").strip()
                except Exception as e3:
                    print(f"[PI suggest steps] invoke failed: {e3}")
                if pi_suggest_steps:
                    session_state["history"][-1] = {"role": "assistant", "content": pi_suggest_steps, "role_id": "principal_investigator"}
                    session_state.setdefault("conversation_log", []).append({
                        "role": "assistant", "content": pi_suggest_steps, "role_id": "principal_investigator",
                        "timestamp": datetime.now().isoformat(),
                    })
                else:
                    session_state["history"].pop()
                yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                appended_draft_and_steps = True
        else:
            # Fallback: AgentExecutor then optional fixed search
            try:
                agent_result = await asyncio.to_thread(
                    session_state["pi_research_agent"].invoke,
                    {"input": text, "protein_context_summary": protein_context_summary},
                )
            except Exception as e:
                import traceback
                print(f"[PI research] AgentExecutor invoke failed: {e}")
                traceback.print_exc()
                agent_result = {}
            raw_steps = agent_result.get("intermediate_steps") or []
            steps_for_ui = []
            for action, observation in raw_steps:
                tname = getattr(action, "tool", None) or (action.get("tool") if isinstance(action, dict) else None)
                tinputs = getattr(action, "tool_input", None) or (action.get("tool_input", {}) if isinstance(action, dict) else {})
                obs_str = observation if isinstance(observation, str) else json.dumps(observation, ensure_ascii=False)
                if tname:
                    steps_for_ui.append((tname, tinputs, obs_str))
            if agent_result.get("output"):
                final_content_from_llm = (agent_result.get("output") or "").strip()
            if steps_for_ui:
                session_state["history"].pop()
            for step_off, (tname, tinputs, obs_str) in enumerate(steps_for_ui, start=len(protein_ctx.tool_history) + 1):
                if len(protein_ctx.tool_history) >= AGENT_CHAT_MAX_TOOL_CALLS:
                    break
                protein_ctx.add_tool_call(step_off, tname, tinputs, obs_str, cached=False)
                session_state.setdefault("tool_executions", []).append({
                    "step": step_off, "tool_name": tname, "inputs": tinputs,
                    "outputs": (obs_str[:1000] + "..." if len(obs_str) > 1000 else obs_str),
                    "timestamp": datetime.now().isoformat(),
                })
                class _Action:
                    tool = tname
                    tool_input = tinputs
                intermediate_steps.append((_Action(), obs_str))
                summary_msg = _format_search_summary(tname, tinputs, obs_str)
                preview = _format_search_preview(tname, obs_str)
                content = summary_msg + ("\n\n" + preview if preview else "")
                session_state["history"].append({"role": "assistant", "content": content, "role_id": "principal_investigator"})
                yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
            if not intermediate_steps:
                session_state["history"][-1] = {"role": "assistant", "content": "**Principal Investigator** is thinking (fallback search).", "role_id": "principal_investigator"}
                yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                search_results_str, search_logged = await asyncio.to_thread(_fetch_search_for_pi_report, text)
                if search_logged:
                    session_state["history"].pop()
                for step_off, (tname, tinputs, toutputs) in enumerate(search_logged, start=len(protein_ctx.tool_history) + 1):
                    if len(protein_ctx.tool_history) >= AGENT_CHAT_MAX_TOOL_CALLS:
                        break
                    protein_ctx.add_tool_call(step_off, tname, tinputs, toutputs, cached=False)
                    session_state.setdefault("tool_executions", []).append({
                        "step": step_off, "tool_name": tname, "inputs": tinputs,
                        "outputs": (str(toutputs)[:1000] + "..." if len(str(toutputs)) > 1000 else str(toutputs)),
                        "timestamp": datetime.now().isoformat(),
                    })
                    summary_msg = _format_search_summary(tname, tinputs, str(toutputs))
                    preview = _format_search_preview(tname, str(toutputs))
                    content = summary_msg + ("\n\n" + preview if preview else "")
                    session_state["history"].append({"role": "assistant", "content": content, "role_id": "principal_investigator"})
                    yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                    class _Action:
                        tool = tname
                        tool_input = tinputs
                    intermediate_steps.append((_Action(), toutputs))
                if search_results_str and search_results_str != "No search results.":
                    report_instruction = (
                        "The references below are numbered [1], [2], [3], etc. You MUST cite them in your report.\n\n"
                        + search_results_str
                        + "\n\nUser question: " + text + "\n\n"
                        "Output your research report with: ## Abstract, ## Introduction, ## Related Work (cite [1],[2],…), ## Tools, ## Methods (steps for CB/MLS), ## References (list all [1], [2], … at the end). No JSON."
                    )
                    try:
                        pi_report = await asyncio.to_thread(session_state["pi_report"].invoke, {
                            "input": report_instruction,
                            "chat_history": [],
                            "protein_context_summary": protein_context_summary,
                            "tool_outputs": json.dumps(tool_outputs_summary, ensure_ascii=False),
                        })
                    except Exception as e2:
                        print(f"[PI report] fallback invoke failed: {e2}")
                        pi_report = "Report generation failed."
            if intermediate_steps and not pi_report:
                search_results_str = _format_pi_steps_for_report(intermediate_steps)
                report_instruction = (
                    "The references below are numbered [1], [2], [3], etc. You MUST cite them in your report.\n\n"
                    + search_results_str
                    + "\n\nUser question: " + text + "\n\n"
                    "Output your research report with: ## Abstract, ## Introduction, ## Related Work (cite [1],[2],…), ## Tools, ## Methods (steps for CB/MLS), ## References (list all [1], [2], … at the end). No JSON."
                )
                try:
                    pi_report = await asyncio.to_thread(session_state["pi_report"].invoke, {
                        "input": report_instruction,
                        "chat_history": [],
                        "protein_context_summary": protein_context_summary,
                        "tool_outputs": json.dumps(tool_outputs_summary, ensure_ascii=False),
                    })
                except Exception as e2:
                    print(f"[PI report] invoke failed: {e2}")
                    pi_report = "Report generation failed."
        if not (pi_report or pi_report.strip()):
            pi_report = (final_content_from_llm or "").strip() or "No search results; no report generated."
        # Append single report message only when we did not show draft + suggest steps (5-section) and we did not replace the last message (tool-limit path)
        if not appended_draft_and_steps and current_tool_count < AGENT_CHAT_MAX_TOOL_CALLS:
            session_state["history"].append({"role": "assistant", "content": pi_report, "role_id": "principal_investigator"})
            yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state

    session_state["memory"].save_context({"input": display_text}, {"output": pi_report})
    if current_tool_count >= AGENT_CHAT_MAX_TOOL_CALLS:
        yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
    # Rebuild tool_outputs_summary after PI research so CB sees latest tool history (including PI's search calls)
    recent_tool_calls = getattr(protein_ctx, "tool_history", [])[-10:]
    tool_outputs_summary = []
    for call in reversed(recent_tool_calls):
        tool_outputs_summary.append({
            "step": call.get("step"),
            "tool": call.get("tool_name"),
            "inputs": call.get("inputs"),
            "cache_key": call.get("cache_key"),
            "cached": call.get("cached", False),
            "timestamp": call.get("timestamp").isoformat() if call.get("timestamp") else None,
            "outputs": call.get("outputs", "")[:500] if isinstance(call.get("outputs"), str) else str(call.get("outputs", ""))[:500],
        })
    yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
    # --- Phase 2: CB planning — from PI draft + Suggest steps to concrete pipeline (JSON steps) ---
    session_state["history"].append({"role": "assistant", "content": "📋 **Computational Biologist** is planning the pipeline…", "role_id": "computational_biologist"})
    yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
    cb_planner_inputs = {
        "pi_report": pi_report,
        "pi_suggest_steps": pi_suggest_steps or "",
        "protein_context_summary": protein_context_summary,
        "tool_outputs": json.dumps(tool_outputs_summary, ensure_ascii=False),
        "skills_metadata": get_skills_metadata_string(),
    }
    try:
        raw_msg = await asyncio.to_thread(session_state["cb_planner_raw"].invoke, cb_planner_inputs)
        content = getattr(raw_msg, "content", None) or str(raw_msg) or ""
        plan = _parse_cb_plan(content)
        if not plan and content:
            print(f"[CB planner] parsed empty plan; raw length={len(content)}, snippet={content[:300]!r}")
    except Exception as e:
        print(f"[CB planner] chain invoke failed: {e}")
        plan = []
    if not isinstance(plan, list):
        plan = []

    # Normalize plan: ensure each item has step, task_description, tool_name, tool_input (CB may return different keys)
    def _normalize_step(i: int, p: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(p, dict):
            return None
        step_num = p.get("step") or p.get("Step") or (i + 1)
        task_desc = (
            p.get("task_description") or p.get("Task_description")
            or p.get("task") or p.get("Task") or p.get("description") or ""
        )
        tool_name = p.get("tool_name") or p.get("Tool_name") or p.get("tool") or p.get("Tool") or p.get("name") or ""
        if not isinstance(tool_name, str):
            tool_name = str(tool_name) if tool_name else ""
        tool_input = p.get("tool_input") or p.get("Tool_input") or p.get("input") or p.get("params") or {}
        if not tool_name:
            return None
        if not isinstance(tool_input, dict):
            tool_input = {}
        return {"step": step_num, "task_description": task_desc, "tool_name": tool_name.strip(), "tool_input": tool_input}

    plan = [_normalize_step(i, p) for i, p in enumerate(plan)]
    plan = [p for p in plan if p is not None]
    # Search steps are already done by PI in the research phase; do not run them again
    plan = [p for p in plan if (p.get("tool_name") or "") not in PI_SEARCH_TOOL_NAMES]

    # Only show "No pipeline steps" when there truly are no steps; if PI suggested steps but plan is empty, parsing failed
    def _pi_suggested_steps_like_content(s: str) -> bool:
        if not s or not s.strip():
            return False
        t = s.strip().lower()
        if "no tools" in t and "no execution" in t:
            return False
        if "无需工具" in s or "无执行步骤" in s or "不需要工具" in s:
            return False
        for token in ("step", "步骤", "tools", "工具", "suggested", "1.", "2.", "3.", "一、", "二、", "read_skill", "uniprot", "biopython", "python_repl"):
            if token in t or token in s:
                return True
        return bool(s.count("\n") >= 2)  # multi-line suggest steps likely has a list

    if not plan:
        if _pi_suggested_steps_like_content(pi_suggest_steps or ""):
            fail_msg = (
                "**Computational Biologist** could not produce a valid pipeline from the suggested steps (parse failed or no tool_name in steps). "
                "Please try again or rephrase; the research draft above remains the answer."
            )
            if content:
                print(f"[CB planner] empty plan but PI suggested steps; raw snippet: {content[:500]!r}")
            session_state["history"][-1] = {"role": "assistant", "content": fail_msg, "role_id": "computational_biologist"}
        else:
            session_state["history"][-1] = {"role": "assistant", "content": "No pipeline steps to run; the research draft above is the answer.", "role_id": "computational_biologist"}
        yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=True, file_count="multiple"), session_state
        return

    # --- CB: replace planning bubble with pipeline, then MLS executes each step ---
    step_lines = [f"**Step {p['step']}.** {p['task_description']}" for p in plan]
    plan_text = "📋 **Pipeline**\n\nHere's what we'll do:\n\n" + "\n\n".join(step_lines)
    session_state["history"][-1] = {"role": "assistant", "content": plan_text, "role_id": "computational_biologist"}
    session_state.setdefault("conversation_log", []).append({
        "role": "assistant", "content": plan_text, "role_id": "computational_biologist",
        "timestamp": datetime.now().isoformat(),
    })
    yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
    step_results = {}
    analysis_log = ""
    collected_references = []

    for step in plan:
        # Free-use limit: max tool calls per chat
        if len(getattr(protein_ctx, "tool_history", [])) >= AGENT_CHAT_MAX_TOOL_CALLS:
            session_state["history"].append({
                "role": "assistant",
                "content": f"**Tool call limit reached.** This chat has used the maximum of {AGENT_CHAT_MAX_TOOL_CALLS} tool calls. Summary below based on results so far.",
                "role_id": "computational_biologist",
            })
            yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
            break

        step_num = step["step"]
        task_desc = step["task_description"]
        tool_name = step["tool_name"]
        tool_input = step["tool_input"]

        # MLS executes pipeline steps (CB only plans; MLS runs tools)
        skill_id = (tool_input or {}).get("skill_id", "") if tool_name == "read_skill" else ""
        if tool_name == "read_skill" and skill_id:
            load_skill_msg = f"📖 **MLS loading skill:** {skill_id}"
            session_state["history"].append({"role": "assistant", "content": load_skill_msg, "role_id": "machine_learning_specialist"})
            session_state.setdefault("conversation_log", []).append({
                "role": "assistant", "content": load_skill_msg, "role_id": "machine_learning_specialist",
                "timestamp": datetime.now().isoformat(),
            })
            yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
        session_state['history'].append({"role": "assistant", "content": f"⏳ **Executing Step {step_num}:** {task_desc}", "role_id": "machine_learning_specialist"})
        yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
        merged_tool_input = _merge_tool_parameters_with_context(protein_ctx, tool_input)
        # CRITICAL: Resolve dependencies BEFORE cache check and tool execution
        for key, value in list(merged_tool_input.items()):
            if isinstance(value, str) and value.startswith("dependency:"):
                parts = value.split(':')
                dep_step = int(parts[1].replace('step_', '').replace('step', ''))
                dep_raw_output = step_results[dep_step]['raw_output']
                if len(parts) > 2:
                    field_name = parts[2]
                    try:
                        parsed = json.loads(dep_raw_output) if isinstance(dep_raw_output, str) else dep_raw_output
                        merged_tool_input[key] = parsed.get(field_name, dep_raw_output)
                    except Exception:
                        merged_tool_input[key] = dep_raw_output
                else:
                    merged_tool_input[key] = dep_raw_output

        step_retry = 0
        step_done = False
        while step_retry <= MAX_STEP_RETRIES and not step_done:
            try:
                cached_entry = get_cached_tool_result(session_state, tool_name, merged_tool_input)

                if cached_entry:
                    raw_output = cached_entry.get("outputs", "")
                    step_results[step_num] = {'raw_output': raw_output, 'cached': True}
                    protein_ctx.add_tool_call(step_num, tool_name, merged_tool_input, raw_output, cached=True)
                    session_state.setdefault("tool_executions", []).append({
                        "step": step_num, "tool_name": tool_name, "inputs": merged_tool_input,
                        "outputs": (str(raw_output)[:1000] + "..." if len(str(raw_output)) > 1000 else str(raw_output)),
                        "cached": True, "timestamp": datetime.now().isoformat(),
                    })

                    # Collect references from cached outputs if present
                    try:
                        cached_outputs = cached_entry.get("outputs", {})
                        if isinstance(cached_outputs, dict) and cached_outputs.get("references"):
                            for rr in cached_outputs.get("references", []):
                                if rr and (rr.get("title") or rr.get("url")):
                                    collected_references.append(rr)
                    except Exception:
                        pass

                    # Every step (including cached): post-step self-check dialog
                    session_state["history"].append({
                        "role": "assistant",
                        "content": MLS_POST_STEP_SELF_CHECK_MSG.format(step_num=step_num),
                        "role_id": "machine_learning_specialist",
                    })
                    yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                    post_ok, post_retry, post_report = await _run_mls_post_step_verify(
                        session_state, step_num, task_desc, tool_name, merged_tool_input, raw_output,
                    )
                    if not post_ok:
                        if post_report:
                            session_state["history"].append({"role": "assistant", "content": f"**MLS report for CB:** {post_report}", "role_id": "machine_learning_specialist"})
                        session_state["history"].append({"role": "assistant", "content": f"**CB:** Step {step_num} (cached) verification failed. " + (post_report or "Output issue.") + " Re-plan or re-run if needed.", "role_id": "computational_biologist"})
                        yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=True), session_state
                        return

                    step_detail = f"**Step {step_num}:** {task_desc}\n\n"
                    step_detail += f"**Tool:** {tool_name} ⚡ (cached result)\n"
                    code_block = _format_mls_code_for_display(tool_name, merged_tool_input, raw_output)
                    if code_block:
                        step_detail += code_block + "\n\n"
                    step_detail += f"**Input:** {json.dumps(merged_tool_input, indent=2)}\n\n"
                    step_detail += f"**Cache Key:** {cached_entry.get('cache_key', 'N/A')}\n\n"
                    step_detail += f"**Output:**\n```\n{str(raw_output)[:500]}{'...' if len(str(raw_output)) > 500 else ''}\n```"
                    image_paths = _extract_image_paths_from_tool_output(raw_output, tool_name)
                    step_detail = _embed_image_paths_in_content(step_detail, image_paths)

                    analysis_log += f"--- Cached Analysis for Step {step_num}: {task_desc} ---\n\n"
                    analysis_log += f"Tool: {tool_name} (cached)\n"
                    analysis_log += f"Input: {json.dumps(merged_tool_input, indent=2)}\n"
                    analysis_log += f"Output: {raw_output}\n\n"

                    # Extract download file from cached output
                    try:
                        cached_data = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
                        if isinstance(cached_data, dict):
                            dl_path = _extract_download_file_from_output(tool_name, cached_data)
                            if dl_path:
                                session_state['latest_tool_output_file'] = dl_path
                    except Exception:
                        pass

                    out_path = _get_output_file_path_from_raw(raw_output, tool_name)
                    preview = _read_output_file_preview(out_path, max_lines=10) if out_path else ""
                    cb_match, cb_note = await _cb_post_step_check(session_state["llm"], step_num, task_desc, tool_name, raw_output, output_file_path=out_path, file_preview=preview if out_path else None)
                    if cb_match:
                        session_state['history'].append({"role": "assistant", "content": f"✅ **Step {step_num} Complete (cached):** {task_desc}\n\n{step_detail}", "role_id": "machine_learning_specialist"})
                        cb_msg = f"✓ **CB verified:** Step {step_num} complete (cached). Execution matches plan. Proceeding to next."
                        session_state['history'].append({"role": "assistant", "content": cb_msg, "role_id": "computational_biologist"})
                        yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                        step_done = True
                        break
                    # CB says cached output does not match — discuss with MLS
                    session_state["history"].append({"role": "assistant", "content": f"**CB:** Output does not match plan ({cb_note}). Re-execute with different parameters, another skill, or code.", "role_id": "computational_biologist"})
                    yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                    cb_retry, cb_report = await _run_mls_debug_with_tools(
                        session_state, step_num, task_desc, tool_name, merged_tool_input,
                        f"CB verification: output does not match plan. {cb_note} Re-execute with different parameters, another skill, or code.",
                    )
                    if cb_retry and step_retry < MAX_STEP_RETRIES:
                        step_retry += 1
                        merged_tool_input.update(cb_retry)
                        session_state["history"].append({"role": "assistant", "content": f"🔧 **MLS:** Retrying Step {step_num} (cached result rejected).", "role_id": "machine_learning_specialist"})
                        yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                        continue
                    if cb_report:
                        session_state["history"].append({"role": "assistant", "content": f"**MLS report for CB:** {cb_report}", "role_id": "machine_learning_specialist"})
                    session_state["history"].append({"role": "assistant", "content": f"**CB:** Step {step_num} output did not match plan. Re-plan or ask user to adjust.", "role_id": "computational_biologist"})
                    yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=True), session_state
                    return

                # Dependencies already resolved above, proceed to tool execution
                worker_executor = session_state['workers'].get(tool_name)
                if not worker_executor:
                    raise ValueError(f"Worker executor '{tool_name}' not found.")
                inputs_json_str = json.dumps(merged_tool_input, ensure_ascii=False, indent=2)

                agent_input_text = (
                    f"Execute this task: {task_desc}\n\n"
                    f"INPUT DATA (Use these parameters explicitly, DO NOT ASK for them):\n"
                    f"```json\n{inputs_json_str}\n```\n"
                    f"Command: Call the tool '{tool_name}' immediately using the data above."
                )


                print(f"\n[AgentExecutor] Step {step_num}: {tool_name}", flush=True)
                executor_result = None
                last_step_error = None
                mls_report_for_cb = None
                for attempt in range(2):
                    try:
                        executor_result = await asyncio.to_thread(
                            worker_executor.invoke,
                            {
                                "input": agent_input_text,
                                **merged_tool_input
                            }
                        )
                        break
                    except Exception as e:
                        last_step_error = e
                        if attempt == 0:
                            session_state["history"].append({"role": "assistant", "content": MLS_SELF_CHECK_MSG, "role_id": "machine_learning_specialist"})
                            yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                            retry_input, mls_report_for_cb = await _run_mls_debug_with_tools(
                                session_state,
                                step_num,
                                task_desc,
                                tool_name,
                                merged_tool_input,
                                str(e),
                            )
                            if session_state["history"] and session_state["history"][-1].get("content") == MLS_SELF_CHECK_MSG:
                                session_state["history"].pop()
                            if retry_input:
                                merged_tool_input.update(retry_input)
                                inputs_json_str = json.dumps(merged_tool_input, ensure_ascii=False, indent=2)
                                agent_input_text = (
                                    f"Execute this task: {task_desc}\n\n"
                                    f"INPUT DATA (Use these parameters explicitly, DO NOT ASK for them):\n"
                                    f"```json\n{inputs_json_str}\n```\n"
                                    f"Command: Call the tool '{tool_name}' immediately using the data above."
                                )
                                session_state["history"].append({"role": "assistant", "content": f"🔧 **MLS debug:** Retrying Step {step_num} with corrected parameters.", "role_id": "machine_learning_specialist"})
                                yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                                continue
                        break
                if executor_result is None and last_step_error is not None:
                    error_message = f"❌ **Error in Step {step_num}:** {task_desc}\n`{str(last_step_error)}`"
                    session_state['history'].append({"role": "assistant", "content": error_message, "role_id": "machine_learning_specialist"})
                    session_state["history"].append({"role": "assistant", "content": MLS_SELF_CHECK_MSG, "role_id": "machine_learning_specialist"})
                    yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                    retry_input, mls_report_for_cb = await _run_mls_debug_with_tools(
                        session_state,
                        step_num,
                        task_desc,
                        tool_name,
                        merged_tool_input,
                        str(last_step_error),
                    )
                    if session_state["history"] and session_state["history"][-1].get("content") == MLS_SELF_CHECK_MSG:
                        session_state["history"].pop()
                    if retry_input and step_retry < MAX_STEP_RETRIES:
                        step_retry += 1
                        merged_tool_input.update(retry_input)
                        session_state["history"].append({"role": "assistant", "content": f"🔧 **MLS self-check:** Retrying Step {step_num} with corrected parameters.", "role_id": "machine_learning_specialist"})
                        yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                        continue
                    if mls_report_for_cb:
                        session_state['history'].append({"role": "assistant", "content": f"**MLS report for CB:** {mls_report_for_cb}", "role_id": "machine_learning_specialist"})
                        session_state['history'].append({"role": "assistant", "content": f"**CB:** Step {step_num} failed. {mls_report_for_cb} Please re-plan the pipeline or ask the user to adjust.", "role_id": "computational_biologist"})
                    else:
                        session_state['history'].append({"role": "assistant", "content": f"**CB suggestion:** Step {step_num} failed. Check parameters or retry; ask again to adjust the pipeline if needed.", "role_id": "computational_biologist"})
                    yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=True), session_state
                    return
                if executor_result is None:
                    continue

                # Extract tool output from intermediate_steps (the actual tool return value)
                raw_output = ''
                if executor_result.get('intermediate_steps'):
                    last_step = executor_result['intermediate_steps'][-1]
                    if len(last_step) >= 2:
                        tool_output = last_step[1]
                        raw_output = tool_output if isinstance(tool_output, str) else json.dumps(tool_output, ensure_ascii=False)
                    else:
                        raw_output = executor_result.get('output', '')
                else:
                    raw_output = executor_result.get('output', '')
                step_results[step_num] = {'raw_output': raw_output, 'cached': False}

                # If tool returned success: false (top-level or nested in result/data), run MLS self-check and retry or report
                try:
                    output_failed, err_str = _tool_output_indicates_failure(raw_output)
                    if output_failed:
                        err_str = err_str or "Tool reported failure or error"
                        session_state["history"].append({"role": "assistant", "content": f"❌ **Step {step_num} failed (tool reported error):** {task_desc}\n`{err_str[:300]}`", "role_id": "machine_learning_specialist"})
                        session_state["history"].append({"role": "assistant", "content": MLS_SELF_CHECK_MSG, "role_id": "machine_learning_specialist"})
                        yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                        retry_input, mls_report_f = await _run_mls_debug_with_tools(
                            session_state,
                            step_num,
                            task_desc,
                            tool_name,
                            merged_tool_input,
                            err_str,
                        )
                        if session_state["history"] and session_state["history"][-1].get("content") == MLS_SELF_CHECK_MSG:
                            session_state["history"].pop()
                        if retry_input and step_retry < MAX_STEP_RETRIES:
                            step_retry += 1
                            merged_tool_input.update(retry_input)
                            session_state["history"].append({"role": "assistant", "content": f"🔧 **MLS self-check:** Retrying Step {step_num} with corrected parameters.", "role_id": "machine_learning_specialist"})
                            yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                            continue
                        if mls_report_f:
                            session_state['history'].append({"role": "assistant", "content": f"**MLS report for CB:** {mls_report_f}", "role_id": "machine_learning_specialist"})
                            session_state['history'].append({"role": "assistant", "content": f"**CB:** Step {step_num} failed. {mls_report_f} Please re-plan the pipeline or ask the user to adjust.", "role_id": "computational_biologist"})
                        else:
                            session_state['history'].append({"role": "assistant", "content": f"**CB suggestion:** Step {step_num} failed. Check parameters or retry; ask again to adjust the pipeline if needed.", "role_id": "computational_biologist"})
                        yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=True), session_state
                        return
                except Exception:
                    pass

                # Every step: post-step self-check (new dialog) — verify output before proceeding
                session_state["history"].append({
                    "role": "assistant",
                    "content": MLS_POST_STEP_SELF_CHECK_MSG.format(step_num=step_num),
                    "role_id": "machine_learning_specialist",
                })
                yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                post_ok, post_retry, post_report = await _run_mls_post_step_verify(
                    session_state, step_num, task_desc, tool_name, merged_tool_input, raw_output,
                )
                if not post_ok:
                    if post_retry and step_retry < MAX_STEP_RETRIES:
                        step_retry += 1
                        merged_tool_input.update(post_retry)
                        session_state["history"].append({"role": "assistant", "content": f"🔧 **Machine Learning Specialist self-check (post-step):** Step {step_num} output issue; retrying with adjusted parameters.", "role_id": "machine_learning_specialist"})
                        yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                        continue
                    if post_report:
                        session_state["history"].append({"role": "assistant", "content": f"**Machine Learning Specialist report for Computational Biologist:** {post_report}", "role_id": "machine_learning_specialist"})
                        session_state["history"].append({"role": "assistant", "content": f"**Computational Biologist:** Step {step_num} verification failed. {post_report} Please re-plan or ask user to adjust.", "role_id": "computational_biologist"})
                    else:
                        session_state["history"].append({"role": "assistant", "content": f"**Computational Biologist:** Step {step_num} post-step check found an issue. Re-plan or retry if needed.", "role_id": "computational_biologist"})
                    yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=True), session_state
                    return

                # Display training progress if this is a training tool
                if tool_name == 'train_protein_model_tool':
                    try:
                        parsed_output = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
                        if isinstance(parsed_output, dict) and 'training_progress' in parsed_output:
                            training_progress = parsed_output['training_progress']
                            if training_progress:
                                progress_display = f"**Training Progress:**\n```\n{training_progress}\n```"
                                session_state['history'].append({"role": "assistant", "content": f"⏳ **Step {step_num}:** {task_desc}\n\n{progress_display}", "role_id": "machine_learning_specialist"})
                                yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state

                    except Exception as e:
                        print(f"Could not parse training progress: {e}")


                # Try to parse and cache the result (only if successful)
                try:
                    try:
                        parsed_output = json.loads(raw_output)
                    except Exception:
                        parsed_output = raw_output

                    cached = save_cached_tool_result(session_state, tool_name, merged_tool_input, parsed_output)
                    if not cached:
                        print(f"⚠ Step {step_num} result not cached due to execution failure")

                    try:
                        if isinstance(parsed_output, dict) and parsed_output.get("references"):
                            for rr in parsed_output.get("references", []):
                                if rr and (rr.get("title") or rr.get("url")):
                                    collected_references.append(rr)
                    except Exception:
                        pass

                except Exception as e:
                    print(f"Failed to process result: {e}")

                # Always record tool call in history
                protein_ctx.add_tool_call(step_num, tool_name, merged_tool_input, raw_output, cached=False)
                session_state.setdefault("tool_executions", []).append({
                    "step": step_num, "tool_name": tool_name, "inputs": merged_tool_input,
                    "outputs": (str(raw_output)[:1000] + "..." if len(str(raw_output)) > 1000 else str(raw_output)),
                    "cached": False, "timestamp": datetime.now().isoformat(),
                })

                # Parse tool output to update context and download file
                try:
                    if tool_name in ['download_ncbi_sequence_by_accession', 'download_alphafold_structure_by_uniprot_id', 'download_uniprot_sequence_by_uniprot_id', 'query_interpro_by_uniprot_id',
                                     'protein_function_prediction', 'functional_residue_prediction',
                                     'protein_property_prediction', 'zero_shot_mutation_sequence_prediction', 'zero_shot_mutation_structure_prediction',
                                     'query_sequence_from_pdb_file', 'download_pdb_structure_by_pdb_id', 'query_literature_by_keywords', 'query_dataset_by_keywords', 'query_web_by_keywords',
                                     'esmfold_structure_prediction', 'query_foldseek_search_by_pdb_file']:

                        output_data = json.loads(raw_output)
                        if output_data.get('success') and 'file_path' in output_data:
                            file_path = output_data['file_path']
                            if tool_name == 'download_alphafold_structure_by_uniprot_id':
                                uniprot_id = merged_tool_input.get('uniprot_id', 'unknown')
                                protein_ctx.add_structure_file(file_path, 'alphafold', uniprot_id)
                            elif tool_name == 'download_ncbi_sequence_by_accession':
                                protein_ctx.add_file(file_path)
                        dl_path = _extract_download_file_from_output(tool_name, output_data)
                        if dl_path:
                            session_state['latest_tool_output_file'] = dl_path
                        if tool_name == 'query_literature_by_keywords' and isinstance(output_data, dict) and output_data.get('references'):
                            try:
                                for rr in output_data.get('references', []):
                                    if rr and (rr.get('title') or rr.get('url')):
                                        collected_references.append(rr)
                            except Exception:
                                pass
                except (json.JSONDecodeError, KeyError):
                    pass

                # Create detailed step result display
                step_detail = f"**Step {step_num}:** {task_desc}\n\n"
                step_detail += f"**Tool:** {tool_name}\n"
                code_block = _format_mls_code_for_display(tool_name, merged_tool_input, raw_output)
                if code_block:
                    step_detail += code_block + "\n\n"
                step_detail += f"**Input:** {json.dumps(merged_tool_input, indent=2)}\n\n"

                if tool_name == 'train_protein_model_tool':
                    try:
                        parsed_output = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
                        if isinstance(parsed_output, dict):
                            if parsed_output.get('success'):
                                step_detail += f"✅ **Training Completed Successfully!**\n\n"
                                if 'model_path' in parsed_output:
                                    step_detail += f"**Model saved to:** `{parsed_output['model_path']}`\n\n"

                                if 'training_progress' in parsed_output:
                                    step_detail += f"**Training Progress:**\n```\n{parsed_output['training_progress']}\n```\n\n"
                            else:
                                step_detail += f"❌ **Training Failed**\n\n"
                                if 'error' in parsed_output:
                                    step_detail += f"**Error:** {parsed_output['error']}\n\n"
                            if 'logs' in parsed_output:
                                step_detail += f"**Recent Logs:**\n```\n{parsed_output['logs']}\n```"
                    except Exception:
                        output_str = str(raw_output) if not isinstance(raw_output, str) else raw_output
                        step_detail += f"**Output:**\n```\n{output_str[:500]}{'...' if len(output_str) > 500 else ''}\n```"
                else:
                    output_str = str(raw_output) if not isinstance(raw_output, str) else raw_output
                    # Show more output for code/REPL tools so written code and plot paths are visible in chat
                    max_len = 3000 if tool_name in ("agent_generated_code", "python_repl") else 500
                    step_detail += f"**Output:**\n```\n{output_str[:max_len]}{'...' if len(output_str) > max_len else ''}\n```"

                image_paths = _extract_image_paths_from_tool_output(raw_output, tool_name)
                step_detail = _embed_image_paths_in_content(step_detail, image_paths)

                analysis_log += f"--- Analysis for Step {step_num}: {task_desc} ---\n\n"
                analysis_log += f"Tool: {tool_name}\n"
                analysis_log += f"Input: {json.dumps(merged_tool_input, indent=2)}\n"
                analysis_log += f"Output: {raw_output}\n\n"

                out_path = _get_output_file_path_from_raw(raw_output, tool_name)
                preview = _read_output_file_preview(out_path, max_lines=10) if out_path else ""
                cb_match, cb_note = await _cb_post_step_check(session_state["llm"], step_num, task_desc, tool_name, raw_output, output_file_path=out_path, file_preview=preview if out_path else None)
                if cb_match:
                    session_state['history'].append({"role": "assistant", "content": f"✅ **Step {step_num} Complete:** {task_desc}\n\n{step_detail}", "role_id": "machine_learning_specialist"})
                    cb_msg = f"✓ **CB verified:** Step {step_num} complete. Execution matches plan. Proceeding to next."
                    session_state['history'].append({"role": "assistant", "content": cb_msg, "role_id": "computational_biologist"})
                    yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                    step_done = True
                    break
                # CB says output does not match plan (null/empty/weird) — discuss with MLS and re-execute or report
                session_state["history"].append({"role": "assistant", "content": f"**CB:** Output does not match plan ({cb_note}). Re-execute this step with different parameters, another skill, or code.", "role_id": "computational_biologist"})
                yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                cb_retry, cb_report = await _run_mls_debug_with_tools(
                    session_state, step_num, task_desc, tool_name, merged_tool_input,
                    f"CB verification: output does not match plan. {cb_note} Re-execute with different parameters, another skill, or code.",
                )
                if cb_retry and step_retry < MAX_STEP_RETRIES:
                    step_retry += 1
                    merged_tool_input.update(cb_retry)
                    session_state["history"].append({"role": "assistant", "content": f"🔧 **MLS:** Retrying Step {step_num} with adjusted parameters/skill/code.", "role_id": "machine_learning_specialist"})
                    yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                    continue
                if cb_report:
                    session_state["history"].append({"role": "assistant", "content": f"**MLS report for CB:** {cb_report}", "role_id": "machine_learning_specialist"})
                session_state["history"].append({"role": "assistant", "content": f"**CB:** Step {step_num} output did not match plan. {cb_note} Re-plan or ask user to adjust.", "role_id": "computational_biologist"})
                yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=True), session_state
                return

            except Exception as e:
                error_message = f"❌ **Error in Step {step_num}:** {task_desc}\n`{str(e)}`"
                session_state['history'].append({"role": "assistant", "content": error_message, "role_id": "machine_learning_specialist"})
                session_state["history"].append({"role": "assistant", "content": MLS_SELF_CHECK_MSG, "role_id": "machine_learning_specialist"})
                yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                retry_input, mls_report_for_cb = await _run_mls_debug_with_tools(
                    session_state,
                    step_num,
                    task_desc,
                    tool_name,
                    merged_tool_input,
                    str(e),
                )
                if session_state["history"] and session_state["history"][-1].get("content") == MLS_SELF_CHECK_MSG:
                    session_state["history"].pop()
                if retry_input and step_retry < MAX_STEP_RETRIES:
                    step_retry += 1
                    merged_tool_input.update(retry_input)
                    session_state["history"].append({"role": "assistant", "content": f"🔧 **Machine Learning Specialist self-check:** Retrying Step {step_num} with corrected parameters.", "role_id": "machine_learning_specialist"})
                    yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                    continue
                if mls_report_for_cb:
                    session_state['history'].append({"role": "assistant", "content": f"**Machine Learning Specialist report for Computational Biologist:** {mls_report_for_cb}", "role_id": "machine_learning_specialist"})
                    session_state['history'].append({"role": "assistant", "content": f"**Computational Biologist:** Step {step_num} failed. {mls_report_for_cb} Please re-plan the pipeline or ask the user to adjust.", "role_id": "computational_biologist"})
                else:
                    session_state['history'].append({"role": "assistant", "content": f"**Computational Biologist suggestion:** Step {step_num} failed. Check parameters or retry; ask again to adjust the pipeline if needed.", "role_id": "computational_biologist"})
                yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=True), session_state
                return

    # SC summarizes only after all steps (separate dialog + avatar)
    session_state['history'].append({"role": "assistant", "content": "📄 **All steps complete. Generating final report...**", "role_id": "scientific_critic"})
    yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state

    # Build full run record so SC sees all agent outputs and tool executions
    run_parts = []
    for h in session_state.get("history") or []:
        role = h.get("role")
        role_id = h.get("role_id") or role
        content = (h.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            run_parts.append("**User:**\n" + content)
        else:
            label = ROLE_DISPLAY_NAMES.get(role_id, (role_id or "Assistant").replace("_", " ").title())
            run_parts.append(f"**{label}:**\n{content}")
    run_parts.append("\n## Tool executions")
    for t in session_state.get("tool_executions") or []:
        run_parts.append(f"\n### {t.get('step')}. {t.get('tool_name', '')}")
        run_parts.append("Input: " + json.dumps(t.get("inputs", {}), ensure_ascii=False, indent=2))
        out_str = str(t.get("outputs", ""))
        run_parts.append("Output: " + (out_str[:2000] + "..." if len(out_str) > 2000 else out_str))
    full_run_record = "\n\n---\n\n".join(run_parts)

    finalizer_inputs = {
        "original_input": text,
        "full_run_record": full_run_record,
        "analysis_log": analysis_log,
        "references": json.dumps(_dedupe_references(collected_references), ensure_ascii=False)
    }

    final_response = await asyncio.to_thread(
        session_state['finalizer'].invoke,
        finalizer_inputs
    )
    session_state['history'][-1] = {"role": "assistant", "content": final_response, "role_id": "scientific_critic"}
    session_state['memory'].save_context({"input": display_text}, {"output": final_response})
    session_state.setdefault("conversation_log", []).append({
        "role": "assistant", "content": final_response, "role_id": "scientific_critic",
        "timestamp": datetime.now().isoformat(),
    })
    yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=True, file_count="multiple"), session_state



def _format_backend_log(session_state: Dict[str, Any]) -> str:
    """Format conversation_log and tool_executions for backend display (compact, clear structure)."""
    if not session_state:
        return "_No session._"
    conv = session_state.get("conversation_log") or []
    tools = session_state.get("tool_executions") or []
    max_conv, max_tools = 20, 40
    conv = conv[-max_conv:] if len(conv) > max_conv else conv
    tools = tools[-max_tools:] if len(tools) > max_tools else tools
    lines = ["*Last 20 messages, 40 tool runs.*", "", "### Conversation", ""]
    for i, entry in enumerate(conv, 1):
        role = entry.get("role", "")
        role_id = entry.get("role_id", "")
        ts = (entry.get("timestamp") or "")[:19]
        content = entry.get("content") or ""
        label = f"{role_id}" if role_id else role
        lines.append(f"**{i}. [{ts}] {label}**")
        lines.append("")
        lines.append(content)
        lines.append("")

    lines.append("### Tool executions")
    lines.append("")
    for j, t in enumerate(tools, 1):
        step = t.get("step", j)
        name = t.get("tool_name", "")
        cached = " (cached)" if t.get("cached") else ""
        ts = (t.get("timestamp") or "")[:19]
        inp = t.get("inputs", {})
        inp_str = json.dumps(inp, ensure_ascii=False)
        if len(inp_str) > 180:
            inp_str = inp_str[:180] + "..."
        out = str(t.get("outputs", ""))[:180] + ("..." if len(str(t.get("outputs", ""))) > 180 else "")
        if name == "read_skill" and isinstance(inp, dict) and inp.get("skill_id"):
            lines.append(f"**{j}.** 📖 **Loaded skill:** `{inp.get('skill_id')}`{cached} — {ts}")
        else:
            lines.append(f"**{j}.** `{name}`{cached} — {ts}")
        lines.append(f"In: `{inp_str}`")
        lines.append(f"Out: `{out}`")
        lines.append("")
    return "\n".join(lines).strip()

def _build_conversation_panel_html(history_list: List[Dict[str, Any]]) -> str:
    """Main panel HTML: conversation area above input. Large min-height so input sits near bottom, minimal whitespace below."""
    panel_style_tag = f"<style>{_CHAT_PANEL_CSS}</style>" if _CHAT_PANEL_CSS else ""
    if not history_list:
        empty_html = (
            '<div class="vf-empty-state">'
            '<div class="vf-empty-title">No messages yet</div>'
            '<div class="vf-empty-hint">Type below to start a conversation.</div>'
            '</div>'
        )
        return f"{panel_style_tag}<div class=\"vf-right-panel\">{empty_html}</div>"
    chat_html = _build_chat_html(history_list)
    return (
        f"{panel_style_tag}"
        '<div class="vf-right-panel" style="height:100%;min-height:77vh;">'
        f'<div class="vf-conversation-panel">{chat_html}</div>'
        "</div>"
    )


def _build_log_html(session_state: Dict[str, Any]) -> str:
    """Log panel HTML for sidebar. Styled with colors so it reads clearly as a log."""
    log_style_tag = f"<style>{_CHAT_LOG_CSS}</style>" if _CHAT_LOG_CSS else ""
    log_md = _format_backend_log(session_state or {})
    log_html = _markdown_to_html(log_md)
    return f"{log_style_tag}<div class=\"vf-log-panel\">{log_html}</div>"


def create_chat_tab(constant: Dict[str, Any]) -> Dict[str, Any]:
    """Create the chat tab: left = main area (conversation + input at bottom); right = sidebar (examples, tips, log)."""
    session_state = gr.State(value=None)  # Lazy-init in send_message (State cannot deepcopy LLM/chains)
    gr.Markdown("""
    💡 *This is a academic demo version of VenusFactory (Max PI search 3, Max tools 40), please [local deployment](https://github.com/AI4Protein/VenusFactory) to unlock the full features or visit [project version](https://venusfactory.cn).*
    """)
    with gr.Row(equal_height=True, elem_id="vf-agent-page-row"):
        # Left (main): conversation area + input row at bottom
        with gr.Column(scale=3, min_width=200, elem_id="vf-agent-main-column"):
            with gr.Group(elem_id="vf-agent-main-group"):
                right_panel_display = gr.HTML(
                    value=_build_conversation_panel_html([]),
                    elem_id="vf-right-panel-root",
                )
                with gr.Row(elem_id="vf-agent-input-row"):
                    model_selector = gr.Dropdown(choices=["ChatGPT-4o", "Gemini-2.5-Pro", "Claude-3.7", "DeepSeek-R1"], value="Gemini-2.5-Pro", show_label=False, scale=2)
                    chat_input = gr.MultimodalTextbox(interactive=True, file_count="multiple", placeholder="💬 Ask me anything about AI protein.", show_label=False, file_types=[".fasta", ".fa", ".pdb", ".csv"], scale=8)
        # Sidebar: log, examples, tips (references chat_input)
        with gr.Column(scale=2, min_width=200, elem_id="vf-agent-sidebar-column"):
            with gr.Group():
                with gr.Accordion("📋 Backend Log", open=True, elem_id="vf-agent-backend-log"):
                    log_display = gr.HTML(value=_build_log_html({}), elem_id="vf-log-display")
                with gr.Accordion("💡 Example Research Questions", open=True):
                    gr.Examples(
                        examples=[
                            ["For UniProt P04040 (SOD1): search the literature for stability-related mutations and mechanisms, download the AlphaFold structure, then run in silico saturation mutagenesis to rank single-point mutations by predicted stability change and suggest top candidates for wet-lab validation."],
                            ["I have a FASTA file of protein sequences: extract UniProt IDs, look up InterPro domains for each, run protein function prediction, then suggest which sequences are best suited for a custom property prediction model and outline the training pipeline."],
                            ["For PDB 1A00: get chain sequences and run FoldSeek to find structurally similar proteins; download the AlphaFold structure of one similar UniProt entry, predict protein property (e.g. stability) for both, and compare the results with a brief literature context."]
                        ],
                        inputs=chat_input,
                        label=None
                    )
                with gr.Accordion("✨ Tips for Prompting VenusFactory", open=True):
                    gr.Markdown("""
**Capabilities**: Sequence/structure analysis, function prediction, mutation impact, DB queries (UniProt/PDB/NCBI).

**Input**: Paste FASTA, give UniProt/PDB ID, or upload .fasta/.pdb/.csv. State your goal clearly.

**Auto workflow**: Decompose tasks → run tools → integrate results.
                    """)

    model_selector.change(
        fn=update_llm_model,
        inputs=[model_selector, session_state],
        outputs=[session_state]
    )

    chat_input.submit(
        fn=send_message,
        inputs=[chat_input, session_state],
        outputs=[right_panel_display, log_display, chat_input, session_state],
        concurrency_limit=3,
        show_progress="full"
    )

    return {
        "chatbot": right_panel_display,
        "chat_input": chat_input,
        "session_state": session_state
    }

if __name__ == "__main__":
    components = create_chat_tab({})
    with gr.Blocks(theme=gr.themes.Soft(), css=AGENT_TAB_CSS or ".gradio-container {max-width: 95% !important;}") as demo:
        create_chat_tab({})

    demo.queue(
        concurrency_count=3,  
        max_size=20,
        api_open=False
    )

    demo.launch(
        share=True,
        max_threads=40,  
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )