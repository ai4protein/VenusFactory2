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
from langchain_classic.memory import ConversationBufferWindowMemory
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
from agent.prompts import get_role_avatar_path, ROLE_DISPLAY_NAMES
from agent.chat_agent import (
    initialize_session_state,
    update_llm_model,
    get_cached_tool_result,
    save_cached_tool_result,
    _merge_tool_parameters_with_context,
    _build_tools_description,
)
from web.utils.chat_helpers import handle_feedback_submit
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
    path = output_data.get("pdb_path") or output_data.get("pdb_file") or output_data.get("fasta_file") or output_data.get("file_path")
    if path and isinstance(path, str) and os.path.isfile(path):
        return path
    return None


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
            html_parts.append(
                "<div class='vf-chat-row vf-chat-assistant'>"
                "<img class='vf-chat-avatar' src='" + avatar_url + "' alt='" + label_esc + "' />"
                "<div class='vf-chat-bubble vf-chat-assistant-bubble'>"
                "<span class='vf-chat-role-label'>" + label_esc + "</span><br><br>" + content_html
                + "</div></div>"
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
        lit_tool = next((t for t in tools if getattr(t, "name", "") == "literature_search"), None)
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
    """Ask MLS to analyze the error and either suggest corrected tool_input to retry or a report for CB to re-plan. Returns (retry_input_dict or None, report_for_cb or None)."""
    prompt = (
        f"Step {step_num} failed during tool execution.\n\n"
        f"**Task:** {task_desc}\n**Tool:** {tool_name}\n**Current input:** {json.dumps(merged_tool_input, ensure_ascii=False)}\n**Error:** {error_str}\n\n"
        "You are the Machine Learning Specialist. Analyze the error. "
        "If the fix is to correct or fill in parameters (e.g. empty string where an integer is required, wrong key), output a JSON object with key \"retry_input\" whose value is a dict of only the parameters to change or the full corrected tool_input. "
        "If the pipeline or tool cannot be fixed without CB re-planning, output a JSON object with key \"report_for_cb\" and value a short string explaining the problem and what CB should do (e.g. which parameter is missing, or that the tool needs different inputs). "
        "Output only one JSON object, no other text."
    )
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = (response.content if hasattr(response, "content") else str(response)).strip()
        if not content:
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
        data = json.loads(content)
        retry = data.get("retry_input") if isinstance(data.get("retry_input"), dict) else None
        report = data.get("report_for_cb") if isinstance(data.get("report_for_cb"), str) else None
        return (retry, report)
    except Exception:
        return (None, None)


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
    if tool_name == "literature_search":
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
    elif tool_name == "web_search":
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
    elif tool_name == "dataset_search":
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
    if tool_name == "literature_search":
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
    if tool_name == "web_search":
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
    if tool_name == "dataset_search":
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
    if tool_name == "literature_search":
        refs = data.get("references") or []
        if isinstance(refs, str):
            try:
                refs = json.loads(refs) if refs.strip().startswith("[") else []
            except Exception:
                refs = []
        return not (isinstance(refs, list) and len(refs) > 0)
    if tool_name == "web_search":
        res = data.get("results") or []
        return not (isinstance(res, list) and len(res) > 0) and not (isinstance(data.get("results"), str) and data.get("results"))
    if tool_name == "dataset_search":
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
        lit_tool = next((t for t in tools if getattr(t, "name", "") == "literature_search"), None)
        web_tool = next((t for t in tools if getattr(t, "name", "") == "web_search"), None)
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
                logged.append(("literature_search", lit_input, raw_lit))
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
                logged.append(("literature_search", lit_input, json.dumps({"success": False, "error": str(e)})))
    if web_tool:
        for source in ("tavily", "duckduckgo"):
            web_input = {"query": query, "max_results": max_results, "source": source}
            try:
                print(f"\n[PI section] Invoking: `web_search` with `{web_input}`", flush=True)
                web_out = web_tool.invoke(web_input)
                raw_web = web_out if isinstance(web_out, str) else json.dumps(web_out, ensure_ascii=False)
                logged.append(("web_search", web_input, raw_web))
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
                logged.append(("web_search", web_input, json.dumps({"success": False, "error": str(e)})))
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
        lit_tool = next((t for t in tools if getattr(t, "name", "") == "literature_search"), None)
        web_tool = next((t for t in tools if getattr(t, "name", "") == "web_search"), None)
        dataset_tool = next((t for t in tools if getattr(t, "name", "") == "dataset_search"), None)
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
                    logged.append(("literature_search", lit_input, raw_lit))
                    data = json.loads(raw_lit) if isinstance(raw_lit, str) and raw_lit.strip().startswith("{") else {}
                    if data.get("success") and data.get("references"):
                        refs = data["references"] if isinstance(data["references"], list) else []
                        lines = _format_literature_citations(refs, max_n=5)
                        if lines:
                            sections.append("**Literature (cite as [1], [2], ...)**\n" + "\n".join(lines))
                            break
                    if not _is_search_result_empty("literature_search", data):
                        break
                except Exception as e:
                    print(f"[PI report] literature_search source={source} failed: {e}")
                    logged.append(("literature_search", lit_input, json.dumps({"success": False, "error": str(e)})))

        # Web: default tavily; if empty/fail try duckduckgo
        if web_tool:
            for source in ("tavily", "duckduckgo"):
                web_input = {"query": query, "max_results": max_results, "source": source}
                try:
                    print(f"\n[PI research] Invoking: `web_search` with `{web_input}`", flush=True)
                    web_out = web_tool.invoke(web_input)
                    raw_web = web_out if isinstance(web_out, str) else json.dumps(web_out, ensure_ascii=False)
                    logged.append(("web_search", web_input, raw_web))
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
                    if not _is_search_result_empty("web_search", data):
                        break
                except Exception as e:
                    print(f"[PI report] web_search source={source} failed: {e}")
                    logged.append(("web_search", web_input, json.dumps({"success": False, "error": str(e)})))

        # Dataset: default github; if empty/fail try hugging_face
        if dataset_tool:
            for source in ("github", "hugging_face"):
                ds_input = {"query": query, "max_results": max_results, "source": source}
                try:
                    print(f"\n[PI research] Invoking: `dataset_search` with `{ds_input}`", flush=True)
                    ds_out = dataset_tool.invoke(ds_input)
                    raw_ds = ds_out if isinstance(ds_out, str) else json.dumps(ds_out, ensure_ascii=False)
                    logged.append(("dataset_search", ds_input, raw_ds))
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
                    if not _is_search_result_empty("dataset_search", data):
                        break
                except Exception as e:
                    print(f"[PI report] dataset_search source={source} failed: {e}")
                    logged.append(("dataset_search", ds_input, json.dumps({"success": False, "error": str(e)})))
        if not sections:
            return ("No search results.", logged)
        intro = "References from search (use [1], [2], etc. in your report):\n\n"
        return (intro + "\n\n".join(sections), logged)
    except Exception as e:
        print(f"[PI report] search failed: {e}")
        return ("", [])



async def send_message(message, session_state):

    """Async message handler with Planner-Worker-Finalizer workflow"""
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
        yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None)
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
        yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=True, file_count="multiple")
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
    yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False)

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
    yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False)
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
                yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False)
                session_state["history"].append({"role": "assistant", "content": "🔍 **Searching…**", "role_id": "principal_investigator"})
                yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False)
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
                    yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False)
                session_state["history"].append({"role": "assistant", "content": "✍️ **Writing sub-report…**", "role_id": "principal_investigator"})
                yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False)
                try:
                    sub_report = await asyncio.to_thread(
                        session_state["pi_sub_report"].invoke,
                        {"section_name": section_title, "focus": section["focus"], "search_results": search_results_str},
                    )
                except Exception as e2:
                    sub_report = f"(Sub-report failed: {e2})"
                if session_state["history"] and session_state["history"][-1].get("content") == "✍️ **Writing sub-report…**":
                    session_state["history"].pop()
                # Include references (with [1], [2], titles, links) so final report PI can see and cite them
                ref_block = f"**References:**\n{search_results_str}" if search_results_str and search_results_str.strip() else ""
                sub_block = f"**Sub-report:**\n{(sub_report or '').strip()}"
                part = f"### {section_title}\n\n{ref_block}\n\n{sub_block}" if ref_block else f"### {section_title}\n\n{sub_block}"
                sub_reports_parts.append(part)
                # Show full sub-report in chat (no truncation); content already has ## Sub-report from prompt
                full_sub = (sub_report or "").strip()
                session_state["history"].append({"role": "assistant", "content": full_sub, "role_id": "principal_investigator"})
                yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False)
            sub_reports_text = "\n\n".join(sub_reports_parts)
            session_state["history"].append({"role": "assistant", "content": "📝 **Principal Investigator** is writing the research draft…", "role_id": "principal_investigator"})
            yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False)
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
            session_state["history"][-1] = {"role": "assistant", "content": f"# {draft_title}\n\n{pi_report}", "role_id": "principal_investigator"}
            yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False)
            # Generate Suggest steps (new dialog) for CB/MLS — show bubble then replace with content
            pi_suggest_steps = ""
            if pi_report and "Report generation failed" not in pi_report:
                session_state["history"].append({"role": "assistant", "content": "📋 **Principal Investigator** is generating Suggest steps…", "role_id": "principal_investigator"})
                yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False)
                try:
                    pi_suggest_steps = await asyncio.to_thread(
                        session_state["pi_suggest_steps"].invoke,
                        {"draft_report": pi_report, "input": text},
                    )
                    pi_suggest_steps = (pi_suggest_steps or "").strip()
                except Exception as e3:
                    print(f"[PI suggest steps] invoke failed: {e3}")
                if pi_suggest_steps:
                    session_state["history"][-1] = {"role": "assistant", "content": f"**Suggest steps**\n\n{pi_suggest_steps}", "role_id": "principal_investigator"}
                else:
                    session_state["history"].pop()
                yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False)
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
                yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False)
            if not intermediate_steps:
                session_state["history"][-1] = {"role": "assistant", "content": "**Principal Investigator** is thinking (fallback search).", "role_id": "principal_investigator"}
                yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False)
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
                    yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False)
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
            yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False)

    session_state["memory"].save_context({"input": display_text}, {"output": pi_report})
    if current_tool_count >= AGENT_CHAT_MAX_TOOL_CALLS:
        yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False)
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
    yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False)
    # --- Phase 2: CB planning — from PI draft + Suggest steps to concrete pipeline (JSON steps) ---
    session_state["history"].append({"role": "assistant", "content": "📋 **Computational Biologist** is planning the pipeline…", "role_id": "computational_biologist"})
    yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False)
    cb_planner_inputs = {
        "pi_report": pi_report,
        "pi_suggest_steps": pi_suggest_steps or "",
        "protein_context_summary": protein_context_summary,
        "tool_outputs": json.dumps(tool_outputs_summary, ensure_ascii=False),
    }
    try:
        plan = await asyncio.to_thread(session_state["cb_planner"].invoke, cb_planner_inputs)
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
        task_desc = p.get("task_description") or p.get("Task_description") or ""
        tool_name = p.get("tool_name") or p.get("Tool_name") or ""
        tool_input = p.get("tool_input") or p.get("Tool_input") or {}
        if not tool_name:
            return None
        if not isinstance(tool_input, dict):
            tool_input = {}
        return {"step": step_num, "task_description": task_desc, "tool_name": tool_name, "tool_input": tool_input}

    plan = [_normalize_step(i, p) for i, p in enumerate(plan)]
    plan = [p for p in plan if p is not None]
    # Search steps are already done by PI in the research phase; do not run them again
    plan = [p for p in plan if (p.get("tool_name") or "") not in PI_SEARCH_TOOL_NAMES]

    if not plan:
        session_state["history"][-1] = {"role": "assistant", "content": "No pipeline steps to run; the research draft above is the answer.", "role_id": "computational_biologist"}
        yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=True, file_count="multiple")
        return

    # --- CB: replace planning bubble with pipeline, then MLS executes each step ---
    step_lines = [f"**Step {p['step']}.** {p['task_description']}" for p in plan]
    plan_text = "📋 **Pipeline (CB planning)**\n\nHere's what we'll do:\n\n" + "\n\n".join(step_lines)
    session_state["history"][-1] = {"role": "assistant", "content": plan_text, "role_id": "computational_biologist"}
    session_state.setdefault("conversation_log", []).append({
        "role": "assistant", "content": plan_text, "role_id": "computational_biologist",
        "timestamp": datetime.now().isoformat(),
    })
    yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False)
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
            yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False)
            break

        step_num = step["step"]
        task_desc = step["task_description"]
        tool_name = step["tool_name"]
        tool_input = step["tool_input"]

        # MLS executes pipeline steps (CB only plans; MLS runs tools)
        session_state['history'].append({"role": "assistant", "content": f"⏳ **Executing Step {step_num}:** {task_desc}", "role_id": "machine_learning_specialist"})
        yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False)
        try:
            merged_tool_input = _merge_tool_parameters_with_context(protein_ctx, tool_input)
            # CRITICAL: Resolve dependencies BEFORE cache check and tool execution
            for key, value in merged_tool_input.items():
                if isinstance(value, str) and value.startswith("dependency:"):
                    parts = value.split(':')
                    dep_step = int(parts[1].replace('step_', '').replace('step', ''))
                    dep_raw_output = step_results[dep_step]['raw_output']
                    if len(parts) > 2:
                        field_name = parts[2]
                        try:
                            parsed = json.loads(dep_raw_output) if isinstance(dep_raw_output, str) else dep_raw_output
                            merged_tool_input[key] = parsed.get(field_name, dep_raw_output)
                        except:
                            merged_tool_input[key] = dep_raw_output
                    else:
                        merged_tool_input[key] = dep_raw_output

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

                step_detail = f"**Step {step_num}:** {task_desc}\n\n"
                step_detail += f"**Tool:** {tool_name} ⚡ (cached result)\n"
                step_detail += f"**Input:** {json.dumps(merged_tool_input, indent=2)}\n\n"
                step_detail += f"**Cache Key:** {cached_entry.get('cache_key', 'N/A')}\n\n"
                step_detail += f"**Output:**\n```\n{str(raw_output)[:500]}{'...' if len(str(raw_output)) > 500 else ''}\n```"

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

                session_state['history'].append({"role": "assistant", "content": f"✅ **Step {step_num} Complete (cached):** {task_desc}\n\n{step_detail}", "role_id": "machine_learning_specialist"})
                session_state['history'].append({"role": "assistant", "content": f"✓ **CB verified:** Step {step_num} complete (cached). Proceeding to next.", "role_id": "computational_biologist"})
                yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False)
                continue

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
                        session_state["history"].append({"role": "assistant", "content": "🔧 **MLS** is analyzing the error…", "role_id": "machine_learning_specialist"})
                        yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False)
                        retry_input, mls_report_for_cb = await asyncio.to_thread(
                            _mls_debug_step,
                            session_state["llm"],
                            step_num,
                            task_desc,
                            tool_name,
                            merged_tool_input,
                            str(e),
                        )
                        if session_state["history"] and session_state["history"][-1].get("content") == "🔧 **MLS** is analyzing the error…":
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
                            yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False)
                            continue
                    break
            if executor_result is None and last_step_error is not None:
                error_message = f"❌ **Error in Step {step_num}:** {task_desc}\n`{str(last_step_error)}`"
                session_state['history'].append({"role": "assistant", "content": error_message, "role_id": "machine_learning_specialist"})
                if mls_report_for_cb:
                    session_state['history'].append({"role": "assistant", "content": f"**MLS report for CB:** {mls_report_for_cb}", "role_id": "machine_learning_specialist"})
                    session_state['history'].append({"role": "assistant", "content": f"**CB:** Step {step_num} failed. {mls_report_for_cb} Please re-plan the pipeline or ask the user to adjust.", "role_id": "computational_biologist"})
                else:
                    session_state['history'].append({"role": "assistant", "content": f"**CB suggestion:** Step {step_num} failed. Check parameters or retry; ask again to adjust the pipeline if needed.", "role_id": "computational_biologist"})
                yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=True)
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

            # Display training progress if this is a training tool
            if tool_name == 'train_protein_model_tool':
                try:
                    parsed_output = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
                    if isinstance(parsed_output, dict) and 'training_progress' in parsed_output:
                        training_progress = parsed_output['training_progress']
                        if training_progress:
                            progress_display = f"**Training Progress:**\n```\n{training_progress}\n```"
                            session_state['history'].append({"role": "assistant", "content": f"⏳ **Step {step_num}:** {task_desc}\n\n{progress_display}", "role_id": "machine_learning_specialist"})
                            yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False)

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
                if tool_name in ['ncbi_sequence_download', 'alphafold_structure_download', 'uniprot_sequence_download', 'interpro_lookup',
                                 'protein_function_prediction', 'functional_residue_prediction',
                                 'protein_property_prediction', 'zero_shot_mutation_sequence_prediction', 'zero_shot_mutation_structure_prediction',
                                 'pdb_sequence_extraction', 'pdb_structure_download', 'literature_search', 'dataset_search', 'web_search',
                                 'esmfold_structure_prediction', 'foldseek_search']:

                    output_data = json.loads(raw_output)
                    if output_data.get('success') and 'file_path' in output_data:
                        file_path = output_data['file_path']
                        if tool_name == 'alphafold_structure_download':
                            uniprot_id = merged_tool_input.get('uniprot_id', 'unknown')
                            protein_ctx.add_structure_file(file_path, 'alphafold', uniprot_id)
                        elif tool_name == 'ncbi_sequence_download':
                            protein_ctx.add_file(file_path)
                    dl_path = _extract_download_file_from_output(tool_name, output_data)
                    if dl_path:
                        session_state['latest_tool_output_file'] = dl_path
                    if tool_name == 'literature_search' and isinstance(output_data, dict) and output_data.get('references'):
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
                step_detail += f"**Output:**\n```\n{output_str[:500]}{'...' if len(output_str) > 500 else ''}\n```"

            analysis_log += f"--- Analysis for Step {step_num}: {task_desc} ---\n\n"
            analysis_log += f"Tool: {tool_name}\n"
            analysis_log += f"Input: {json.dumps(merged_tool_input, indent=2)}\n"
            analysis_log += f"Output: {raw_output}\n\n"

            session_state['history'].append({"role": "assistant", "content": f"✅ **Step {step_num} Complete:** {task_desc}\n\n{step_detail}", "role_id": "machine_learning_specialist"})
            session_state['history'].append({"role": "assistant", "content": f"✓ **CB verified:** Step {step_num} complete. Proceeding to next.", "role_id": "computational_biologist"})
            yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False)

        except Exception as e:
            error_message = f"❌ **Error in Step {step_num}:** {task_desc}\n`{str(e)}`"
            session_state['history'].append({"role": "assistant", "content": error_message, "role_id": "machine_learning_specialist"})
            session_state["history"].append({"role": "assistant", "content": "🔧 **MLS** is analyzing the error…", "role_id": "machine_learning_specialist"})
            yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False)
            try:
                merged_for_debug = _merge_tool_parameters_with_context(protein_ctx, step.get("tool_input") or {})
            except Exception:
                merged_for_debug = step.get("tool_input") or {}
            _, mls_report_for_cb = await asyncio.to_thread(
                _mls_debug_step,
                session_state["llm"],
                step_num,
                task_desc,
                tool_name,
                merged_for_debug,
                str(e),
            )
            if session_state["history"] and session_state["history"][-1].get("content") == "🔧 **MLS** is analyzing the error…":
                session_state["history"].pop()
            if mls_report_for_cb:
                session_state['history'].append({"role": "assistant", "content": f"**MLS report for CB:** {mls_report_for_cb}", "role_id": "machine_learning_specialist"})
                session_state['history'].append({"role": "assistant", "content": f"**CB:** Step {step_num} failed. {mls_report_for_cb} Please re-plan the pipeline or ask the user to adjust.", "role_id": "computational_biologist"})
            else:
                session_state['history'].append({"role": "assistant", "content": f"**CB suggestion:** Step {step_num} failed. Check parameters or retry; ask again to adjust the pipeline if needed.", "role_id": "computational_biologist"})
            yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=True)
            return

    # SC summarizes only after all steps (separate dialog + avatar)
    session_state['history'].append({"role": "assistant", "content": "📄 **All steps complete. Generating final report...**", "role_id": "scientific_critic"})
    yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False)

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
    yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=True, file_count="multiple")



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
        inp = json.dumps(t.get("inputs", {}), ensure_ascii=False)
        if len(inp) > 180:
            inp = inp[:180] + "..."
        out = str(t.get("outputs", ""))[:180] + ("..." if len(str(t.get("outputs", ""))) > 180 else "")
        lines.append(f"**{j}.** `{name}`{cached} — {ts}")
        lines.append(f"In: `{inp}`")
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
    """Create the chat tab: left = main area (conversation + input at bottom); right = sidebar (examples, tips, feedback, log)."""
    session_state = gr.State(value=initialize_session_state())
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
        # Left: examples, tips, feedback, log (references chat_input)
        with gr.Column(scale=2, min_width=200, elem_id="vf-agent-sidebar-column"):
            with gr.Group():
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
                with gr.Accordion("📝 Provide Feedback", open=False):
                    feedback_input = gr.Textbox(
                        placeholder="Enter your feedback here...",
                        lines=1,
                        show_label=False
                    )
                    feedback_submit = gr.Button("Submit", variant="primary", size="sm")
                    feedback_status = gr.Markdown(visible=False)
                with gr.Accordion("📋 Backend Log", open=True, elem_id="vf-agent-backend-log"):
                    log_display = gr.HTML(value=_build_log_html({}), elem_id="vf-log-display")
    # Feedback submission handler
    feedback_submit.click(
        fn=handle_feedback_submit,
        inputs=[feedback_input],
        outputs=[feedback_input, feedback_status]
    )

    model_selector.change(
        fn=update_llm_model,
        inputs=[model_selector, session_state],
        outputs=[session_state]
    )

    chat_input.submit(
        fn=send_message,
        inputs=[chat_input, session_state],
        outputs=[right_panel_display, log_display, chat_input],
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