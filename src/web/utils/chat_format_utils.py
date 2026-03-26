import json
import os
import base64
from typing import Dict, Any, List
import gradio as gr
from agent.prompts import get_role_avatar_path, ROLE_DISPLAY_NAMES
from web.utils.css_loader import load_css_file
from pathlib import Path

# Load chat tab CSS from assets (cached at module level)
_CHAT_DIALOG_CSS = load_css_file("chat_tab_dialog.css")
_CHAT_PANEL_CSS = load_css_file("chat_tab_panel.css")
_CHAT_LOG_CSS = load_css_file("chat_tab_log.css")

try:
    import markdown
except ImportError:
    markdown = None

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

def _format_literature_citations(refs: list, max_n: int = 5) -> list:
    """Format literature references as [n] [title](url) for PI citations (Markdown link format).
    Only shows available fields, no NA or empty values.
    """
    out = []
    for i, r in enumerate((refs or [])[:max_n], 1):
        if not isinstance(r, dict):
            continue
        title = (r.get("title") or r.get("citation") or "No title").strip()

        # Build metadata parts, only include non-empty values
        authors = r.get("authors") or r.get("author") or ""
        if isinstance(authors, list):
            authors = ", ".join(str(a) for a in authors[:5] if a)
        year = r.get("year") or r.get("published") or ""

        # Build metadata string with only available fields
        meta_parts = []
        if authors:
            meta_parts.append(authors)
        if year:
            meta_parts.append(str(year))
        meta_str = f" — {', '.join(meta_parts)}" if meta_parts else ""

        url = r.get("url") or r.get("link") or ""
        if url:
            out.append(f"[{i}] [{title}]({url}){meta_str}")
        else:
            out.append(f"[{i}] {title}{meta_str}")
    return out

def _format_literature_for_reading(refs: list, max_n: int = 5, abstract_max: int = 400) -> list:
    """Format literature with abstract so PI can read and cite. Each item: [title](url) — authors, year.
    Only shows available fields, no NA or empty values.
    """
    out = []
    for i, r in enumerate((refs or [])[:max_n], 1):
        if not isinstance(r, dict):
            continue
        title = (r.get("title") or r.get("citation") or "No title").strip()

        # Build metadata parts, only include non-empty values
        authors = r.get("authors") or r.get("author") or ""
        if isinstance(authors, list):
            authors = ", ".join(str(a) for a in authors[:5] if a)
        year = r.get("year") or r.get("published") or ""

        meta_parts = []
        if authors:
            meta_parts.append(authors)
        if year:
            meta_parts.append(str(year))
        meta_str = f" — {', '.join(meta_parts)}" if meta_parts else ""

        url = r.get("url") or r.get("link") or ""
        if url:
            line = f"[{title}]({url}){meta_str}"
        else:
            line = f"{title}{meta_str}"

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
                line = f"[{title or 'Link'}]({url})"
            else:
                line = f"{title or str(r)}"
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
    """Format PI agent intermediate_steps (list of (action, observation)) into a single string for pi_report_chain input.
    References are numbered and each on a separate line.
    """
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
                    # Each reference on its own line with proper line breaks
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
    from agent.chat_agent_utils import _extract_deepsearch_data
    query = (tool_inputs.get("query") or "") if isinstance(tool_inputs, dict) else ""
    source = (tool_inputs.get("source") or "") if isinstance(tool_inputs, dict) else ""
    q = (query[:80] + "…") if len(query) > 80 else query
    label = tool_name.replace("_", " ").title()
    if source and source.lower() != "all":
        label = f"{label} ({source})"
    intro = f"**{label}** — searched for \"{q}\"." if q else f"**{label}**."
    try:
        data = json.loads(observation_str) if isinstance(observation_str, str) and observation_str.strip().startswith("{") else {}
        data = _extract_deepsearch_data(data)
    except Exception:
        return intro + " Error."
    if data.get("success") is False:
        return intro + " Error."
    if tool_name in ("query_pubmed", "query_semantic_scholar", "query_arxiv", "query_literature_by_keywords"):
        refs = data.get("references") or data.get("results") or data.get("papers") or data or []
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
    if tool_name in ("query_tavily", "query_duckduckgo", "query_web_by_keywords"):
        res = data.get("results") or data or []
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
    if tool_name in ("query_github", "query_hugging_face", "query_dataset_by_keywords"):
        ds = data.get("datasets") or data.get("results") or data or []
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
        success_icon = "✅" if t.get("success", True) else "❌"
        out_path = t.get("output_file_path") or ""
        inp = t.get("inputs", {})
        inp_str = json.dumps(inp, ensure_ascii=False)
        if len(inp_str) > 180:
            inp_str = inp_str[:180] + "..."
        out = str(t.get("outputs", ""))[:180] + ("..." if len(str(t.get("outputs", ""))) > 180 else "")
        if name == "read_skill" and isinstance(inp, dict) and inp.get("skill_id"):
            lines.append(f"**{j}.** {success_icon} 📖 **Loaded skill:** `{inp.get('skill_id')}`{cached} — {ts}")
        else:
            lines.append(f"**{j}.** {success_icon} `{name}`{cached} — {ts}")
        if out_path:
            lines.append(f"File: `{out_path}`")
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
    # Auto-scroll script: scroll to bottom anchor on each update
    scroll_script = '''
    <div id="vf-chat-bottom-anchor"></div>
    <script>
    (function() {
        setTimeout(function() {
            var anchor = document.getElementById('vf-chat-bottom-anchor');
            if (anchor) {
                anchor.scrollIntoView({ behavior: 'smooth', block: 'end' });
            }
        }, 50);
    })();
    </script>
    '''
    return (
        f"{panel_style_tag}"
        '<div class="vf-right-panel" style="height:100%;min-height:77vh;">'
        f'<div class="vf-conversation-panel">{chat_html}{scroll_script}</div>'
        "</div>"
    )

def _build_log_html(session_state: Dict[str, Any]) -> str:
    """Log panel HTML for sidebar. Styled with colors so it reads clearly as a log."""
    log_style_tag = f"<style>{_CHAT_LOG_CSS}</style>" if _CHAT_LOG_CSS else ""
    log_md = _format_backend_log(session_state or {})
    log_html = _markdown_to_html(log_md)
    return f"{log_style_tag}<div class=\"vf-log-panel\">{log_html}</div>"
