"""PI per-section search nodes (one query per invocation)."""
from __future__ import annotations

import asyncio

from langchain_core.runnables import RunnableConfig

from agent.chat_agent_utils import AGENT_CHAT_MAX_TOOL_CALLS, _run_section_search
from agent.graph.common.lang import _resolve_ui_lang
from agent.graph.state import AgentState


def _summarize_query_results(
    query: str,
    search_results_list: list,
    search_logged: list,
    ui_lang: str,
) -> str:
    """One compact history line for a multi-source query (no per-tool spam)."""
    hits = len(search_results_list or [])
    tools_ok = []
    for tname, _tinputs, toutputs in search_logged or []:
        raw = str(toutputs or "")
        if '"success": false' in raw.lower() or "no results" in raw.lower():
            continue
        # Cheap signal: non-empty structured payload with references/results.
        if any(k in raw for k in ("references", "results", "papers", "title")):
            short = (tname or "").replace("query_", "")
            if short and short not in tools_ok:
                tools_ok.append(short)
    sources = " · ".join(tools_ok) if tools_ok else ("none" if ui_lang != "zh" else "无")
    q = (query or "").strip()
    if len(q) > 72:
        q = q[:72] + "…"
    if ui_lang == "zh":
        return (
            f"🔎 检索完成：**{q or '…'}** — 汇总 {hits} 条（来源：{sources}）"
        )
    return (
        f"🔎 Search done: **{q or '…'}** — {hits} merged hit(s) (sources: {sources})"
    )


async def research_search_start_node(state: AgentState, config: RunnableConfig):
    """Announce the upcoming parallel deep-research call to the chat UI."""
    research_idx = state.get("research_idx", 0)
    search_idx = state.get("search_idx", 0)
    sections = state.get("research_sections", [])
    history = list(state.get("history", []))
    ui_lang = _resolve_ui_lang(state)
    if research_idx >= len(sections):
        return {"status": "research_steps_done"}
    section = sections[research_idx]
    queries = section.get("search_queries", [])
    if search_idx >= len(queries):
        return {}
    sq = (queries[search_idx] or "").strip()[:80]
    if sq and len(sq) == 80:
        sq = sq + "…"

    # Keep the announcement short — detailed per-tool chatter was drowning
    # the eventual sub-report in the timeline.
    if ui_lang == "zh":
        content = f"🔍 **Principal Investigator** 正在检索：**{sq or '…'}**"
    else:
        content = f"🔍 **Principal Investigator** is searching: **{sq or '…'}**"
    history.append({
        "role": "assistant",
        "content": content,
        "role_id": "principal_investigator",
        "phase": "research_search",
        "kind": "status",
    })
    return {"history": history}


async def research_search_node(state: AgentState, config: RunnableConfig):
    """PI phase 2a: Process ONE search query from the current section."""
    protein_ctx = state["protein_context"]
    research_idx = state.get("research_idx", 0)
    search_idx = state.get("search_idx", 0)
    sections = state.get("research_sections", [])
    history = list(state.get("history", []))
    ui_lang = _resolve_ui_lang(state)
    executions = list(state.get("tool_executions", []))
    current_search_results = list(state.get("current_search_results", []))

    if research_idx >= len(sections) or len(protein_ctx.tool_history) >= AGENT_CHAT_MAX_TOOL_CALLS:
        return {"status": "research_steps_done"}

    section = sections[research_idx]
    queries = section.get("search_queries", [])

    if search_idx == 0:
        section_title = (
            f"**第 {research_idx + 1}/{len(sections)} 节：** {section['section_name']}"
            if ui_lang == "zh"
            else f"**Section {research_idx + 1}/{len(sections)}:** {section['section_name']}"
        )
        history.append({
            "role": "assistant",
            "content": section_title,
            "role_id": "principal_investigator",
            "phase": "research_section",
            "kind": "status",
        })

    # Drop the "is searching…" status bubble once results are ready.
    if history and history[-1].get("phase") == "research_search":
        history.pop()

    if search_idx < len(queries):
        sq = queries[search_idx]
        # Lighter fan-out for Expert UX: fewer sources, less timeline noise.
        search_results_list, search_logged = await asyncio.to_thread(
            _run_section_search, sq, None, "lite"
        )
        current_search_results.extend(search_results_list)

        # Record in protein_ctx only — do NOT push into chat tool_executions
        # (those cards drown sub-reports and inflate PipelineProgress counts).
        step_off = len(protein_ctx.tool_history) + 1
        summary_inputs = {"query": sq, "sources": [t[0] for t in (search_logged or [])]}
        summary_outputs = {
            "success": True,
            "merged_hits": len(search_results_list or []),
            "tools": [
                {"tool": t[0], "ok": '"success": false' not in str(t[2] or "").lower()}
                for t in (search_logged or [])
            ],
        }
        protein_ctx.add_tool_call(
            step_off, "research_search", summary_inputs, summary_outputs, cached=False
        )
        # Keep raw calls in protein_ctx for debugging / report citations, but
        # do NOT push each query_arxiv/… into chat history or tool_executions.
        for tname, tinputs, toutputs in search_logged or []:
            protein_ctx.add_tool_call(
                len(protein_ctx.tool_history) + 1,
                tname,
                tinputs,
                toutputs,
                cached=False,
            )

        history.append({
            "role": "assistant",
            "content": _summarize_query_results(
                sq, search_results_list, search_logged, ui_lang
            ),
            "role_id": "principal_investigator",
            "phase": "research_search_done",
            "kind": "status",
        })

    return {
        "history": history,
        "tool_executions": executions,
        "current_search_results": current_search_results,
        "search_idx": search_idx + 1,
        "status": "research_search_done",
    }
