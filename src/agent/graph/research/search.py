"""PI per-section search nodes (one query per invocation)."""
from __future__ import annotations

import asyncio
from datetime import datetime

from langchain_core.runnables import RunnableConfig

from agent.chat_agent_utils import AGENT_CHAT_MAX_TOOL_CALLS, _run_section_search
from agent.graph.common.lang import _detect_ui_lang
from agent.graph.state import AgentState
from web.utils.chat_format_utils import _format_search_preview, _format_search_summary


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
        section_title = (
            f"**第 {research_idx + 1} 节：** {section['section_name']}"
            if ui_lang == "zh"
            else f"**Section {research_idx + 1}:** {section['section_name']}"
        )
        history.append({"role": "assistant", "content": section_title, "role_id": "principal_investigator"})

    if search_idx < len(queries):
        sq = queries[search_idx]
        search_results_list, search_logged = await asyncio.to_thread(_run_section_search, sq)
        current_search_results.extend(search_results_list)

        for tname, tinputs, toutputs in search_logged:
            step_off = len(protein_ctx.tool_history) + 1
            protein_ctx.add_tool_call(step_off, tname, tinputs, toutputs, cached=False)
            executions.append({
                "step": step_off,
                "tool_name": tname,
                "inputs": tinputs,
                "outputs": (str(toutputs)[:1000] + "..." if len(str(toutputs)) > 1000 else str(toutputs)),
                "timestamp": datetime.now().isoformat(),
            })
            summary_msg = _format_search_summary(tname, tinputs, str(toutputs))
            preview = _format_search_preview(tname, str(toutputs))
            history.append({
                "role": "assistant",
                "content": summary_msg + ("\n\n" + preview if preview else ""),
                "role_id": "principal_investigator",
            })

    return {
        "history": history,
        "tool_executions": executions,
        "current_search_results": current_search_results,
        "search_idx": search_idx + 1,
        "status": "research_search_done",
    }
