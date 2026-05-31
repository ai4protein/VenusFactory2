"""PI research planning nodes."""
from __future__ import annotations

import asyncio

from langchain_core.runnables import RunnableConfig

from agent.graph.common.lang import _detect_ui_lang
from agent.graph.helpers.chat_history import _format_conversation_history
from agent.graph.helpers.plan_helpers import (
    _looks_like_execution_request,
    _parse_sections,
)
from agent.graph.state import AgentState
from logger import get_logger

_logger = get_logger("agent.graph")


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
        _logger.warning("PI sections failed: %s", e)
        sections_list = []

    if not sections_list and _looks_like_execution_request(text):
        # Planner parse can fail for some base models; for execution-style requests
        # fall through to internal plan node instead of PI chat free-form reply.
        return {"status": "resume_plan", "pi_report": "", "pi_suggest_steps": "", "ui_lang": ui_lang}

    if not sections_list:
        return {"status": "chat_mode", "pi_report": "", "pi_suggest_steps": "", "ui_lang": ui_lang}

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
        "status": "research_planning_done",
    }
