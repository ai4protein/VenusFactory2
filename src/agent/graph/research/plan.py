"""PI research planning nodes."""
from __future__ import annotations

import asyncio

from langchain_core.runnables import RunnableConfig

from agent.graph.common.lang import _resolve_ui_lang
from agent.graph.helpers.chat_history import _format_conversation_history
from agent.graph.helpers.plan_helpers import (
    _looks_like_execution_request,
    _looks_like_research_request,
    _parse_sections,
)
from agent.graph.state import AgentState
from logger import get_logger

_logger = get_logger("agent.graph")


def _pop_plan_thinking(history: list) -> list:
    """Remove the in-flight PI thinking bubble from research_plan_start_node."""
    if history and (
        history[-1].get("kind") == "thinking"
        or history[-1].get("phase") == "thinking"
        or "分析你的请求" in history[-1].get("content", "")
        or "analyzing your request" in history[-1].get("content", "").lower()
    ):
        history.pop()
    return history


async def research_plan_start_node(state: AgentState, config: RunnableConfig):
    """Show PI Thinking immediately — before the sections LLM runs."""
    history = list(state.get("history", []))
    ui_lang = _resolve_ui_lang(state)
    history.append({
        "role": "assistant",
        "content": (
            "🤔 **Principal Investigator** 正在分析你的请求并制定研究计划…"
            if ui_lang == "zh"
            else "🤔 **Principal Investigator** is analyzing your request and creating a research plan…"
        ),
        "role_id": "principal_investigator",
        "kind": "thinking",
        "phase": "thinking",
    })
    return {"history": history, "status": "analyzing", "ui_lang": ui_lang}


async def research_plan_node(state: AgentState, config: RunnableConfig):
    """PI phase 1: Create a research plan with sections.

    Always run PI analysis first. Explicit Skip Research is handled via
    clarification answers / ``waiting_for=skip_to_plan`` — do not silently
    jump to CB based on keyword heuristics.

    Post-clarification path (``waiting_for == clarification_answered``): regenerate
    sections/queries using the user's answers (already folded into the message /
    history), then resume search — do not loop back into clarification.
    """
    chains = config.get("configurable", {}).get("chains", {})
    protein_ctx = state["protein_context"]
    text = state["messages"][-1].content
    ui_lang = _resolve_ui_lang(state, text)
    history = list(state.get("history", []))
    post_clarification = (
        state.get("waiting_for") == "clarification_answered"
        or state.get("status") == "replan_research"
    )

    protein_context_summary = protein_ctx.get_context_summary()
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

    if (
        not sections_list
        and _looks_like_execution_request(text)
        and not _looks_like_research_request(text)
    ):
        # PI tried but produced no sections; still speak as PI before CB.
        _logger.info(
            "research_plan: empty sections but execution intent → resume_plan"
        )
        history = _pop_plan_thinking(history)
        history.append({
            "role": "assistant",
            "content": (
                "🧭 **Principal Investigator**：这是偏工具执行的请求，"
                "我将跳过文献调研，直接交给 Computational Biologist 设计流程。"
                if ui_lang == "zh" else
                "🧭 **Principal Investigator**: This looks like a tool-execution request, "
                "so I'll skip literature research and hand off to the Computational Biologist."
            ),
            "role_id": "principal_investigator",
            "phase": "research_plan",
        })
        return {
            "status": "resume_plan",
            "pi_report": "",
            "pi_suggest_steps": "",
            "research_sections": [],
            "waiting_for": None,
            "history": history,
            "ui_lang": ui_lang,
        }

    if not sections_list:
        # Greeting / chitchat, or literature ask that failed to parse sections.
        history = _pop_plan_thinking(history)
        return {
            "status": "chat_mode",
            "pi_report": "",
            "pi_suggest_steps": "",
            "waiting_for": None,
            "history": history,
            "ui_lang": ui_lang,
        }

    # Keep the start-node Thinking bubble visible until clarification_start
    # (or search) replaces it — do not append a second "analyzing" line here.
    result = {
        "research_sections": sections_list,
        "research_idx": 0,
        "search_idx": 0,
        "current_search_results": [],
        "research_sub_reports": [],
        "history": history,
        "ui_lang": ui_lang,
    }

    if post_clarification:
        _logger.info(
            "research_plan: post-clarification replan → resume_research (%d sections)",
            len(sections_list),
        )
        result["waiting_for"] = None
        result["status"] = "resume_research"
        return result

    result["status"] = "research_planning_done"
    return result
