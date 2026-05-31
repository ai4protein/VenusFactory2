"""PI clarification-question generation nodes."""
from __future__ import annotations

import asyncio
import json

from langchain_core.runnables import RunnableConfig

from agent.graph.common.lang import _detect_ui_lang
from agent.graph.common.ui_text import _ui_text
from agent.graph.helpers.chat_history import _format_conversation_history
from agent.graph.helpers.plan_helpers import _parse_clarification_questions
from agent.graph.state import AgentState
from logger import get_logger

_logger = get_logger("agent.graph")


async def clarification_start_node(state: AgentState, config: RunnableConfig):
    """Show 'PI is preparing clarification questions'."""
    history = list(state.get("history", []))
    ui_lang = state.get("ui_lang") or _detect_ui_lang(state["messages"][-1].content)
    if history and (
        "分析你的请求" in history[-1].get("content", "")
        or "analyzing your request" in history[-1].get("content", "").lower()
    ):
        history.pop()
    return {"history": history, "ui_lang": ui_lang}


async def clarification_node(state: AgentState, config: RunnableConfig):
    """PI generates 2-4 clarification questions for the user to answer."""
    chains = config.get("configurable", {}).get("chains", {})
    text = state["messages"][-1].content
    ui_lang = state.get("ui_lang") or _detect_ui_lang(text)
    protein_ctx = state["protein_context"]
    history = list(state.get("history", []))
    sections = state.get("research_sections", [])
    conversation_history = _format_conversation_history(chains, history, text)
    sections_str = json.dumps(sections, ensure_ascii=False)

    questions = []
    try:
        raw = await asyncio.to_thread(
            chains["pi_clarification"].invoke,
            {
                "input": text,
                "protein_context_summary": protein_ctx.get_context_summary(),
                "conversation_history": conversation_history,
                "research_sections": sections_str,
            },
        )
        questions = _parse_clarification_questions(raw)
    except Exception as e:
        _logger.warning("PI clarification generation failed: %s", e)

    if not questions:
        return {
            "clarification_questions": [],
            "waiting_for": None,
            "status": "resume_research",
            "ui_lang": ui_lang,
        }

    # Always add an explicit switch to let users skip literature research.
    if ui_lang == "zh":
        questions.append(
            {
                "question": "研究阶段是否继续？",
                "options": ["继续 Research（推荐）", "跳过 Research，直接进入工具执行"],
                "allow_multiple": False,
            }
        )
    else:
        questions.append(
            {
                "question": "How should we handle the research phase?",
                "options": ["Continue research (recommended)", "Skip research and go straight to tool execution"],
                "allow_multiple": False,
            }
        )

    title = _ui_text(ui_lang, "clarification_title")
    history.append({
        "role": "assistant",
        "content": title,
        "role_id": "principal_investigator",
    })

    return {
        "clarification_questions": questions,
        "waiting_for": "clarification",
        "history": history,
        "ui_lang": ui_lang,
        "status": "waiting_for_clarification",
    }
