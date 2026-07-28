"""PI clarification-question generation nodes."""
from __future__ import annotations

import asyncio
import json

from langchain_core.runnables import RunnableConfig

from agent.graph.common.lang import _resolve_ui_lang
from agent.graph.common.streaming import _stream_text
from agent.graph.common.ui_text import _ui_text
from agent.graph.helpers.chat_history import _format_conversation_history
from agent.graph.helpers.plan_helpers import (
    _parse_clarification_questions,
)
from agent.graph.state import AgentState
from logger import get_logger

_logger = get_logger("agent.graph")

# Cap content questions so users are not forced through 2–4 answers.
_MAX_CONTENT_QUESTIONS = 2


def _format_clarification_message(ui_lang: str, questions: list[dict]) -> str:
    """Build the PI prose that streams before the interactive form mounts.

    Question/option chips are revealed by the frontend form — keep this text
    as a short intro so options are not dumped twice.
    """
    title = _ui_text(ui_lang, "clarification_title")
    n = len(questions)
    if ui_lang == "zh":
        hint = f"\n\n请依次确认以下 {n} 个问题："
    else:
        hint = f"\n\nPlease answer the following {n} question(s):"
    return f"{title}{hint}\n"


async def clarification_start_node(state: AgentState, config: RunnableConfig):
    """Show PI Thinking while clarification options are generated."""
    history = list(state.get("history", []))
    ui_lang = _resolve_ui_lang(state)
    # Replace prior research-plan Thinking / analyzing placeholder.
    if history and (
        history[-1].get("kind") == "thinking"
        or history[-1].get("phase") == "thinking"
        or "分析你的请求" in history[-1].get("content", "")
        or "analyzing your request" in history[-1].get("content", "").lower()
        or "准备澄清" in history[-1].get("content", "")
        or "preparing clarification" in history[-1].get("content", "").lower()
    ):
        history.pop()
    history.append({
        "role": "assistant",
        "content": (
            "🤔 **Principal Investigator** 正在准备澄清问题…"
            if ui_lang == "zh"
            else "🤔 **Principal Investigator** is preparing clarification questions…"
        ),
        "role_id": "principal_investigator",
        # kind=thinking → frontend ThinkingBlock (live "Thinking…" pulse).
        "kind": "thinking",
        "phase": "thinking",
    })
    return {"history": history, "ui_lang": ui_lang}


async def clarification_node(state: AgentState, config: RunnableConfig):
    """PI generates clarification questions (capped) for the user to answer."""
    chains = config.get("configurable", {}).get("chains", {})
    text = state["messages"][-1].content
    ui_lang = _resolve_ui_lang(state, text)
    protein_ctx = state["protein_context"]
    history = list(state.get("history", []))
    sections = state.get("research_sections", [])

    # Always ask clarification (includes explicit Skip Research option).
    # Do not auto-jump to CB — that made Expert look like it started at CB.
    conversation_history = _format_conversation_history(chains, history, text)
    sections_str = json.dumps(sections, ensure_ascii=False)

    # Bound LLM wait so the "preparing clarification…" Thinking bubble does
    # not feel stuck for a full minute when the provider is slow.
    _CLARIFICATION_TIMEOUT_S = 25.0
    questions = []
    try:
        raw = await asyncio.wait_for(
            asyncio.to_thread(
                chains["pi_clarification"].invoke,
                {
                    "input": text,
                    "protein_context_summary": protein_ctx.get_context_summary(),
                    "conversation_history": conversation_history,
                    "research_sections": sections_str,
                },
            ),
            timeout=_CLARIFICATION_TIMEOUT_S,
        )
        questions = _parse_clarification_questions(raw)
    except asyncio.TimeoutError:
        _logger.warning(
            "PI clarification generation timed out after %.0fs; using skip-research only",
            _CLARIFICATION_TIMEOUT_S,
        )
    except Exception as e:
        _logger.warning("PI clarification generation failed: %s", e)

    if history and (
        history[-1].get("kind") == "thinking"
        or history[-1].get("phase") == "thinking"
        or "准备澄清" in history[-1].get("content", "")
        or "preparing clarification" in history[-1].get("content", "").lower()
    ):
        history.pop()

    # At most 2 content questions + an always-present research skip switch.
    # Even if the LLM timed out / failed, still gate on Skip Research so the
    # user is not silently auto-advanced past clarification.
    questions = list(questions or [])[:_MAX_CONTENT_QUESTIONS]

    # Put "Skip Research" first so it is the easiest default choice.
    if ui_lang == "zh":
        skip_q = {
            "question": "研究阶段是否继续？",
            "options": ["跳过 Research，直接进入工具执行", "继续 Research"],
            "allow_multiple": False,
        }
    else:
        skip_q = {
            "question": "How should we handle the research phase?",
            "options": [
                "Skip research and go straight to tool execution",
                "Continue research",
            ],
            "allow_multiple": False,
        }
    questions = [skip_q] + questions

    message = _format_clarification_message(ui_lang, questions)
    await _stream_text(message, role_id="principal_investigator")
    history.append({
        "role": "assistant",
        "content": message,
        "role_id": "principal_investigator",
    })

    return {
        "clarification_questions": questions,
        "waiting_for": "clarification",
        "history": history,
        "ui_lang": ui_lang,
        "status": "waiting_for_clarification",
    }
