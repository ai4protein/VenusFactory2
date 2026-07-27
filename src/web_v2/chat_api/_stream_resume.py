"""Checkpoint-resume streamer used by the ``interactive`` router.

Split from ``_stream`` to keep both files under the size limit. Imports the
shared constants + finalize helper from ``_stream``.
"""
from typing import Any

from langchain_core.messages import HumanMessage

from web_v2.chat_api._hooks_runtime import (
    _get_compiled_graph,
    _is_cancelled,
    _session_store,
)
from web_v2.chat_api._shared import (
    _is_zh_text,
    _snapshot,
    _to_json,
    clear_agent_gates,
)
from web_v2.chat_api._stream import (
    _drain_graph_astream,
    _finalize_after_stream,
)


async def _stream_graph_resume(
    state: dict[str, Any],
    waiting_for: str,
    extra_state: dict[str, Any] | None = None,
):
    """Resume a graph run from a checkpoint (clarification answered or plan confirmed)."""
    if await _is_cancelled(state["session_id"]):
        state["status"] = "stopped"
        await _session_store.save(state["session_id"])
        yield f"event: state\ndata: {_to_json(_snapshot(state))}\n\n"
        yield "event: done\ndata: {}\n\n"
        return

    agent_session_dir = state.get("agent_session_dir")
    if not agent_session_dir:
        raise RuntimeError("Agent session directory is missing.")

    protein_ctx = state["protein_context"]
    is_zh = _is_zh_text(state.get("last_user_text", ""))
    original_text = state.get("last_user_text", "")

    initial_state = {
        "messages": [HumanMessage(content=original_text)],
        "protein_context": protein_ctx,
        "session_id": state["session_id"],
        "agent_session_dir": agent_session_dir,
        "history": list(state["history"]),
        "conversation_log": list(state.get("conversation_log", [])),
        "dialogue_memory": list(state.get("dialogue_memory", [])),
        "tool_executions": list(state.get("tool_executions", [])),
        "tool_cache": dict(state.get("tool_cache", {})),
        "status": "started",
        "pi_report": state.get("pi_report", ""),
        "pi_suggest_steps": state.get("pi_suggest_steps", ""),
        "plan": list(state.get("plan", [])),
        "current_step_index": state.get("current_step_index", 0),
        "step_results": dict(state.get("step_results", {})),
        "error": None,
        "research_sections": list(state.get("research_sections", [])),
        # Preserve research cursor — resetting to 0 made Continue/Rewrite
        # restart the first section instead of advancing.
        "research_idx": int(state.get("research_idx") or 0),
        "search_idx": int(state.get("search_idx") or 0),
        "current_search_results": list(state.get("current_search_results") or []),
        "research_sub_reports": list(state.get("research_sub_reports", [])),
        "sub_report_rewrite_comment": state.get("sub_report_rewrite_comment", ""),
        "auto_execute": state.get("auto_execute") is not False,
        "execution_failed": False,
        "failed_step": None,
        "failed_reason": None,
        "clarification_questions": list(state.get("clarification_questions", [])),
        "clarification_answers": list(state.get("clarification_answers", [])),
        "waiting_for": waiting_for,
        "review_sub_reports": state.get("review_sub_reports") is True,
        "full_manuscript": True if state.get("full_manuscript") is None else bool(state.get("full_manuscript")),
        "user_lang": state.get("user_lang") or "",
        "ui_lang": state.get("user_lang") or state.get("ui_lang") or "",
    }
    if extra_state:
        initial_state.update(extra_state)

    graph = _get_compiled_graph()
    config = {
        "configurable": {"chains": state, "session_id": state["session_id"]},
        "recursion_limit": 100,
    }

    state["status"] = "started"
    state["engine"] = "graph"
    state["chat_mode"] = "science_expert"
    clear_agent_gates(state)
    yield f"event: state\ndata: {_to_json(_snapshot(state))}\n\n"

    async for chunk in _drain_graph_astream(
        graph, initial_state, config, state, is_zh=is_zh
    ):
        yield chunk
        if chunk.startswith("event: done"):
            return

    await _finalize_after_stream(state, original_text)
    await _session_store.save(state["session_id"])
    yield f"event: state\ndata: {_to_json(_snapshot(state))}\n\n"
    yield "event: done\ndata: {}\n\n"
