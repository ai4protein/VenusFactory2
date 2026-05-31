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
from web_v2.chat_api._shared import _is_zh_text, _snapshot, _to_json
from web_v2.chat_api._stream import (
    _STREAM_STATE_KEYS,
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
        "research_idx": 0,
        "search_idx": 0,
        "current_search_results": [],
        "research_sub_reports": list(state.get("research_sub_reports", [])),
        "sub_report_rewrite_comment": state.get("sub_report_rewrite_comment", ""),
        "auto_execute": state.get("auto_execute", False),
        "execution_failed": False,
        "failed_step": None,
        "failed_reason": None,
        "clarification_questions": list(state.get("clarification_questions", [])),
        "clarification_answers": list(state.get("clarification_answers", [])),
        "waiting_for": waiting_for,
    }
    if extra_state:
        initial_state.update(extra_state)

    graph = _get_compiled_graph()
    config = {
        "configurable": {"chains": state, "session_id": state["session_id"]},
        "recursion_limit": 100,
    }

    state["status"] = "started"
    yield f"event: state\ndata: {_to_json(_snapshot(state))}\n\n"

    async for stream_mode, data in graph.astream(
        initial_state, config=config, stream_mode=["updates", "custom"]
    ):
        if await _is_cancelled(state["session_id"]):
            state["status"] = "stopped"
            state.setdefault("history", []).append({
                "role": "assistant",
                "content": "用户已停止本次运行。" if is_zh else "Run stopped by user.",
                "role_id": "principal_investigator",
            })
            await _session_store.save(state["session_id"])
            yield f"event: state\ndata: {_to_json(_snapshot(state))}\n\n"
            yield "event: done\ndata: {}\n\n"
            return

        if stream_mode == "custom":
            event_type = data.get("type", "token") if isinstance(data, dict) else "token"
            yield f"event: {event_type}\ndata: {_to_json(data)}\n\n"
        elif stream_mode == "updates":
            # If user already requested cancel, preserve our stopping/stopped
            # status so the graph's in-flight node updates don't clobber it.
            cancelled = await _is_cancelled(state["session_id"])
            for _, updates in data.items():
                if updates:
                    for key, val in updates.items():
                        if key in _STREAM_STATE_KEYS:
                            if cancelled and key == "status":
                                continue  # don't overwrite stopping/stopped
                            state[key] = val
            yield f"event: state\ndata: {_to_json(_snapshot(state))}\n\n"

    await _finalize_after_stream(state, original_text)
    await _session_store.save(state["session_id"])
    yield f"event: state\ndata: {_to_json(_snapshot(state))}\n\n"
    yield "event: done\ndata: {}\n\n"
