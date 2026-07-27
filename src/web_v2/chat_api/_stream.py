"""LangGraph streaming for fresh agent runs (the ``messages`` router).

The checkpoint-resume streamer used by the ``interactive`` router lives in
``_stream_resume`` and shares the constants + ``_finalize_after_stream``
helper defined here.
"""
import asyncio
import os
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage

from agent.chat_agent_utils import (
    AGENT_CHAT_MAX_MESSAGES,
    extract_sequence_from_message,
    extract_uniprot_id_from_message,
)
from agent.graph.common.streaming import (
    attach_sse_to_session_state,
    bind_sse_queue,
    detach_sse_from_session_state,
    reset_sse_queue,
    sse_config_keys,
)

from web_v2.chat_api._hooks_runtime import (
    _dispatch_hook,
    _get_compiled_graph,
    _is_cancelled,
    _session_store,
)
from web_v2.chat_api._shared import (
    _append_dialogue_memory,
    _is_zh_text,
    _snapshot,
    _to_json,
    clear_agent_gates,
)
from web_v2.chat_api._uploads import _archive_conversation, _normalize_uploaded_file


# State keys to copy back from the LangGraph stream into the persistent
# session state on each "updates" event. Shared by both stream functions.
_STREAM_STATE_KEYS: frozenset[str] = frozenset({
    "history", "conversation_log", "tool_executions", "status",
    "pi_report", "pi_suggest_steps", "plan", "protein_context",
    "current_step_index", "step_results", "research_sections",
    "research_idx", "search_idx", "current_search_results",
    "research_sub_reports", "sub_report_rewrite_comment",
    "auto_execute", "tool_cache", "execution_failed",
    "failed_step", "failed_reason",
    "clarification_questions", "clarification_answers", "waiting_for",
    "review_sub_reports", "full_manuscript", "ui_lang", "user_lang", "error",
})

# Statuses that should NOT trigger end-of-run finalization (archive + memory).
_WAITING_STATUSES: tuple[str, ...] = (
    "waiting_for_clarification",
    "waiting_for_plan_confirmation",
    "waiting_for_iteration",
    "waiting_for_step_review",
    "waiting_for_sub_report_review",
)

# Terminal failure statuses that must NOT be rewritten to completed.
_FAILURE_STATUSES: frozenset[str] = frozenset({
    "planning_failed",
    "error",
    "execution_failed",
    "stopped",
})


async def _drain_graph_astream(
    graph,
    initial_state: dict[str, Any],
    config: dict[str, Any],
    state: dict[str, Any],
    *,
    is_zh: bool,
):
    """Run ``graph.astream`` while flushing token SSE via the side-channel queue.

    LangGraph may buffer ``get_stream_writer()`` custom events until a node
    returns; ``_stream_chain`` / ``_stream_text`` also push into a bound
    asyncio.Queue so the client sees tokens as they are produced.
    """
    sse_q: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    bind_tokens = bind_sse_queue(sse_q, loop)
    # Also stash on session state (configurable["chains"]) — most reliable
    # resolution path inside LangGraph nodes.
    attach_sse_to_session_state(state, sse_q, loop)
    astream_q: asyncio.Queue = asyncio.Queue()
    # Stash queue on runnable config so graph nodes can emit even when
    # contextvars are not inherited into LangGraph's node tasks.
    merged_config = dict(config or {})
    conf = dict(merged_config.get("configurable") or {})
    conf.update(sse_config_keys(sse_q, loop))
    # Ensure chains points at the live session state (with SSE stash).
    conf.setdefault("chains", state)
    merged_config["configurable"] = conf
    # Count side-channel token/stream_start emits so we only skip LangGraph
    # custom duplicates when the side-channel actually delivered them.
    side_channel_live_events = 0

    async def _run_astream() -> None:
        try:
            async for stream_mode, data in graph.astream(
                initial_state, config=merged_config, stream_mode=["updates", "custom"]
            ):
                await astream_q.put((stream_mode, data))
        finally:
            await astream_q.put(None)

    task = asyncio.create_task(_run_astream())
    astream_waiter: asyncio.Task | None = asyncio.create_task(astream_q.get())
    sse_waiter: asyncio.Task | None = asyncio.create_task(sse_q.get())
    astream_done = False

    try:
        while True:
            if await _is_cancelled(state["session_id"]):
                state["status"] = "stopped"
                state.setdefault("history", []).append(
                    {
                        "role": "assistant",
                        "content": "用户已停止本次运行。" if is_zh else "Run stopped by user.",
                        "role_id": "principal_investigator",
                    }
                )
                await _session_store.save(state["session_id"])
                yield f"event: state\ndata: {_to_json(_snapshot(state))}\n\n"
                yield "event: done\ndata: {}\n\n"
                task.cancel()
                return

            wait_set = {t for t in (astream_waiter, sse_waiter) if t is not None}
            if not wait_set:
                break
            done, _ = await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)

            if sse_waiter is not None and sse_waiter in done:
                try:
                    event = sse_waiter.result()
                except Exception:
                    event = None
                if isinstance(event, dict):
                    event_type = event.get("type", "token")
                    if event_type in ("token", "stream_start", "thinking"):
                        side_channel_live_events += 1
                    yield f"event: {event_type}\ndata: {_to_json(event)}\n\n"
                sse_waiter = None if astream_done else asyncio.create_task(sse_q.get())

            if astream_waiter is not None and astream_waiter in done:
                item = astream_waiter.result()
                if item is None:
                    astream_done = True
                    astream_waiter = None
                    # Drain any remaining side-channel tokens, then stop.
                    while True:
                        try:
                            event = sse_q.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                        if isinstance(event, dict):
                            event_type = event.get("type", "token")
                            if event_type in ("token", "stream_start", "thinking"):
                                side_channel_live_events += 1
                            yield f"event: {event_type}\ndata: {_to_json(event)}\n\n"
                    if sse_waiter is not None and not sse_waiter.done():
                        sse_waiter.cancel()
                    sse_waiter = None
                    break

                stream_mode, data = item
                # Custom events are usually flushed via the side-channel when
                # nodes use ``_emit_custom``. Forward them when the side-channel
                # did not deliver (otherwise Expert looks like a one-shot dump).
                if stream_mode == "custom":
                    event_type = data.get("type", "token") if isinstance(data, dict) else "token"
                    if (
                        isinstance(data, dict)
                        and event_type in ("token", "stream_start", "thinking")
                        and side_channel_live_events > 0
                    ):
                        pass
                    else:
                        yield f"event: {event_type}\ndata: {_to_json(data)}\n\n"
                elif stream_mode == "updates":
                    cancelled = await _is_cancelled(state["session_id"])
                    for _, updates in data.items():
                        if updates:
                            for key, val in updates.items():
                                if key in _STREAM_STATE_KEYS:
                                    if cancelled and key == "status":
                                        continue
                                    state[key] = val
                    yield f"event: state\ndata: {_to_json(_snapshot(state))}\n\n"
                astream_waiter = asyncio.create_task(astream_q.get())
    finally:
        detach_sse_from_session_state(state)
        reset_sse_queue(bind_tokens)
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        for waiter in (astream_waiter, sse_waiter):
            if waiter is not None and not waiter.done():
                waiter.cancel()


async def _finalize_after_stream(state: dict[str, Any], user_text: str) -> None:
    """Run post-astream finalization shared by ``_stream_graph`` + ``_stream_graph_resume``.

    If the run did not stop at a waiting checkpoint, append dialogue memory,
    log the assistant's final message, mark status=completed, and archive the
    conversation (fire-and-forget).
    """
    status = str(state.get("status", "") or "")
    if status in _WAITING_STATUSES:
        return
    final_content = state["history"][-1]["content"] if state.get("history") else ""
    _append_dialogue_memory(state, user_text, final_content)
    if final_content:
        state.setdefault("conversation_log", []).append(
            {"role": "assistant", "content": final_content, "timestamp": datetime.now().isoformat()}
        )
    try:
        state["memory"].save_context({"input": user_text}, {"output": final_content})
    except Exception:
        pass
    # Keep planning_failed / error / stopped visible for PipelineProgress + Retry.
    if status not in _FAILURE_STATUSES:
        state["status"] = "completed"
    asyncio.create_task(_archive_conversation(state))


async def _stream_graph(
    state: dict[str, Any],
    text: str,
    attachment_paths: list[str],
):
    if await _is_cancelled(state["session_id"]):
        state["status"] = "stopped"
        await _session_store.save(state["session_id"])
        yield f"event: state\ndata: {_to_json(_snapshot(state))}\n\n"
        yield "event: done\ndata: {}\n\n"
        return

    agent_session_dir = state.get("agent_session_dir")
    if not agent_session_dir:
        raise RuntimeError("Agent session directory is missing.")
    os.makedirs(agent_session_dir, exist_ok=True)

    valid_attachments: list[str] = []
    for p in attachment_paths or []:
        normalized = await _normalize_uploaded_file(
            p,
            agent_session_dir,
            state.setdefault("temp_files", []),
            str(state.get("owner_key", "")),
        )
        if normalized:
            valid_attachments.append(normalized)

    display_text = text or ""
    is_zh = _is_zh_text(display_text)
    if valid_attachments:
        names = ", ".join([os.path.basename(p) for p in valid_attachments])
        attached_label = "已附加" if is_zh else "Attached"
        display_text = (display_text + f"\n📎 *{attached_label}: {names}*").strip()

    if not display_text:
        await _session_store.save(state["session_id"])
        yield f"event: state\ndata: {_to_json(_snapshot(state))}\n\n"
        yield "event: done\ndata: {}\n\n"
        return

    history_len = len(state.get("history") or [])
    if history_len >= AGENT_CHAT_MAX_MESSAGES:
        limit_msg = (
            f"已达上限。本会话最多允许 {AGENT_CHAT_MAX_MESSAGES} 条消息，请新建会话继续。"
            if is_zh else
            f"Limit reached. This chat has reached the maximum of {AGENT_CHAT_MAX_MESSAGES} messages. Start a new chat to continue."
        )
        state["history"].append(
            {"role": "assistant", "content": limit_msg, "role_id": "principal_investigator"}
        )
        await _session_store.save(state["session_id"])
        yield f"event: state\ndata: {_to_json(_snapshot(state))}\n\n"
        yield "event: done\ndata: {}\n\n"
        return

    state["history"].append({"role": "user", "content": display_text})
    state.setdefault("conversation_log", []).append(
        {"role": "user", "content": display_text, "timestamp": datetime.now().isoformat()}
    )

    protein_ctx = state["protein_context"]
    sequence = extract_sequence_from_message(text)
    uniprot_id = extract_uniprot_id_from_message(text)
    if sequence:
        protein_ctx.add_sequence(sequence)
    if uniprot_id:
        protein_ctx.add_uniprot_id(uniprot_id)
    for fp in valid_attachments:
        protein_ctx.add_file(fp)

    state["last_user_text"] = text
    state["last_attachment_paths"] = valid_attachments
    state["status"] = "started"
    state["engine"] = "graph"
    state["chat_mode"] = "science_expert"
    # Expert path is graph-only — never inherit Agent AskUser/Approve gates.
    clear_agent_gates(state)
    state["waiting_for"] = "skip_to_plan" if (
        bool(state.get("has_prior_research")) and bool(state.get("pi_report"))
    ) else ""
    state["clarification_questions"] = []
    state["clarification_answers"] = []

    skip_research = bool(state.get("has_prior_research")) and bool(state.get("pi_report"))
    if skip_research:
        state["has_prior_research"] = False

    initial_state = {
        "messages": [HumanMessage(content=display_text)],
        "protein_context": protein_ctx,
        "session_id": state["session_id"],
        "agent_session_dir": agent_session_dir,
        "history": list(state["history"]),
        "conversation_log": list(state.get("conversation_log", [])),
        "dialogue_memory": list(state.get("dialogue_memory", [])),
        "tool_executions": list(state.get("tool_executions", [])),
        "tool_cache": dict(state.get("tool_cache", {})),
        "status": "started",
        "pi_report": state.get("pi_report", "") if skip_research else "",
        "pi_suggest_steps": state.get("pi_suggest_steps", "") if skip_research else "",
        "plan": [],
        "current_step_index": 0,
        "step_results": {},
        "error": None,
        "research_sections": list(state.get("research_sections", [])) if skip_research else [],
        "research_idx": 0,
        "search_idx": 0,
        "current_search_results": [],
        "research_sub_reports": list(state.get("research_sub_reports", [])) if skip_research else [],
        "execution_failed": False,
        "failed_step": None,
        "failed_reason": None,
        "clarification_questions": [],
        "clarification_answers": [],
        "waiting_for": "skip_to_plan" if skip_research else None,
        # Expert defaults: do not pause after every sub-report; paper-level SC.
        "review_sub_reports": state.get("review_sub_reports") is True,
        "full_manuscript": True if state.get("full_manuscript") is None else bool(state.get("full_manuscript")),
        # Prefer UI locale (user_lang) so graph nodes don't CJK-guess wrong.
        "user_lang": state.get("user_lang") or "",
        "ui_lang": state.get("user_lang") or state.get("ui_lang") or "",
    }
    graph = _get_compiled_graph()
    config = {
        "configurable": {"chains": state, "session_id": state["session_id"]},
        "recursion_limit": 100,
    }

    yield f"event: state\ndata: {_to_json(_snapshot(state))}\n\n"

    # Fire on_run_start hook (fire-and-forget so it can't slow the stream)
    _stream_hooks = state.get("hooks")
    if _stream_hooks is not None:
        _dispatch_hook(_stream_hooks.on_run_start(
            session_id=state["session_id"],
            user_message=display_text[:500],
        ))

    async for chunk in _drain_graph_astream(
        graph, initial_state, config, state, is_zh=is_zh
    ):
        yield chunk
        if chunk.startswith("event: done"):
            return

    await _finalize_after_stream(state, display_text)
    await _session_store.save(state["session_id"])
    yield f"event: state\ndata: {_to_json(_snapshot(state))}\n\n"
    yield "event: done\ndata: {}\n\n"

    # Fire on_run_end hook after the SSE stream is closed out.
    if _stream_hooks is not None:
        _dispatch_hook(_stream_hooks.on_run_end(
            session_id=state["session_id"],
            success=state.get("status") not in ("error", "execution_failed"),
            total_steps=int(state.get("current_step_index", 0) or 0),
            total_tool_calls=len(state.get("tool_executions", []) or []),
        ))


