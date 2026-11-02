"""Streaming chat message endpoints (new message + retry)."""
import asyncio

from fastapi import APIRouter, HTTPException, Request

from agent.chat_agent import (
    update_llm_model,
    update_llm_openai_style_config,
)
from agent.model_config import apply_to_llm, get_expert_model_id, load_model_config
from backend.web_v2.agent_context import ContextPolicy, ContextLimit, check_admission
from backend.web_v2.agent_execution import admit as admit_agent
from backend.web_v2.agent_execution import options as agent_options
from backend.web_v2.chat_api._hooks_runtime import (
    _BUILTIN_MODEL_LABELS,
    _get_lock,
    _set_cancel,
)
from agent.harness.interactions import InteractionResponse
from backend.web_v2.chat_api._models import ChatStreamRequest
from backend.web_v2.chat_api._shared import (
    _assert_session_access,
    _consume_online_chat_quota_or_429,
    _extract_client_ip,
    _get_session_or_404,
    _record_access_event,
    _resolve_chat_mode,
    _resolve_engine,
    _runtime_mode,
    _session_owner_key_for_request,
    _sse_response,
)
from backend.web_v2.chat_api._stream import _stream_graph
from backend.web_v2.chat_api._stream_agent import _stream_agent
from backend.web_v2.chat_api._stream_kimi import _stream_kimi
from backend.web_v2.science_goals import service as goal_service

router = APIRouter()


async def _assert_no_goal(session_id):
    if await asyncio.to_thread(goal_service.has_running_goal, session_id):
        raise HTTPException(409, "Pause the running Science goal before starting a conversation turn.")


@router.get("/agent-execution-options")
async def execution_options():
    # Availability comes from the same free-only catalog in both deployment modes.
    return await asyncio.to_thread(agent_options)


async def selected_agent_response(state, request, payload=None, *, retry=False):
    from backend.web_v2.chat_runs import submit
    return await submit(state, request, payload, retry=retry)


@router.get("/sessions/{session_id}/runs/{run_id}/events")
async def run_events(session_id: str, run_id: str, request: Request, after: str = ""):
    from backend.web_v2.chat_runs import response
    from backend.web_v2.job_queue import JobQueue
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)
    owner = _session_owner_key_for_request(request)
    queue = JobQueue()
    try:
        run = await asyncio.to_thread(queue.get, run_id, owner)
    except FileNotFoundError:
        raise HTTPException(404, "Run not found") from None
    if run.get("session_key") != "chat:" + session_id:
        raise HTTPException(404, "Run not found")
    return response(queue, run_id, owner, session_id, after or request.headers.get("last-event-id", ""))


@router.post("/sessions/{session_id}/messages/stream")
async def stream_message(session_id: str, payload: ChatStreamRequest, request: Request):
    _record_access_event(request, "/api/chat/sessions/{id}/messages/stream")
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)
    await _assert_no_goal(session_id)
    if payload.chat_mode == "science_expert" or state.get("chat_mode") == "science_expert":
        raise HTTPException(409, "Use Start research in Scientist mode for a bounded online study")
    if payload.execution is not None or state.get("agent_execution") or _resolve_chat_mode(_resolve_engine(payload.engine, payload.model, payload.chat_mode), payload.chat_mode) == "science_agent":
        return await selected_agent_response(state, request, payload)
    raise HTTPException(409, "Use Start research in Scientist mode")


@router.post("/sessions/{session_id}/messages/retry/stream")
async def stream_retry(session_id: str, request: Request):
    _record_access_event(request, "/api/chat/sessions/{id}/messages/retry/stream")
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)
    await _assert_no_goal(session_id)
    if state.get("chat_mode") == "science_expert":
        raise HTTPException(409, "Use the Scientist study controls to continue research")
    return await selected_agent_response(state, request, retry=True)


@router.post("/sessions/{session_id}/interactions/respond/stream")
async def respond_interaction(session_id: str, payload: InteractionResponse, request: Request):
    from backend.web_v2.chat_runs import submit
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)
    await _assert_no_goal(session_id)
    if state.get("chat_mode") != "science_agent":
        raise HTTPException(409, "Use the research input endpoint for Scientist")
    return await submit(state, request, interaction=payload)
