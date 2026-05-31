"""Streaming chat message endpoints (new message + retry)."""
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from agent.chat_agent import (
    update_llm_model,
    update_llm_openai_style_config,
)

from web_v2.chat_api._hooks_runtime import (
    _BUILTIN_MODEL_LABELS,
    _get_lock,
    _set_cancel,
)
from web_v2.chat_api._models import ChatStreamRequest
from web_v2.chat_api._shared import (
    _assert_session_access,
    _consume_online_chat_quota_or_429,
    _extract_client_ip,
    _get_session_or_404,
    _record_access_event,
    _runtime_mode,
    _session_owner_key_for_request,
)
from web_v2.chat_api._stream import _stream_graph

router = APIRouter()


@router.post("/sessions/{session_id}/messages/stream")
async def stream_message(session_id: str, payload: ChatStreamRequest, request: Request):
    _record_access_event(request, "/api/chat/sessions/{id}/messages/stream")
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)
    if _runtime_mode() != "local" and (payload.custom_model_config or payload.custom_model_id):
        raise HTTPException(status_code=403, detail="Custom models are available only in local mode.")
    await _consume_online_chat_quota_or_429(request)
    await _set_cancel(session_id, False)
    state["client_ip"] = _extract_client_ip(request)
    state["owner_key"] = _session_owner_key_for_request(request)
    if payload.model in _BUILTIN_MODEL_LABELS:
        llm = state.get("llm")
        if llm is not None:
            default_api_key = str(state.get("default_llm_api_key", "") or "")
            default_base_url = str(state.get("default_llm_base_url", "") or "")
            if default_api_key:
                llm.api_key = default_api_key
            if default_base_url:
                llm.base_url = default_base_url
    if payload.custom_model_config and _runtime_mode() == "local":
        cfg = payload.custom_model_config
        update_llm_openai_style_config(
            state=state,
            model_name=str(cfg.get("model_name", "") or ""),
            api_key=str(cfg.get("api_key", "") or ""),
            base_url=str(cfg.get("base_url", "") or ""),
        )
        if payload.custom_model_id:
            state["active_custom_model_id"] = payload.custom_model_id
    elif payload.model in _BUILTIN_MODEL_LABELS:
        state["active_custom_model_id"] = ""
    if payload.model:
        update_llm_model(payload.model, state)
    lock = await _get_lock(session_id)

    async def event_gen():
        async with lock:
            async for chunk in _stream_graph(
                state,
                payload.text or "",
                payload.attachment_paths or [],
            ):
                yield chunk

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@router.post("/sessions/{session_id}/messages/retry/stream")
async def stream_retry(session_id: str, request: Request):
    _record_access_event(request, "/api/chat/sessions/{id}/messages/retry/stream")
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)
    await _consume_online_chat_quota_or_429(request)
    await _set_cancel(session_id, False)
    state["client_ip"] = _extract_client_ip(request)
    state["owner_key"] = _session_owner_key_for_request(request)
    lock = await _get_lock(session_id)
    last_text = state.get("last_user_text", "")
    last_paths = state.get("last_attachment_paths", [])

    if not last_text and not last_paths:
        raise HTTPException(status_code=400, detail="No previous user message to retry.")

    async def event_gen():
        async with lock:
            async for chunk in _stream_graph(state, last_text, last_paths):
                yield chunk

    return StreamingResponse(event_gen(), media_type="text/event-stream")
