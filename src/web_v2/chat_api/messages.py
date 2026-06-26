"""Streaming chat message endpoints (new message + retry)."""
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from agent.chat_agent import (
    update_llm_model,
    update_llm_openai_style_config,
)
from agent.model_registry import get_model

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
from web_v2.chat_api._stream_kimi import _stream_kimi

router = APIRouter()


def _resolve_engine(payload_engine: str | None, model_id: str | None) -> str:
    """Decide whether this turn runs through the kimi-code daemon or the
    legacy LangGraph pipeline.

    Precedence:
      1. The model's registry entry. A model with `engine: kimi-code` forces
         the kimi path regardless of the request (so the UI's model selector
         is the single source of truth).
      2. Explicit `engine` field on the request payload (advanced clients).
      3. Default: "graph".
    """
    if model_id:
        spec = get_model(model_id)
        if spec is not None and (spec.engine or "graph") == "kimi-code":
            return "kimi-code"
    if payload_engine in ("kimi-code", "graph"):
        return payload_engine
    return "graph"


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
        # Skip LLM hot-swap for kimi-code engine — kimi manages its own model
        # selection internally and trying to instantiate a LangChain LLM for
        # the "kimi-code" pseudo-model fails.
        engine_for_swap = _resolve_engine(payload.engine, payload.model)
        if engine_for_swap != "kimi-code":
            update_llm_model(payload.model, state)
    # Pin the response language from the UI locale ("en" | "zh"). Stored on
    # state so retries (which carry no fresh payload) inherit it; also
    # stamped on the LLM instance so chat_agent._build_message_dicts can
    # prepend a forced-language directive on every graph-engine call.
    if payload.lang in ("en", "zh"):
        state["user_lang"] = payload.lang
    _llm = state.get("llm")
    if _llm is not None:
        _llm._user_lang = state.get("user_lang") or ""
    lock = await _get_lock(session_id)
    engine = _resolve_engine(payload.engine, payload.model)
    streamer = _stream_kimi if engine == "kimi-code" else _stream_graph

    async def event_gen():
        async with lock:
            async for chunk in streamer(
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

    # On retry we don't have a fresh payload; re-derive the engine from the
    # currently-active model so retries use the same backend as the original.
    llm = state.get("llm")
    model_id = getattr(llm, "model_name", "") if llm is not None else ""
    engine = _resolve_engine(None, model_id)
    streamer = _stream_kimi if engine == "kimi-code" else _stream_graph
    # Carry forward the last-known user_lang so retries respect the same
    # forced-language policy as the original turn.
    if llm is not None:
        llm._user_lang = state.get("user_lang") or ""

    async def event_gen():
        async with lock:
            async for chunk in streamer(state, last_text, last_paths):
                yield chunk

    return StreamingResponse(event_gen(), media_type="text/event-stream")
