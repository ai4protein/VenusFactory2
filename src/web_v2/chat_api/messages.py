"""Streaming chat message endpoints (new message + retry)."""
from fastapi import APIRouter, HTTPException, Request

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
    _resolve_chat_mode,
    _resolve_engine,
    _runtime_mode,
    _session_owner_key_for_request,
    _sse_response,
)
from web_v2.chat_api._stream import _stream_graph
from web_v2.chat_api._stream_kimi import _stream_kimi

router = APIRouter()

# Online deployments pin Science Expert (graph) to DeepSeek — clients cannot
# pick GPT/Claude/etc. Science Agent still routes through kimi-code.
_ONLINE_FIXED_GRAPH_MODEL = "deepseek-v4-pro"


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

    engine = _resolve_engine(payload.engine, payload.model, payload.chat_mode)
    chat_mode = _resolve_chat_mode(engine, payload.chat_mode)
    # Online: ignore client model for graph / Science Expert — always DeepSeek.
    # Science Agent (kimi-code) keeps client model in local mode so the picker
    # can choose the underlying LLM forwarded to kimi create_session.
    graph_model = payload.model
    if _runtime_mode() != "local" and engine != "kimi-code":
        graph_model = _ONLINE_FIXED_GRAPH_MODEL
        state["active_custom_model_id"] = ""

    if engine == "kimi-code":
        from agent.kimi_model import to_kimi_model_id

        # Online Agent: always kimi default. Local Agent: honor picker.
        selected = "" if _runtime_mode() != "local" else (graph_model or "")
        kimi_model = to_kimi_model_id(selected)
        prev = str(state.get("kimi_model") or "")
        state["kimi_model"] = kimi_model or ""
        # Recreate kimi session when the underlying model changes.
        if prev != state["kimi_model"]:
            state["kimi_session_id"] = ""
            state.pop("_kimi_bound_model", None)
        # Snapshot / retry display: keep a real registry LLM when possible.
        # Never instantiate LangChain for the sentinel id "kimi-code".
        if selected and selected != "kimi-code":
            update_llm_model(selected, state)
    elif graph_model:
        update_llm_model(graph_model, state)
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
    state["engine"] = engine
    state["chat_mode"] = chat_mode
    streamer = _stream_kimi if engine == "kimi-code" else _stream_graph

    async def event_gen():
        async with lock:
            async for chunk in streamer(
                state,
                payload.text or "",
                payload.attachment_paths or [],
            ):
                yield chunk

    return _sse_response(event_gen())


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
    # currently-active model / session chat_mode so retries use the same backend.
    llm = state.get("llm")
    model_id = getattr(llm, "model_name", "") if llm is not None else ""
    engine = _resolve_engine(state.get("engine"), model_id, state.get("chat_mode"))
    chat_mode = _resolve_chat_mode(engine, state.get("chat_mode"))
    # Online graph retries stay pinned to DeepSeek even if session LLM drifted.
    if _runtime_mode() != "local" and engine != "kimi-code":
        update_llm_model(_ONLINE_FIXED_GRAPH_MODEL, state)
        state["active_custom_model_id"] = ""
        llm = state.get("llm")
    state["engine"] = engine
    state["chat_mode"] = chat_mode
    streamer = _stream_kimi if engine == "kimi-code" else _stream_graph
    # Carry forward the last-known user_lang so retries respect the same
    # forced-language policy as the original turn.
    if llm is not None:
        llm._user_lang = state.get("user_lang") or ""

    async def event_gen():
        async with lock:
            async for chunk in streamer(state, last_text, last_paths):
                yield chunk

    return _sse_response(event_gen())
