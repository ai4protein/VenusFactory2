"""Session CRUD + cancel + custom-model cache endpoints."""
import asyncio
import shutil
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response

from backend.web_v2.chat_runtime import initialize_session_state

from backend.web_v2.chat_api._hooks_runtime import (
    _SESSION_CANCEL_FLAGS,
    _SESSION_LOCKS,
    _SESSIONS_GUARD,
    _get_lock,
    _logger,
    _session_store,
    _set_cancel,
)
from backend.web_v2.chat_api._models import CreateSessionRequest, CreateSessionResponse, SessionStateResponse
from backend.web_v2.chat_api._shared import (
    _assert_session_access,
    _extract_client_ip,
    _get_session_or_404,
    _issue_session_access_token,
    _historical_session_ids,
    _record_access_event,
    _runtime_mode,
    _session_owner_key_for_request,
    _snapshot,
)

from backend.web_v2.science_goals import service as goal_service
from backend.web_v2.chat_history import include_research_content

router = APIRouter()


def visible_run_status(session_id, status, queue=None):
    from backend.web_v2.job_queue import JobQueue
    run = (queue or JobQueue()).latest_run('chat:' + session_id)
    if run:
        return 'stopping' if run['cancel_requested'] and run['status']=='running' else {'running':'running','queued':'queued','interrupted':'interrupted','failed':'error','cancelled':'stopped'}.get(run['status'],status)
    # A finished/old cancellation must not leave a permanent running indicator.
    lock = _SESSION_LOCKS.get(session_id)
    if status == "stopping" and not (lock and lock.locked()):
        return "stopped"
    return status


@router.post("/sessions", response_model=CreateSessionResponse)
async def create_session(request: Request, payload: CreateSessionRequest | None = None):
    _record_access_event(request, "/api/chat/sessions:create")
    from backend.web_v2.online_access import check_new_conversation
    await asyncio.to_thread(check_new_conversation, request)
    state = initialize_session_state()
    state["chat_mode"] = (payload or CreateSessionRequest()).chat_mode
    token, token_expires_at = _issue_session_access_token(state, request)
    state["client_ip"] = _extract_client_ip(request)
    state["owner_key"] = _session_owner_key_for_request(request)
    session_id = state["session_id"]
    await _session_store.create(state)
    async with _SESSIONS_GUARD:
        from backend.web_v2.session_runtime import SharedSessionLock
        from backend.web_v2.job_queue import JobQueue
        _SESSION_LOCKS[session_id] = SharedSessionLock(JobQueue().root, session_id)
        _SESSION_CANCEL_FLAGS[session_id] = False
    return CreateSessionResponse(
        session_id=session_id,
        created_at=str(state.get("created_at", "")),
        model_name=getattr(state.get("llm"), "model_name", ""),
        session_access_token=token,
        token_expires_at=token_expires_at,
    )


@router.get("/sessions")
async def list_sessions(request: Request):
    _record_access_event(request, "/api/chat/sessions:list")
    owner_key = None if _runtime_mode() != "online" else _session_owner_key_for_request(request)
    data: list[dict[str, Any]] = []
    # Prefer cheap summary listing (no runtime rebuild). Fallback to legacy
    # list_ids + get() if SessionStore hasn't grown list_summaries yet.
    list_summaries = getattr(_session_store, "list_summaries", None)
    used_summaries = False
    if list_summaries is not None:
        try:
            summaries = await list_summaries(owner_key, extra_session_ids=await asyncio.to_thread(_historical_session_ids, request)) if owner_key is not None else await list_summaries(None)
            data = [
                {
                    "session_id": s["session_id"],
                    "created_at": str(s.get("created_at", "")),
                    "model_name": s.get("model_name", ""),
                    "history_size": s.get("history_size", 0),
                    "status": s.get("status", ""),
                    "title": s.get("title") or "",
                    "has_content": s.get("has_content", bool(s.get("history_size") or s.get("title"))),
                    "chat_mode": s.get("chat_mode", "science_agent"),
                }
                for s in summaries
            ]
            used_summaries = True
        except Exception:
            _logger.debug("list_summaries failed; falling back to list_ids", exc_info=True)
    if not used_summaries:
        # TODO: remove this fallback once SessionStore.list_summaries is guaranteed.
        from agent.session_store import session_title_from_history, session_has_content, session_chat_mode

        sids = await _session_store.list_ids(owner_key)
        for sid in sids:
            s = await _session_store.get(sid)
            if s is None:
                continue
            history = s.get("history") or []
            data.append(
                {
                    "session_id": sid,
                    "created_at": str(s.get("created_at", "")),
                    "model_name": getattr(s.get("llm"), "model_name", ""),
                    "history_size": len(history),
                    "status": s.get("status", ""),
                    "title": session_title_from_history(history),
                    "has_content": session_has_content(s),
                    "chat_mode": session_chat_mode(s),
                }
            )
    if _runtime_mode() == "online":
        from backend.web_v2.online_visibility import OnlineVisibility
        hidden = await asyncio.to_thread(OnlineVisibility().hidden_ids, "chat")
        data = [row for row in data if row["session_id"] not in hidden]
    def with_research():
        from backend.web_v2.job_queue import JobQueue
        queue = JobQueue()
        for item in data:
            item["status"] = visible_run_status(item["session_id"], item["status"], queue)
        return [include_research_content(item, goal_service.session_folder(item["session_id"]), goal_service.live)
                for item in data]
    return {"sessions": await asyncio.to_thread(with_research)}


@router.get("/sessions/{session_id}", response_model=SessionStateResponse)
async def get_session(session_id: str, request: Request, response: Response):
    _record_access_event(request, "/api/chat/sessions/{id}:get")
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)
    # Soft-navigate back must see live checkpoint state, not a cached Thinking bubble.
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    snap = _snapshot(state)
    snap["status"] = await asyncio.to_thread(visible_run_status, session_id, snap.get("status", ""))
    from backend.web_v2.job_queue import JobQueue
    snap["active_run"] = await asyncio.to_thread(JobQueue().latest_run, "chat:" + session_id)
    if snap["active_run"] and not snap.get("event_cursor", "").startswith(snap["active_run"]["id"] + ":"):
        snap["event_cursor"] = ""
    return SessionStateResponse(**snap)


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, request: Request):
    _record_access_event(request, "/api/chat/sessions/{id}:delete")
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)
    from backend.web_v2.job_queue import JobQueue
    run = await asyncio.to_thread(JobQueue().latest_run, "chat:" + session_id)
    lock = await _get_lock(session_id)
    if lock.locked() or run and run["status"] in {"queued", "running"}:
        raise HTTPException(status_code=409, detail="Session is currently running.")

    async with lock:
        current = await _session_store.get(session_id)
        if current is None:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
        _assert_session_access(current, request)
        if await asyncio.to_thread(goal_service.has_unfinished_goal, session_id):
            raise HTTPException(409, "Cancel the unfinished Science goal before deleting this session.")
        if _runtime_mode() == "online":
            from backend.web_v2.online_visibility import OnlineVisibility
            roots = [state.get("agent_session_dir"), goal_service.session_folder(session_id)]
            await asyncio.to_thread(OnlineVisibility().hide, "chat", session_id, roots)
        else:
            await _session_store.delete(session_id)
        async with _SESSIONS_GUARD:
            _SESSION_LOCKS.pop(session_id, None)
            _SESSION_CANCEL_FLAGS.pop(session_id, None)

        session_dir = state.get("agent_session_dir")
        if session_dir and _runtime_mode() != "online":
            try:
                shutil.rmtree(session_dir, ignore_errors=True)
            except Exception:
                pass
    return {"success": True, "session_id": session_id}


@router.get("/sessions/{session_id}/files")
async def list_session_files(session_id: str, request: Request):
    from backend.web_v2.research_files import session_roots, list_files
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)
    return await asyncio.to_thread(list_files, session_roots(state, goal_service, session_id))


@router.post("/sessions/{session_id}/cancel")
async def cancel_session_run(session_id: str, request: Request):
    _record_access_event(request, "/api/chat/sessions/{id}/cancel")
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)
    goals = await asyncio.to_thread(goal_service.list, session_id)
    for goal in goals:
        if goal["status"] == "active":
            await asyncio.to_thread(goal_service.control, session_id, goal["id"], "pause")
    if goals and any(g["worker_alive"] or g["status"] == "active" for g in goals):
        return {"success": True, "status": "paused"}
    from backend.web_v2.job_queue import JobQueue
    queue = JobQueue()
    run = await asyncio.to_thread(queue.latest_run, 'chat:' + session_id)
    if run and run['status'] in {'queued','running'}:
        await _set_cancel(session_id, True)
        result = await asyncio.to_thread(queue.cancel, run['id'], state.get('owner_key') or 'local')
        return {'success': True, 'status': 'stopping' if result['status']=='running' else 'stopped'}
    state["status"] = visible_run_status(session_id, "stopping")
    await _set_cancel(session_id, True)
    await _session_store.save(session_id)
    return {"success": True, "status": state["status"]}


@router.delete("/models/custom/{custom_model_id}")
async def delete_custom_model_cache(custom_model_id: str, request: Request):
    _record_access_event(request, "/api/chat/models/custom/{id}:delete")
    removed_sessions: list[str] = []
    owner_key_for_req = _session_owner_key_for_request(request)
    online = _runtime_mode() == "online"
    sids = await _session_store.list_ids(owner_key_for_req if online else None)
    for sid in sids:
        state = await _session_store.get(sid)
        if state is None:
            continue
        if online and str(state.get("owner_key", "")) != owner_key_for_req:
            continue
        if str(state.get("active_custom_model_id", "")) != custom_model_id:
            continue
        llm = state.get("llm")
        if llm is not None:
            default_api_key = str(state.get("default_llm_api_key", "") or "")
            default_base_url = str(state.get("default_llm_base_url", "") or "")
            default_model_name = str(state.get("default_llm_model_name", "") or "")
            if default_api_key:
                llm.api_key = default_api_key
            if default_base_url:
                llm.base_url = default_base_url
            if default_model_name:
                llm.model_name = default_model_name
        state["active_custom_model_id"] = ""
        await _session_store.save(sid)
        removed_sessions.append(sid)
    return {"success": True, "custom_model_id": custom_model_id, "cleared_sessions": removed_sessions}
