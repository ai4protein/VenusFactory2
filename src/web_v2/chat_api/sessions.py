"""Session CRUD + cancel + custom-model cache endpoints."""
import asyncio
import shutil
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from agent.chat_agent import initialize_session_state

from web_v2.chat_api._hooks_runtime import (
    _SESSION_CANCEL_FLAGS,
    _SESSION_LOCKS,
    _SESSIONS_GUARD,
    _get_lock,
    _logger,
    _session_store,
    _set_cancel,
)
from web_v2.chat_api._models import CreateSessionResponse, SessionStateResponse
from web_v2.chat_api._shared import (
    _assert_session_access,
    _extract_client_ip,
    _get_session_or_404,
    _issue_session_access_token,
    _record_access_event,
    _runtime_mode,
    _session_owner_key_for_request,
    _snapshot,
)

router = APIRouter()


@router.post("/sessions", response_model=CreateSessionResponse)
async def create_session(request: Request):
    _record_access_event(request, "/api/chat/sessions:create")
    state = initialize_session_state()
    token, token_expires_at = _issue_session_access_token(state, request)
    state["client_ip"] = _extract_client_ip(request)
    state["owner_key"] = _session_owner_key_for_request(request)
    session_id = state["session_id"]
    await _session_store.create(state)
    async with _SESSIONS_GUARD:
        _SESSION_LOCKS[session_id] = asyncio.Lock()
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
            summaries = await list_summaries(owner_key)
            data = [
                {
                    "session_id": s["session_id"],
                    "created_at": str(s.get("created_at", "")),
                    "model_name": s.get("model_name", ""),
                    "history_size": s.get("history_size", 0),
                    "status": s.get("status", ""),
                    "title": s.get("title") or "",
                }
                for s in summaries
            ]
            used_summaries = True
        except Exception:
            _logger.debug("list_summaries failed; falling back to list_ids", exc_info=True)
    if not used_summaries:
        # TODO: remove this fallback once SessionStore.list_summaries is guaranteed.
        from agent.session_store import session_title_from_history

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
                }
            )
    return {"sessions": data}


@router.get("/sessions/{session_id}", response_model=SessionStateResponse)
async def get_session(session_id: str, request: Request):
    _record_access_event(request, "/api/chat/sessions/{id}:get")
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)
    snap = _snapshot(state)
    return SessionStateResponse(**snap)


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, request: Request):
    _record_access_event(request, "/api/chat/sessions/{id}:delete")
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)
    lock = await _get_lock(session_id)
    if lock.locked():
        raise HTTPException(status_code=409, detail="Session is currently running.")

    async with lock:
        current = await _session_store.get(session_id)
        if current is None:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
        _assert_session_access(current, request)
        await _session_store.delete(session_id)
        async with _SESSIONS_GUARD:
            _SESSION_LOCKS.pop(session_id, None)
            _SESSION_CANCEL_FLAGS.pop(session_id, None)

        session_dir = state.get("agent_session_dir")
        if session_dir:
            try:
                shutil.rmtree(session_dir, ignore_errors=True)
            except Exception:
                pass
    return {"success": True, "session_id": session_id}


@router.get("/sessions/{session_id}/files")
async def list_session_files(session_id: str, request: Request):
    """List every file under this session's working directory.

    Used by the frontend's session files panel. Returns paths relative to
    `agent_session_dir` plus size + mtime so the UI can render a tree and
    link each entry through `/api/files/inline?path=...` for inline preview.
    """
    import os as _os
    _record_access_event(request, "/api/chat/sessions/{id}/files")
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)
    sdir = state.get("agent_session_dir") or ""
    if not sdir or not _os.path.isdir(sdir):
        return {"session_dir": sdir, "files": []}

    files: list[dict[str, Any]] = []
    max_entries = 500
    # Sort by mtime desc so newest artifacts surface first.
    for dirpath, _dirs, filenames in _os.walk(sdir):
        for name in filenames:
            full = _os.path.join(dirpath, name)
            try:
                st = _os.stat(full)
            except OSError:
                continue
            rel = _os.path.relpath(full, sdir)
            files.append({
                "name": name,
                "rel": rel,
                "abs": full,
                "size": st.st_size,
                "mtime": st.st_mtime,
                "ext": _os.path.splitext(name)[1].lower().lstrip("."),
            })
            if len(files) >= max_entries:
                break
        if len(files) >= max_entries:
            break
    files.sort(key=lambda f: f["mtime"], reverse=True)
    return {"session_dir": sdir, "files": files}


@router.post("/sessions/{session_id}/cancel")
async def cancel_session_run(session_id: str, request: Request):
    _record_access_event(request, "/api/chat/sessions/{id}/cancel")
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)
    state["status"] = "stopping"
    await _set_cancel(session_id, True)
    await _session_store.save(session_id)
    return {"success": True, "status": "stopping"}


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
