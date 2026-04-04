import asyncio
import hashlib
import hmac
import json
import os
import secrets
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from agent.chat_agent import initialize_session_state, update_llm_model
from agent.chat_agent_utils import (
    AGENT_CHAT_MAX_MESSAGES,
    extract_sequence_from_message,
    extract_uniprot_id_from_message,
)
from agent.chat_graph import create_agent_graph
from web.utils.common_utils import (
    make_web_v2_upload_name,
    redact_path_text,
    resolve_web_v2_client_path,
    to_web_v2_public_path,
)


router = APIRouter(prefix="/api/chat", tags=["chat-v2"])

_SESSIONS: Dict[str, Dict[str, Any]] = {}
_SESSION_LOCKS: Dict[str, asyncio.Lock] = {}
_SESSION_CANCEL_FLAGS: Dict[str, bool] = {}
_SESSIONS_GUARD = asyncio.Lock()
_IP_DAILY_CHAT_USAGE: Dict[str, Dict[str, Any]] = {}
_IP_USAGE_GUARD = asyncio.Lock()


def _parse_daily_chat_limit() -> int:
    raw = os.getenv("WEBUI_V2_ONLINE_DAILY_CHAT_LIMIT", "10").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 10
    return max(1, value)


_ONLINE_DAILY_CHAT_LIMIT = _parse_daily_chat_limit()
def _parse_session_token_ttl_hours() -> int:
    raw = os.getenv("WEBUI_V2_SESSION_TOKEN_TTL_HOURS", "24").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 24
    return max(1, value)


_SESSION_TOKEN_TTL_HOURS = _parse_session_token_ttl_hours()
_SESSION_TOKEN_SECRET = os.getenv("WEBUI_V2_SESSION_TOKEN_SECRET", "").strip() or secrets.token_hex(32)


def _runtime_mode() -> str:
    mode = os.getenv("WEBUI_V2_MODE", "local").strip().lower()
    return mode if mode in {"local", "online"} else "local"


def _extract_user_agent(request: Request) -> str:
    return (request.headers.get("user-agent", "") or "").strip().lower()


def _extract_origin(request: Request) -> str:
    return (request.headers.get("origin", "") or "").strip().lower()


def _build_owner_fingerprint(request: Request) -> str:
    ip = _extract_client_ip(request)
    ua = _extract_user_agent(request)
    origin = _extract_origin(request)
    raw = f"{ip}|{ua}|{origin}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _session_owner_key_for_request(request: Request) -> str:
    if _runtime_mode() != "online":
        return "local"
    return _build_owner_fingerprint(request)


def _hash_session_token(raw_token: str) -> str:
    return hmac.new(_SESSION_TOKEN_SECRET.encode("utf-8"), raw_token.encode("utf-8"), hashlib.sha256).hexdigest()


def _issue_session_access_token(state: Dict[str, Any], request: Request) -> tuple[str, str]:
    raw_token = secrets.token_urlsafe(48)
    expires_at = datetime.now(timezone.utc) + timedelta(hours=_SESSION_TOKEN_TTL_HOURS)
    state["session_token_hash"] = _hash_session_token(raw_token)
    state["token_expires_at"] = expires_at.isoformat()
    state["owner_key"] = _session_owner_key_for_request(request)
    return raw_token, state["token_expires_at"]


def _extract_session_token(request: Request) -> str:
    custom = (request.headers.get("x-session-access-token", "") or "").strip()
    if custom:
        return custom
    auth = (request.headers.get("authorization", "") or "").strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return ""


async def get_visible_session_ids(request: Request) -> set[str]:
    owner_key = _session_owner_key_for_request(request)
    async with _SESSIONS_GUARD:
        if _runtime_mode() != "online":
            return set(_SESSIONS.keys())
        return {sid for sid, state in _SESSIONS.items() if str(state.get("owner_key", "")) == owner_key}


def _is_zh_text(text: str) -> bool:
    raw = str(text or "")
    return any("\u4e00" <= ch <= "\u9fff" for ch in raw)


class CreateSessionResponse(BaseModel):
    session_id: str
    created_at: str
    model_name: str
    session_access_token: str = ""
    token_expires_at: str = ""


class SessionStateResponse(BaseModel):
    session_id: str
    model_name: str
    created_at: str
    history: List[Dict[str, Any]]
    conversation_log: List[Dict[str, Any]]
    tool_executions: List[Dict[str, Any]]


class ChatStreamRequest(BaseModel):
    text: str = Field(default="")
    model: Optional[str] = Field(default=None)
    attachment_paths: List[str] = Field(default_factory=list)


def _to_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def _redact_obj(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _redact_obj(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_obj(v) for v in value]
    if isinstance(value, str):
        return redact_path_text(value)
    return value


def _snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "session_id": state.get("session_id"),
        "model_name": getattr(state.get("llm"), "model_name", ""),
        "created_at": str(state.get("created_at", "")),
        "history": _redact_obj(list(state.get("history", []))),
        "conversation_log": _redact_obj(list(state.get("conversation_log", []))),
        "tool_executions": _redact_obj(list(state.get("tool_executions", []))),
        "status": state.get("status", ""),
    }


async def _get_session_or_404(session_id: str) -> Dict[str, Any]:
    async with _SESSIONS_GUARD:
        state = _SESSIONS.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    return state


def _assert_session_access(state: Dict[str, Any], request: Request) -> None:
    if _runtime_mode() != "online":
        return
    expected_owner = _session_owner_key_for_request(request)
    actual_owner = str(state.get("owner_key", ""))
    if actual_owner != expected_owner:
        raise HTTPException(
            status_code=403,
            detail={"code": "SESSION_OWNER_MISMATCH", "message": "You do not have access to this session."},
        )
    token = _extract_session_token(request)
    if not token:
        raise HTTPException(
            status_code=401,
            detail={"code": "SESSION_TOKEN_REQUIRED", "message": "Session access token is required."},
        )
    expected_hash = str(state.get("session_token_hash", ""))
    if not expected_hash or not hmac.compare_digest(expected_hash, _hash_session_token(token)):
        raise HTTPException(
            status_code=403,
            detail={"code": "SESSION_TOKEN_INVALID", "message": "Session access token is invalid."},
        )
    expires_raw = str(state.get("token_expires_at", "")).strip()
    if expires_raw:
        try:
            expires_at = datetime.fromisoformat(expires_raw.replace("Z", "+00:00"))
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
        except ValueError:
            expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        if datetime.now(timezone.utc) >= expires_at:
            raise HTTPException(
                status_code=401,
                detail={"code": "SESSION_TOKEN_EXPIRED", "message": "Session access token has expired."},
            )


async def _get_lock(session_id: str) -> asyncio.Lock:
    async with _SESSIONS_GUARD:
        if session_id not in _SESSION_LOCKS:
            _SESSION_LOCKS[session_id] = asyncio.Lock()
        return _SESSION_LOCKS[session_id]


async def _is_cancelled(session_id: str) -> bool:
    async with _SESSIONS_GUARD:
        return _SESSION_CANCEL_FLAGS.get(session_id, False)


async def _set_cancel(session_id: str, value: bool) -> None:
    async with _SESSIONS_GUARD:
        _SESSION_CANCEL_FLAGS[session_id] = value


def _extract_client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        parts = [p.strip() for p in xff.split(",") if p.strip()]
        if parts:
            return parts[0]
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


async def _consume_online_chat_quota_or_429(request: Request) -> None:
    if os.getenv("WEBUI_V2_MODE", "local").strip().lower() != "online":
        return

    owner_key = _session_owner_key_for_request(request)
    today = datetime.now().strftime("%Y-%m-%d")
    async with _IP_USAGE_GUARD:
        usage = _IP_DAILY_CHAT_USAGE.get(owner_key)
        if not usage or usage.get("date") != today:
            usage = {"date": today, "count": 0}
            _IP_DAILY_CHAT_USAGE[owner_key] = usage

        if int(usage.get("count", 0)) >= _ONLINE_DAILY_CHAT_LIMIT:
            raise HTTPException(
                status_code=429,
                detail={
                    "code": "CHAT_DAILY_LIMIT_REACHED",
                    "message": f"Online mode limit reached: up to {_ONLINE_DAILY_CHAT_LIMIT} chats per user per day.",
                },
            )
        usage["count"] = int(usage.get("count", 0)) + 1


async def _get_online_chat_quota_status(request: Request) -> Dict[str, Any]:
    mode = os.getenv("WEBUI_V2_MODE", "local").strip().lower()
    if mode != "online":
        return {
            "mode": mode,
            "enforced": False,
            "limit": None,
            "used": 0,
            "remaining": None,
        }

    owner_key = _session_owner_key_for_request(request)
    today = datetime.now().strftime("%Y-%m-%d")
    async with _IP_USAGE_GUARD:
        usage = _IP_DAILY_CHAT_USAGE.get(owner_key)
        if not usage or usage.get("date") != today:
            usage = {"date": today, "count": 0}
            _IP_DAILY_CHAT_USAGE[owner_key] = usage
        used = int(usage.get("count", 0))
    return {
        "mode": mode,
        "enforced": True,
        "limit": _ONLINE_DAILY_CHAT_LIMIT,
        "used": used,
        "remaining": max(0, _ONLINE_DAILY_CHAT_LIMIT - used),
    }


async def _normalize_uploaded_file(
    src_path: str,
    agent_session_dir: str,
    temp_files: List[str],
    owner_key: str,
) -> Optional[str]:
    if not src_path:
        return None
    src_file = Path(src_path)
    if not src_file.is_file():
        try:
            src_file = resolve_web_v2_client_path(src_path, allowed_areas=("uploads", "sessions"))
        except ValueError:
            return None
    if not src_file.is_file():
        return None
    if _runtime_mode() == "online":
        try:
            rel_path = to_web_v2_public_path(src_file)
        except Exception:
            rel_path = ""
        if rel_path.startswith("sessions/"):
            parts = [p for p in rel_path.split("/") if p]
            source_session_id = parts[1] if len(parts) > 1 else ""
            if source_session_id:
                async with _SESSIONS_GUARD:
                    source_session = _SESSIONS.get(source_session_id)
                if not source_session or str(source_session.get("owner_key", "")) != owner_key:
                    return None
    os.makedirs(agent_session_dir, exist_ok=True)
    existing = len([p for p in Path(agent_session_dir).glob("u_*__*") if p.is_file()])
    dst_name = make_web_v2_upload_name(existing + 1, src_file.name)
    dst = os.path.join(agent_session_dir, dst_name)
    shutil.copy2(str(src_file), dst)
    normalized = dst.replace("\\", "/")
    temp_files.append(normalized)
    return normalized


async def _stream_graph(
    state: Dict[str, Any],
    text: str,
    attachment_paths: List[str],
):
    if await _is_cancelled(state["session_id"]):
        state["status"] = "stopped"
        yield f"event: state\ndata: {_to_json(_snapshot(state))}\n\n"
        yield "event: done\ndata: {}\n\n"
        return

    agent_session_dir = state.get("agent_session_dir")
    if not agent_session_dir:
        raise RuntimeError("Agent session directory is missing.")
    os.makedirs(agent_session_dir, exist_ok=True)

    valid_attachments: List[str] = []
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

    initial_state = {
        "messages": [HumanMessage(content=display_text)],
        "protein_context": protein_ctx,
        "session_id": state["session_id"],
        "agent_session_dir": agent_session_dir,
        "history": list(state["history"]),
        "conversation_log": list(state.get("conversation_log", [])),
        "tool_executions": list(state.get("tool_executions", [])),
        "tool_cache": dict(state.get("tool_cache", {})),
        "status": "started",
        "pi_report": "",
        "pi_suggest_steps": "",
        "plan": [],
        "current_step_index": 0,
        "step_results": {},
        "error": None,
        "research_sections": [],
        "research_idx": 0,
        "search_idx": 0,
        "current_search_results": [],
        "research_sub_reports": [],
        "execution_failed": bool(state.get("execution_failed", False)),
        "failed_step": state.get("failed_step"),
        "failed_reason": state.get("failed_reason"),
    }
    graph = create_agent_graph()
    config = {
        "configurable": {"chains": state, "session_id": state["session_id"]},
        "recursion_limit": 100,
    }

    yield f"event: state\ndata: {_to_json(_snapshot(state))}\n\n"

    async for event in graph.astream(initial_state, config=config):
        if await _is_cancelled(state["session_id"]):
            state["status"] = "stopped"
            state.setdefault("history", []).append(
                {
                    "role": "assistant",
                    "content": "用户已停止本次运行。" if is_zh else "Run stopped by user.",
                    "role_id": "principal_investigator",
                }
            )
            yield f"event: state\ndata: {_to_json(_snapshot(state))}\n\n"
            yield "event: done\ndata: {}\n\n"
            return
        for _, updates in event.items():
            if updates:
                for key, val in updates.items():
                    if key in (
                        "history",
                        "conversation_log",
                        "tool_executions",
                        "status",
                        "pi_report",
                        "pi_suggest_steps",
                        "plan",
                        "protein_context",
                        "current_step_index",
                        "step_results",
                        "research_sections",
                        "research_idx",
                        "search_idx",
                        "current_search_results",
                        "research_sub_reports",
                        "tool_cache",
                        "execution_failed",
                        "failed_step",
                        "failed_reason",
                    ):
                        state[key] = val
        yield f"event: state\ndata: {_to_json(_snapshot(state))}\n\n"

    final_content = state["history"][-1]["content"] if state.get("history") else ""
    try:
        state["memory"].save_context({"input": display_text}, {"output": final_content})
    except Exception:
        pass
    state["status"] = "completed"
    yield f"event: state\ndata: {_to_json(_snapshot(state))}\n\n"
    yield "event: done\ndata: {}\n\n"


@router.post("/sessions", response_model=CreateSessionResponse)
async def create_session(request: Request):
    state = initialize_session_state()
    token, token_expires_at = _issue_session_access_token(state, request)
    session_id = state["session_id"]
    async with _SESSIONS_GUARD:
        _SESSIONS[session_id] = state
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
    visible = await get_visible_session_ids(request)
    async with _SESSIONS_GUARD:
        data = [
            {
                "session_id": sid,
                "created_at": str(s.get("created_at", "")),
                "model_name": getattr(s.get("llm"), "model_name", ""),
                "history_size": len(s.get("history") or []),
                "status": s.get("status", ""),
            }
            for sid, s in _SESSIONS.items()
            if sid in visible
        ]
    return {"sessions": data}


@router.get("/sessions/{session_id}", response_model=SessionStateResponse)
async def get_session(session_id: str, request: Request):
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)
    snap = _snapshot(state)
    return SessionStateResponse(**snap)


@router.post("/sessions/{session_id}/attachments")
async def upload_attachments(
    session_id: str,
    request: Request,
    files: List[UploadFile] = File(default_factory=list),
):
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)
    target_dir = state.get("agent_session_dir")
    os.makedirs(target_dir, exist_ok=True)
    stored = []
    for f in files:
        filename = os.path.basename(f.filename or f"upload-{uuid.uuid4().hex}")
        existing = len([p for p in Path(target_dir).glob("u_*__*") if p.is_file()])
        dst_name = make_web_v2_upload_name(existing + 1, filename)
        dst = os.path.join(target_dir, dst_name)
        content = await f.read()
        with open(dst, "wb") as out:
            out.write(content)
        normalized = dst.replace("\\", "/")
        state.setdefault("temp_files", []).append(normalized)
        stored.append(
            {
                "name": filename,
                "stored_name": dst_name,
                "path": to_web_v2_public_path(normalized),
                "size": len(content),
            }
        )
    return {"files": stored}


@router.post("/sessions/{session_id}/messages/stream")
async def stream_message(session_id: str, payload: ChatStreamRequest, request: Request):
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)
    await _consume_online_chat_quota_or_429(request)
    await _set_cancel(session_id, False)
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
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)
    await _consume_online_chat_quota_or_429(request)
    await _set_cancel(session_id, False)
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


@router.get("/quota")
async def get_chat_quota(request: Request):
    return await _get_online_chat_quota_status(request)


@router.post("/sessions/{session_id}/cancel")
async def cancel_session_run(session_id: str, request: Request):
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)
    state["status"] = "stopping"
    await _set_cancel(session_id, True)
    return {"success": True, "status": "stopping"}
