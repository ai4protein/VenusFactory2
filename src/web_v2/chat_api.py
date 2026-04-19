import asyncio
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, List, Optional

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from agent.chat_agent import initialize_session_state, update_llm_model
from agent.chat_agent_utils import (
    AGENT_CHAT_MAX_MESSAGES,
    extract_sequence_from_message,
    extract_uniprot_id_from_message,
)
from agent.chat_graph import _format_clarification_answers, create_agent_graph
from config import get_config
from logger import get_logger
from web.utils.common_utils import (
    make_web_v2_upload_name,
    redact_path_text,
    resolve_web_v2_client_path,
    to_web_v2_public_path,
)
from web_v2.analytics_store import analytics_store
from web_v2.feedback_webhook import dispatch_webhook

_logger = get_logger("web_v2.chat_api")
_cfg = get_config()

router = APIRouter(prefix="/api/chat", tags=["chat-v2"])

_SESSIONS: dict[str, dict[str, Any]] = {}
_SESSION_LOCKS: dict[str, asyncio.Lock] = {}
_SESSION_CANCEL_FLAGS: dict[str, bool] = {}
_SESSIONS_GUARD = asyncio.Lock()
_OWNER_DAILY_CHAT_USAGE: dict[str, dict[str, Any]] = {}
_OWNER_USAGE_GUARD = asyncio.Lock()

_ONLINE_DAILY_CHAT_LIMIT = _cfg.online_limits.daily_chat_limit
_SESSION_TOKEN_TTL_HOURS = _cfg.online_limits.session_token_ttl_hours
_SESSION_TOKEN_SECRET = os.getenv("WEBUI_V2_SESSION_TOKEN_SECRET", "").strip() or secrets.token_hex(32)


def _runtime_mode() -> str:
    return _cfg.server.mode


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


def _issue_session_access_token(state: dict[str, Any], request: Request) -> tuple[str, str]:
    raw_token = secrets.token_urlsafe(48)
    expires_at = datetime.now(UTC) + timedelta(hours=_SESSION_TOKEN_TTL_HOURS)
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
    history: list[dict[str, Any]]
    conversation_log: list[dict[str, Any]]
    tool_executions: list[dict[str, Any]]
    status: str = ""
    clarification_questions: list[dict[str, Any]] = Field(default_factory=list)
    plan: list[dict[str, Any]] = Field(default_factory=list)
    waiting_for: str = ""


class ChatStreamRequest(BaseModel):
    text: str = Field(default="")
    model: Optional[str] = Field(default=None)
    attachment_paths: list[str] = Field(default_factory=list)


class ClarificationAnswer(BaseModel):
    question_index: int = 0
    selected_options: list[int] = Field(default_factory=list)
    custom_text: str = ""


class ClarificationResponseRequest(BaseModel):
    answers: list[ClarificationAnswer] = Field(default_factory=list)


class PlanConfirmRequest(BaseModel):
    plan: list[dict[str, Any]] = Field(default_factory=list)
    auto_execute: bool = Field(default=False)


class IterationDecideRequest(BaseModel):
    action: str = Field(default="satisfied")


class StepDecideRequest(BaseModel):
    action: str = Field(default="continue")


class SubReportDecideRequest(BaseModel):
    action: str = Field(default="continue")
    comment: str = Field(default="")


class FeedbackRequest(BaseModel):
    message_index: int = Field(ge=0)
    rating: str = Field(pattern="^(like|dislike)$")
    comment: str = Field(default="")


def _to_json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def _redact_obj(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _redact_obj(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_obj(v) for v in value]
    if isinstance(value, str):
        return redact_path_text(value)
    return value


def _snapshot(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_id": state.get("session_id"),
        "model_name": getattr(state.get("llm"), "model_name", ""),
        "created_at": str(state.get("created_at", "")),
        "history": _redact_obj(list(state.get("history", []))),
        "conversation_log": _redact_obj(list(state.get("conversation_log", []))),
        "tool_executions": _redact_obj(list(state.get("tool_executions", []))),
        "status": state.get("status", ""),
        "clarification_questions": list(state.get("clarification_questions", [])),
        "plan": list(state.get("plan", [])),
        "waiting_for": state.get("waiting_for", ""),
    }


def _append_dialogue_memory(state: dict[str, Any], user_input: str, final_output: str) -> None:
    user = (user_input or "").strip()
    assistant = (final_output or "").strip()
    if not user or not assistant:
        return
    memory = state.setdefault("dialogue_memory", [])
    memory.append(
        {
            "user": user,
            "assistant": assistant,
            "timestamp": datetime.now().isoformat(),
        }
    )
    if len(memory) > 10:
        del memory[:-10]


async def _get_session_or_404(session_id: str) -> dict[str, Any]:
    async with _SESSIONS_GUARD:
        state = _SESSIONS.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    return state


def _assert_session_access(state: dict[str, Any], request: Request) -> None:
    if _runtime_mode() == "online":
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
                expires_at = expires_at.replace(tzinfo=UTC)
        except ValueError:
            expires_at = datetime.now(UTC) - timedelta(seconds=1)
        if datetime.now(UTC) >= expires_at:
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
    if not _cfg.server.is_online:
        return

    owner_key = _session_owner_key_for_request(request)
    today = datetime.now().strftime("%Y-%m-%d")
    async with _OWNER_USAGE_GUARD:
        usage = _OWNER_DAILY_CHAT_USAGE.get(owner_key)
        if not usage or usage.get("date") != today:
            usage = {"date": today, "count": 0}
            _OWNER_DAILY_CHAT_USAGE[owner_key] = usage

        if int(usage.get("count", 0)) >= _ONLINE_DAILY_CHAT_LIMIT:
            raise HTTPException(
                status_code=429,
                detail={
                    "code": "CHAT_DAILY_LIMIT_REACHED",
                    "message": f"Online mode limit reached: up to {_ONLINE_DAILY_CHAT_LIMIT} chats per user per day.",
                },
            )
        usage["count"] = int(usage.get("count", 0)) + 1


async def _get_online_chat_quota_status(request: Request) -> dict[str, Any]:
    mode = _cfg.server.mode
    if not _cfg.server.is_online:
        return {
            "mode": mode,
            "enforced": False,
            "limit": None,
            "used": 0,
            "remaining": None,
        }

    owner_key = _session_owner_key_for_request(request)
    today = datetime.now().strftime("%Y-%m-%d")
    async with _OWNER_USAGE_GUARD:
        usage = _OWNER_DAILY_CHAT_USAGE.get(owner_key)
        if not usage or usage.get("date") != today:
            usage = {"date": today, "count": 0}
            _OWNER_DAILY_CHAT_USAGE[owner_key] = usage
        used = int(usage.get("count", 0))
    return {
        "mode": mode,
        "enforced": True,
        "limit": _ONLINE_DAILY_CHAT_LIMIT,
        "used": used,
        "remaining": max(0, _ONLINE_DAILY_CHAT_LIMIT - used),
    }


def _record_access_event(request: Request, endpoint: str) -> None:
    try:
        analytics_store.record_access_event(
            ts=datetime.now(UTC).isoformat(),
            endpoint=endpoint,
            owner_key=_session_owner_key_for_request(request),
            ip=_extract_client_ip(request),
            user_agent=_extract_user_agent(request),
        )
    except Exception:
        pass


async def _archive_conversation(state: dict[str, Any]) -> None:
    if not _cfg.feedback.collect_conversations:
        return
    try:
        history = state.get("history", [])
        if not history:
            return
        session_id = state.get("session_id", "")
        model_name = getattr(state.get("llm"), "model_name", "")
        messages_json = json.dumps(
            _redact_obj(list(history)), ensure_ascii=False, default=str
        )
        analytics_store.record_conversation(
            ts=datetime.now(UTC).isoformat(),
            session_id=session_id,
            model_name=model_name,
            messages=messages_json,
            message_count=len(history),
            owner_key=str(state.get("owner_key", "")),
            ip=str(state.get("client_ip", "")),
        )
        asyncio.create_task(dispatch_webhook("conversation_archived", {
            "session_id": session_id,
            "model_name": model_name,
            "message_count": len(history),
            "messages": _redact_obj(list(history)),
        }))
    except Exception:
        _logger.debug("Failed to archive conversation", exc_info=True)


async def _normalize_uploaded_file(
    src_path: str,
    agent_session_dir: str,
    temp_files: list[str],
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
    state: dict[str, Any],
    text: str,
    attachment_paths: list[str],
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
    }
    graph = create_agent_graph()
    config = {
        "configurable": {"chains": state, "session_id": state["session_id"]},
        "recursion_limit": 100,
    }

    yield f"event: state\ndata: {_to_json(_snapshot(state))}\n\n"

    _STREAM_STATE_KEYS = {
        "history", "conversation_log", "tool_executions", "status",
        "pi_report", "pi_suggest_steps", "plan", "protein_context",
        "current_step_index", "step_results", "research_sections",
        "research_idx", "search_idx", "current_search_results",
        "research_sub_reports", "sub_report_rewrite_comment",
        "auto_execute", "tool_cache", "execution_failed",
        "failed_step", "failed_reason",
        "clarification_questions", "clarification_answers", "waiting_for",
    }

    async for stream_mode, data in graph.astream(
        initial_state, config=config, stream_mode=["updates", "custom"]
    ):
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

        if stream_mode == "custom":
            event_type = data.get("type", "token") if isinstance(data, dict) else "token"
            yield f"event: {event_type}\ndata: {_to_json(data)}\n\n"
        elif stream_mode == "updates":
            for _, updates in data.items():
                if updates:
                    for key, val in updates.items():
                        if key in _STREAM_STATE_KEYS:
                            state[key] = val
            yield f"event: state\ndata: {_to_json(_snapshot(state))}\n\n"

    final_status = state.get("status", "")
    _WAITING_STATUSES = ("waiting_for_clarification", "waiting_for_plan_confirmation", "waiting_for_iteration", "waiting_for_step_review", "waiting_for_sub_report_review")
    if final_status not in _WAITING_STATUSES:
        final_content = state["history"][-1]["content"] if state.get("history") else ""
        _append_dialogue_memory(state, display_text, final_content)
        if final_content:
            state.setdefault("conversation_log", []).append(
                {"role": "assistant", "content": final_content, "timestamp": datetime.now().isoformat()}
            )
        try:
            state["memory"].save_context({"input": display_text}, {"output": final_content})
        except Exception:
            pass
        state["status"] = "completed"
        await _archive_conversation(state)
    yield f"event: state\ndata: {_to_json(_snapshot(state))}\n\n"
    yield "event: done\ndata: {}\n\n"


@router.post("/sessions", response_model=CreateSessionResponse)
async def create_session(request: Request):
    _record_access_event(request, "/api/chat/sessions:create")
    state = initialize_session_state()
    token, token_expires_at = _issue_session_access_token(state, request)
    state["client_ip"] = _extract_client_ip(request)
    state["owner_key"] = _session_owner_key_for_request(request)
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
    _record_access_event(request, "/api/chat/sessions:list")
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
        async with _SESSIONS_GUARD:
            current = _SESSIONS.get(session_id)
            if current is None:
                raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
            _assert_session_access(current, request)
            _SESSIONS.pop(session_id, None)
            _SESSION_LOCKS.pop(session_id, None)
            _SESSION_CANCEL_FLAGS.pop(session_id, None)

        session_dir = state.get("agent_session_dir")
        if session_dir:
            try:
                shutil.rmtree(session_dir, ignore_errors=True)
            except Exception:
                pass
    return {"success": True, "session_id": session_id}


@router.post("/sessions/{session_id}/attachments")
async def upload_attachments(
    session_id: str,
    request: Request,
    files: List[UploadFile] = File(default_factory=list),
):
    _record_access_event(request, "/api/chat/sessions/{id}/attachments")
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
    _record_access_event(request, "/api/chat/sessions/{id}/messages/stream")
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)
    await _consume_online_chat_quota_or_429(request)
    await _set_cancel(session_id, False)
    state["client_ip"] = _extract_client_ip(request)
    state["owner_key"] = _session_owner_key_for_request(request)
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


@router.get("/quota")
async def get_chat_quota(request: Request):
    _record_access_event(request, "/api/chat/quota")
    return await _get_online_chat_quota_status(request)


@router.post("/sessions/{session_id}/cancel")
async def cancel_session_run(session_id: str, request: Request):
    _record_access_event(request, "/api/chat/sessions/{id}/cancel")
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)
    state["status"] = "stopping"
    await _set_cancel(session_id, True)
    return {"success": True, "status": "stopping"}


async def _stream_graph_resume(
    state: dict[str, Any],
    waiting_for: str,
    extra_state: dict[str, Any] | None = None,
):
    """Resume a graph run from a checkpoint (clarification answered or plan confirmed)."""
    if await _is_cancelled(state["session_id"]):
        state["status"] = "stopped"
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

    graph = create_agent_graph()
    config = {
        "configurable": {"chains": state, "session_id": state["session_id"]},
        "recursion_limit": 100,
    }

    state["status"] = "started"
    yield f"event: state\ndata: {_to_json(_snapshot(state))}\n\n"

    _STREAM_STATE_KEYS = {
        "history", "conversation_log", "tool_executions", "status",
        "pi_report", "pi_suggest_steps", "plan", "protein_context",
        "current_step_index", "step_results", "research_sections",
        "research_idx", "search_idx", "current_search_results",
        "research_sub_reports", "sub_report_rewrite_comment",
        "auto_execute", "tool_cache", "execution_failed",
        "failed_step", "failed_reason",
        "clarification_questions", "clarification_answers", "waiting_for",
    }

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
            yield f"event: state\ndata: {_to_json(_snapshot(state))}\n\n"
            yield "event: done\ndata: {}\n\n"
            return

        if stream_mode == "custom":
            event_type = data.get("type", "token") if isinstance(data, dict) else "token"
            yield f"event: {event_type}\ndata: {_to_json(data)}\n\n"
        elif stream_mode == "updates":
            for _, updates in data.items():
                if updates:
                    for key, val in updates.items():
                        if key in _STREAM_STATE_KEYS:
                            state[key] = val
            yield f"event: state\ndata: {_to_json(_snapshot(state))}\n\n"

    final_status = state.get("status", "")
    _WAITING_STATUSES = ("waiting_for_clarification", "waiting_for_plan_confirmation", "waiting_for_iteration", "waiting_for_step_review", "waiting_for_sub_report_review")
    if final_status not in _WAITING_STATUSES:
        final_content = state["history"][-1]["content"] if state.get("history") else ""
        _append_dialogue_memory(state, original_text, final_content)
        if final_content:
            state.setdefault("conversation_log", []).append({
                "role": "assistant", "content": final_content,
                "timestamp": datetime.now().isoformat(),
            })
        try:
            state["memory"].save_context({"input": original_text}, {"output": final_content})
        except Exception:
            pass
        state["status"] = "completed"
        await _archive_conversation(state)

    yield f"event: state\ndata: {_to_json(_snapshot(state))}\n\n"
    yield "event: done\ndata: {}\n\n"


@router.post("/sessions/{session_id}/clarification/respond/stream")
async def stream_clarification_response(
    session_id: str,
    payload: ClarificationResponseRequest,
    request: Request,
):
    _record_access_event(request, "/api/chat/sessions/{id}/clarification/respond/stream")
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)
    await _set_cancel(session_id, False)

    if state.get("waiting_for") != "clarification" and state.get("status") != "waiting_for_clarification":
        raise HTTPException(status_code=400, detail="Session is not waiting for clarification.")

    questions = state.get("clarification_questions", [])
    answers_data = [a.model_dump() for a in payload.answers]
    state["clarification_answers"] = answers_data

    answers_summary = _format_clarification_answers(questions, answers_data)
    is_zh = _is_zh_text(state.get("last_user_text", ""))
    state["history"].append({
        "role": "user",
        "content": ("📝 **需求补充说明：**\n\n" if is_zh else "📝 **Clarification Details:**\n\n") + answers_summary,
    })

    original_text = state.get("last_user_text", "")
    enriched_text = f"{original_text}\n\n[Clarification Details]\n{answers_summary}"
    state["last_user_text"] = enriched_text

    lock = await _get_lock(session_id)

    async def event_gen():
        async with lock:
            async for chunk in _stream_graph_resume(
                state,
                waiting_for="clarification_answered",
            ):
                yield chunk

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@router.post("/sessions/{session_id}/plan/confirm/stream")
async def stream_plan_confirmation(
    session_id: str,
    payload: PlanConfirmRequest,
    request: Request,
):
    _record_access_event(request, "/api/chat/sessions/{id}/plan/confirm/stream")
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)
    await _set_cancel(session_id, False)

    if state.get("waiting_for") != "plan_confirmation" and state.get("status") != "waiting_for_plan_confirmation":
        raise HTTPException(status_code=400, detail="Session is not waiting for plan confirmation.")

    confirmed_plan = payload.plan
    auto_execute = payload.auto_execute
    state["plan"] = confirmed_plan
    state["auto_execute"] = auto_execute

    is_zh = _is_zh_text(state.get("last_user_text", ""))
    if auto_execute:
        state["history"].append({
            "role": "user",
            "content": "✅ 已确认执行计划，自动执行所有步骤。" if is_zh else "✅ Plan confirmed. Auto-executing all steps.",
        })
    else:
        state["history"].append({
            "role": "user",
            "content": "✅ 已确认执行计划。" if is_zh else "✅ Plan confirmed.",
        })

    lock = await _get_lock(session_id)

    async def event_gen():
        async with lock:
            async for chunk in _stream_graph_resume(
                state,
                waiting_for="plan_confirmed",
                extra_state={
                    "plan": confirmed_plan,
                    "current_step_index": 0,
                    "step_results": {},
                    "auto_execute": auto_execute,
                },
            ):
                yield chunk

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@router.post("/sessions/{session_id}/iteration/decide")
async def iteration_decide(
    session_id: str,
    payload: IterationDecideRequest,
    request: Request,
):
    _record_access_event(request, "/api/chat/sessions/{id}/iteration/decide")
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)

    if state.get("waiting_for") != "iteration" and state.get("status") != "waiting_for_iteration":
        raise HTTPException(status_code=400, detail="Session is not waiting for iteration decision.")

    action = payload.action
    is_zh = _is_zh_text(state.get("last_user_text", ""))

    if action == "modify_plan":
        state["status"] = "waiting_for_plan_confirmation"
        state["waiting_for"] = "plan_confirmation"
        state["history"].append({
            "role": "user",
            "content": "🔄 希望修改计划并重新执行。" if is_zh else "🔄 I'd like to modify the plan and re-execute.",
        })
        return {
            "success": True,
            "status": "waiting_for_plan_confirmation",
            "plan": list(state.get("plan", [])),
        }

    if action == "continue":
        user_msg = "➕ 继续分析，我有新的指令。" if is_zh else "➕ Continue analysis with new instructions."
        state["has_prior_research"] = True
    else:
        user_msg = "✅ 对结果满意，任务完成。" if is_zh else "✅ Satisfied with the results. Task complete."
        state["has_prior_research"] = False

    state["status"] = "completed"
    state["waiting_for"] = None
    state["history"].append({"role": "user", "content": user_msg})

    original_text = state.get("last_user_text", "")
    final_content = ""
    skip_markers = ("iteration_prompt", "请选择下一步", "Please choose")
    for item in reversed(state.get("history", [])):
        if item.get("role") == "assistant" and item.get("role_id") == "principal_investigator":
            content = item.get("content", "")
            if not any(m in content for m in skip_markers) and len(content) > 10:
                final_content = content
                break
    _append_dialogue_memory(state, original_text, final_content)
    try:
        state["memory"].save_context({"input": original_text}, {"output": final_content})
    except Exception:
        pass
    await _archive_conversation(state)
    return {"success": True, "status": "completed"}


@router.post("/sessions/{session_id}/step/decide/stream")
async def stream_step_decide(
    session_id: str,
    payload: StepDecideRequest,
    request: Request,
):
    _record_access_event(request, "/api/chat/sessions/{id}/step/decide/stream")
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)
    await _set_cancel(session_id, False)

    if state.get("waiting_for") != "step_review" and state.get("status") != "waiting_for_step_review":
        raise HTTPException(status_code=400, detail="Session is not waiting for step review.")

    action = payload.action
    is_zh = _is_zh_text(state.get("last_user_text", ""))

    if action == "abort":
        state["history"].append({
            "role": "user",
            "content": "⏹️ 跳过剩余步骤，直接汇总。" if is_zh else "⏹️ Skip remaining steps and go to summary.",
        })
        waiting_for = "step_abort"
    else:
        state["history"].append({
            "role": "user",
            "content": "▶️ 继续执行下一步。" if is_zh else "▶️ Continue to the next step.",
        })
        waiting_for = "step_continue"

    lock = await _get_lock(session_id)

    async def event_gen():
        async with lock:
            async for chunk in _stream_graph_resume(
                state,
                waiting_for=waiting_for,
            ):
                yield chunk

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@router.post("/sessions/{session_id}/sub-report/decide/stream")
async def stream_sub_report_decide(
    session_id: str,
    payload: SubReportDecideRequest,
    request: Request,
):
    _record_access_event(request, "/api/chat/sessions/{id}/sub-report/decide/stream")
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)
    await _set_cancel(session_id, False)

    if state.get("waiting_for") != "sub_report_review" and state.get("status") != "waiting_for_sub_report_review":
        raise HTTPException(status_code=400, detail="Session is not waiting for sub-report review.")

    action = payload.action
    is_zh = _is_zh_text(state.get("last_user_text", ""))

    if action == "rewrite":
        comment_text = (payload.comment or "").strip()
        if not comment_text:
            raise HTTPException(status_code=400, detail="Comment is required for rewrite action.")
        current_idx = state.get("research_idx", 1)
        state["research_idx"] = max(0, current_idx - 1)
        sub_reports = list(state.get("research_sub_reports", []))
        if sub_reports:
            sub_reports.pop()
        state["research_sub_reports"] = sub_reports
        state["sub_report_rewrite_comment"] = comment_text
        state["history"].append({
            "role": "user",
            "content": f"✏️ 修改意见：{comment_text}" if is_zh else f"✏️ Revision feedback: {comment_text}",
        })
        waiting_for = "sub_report_rewrite"
    elif action == "skip":
        state["history"].append({
            "role": "user",
            "content": "⏭️ 跳过剩余小节，直接生成报告。" if is_zh else "⏭️ Skip remaining sections and generate report.",
        })
        waiting_for = "sub_report_skip"
    else:
        state["history"].append({
            "role": "user",
            "content": "▶️ 继续调研下一个小节。" if is_zh else "▶️ Continue to the next section.",
        })
        waiting_for = "sub_report_continue"

    extra = {}
    if action == "rewrite":
        extra = {
            "sub_report_rewrite_comment": comment_text,
            "research_idx": state["research_idx"],
            "research_sub_reports": list(state.get("research_sub_reports", [])),
        }

    lock = await _get_lock(session_id)

    async def event_gen():
        async with lock:
            async for chunk in _stream_graph_resume(
                state,
                waiting_for=waiting_for,
                extra_state=extra if extra else None,
            ):
                yield chunk

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@router.post("/sessions/{session_id}/feedback")
async def submit_feedback(
    session_id: str,
    payload: FeedbackRequest,
    request: Request,
):
    _record_access_event(request, "/api/chat/sessions/{id}/feedback")
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)

    history = state.get("history", [])
    if payload.message_index >= len(history):
        raise HTTPException(status_code=400, detail="Invalid message index.")
    msg = history[payload.message_index]
    if msg.get("role") == "user":
        raise HTTPException(status_code=400, detail="Cannot rate user messages.")

    model_name = getattr(state.get("llm"), "model_name", "")
    ip = _extract_client_ip(request)
    owner_key = _session_owner_key_for_request(request)

    analytics_store.record_feedback(
        ts=datetime.now(UTC).isoformat(),
        session_id=session_id,
        message_index=payload.message_index,
        rating=payload.rating,
        comment=payload.comment,
        owner_key=owner_key,
        ip=ip,
        model_name=model_name,
    )

    webhook_data = {
        "session_id": session_id,
        "message_index": payload.message_index,
        "rating": payload.rating,
        "comment": payload.comment,
        "message_content": msg.get("content", "")[:500],
        "model_name": model_name,
        "owner_key": owner_key,
    }
    asyncio.create_task(dispatch_webhook("feedback_submitted", webhook_data))

    return {"success": True, "session_id": session_id, "message_index": payload.message_index}


def _generate_experiment_report(state: dict[str, Any]) -> str:
    """Build a comprehensive long-form Markdown research report from session state."""
    is_zh = _is_zh_text(state.get("last_user_text", ""))
    model_name = getattr(state.get("llm"), "model_name", "unknown")
    session_id = state.get("session_id", "unknown")
    created_at = str(state.get("created_at", ""))
    generated_at = datetime.now().isoformat()
    history = state.get("history", [])

    lines: list[str] = []

    # ── Title & Metadata ──
    lines.append("# " + ("实验研究报告" if is_zh else "Experiment Research Report"))
    lines.append("")
    lines.append(f"| {'字段' if is_zh else 'Field'} | {'值' if is_zh else 'Value'} |")
    lines.append("|---|---|")
    lines.append(f"| Session ID | `{session_id}` |")
    lines.append(f"| {'模型' if is_zh else 'Model'} | {model_name} |")
    lines.append(f"| {'创建时间' if is_zh else 'Created'} | {created_at} |")
    lines.append(f"| {'报告生成时间' if is_zh else 'Report Generated'} | {generated_at} |")
    lines.append(f"| {'状态' if is_zh else 'Status'} | {state.get('status', '')} |")
    lines.append("")

    # ── Table of Contents ──
    lines.append("## " + ("目录" if is_zh else "Table of Contents"))
    lines.append("")
    toc_items = [
        ("蛋白质上下文", "Protein Context"),
        ("研究背景与用户需求", "Background & User Request"),
        ("文献调研", "Literature Research"),
        ("调研报告草稿", "Research Report Draft"),
        ("实验设计", "Experimental Design"),
        ("实验过程与结果", "Experimental Process & Results"),
        ("图表与可视化", "Figures & Visualizations"),
        ("讨论与结论", "Discussion & Conclusion"),
    ]
    for i, (zh, en) in enumerate(toc_items, 1):
        lines.append(f"{i}. [{zh if is_zh else en}](#{i})")
    lines.append("")

    # ── 1. Protein Context ──
    lines.append("## 1. " + ("蛋白质上下文" if is_zh else "Protein Context"))
    lines.append("")
    protein_ctx = state.get("protein_context")
    if protein_ctx:
        ctx_summary = protein_ctx.get_context_summary()
        if ctx_summary and ctx_summary != "No protein data in memory":
            lines.append(redact_path_text(ctx_summary))
        else:
            lines.append("_" + ("本次实验未涉及特定蛋白质。" if is_zh else "No specific protein context in this experiment.") + "_")
    else:
        lines.append("_" + ("本次实验未涉及特定蛋白质。" if is_zh else "No specific protein context in this experiment.") + "_")
    lines.append("")

    # ── 2. Background & User Request ──
    lines.append("## 2. " + ("研究背景与用户需求" if is_zh else "Background & User Request"))
    lines.append("")
    original_text = state.get("last_user_text", "")
    if original_text:
        lines.append(redact_path_text(original_text))
        lines.append("")

    questions = state.get("clarification_questions", [])
    answers = state.get("clarification_answers", [])
    if questions and answers:
        lines.append("### " + ("需求澄清" if is_zh else "Clarification Q&A"))
        lines.append("")
        formatted = _format_clarification_answers(questions, answers)
        if formatted:
            lines.append(redact_path_text(formatted))
            lines.append("")

    # ── 3. Literature Research ──
    sections = state.get("research_sections", [])
    sub_reports = state.get("research_sub_reports", [])
    if sections or sub_reports:
        lines.append("## 3. " + ("文献调研" if is_zh else "Literature Research"))
        lines.append("")

        if sections:
            lines.append("### " + ("调研计划" if is_zh else "Research Plan"))
            lines.append("")
            for i, sec in enumerate(sections, 1):
                name = sec.get("section_name", f"Section {i}")
                focus = sec.get("focus", "")
                queries = sec.get("search_queries", [])
                lines.append(f"**{i}. {name}**")
                if focus:
                    lines.append(f"  - {'研究重点' if is_zh else 'Focus'}: {focus}")
                if queries:
                    lines.append(f"  - {'检索词' if is_zh else 'Search queries'}:")
                    for q in queries:
                        lines.append(f"    - {q}")
                lines.append("")

        if sub_reports:
            lines.append("### " + ("各节调研结果" if is_zh else "Section-by-Section Research Results"))
            lines.append("")
            for sr in sub_reports:
                lines.append(redact_path_text(sr))
                lines.append("")
                lines.append("---")
                lines.append("")

    # ── 4. Research Report Draft (PI final report) ──
    pi_report = state.get("pi_report", "")
    if pi_report:
        lines.append("## 4. " + ("调研报告草稿" if is_zh else "Research Report Draft"))
        lines.append("")
        lines.append(redact_path_text(pi_report))
        lines.append("")

    # ── 5. Experimental Design ──
    plan = state.get("plan", [])
    if plan:
        lines.append("## 5. " + ("实验设计" if is_zh else "Experimental Design"))
        lines.append("")
        pi_suggest = state.get("pi_suggest_steps", "")
        if pi_suggest:
            lines.append("### " + ("PI 建议方案" if is_zh else "PI Suggested Approach"))
            lines.append("")
            lines.append(redact_path_text(pi_suggest))
            lines.append("")

        lines.append("### " + ("最终执行计划" if is_zh else "Final Execution Plan"))
        lines.append("")
        lines.append(f"| {'步骤' if is_zh else 'Step'} | {'工具' if is_zh else 'Tool'} | {'描述' if is_zh else 'Description'} |")
        lines.append("|---|---|---|")
        for step in plan:
            step_num = step.get("step", "?")
            tool = step.get("tool_name", "?")
            desc = step.get("task_description", "").replace("\n", " ").replace("|", "\\|")
            lines.append(f"| {step_num} | `{tool}` | {desc} |")
        lines.append("")

    # ── 6. Experimental Process & Results ──
    tool_executions = state.get("tool_executions", [])
    if tool_executions:
        lines.append("## 6. " + ("实验过程与结果" if is_zh else "Experimental Process & Results"))
        lines.append("")

        for entry in tool_executions:
            step = entry.get("step", "?")
            tool = entry.get("tool_name", "unknown")
            ts = entry.get("timestamp", "")
            inputs = entry.get("inputs", {})
            outputs = entry.get("outputs", "")
            oss_url = entry.get("oss_url")

            step_plan_desc = ""
            for p in plan:
                if str(p.get("step", "")) == str(step):
                    step_plan_desc = p.get("task_description", "")
                    break

            lines.append(f"### Step {step}: `{tool}`")
            lines.append("")
            if step_plan_desc:
                lines.append(f"**{'任务目标' if is_zh else 'Objective'}:** {step_plan_desc}")
                lines.append("")
            if ts:
                lines.append(f"**{'执行时间' if is_zh else 'Timestamp'}:** {ts}")
                lines.append("")

            if inputs:
                lines.append(f"**{'输入参数' if is_zh else 'Input Parameters'}:**")
                lines.append("")
                lines.append("```json")
                lines.append(redact_path_text(json.dumps(inputs, ensure_ascii=False, indent=2)))
                lines.append("```")
                lines.append("")

            if outputs:
                output_str = redact_path_text(str(outputs))
                lines.append(f"**{'输出结果' if is_zh else 'Output'}:**")
                lines.append("")
                lines.append("```")
                lines.append(output_str[:4000] + ("..." if len(output_str) > 4000 else ""))
                lines.append("```")
                lines.append("")

            if oss_url:
                name = os.path.basename(oss_url)
                lines.append(f"**{'云端下载' if is_zh else 'Cloud Download'}:** [{name}]({oss_url})")
                lines.append("")

            # Extract per-step MLS feedback from history (cloud links, file previews, images)
            mls_feedback = _extract_step_feedback_from_history(history, step, tool)
            if mls_feedback:
                lines.append(f"**{'详细反馈' if is_zh else 'Detailed Feedback'}:**")
                lines.append("")
                lines.append(redact_path_text(mls_feedback))
                lines.append("")

            lines.append("---")
            lines.append("")

    # ── 7. Figures & Visualizations ──
    figure_links = _extract_figure_links_from_history(history)
    if figure_links:
        lines.append("## 7. " + ("图表与可视化" if is_zh else "Figures & Visualizations"))
        lines.append("")
        for i, (fig_name, fig_url) in enumerate(figure_links, 1):
            lines.append(f"### Figure {i}: {fig_name}")
            lines.append("")
            lines.append(f"![{fig_name}]({fig_url})")
            lines.append("")
            lines.append(f"[{'下载' if is_zh else 'Download'}]({fig_url})")
            lines.append("")

    # ── 8. Discussion & Conclusion ──
    lines.append("## 8. " + ("讨论与结论" if is_zh else "Discussion & Conclusion"))
    lines.append("")

    # Collect the finalizer summary (Scientific Critic / PI)
    skip_markers = (
        "iteration_prompt", "请选择下一步", "Please choose",
        "正在分析", "is analyzing", "正在汇总", "is summarizing",
        "Thinking", "思考中", "撰写小报告", "writing sub-report",
        "撰写研究草案", "writing the draft report",
        "⏳", "✍️", "📝", "sub_report_checkpoint",
        "step_checkpoint", "🔍 **第", "🔍 **Step",
    )
    final_summaries: list[str] = []
    for item in reversed(history):
        if item.get("role") != "assistant":
            continue
        content = item.get("content", "")
        role_id = item.get("role_id", "")
        if not content or len(content) < 50:
            continue
        if any(m in content for m in skip_markers):
            continue
        if role_id == "principal_investigator":
            final_summaries.append(content)
            if len(final_summaries) >= 2:
                break

    if final_summaries:
        for summary in reversed(final_summaries):
            lines.append(redact_path_text(summary))
            lines.append("")
    else:
        lines.append("_" + ("暂无总结。" if is_zh else "No conclusion available yet.") + "_")
        lines.append("")

    # Failure information
    if state.get("execution_failed"):
        lines.append("### " + ("失败信息" if is_zh else "Failure Details"))
        lines.append("")
        lines.append(f"- **{'失败步骤' if is_zh else 'Failed Step'}:** {state.get('failed_step', 'N/A')}")
        lines.append(f"- **{'失败原因' if is_zh else 'Failure Reason'}:** {state.get('failed_reason', 'Unknown')}")
        lines.append("")

    # ── Footer ──
    lines.append("---")
    lines.append("")
    lines.append(f"*{'由 VenusFactory 多智能体系统自动生成' if is_zh else 'Auto-generated by VenusFactory Multi-Agent System'}* | "
                 f"Session `{session_id[:8]}` | {generated_at}")
    lines.append("")

    return "\n".join(lines)


def _extract_step_feedback_from_history(
    history: list[dict[str, Any]], step: Any, tool_name: str
) -> str:
    """Extract the MLS feedback message for a specific execution step from conversation history."""
    step_str = str(step)
    executing_marker = f"Step {step_str}"
    found_step = False
    for item in history:
        if item.get("role") != "assistant" or item.get("role_id") != "machine_learning_specialist":
            continue
        content = item.get("content", "")
        if not found_step:
            if executing_marker in content and ("⏳" in content or "executing" in content.lower() or "正在执行" in content):
                found_step = True
            continue
        # This is the feedback message right after the executing message
        if "📎" in content or "🖼️" in content or "Cloud Download" in content or "云端下载" in content or "File Preview" in content or "文件预览" in content:
            return content
        if len(content) > 100 and (tool_name in content or "Summary" in content or "summary" in content or "输出" in content):
            return content
        break
    return ""


def _extract_figure_links_from_history(history: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Extract all generated image links from the conversation history."""
    _IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".bmp", ".tiff")
    figures: list[tuple[str, str]] = []
    seen_urls: set[str] = set()
    img_emoji_pat = re.compile(r"🖼️[^[]*\[([^\]]+)\]\(([^)]+)\)")
    md_img_pat = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
    md_link_pat = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
    for item in history:
        content = item.get("content", "")
        if not content:
            continue
        for m in img_emoji_pat.finditer(content):
            name, url = m.group(1), m.group(2)
            if url and url not in seen_urls:
                seen_urls.add(url)
                figures.append((name or "figure", url))
        for m in md_img_pat.finditer(content):
            name, url = m.group(1), m.group(2)
            if url and url not in seen_urls:
                seen_urls.add(url)
                figures.append((name or "figure", url))
        for m in md_link_pat.finditer(content):
            name, url = m.group(1), m.group(2)
            if url not in seen_urls and any(url.lower().endswith(ext) for ext in _IMAGE_EXTS):
                seen_urls.add(url)
                figures.append((name, url))
    return figures


@router.get("/sessions/{session_id}/report")
async def get_experiment_report(session_id: str, request: Request):
    _record_access_event(request, "/api/chat/sessions/{id}/report")
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)

    report = _generate_experiment_report(state)
    filename = f"experiment_report_{session_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    return Response(
        content=report,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
