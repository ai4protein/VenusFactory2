"""Shared helpers for chat_api sub-routers (request fingerprinting, snapshot,
session access control, online quota).

Pydantic models live in ``_models``. Heavy upload/archive helpers live in
``_uploads``. Both are re-exported from the package ``__init__`` for
external compatibility.
"""
import hashlib
import hmac
import json
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import HTTPException, Request

from web.utils.common_utils import redact_path_text
from web_v2.analytics_store import analytics_store

from web_v2.chat_api._hooks_runtime import (
    _ONLINE_DAILY_CHAT_LIMIT,
    _SESSION_TOKEN_SECRET,
    _SESSION_TOKEN_TTL_HOURS,
    _cfg,
    _session_store,
)


# ── Runtime mode + request fingerprinting ──────────────────────────────────

def _runtime_mode() -> str:
    return _cfg.server.mode


def _extract_user_agent(request: Request) -> str:
    return (request.headers.get("user-agent", "") or "").strip().lower()


def _extract_origin(request: Request) -> str:
    return (request.headers.get("origin", "") or "").strip().lower()


def _extract_client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        parts = [p.strip() for p in xff.split(",") if p.strip()]
        if parts:
            return parts[0]
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _build_owner_fingerprint(request: Request) -> str:
    # NOTE: do NOT include Origin in the fingerprint. Browsers omit the
    # Origin header on default GET fetch() requests but include it on POST,
    # so mixing it in causes CREATE (POST, origin set) and LIST (GET, origin
    # empty) for the same user to produce different fingerprints — and the
    # user's own session never shows in their sidebar.
    ip = _extract_client_ip(request)
    ua = _extract_user_agent(request)
    raw = f"{ip}|{ua}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _session_owner_key_for_request(request: Request) -> str:
    if _runtime_mode() != "online":
        return "local"
    return _build_owner_fingerprint(request)


# ── Session access tokens (HMAC) ───────────────────────────────────────────

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
    if _runtime_mode() != "online":
        sids = await _session_store.list_ids(None)
        return set(sids)
    sids = await _session_store.list_ids(owner_key)
    return set(sids)


# ── Text utilities ─────────────────────────────────────────────────────────

def _is_zh_text(text: str) -> bool:
    raw = str(text or "")
    return any("一" <= ch <= "鿿" for ch in raw)


def _should_skip_research(questions: list[dict[str, Any]], answers: list[dict[str, Any]]) -> bool:
    for i, ans in enumerate(answers):
        if i >= len(questions):
            break
        opts = questions[i].get("options", []) or []
        selected = ans.get("selected_options", []) or []
        custom = str(ans.get("custom_text", "") or "").strip().lower()
        for idx in selected:
            if isinstance(idx, int) and 0 <= idx < len(opts):
                text = str(opts[idx]).strip().lower()
                if ("skip research" in text) or ("跳过 research" in text):
                    return True
        if custom and ("skip research" in custom or "跳过research" in custom or "跳过 research" in custom):
            return True
    return False


# ── Snapshot / redaction ───────────────────────────────────────────────────

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
    waiting_for = state.get("waiting_for", "")
    waiting_for_str = waiting_for if isinstance(waiting_for, str) else ""
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
        "waiting_for": waiting_for_str,
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


# ── Session retrieval + access check ───────────────────────────────────────

async def _get_session_or_404(session_id: str) -> dict[str, Any]:
    state = await _session_store.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    return state


def _assert_session_access(state: dict[str, Any], request: Request) -> None:
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
                expires_at = expires_at.replace(tzinfo=UTC)
        except ValueError:
            expires_at = datetime.now(UTC) - timedelta(seconds=1)
        if datetime.now(UTC) >= expires_at:
            raise HTTPException(
                status_code=401,
                detail={"code": "SESSION_TOKEN_EXPIRED", "message": "Session access token has expired."},
            )


# ── Online quota gating ────────────────────────────────────────────────────

async def _consume_online_chat_quota_or_429(request: Request) -> None:
    if not _cfg.server.is_online:
        return

    ip = _extract_client_ip(request)
    today = datetime.now().strftime("%Y-%m-%d")

    current = analytics_store.get_ip_chat_usage(ip, today)
    if current >= _ONLINE_DAILY_CHAT_LIMIT:
        raise HTTPException(
            status_code=429,
            detail={
                "code": "CHAT_DAILY_LIMIT_REACHED",
                "message": f"Online mode limit reached: up to {_ONLINE_DAILY_CHAT_LIMIT} chats per IP per day.",
            },
        )
    analytics_store.increment_ip_chat_usage(ip, today)


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

    ip = _extract_client_ip(request)
    today = datetime.now().strftime("%Y-%m-%d")
    used = analytics_store.get_ip_chat_usage(ip, today)
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
