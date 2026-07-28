"""Science Agent (kimi-code) session controls aligned with Kimi Code docs.

Exposes compact / status / plan-mode / reset-context / fork for the WebUI.
Permission mode stays ``manual`` so VF security + ApprovalCard remain the gate.
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from agent.kimi_client import KimiAPIError, KimiClient
from agent.kimi_daemon import base_url as kimi_base_url
from config import get_config
from logger import get_logger

from web_v2.chat_api._hooks_runtime import _get_lock, _session_store
from web_v2.chat_api._shared import (
    _assert_session_access,
    _get_session_or_404,
    _infer_chat_mode,
    _record_access_event,
    _snapshot,
)
from web_v2.chat_api._stream_kimi import _ensure_kimi_session, _refresh_kimi_context

_logger = get_logger("chat_api.agent_controls")
router = APIRouter()


class AgentCompactRequest(BaseModel):
    instruction: str = Field(default="")


class AgentProfileRequest(BaseModel):
    plan_mode: Optional[bool] = None


class AgentForkRequest(BaseModel):
    title: str = Field(default="")


def _require_agent(state: dict[str, Any]) -> None:
    if _infer_chat_mode(state) != "science_agent" and state.get("engine") != "kimi-code":
        raise HTTPException(
            status_code=400,
            detail="These controls are only available for Science Agent (kimi-code).",
        )


async def _kimi_client_for_state(state: dict[str, Any]) -> KimiClient:
    mode = get_config().server.mode or "local"
    if mode == "online":
        from agent.kimi_session_pool import get_pool as get_kimi_pool
        inst = await get_kimi_pool().acquire(state["session_id"])
        return KimiClient(base_url=inst.base_url)
    return KimiClient(base_url=kimi_base_url())


@router.get("/sessions/{session_id}/agent/status")
async def agent_status(session_id: str, request: Request):
    _record_access_event(request, "/api/chat/sessions/{id}/agent/status")
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)
    _require_agent(state)
    kimi_sid = str(state.get("kimi_session_id") or "")
    if not kimi_sid:
        return {
            "success": True,
            "kimi_session_id": "",
            "kimi_context": state.get("kimi_context") or {},
            "kimi_plan_mode": bool(state.get("kimi_plan_mode")),
        }
    client = await _kimi_client_for_state(state)
    try:
        await _refresh_kimi_context(client, state, kimi_sid)
        await _session_store.save(session_id)
        return {
            "success": True,
            "kimi_session_id": kimi_sid,
            "kimi_context": state.get("kimi_context") or {},
            "kimi_plan_mode": bool(state.get("kimi_plan_mode")),
        }
    except KimiAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    finally:
        await client.aclose()


@router.post("/sessions/{session_id}/agent/compact")
async def agent_compact(session_id: str, payload: AgentCompactRequest, request: Request):
    """Manual context compression — Kimi Code ``/compact``."""
    _record_access_event(request, "/api/chat/sessions/{id}/agent/compact")
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)
    _require_agent(state)
    lock = await _get_lock(session_id)
    async with lock:
        client = await _kimi_client_for_state(state)
        try:
            kimi_sid = await _ensure_kimi_session(client, state)
            instruction = (payload.instruction or "").strip() or (
                "Preserve the user's goals, key tool results, file paths, IDs, "
                "and recent decisions. Drop verbose tool dumps."
            )
            await client.compact(kimi_sid, instruction=instruction)
            ui_lang = str(state.get("user_lang") or state.get("ui_lang") or "")
            state.setdefault("history", []).append({
                "role": "assistant",
                "content": (
                    "🗜️ **Context compressed** (manual)."
                    if ui_lang != "zh"
                    else "🗜️ **上下文已压缩**（手动）。"
                ),
                "kind": "compaction",
                "phase": "compaction",
                "role_id": "assistant",
            })
            await _refresh_kimi_context(client, state, kimi_sid)
            await _session_store.save(session_id)
            return {
                "success": True,
                "snapshot": _snapshot(state),
            }
        except KimiAPIError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        finally:
            await client.aclose()


@router.post("/sessions/{session_id}/agent/profile")
async def agent_profile(session_id: str, payload: AgentProfileRequest, request: Request):
    """Toggle Plan mode (Kimi ``/plan``). Permission stays manual for VF gates."""
    _record_access_event(request, "/api/chat/sessions/{id}/agent/profile")
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)
    _require_agent(state)
    if payload.plan_mode is None:
        raise HTTPException(status_code=400, detail="Provide plan_mode.")
    lock = await _get_lock(session_id)
    async with lock:
        state["kimi_plan_mode"] = bool(payload.plan_mode)
        kimi_sid = str(state.get("kimi_session_id") or "")
        if kimi_sid:
            client = await _kimi_client_for_state(state)
            try:
                await client.update_profile(
                    kimi_sid,
                    agent_config={"plan_mode": bool(payload.plan_mode)},
                )
                await _refresh_kimi_context(client, state, kimi_sid)
            except KimiAPIError as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            finally:
                await client.aclose()
        await _session_store.save(session_id)
        return {
            "success": True,
            "kimi_plan_mode": bool(state.get("kimi_plan_mode")),
            "snapshot": _snapshot(state),
        }


@router.post("/sessions/{session_id}/agent/reset-context")
async def agent_reset_context(session_id: str, request: Request):
    """Start a fresh kimi session (Kimi ``/new`` / ``/clear``) for this chat."""
    _record_access_event(request, "/api/chat/sessions/{id}/agent/reset-context")
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)
    _require_agent(state)
    lock = await _get_lock(session_id)
    async with lock:
        state["kimi_session_id"] = ""
        state.pop("_kimi_bound_model", None)
        state.pop("kimi_current_prompt_id", None)
        state["kimi_context"] = {}
        ui_lang = str(state.get("user_lang") or state.get("ui_lang") or "")
        state.setdefault("history", []).append({
            "role": "assistant",
            "content": (
                "🧹 **Agent context cleared** — next message starts a fresh kimi session."
                if ui_lang != "zh"
                else "🧹 **Agent 上下文已清空** — 下一条消息将开启新的 kimi session。"
            ),
            "kind": "status",
            "phase": "agent_reset",
            "role_id": "assistant",
        })
        await _session_store.save(session_id)
        return {"success": True, "snapshot": _snapshot(state)}


@router.post("/sessions/{session_id}/agent/fork")
async def agent_fork(session_id: str, payload: AgentForkRequest, request: Request):
    """Fork the kimi session (Kimi ``/fork``) and bind this chat to the fork."""
    _record_access_event(request, "/api/chat/sessions/{id}/agent/fork")
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)
    _require_agent(state)
    lock = await _get_lock(session_id)
    async with lock:
        client = await _kimi_client_for_state(state)
        try:
            kimi_sid = await _ensure_kimi_session(client, state)
            data = await client.fork_session(
                kimi_sid,
                title=(payload.title or "").strip() or None,
            )
            new_id = ""
            if isinstance(data, dict):
                new_id = str(data.get("id") or (data.get("session") or {}).get("id") or "")
            if not new_id:
                raise HTTPException(status_code=502, detail=f"Fork returned no id: {data}")
            state["kimi_session_id"] = new_id
            ui_lang = str(state.get("user_lang") or state.get("ui_lang") or "")
            state.setdefault("history", []).append({
                "role": "assistant",
                "content": (
                    f"🌿 **Session forked** — continuing on `{new_id[:8]}…`."
                    if ui_lang != "zh"
                    else f"🌿 **会话已分叉** — 后续在 `{new_id[:8]}…` 上继续。"
                ),
                "kind": "status",
                "phase": "agent_fork",
                "role_id": "assistant",
            })
            await _refresh_kimi_context(client, state, new_id)
            await _session_store.save(session_id)
            return {"success": True, "kimi_session_id": new_id, "snapshot": _snapshot(state)}
        except KimiAPIError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        finally:
            await client.aclose()
