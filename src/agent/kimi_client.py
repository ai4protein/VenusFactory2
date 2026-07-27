"""Async REST + WebSocket client for kimi-code server (v0.19.x).

All REST responses share the envelope `{code, msg, data, request_id}`; this
client unwraps `data` and raises `KimiAPIError` when `code != 0` or HTTP
status is non-2xx.

WebSocket protocol (see /asyncapi.json on the running server):

  client → server:
    {"type": "client_hello", "id": "<uuid>", "payload": {"client_id": "..."}}
    {"type": "subscribe",   "id": "<uuid>", "payload": {"session_ids": ["<sid>"]}}
    {"type": "unsubscribe", "id": "<uuid>", "payload": {"session_ids": ["<sid>"]}}
    {"type": "ping",        "id": "<uuid>"}

  server → client:
    {"type": "server_hello", ...}
    {"type": "client_hello_ack", "id": ..., "payload": ...}
    {"type": "subscribe_ack", "id": ..., "payload": ...}
    {"type": "session_event", "seq": N, "epoch": "...", "offset": N,
     "session_id": "...", "timestamp": "...",
     "payload": {"type": "<event-type>", ...}}

session_event payload `type` values we map to KimiEvent:
    thinking.delta    → kind="thinking",  text=delta
    assistant.delta   → kind="text",      text=delta
    tool.call.started → kind="tool_call_start"
    tool.result       → kind="tool_result"
    turn.started      → kind="turn_started"
    turn.ended        → kind="turn_ended"
    error             → kind="error"
    (other event types are passed through with kind="other" and the raw payload
    in `raw` so callers can inspect without us hard-coding every variant.)

stream_session_events() also yields one synthetic `kind="subscribed"` event
after the server acknowledges the subscribe frame. Callers should wait for it
before POSTing a prompt to avoid racing the first session_event.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

import httpx
import websockets

from logger import get_logger

_logger = get_logger("agent.kimi_client")


class KimiAPIError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, code: int | None = None,
                 detail: Any | None = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.detail = detail


@dataclass
class KimiEvent:
    kind: str  # "thinking" | "text" | "tool_call_start" | "tool_result"
               # | "turn_started" | "turn_ended" | "error" | "other"
    turn_id: str = ""
    text: str = ""
    tool_call_id: str = ""
    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)
    tool_output: Any = None
    is_error: bool = False
    error_code: str = ""
    raw_type: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


def _unwrap(resp: httpx.Response) -> Any:
    try:
        body = resp.json()
    except ValueError as exc:
        raise KimiAPIError(f"non-JSON response from kimi: {resp.text[:200]}",
                           status=resp.status_code) from exc
    if resp.status_code >= 400:
        raise KimiAPIError(
            f"kimi HTTP {resp.status_code}: {body}",
            status=resp.status_code,
            code=body.get("code") if isinstance(body, dict) else None,
            detail=body,
        )
    if isinstance(body, dict) and body.get("code", 0) != 0:
        raise KimiAPIError(
            f"kimi error code={body.get('code')}: {body.get('msg')}",
            status=resp.status_code,
            code=body.get("code"),
            detail=body,
        )
    return body.get("data") if isinstance(body, dict) else body


def _map_event(env: dict[str, Any]) -> KimiEvent:
    """Translate one session-scoped frame into KimiEvent.

    Wire shape (v0.19.x — empirically verified, differs from /asyncapi.json):
        {"type": "<event-type>",     # e.g. "thinking.delta" — TOP-LEVEL, no
                                     #  "session_event" wrapper as docs imply
         "seq": N, "epoch": "...", "session_id": "...", "timestamp": "...",
         "payload": {"type": <same>, "turnId": "...", "delta": "...", ...}}
    """
    payload = env.get("payload") or {}
    et = str(env.get("type") or payload.get("type") or "")
    common = {
        "raw_type": et,
        "raw": payload,
        "turn_id": str(payload.get("turnId") or ""),
    }
    if et == "thinking.delta":
        return KimiEvent(kind="thinking", text=str(payload.get("delta") or ""), **common)
    if et == "assistant.delta":
        return KimiEvent(kind="text", text=str(payload.get("delta") or ""), **common)
    if et == "tool.call.started":
        return KimiEvent(
            kind="tool_call_start",
            tool_call_id=str(payload.get("toolCallId") or ""),
            tool_name=str(payload.get("name") or ""),
            tool_args=payload.get("args") or {},
            **common,
        )
    if et == "tool.result":
        return KimiEvent(
            kind="tool_result",
            tool_call_id=str(payload.get("toolCallId") or ""),
            tool_output=payload.get("output"),
            is_error=bool(payload.get("isError")),
            **common,
        )
    if et == "turn.started":
        return KimiEvent(kind="turn_started", **common)
    if et == "turn.ended":
        return KimiEvent(
            kind="turn_ended",
            text=str(payload.get("reason") or ""),
            is_error=bool(payload.get("error")),
            **common,
        )
    if et == "error":
        return KimiEvent(
            kind="error",
            text=str(payload.get("message") or ""),
            error_code=str(payload.get("code") or ""),
            **common,
        )
    return KimiEvent(kind="other", **common)


class KimiClient:
    def __init__(self, base_url: str, *, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(timeout=timeout, base_url=self.base_url)

    async def aclose(self) -> None:
        await self._http.aclose()

    # ── REST ──────────────────────────────────────────────────────────────
    async def healthz(self) -> bool:
        try:
            r = await self._http.get("/api/v1/healthz")
            return r.status_code == 200 and (r.json().get("data") or {}).get("ok") is True
        except Exception:
            return False

    async def auth_status(self) -> dict[str, Any]:
        return await _unwrap_async(self._http.get("/api/v1/auth"))

    async def list_providers(self) -> list[dict[str, Any]]:
        data = await _unwrap_async(self._http.get("/api/v1/providers"))
        return list((data or {}).get("items") or [])

    async def list_mcp_servers(self) -> list[dict[str, Any]]:
        data = await _unwrap_async(self._http.get("/api/v1/mcp/servers"))
        return list((data or {}).get("servers") or [])

    async def restart_mcp_server(self, server_id: str) -> dict[str, Any]:
        return await _unwrap_async(
            self._http.post(f"/api/v1/mcp/servers/{server_id}:restart")
        )

    async def create_session(
        self,
        *,
        cwd: str,
        title: str = "",
        mcp_servers: list[str] | None = None,
        thinking: str = "high",
        permission_mode: str = "yolo",
        model: str | None = None,
        system_prompt: str | None = None,
    ) -> str:
        agent_config: dict[str, Any] = {
            "thinking": thinking,
            "permission_mode": permission_mode,
        }
        if mcp_servers:
            agent_config["mcp_servers"] = list(mcp_servers)
        if model:
            agent_config["model"] = model
        if system_prompt:
            agent_config["system_prompt"] = system_prompt
        body: dict[str, Any] = {
            "metadata": {"cwd": cwd},
            "agent_config": agent_config,
        }
        if title:
            body["title"] = title
        data = await _unwrap_async(self._http.post("/api/v1/sessions", json=body))
        sid = str((data or {}).get("id") or "")
        if not sid:
            raise KimiAPIError(f"create_session: no id in response: {data}")
        # Quirk (v0.19.x): POST /sessions silently drops `permission_mode`
        # from agent_config — the new session always boots in "manual". A
        # follow-up POST /sessions/{id}/profile WITH the same agent_config
        # under the (nested) `agent_config` key actually persists it. Without
        # this, every tool call waits 60 s for a UI approval that never
        # comes through our REST/WS layer. See session/status to verify.
        try:
            await self._http.post(
                f"/api/v1/sessions/{sid}/profile",
                json={"agent_config": {"permission_mode": permission_mode}},
            )
        except Exception:  # noqa: BLE001 - best-effort; fall through if it fails
            pass
        return sid

    async def get_session(self, session_id: str) -> dict[str, Any]:
        return await _unwrap_async(self._http.get(f"/api/v1/sessions/{session_id}"))

    async def delete_session(self, session_id: str) -> None:
        # The OpenAPI exposes session deletion via the catch-all
        # POST /sessions/{id}/{tail}; the simplest is to call archive. For now
        # we just leave the session in kimi's store — it has its own GC.
        # Reserved for later; no-op to keep callers compiling.
        return None

    async def submit_prompt(self, session_id: str, text: str) -> dict[str, Any]:
        body = {"content": [{"type": "text", "text": text}]}
        return await _unwrap_async(
            self._http.post(f"/api/v1/sessions/{session_id}/prompts", json=body)
        )

    async def list_sessions(self) -> list[dict[str, Any]]:
        """All sessions kimi currently knows about.

        Used by the long-running approval worker to drain queued approvals
        across sessions whose chat stream isn't active (otherwise approvals
        time out after 60s when nobody's listening).
        """
        data = await _unwrap_async(self._http.get("/api/v1/sessions"))
        return list((data or {}).get("items") or [])

    async def list_pending_approvals(self, session_id: str) -> list[dict[str, Any]]:
        """Return tool-call approvals currently waiting for a decision.

        Kimi raises approvals even when `permission_mode=yolo` is set at the
        session level — yolo only auto-approves a *whitelist* of low-risk
        builtins (Read/Write/FetchURL/Skill). MCP tools, Bash, and Edit
        still go through the approval queue. Without a UI handler they
        time out after 60s.
        """
        data = await _unwrap_async(
            self._http.get(
                f"/api/v1/sessions/{session_id}/approvals",
                params={"status": "pending"},
            )
        )
        return list((data or {}).get("items") or [])

    async def approve(
        self,
        session_id: str,
        approval_id: str,
        *,
        scope: str = "session",
        feedback: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"decision": "approved", "scope": scope}
        if feedback:
            body["feedback"] = feedback
        return await _unwrap_async(
            self._http.post(
                f"/api/v1/sessions/{session_id}/approvals/{approval_id}",
                json=body,
            )
        )

    async def reject(
        self,
        session_id: str,
        approval_id: str,
        *,
        feedback: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"decision": "rejected"}
        if feedback:
            body["feedback"] = feedback
        return await _unwrap_async(
            self._http.post(
                f"/api/v1/sessions/{session_id}/approvals/{approval_id}",
                json=body,
            )
        )

    async def decide_approval(
        self,
        session_id: str,
        approval_id: str,
        *,
        decision: str,
        scope: str | None = None,
        feedback: str | None = None,
        selected_label: str | None = None,
    ) -> dict[str, Any]:
        """POST an approval decision (`approved`/`rejected`/`cancelled`).

        `selected_label` is used by ExitPlanMode when the agent offered
        alternate approaches via the tool's `options` parameter.
        """
        body: dict[str, Any] = {"decision": decision}
        if scope:
            body["scope"] = scope
        if feedback:
            body["feedback"] = feedback
        if selected_label:
            body["selected_label"] = selected_label
        return await _unwrap_async(
            self._http.post(
                f"/api/v1/sessions/{session_id}/approvals/{approval_id}",
                json=body,
            )
        )

    async def list_pending_questions(self, session_id: str) -> list[dict[str, Any]]:
        """Return AskUserQuestion requests waiting for a structured answer.

        Kimi session status becomes `awaiting_question` until each item is
        resolved via `answer_question` or dismissed. Without a host handler
        the AskUserQuestion tool fails and the agent falls back to prose.
        """
        data = await _unwrap_async(
            self._http.get(
                f"/api/v1/sessions/{session_id}/questions",
                params={"status": "pending"},
            )
        )
        return list((data or {}).get("items") or [])

    async def answer_question(
        self,
        session_id: str,
        question_id: str,
        *,
        answers: dict[str, Any],
        method: str = "click",
        note: str | None = None,
    ) -> dict[str, Any]:
        """Resolve a pending AskUserQuestion with structured answers.

        `answers` maps each sub-question id →
        `{kind: single|multi|other|multi_with_other|skipped, ...}`.
        """
        body: dict[str, Any] = {"answers": answers, "method": method}
        if note:
            body["note"] = note
        return await _unwrap_async(
            self._http.post(
                f"/api/v1/sessions/{session_id}/questions/{question_id}",
                json=body,
            )
        )

    async def dismiss_question(
        self,
        session_id: str,
        question_id: str,
    ) -> dict[str, Any]:
        """Dismiss a pending question without answering (tail `:dismiss`)."""
        return await _unwrap_async(
            self._http.post(
                f"/api/v1/sessions/{session_id}/questions/{question_id}:dismiss",
                json={},
            )
        )

    # ── WebSocket ─────────────────────────────────────────────────────────
    def ws_url(self) -> str:
        if self.base_url.startswith("https://"):
            return "wss://" + self.base_url[len("https://") :] + "/api/v1/ws"
        if self.base_url.startswith("http://"):
            return "ws://" + self.base_url[len("http://") :] + "/api/v1/ws"
        return self.base_url + "/api/v1/ws"

    async def stream_session_events(
        self,
        session_id: str,
        *,
        client_id: str | None = None,
    ) -> AsyncIterator[KimiEvent]:
        """Connect to /api/v1/ws, subscribe to session_id, yield KimiEvents.

        Caller should iterate this in a task; on cancel the connection closes.

        Wire-level note (v0.19.x): kimi acknowledges client_hello AND
        explicit subscribe frames with a single `{"type":"ack","id":<our id>,
        "code":0,"payload":{...}}` shape (NOT separate "client_hello_ack" /
        "subscribe_ack" types). We track the IDs we sent and synthesize
        `kind="subscribed"` on the first matching ack so the caller can
        safely submit prompts without racing the first session_event.
        """
        url = self.ws_url()
        client_id = client_id or f"venusfactory-{uuid.uuid4().hex[:8]}"
        hello_id = uuid.uuid4().hex
        sub_id = uuid.uuid4().hex
        async with websockets.connect(url, max_size=2**24) as ws:
            await ws.send(json.dumps({
                "type": "client_hello",
                "id": hello_id,
                "payload": {"client_id": client_id, "subscriptions": [session_id]},
            }))
            # We don't strictly require client_hello_ack before subscribe;
            # subscribe explicitly to be safe even if `client_hello` already
            # subscribed via `subscriptions`.
            await ws.send(json.dumps({
                "type": "subscribe",
                "id": sub_id,
                "payload": {"session_ids": [session_id]},
            }))
            subscribed_emitted = False
            async for msg in ws:
                try:
                    env = json.loads(msg)
                except (TypeError, ValueError):
                    continue
                if not isinstance(env, dict):
                    continue
                t = env.get("type")
                # kimi sends application-level ping frames (NOT WS protocol
                # pings, which the websockets lib auto-handles). server_hello
                # advertises heartbeat_ms (default 30 000); if we don't reply,
                # kimi closes the socket with code 1006 after the timeout.
                # Wire shape: ping = {type, timestamp, payload:{nonce}}; pong
                # must echo {type:"pong", payload:{nonce:<same>}}.
                if t == "ping":
                    nonce = (env.get("payload") or {}).get("nonce") or ""
                    try:
                        await ws.send(json.dumps({
                            "type": "pong",
                            "payload": {"nonce": nonce},
                        }))
                    except Exception:
                        pass
                    continue
                if t == "ack" and not subscribed_emitted:
                    ack_id = env.get("id")
                    payload = env.get("payload") or {}
                    # Either the client_hello ack (carries
                    # `accepted_subscriptions`) or the explicit subscribe ack
                    # (carries `accepted`) is enough proof we're subscribed.
                    accepted = (payload.get("accepted_subscriptions")
                                or payload.get("accepted")
                                or [])
                    if (ack_id in (hello_id, sub_id) and env.get("code", 0) == 0
                            and session_id in (accepted or [])):
                        subscribed_emitted = True
                        yield KimiEvent(kind="subscribed", raw_type="ack", raw=payload)
                    continue
                # Any frame carrying a matching session_id is a session-scoped
                # event (per actual wire format — the asyncapi spec is wrong
                # about a "session_event" envelope wrapper).
                if env.get("session_id") == session_id:
                    yield _map_event(env)
                    continue
                if t == "error":
                    payload = env.get("payload") or {}
                    yield KimiEvent(
                        kind="error",
                        text=str(payload.get("message") or env.get("message") or ""),
                        error_code=str(payload.get("code") or ""),
                        raw_type="error",
                        raw=payload,
                    )
                # Ignore other control frames (server_hello, pong, ...).


async def _unwrap_async(coro_or_resp) -> Any:
    """Helper: await an httpx coroutine and unwrap the envelope."""
    resp = await coro_or_resp if asyncio.iscoroutine(coro_or_resp) else coro_or_resp
    return _unwrap(resp)
