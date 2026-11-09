"""Authenticated Kimi 2.5 REST/WS transport with recoverable transcript cursors."""
from __future__ import annotations

import asyncio
import json
import math
import re
import uuid
from urllib.parse import urlsplit

import httpx
import websockets
from websockets.exceptions import ConnectionClosed

from agent.kimi_client import KimiClient, KimiEvent, KimiAPIError, _unwrap
from .kimi_web_events import KimiTranscript, TranscriptGap


class KimiWebClient(KimiClient):
    def __init__(self, base_url, *, token, timeout=30.0, transport=None):
        parsed = urlsplit(base_url)
        if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
            raise ValueError("Kimi server must use its isolated loopback endpoint")
        if not re.fullmatch(r"[A-Za-z0-9._~-]+", token):
            raise ValueError("Invalid Kimi server token")
        self.base_url = base_url.rstrip("/")
        self._token = token
        self._http = httpx.AsyncClient(base_url=self.base_url, timeout=timeout, trust_env=False,
            headers={"Authorization": "Bearer " + token}, transport=transport)
        self._transcripts = {}

    async def get_status(self, session_id):
        stream = self._wire_frames(session_id)
        try:
            async with asyncio.timeout(15):
                async for item in stream:
                    if item.get("type") == "agent" and item.get("session_id") == session_id and item.get("agent_id") == "main":
                        required = {"model", "thinking", "permission_mode", "plan_mode", "swarm_mode"}
                        if not required <= item.keys():
                            raise KimiAPIError("Kimi agent did not report its effective configuration")
                        return {"status": item.get("status"), "model": item["model"],
                            "thinking_level": item["thinking"], "permission": item["permission_mode"],
                            "plan_mode": item["plan_mode"], "swarm_mode": item["swarm_mode"]}
                raise KimiAPIError("Kimi agent status stream ended")
        finally:
            await stream.aclose()

    async def usage_rollup(self, session_id):
        """Read a cumulative runtime snapshot; never sum repeated WS updates."""
        usage = (await self.get_session(session_id)).get("usage")
        if not isinstance(usage, dict):
            return None
        fields = ("input_tokens", "output_tokens", "cache_read_tokens",
                  "cache_creation_tokens", "context_tokens", "context_limit", "turn_count")
        result = {}
        for field in fields:
            if field in usage:
                value = usage[field]
                if type(value) is not int or not 0 <= value <= 2**53 - 1:
                    raise KimiAPIError("Kimi session reported invalid token usage")
                result[field] = value
        if "total_cost_usd" in usage:
            value = usage["total_cost_usd"]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise KimiAPIError("Kimi session reported invalid usage cost")
            result["total_cost_usd"] = value
        return result or None

    def transcript(self, sid):
        return self._transcripts.setdefault(sid, KimiTranscript())

    async def list_pending_approvals(self, session_id):
        return [dict(item) for item in self.transcript(session_id).interactions.values()
                if item["kind"] == "approval"]

    async def list_pending_questions(self, session_id):
        return [dict(item) for item in self.transcript(session_id).interactions.values()
                if item["kind"] == "question"]

    async def _respond(self, sid, identity, body):
        result = _unwrap(await self._http.post(
            f"/api/v1/sessions/{sid}/interactions/{identity}", json=body))
        self.transcript(sid).interaction({"interaction_id": identity, "status": "ended"})
        return result

    async def decide_approval(self, session_id, approval_id, *, decision,
                              scope=None, feedback=None, selected_label=None):
        if decision not in {"approved", "rejected", "cancelled"}:
            raise ValueError("Invalid Kimi approval decision")
        body = {"kind": "approval", "decision": decision}
        for key, value in (("scope", scope), ("feedback", feedback), ("selected_label", selected_label)):
            if value is not None:
                body[key] = value
        return await self._respond(session_id, approval_id, body)

    async def approve(self, session_id, approval_id, *, scope="session", feedback=None):
        return await self.decide_approval(session_id, approval_id, decision="approved", scope=scope, feedback=feedback)

    async def reject(self, session_id, approval_id, *, feedback=None):
        return await self.decide_approval(session_id, approval_id, decision="rejected", feedback=feedback)

    async def answer_question(self, session_id, question_id, *, answers, method="click", note=None):
        body = {"kind": "question", "answers": answers, "method": method}
        if note:
            body["note"] = note
        return await self._respond(session_id, question_id, body)

    async def dismiss_question(self, session_id, question_id):
        return await self._respond(session_id, question_id, {"kind": "question", "dismissed": True})

    async def catch_up(self, sid):
        state = self.transcript(sid)
        data = _unwrap(await self._http.get(f"/api/v1/sessions/{sid}/transcript/ops",
            params={"agent_id": "main", "since_seq": state.seq}))
        events = []
        if data.get("complete"):
            for batch in data.get("batches", []):
                events.extend(state.consume(batch))
            if state.seq != data.get("latest_seq"):
                raise TranscriptGap("Kimi replay did not reach its advertised cursor")
            return events
        data = _unwrap(await self._http.get(f"/api/v1/sessions/{sid}/transcript",
            params={"agent_id": "main", "page_size": 100}))
        if data.get("has_more"):
            raise TranscriptGap("Kimi transcript recovery exceeds one bounded session page")
        return state.reset(data, data["seq"])

    async def _wire_frames(self, session_id, since=0):
        """One authenticated connection, including subscription and heartbeat."""
        async with websockets.connect(self.base_url.replace("http://", "ws://", 1) + "/api/v1/ws",
                subprotocols=["kimi-code.bearer." + self._token], max_size=2**24) as ws:
            hello, subscription = uuid.uuid4().hex, uuid.uuid4().hex
            await ws.send(json.dumps({"type": "client_hello", "id": hello,
                "payload": {"client_id": "venusfactory-harness"}}))
            async for raw in ws:
                item = json.loads(raw)
                if item.get("type") == "ping":
                    await ws.send(json.dumps({"type": "pong", "payload": {"nonce": (item.get("payload") or {}).get("nonce")}}))
                elif item.get("type") == "ack":
                    if item.get("code", 0) != 0:
                        raise KimiAPIError("Kimi WebSocket request was rejected")
                    if item.get("id") == hello:
                        payload = {"session_id": session_id, "transcript": {"main": "delta"}}
                        if since:
                            payload["transcript_since"] = {"main": since}
                        await ws.send(json.dumps({"type": "subscribe_v2", "id": subscription, "payload": payload}))
                    elif item.get("id") == subscription:
                        if session_id not in item.get("payload", {}).get("accepted", []):
                            raise KimiAPIError("Kimi session subscription was rejected")
                        yield {"type": "subscribed", "session_id": session_id}
                else:
                    yield item

    async def stream_session_events(self, session_id, **_):
        """Reconnect without resubmitting the prompt or re-emitting prior output."""
        state = self.transcript(session_id)
        subscribed = False
        for attempt in range(3):
            stream = self._wire_frames(session_id, state.seq)
            try:
                async for item in stream:
                    kind = item.get("type")
                    if item.get("session_id") != session_id:
                        continue
                    payload = item.get("payload") or {}
                    if (item.get("agent_id") or payload.get("agent_id") or "main") != "main":
                        continue
                    if kind == "subscribed":
                        if not subscribed:
                            subscribed = True
                            yield KimiEvent(kind="subscribed", raw_type="subscribe_v2")
                        else:
                            for event in await self.catch_up(session_id):
                                yield event
                    elif kind == "interaction":
                        state.interaction(item)
                    elif kind == "transcript.reset":
                        for event in state.reset(payload["snapshot"], payload["seq"]):
                            yield event
                    elif kind == "transcript.ops":
                        if payload.get("seq", 0) > state.seq + 1:
                            for event in await self.catch_up(session_id):
                                yield event
                        for event in state.consume(payload):
                            yield event
                    elif kind == "usage":
                        yield KimiEvent(kind="other", raw_type="usage", raw={"usage": item.get("usage", {})})
                    elif kind == "error":
                        raise KimiAPIError("Kimi session reported an execution error")
            except ConnectionClosed:
                pass
            finally:
                await stream.aclose()
            if attempt == 2:
                raise KimiAPIError("Kimi event stream disconnected before completion")
            await asyncio.sleep(0.2 * (attempt + 1))
