"""Research-only OpenAI relay enforcing model, sampling and tool exposure.

The local runtime receives a disposable relay token, never the upstream key.
Only public request metadata and usage are logged; no prompts or credentials.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import secrets
import time
from pathlib import Path

import httpx
from aiohttp import web


def classify_upstream_error(raw, *, truncated=False):
    """Record fixed diagnostic signals, never free-text provider errors.

    Provider errors can echo prompts or authentication. Even code/type/param
    values are untrusted, so only recognized categories and field names leave
    this parser. Signals describe reported text, not a proven root cause.
    """
    result = {
        "body_prefix_sha256": hashlib.sha256(raw).hexdigest(),
        "inspected_bytes": len(raw),
        "truncated": truncated,
        "format": "unparsed",
        "reported_signals": [],
        "mentioned_fields": [],
    }
    if truncated:
        return result
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeError):
        return result
    result["format"] = "json"
    if not isinstance(data, dict):
        return result
    error = data.get("error", data)
    if isinstance(error, dict):
        values = [error.get(k) for k in ("code", "type", "message", "param")]
    else:
        values = [error]
    message = " ".join(str(v) for v in values if isinstance(v, (str, int))).lower()
    categories = {
        "model_unavailable": ("unavailablemodel", "model_not_found", "model not found", "deprecated"),
        "context_limit": ("context_length_exceeded", "maximum context", "context window"),
        "rate_limit": ("rate_limit", "rate limit", "too many requests"),
        "credential_rejected": ("invalid_api_key", "invalid api key", "authentication_error", "unauthorized"),
        "quota": ("insufficient_quota", "insufficient balance", "quota exceeded"),
        "tool_protocol": ("tool_call", "tool call", "function_call", "function call"),
        "reasoning_protocol": ("reasoning_content", "thinking"),
        "invalid_parameter": ("invalid_request", "invalid parameter", "unsupported parameter", "invalid_argument"),
        "content_filter": ("content_filter", "content filter", "sensitive content"),
        "capacity": ("overloaded", "no available channel", "service unavailable"),
    }
    result["reported_signals"] = [name for name, patterns in categories.items() if any(p in message for p in patterns)]
    result["mentioned_fields"] = [
        field
        for field in (
            "reasoning_content",
            "tool_calls",
            "tool_call_id",
            "tool_choice",
            "parallel_tool_calls",
            "max_tokens",
            "temperature",
            "thinking",
            "messages",
            "response_format",
        )
        if field in message
    ]
    return result


class GatewayPolicy:
    def __init__(self, *, model, mode, scientific_tools, max_calls=24, temperature=0.1, max_tokens=4096, thinking=None):
        if mode not in {"science_agent", "science_expert"}:
            raise ValueError("unsupported mode")
        self.model = model
        self.mode = mode
        self.scientific_tools = frozenset(scientific_tools)
        self.max_calls = max_calls
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking = thinking

    def apply(self, payload):
        result = copy.deepcopy(payload)
        if result.get("model") != self.model:
            raise ValueError("model identity mismatch")
        original = result.get("tools", [])
        if not isinstance(original, list) or any(t.get("type") != "function" for t in original):
            raise ValueError("only function tools supported")
        names = [t["function"]["name"] for t in original]
        if len(names) != len(set(names)):
            raise ValueError("duplicate function identity")
        if self.mode == "science_agent":
            allowed = {"mcp__venusfactory__" + n for n in self.scientific_tools}
            kept = [t for t in original if t["function"]["name"] in allowed]
        else:
            # Role-output schemas are internal graph outputs, not scientific actions.
            # They remain visible; scientific execution is constrained by the MCP adapter.
            kept = original
        if kept:
            result["tools"] = kept
        else:
            result.pop("tools", None)
        choice = result.get("tool_choice")
        if isinstance(choice, dict) and choice.get("function", {}).get("name") not in {
            t["function"]["name"] for t in kept
        }:
            raise ValueError("selected tool is outside the experiment interface")
        if not kept:
            result.pop("tool_choice", None)
            result.pop("parallel_tool_calls", None)
        if self.thinking is not None:
            result["thinking"] = {"type": self.thinking, "clear_thinking": True}
            result.pop("reasoning_effort", None)
        result["temperature"] = self.temperature
        result["max_tokens"] = self.max_tokens
        result.pop("max_completion_tokens", None)
        # Avoid an asymmetric hidden nucleus/seed setting. Different system prompts
        # remain an intentional architecture difference, recorded in the protocol.
        result.pop("top_p", None)
        result.pop("seed", None)
        encoded = json.dumps(result, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
        metadata = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "tools_before": names,
            "tools_sent": [t["function"]["name"] for t in kept],
            "tool_schema_sha256": hashlib.sha256(json.dumps(kept, sort_keys=True).encode()).hexdigest(),
            "request_sha256": hashlib.sha256(encoded).hexdigest(),
            "request_bytes": len(encoded),
            "stream": bool(result.get("stream")),
            "thinking_sent": result.get("thinking"),
            "reasoning_effort_sent": result.get("reasoning_effort"),
            "parallel_tool_calls_sent": result.get("parallel_tool_calls"),
        }
        return result, metadata


class ModelGateway:
    def __init__(self, *, policy, upstream_url, api_key, output, transport=None, timeout_s=95):
        self.policy = policy
        self.upstream_url = upstream_url.rstrip("/")
        self._api_key = api_key
        self.output = Path(output)
        self.token = secrets.token_urlsafe(32)
        self.calls = 0
        self.transport = transport
        self.timeout_s = timeout_s
        self.runner = None
        self.client = None
        self.base_url = None
        self._active_requests = set()
        self._closing = False

    def record(self, value):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.output.open("a") as f:
            f.write(json.dumps(value, ensure_ascii=False) + "\n")

    async def record_upstream_rejection(self, response, call):
        raw = bytearray()
        truncated = False
        async for chunk in response.aiter_bytes():
            remaining = 65536 - len(raw)
            raw.extend(chunk[:remaining])
            if len(chunk) > remaining:
                truncated = True
                break
        evidence = {
            "event": "upstream_rejection",
            "call": call,
            "status": response.status_code,
            **classify_upstream_error(bytes(raw), truncated=truncated),
        }
        self.record(evidence)
        return evidence

    async def handle(self, request):
        if not secrets.compare_digest(request.headers.get("Authorization", ""), "Bearer " + self.token):
            return web.json_response({"error": {"message": "relay authentication failed"}}, status=401)
        if request.method == "GET" and request.path == "/v1/models":
            return web.json_response(
                {
                    "object": "list",
                    "data": [{"id": self.policy.model, "object": "model", "owned_by": "configured-provider"}],
                }
            )
        if request.method != "POST" or request.path != "/v1/chat/completions":
            return web.json_response({"error": {"message": "unsupported relay route"}}, status=404)
        try:
            payload, metadata = self.policy.apply(await request.json())
        except (ValueError, KeyError, TypeError):
            self.record({"event": "policy_rejected"})
            return web.json_response({"error": {"message": "research request policy rejected"}}, status=400)
        if self.calls >= self.policy.max_calls:
            self.record({"event": "budget_rejected", "max_calls": self.policy.max_calls})
            return web.json_response({"error": {"message": "research model call budget exhausted"}}, status=429)
        self.calls += 1
        call = self.calls
        started = time.monotonic()
        self.record({"event": "request", "call": call, "total_deadline_s": self.timeout_s, **metadata})
        status = None
        usage = None
        completed = False
        error_type = None
        try:
            async with (
                asyncio.timeout(self.timeout_s),
                self.client.stream(
                    "POST",
                    self.upstream_url + "/chat/completions",
                    json=payload,
                    headers={"Authorization": "Bearer " + self._api_key},
                ) as response,
            ):
                status = response.status_code
                if status >= 400:
                    await self.record_upstream_rejection(response, call)
                    return web.json_response(
                        {
                            "error": {
                                "message": "configured provider rejected request",
                                "code": "upstream_http_" + str(status),
                            }
                        },
                        status=status,
                    )
                if not payload.get("stream"):
                    raw = await response.aread()
                    data = json.loads(raw)
                    usage = data.get("usage")
                    if not data.get("choices") or not all(c.get("finish_reason") for c in data["choices"]):
                        raise ValueError("provider response has no completed choice")
                    completed = True
                    return web.Response(body=raw, content_type="application/json", status=status)
                result = web.StreamResponse(
                    status=status, headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
                )
                await result.prepare(request)
                pending = b""
                saw_done = False
                finished_choices = set()
                async for chunk in response.aiter_bytes():
                    pending += chunk
                    # Inspect only usage, without saving streamed contents/reasoning.
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                        if line.startswith(b"data:"):
                            data = line[5:].strip()
                            if data == b"[DONE]":
                                saw_done = True
                            elif data:
                                obj = json.loads(data)
                                if obj.get("error"):
                                    raise ValueError("provider stream contained an error")
                                if obj.get("usage"):
                                    usage = obj["usage"]
                                for choice in obj.get("choices", []):
                                    if choice.get("finish_reason"):
                                        finished_choices.add(choice.get("index", 0))
                    if len(pending) > 8_000_000:
                        raise ValueError("provider event too large")
                    await result.write(chunk)
                if not saw_done or not finished_choices:
                    raise ValueError("provider stream ended without completion")
                await result.write_eof()
                completed = True
                return result
        except asyncio.CancelledError:
            error_type = "CancelledError"
            self.record({"event": "transport_error", "call": call, "error_type": error_type})
            raise
        except (httpx.HTTPError, ValueError, OSError, TimeoutError) as exc:
            error_type = type(exc).__name__
            self.record({"event": "transport_error", "call": call, "error_type": type(exc).__name__})
            if "result" in locals() and result.prepared:
                result.force_close()
                return result
            return web.json_response({"error": {"message": "research relay transport failure"}}, status=502)
        finally:
            self.record(
                {
                    "event": "response",
                    "call": call,
                    "status": status,
                    "completed": completed,
                    "error_type": error_type,
                    "total_deadline_s": self.timeout_s,
                    "usage": usage,
                    "wall_seconds": round(time.monotonic() - started, 3),
                }
            )

    async def _tracked_handle(self, request):
        if self._closing:
            return web.json_response({"error": "relay is closing"}, status=503)
        task = asyncio.current_task()
        self._active_requests.add(task)
        try:
            return await self.handle(request)
        finally:
            self._active_requests.discard(task)

    async def __aenter__(self):
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(None, connect=min(15, self.timeout_s)), transport=self.transport, trust_env=False
        )
        app = web.Application(client_max_size=8_000_000)
        app.router.add_route("*", "/{tail:.*}", self._tracked_handle)
        self.runner = web.AppRunner(app, access_log=None, shutdown_timeout=1)
        try:
            await self.runner.setup()
            site = web.TCPSite(self.runner, "127.0.0.1", 0)
            await site.start()
            self.base_url = "http://127.0.0.1:" + str(site._server.sockets[0].getsockname()[1]) + "/v1"
            return self
        except BaseException:
            await self.__aexit__(None, None, None)
            raise

    async def _close(self):
        self._closing = True
        # aiohttp shutdown may return before cancelled request handlers finish.
        # Drain their finally blocks before the host takes its terminal audit.
        pending = tuple(self._active_requests)
        for task in pending:
            task.cancel()
        try:
            await asyncio.gather(*pending, return_exceptions=True)
        finally:
            try:
                if self.runner:
                    await self.runner.cleanup()
            finally:
                if self.client:
                    await self.client.aclose()

    async def __aexit__(self, *_):
        cleanup = asyncio.create_task(self._close())
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await cleanup
            raise
