"""Experimental text/MCP Messages bridge to an explicitly admitted free Chat API.

This is not a vendor-supported non-Claude backend. Buffer and validate the entire
upstream answer before exposing tool arguments to a native client. Authentication,
free-provider evidence, request accounting and shutdown remain host-owned.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import secrets
import time

import httpx
from aiohttp import web
from .sse_terminal import StreamTerminal
from .provider_config import request_headers, matches_snapshot, local_route

from .contracts import digest
from .gateway import EvidenceGateway, inspect_response

TRANSPORT = "messages_to_free_chat_buffered_v1"
PREFIX = "mcp__venusfactory__"

# Observed in the retained native DMX probe. This maps only a response identity;
# the request and profile keep the admitted free model. No model is substituted.
_RESPONSE_MODEL_ALIASES = {
    ("https://www.dmxapi.cn/v1", "unisound-u2-flash"): frozenset({"u2-flash"}),
}


def _object(value):
    if not isinstance(value, dict):
        raise ValueError("Expected a structured mapping")
    return value


def _text(value):
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise ValueError("Only text content is supported")
    parts = []
    for part in value:
        part = _object(part)
        if part.get("type") != "text" or not isinstance(part.get("text"), str):
            raise ValueError("Unsupported content block")
        if set(part) - {"type", "text", "cache_control"}:
            raise ValueError("Unsupported text-block semantics")
        parts.append(part["text"])
    return "\n".join(parts)


from .live_text import PublicTextStream


def messages_to_chat(body, policy):
    """Strict text/tool subset; no silent model substitution or opaque blocks."""
    body = copy.deepcopy(_object(body))
    if set(body) - {
        "model",
        "messages",
        "system",
        "tools",
        "tool_choice",
        "stream",
        "max_tokens",
        "temperature",
        "thinking",
        "output_config",
        "context_management",
        "metadata",
    }:
        raise ValueError("Unsupported native request field")
    configured_defaults = policy.thinking is None and bool(getattr(policy, 'upstream_url', None)) and local_route(
        policy.model, policy.upstream_url, 'chat_completions'
    ) is not None
    if body.get("model") != policy.model or (policy.thinking != "disabled" and not configured_defaults):
        raise ValueError("Bridge requires the exact admitted model and text/tool output")
    if type(body.get("max_tokens")) is not int or body["max_tokens"] != policy.max_tokens:
        raise ValueError("Native output budget differs from the frozen profile")
    if type(body.get("stream", False)) is not bool:
        raise ValueError("Invalid streaming control")
    if body.get("thinking") not in (None, {"type": "disabled"}):
        raise ValueError("Thinking blocks have no supported round-trip contract")
    if body.get("output_config") not in (None, {}):
        raise ValueError("Unsupported native output controls")
    if body.get("context_management") not in (
        None,
        {},
        {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
    ):
        raise ValueError("Unsupported context management")
    tools = body.get("tools", [])
    if not isinstance(tools, list):
        raise ValueError("Malformed native tool list")
    allowed = {PREFIX + n for n in policy.scientific_tools}
    names, converted_tools = set(), []
    for tool in tools:
        tool = _object(tool)
        name = tool.get("name")
        if not isinstance(name, str) or name not in allowed or name in names:
            raise ValueError("Undeclared or duplicate native tool")
        if set(tool) - {"name", "description", "input_schema", "cache_control"}:
            raise ValueError("Unsupported native tool semantics")
        if not isinstance(tool.get("input_schema"), dict):
            raise ValueError("Missing tool input schema")
        fn = {"name": name, "parameters": tool["input_schema"]}
        if "description" in tool:
            if not isinstance(tool["description"], str):
                raise ValueError("Malformed tool description")
            fn["description"] = tool["description"]
        converted_tools.append({"type": "function", "function": fn})
        names.add(name)
    messages = []
    if "system" in body:
        messages.append({"role": "system", "content": _text(body["system"])})
    history = body.get("messages")
    if not isinstance(history, list) or not history:
        raise ValueError("Missing native conversation")
    used, pending = set(), set()
    for message in history:
        message = _object(message)
        if set(message) != {"role", "content"} or message["role"] not in {"user", "assistant"}:
            raise ValueError("Unsupported conversation role or fields")
        content = message["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        if not isinstance(content, list) or not content:
            raise ValueError("Empty or invalid native message")
        text, calls = [], []
        for part in content:
            part = _object(part)
            kind = part.get("type")
            if kind == "text":
                if pending:
                    raise ValueError("Text arrived before outstanding tool results")
                text.append(_text([part]))
            elif kind == "tool_use" and message["role"] == "assistant":
                if set(part) - {"type", "id", "name", "input", "cache_control"}:
                    raise ValueError("Unsupported tool-call semantics")
                cid, name = part.get("id"), part.get("name")
                if not isinstance(cid, str) or not cid or cid in used or name not in names:
                    raise ValueError("Invalid or undeclared historical tool identity")
                args = _object(part.get("input"))
                calls.append(
                    {
                        "id": cid,
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args, allow_nan=False)},
                    }
                )
                used.add(cid)
            elif kind == "tool_result" and message["role"] == "user":
                if set(part) - {"type", "tool_use_id", "content", "is_error", "cache_control"}:
                    raise ValueError("Unsupported tool-result semantics")
                cid = part.get("tool_use_id")
                if cid not in pending or text or type(part.get("is_error", False)) is not bool:
                    raise ValueError("Unpaired, reordered or repeated tool result")
                output = _text(part.get("content", ""))
                if part.get("is_error", False):
                    output = json.dumps({"is_error": True, "content": output}, ensure_ascii=False)
                messages.append({"role": "tool", "tool_call_id": cid, "content": output})
                pending.remove(cid)
            else:
                raise ValueError("Unsupported native content type")
        if message["role"] == "assistant":
            if pending:
                raise ValueError("Assistant message before tool results")
            row = {"role": "assistant", "content": "\n".join(text) or None}
            if calls:
                row["tool_calls"] = calls
                pending = {c["id"] for c in calls}
            messages.append(row)
        elif text:
            messages.append({"role": "user", "content": "\n".join(text)})
    if pending:
        raise ValueError("Conversation ends with unfinished tools")
    payload = {"model": policy.model, "messages": messages, "stream": False, "max_tokens": policy.max_tokens}
    if converted_tools:
        payload["tools"] = converted_tools
    choice = body.get("tool_choice")
    if choice is not None:
        choice = _object(choice)
        if set(choice) - {"type", "name", "disable_parallel_tool_use"}:
            raise ValueError("Unsupported tool-choice controls")
        if "disable_parallel_tool_use" in choice:
            if type(choice["disable_parallel_tool_use"]) is not bool:
                raise ValueError("Invalid parallel tool control")
            payload["parallel_tool_calls"] = not choice["disable_parallel_tool_use"]
        kind = choice.get("type")
        if kind == "tool" and choice.get("name") in names:
            payload["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
        elif kind in {"auto", "any", "none"} and "name" not in choice:
            if kind == "any" and not names:
                raise ValueError("Required tool choice without tools")
            payload["tool_choice"] = {"auto": "auto", "any": "required", "none": "none"}[kind]
        else:
            raise ValueError("Invalid native tool choice")
    return payload, {
        "native_request_sha256": digest(body),
        "native_stream": body.get("stream", False),
        "native_tool_names": sorted(names),
        "translation": TRANSPORT,
        "local_only_fields": sorted(set(body) & {"metadata", "context_management", "output_config"}),
        "cache_hints_forwarded": False,
    }


def chat_to_message(raw, *, model, allowed_tools, max_tokens, upstream_url=None):
    data = _object(json.loads(raw))
    choices = data.get("choices")
    aliases = _RESPONSE_MODEL_ALIASES.get((upstream_url, model), frozenset())
    from agent.rsi.engine.openrouter_policy import is_openrouter
    if upstream_url and is_openrouter(upstream_url) and model.endswith(':free'):
        # Same exact alias rule as the preceding zero-cost provider verification.
        aliases = aliases | {model.removesuffix(':free')}
    response_model = data.get("model")
    if (
        data.get("error")
        or not isinstance(response_model, str)
        or (response_model not in aliases | {model} and not matches_snapshot(model, response_model, upstream_url))
        or not isinstance(data.get("id"), str)
        or not data["id"]
    ):
        raise ValueError("Provider response identity mismatch")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("Exactly one complete choice is required")
    choice = _object(choices[0])
    message = _object(choice.get("message"))
    if choice.get("index", 0) != 0 or message.get("role") != "assistant":
        raise ValueError("Malformed provider choice")
    content = []
    text = message.get("content")
    if text is not None:
        if not isinstance(text, str):
            raise ValueError("Unsupported provider content")
        if text:
            content.append({"type": "text", "text": text})
    calls = message.get("tool_calls") or []
    if not isinstance(calls, list):
        raise ValueError("Malformed provider tool calls")
    ids = set()
    for call in calls:
        call = _object(call)
        fn = _object(call.get("function"))
        cid = call.get("id")
        if (
            call.get("type") != "function"
            or fn.get("name") not in allowed_tools
            or not isinstance(cid, str)
            or not cid
            or cid in ids
        ):
            raise ValueError("Unadmitted or duplicated provider tool")
        if not isinstance(fn.get("arguments"), str):
            raise ValueError("Malformed provider tool arguments")
        args = _object(json.loads(fn["arguments"]))
        json.dumps(args, allow_nan=False)
        content.append({"type": "tool_use", "id": cid, "name": fn["name"], "input": args})
        ids.add(cid)
    if choice.get("finish_reason") != ("tool_calls" if calls else "stop") or not content:
        raise ValueError("Truncated, filtered or incomplete provider response")
    if message.get("function_call") or message.get("refusal"):
        raise ValueError("Unsupported provider action or refusal")
    usage = _object(data.get("usage"))
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if type(usage.get(k)) is not int or usage[k] < 0:
            raise ValueError("Missing or malformed provider token evidence")
    if (
        usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]
        or usage["completion_tokens"] > max_tokens
    ):
        raise ValueError("Inconsistent usage or output-budget violation")
    result = {
        "id": "msg_" + digest(data["id"])[:32],
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": "tool_use" if calls else "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": usage["prompt_tokens"], "output_tokens": usage["completion_tokens"]},
    }
    return result, usage


def message_sse(message):
    events = []
    initial = {**message, "content": [], "stop_reason": None, "usage": {**message["usage"], "output_tokens": 0}}
    events.append({"type": "message_start", "message": initial})
    for index, part in enumerate(message["content"]):
        if part["type"] == "text":
            initial_part = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": part["text"]}
        else:
            initial_part = {**part, "input": {}}
            delta = {
                "type": "input_json_delta",
                "partial_json": json.dumps(part["input"], ensure_ascii=False, allow_nan=False),
            }
        events.extend(
            [
                {"type": "content_block_start", "index": index, "content_block": initial_part},
                {"type": "content_block_delta", "index": index, "delta": delta},
                {"type": "content_block_stop", "index": index},
            ]
        )
    events.extend(
        [
            {
                "type": "message_delta",
                "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
                "usage": {"output_tokens": message["usage"]["output_tokens"]},
            },
            {"type": "message_stop"},
        ]
    )
    return "".join(
        "event: " + row["type"] + "\ndata: " + json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n\n"
        for row in events
    ).encode()


class MessagesBridge(EvidenceGateway):
    supports_free_response_evidence = True
    native_route = "/v1/messages"
    transport_name = TRANSPORT

    def __init__(self, *, request_recovery="none", **kwargs):
        if request_recovery != "none":
            raise ValueError("Bridge retries require separate conformance evidence")
        super().__init__(**kwargs)

    async def handle(self, request):
        # Role calls and native Messages consume this same physical request counter.
        if request.path == "/v1/chat/completions":
            return await super().handle(request)
        auth = request.headers.get("Authorization", "")
        key = request.headers.get("x-api-key", "")
        if not (secrets.compare_digest(auth, "Bearer " + self.token) or secrets.compare_digest(key, self.token)):
            return self.error("authentication_error", "Invalid relay credential", 401)
        if request.path != self.native_route or request.method != "POST":
            return self.error("invalid_request_error", "Unsupported relay route", 404)
        from .research_handoff import closed_response
        handoff = closed_response(self.policy, self.record)
        if handoff is not None:
            return handoff
        try:
            payload, native = self.translate_request(await request.json())
            if self.on_text is not None:
                payload["stream"] = True
            payload, metadata = self.policy.apply(payload)
            from .scientist_policy import request_policy_witness

            metadata.update(request_policy_witness(payload))
        except (ValueError, KeyError, TypeError):
            self.record({"event": "policy_rejected"})
            return self.error("invalid_request_error", "Request policy rejected", 400)
        if self._rate_limited:
            self.record({"event": "upstream_retry_suppressed", "status": 429})
            return self.error("api_error", "upstream_http_429", 400)
        handoff = closed_response(self.policy, self.record)
        if handoff is not None:
            return handoff
        if self.calls >= self.policy.max_calls:
            self.record({"event": "budget_rejected", "code": "model_call_budget_exhausted"})
            return self.error("invalid_request_error", "Model call budget exhausted", 400)
        self.calls += 1
        call, started = self.calls, time.monotonic()
        self.record({"event": "request", "call": call, "transport": self.transport_name, **metadata, **native})
        status, usage, completed, error_type, evidence = None, None, False, None, {}
        raw = bytearray()
        try:
            async with (
                asyncio.timeout(self.timeout_s),
                self.client.stream(
                    "POST",
                    self.upstream_url + "/chat/completions",
                    json=payload,
                    headers=request_headers(self.upstream_url, self._api_key),
                ) as response,
            ):
                status = response.status_code
                if status != 200:
                    await self.record_upstream_rejection(response, call)
                    client_status = 400 if status == 429 and (self.on_text is not None or self.stop_on_rate_limit) else status
                    self._rate_limited = client_status != status
                    return self.error("api_error", "upstream_http_" + str(status), client_status)
                preview = PublicTextStream(self.on_text, call) if self.on_text is not None else None
                terminal = StreamTerminal('chat') if payload.get('stream') else None
                async for chunk in response.aiter_bytes():
                    raw.extend(chunk)
                    if len(raw) > 16_000_000:
                        raise ValueError("Provider response exceeds byte budget")
                    if preview is not None:
                        await preview.feed(chunk)
                    if terminal is not None and terminal.feed(chunk):
                        break
                if preview is not None:
                    inspect_response(bytes(raw), True)
                chat_raw = preview.completion() if preview is not None else bytes(raw)
                provider = self.verify_provider_response(bytes(raw), preview is not None)
                from .tool_response_names import normalize
                chat_raw, aliases = normalize(chat_raw, False, payload.get('tools', []))
                if aliases:
                    self.record({'event': 'tool_name_normalized', 'call': call, 'aliases': aliases})
                body, usage = self.translate_response(chat_raw, metadata, native)
                evidence = {
                    "response_sha256": hashlib.sha256(raw).hexdigest(),
                    "translated_response_sha256": hashlib.sha256(body).hexdigest(),
                    "requested_model": self.policy.model,
                    "response_model": json.loads(chat_raw)["model"],
                    "response_model_alias_used": json.loads(chat_raw)["model"] != self.policy.model,
                }
                if provider is not None:
                    evidence["provider_evidence"] = provider
                completed = True
                return web.Response(
                    body=body, content_type="text/event-stream" if native["native_stream"] else "application/json"
                )
        except asyncio.CancelledError:
            error_type = "CancelledError"
            self.record({"event": "transport_error", "call": call, "error_type": error_type})
            raise
        except (httpx.HTTPError, ValueError, KeyError, TypeError, OSError, TimeoutError) as exc:
            error_type = type(exc).__name__
            self.record({"event": "transport_error", "call": call, "error_type": error_type})
            return self.error("api_error", "Incomplete or inadmissible upstream response", 502)
        finally:
            self.record(
                {
                    "event": "response",
                    "call": call,
                    "status": status,
                    "completed": completed,
                    "received_bytes": len(raw),
                    "error_type": error_type,
                    "usage": usage,
                    **evidence,
                    "wall_seconds": round(time.monotonic() - started, 3),
                }
            )

    def translate_request(self, body):
        return messages_to_chat(body, self.policy)

    def translate_response(self, chat_raw, metadata, native):
        message, usage = chat_to_message(
            chat_raw, model=self.policy.model, allowed_tools=set(metadata["tools_sent"]),
            max_tokens=self.policy.max_tokens, upstream_url=self.upstream_url,
        )
        body = (message_sse(message) if native["native_stream"]
                else json.dumps(message, ensure_ascii=False, allow_nan=False).encode())
        return body, usage

    @staticmethod
    def error(kind, message, status):
        return web.json_response({"type": "error", "error": {"type": kind, "message": message}}, status=status)
