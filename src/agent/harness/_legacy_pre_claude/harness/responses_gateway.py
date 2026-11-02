"""Bounded Responses transport; no Chat Completions translation or retry.

The supported subset is stateless text/function-tool Responses over HTTP/SSE.
Complete provider evidence is validated before any bytes reach the harness.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import re
import time

import httpx
from aiohttp import web

from agent.rsi.live_model_gateway import ModelGateway

from .contracts import digest

TRANSPORT = "buffered_responses_v1"
REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"}


def canonical_function(name, namespace=None):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_]+", name):
        raise ValueError("Invalid function identity")
    if namespace is None:
        return name
    if not isinstance(namespace, str) or not re.fullmatch(r"[A-Za-z0-9_]+", namespace):
        raise ValueError("Invalid function namespace")
    if namespace == "mcp__venusfactory":
        return namespace + "__" + name
    # A different namespace cannot spoof a permitted flattened MCP identity.
    return namespace + "." + name


class ResponsesPolicy:
    def __init__(
        self, *, model, scientific_tools, max_calls=24, temperature=None, max_output_tokens=4096, reasoning_effort=None
    ):
        if not isinstance(model, str) or not model:
            raise ValueError("Exact model identity required")
        if type(max_calls) is not int or max_calls < 1 or type(max_output_tokens) is not int or max_output_tokens < 1:
            raise ValueError("Positive integer model budgets required")
        if temperature is not None and (
            type(temperature) not in (int, float) or not math.isfinite(temperature) or not 0 <= temperature <= 2
        ):
            raise ValueError("Invalid temperature")
        if reasoning_effort is not None and reasoning_effort not in REASONING_EFFORTS:
            raise ValueError("Unsupported explicit reasoning effort")
        self.model, self.max_calls = model, max_calls
        self.temperature, self.max_output_tokens = temperature, max_output_tokens
        self.reasoning_effort = reasoning_effort
        self.allowed = {"mcp__venusfactory__" + name for name in scientific_tools}

    @classmethod
    def from_profile(cls, profile, tools):
        if profile.model.protocol != "responses" or profile.request_recovery != "none":
            raise ValueError("Responses requires an explicit protocol and no implicit recovery")
        if profile.budget.thinking is not None:
            raise ValueError("Set thinking=null and choose an explicit Responses reasoning_effort")
        return cls(
            model=profile.model.model,
            scientific_tools=tools,
            max_calls=profile.budget.model_calls,
            temperature=profile.budget.temperature,
            max_output_tokens=profile.budget.output_tokens,
            reasoning_effort=profile.budget.reasoning_effort,
        )

    def apply(self, payload):
        if not isinstance(payload, dict) or payload.get("model") != self.model:
            raise ValueError("Model identity mismatch")
        if "messages" in payload or "max_tokens" in payload or "thinking" in payload:
            raise ValueError("Chat Completions fields cannot be translated implicitly")
        if not isinstance(payload.get("input"), (str, list)):
            raise ValueError("Explicit stateless input required")
        if any(payload.get(key) is not None for key in ("previous_response_id", "conversation")):
            raise ValueError("Unbound server-side history is not supported")
        for key in ("stream", "store", "background"):
            if key in payload and type(payload[key]) is not bool:
                raise ValueError("Boolean transport controls required")
        if payload.get("store") or payload.get("background"):
            raise ValueError("Only stateless foreground execution is supported")
        result = copy.deepcopy(payload)
        tools = result.get("tools", [])
        if not isinstance(tools, list) or any(not isinstance(tool, dict) for tool in tools):
            raise ValueError("Invalid tool declarations")
        names, kept, sent = [], [], set()
        for tool in tools:
            if tool.get("type") == "function":
                identity = canonical_function(tool.get("name"))
                names.append(identity)
                if identity in self.allowed:
                    kept.append(tool)
                    sent.add(identity)
            elif tool.get("type") == "namespace":
                children = tool.get("tools")
                namespace = tool.get("name")
                canonical_function("validate_namespace", namespace)
                if (
                    namespace is None
                    or not isinstance(children, list)
                    or any(not isinstance(child, dict) for child in children)
                ):
                    raise ValueError("Invalid namespace tool declaration")
                selected = []
                for child in children:
                    if child.get("type") == "function":
                        identity = canonical_function(child.get("name"), namespace)
                        names.append(identity)
                        if identity in self.allowed:
                            selected.append(child)
                            sent.add(identity)
                if selected:
                    kept.append({**tool, "tools": selected})
        if len(names) != len(set(names)):
            raise ValueError("Invalid or duplicate function identity")
        choice = result.get("tool_choice", "auto")
        if isinstance(choice, dict):
            if (
                not {"type", "name"} <= set(choice)
                or set(choice) - {"type", "name", "namespace"}
                or choice["type"] != "function"
                or canonical_function(choice["name"], choice.get("namespace")) not in sent
            ):
                raise ValueError("Selected tool is outside the admitted interface")
        elif choice not in {"auto", "none", "required"} or (choice == "required" and not kept):
            raise ValueError("Invalid tool choice")
        if kept:
            result["tools"] = kept
        else:
            result.pop("tools", None)
            result.pop("tool_choice", None)
            result.pop("parallel_tool_calls", None)
        if self.temperature is None:
            result.pop("temperature", None)
        else:
            result["temperature"] = self.temperature
        reasoning = result.get("reasoning", {})
        if not isinstance(reasoning, dict) or set(reasoning) - {"effort", "summary"}:
            raise ValueError("Unsupported reasoning controls")
        if reasoning.get("summary") not in {None, "auto", "concise", "detailed"}:
            raise ValueError("Invalid reasoning summary control")
        if self.reasoning_effort is None:
            reasoning.pop("effort", None)
        else:
            reasoning["effort"] = self.reasoning_effort
        if reasoning:
            result["reasoning"] = reasoning
        else:
            result.pop("reasoning", None)
        result["max_output_tokens"] = self.max_output_tokens
        result["store"] = False
        result.pop("top_p", None)
        result.pop("seed", None)
        encoded = json.dumps(
            result, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
        metadata = {
            "model": self.model,
            "protocol": "responses",
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "reasoning_effort_sent": self.reasoning_effort,
            "tools_sent": sorted(sent),
            "tool_schema_sha256": digest(kept),
            "request_sha256": hashlib.sha256(encoded).hexdigest(),
            "request_bytes": len(encoded),
            "stream": result.get("stream", False),
            "store": False,
        }
        return result, metadata


def _usage(value, max_output_tokens):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Malformed usage")
    out = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        number = value.get(key)
        if type(number) is not int or number < 0:
            raise ValueError("Malformed token counts")
        out[key] = number
    if out["total_tokens"] != out["input_tokens"] + out["output_tokens"] or out["output_tokens"] > max_output_tokens:
        raise ValueError("Usage exceeds the declared output budget or is inconsistent")
    for key, field in (("input_tokens_details", "cached_tokens"), ("output_tokens_details", "reasoning_tokens")):
        detail = value.get(key)
        if detail is not None:
            if not isinstance(detail, dict) or type(detail.get(field)) is not int or detail[field] < 0:
                raise ValueError("Malformed token detail")
            if detail[field] > out["input_tokens" if key.startswith("input") else "output_tokens"]:
                raise ValueError("Token detail exceeds its total")
            out[key] = {field: detail[field]}
    return out


def _completed(response, *, model, allowed_tools, max_output_tokens):
    if (
        not isinstance(response, dict)
        or response.get("status") != "completed"
        or response.get("error")
        or response.get("incomplete_details")
    ):
        raise ValueError("Provider response is incomplete or failed")
    if not isinstance(response.get("id"), str) or not response["id"] or response.get("model") != model:
        raise ValueError("Provider response identity mismatch")
    output = response.get("output")
    if not isinstance(output, list) or not output:
        raise ValueError("Provider returned no output")
    item_ids, calls, meaningful = set(), set(), False
    for item in output:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("id"), str)
            or not item["id"]
            or item["id"] in item_ids
        ):
            raise ValueError("Invalid or repeated output item identity")
        item_ids.add(item["id"])
        if item.get("type") == "function_call":
            if canonical_function(item.get("name"), item.get("namespace")) not in allowed_tools or item.get(
                "status"
            ) not in {None, "completed"}:
                raise ValueError("Undeclared or incomplete function call")
            cid = item.get("call_id")
            if not isinstance(cid, str) or not cid or cid in calls or not isinstance(item.get("arguments"), str):
                raise ValueError("Invalid function-call identity")
            if not isinstance(json.loads(item["arguments"]), dict):
                raise ValueError("Function arguments must be a JSON object")
            calls.add(cid)
            meaningful = True
        elif item.get("type") == "message":
            if (
                item.get("role") != "assistant"
                or item.get("status") != "completed"
                or not isinstance(item.get("content"), list)
            ):
                raise ValueError("Incomplete assistant message")
            for part in item["content"]:
                if not isinstance(part, dict) or part.get("type") not in {"output_text", "refusal"}:
                    raise ValueError("Unsupported message content")
                text = part.get("text" if part["type"] == "output_text" else "refusal")
                if not isinstance(text, str):
                    raise ValueError("Invalid message text")
                meaningful |= bool(text.strip())
        elif item.get("type") != "reasoning":
            raise ValueError("Unsupported provider tool or output type")
    if not meaningful:
        raise ValueError("Response contains no answer or function call")
    return _usage(response.get("usage"), max_output_tokens)


def _sse(raw):
    text = raw.decode("utf-8").replace("\r\n", "\n")
    if not text.endswith("\n\n"):
        raise ValueError("Unterminated provider SSE event")
    for block in text.split("\n\n"):
        fields = block.splitlines()
        data = "\n".join(line[5:].lstrip(" ") for line in fields if line.startswith("data:"))
        event = [line[6:].strip() for line in fields if line.startswith("event:")]
        if not data:
            continue
        if data == "[DONE]":
            yield None
            continue
        obj = json.loads(data)
        if not isinstance(obj, dict) or (event and event != [obj.get("type")]):
            raise ValueError("Inconsistent SSE event type")
        yield obj


def inspect_responses(raw, streaming, *, model, allowed_tools, max_output_tokens):
    options = {"model": model, "allowed_tools": allowed_tools, "max_output_tokens": max_output_tokens}
    if not streaming:
        response = json.loads(raw)
        return {"usage": _completed(response, **options), "response_id_sha256": digest(response["id"])}
    created, completed, done_marker = None, None, False
    added, finished, deltas, text_done, argument_done = {}, {}, {}, {}, {}
    sequence = 0
    auxiliary = {
        "response.in_progress",
        "response.content_part.added",
        "response.content_part.done",
        "response.reasoning_summary_part.added",
        "response.reasoning_summary_part.done",
        "response.reasoning_summary_text.delta",
        "response.reasoning_summary_text.done",
        "response.reasoning_text.delta",
        "response.reasoning_text.done",
    }
    for event in _sse(raw):
        if event is None:
            if completed is None or done_marker:
                raise ValueError("Premature or repeated SSE terminator")
            done_marker = True
            continue
        if (
            completed is not None
            or event.get("sequence_number") != sequence
            or type(event.get("sequence_number")) is not int
        ):
            raise ValueError("Missing, duplicated, reordered or post-terminal event")
        sequence += 1
        kind = event.get("type")
        if kind == "response.created":
            if created is not None or sequence != 1 or not isinstance(event.get("response"), dict):
                raise ValueError("Invalid response start")
            created = event["response"].get("id")
            if not isinstance(created, str) or not created:
                raise ValueError("Missing response identity")
        elif created is None:
            raise ValueError("Missing response.created")
        elif kind == "response.completed":
            completed = event.get("response")
            usage = _completed(completed, **options)
            if completed["id"] != created:
                raise ValueError("Mixed response identities")
        elif kind in {"response.output_item.added", "response.output_item.done"}:
            index, item = event.get("output_index"), event.get("item")
            if type(index) is not int or index < 0 or not isinstance(item, dict):
                raise ValueError("Malformed output item event")
            if kind.endswith("added"):
                if index in added:
                    raise ValueError("Repeated output item")
                added[index] = item
            else:
                if (
                    index not in added
                    or index in finished
                    or item.get("id") != added[index].get("id")
                    or item.get("type") != added[index].get("type")
                ):
                    raise ValueError("Unpaired output item")
                finished[index] = item
        elif kind in {
            "response.output_text.delta",
            "response.refusal.delta",
            "response.function_call_arguments.delta",
            "response.output_text.done",
            "response.refusal.done",
            "response.function_call_arguments.done",
        }:
            index = event.get("output_index")
            if (
                type(index) is not int
                or index not in added
                or index in finished
                or event.get("item_id") != added[index].get("id")
            ):
                raise ValueError("Output event has no active matching item")
            is_arguments = "function_call_arguments" in kind
            if added[index].get("type") != ("function_call" if is_arguments else "message"):
                raise ValueError("Output event disagrees with item type")
            content = None if is_arguments else event.get("content_index")
            if not is_arguments and (type(content) is not int or content < 0):
                raise ValueError("Invalid content index")
            key = (index, content)
            table = argument_done if is_arguments else text_done
            if key in table:
                raise ValueError("Output delta after completion")
            if kind.endswith("delta"):
                if not isinstance(event.get("delta"), str):
                    raise ValueError("Invalid output delta")
                deltas[key] = deltas.get(key, "") + event["delta"]
            else:
                field = "arguments" if is_arguments else "refusal" if "refusal" in kind else "text"
                value = event.get(field)
                if not isinstance(value, str) or deltas.get(key, "") != value:
                    raise ValueError("Output deltas disagree with completed content")
                table[key] = value
        elif kind not in auxiliary:
            raise ValueError("Failed, incomplete or unsupported provider event")
        elif "response" in event and (
            not isinstance(event["response"], dict) or event["response"].get("id") != created
        ):
            raise ValueError("Mixed response identities")
    if completed is None or set(finished) != set(range(len(completed["output"]))) or set(added) != set(finished):
        raise ValueError("Provider stream ended without complete output evidence")
    for index, item in enumerate(completed["output"]):
        if finished[index] != item:
            raise ValueError("Terminal output disagrees with item events")
        if item["type"] == "function_call" and argument_done.get((index, None)) != item["arguments"]:
            raise ValueError("Function arguments were not fully streamed")
        if item["type"] == "message":
            for content_index, part in enumerate(item["content"]):
                value = part["text"] if part["type"] == "output_text" else part["refusal"]
                if text_done.get((index, content_index)) != value:
                    raise ValueError("Message content was not fully streamed")
    return {"usage": usage, "response_id_sha256": digest(created), "stream_events": sequence}


class ResponsesGateway(ModelGateway):
    def __init__(self, *, request_recovery="none", **kwargs):
        if request_recovery != "none":
            raise ValueError("Responses recovery requires its own declared and tested policy")
        super().__init__(**kwargs)

    async def handle(self, request):
        if request.headers.get("Authorization") != "Bearer " + self.token:
            return web.json_response({"error": "invalid relay credential"}, status=401)
        if request.path != "/v1/responses" or request.method != "POST":
            return web.json_response({"error": "unsupported model route"}, status=404)
        try:
            payload, metadata = self.policy.apply(await request.json())
            from .scientist_policy import request_policy_witness

            metadata.update(request_policy_witness(payload))
        except (ValueError, KeyError, TypeError, AttributeError):
            self.record({"event": "policy_rejected"})
            return web.json_response({"error": "request policy rejected"}, status=400)
        if self.calls >= self.policy.max_calls:
            self.record({"event": "budget_rejected"})
            return web.json_response({"error": "model call budget exhausted"}, status=429)
        self.calls += 1
        call, started = self.calls, time.monotonic()
        self.record(
            {"event": "request", "call": call, "total_deadline_s": self.timeout_s, "transport": TRANSPORT, **metadata}
        )
        status, completed, error_type, evidence = None, False, None, {}
        try:
            async with (
                asyncio.timeout(self.timeout_s),
                self.client.stream(
                    "POST",
                    self.upstream_url + "/responses",
                    json=payload,
                    headers={"Authorization": "Bearer " + self._api_key},
                ) as response,
            ):
                status = response.status_code
                if status != 200:
                    await self.record_upstream_rejection(response, call)
                    return web.json_response({"error": "upstream_http_" + str(status)}, status=status)
                raw = bytearray()
                async for chunk in response.aiter_bytes():
                    raw.extend(chunk)
                    if len(raw) > 16_000_000:
                        raise ValueError("Provider response exceeds byte budget")
                evidence = inspect_responses(
                    bytes(raw),
                    payload.get("stream", False),
                    model=self.policy.model,
                    allowed_tools=set(metadata["tools_sent"]),
                    max_output_tokens=self.policy.max_output_tokens,
                )
                evidence["response_sha256"] = hashlib.sha256(raw).hexdigest()
                completed = True
                return web.Response(
                    body=bytes(raw), content_type="text/event-stream" if payload.get("stream") else "application/json"
                )
        except asyncio.CancelledError:
            error_type = "CancelledError"
            self.record({"event": "transport_error", "call": call, "error_type": error_type})
            raise
        except (httpx.HTTPError, ValueError, OSError, TimeoutError, KeyError, TypeError) as error:
            error_type = type(error).__name__
            self.record({"event": "transport_error", "call": call, "error_type": error_type})
            return web.json_response({"error": "incomplete upstream response"}, status=502)
        finally:
            self.record(
                {
                    "event": "response",
                    "call": call,
                    "status": status,
                    "completed": completed,
                    "error_type": error_type,
                    **evidence,
                    "total_deadline_s": self.timeout_s,
                    "wall_seconds": round(time.monotonic() - started, 3),
                }
            )
