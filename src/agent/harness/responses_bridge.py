"""Explicit native Responses / upstream Chat Completions bridge.

Model identity, authorization, usage and request budgets remain upstream Chat
Completions facts. Native tools execute only after a complete validated response.
"""
from __future__ import annotations

import json
import time

from aiohttp import web

from .contracts import digest
from .messages_bridge import MessagesBridge, chat_to_message
from .responses_gateway import ResponsesPolicy, canonical_function, inspect_responses

TRANSPORT = "shared_chat_responses_free_bridge.v1"


def validate_profile(profile):
    if (profile.model.protocol != "chat_completions"
            or profile.harness not in {"codex", "expert-codex"}
            or profile.request_recovery != "none"
            or profile.budget.temperature is None
            or profile.budget.reasoning_effort is not None):
        raise ValueError("Codex Chat bridge requires explicit Chat controls and no implicit recovery")


def text_content(value):
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise ValueError("Text content required")
    parts = []
    for part in value:
        if (not isinstance(part, dict) or part.get("type") not in {"input_text", "output_text"}
                or not isinstance(part.get("text"), str)):
            raise ValueError("The Codex Chat bridge supports text and MCP tool results")
        parts.append(part["text"])
    return "\n".join(parts)


def responses_to_chat(body, policy):
    native_policy = ResponsesPolicy(
        model=policy.model, scientific_tools=policy.scientific_tools,
        max_calls=policy.max_calls, temperature=policy.temperature,
        max_output_tokens=policy.max_tokens,
    )
    filtered, _ = native_policy.apply(body)
    text_config = filtered.get("text", {})
    if not isinstance(text_config, dict):
        raise ValueError("Invalid native text configuration")
    format_config = text_config.get("format", {"type": "text"})
    if format_config != {"type": "text"}:
        raise ValueError("Native structured-output formats require a separate compatible bridge")
    messages = []
    instructions = filtered.get("instructions")
    if instructions is not None:
        if not isinstance(instructions, str):
            raise ValueError("Text instructions required")
        if instructions:
            messages.append({"role": "system", "content": instructions})
    items = filtered["input"]
    if isinstance(items, str):
        items = [{"role": "user", "content": items}]
    pending, completed, seen_calls = set(), set(), set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Invalid native input item")
        kind = item.get("type", "message")
        if kind == "message":
            if pending:
                raise ValueError("Unresolved tool calls before the next message")
            role = item.get("role")
            if role not in {"system", "developer", "user", "assistant"}:
                raise ValueError("Invalid message role")
            messages.append({"role": "system" if role == "developer" else role,
                             "content": text_content(item.get("content"))})
        elif kind == "function_call":
            name = canonical_function(item.get("name"), item.get("namespace"))
            cid = item.get("call_id")
            arguments = item.get("arguments")
            if (name not in native_policy.allowed or not isinstance(cid, str) or not cid
                    or cid in seen_calls or not isinstance(arguments, str)
                    or not isinstance(json.loads(arguments), dict)):
                raise ValueError("Invalid historical function call")
            json.dumps(json.loads(arguments), allow_nan=False)
            if not messages or messages[-1]["role"] != "assistant":
                if pending:
                    raise ValueError("Interleaved parallel tool results")
                messages.append({"role": "assistant", "content": None})
            messages[-1].setdefault("tool_calls", []).append({
                "id": cid, "type": "function", "function": {"name": name, "arguments": arguments},
            })
            pending.add(cid)
            seen_calls.add(cid)
        elif kind == "function_call_output":
            cid = item.get("call_id")
            if not isinstance(cid, str) or cid not in pending or cid in completed:
                raise ValueError("Unpaired or duplicated tool output")
            messages.append({"role": "tool", "tool_call_id": cid, "content": text_content(item.get("output"))})
            pending.remove(cid)
            completed.add(cid)
        else:
            raise ValueError("Unsupported native history item")
    if pending or not messages:
        raise ValueError("Complete stateless message history required")
    tools, identities = [], {}
    for tool in filtered.get("tools", []):
        namespace = tool.get("name") if tool["type"] == "namespace" else None
        for child in tool["tools"] if namespace else [tool]:
            name = canonical_function(child["name"], namespace)
            definition = {k: child[k] for k in ("description", "parameters", "strict") if k in child}
            definition["name"] = name
            tools.append({"type": "function", "function": definition})
            identities[name] = {"name": child["name"], **({"namespace": namespace} if namespace else {})}
    payload = {"model": policy.model, "messages": messages, "stream": False}
    if tools:
        payload["tools"] = tools
        choice = filtered.get("tool_choice", "auto")
        payload["tool_choice"] = ({"type": "function", "function": {
            "name": canonical_function(choice["name"], choice.get("namespace"))}}
            if isinstance(choice, dict) else choice)
        if "parallel_tool_calls" in filtered:
            payload["parallel_tool_calls"] = filtered["parallel_tool_calls"]
    return payload, {
        "native_protocol": "responses", "upstream_protocol": "chat_completions",
        "native_stream": body.get("stream", False), "native_request_sha256": digest(body),
        # Identifiers only; this mapping is also used for round-trip namespace identity.
        "native_tool_identities": identities,
    }


def chat_to_response(raw, *, model, allowed_tools, max_tokens, upstream_url, identities):
    message, usage = chat_to_message(
        raw, model=model, allowed_tools=allowed_tools, max_tokens=max_tokens, upstream_url=upstream_url,
    )
    output = []
    for index, part in enumerate(message["content"]):
        item_id = ("msg_" if part["type"] == "text" else "fc_") + digest([message["id"], index])[:32]
        if part["type"] == "text":
            output.append({"type": "message", "id": item_id, "role": "assistant", "status": "completed",
                           "content": [{"type": "output_text", "text": part["text"], "annotations": []}]})
        else:
            identity = identities.get(part["name"])
            if identity is None:
                raise ValueError("Response tool lacks a matching native declaration")
            output.append({"type": "function_call", "id": item_id, "status": "completed",
                           "call_id": part["id"], **identity,
                           "arguments": json.dumps(part["input"], ensure_ascii=False, allow_nan=False)})
    response = {
        "id": "resp_" + digest(message["id"])[:32], "object": "response",
        "created_at": int(time.time()), "status": "completed", "error": None,
        "incomplete_details": None, "model": model, "output": output,
        "usage": {"input_tokens": usage["prompt_tokens"], "output_tokens": usage["completion_tokens"],
                  "total_tokens": usage["total_tokens"]},
        "parallel_tool_calls": True, "store": False,
    }
    return response, usage


def response_sse(response):
    events = []
    def emit(kind, **value):
        events.append({"type": kind, "sequence_number": len(events), **value})
    initial = {**response, "output": [], "status": "in_progress", "usage": None}
    emit("response.created", response=initial)
    emit("response.in_progress", response=initial)
    for index, item in enumerate(response["output"]):
        initial_item = {**item, "status": "in_progress"}
        if item["type"] == "message":
            initial_item["content"] = []
        else:
            initial_item["arguments"] = ""
        emit("response.output_item.added", output_index=index, item=initial_item)
        if item["type"] == "message":
            for content_index, part in enumerate(item["content"]):
                common = {"item_id": item["id"], "output_index": index, "content_index": content_index}
                emit("response.content_part.added", **common, part={**part, "text": ""})
                emit("response.output_text.delta", **common, delta=part["text"])
                emit("response.output_text.done", **common, text=part["text"])
                emit("response.content_part.done", **common, part=part)
        else:
            common = {"item_id": item["id"], "output_index": index}
            emit("response.function_call_arguments.delta", **common, delta=item["arguments"])
            emit("response.function_call_arguments.done", **common, arguments=item["arguments"])
        emit("response.output_item.done", output_index=index, item=item)
    emit("response.completed", response=response)
    return "".join("event: " + row["type"] + "\ndata: " +
                   json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n\n" for row in events).encode()


class ResponsesChatBridge(MessagesBridge):
    native_route = "/v1/responses"
    transport_name = TRANSPORT

    def translate_request(self, body):
        return responses_to_chat(body, self.policy)

    def translate_response(self, chat_raw, metadata, native):
        response, usage = chat_to_response(
            chat_raw, model=self.policy.model, allowed_tools=set(metadata["tools_sent"]),
            max_tokens=self.policy.max_tokens, upstream_url=self.upstream_url,
            identities=native["native_tool_identities"],
        )
        body = (response_sse(response) if native["native_stream"]
                else json.dumps(response, ensure_ascii=False, allow_nan=False).encode())
        inspect_responses(body, native["native_stream"], model=self.policy.model,
                          allowed_tools=set(metadata["tools_sent"]), max_output_tokens=self.policy.max_tokens)
        return body, usage

    @staticmethod
    def error(kind, message, status):
        return web.json_response({"error": {"type": kind, "message": message}}, status=status)
