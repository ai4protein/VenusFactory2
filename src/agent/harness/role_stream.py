"""Collect a bounded text-only scientific role from the common streaming protocol."""
from __future__ import annotations

import json


async def read_role_stream(response):
    text, model, finish, done, size = [], None, None, False, 0
    async for line in response.aiter_lines():
        size += len(line.encode("utf-8"))
        if size > 2_000_000:
            raise ValueError("Scientific role response is too large")
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            done = True
            break
        if not payload:
            continue
        item = json.loads(payload)
        if not isinstance(item, dict) or item.get("error"):
            raise ValueError("Scientific role stream returned an error")
        model = item.get("model", model)
        for choice in item.get("choices", []):
            if choice.get("index", 0) != 0:
                raise ValueError("Scientific role returned multiple choices")
            delta = choice.get("delta") or {}
            if delta.get("tool_calls") or delta.get("function_call"):
                raise ValueError("Scientific roles may return complete text only")
            if delta.get("role", "assistant") != "assistant":
                raise ValueError("Invalid scientific role response")
            content = delta.get("content")
            if content is not None:
                if not isinstance(content, str):
                    raise ValueError("Scientific role response is not text")
                text.append(content)
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    if not done or not finish:
        raise ValueError("Scientific role stream ended without completion")
    return {"model": model, "choices": [{"finish_reason": finish,
            "message": {"role": "assistant", "content": "".join(text)}}]}
