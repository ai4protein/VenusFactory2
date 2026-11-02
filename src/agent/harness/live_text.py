"""Incremental public text for Web chat, independent of native tool execution.

Only assistant text is previewed. Complete provider responses still pass the
gateway's normal validation before they reach a harness or execute tools.
"""

from __future__ import annotations

import json


class PublicTextStream:
    def __init__(self, callback, call, *, protocol="chat"):
        self.callback = callback
        self.call = call
        self.protocol = protocol
        self.pending = b""
        self.data = []
        self.metadata = {}
        self.choices = {}
        self.response = None

    async def feed(self, chunk):
        self.pending += chunk
        while b"\n" in self.pending:
            line, self.pending = self.pending.split(b"\n", 1)
            line = line.rstrip(b"\r")
            if line.startswith(b"data:"):
                self.data.append(line[5:].lstrip(b" "))
            elif not line and self.data:
                data, self.data = b"\n".join(self.data), []
                if data != b"[DONE]":
                    await self.event(json.loads(data))

    async def emit(self, text):
        if isinstance(text, str) and text:
            await self.callback(self.call, text)

    async def event(self, event):
        if not isinstance(event, dict) or event.get("error"):
            raise ValueError("Invalid provider text event")
        if self.protocol == "responses":
            if event.get("type") in {"response.output_text.delta", "response.refusal.delta"}:
                await self.emit(event.get("delta"))
            elif event.get("type") == "response.completed":
                self.response = event.get("response")
            return
        self.metadata.update({k: v for k, v in event.items() if k != "choices"})
        for choice in event.get("choices", []):
            index = choice.get("index", 0)
            current = self.choices.setdefault(index, {"index": index, "message": {"role": "assistant"}, "finish_reason": None})
            message = current["message"]
            delta = choice.get("delta") or {}
            if delta.get("role"):
                message["role"] = delta["role"]
            for field in ("content", "refusal"):
                value = delta.get(field)
                if value is not None:
                    if not isinstance(value, str):
                        raise ValueError("Unsupported public text delta")
                    message[field] = message.get(field, "") + value
                    if index == 0:
                        await self.emit(value)
            # Keep tool arguments internal; they are delivered only after validation.
            for call in delta.get("tool_calls") or []:
                tools = current.setdefault("tools", {})
                target = tools.setdefault(call["index"], {"type": "function", "function": {}})
                for field in ("id", "type"):
                    if call.get(field):
                        target[field] = call[field]
                for field, value in call.get("function", {}).items():
                    if value is not None:
                        target["function"][field] = target["function"].get(field, "") + value
            if choice.get("finish_reason"):
                current["finish_reason"] = choice["finish_reason"]

    def completion(self):
        if self.protocol == "responses":
            if not self.response:
                raise ValueError("Missing completed response")
            return json.dumps(self.response, ensure_ascii=False).encode()
        choices = []
        for _, item in sorted(self.choices.items()):
            if not item["finish_reason"]:
                raise ValueError("Missing completed chat choice")
            message = dict(item["message"])
            message.setdefault("content", None)
            if item.get("tools"):
                message["tool_calls"] = [value for _, value in sorted(item["tools"].items())]
            choices.append({"index": item["index"], "message": message, "finish_reason": item["finish_reason"]})
        if not choices:
            raise ValueError("Missing chat choices")
        return json.dumps({**self.metadata, "object": "chat.completion", "choices": choices}, ensure_ascii=False).encode()
