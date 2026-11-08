"""Normalize complete native Claude events without double-counting partial SSE."""

from __future__ import annotations

import json
import math

from .messages_bridge import PREFIX, _object, _text


class ClaudeEvents:
    def __init__(self, allowed_tools, limit, validator=None):
        self.allowed = {PREFIX + name for name in allowed_tools}
        self.limit, self.validator = limit, validator
        self.session_id = None
        self.started, self.finished = {}, set()
        self.final_text = ""
        self.completed, self.failed = False, False
        self.tool_errors = 0
        self.event_ids = set()

    def consume(self, event):
        event = _object(event)
        if self.completed or self.failed:
            raise ValueError("Native event after terminal result")
        kind = event.get("type")
        if kind == "system":
            if event.get("subtype") == "init":
                sid = event.get("session_id")
                if self.session_id is not None or not isinstance(sid, str) or not sid:
                    raise ValueError("Invalid native session initialization")
                self.session_id = sid
            elif (
                event.get("subtype") != "status"
                or self.session_id is None
                or event.get("session_id") != self.session_id
            ):
                raise ValueError("Unknown native system event")
            return []
        if self.session_id is None or event.get("session_id") != self.session_id:
            raise ValueError("Native session identity mismatch")
        if kind == "tool_progress":
            # Claude emits heartbeats during long MCP calls. Their parent is the
            # existing call, not a delegated agent or a new executable action.
            parent = event.get("parent_tool_use_id")
            cid = parent if event.get("heartbeat") is True else event.get("tool_use_id")
            elapsed = event.get("elapsed_time_seconds")
            if (cid not in self.started or cid in self.finished
                    or event.get("tool_name") != self.started[cid]
                    or (parent is not None and event.get("heartbeat") is not True)
                    or type(elapsed) not in (int, float)
                    or not math.isfinite(elapsed) or elapsed < 0):
                raise ValueError("Invalid native tool progress")
            return []
        if event.get("parent_tool_use_id") is not None:
            raise ValueError("Nested native execution has no admitted contract")
        if kind == "stream_event":
            # Complete assistant/user events are the sole scientific action basis.
            if not isinstance(event.get("event"), dict):
                raise ValueError("Malformed partial native event")
            return []
        if kind == "result":
            if (
                event.get("is_error") is not False
                or event.get("subtype") != "success"
                or event.get("permission_denials")
                or event.get("safety_stops")
            ):
                self.failed = True
                return []
            if self.started.keys() != self.finished:
                raise ValueError("Terminal result with outstanding tool calls")
            if not isinstance(event.get("result"), str):
                raise ValueError("Missing native final text")
            self.final_text = event["result"]
            self.completed = True
            return []
        if kind not in {"assistant", "user"}:
            raise ValueError("Unsupported native action")
        uid = event.get("uuid")
        if not isinstance(uid, str) or not uid or uid in self.event_ids:
            raise ValueError("Missing or duplicate complete native event identity")
        self.event_ids.add(uid)
        message = _object(event.get("message"))
        if message.get("role") != kind or not isinstance(message.get("content"), list):
            raise ValueError("Malformed complete native message")
        rows = []
        for part in message["content"]:
            part = _object(part)
            if part.get("type") == "text":
                text = _text([part])
                if kind == "assistant":
                    self.final_text = text
                continue
            if kind == "assistant" and part.get("type") == "tool_use":
                cid, name = part.get("id"), part.get("name")
                if not isinstance(cid, str) or not cid or cid in self.started or name not in self.allowed:
                    raise ValueError("Undeclared or repeated native tool")
                args = _object(part.get("input"))
                json.dumps(args, allow_nan=False)
                if len(self.started) >= self.limit:
                    raise ValueError("Native tool budget exceeded")
                self.started[cid] = name
                rows.append({"kind": "tool_call_start", "tool_call_id": cid, "tool_name": name, "args": args})
            elif kind == "user" and part.get("type") == "tool_result":
                cid = part.get("tool_use_id")
                if cid not in self.started or cid in self.finished:
                    raise ValueError("Unpaired or repeated native tool result")
                error = part.get("is_error", False)
                if type(error) is not bool:
                    raise ValueError("Invalid native tool error flag")
                text = _text(part.get("content", ""))
                if not error and self.validator:
                    data = _object(json.loads(text))
                    self.validator(self.started[cid][len(PREFIX) :], data)
                self.finished.add(cid)
                self.tool_errors += int(error)
                rows.append({"kind": "tool_result", "tool_call_id": cid, "is_error": error, "output": text})
            else:
                raise ValueError("Unsupported native message block")
        return rows
