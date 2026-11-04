"""Bound a live chat's tool dispatch and reserve a final evidence-based answer."""
from __future__ import annotations

import copy
import json


class ChatToolBudget:
    def __init__(self, limit):
        if type(limit) is not int or limit < 1:
            raise ValueError("Invalid chat tool limit")
        self.limit = limit
        self.used = 0

    def prepare(self, payload, *, final_call=False):
        payload = copy.deepcopy(payload)
        remaining = max(0, self.limit - self.used)
        finalizing = remaining == 0 or final_call
        if finalizing:
            for key in ("tools", "tool_choice", "parallel_tool_calls"):
                payload.pop(key, None)
            instruction = (
                "Tool exploration is finished for this turn. Give your final answer now from "
                "the tool results already received. Do not request more tools or claim that "
                "unperformed computations succeeded. State any remaining limitations briefly."
            )
        else:
            instruction = (
                f"At most {remaining} tool calls remain in this turn, including search and describe. "
                "Use a small number of targeted searches, then answer. Do not enumerate every "
                "category for a general capability question. A parallel batch must fit this limit."
            )
        messages = payload.setdefault("messages", [])
        system = next((m for m in messages if m.get("role") == "system"), None)
        if system is not None and isinstance(system.get("content"), str):
            system["content"] += "\n\n" + instruction
        else:
            messages.insert(0, {"role": "system", "content": instruction})
        return payload, {"tool_calls_remaining": remaining, "answer_only": finalizing}

    def deliver(self, raw, streaming):
        remaining = max(0, self.limit - self.used)
        if not streaming:
            obj = json.loads(raw)
            total = sum(len(c.get("message", {}).get("tool_calls") or []) for c in obj.get("choices", []))
            if total and not remaining:
                raise ValueError("Tool budget exhausted")
            kept = 0
            for choice in obj.get("choices", []):
                message = choice.get("message", {})
                calls = message.get("tool_calls") or []
                selected = calls[:max(0, remaining - kept)]
                if calls:
                    message["tool_calls"] = selected
                kept += len(selected)
            result = json.dumps(obj, ensure_ascii=False).encode()
        else:
            lines = raw.splitlines(keepends=True)
            objects, groups = {}, {}
            for i, line in enumerate(lines):
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if not data or data == b"[DONE]":
                    continue
                obj = objects[i] = json.loads(data)
                for choice in obj.get("choices", []):
                    for call in choice.get("delta", {}).get("tool_calls") or []:
                        key = (choice.get("index", 0), call.get("index", 0))
                        groups.setdefault(key, len(groups))
            total = len(groups)
            if total and not remaining:
                raise ValueError("Tool budget exhausted")
            kept = min(total, remaining)
            allowed = set(list(groups)[:kept])
            for i, obj in objects.items():
                for choice in obj.get("choices", []):
                    delta = choice.get("delta", {})
                    if "tool_calls" in delta:
                        delta["tool_calls"] = [
                            c for c in delta["tool_calls"]
                            if (choice.get("index", 0), c.get("index", 0)) in allowed
                        ]
                lines[i] = b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n"
            result = b"".join(lines)
        self.used += kept
        return result, {"admitted": kept, "not_dispatched": total - kept, "used": self.used}
