"""Content-free completion evidence for provider responses."""
from __future__ import annotations
import hashlib
import json

_REASONS = {"stop", "length", "tool_calls", "function_call", "content_filter", "error"}

def response_evidence(raw, streaming):
    result = {"response_sha256": hashlib.sha256(raw).hexdigest(),
              "response_bytes": len(raw), "streaming": bool(streaming),
              "done": False, "malformed": False, "provider_error": False,
              "content_present": False, "tools_present": False,
              "usage_present": False, "finish_reasons": []}
    reasons = set()
    frames = (line[5:].strip() for line in raw.splitlines() if line.startswith(b"data:")) if streaming else [raw]
    for frame in frames:
        if frame == b"[DONE]":
            result["done"] = True
            continue
        if not frame:
            continue
        try:
            obj = json.loads(frame)
            if not isinstance(obj, dict):
                result["malformed"] = True
                continue
            result["provider_error"] |= bool(obj.get("error"))
            result["usage_present"] |= isinstance(obj.get("usage"), dict)
            for choice in obj.get("choices") or []:
                if not isinstance(choice, dict):
                    result["malformed"] = True
                    continue
                reason = choice.get("finish_reason")
                if reason is not None:
                    reasons.add(reason if isinstance(reason, str) and reason in _REASONS else "unknown")
                message = choice.get("delta" if streaming else "message") or {}
                if isinstance(message, dict):
                    result["content_present"] |= bool(message.get("content"))
                    result["tools_present"] |= bool(message.get("tool_calls") or message.get("function_call"))
                else:
                    result["malformed"] = True
        except (ValueError, TypeError, UnicodeError):
            result["malformed"] = True
    result["finish_reasons"] = sorted(reasons)
    return result
