"""Result metadata and failure detection for computational tools."""

import json
from typing import Any


def result_patch(**values):
    # Existing HTTP consumers read value/visible; no UI framework is involved.
    return values


def ignore_progress(*args, **kwargs):
    pass


def tool_output_indicates_failure(raw_output: Any) -> tuple[bool, str]:
    """Detect if tool output indicates failure: top-level success:false, or nested result/data with success:false or error (e.g. BLAST timeout in result string). Returns (is_failure, error_message)."""
    if raw_output is None:
        return (False, "")
    text = str(raw_output).strip()
    if not text:
        return (False, "")
    # Top-level parse
    try:
        parsed = json.loads(text) if isinstance(raw_output, str) else raw_output
    except Exception:
        parsed = None
    if not isinstance(parsed, dict):
        return (False, "")

    # 1) Top-level success is False
    if parsed.get("success") is False:
        err = parsed.get("error") or parsed.get("message") or str(parsed)
        return (True, err[:500] if isinstance(err, str) else str(err)[:500])

    # 1b) Top-level status is "error" (e.g. UniProt download tools)
    if parsed.get("status") == "error":
        err_obj = parsed.get("error")
        if isinstance(err_obj, dict):
            err = err_obj.get("message") or err_obj.get("type") or str(err_obj)
        else:
            err = err_obj or parsed.get("message") or str(parsed)
        return (True, (err[:500] if isinstance(err, str) else str(err)[:500]))

    # 2) Top-level success is True but nested payload indicates failure (e.g. result/data/output as JSON string or dict)
    for key in ("result", "data", "output", "response", "body"):
        val = parsed.get(key)
        if val is None:
            continue
        if isinstance(val, dict):
            if val.get("success") is False:
                err = val.get("error") or val.get("message") or str(val)
                return (True, (err[:500] if isinstance(err, str) else str(err)[:500]))
            if val.get("error"):
                return (True, str(val.get("error"))[:500])
        if isinstance(val, str):
            val_strip = val.strip()
            if not val_strip or val_strip[0] not in ("{", "["):
                if "timeout" in val_strip.lower() or "error" in val_strip.lower():
                    return (True, val_strip[:500])
                continue
            try:
                inner = json.loads(val_strip)
                if isinstance(inner, dict) and inner.get("success") is False:
                    err = inner.get("error") or inner.get("message") or val_strip
                    return (True, (err[:500] if isinstance(err, str) else str(err)[:500]))
                if isinstance(inner, dict) and inner.get("error"):
                    return (True, str(inner.get("error"))[:500])
            except Exception:
                if "success\": false" in val_strip or '"success":false' in val_strip or "Timeout" in val_strip:
                    return (True, val_strip[:500])
    return (False, "")
