"""Compact contract errors with the root cause retained after redaction."""
from __future__ import annotations

import re
from pathlib import Path

from .session_boundary import public_value


def bounded_message(details, workspace, limit=1600):
    text = public_value(str(details), workspace)
    if len(text) > limit:
        marker = "\n... [trace shortened] ...\n"
        head = min(240, limit//4)
        text = text[:head] + marker + text[-(limit-head-len(marker)):]
    return text


def contract_error(exc, workspace, limit=1600):
    from pydantic import ValidationError

    if isinstance(exc, ValidationError):
        details = "; ".join(
            ".".join(map(str, item["loc"])) + ": " + item["msg"]
            for item in exc.errors(include_input=False, include_url=False)[:8])
    else:
        details = str(exc)
    text = bounded_message(details, workspace, limit)
    return type(exc).__name__ + (": " + text if text else "")


def _source_location(details, workspace, source_file, source_text):
    """Map a generated runtime traceback back to its unchanged authored prefix."""
    if not source_file or not isinstance(source_text, str):
        return None
    source_lines = source_text.splitlines()
    allowed = {"runtime.py", "/workspace/runtime.py", str(Path(workspace) / "runtime.py")}
    frames = re.finditer(r'^\s*File "([^"]+)", line (\d+)(?:, in [^\n]+)?$', str(details), re.MULTILINE)
    locations = [int(match[2]) for match in frames
                 if match[1] in allowed and 1 <= int(match[2]) <= len(source_lines)]
    if not locations:
        return None
    number = locations[-1]
    # Read the displayed line from our source snapshot, not stderr-supplied code.
    return f"At {source_file}:{number}: {source_lines[number-1].strip()[:200]}"


def python_failure(details, workspace, limit=800, *, source_file=None, source_text=None):
    """Keep the final cause and, when known, one authored line; full logs stay in receipts."""
    lines = str(details).strip().splitlines()
    cause = str(details)
    for index in range(len(lines) - 1, -1, -1):
        if re.match(r"^[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception):", lines[index]):
            cause = chr(10).join(lines[index:])
            break
    location = _source_location(details, workspace, source_file, source_text)
    if location and limit >= 160:
        location = bounded_message(location, workspace, min(260, limit//3))
        return bounded_message(cause, workspace, limit-len(location)-1) + "\n" + location
    return bounded_message(cause, workspace, limit)
