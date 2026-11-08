"""Compact contract errors with the root cause retained after redaction."""
from __future__ import annotations

import re

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


def python_failure(details, workspace, limit=800):
    """Expose the final exception; the complete execution log remains in its receipt."""
    lines = str(details).strip().splitlines()
    for index in range(len(lines) - 1, -1, -1):
        if re.match(r"^[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception):", lines[index]):
            return bounded_message(chr(10).join(lines[index:]), workspace, limit)
    return bounded_message(details, workspace, limit)
