"""Public failure hints from gateway evidence, without provider bodies or paths."""

import json
from collections import deque
from pathlib import Path


def public_failure(path, fallback="runtime_error"):
    code = fallback
    try:
        with Path(path).open() as log:
            rows = [json.loads(line) for line in deque(log, maxlen=64)]
    except (OSError, ValueError):
        rows = []
    for row in reversed(rows):
        if row.get("event") == "response":
            if row.get("completed"):
                shape = row.get("response_completion")
                if (
                    row.get("status") == 200 and isinstance(shape, dict)
                    and shape.get("malformed") is False and shape.get("provider_error") is False
                    and shape.get("content_present") is False and shape.get("tools_present") is False
                    and shape.get("finish_reasons") in (["stop"], ["length"])
                    and (shape.get("streaming") is False or shape.get("done") is True)
                ):
                    code = "model_response_empty"
                break
            status = row.get("status")
            if status == 429:
                code = "upstream_rate_limited"
            elif status in {401, 403}:
                code = f"model_access_{status}"
            elif status is not None and status != 200:
                code = "model_unavailable"
            elif row.get("error_type") in {"TimeoutError", "ReadTimeout", "ConnectTimeout", "PoolTimeout"}:
                code = "upstream_timeout"
            else:
                code = "model_response_invalid"
            break
        if row.get("event") == "budget_rejected":
            code = "model_call_budget_exhausted"
            break
        if row.get("event") == "policy_rejected":
            code = "model_protocol_incompatible"
            break
    messages = {
        "model_call_budget_exhausted": "This run reached its model request limit.",
        "upstream_rate_limited": "Model provider returned 429 (rate limited). Please try again later.",
        "upstream_timeout": "Model response timed out. Please retry.",
        "model_unavailable": "Model temporarily unavailable. Choose another model or retry later.",
        "model_access_401": "Model provider authentication failed (401).",
        "model_access_403": "Model provider access denied (403).",
        "model_response_invalid": "Model returned an incomplete response. Please retry.",
        "model_response_empty": "The model returned no answer. Please retry or choose another model.",
        "model_protocol_incompatible": "Model request is incompatible with the selected harness.",
    }
    return {"code": code, "message": messages.get(code, "Agent turn did not complete: " + fallback)}


def research_failure(path, error):
    """Explain generic native failures without replacing specific research errors.

    This is diagnostic only; service-block and retry policy remain separate.
    """
    message = str(error)
    result = {"error_type": type(error).__name__, "message": message[:400]}
    if type(error) is RuntimeError and message in {
        "runtime_error", "protocol_failed", "incomplete", "timeout", "Native turn failed"
    }:
        hint = public_failure(path, fallback=message)
        if hint["code"] != message:
            result.update(code=hint["code"], message=hint["message"])
    return result
