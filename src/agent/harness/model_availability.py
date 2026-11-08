"""Classify terminal model-service failures from trusted gateway receipts."""
import json
from pathlib import Path

_MESSAGES = {
    "quota": "The selected model quota is exhausted. Research is saved; continue when access is restored.",
    "authentication": "The selected model credentials were rejected. Research is saved; continue after configuration is restored.",
    "access": "The selected model service denied access. Research is saved; continue when access is restored.",
    "rate_limit": "The selected model service is rate-limited. Research is saved; continue when it is available.",
    "configuration": "The selected model route is unavailable. Research is saved; continue after configuration is restored.",
}


def model_service_failure(status, signals=()):
    """Static diagnostics for an observed HTTP response, without its payload."""
    if type(status) is not int or status not in {401, 402, 403, 404, 429}:
        return None
    code = ("quota" if status == 402 or "quota" in signals else
            "authentication" if status == 401 else
            "rate_limit" if status == 429 else
            "configuration" if status == 404 else "access")
    return {"source": "model_gateway", "code": code, "http_status": status,
            "message": _MESSAGES[code], "resume": "after_service_recovery"}


class ModelServiceUnavailable(ValueError):
    def __init__(self, status):
        self.diagnostic = model_service_failure(status)
        if self.diagnostic is None:
            raise ValueError("Unsupported model-service status")
        super().__init__(self.diagnostic["message"])


def model_service_blocker(path):
    """Only gateway responses count; never parse model prose or tool errors."""
    response = None
    rejections = {}
    try:
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if not isinstance(row, dict):
                    continue
                if row.get("event") == "response":
                    response = row
                elif row.get("event") == "upstream_rejection" and isinstance(row.get("call"), (int, str)):
                    rejections[row.get("call")] = row
    except FileNotFoundError:
        return None
    if not response or response.get("completed") is True:
        return None
    status = response.get("status")
    if type(status) is not int or status not in {401, 402, 403, 404, 429}:
        return None
    call = response.get("call")
    signals = rejections.get(call if isinstance(call, (int, str)) else None, {}).get("reported_signals", [])
    signals = signals if isinstance(signals, list) else []
    return model_service_failure(status, signals)
