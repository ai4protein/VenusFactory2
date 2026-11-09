"""JSON encoding shared by model API event streams."""
import json
import math
from typing import Any


def _finite_json(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _finite_json(item) for key, item in value.items()}
    return value


def encode_event(event: str, data: dict[str, Any], *, ensure_ascii: bool = False) -> str:
    """Keep missing numeric values JSON-compatible without mutating the result."""
    payload = json.dumps(_finite_json(data), ensure_ascii=ensure_ascii, allow_nan=False)
    return f"event: {event}\ndata: {payload}\n\n"
