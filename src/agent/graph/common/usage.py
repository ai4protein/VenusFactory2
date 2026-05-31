"""LLM usage-token extraction from raw outputs."""
from __future__ import annotations

import json
from typing import Any


def _extract_usage_from_output(raw_output: Any) -> tuple[int, int, int, bool]:
    data = None
    if isinstance(raw_output, dict):
        data = raw_output
    elif isinstance(raw_output, str):
        text = raw_output.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    data = parsed
            except Exception:
                data = None
    if not isinstance(data, dict):
        return 0, 0, 0, True
    usage = data.get("usage")
    if isinstance(usage, dict):
        prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        total = int(usage.get("total_tokens") or (prompt + completion))
        return prompt, completion, total, False
    prompt = int(data.get("prompt_tokens") or data.get("input_tokens") or 0)
    completion = int(data.get("completion_tokens") or data.get("output_tokens") or 0)
    total = int(data.get("total_tokens") or (prompt + completion))
    if prompt == 0 and completion == 0 and total == 0:
        return 0, 0, 0, True
    return prompt, completion, total, False
