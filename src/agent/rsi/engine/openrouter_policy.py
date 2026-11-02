"""Text-only OpenRouter free routes with explicit price and fallback constraints."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from decimal import Decimal, InvalidOperation

BASE_URL = "https://openrouter.ai/api/v1"
FREE_ROUTER = "openrouter/free"
POLICY = "openrouter_free_text.v1"
PROVIDER = {
    "allow_fallbacks": False,
    "require_parameters": True,
    "max_price": {"prompt": 0, "completion": 0, "request": 0, "image": 0},
}
FIELDS = {
    "model",
    "messages",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "stream",
    "stream_options",
    "temperature",
    "max_tokens",
    "thinking",
    "reasoning",
    "reasoning_effort",
    "include_reasoning",
    "response_format",
    "stop",
    "frequency_penalty",
    "presence_penalty",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "user",
    "metadata",
    "provider",
    "usage",
}


def is_openrouter(base_url):
    return isinstance(base_url, str) and base_url.rstrip("/") == BASE_URL


def free_model_id(model):
    return isinstance(model, str) and (
        model == FREE_ROUTER or bool(re.fullmatch(r"[a-z0-9._-]+/[a-z0-9._-]+:free", model))
    )


def zero_price(value):
    if isinstance(value, bool) or value is None:
        return False
    try:
        number = Decimal(str(value))
        return number.is_finite() and number == 0
    except (InvalidOperation, ValueError):
        return False


def free_catalog(payload):
    """Admit only explicit free routes with every reported price exactly zero."""
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("Malformed OpenRouter model directory")
    selected = []
    for entry in payload["data"]:
        if not isinstance(entry, dict) or not free_model_id(entry.get("id")):
            continue
        prices = entry.get("pricing")
        parameters = entry.get("supported_parameters", [])
        if (
            not isinstance(prices, dict)
            or not {"prompt", "completion"} <= prices.keys()
            or not all(zero_price(value) for value in prices.values())
            or not isinstance(parameters, list)
            or "tools" not in parameters
        ):
            continue
        selected.append(
            {
                key: entry[key]
                for key in ("id", "context_length", "pricing", "supported_parameters", "top_provider")
                if key in entry
            }
        )
    if len({entry["id"] for entry in selected}) != len(selected):
        raise ValueError("Duplicate OpenRouter model identities")
    return sorted(selected, key=lambda entry: entry["id"])


def enforce_free_request(payload, metadata):
    """Run after common budget/tool filtering; witness the actual transmitted body."""
    if not isinstance(payload, dict) or not free_model_id(payload.get("model")):
        raise ValueError("Only explicit OpenRouter free models are allowed")
    if set(payload) - FIELDS:
        raise ValueError("Unapproved OpenRouter extension or fallback field")
    if "provider" in payload and payload["provider"] != PROVIDER:
        raise ValueError("Caller cannot override zero-price provider routing")
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise ValueError("OpenRouter text messages required")
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Malformed OpenRouter message")
        content = message.get("content")
        if isinstance(content, list):
            if any(
                not isinstance(part, dict)
                or set(part) != {"type", "text"}
                or part["type"] != "text"
                or not isinstance(part["text"], str)
                for part in content
            ):
                raise ValueError("Only plain text content is admitted on free research routes")
        elif content is not None and not isinstance(content, str):
            raise ValueError("Only text content is admitted on free research routes")
        if set(message) - {
            "role",
            "content",
            "name",
            "tool_calls",
            "tool_call_id",
            "reasoning",
            "reasoning_details",
            "refusal",
        }:
            raise ValueError("Unapproved message extension")
    result = copy.deepcopy(payload)
    thinking = result.pop("thinking", None)
    if thinking is not None:
        if not isinstance(thinking, dict) or thinking.get("type") not in {"enabled", "disabled"}:
            raise ValueError("Unsupported thinking setting")
        result["reasoning"] = {"enabled": thinking["type"] == "enabled"}
        result.pop("reasoning_effort", None)
    result["provider"] = copy.deepcopy(PROVIDER)
    result["usage"] = {"include": True}
    encoded = json.dumps(result, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    metadata = {
        **metadata,
        "free_route": {"policy": POLICY, "provider": result["provider"]},
        "thinking_sent": None,
        "reasoning_sent": result.get("reasoning"),
        "request_sha256": hashlib.sha256(encoded).hexdigest(),
        "request_bytes": len(encoded),
    }
    return result, metadata


def verify_free_response(raw, streaming, model):
    """Keep the provider's actual model and billed cost as response evidence."""
    if streaming:
        objects = []
        for line in raw.splitlines():
            if line.startswith(b"data:"):
                value = line[5:].strip()
                if value and value != b"[DONE]":
                    objects.append(json.loads(value))
    else:
        objects = [json.loads(raw)]
    reported = sorted({item["model"] for item in objects if isinstance(item.get("model"), str)})
    if not reported:
        raise ValueError("OpenRouter response lacks model identity")
    if model != FREE_ROUTER and not set(reported) <= {model, model.removesuffix(":free")}:
        raise ValueError("OpenRouter returned a different model")
    usages = [item["usage"] for item in objects if isinstance(item.get("usage"), dict)]
    if not usages or any(not zero_price(usage.get("cost")) for usage in usages):
        raise ValueError("OpenRouter response did not attest zero billed cost")
    return {"policy": POLICY, "requested_model": model, "reported_models": reported, "billed_cost": 0.0}


def enforce_free_responses_request(payload, metadata):
    """Explicit Responses routing with the same zero-price constraints as Chat."""
    payload = copy.deepcopy(payload)
    payload.pop("client_metadata", None)  # Native client telemetry is not part of inference.
    # OpenRouter owns its own caching and exposes plain reasoning, not OpenAI
    # encrypted reasoning blocks or native-client reasoning summaries.
    payload.pop("prompt_cache_key", None)
    payload.pop("include", None)
    payload.pop("parallel_tool_calls", None)
    if isinstance(payload.get("reasoning"), dict):
        payload["reasoning"].pop("summary", None)
        if not payload["reasoning"]:
            payload.pop("reasoning")

    fields = {
        "model",
        "input",
        "instructions",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "stream",
        "store",
        "background",
        "max_output_tokens",
        "temperature",
        "reasoning",
        "text",
        "include",
        "metadata",
        "provider",
        "prompt_cache_key",
        "service_tier",
        "safety_identifier",
    }
    if not isinstance(payload, dict) or not free_model_id(payload.get("model")) or set(payload) - fields:
        raise ValueError("Only explicit free Responses routes are allowed")
    if payload.get("provider", PROVIDER) != PROVIDER or payload.get("store") or payload.get("background"):
        raise ValueError("Free stateless routing cannot be overridden")
    if payload.get("service_tier") not in {None, "auto", "default"}:
        raise ValueError("Paid priority service is unavailable")
    result = copy.deepcopy(payload)
    result.pop("service_tier", None)
    result["provider"] = copy.deepcopy(PROVIDER)
    encoded = json.dumps(result, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    return result, {
        **metadata,
        "free_route": {"policy": POLICY, "provider": result["provider"]},
        "request_sha256": hashlib.sha256(encoded).hexdigest(),
        "request_bytes": len(encoded),
    }


def verify_free_responses_response(raw, streaming, model):
    if streaming:
        rows = [
            json.loads(line[5:].strip())
            for line in raw.splitlines()
            if line.startswith(b"data:") and line[5:].strip() not in {b"", b"[DONE]"}
        ]
        values = [row.get("response", {}) for row in rows if row.get("type") == "response.completed"]
    else:
        values = [json.loads(raw)]
    if len(values) != 1 or values[0].get("status") != "completed":
        raise ValueError("Missing completed Responses evidence")
    value = values[0]
    if value.get("model") not in {model, model.removesuffix(":free")}:
        raise ValueError("OpenRouter returned a different model")
    if not zero_price(value.get("usage", {}).get("cost")):
        raise ValueError("OpenRouter response did not attest zero billed cost")
    return {"policy": POLICY, "requested_model": model, "reported_models": [value["model"]], "billed_cost": 0.0}
