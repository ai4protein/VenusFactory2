"""Model identity checks for research runs; credentials never enter public records."""

from __future__ import annotations

import tomllib
from pathlib import Path
from urllib.parse import urlsplit


class ModelBindingError(ValueError):
    pass


def public_identity(config: dict) -> dict:
    selected = config.get("default_model")
    models = config.get("models", {})
    if not selected or selected not in models:
        raise ModelBindingError("Kimi must have an explicit configured default model")
    model = models[selected]
    provider = config.get("providers", {}).get(model.get("provider"))
    if not isinstance(provider, dict) or not model.get("model"):
        raise ModelBindingError("Selected model must resolve to an explicit provider")
    endpoint = str(provider.get("base_url") or "").rstrip("/")
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ModelBindingError("Provider endpoint must be an HTTP(S) URL without embedded credentials or query")
    allowed = ("provider", "model", "max_context_size", "max_output_size", "capabilities", "reasoning_key")
    return {
        "configured_id": selected,
        **{k: model[k] for k in allowed if k in model},
        "base_url": endpoint,
        "provider_type": provider.get("type"),
        "snapshot_pinned": False,
        "snapshot_note": "Provider model alias; immutable snapshot not verified.",
    }


def read_model_configuration(
    path: Path, *, expected: dict | None = None, required: dict | None = None
) -> tuple[str, dict]:
    text = path.read_text()
    identity = public_identity(tomllib.loads(text))
    for constraints in (expected, required):
        if constraints:
            for key, value in constraints.items():
                if key not in identity or identity[key] != value:
                    # Do not interpolate values: a malformed config may contain secrets.
                    raise ModelBindingError(f"Research model binding mismatch in field {key}")
    return text, identity
