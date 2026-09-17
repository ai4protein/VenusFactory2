"""Runtime LLM config from ``model-config.yaml``.

Supports multiple named endpoints (dmx / gpt / claude / …). Switch with
``active: <id>``. The file is re-read on every call so changing endpoint,
model, or base_url takes effect on the next page load or LLM request.

Search order:
  1. ``$VENUS_MODEL_CONFIG``
  2. repo-root ``model-config.yaml`` (local; gitignored)
  3. ``src/agent/model-config.yaml``
  4. repo-root ``model-config.yaml.example`` (committed template)
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from logger import get_logger

_logger = get_logger("agent.model_config")

_ENV_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class EndpointSpec:
    id: str
    model: str
    label: str
    provider: str
    base_url: str
    api_compatible: str = "openai"
    api_key: str = ""
    api_key_env: str = ""
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    expert_model: str = ""

    def to_public_dict(self, *, has_key: bool = False) -> dict[str, Any]:
        return {
            "id": self.id,
            "model": self.model,
            "label": self.label,
            "provider": self.provider,
            "base_url": self.base_url,
            "api_compatible": self.api_compatible,
            "api_key_env": self.api_key_env,
            "has_key": has_key,
        }


@dataclass(frozen=True)
class ModelRuntimeConfig:
    """Resolved active endpoint. Edit ``model-config.yaml`` to change."""

    endpoint_id: str = "dmx"
    model: str = "glm-4-flash"
    label: str = "GLM 4 Flash"
    provider: str = "dmx"
    base_url: str = "https://www.dmxapi.cn/v1"
    api_compatible: str = "openai"
    api_key: str = ""
    api_key_env: str = "DMXAPI_API_KEY"
    temperature: float = 0.2
    max_tokens: int = 8192
    code_max_tokens: int = 10000
    expert_model: str = ""
    endpoints: tuple[EndpointSpec, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.expert_model:
            object.__setattr__(self, "expert_model", self.model)

    def resolve_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            from_env = os.getenv(self.api_key_env, "").strip()
            if from_env:
                return from_env
        return (
            os.getenv("CHAT_API_KEY", "").strip()
            or os.getenv("DMXAPI_API_KEY", "").strip()
            or os.getenv("OPENAI_API_KEY", "").strip()
            or os.getenv("ANTHROPIC_API_KEY", "").strip()
            or os.getenv("GOOGLE_API_KEY", "").strip()
            or os.getenv("DEEPSEEK_API_KEY", "").strip()
        )

    def model_ids(self) -> set[str]:
        return {item for item in (self.model, self.expert_model) if item}

    def endpoint_ids(self) -> tuple[str, ...]:
        return tuple(ep.id for ep in self.endpoints)


def _expand(value: str) -> str:
    def _repl(match: re.Match[str]) -> str:
        return os.getenv(match.group(1), "")

    return _ENV_PLACEHOLDER.sub(_repl, value)


def _config_paths() -> list[Path]:
    paths: list[Path] = []
    override = (os.getenv("VENUS_MODEL_CONFIG") or "").strip()
    if override:
        paths.append(Path(override).expanduser())
    paths.append(_REPO_ROOT / "model-config.yaml")
    paths.append(_PACKAGE_DIR / "model-config.yaml")
    paths.append(_REPO_ROOT / "model-config.yaml.example")
    return paths


def find_model_config_path() -> Optional[Path]:
    for path in _config_paths():
        if path.is_file():
            return path
    return None


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_endpoint(endpoint_id: str, raw: dict[str, Any], defaults: dict[str, Any]) -> EndpointSpec:
    model = str(raw.get("model") or defaults.get("model") or "glm-4-flash").strip()
    return EndpointSpec(
        id=str(endpoint_id).strip(),
        model=model,
        label=str(raw.get("label") or defaults.get("label") or model).strip(),
        provider=str(raw.get("provider") or defaults.get("provider") or endpoint_id).strip(),
        base_url=str(raw.get("base_url") or defaults.get("base_url") or "").strip().rstrip("/"),
        api_compatible=str(raw.get("api_compatible") or defaults.get("api_compatible") or "openai").strip()
        or "openai",
        api_key=_expand(str(raw.get("api_key") or defaults.get("api_key") or "")).strip(),
        api_key_env=str(raw.get("api_key_env") or defaults.get("api_key_env") or "").strip(),
        temperature=_as_float(raw["temperature"], 0.2) if "temperature" in raw else None,
        max_tokens=_as_int(raw["max_tokens"], 8192) if "max_tokens" in raw else None,
        expert_model=str(raw.get("expert_model") or model).strip(),
    )


def _parse_raw(raw: dict[str, Any]) -> ModelRuntimeConfig:
    data = raw or {}
    endpoints_raw = data.get("endpoints")
    parsed_endpoints: list[EndpointSpec] = []
    active_ep: Optional[EndpointSpec] = None

    if isinstance(endpoints_raw, dict) and endpoints_raw:
        for key, value in endpoints_raw.items():
            if not isinstance(value, dict):
                continue
            parsed_endpoints.append(_parse_endpoint(str(key), value, data))
        requested = str(data.get("active") or "").strip()
        if requested:
            active_ep = next((ep for ep in parsed_endpoints if ep.id == requested), None)
            if active_ep is None:
                _logger.warning(
                    "model-config.yaml active=%r is not in endpoints %s; using first",
                    requested,
                    [ep.id for ep in parsed_endpoints],
                )
        if active_ep is None:
            active_ep = parsed_endpoints[0]
    else:
        # Legacy flat file: treat the whole mapping as one endpoint.
        fallback_id = str(data.get("provider") or data.get("active") or "default").strip() or "default"
        active_ep = _parse_endpoint(fallback_id, data, {})
        parsed_endpoints = [active_ep]

    temperature = (
        active_ep.temperature
        if active_ep.temperature is not None
        else _as_float(data.get("temperature", 0.2), 0.2)
    )
    max_tokens = (
        active_ep.max_tokens
        if active_ep.max_tokens is not None
        else _as_int(data.get("max_tokens", 8192), 8192)
    )
    return ModelRuntimeConfig(
        endpoint_id=active_ep.id,
        model=active_ep.model,
        label=active_ep.label,
        provider=active_ep.provider,
        base_url=active_ep.base_url,
        api_compatible=active_ep.api_compatible,
        api_key=active_ep.api_key,
        api_key_env=active_ep.api_key_env,
        temperature=temperature,
        max_tokens=max(1, max_tokens),
        code_max_tokens=max(1, _as_int(data.get("code_max_tokens", 10000), 10000)),
        expert_model=str(data.get("expert_model") or active_ep.expert_model or active_ep.model).strip(),
        endpoints=tuple(parsed_endpoints),
    )


def load_model_config() -> ModelRuntimeConfig:
    """Read ``model-config.yaml`` from disk. Not cached — every call is fresh."""
    path = find_model_config_path()
    if path is None:
        return ModelRuntimeConfig()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            _logger.warning("model-config.yaml at %s is not a mapping; using defaults", path)
            return ModelRuntimeConfig()
        return _parse_raw(raw)
    except Exception:
        _logger.exception("Failed to read model-config.yaml at %s; using defaults", path)
        return ModelRuntimeConfig()


def get_expert_model_id() -> str:
    cfg = load_model_config()
    return cfg.expert_model or cfg.model


def applies_to_model(model_id: str | None) -> bool:
    mid = (model_id or "").strip()
    if not mid:
        return True
    return mid in load_model_config().model_ids()


def apply_to_llm(llm: Any) -> None:
    """Push the current yaml (model / base_url / key) onto an LLM instance."""
    if llm is None:
        return
    cfg = load_model_config()
    name = str(getattr(llm, "model_name", "") or "")
    uses_runtime = bool(getattr(llm, "_uses_runtime_config", False))
    if name and not uses_runtime and name not in cfg.model_ids():
        return
    llm.model_name = cfg.model
    if cfg.base_url:
        llm.base_url = cfg.base_url
    key = cfg.resolve_api_key()
    if key:
        llm.api_key = key
    llm._uses_runtime_config = True


def runtime_model_public_fields() -> dict[str, Any]:
    """Fields safe to expose on ``/api/models`` (no secrets)."""
    cfg = load_model_config()
    endpoints = []
    for ep in cfg.endpoints:
        key_present = bool(ep.api_key) or bool(ep.api_key_env and os.getenv(ep.api_key_env, "").strip())
        if ep.id == cfg.endpoint_id:
            key_present = key_present or bool(cfg.resolve_api_key())
        endpoints.append(ep.to_public_dict(has_key=key_present))
    return {
        "expert_model": cfg.expert_model or cfg.model,
        "runtime_model": cfg.model,
        "runtime_base_url": cfg.base_url,
        "runtime_provider": cfg.provider,
        "runtime_label": cfg.label,
        "runtime_endpoint": cfg.endpoint_id,
        "runtime_endpoints": endpoints,
        "config_path": str(find_model_config_path() or ""),
    }


__all__ = [
    "EndpointSpec",
    "ModelRuntimeConfig",
    "load_model_config",
    "find_model_config_path",
    "get_expert_model_id",
    "applies_to_model",
    "apply_to_llm",
    "runtime_model_public_fields",
]
