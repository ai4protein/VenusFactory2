"""Model registry: load models.yaml + manage API key lookup.

Static metadata lives in ``models.yaml`` (URLs, capabilities, defaults).
User-private API keys live in ``~/.venusfactory/keys.json`` and/or per-provider
env vars (e.g. ``DEEPSEEK_API_KEY``). Keys are never persisted to the yaml.

Lookup precedence for a model's API key:
  1. Key explicitly passed to ``resolve_endpoint(model_id, api_key=...)``.
  2. User key store (~/.venusfactory/keys.json) keyed by provider id.
  3. Environment variable named by the model's ``api_key_env`` field.
  4. Empty string (caller decides how to handle).

Gateway behaviour:
  If ``CHAT_FORCE_GATEWAY=<gateway_id>`` env is set (or ``set_active_gateway``
  is called at runtime), ``resolve_endpoint`` returns the gateway's base_url +
  the gateway's key. The model id passed to the provider is translated via the
  gateway's ``model_aliases`` table when present.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml

from logger import get_logger

_logger = get_logger("agent.model_registry")

_REGISTRY_PATH = Path(__file__).resolve().parent / "models.yaml"
_USER_CONFIG_DIR = Path(
    os.getenv("VENUSFACTORY_CONFIG_DIR") or (Path.home() / ".venusfactory")
)
_KEYS_FILE = _USER_CONFIG_DIR / "keys.json"
_KEYS_LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """Static metadata for a single model."""
    id: str
    label: str
    provider: str
    base_url: str
    api_compatible: str = "openai"  # openai | anthropic
    api_key_env: str = ""
    supports_tool_use: bool = True
    supports_json_schema: bool = False
    supports_prompt_caching: bool = False
    max_context_tokens: int = 128000
    max_output_tokens: int = 8192
    requires_adapter: bool = False
    hidden: bool = False

    def to_public_dict(self) -> dict[str, Any]:
        """JSON-serializable dict for /api/models endpoint (no keys)."""
        return {
            "id": self.id,
            "label": self.label,
            "provider": self.provider,
            "base_url": self.base_url,
            "api_compatible": self.api_compatible,
            "api_key_env": self.api_key_env,
            "supports_tool_use": self.supports_tool_use,
            "supports_json_schema": self.supports_json_schema,
            "supports_prompt_caching": self.supports_prompt_caching,
            "max_context_tokens": self.max_context_tokens,
            "max_output_tokens": self.max_output_tokens,
            "requires_adapter": self.requires_adapter,
        }


@dataclass(frozen=True)
class GatewaySpec:
    id: str
    label: str
    base_url: str
    api_compatible: str = "openai"
    api_key_env: str = ""
    model_aliases: dict[str, str] = field(default_factory=dict)
    extra_models: list[ModelSpec] = field(default_factory=list)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "base_url": self.base_url,
            "api_compatible": self.api_compatible,
            "api_key_env": self.api_key_env,
            "model_aliases": dict(self.model_aliases),
            "extra_models": [m.to_public_dict() for m in self.extra_models],
        }


@dataclass(frozen=True)
class ResolvedEndpoint:
    """The concrete (base_url, api_key, model_id) used to call a provider."""
    model_id: str           # what to send as the "model" field in the HTTP body
    base_url: str
    api_key: str
    api_compatible: str
    via_gateway: Optional[str] = None  # gateway id if routed through one
    spec: Optional[ModelSpec] = None


# ---------------------------------------------------------------------------
# Registry loading (cached)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_yaml() -> dict[str, Any]:
    if not _REGISTRY_PATH.exists():
        _logger.warning("models.yaml not found at %s; returning empty registry", _REGISTRY_PATH)
        return {}
    try:
        return yaml.safe_load(_REGISTRY_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        _logger.exception("Failed to parse models.yaml; returning empty registry")
        return {}


@lru_cache(maxsize=1)
def _load_models() -> dict[str, ModelSpec]:
    raw = _load_yaml()
    out: dict[str, ModelSpec] = {}
    for entry in raw.get("models", []) or []:
        try:
            spec = ModelSpec(**{k: v for k, v in entry.items() if k in ModelSpec.__dataclass_fields__})
            out[spec.id] = spec
        except Exception as e:
            _logger.warning("Skipping malformed model entry %r: %s", entry, e)
    # Gateways' extra_models also resolvable by id
    for gw in raw.get("gateways", []) or []:
        for entry in gw.get("extra_models", []) or []:
            try:
                spec = ModelSpec(**{k: v for k, v in entry.items() if k in ModelSpec.__dataclass_fields__})
                if spec.id not in out:
                    out[spec.id] = spec
            except Exception:
                pass
    return out


@lru_cache(maxsize=1)
def _load_gateways() -> dict[str, GatewaySpec]:
    raw = _load_yaml()
    out: dict[str, GatewaySpec] = {}
    for entry in raw.get("gateways", []) or []:
        try:
            extra_specs: list[ModelSpec] = []
            for m_entry in entry.pop("extra_models", []) or []:
                try:
                    extra_specs.append(
                        ModelSpec(**{k: v for k, v in m_entry.items() if k in ModelSpec.__dataclass_fields__})
                    )
                except Exception:
                    pass
            entry["extra_models"] = extra_specs
            spec = GatewaySpec(**{k: v for k, v in entry.items() if k in GatewaySpec.__dataclass_fields__})
            out[spec.id] = spec
        except Exception as e:
            _logger.warning("Skipping malformed gateway entry %r: %s", entry, e)
    return out


def reload_registry() -> None:
    """Drop caches so the yaml is re-read on next access (for hot reload / tests)."""
    _load_yaml.cache_clear()
    _load_models.cache_clear()
    _load_gateways.cache_clear()


# ---------------------------------------------------------------------------
# Public registry queries
# ---------------------------------------------------------------------------


def get_default_model_id() -> str:
    raw = _load_yaml()
    return raw.get("default_model") or "deepseek-v4-pro"


def list_models(include_hidden: bool = False) -> list[ModelSpec]:
    out = []
    for spec in _load_models().values():
        if not include_hidden and spec.hidden:
            continue
        out.append(spec)
    return out


def get_model(model_id: str) -> Optional[ModelSpec]:
    return _load_models().get(model_id)


def list_gateways() -> list[GatewaySpec]:
    return list(_load_gateways().values())


def get_gateway(gateway_id: str) -> Optional[GatewaySpec]:
    return _load_gateways().get(gateway_id)


# ---------------------------------------------------------------------------
# Active gateway (in-process global; can be overridden by env)
# ---------------------------------------------------------------------------

_ACTIVE_GATEWAY_LOCK = threading.RLock()
_active_gateway: Optional[str] = None


def get_active_gateway() -> Optional[str]:
    """Returns active gateway id or None. env CHAT_FORCE_GATEWAY wins."""
    forced = (os.getenv("CHAT_FORCE_GATEWAY") or "").strip()
    if forced:
        return forced
    with _ACTIVE_GATEWAY_LOCK:
        return _active_gateway


def set_active_gateway(gateway_id: Optional[str]) -> None:
    """Set or clear the process-wide active gateway. Pass None to clear."""
    global _active_gateway
    with _ACTIVE_GATEWAY_LOCK:
        _active_gateway = gateway_id


# ---------------------------------------------------------------------------
# User key store: ~/.venusfactory/keys.json
# ---------------------------------------------------------------------------
#
# Schema:
# {
#   "providers": {"deepseek": "sk-...", "openai": "sk-...", "google": "...", "anthropic": "...", "dmx": "sk-..."},
#   "custom": {"<arbitrary-id>": {"label": "...", "api_key": "...", "base_url": "..."}}
# }
#
# Keys are written with 0600 permissions to limit accidental exposure.


def _ensure_keys_file() -> None:
    try:
        _USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    if not _KEYS_FILE.exists():
        try:
            _KEYS_FILE.write_text(json.dumps({"providers": {}, "custom": {}}), encoding="utf-8")
            try:
                os.chmod(_KEYS_FILE, 0o600)
            except OSError:
                pass
        except OSError:
            pass


def _load_keys() -> dict[str, Any]:
    with _KEYS_LOCK:
        _ensure_keys_file()
        try:
            return json.loads(_KEYS_FILE.read_text(encoding="utf-8")) or {}
        except (OSError, json.JSONDecodeError):
            return {"providers": {}, "custom": {}}


def _save_keys(data: dict[str, Any]) -> None:
    with _KEYS_LOCK:
        _ensure_keys_file()
        try:
            _KEYS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            try:
                os.chmod(_KEYS_FILE, 0o600)
            except OSError:
                pass
        except OSError:
            _logger.warning("Failed to write user keys to %s", _KEYS_FILE)


def get_user_key(provider: str) -> str:
    """Return the user-stored key for a provider, or empty string."""
    data = _load_keys()
    return str(((data.get("providers") or {}).get(provider)) or "")


def set_user_key(provider: str, api_key: str) -> None:
    """Persist (or clear, when empty) a provider's API key in the user store."""
    data = _load_keys()
    providers = data.setdefault("providers", {})
    if api_key:
        providers[provider] = api_key
    else:
        providers.pop(provider, None)
    _save_keys(data)


def list_user_key_providers() -> dict[str, bool]:
    """Return {provider_id: has_key_bool} for all known providers (does NOT leak keys).

    "has_key" = TRUE if the provider has either:
      - a key stored in ~/.venusfactory/keys.json, OR
      - any model/gateway under this provider whose api_key_env var is non-empty, OR
      - an active gateway (CHAT_FORCE_GATEWAY) holds a key and routes this provider's
        models — mirrors resolve_endpoint(), which dispatches such calls through the
        gateway using the gateway's key.
    The UI uses this to decide whether to prompt the user for a key.
    """
    models = list(_load_models().values())
    gateways = list(_load_gateways().values())
    providers = {spec.provider for spec in models}
    providers.update({gw.id for gw in gateways})

    data = _load_keys()
    stored = data.get("providers") or {}

    # Index env vars by provider
    env_by_provider: dict[str, set[str]] = {}
    for spec in models:
        if spec.api_key_env:
            env_by_provider.setdefault(spec.provider, set()).add(spec.api_key_env)
    for gw in gateways:
        if gw.api_key_env:
            env_by_provider.setdefault(gw.id, set()).add(gw.api_key_env)

    # Gateway propagation: when an active gateway holds a key, every provider whose
    # model is routed through it inherits has_key (matches resolve_endpoint()).
    gateway_lifted: set[str] = set()
    active_gw_id = get_active_gateway()
    if active_gw_id:
        active_gw = next((g for g in gateways if g.id == active_gw_id), None)
        if active_gw is not None:
            gw_has_key = bool(stored.get(active_gw.id)) or (
                bool(active_gw.api_key_env)
                and bool(os.getenv(active_gw.api_key_env, "").strip())
            )
            if gw_has_key:
                models_by_id = {m.id: m for m in models}
                for aliased_model_id in (active_gw.model_aliases or {}).keys():
                    m = models_by_id.get(aliased_model_id)
                    if m and m.provider:
                        gateway_lifted.add(m.provider)
                for em in active_gw.extra_models or []:
                    if em.provider:
                        gateway_lifted.add(em.provider)

    result: dict[str, bool] = {}
    for p in sorted(providers):
        if stored.get(p):
            result[p] = True
            continue
        if any(os.getenv(var, "").strip() for var in env_by_provider.get(p, set())):
            result[p] = True
            continue
        if p in gateway_lifted:
            result[p] = True
            continue
        result[p] = False
    return result


# ---------------------------------------------------------------------------
# Endpoint resolution
# ---------------------------------------------------------------------------


def resolve_endpoint(model_id: str, api_key: str = "") -> ResolvedEndpoint:
    """Compute the (base_url, api_key, effective_model_id) tuple to call.

    Honors active gateway if set: routes via the gateway with translated model id.

    api_key precedence:
        1. explicit ``api_key`` argument
        2. user key store (~/.venusfactory/keys.json) keyed by provider/gateway id
        3. env var named by ``api_key_env`` on the model/gateway
        4. legacy fallback: OPENAI_API_KEY (for backward-compat with old single-key setups)
    """
    spec = get_model(model_id)
    active_gw_id = get_active_gateway()
    if active_gw_id:
        gw = get_gateway(active_gw_id)
        if gw is not None:
            effective_model = gw.model_aliases.get(model_id, model_id)
            resolved_key = (
                api_key
                or get_user_key(gw.id)
                or (os.getenv(gw.api_key_env) if gw.api_key_env else "")
                or os.getenv("OPENAI_API_KEY", "")
            )
            return ResolvedEndpoint(
                model_id=effective_model,
                base_url=gw.base_url,
                api_key=resolved_key or "",
                api_compatible=gw.api_compatible,
                via_gateway=gw.id,
                spec=spec,
            )

    if spec is None:
        # Unknown model id: fall back to env-only (legacy behaviour).
        _logger.warning("resolve_endpoint: unknown model_id=%s; using env-only fallback", model_id)
        return ResolvedEndpoint(
            model_id=model_id,
            base_url=os.getenv("CHAT_BASE_URL", "https://api.deepseek.com"),
            api_key=api_key or os.getenv("OPENAI_API_KEY", "") or os.getenv("DEEPSEEK_API_KEY", ""),
            api_compatible="openai",
            via_gateway=None,
            spec=None,
        )

    resolved_key = (
        api_key
        or get_user_key(spec.provider)
        or (os.getenv(spec.api_key_env) if spec.api_key_env else "")
        or os.getenv("OPENAI_API_KEY", "")
    )
    return ResolvedEndpoint(
        model_id=model_id,
        base_url=spec.base_url,
        api_key=resolved_key or "",
        api_compatible=spec.api_compatible,
        via_gateway=None,
        spec=spec,
    )


# ---------------------------------------------------------------------------
# Public package API
# ---------------------------------------------------------------------------

__all__ = [
    "ModelSpec",
    "GatewaySpec",
    "ResolvedEndpoint",
    "get_default_model_id",
    "list_models",
    "get_model",
    "list_gateways",
    "get_gateway",
    "get_active_gateway",
    "set_active_gateway",
    "resolve_endpoint",
    "get_user_key",
    "set_user_key",
    "list_user_key_providers",
    "reload_registry",
]
