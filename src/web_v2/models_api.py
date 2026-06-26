"""Model registry + user API key management endpoints.

Exposes read-only listing of registered models/gateways and lightweight
mutation endpoints for the user's per-provider API keys and the active
gateway selection. All responses are scrubbed via ``ModelSpec.to_public_dict``
/ ``GatewaySpec.to_public_dict`` and ``list_user_key_providers`` so secret
material never leaves the server.
"""
from __future__ import annotations

import asyncio
import re
import time

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from agent.kimi_daemon import base_url as kimi_base_url
from agent.model_registry import (
    get_active_gateway,
    get_default_model_id,
    get_gateway,
    list_gateways,
    list_models,
    list_user_key_providers,
    set_active_gateway,
    set_user_key,
)
from logger import get_logger

_logger = get_logger("web_v2.models_api")

router = APIRouter(prefix="/api/models", tags=["models"])

_PROVIDER_RE = re.compile(r"^[a-z0-9_-]{1,32}$")


class SetKeyRequest(BaseModel):
    api_key: str = Field(default="", max_length=512)


class SetGatewayRequest(BaseModel):
    gateway_id: str | None = None


_KIMI_READY_CACHE: dict[str, float | bool | str] = {"ts": 0.0, "ready": False, "reason": ""}
_KIMI_READY_TTL = 5.0  # seconds


async def _kimi_ready_status() -> tuple[bool, str]:
    """Return (ready, reason). Cached briefly to avoid hammering kimi /auth."""
    now = time.monotonic()
    if now - float(_KIMI_READY_CACHE["ts"]) < _KIMI_READY_TTL:
        return bool(_KIMI_READY_CACHE["ready"]), str(_KIMI_READY_CACHE["reason"])
    ready, reason = False, "Kimi daemon unreachable. Start the API server or set KIMI_EXTERNAL=1."
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            r = await client.get(f"{kimi_base_url()}/api/v1/auth")
            if r.status_code == 200:
                data = (r.json() or {}).get("data") or {}
                ready = bool(data.get("ready"))
                if not ready:
                    reason = (
                        "Kimi has no provider/default model configured. In a terminal run one of: "
                        "`kimi provider catalog add deepseek --api-key <KEY> --default-model deepseek-v4-pro`, "
                        "or `kimi provider catalog add moonshot --api-key <KEY> --default-model kimi-k2-thinking`, "
                        "or `kimi login` for the hosted Kimi account."
                    )
                else:
                    reason = ""
    except Exception as exc:  # noqa: BLE001
        reason = f"Kimi daemon not reachable: {exc}"
    _KIMI_READY_CACHE.update({"ts": now, "ready": ready, "reason": reason})
    return ready, reason


@router.get("")
async def list_all_models() -> dict:
    """Return the full registry snapshot (models, gateways, key status).

    Kimi-code models are decorated with `disabled` + `disabled_reason` when
    the local kimi daemon is unreachable or has no provider configured.
    """
    kimi_ready, kimi_reason = await _kimi_ready_status()
    models_out = []
    for m in list_models():
        d = m.to_public_dict()
        if d.get("engine") == "kimi-code" and not kimi_ready:
            d["disabled"] = True
            d["disabled_reason"] = kimi_reason
        models_out.append(d)
    return {
        "default_model": get_default_model_id(),
        "models": models_out,
        "gateways": [g.to_public_dict() for g in list_gateways()],
        "active_gateway": get_active_gateway(),
        "key_status": list_user_key_providers(),
    }


@router.get("/keys")
async def list_keys() -> dict:
    """Return ``{provider_id: has_key}`` map without leaking key values."""
    return {"providers": list_user_key_providers()}


@router.put("/keys/{provider}")
async def upsert_key(provider: str, body: SetKeyRequest) -> dict:
    """Set or clear (empty string) the user API key for ``provider``."""
    if not _PROVIDER_RE.match(provider):
        raise HTTPException(status_code=400, detail="Invalid provider id")
    try:
        set_user_key(provider, body.api_key)
    except Exception as exc:  # noqa: BLE001 - surface as 400 to the client
        raise HTTPException(status_code=400, detail=f"Failed to set key: {exc}") from exc
    has_key = bool(body.api_key)
    # Never log key contents; only the provider id and the set/clear action.
    _logger.info("User key updated for provider=%s set=%s", provider, has_key)
    return {"provider": provider, "set": has_key}


@router.post("/gateway")
async def switch_gateway(body: SetGatewayRequest) -> dict:
    """Set the active gateway (``None`` to clear)."""
    if body.gateway_id is not None:
        if get_gateway(body.gateway_id) is None:
            raise HTTPException(
                status_code=404,
                detail=f"Gateway not found: {body.gateway_id}",
            )
    try:
        set_active_gateway(body.gateway_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to set gateway: {exc}") from exc
    _logger.info("Active gateway updated: %s", body.gateway_id)
    return {"active_gateway": get_active_gateway()}
