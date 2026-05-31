"""Model registry + user API key management endpoints.

Exposes read-only listing of registered models/gateways and lightweight
mutation endpoints for the user's per-provider API keys and the active
gateway selection. All responses are scrubbed via ``ModelSpec.to_public_dict``
/ ``GatewaySpec.to_public_dict`` and ``list_user_key_providers`` so secret
material never leaves the server.
"""
from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

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


@router.get("")
async def list_all_models() -> dict:
    """Return the full registry snapshot (models, gateways, key status)."""
    return {
        "default_model": get_default_model_id(),
        "models": [m.to_public_dict() for m in list_models()],
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
