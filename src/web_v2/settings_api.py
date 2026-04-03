import re
import os
import shutil
from pathlib import Path
from typing import Dict, List

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field


router = APIRouter(prefix="/api/v2/settings", tags=["settings-v2"])
_ENV_PATH = Path(".env")
_ENV_EXAMPLE_PATH = Path(".env.example")
_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class EnvEntry(BaseModel):
    key: str = Field(min_length=1)
    value: str = ""


class UpdateEnvRequest(BaseModel):
    entries: List[EnvEntry] = Field(default_factory=list)


def _parse_env_lines(raw: str) -> List[EnvEntry]:
    entries: List[EnvEntry] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        entries.append(EnvEntry(key=key, value=value))
    return entries


def _parse_env_map(raw: str) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for entry in _parse_env_lines(raw):
        values[entry.key] = entry.value
    return values


def _template_keys() -> List[str]:
    if not _ENV_EXAMPLE_PATH.exists():
        return []
    keys: List[str] = []
    seen = set()
    raw = _ENV_EXAMPLE_PATH.read_text(encoding="utf-8", errors="ignore")
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            continue
        key = line.split("=", 1)[0].strip()
        if not key or key in seen or not _KEY_PATTERN.match(key):
            continue
        seen.add(key)
        keys.append(key)
    return keys


def _required_template_keys() -> List[str]:
    keys = _template_keys()
    if not keys:
        raise HTTPException(
            status_code=503,
            detail="Settings template unavailable: .env.example is missing or contains no valid keys.",
        )
    return keys


def _ensure_settings_access(request: Request) -> None:
    if os.getenv("WEBUI_V2_ALLOW_REMOTE_SETTINGS", "0").strip() in {"1", "true", "TRUE"}:
        return
    host = request.client.host if request.client else ""
    if host in {"127.0.0.1", "::1", "localhost"}:
        return
    raise HTTPException(
        status_code=403,
        detail="Remote settings access is disabled. Use localhost or set WEBUI_V2_ALLOW_REMOTE_SETTINGS=1.",
    )


@router.get("/env")
async def get_env_entries(request: Request):
    _ensure_settings_access(request)
    keys = _required_template_keys()
    created_from_example = False
    if not _ENV_PATH.exists() and _ENV_EXAMPLE_PATH.exists():
        shutil.copyfile(_ENV_EXAMPLE_PATH, _ENV_PATH)
        created_from_example = True
    if not _ENV_PATH.exists():
        return {
            "entries": [],
            "exists": False,
            "created_from_example": False,
        }
    raw = _ENV_PATH.read_text(encoding="utf-8", errors="ignore")
    env_values = _parse_env_map(raw)
    entries = [EnvEntry(key=k, value=env_values.get(k, "")) for k in keys]
    return {
        "entries": [e.model_dump() for e in entries],
        "exists": True,
        "created_from_example": created_from_example,
    }


@router.post("/env")
async def save_env_entries(payload: UpdateEnvRequest, request: Request):
    _ensure_settings_access(request)
    allowed_keys = _required_template_keys()
    clean_entries: List[EnvEntry] = []
    seen_keys = set()
    for entry in payload.entries:
        key = entry.key.strip()
        if not key:
            continue
        if not _KEY_PATTERN.match(key):
            raise HTTPException(status_code=400, detail=f"Invalid env key: {key}")
        if key not in allowed_keys:
            raise HTTPException(status_code=400, detail=f"Unsupported env key: {key}")
        if key in seen_keys:
            raise HTTPException(status_code=400, detail=f"Duplicate env key: {key}")
        seen_keys.add(key)
        clean_entries.append(EnvEntry(key=key, value=entry.value))

    entry_map = {e.key: e.value for e in clean_entries}
    lines = [f"{k}={entry_map.get(k, '')}" for k in allowed_keys]
    content = "\n".join(lines).rstrip() + "\n"
    _ENV_PATH.write_text(content, encoding="utf-8")
    return {"success": True, "count": len(clean_entries)}
