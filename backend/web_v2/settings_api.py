import re
import os
import shutil
import hashlib
import tempfile
from threading import Lock
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from backend.web_v2.analytics_store import analytics_store, normalize_time_range

router = APIRouter(prefix="/api/settings", tags=["settings-v2"])
_ENV_PATH = Path(".env")
_ENV_EXAMPLE_PATH = Path(".env.example")
_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ENV_SAVE_LOCK = Lock()
_SECTION_PATTERN = re.compile(r"^#\s*=+\s*(.*?)\s*=+\s*$")


class EnvEntry(BaseModel):
    key: str = Field(min_length=1)
    value: str = ""
    section: Optional[str] = None


class UpdateEnvRequest(BaseModel):
    entries: List[EnvEntry] = Field(default_factory=list)
    revision: Optional[str] = None


def _insights_payload(*, from_iso: str, to_iso: str, filters_applied: Dict[str, Optional[str]], data: Dict[str, object]):
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "time_range": {"from": from_iso, "to": to_iso},
        "filters_applied": filters_applied,
        "data": data,
    }


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
    return [key for _, key in _template_entries()]


def _template_entries() -> List[Tuple[str, str]]:
    if not _ENV_EXAMPLE_PATH.exists():
        return []
    entries: List[Tuple[str, str]] = []
    seen = set()
    section = "General"
    raw = _ENV_EXAMPLE_PATH.read_text(encoding="utf-8", errors="ignore")
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            heading_match = _SECTION_PATTERN.match(stripped)
            if heading_match:
                heading = heading_match.group(1).strip()
                if heading:
                    section = heading
            continue
        if "=" not in line:
            continue
        key = line.split("=", 1)[0].strip()
        if not key or key in seen or not _KEY_PATTERN.match(key):
            continue
        seen.add(key)
        entries.append((section, key))
    return entries


def _required_template_keys() -> List[str]:
    keys = _template_keys()
    if not keys:
        raise HTTPException(
            status_code=503,
            detail="Settings template unavailable: .env.example is missing or contains no valid keys.",
        )
    return keys


def _ensure_settings_access(request: Request) -> None:
    # Server mode is authoritative, including loopback and explicitly allowed remote clients.
    if os.getenv("WEBUI_V2_MODE", "local").strip().lower() == "online":
        raise HTTPException(status_code=403, detail="Environment settings are available only in local mode.")
    if os.getenv("WEBUI_V2_ALLOW_REMOTE_SETTINGS", "0").strip() in {"1", "true", "TRUE"}:
        return
    host = request.client.host if request.client else ""
    if host in {"127.0.0.1", "::1", "localhost"}:
        return
    raise HTTPException(
        status_code=403,
        detail="Remote settings access is disabled. Use localhost or set WEBUI_V2_ALLOW_REMOTE_SETTINGS=1.",
    )


def _ensure_insights_access(request: Request) -> None:
    mode = os.getenv("WEBUI_V2_MODE", "local").strip().lower()
    if mode == "online":
        return
    _ensure_settings_access(request)


@router.get("/env")
async def get_env_entries(request: Request):
    _ensure_settings_access(request)
    return read_env_entries()


def read_env_entries():
    keys = _required_template_keys()
    template_entries = _template_entries()
    section_map = {key: section for section, key in template_entries}
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
    entries = [EnvEntry(key=k, value=env_values.get(k, ""), section=section_map.get(k, "General")) for k in keys]
    return {
        "entries": [e.model_dump() for e in entries],
        "exists": True,
        "created_from_example": created_from_example,
        "path": str(_ENV_PATH),
        "source": str(_ENV_EXAMPLE_PATH),
        "revision": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
    }


@router.post("/env")
async def save_env_entries(payload: UpdateEnvRequest, request: Request):
    _ensure_settings_access(request)
    return write_env_entries(payload)


def write_env_entries(payload):
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

    for entry in clean_entries:
        if any(c in entry.value for c in ("\n", "\r", "\x00")):
            raise HTTPException(status_code=400, detail="Settings values must be a single line.")
    from backend.web_v2.agent_context import SETTINGS, setting_value
    for entry in clean_entries:
        if entry.key in SETTINGS:
            try:
                setting_value(entry.key, entry.value)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from None
    from backend.web_v2.web_analytics import GA_MEASUREMENT_ID_ENV, validate_measurement_id
    for entry in clean_entries:
        if entry.key == GA_MEASUREMENT_ID_ENV:
            try:
                entry.value = validate_measurement_id(entry.value)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from None
    updates = {e.key: e.value for e in clean_entries}
    from agent.harness.provider_config import configuration_keys, configured_models
    if set(updates).intersection(configuration_keys()):
        current = _parse_env_map(_ENV_PATH.read_text(encoding="utf-8")) if _ENV_PATH.exists() else {}
        try:
            configured_models({**os.environ, **current, **updates})
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail="Invalid provider configuration: " + str(exc)) from None
    with _ENV_SAVE_LOCK:
        raw = _ENV_PATH.read_text(encoding="utf-8") if _ENV_PATH.exists() else ""
        revision = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        if payload.revision is not None and payload.revision != revision:
            raise HTTPException(status_code=409, detail="Settings changed. Reload before saving.")
        lines = raw.splitlines(keepends=True)
        updated = set()
        for i, line in enumerate(lines):
            match = re.match(r"^([ \t]*(?:export[ \t]+)?)([A-Za-z_][A-Za-z0-9_]*)([ \t]*=)", line)
            if not match or match.group(2) not in updates:
                continue
            key = match.group(2)
            ending = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
            lines[i] = f"{match.group(1)}{key}{match.group(3)}{updates[key]}{ending}"
            updated.add(key)
        content = "".join(lines)
        for key, value in updates.items():
            if key in updated:
                continue
            if content and not content.endswith("\n"):
                content += "\n"
            content += f"{key}={value}\n"
        if updates:
            mode = _ENV_PATH.stat().st_mode & 0o777 if _ENV_PATH.exists() else 0o600
            fd, temporary = tempfile.mkstemp(prefix=".env-", dir=_ENV_PATH.resolve().parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.chmod(temporary, mode)
                os.replace(temporary, _ENV_PATH)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
    return {"success": True, "count": len(clean_entries), "path": str(_ENV_PATH),
            "revision": hashlib.sha256(content.encode("utf-8")).hexdigest()}



@router.get("/insights/overview")
async def insights_overview(
    request: Request,
    from_value: Optional[str] = Query(default=None, alias="from"),
    to_value: Optional[str] = Query(default=None, alias="to"),
):
    _ensure_insights_access(request)
    from_iso, to_iso = normalize_time_range(from_value, to_value)
    data = analytics_store.query_overview(from_iso, to_iso)
    return _insights_payload(
        from_iso=from_iso,
        to_iso=to_iso,
        filters_applied={"from": from_value, "to": to_value},
        data=data,
    )


@router.get("/insights/tool-calls")
async def insights_tool_calls(
    request: Request,
    group_by: str = Query(default="day", pattern="^(tool|day|hour)$"),
    from_value: Optional[str] = Query(default=None, alias="from"),
    to_value: Optional[str] = Query(default=None, alias="to"),
):
    _ensure_insights_access(request)
    from_iso, to_iso = normalize_time_range(from_value, to_value)
    rows = analytics_store.query_tool_calls(from_iso, to_iso, group_by)
    return _insights_payload(
        from_iso=from_iso,
        to_iso=to_iso,
        filters_applied={"group_by": group_by, "from": from_value, "to": to_value},
        data={"group_by": group_by, "rows": rows},
    )


@router.get("/insights/ip-distribution")
async def insights_ip_distribution(
    request: Request,
    level: str = Query(default="country", pattern="^(country|region)$"),
    from_value: Optional[str] = Query(default=None, alias="from"),
    to_value: Optional[str] = Query(default=None, alias="to"),
):
    _ensure_insights_access(request)
    from_iso, to_iso = normalize_time_range(from_value, to_value)
    rows = analytics_store.query_ip_distribution(from_iso, to_iso, level)
    return _insights_payload(
        from_iso=from_iso,
        to_iso=to_iso,
        filters_applied={"level": level, "from": from_value, "to": to_value},
        data={"level": level, "rows": rows},
    )


@router.get("/insights/token-usage")
async def insights_token_usage(
    request: Request,
    group_by: str = Query(default="day", pattern="^(model|tool|day)$"),
    from_value: Optional[str] = Query(default=None, alias="from"),
    to_value: Optional[str] = Query(default=None, alias="to"),
):
    _ensure_insights_access(request)
    from_iso, to_iso = normalize_time_range(from_value, to_value)
    rows = analytics_store.query_token_usage(from_iso, to_iso, group_by)
    return _insights_payload(
        from_iso=from_iso,
        to_iso=to_iso,
        filters_applied={"group_by": group_by, "from": from_value, "to": to_value},
        data={"group_by": group_by, "rows": rows},
    )


@router.get("/insights/feedback")
async def insights_feedback(
    request: Request,
    from_value: Optional[str] = Query(default=None, alias="from"),
    to_value: Optional[str] = Query(default=None, alias="to"),
):
    _ensure_insights_access(request)
    from_iso, to_iso = normalize_time_range(from_value, to_value)
    summary = analytics_store.query_feedback_summary(from_iso, to_iso)
    items = analytics_store.query_feedback(from_iso, to_iso)
    return _insights_payload(
        from_iso=from_iso,
        to_iso=to_iso,
        filters_applied={"from": from_value, "to": to_value},
        data={"summary": summary, "items": items},
    )


@router.get("/insights/conversations")
async def insights_conversations(
    request: Request,
    from_value: Optional[str] = Query(default=None, alias="from"),
    to_value: Optional[str] = Query(default=None, alias="to"),
    limit: int = Query(default=50, ge=1, le=500),
):
    _ensure_insights_access(request)
    from_iso, to_iso = normalize_time_range(from_value, to_value)
    rows = analytics_store.query_conversations(from_iso, to_iso, limit)
    return _insights_payload(
        from_iso=from_iso,
        to_iso=to_iso,
        filters_applied={"from": from_value, "to": to_value, "limit": str(limit)},
        data={"rows": rows, "count": len(rows)},
    )


@router.get("/insights/map")
async def insights_map(
    request: Request,
    from_value: Optional[str] = Query(default=None, alias="from"),
    to_value: Optional[str] = Query(default=None, alias="to"),
):
    _ensure_insights_access(request)
    from_iso, to_iso = normalize_time_range(from_value, to_value)
    rows = analytics_store.query_map(from_iso, to_iso)
    return _insights_payload(
        from_iso=from_iso,
        to_iso=to_iso,
        filters_applied={"from": from_value, "to": to_value},
        data={"rows": rows},
    )
