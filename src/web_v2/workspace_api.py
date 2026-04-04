import json
import mimetypes
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile

from web.utils.common_utils import (
    build_run_id_utc,
    create_run_manifest,
    get_web_v2_area_dir,
    make_web_v2_upload_name,
    resolve_web_v2_client_path,
    to_web_v2_public_path,
)
try:
    from src.web_v2 import chat_api as chat_scope
except ModuleNotFoundError:
    from web_v2 import chat_api as chat_scope

router = APIRouter(prefix="/api/workspace", tags=["workspace-v2"])

_ALLOWED_UPLOAD_SUFFIX = {
    ".fasta",
    ".fa",
    ".faa",
    ".fna",
    ".txt",
    ".csv",
    ".tsv",
    ".json",
    ".pdb",
    ".cif",
}
_READABLE_TEXT_SUFFIX = {
    ".txt",
    ".fasta",
    ".fa",
    ".faa",
    ".fna",
    ".csv",
    ".tsv",
    ".json",
    ".pdb",
    ".cif",
    ".md",
}


def _runtime_mode() -> str:
    mode = os.getenv("WEBUI_V2_MODE", "local").strip().lower()
    return mode if mode in {"local", "online"} else "local"


def _owner_key_for_request(request: Request) -> str:
    if _runtime_mode() != "online":
        return "local"
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        parts = [p.strip() for p in xff.split(",") if p.strip()]
        if parts:
            return parts[0]
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _workspace_index_path() -> Path:
    return get_web_v2_area_dir("cache", tool="workspace") / "workspace_index.json"


def _load_workspace_index() -> dict[str, dict[str, Any]]:
    index_path = _workspace_index_path()
    if not index_path.exists():
        return {}
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return {str(k): v for k, v in payload.items() if isinstance(v, dict)}
    except Exception:
        return {}
    return {}


def _save_workspace_index(index: dict[str, dict[str, Any]]) -> None:
    index_path = _workspace_index_path()
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


def _guess_category(suffix: str) -> str:
    normalized = (suffix or "").lower()
    if normalized in {".fasta", ".fa", ".faa", ".fna"}:
        return "sequence"
    if normalized in {".pdb", ".cif"}:
        return "structure"
    if normalized in {".csv", ".tsv", ".json", ".txt"}:
        return "table_or_text"
    return "other"


def _source_from_relative(relative_path: str) -> str:
    parts = [p for p in relative_path.split("/") if p]
    if not parts:
        return "unknown"
    # uploads/<tool>/... or sessions/<tool>/...
    if parts[0] in {"uploads", "sessions"} and len(parts) > 1:
        return parts[1]
    return parts[0]


def _bucket_from_relative(relative_path: str) -> str:
    parts = [p for p in relative_path.split("/") if p]
    if len(parts) >= 2 and parts[0] == "uploads" and parts[1] == "workspace":
        return "user_upload"
    if len(parts) >= 2 and parts[0] == "sessions":
        return "chat_session"
    if parts and parts[0] == "uploads":
        return "tool_upload"
    return "other"


def _session_id_from_relative(relative_path: str) -> str:
    parts = [p for p in relative_path.split("/") if p]
    if len(parts) >= 2 and parts[0] == "sessions":
        return parts[1]
    return ""


async def _assert_online_workspace_access(request: Request, rel_path: str, item_bucket: str, index: dict[str, dict[str, Any]]) -> None:
    if _runtime_mode() != "online":
        return
    if item_bucket == "chat_session":
        sid = _session_id_from_relative(rel_path)
        visible = await chat_scope.get_visible_session_ids(request)
        if not sid or sid not in visible:
            raise HTTPException(status_code=403, detail="You do not have access to this session file.")
        return
    owner_key = _owner_key_for_request(request)
    meta_owner = str(index.get(rel_path, {}).get("owner_key", ""))
    if meta_owner != owner_key:
        raise HTTPException(status_code=403, detail="You do not have access to this workspace file.")


def _assert_user_upload_bucket(rel_path: str) -> None:
    if _bucket_from_relative(rel_path) != "user_upload":
        raise HTTPException(status_code=403, detail="Only user uploaded workspace files can be modified.")


def _build_workspace_item(path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    rel_path = to_web_v2_public_path(path)
    stat = path.stat()
    suffix = path.suffix.lower()
    created_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    source = _source_from_relative(rel_path)
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    category = str(metadata.get("category") or _guess_category(suffix))
    tags = metadata.get("tags")
    if not isinstance(tags, list):
        tags = []

    bucket = _bucket_from_relative(rel_path)

    return {
        "id": str(metadata.get("id") or rel_path),
        "display_name": str(metadata.get("display_name") or path.name),
        "storage_path": rel_path,
        "source": source,
        "mime": mime,
        "suffix": suffix,
        "size": stat.st_size,
        "created_at": created_at,
        "tags": tags,
        "category": category,
        "bucket": bucket,
        "ttl_policy": str(metadata.get("ttl_policy") or "default"),
    }


def _iter_area_files(area: str) -> list[Path]:
    area_root = get_web_v2_area_dir(area)
    return [p for p in area_root.rglob("*") if p.is_file()]


@router.get("/files")
async def list_workspace_files(
    request: Request,
    q: str = "",
    source: str = "",
    file_type: str = "",
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    sort: str = "created_desc",
    include_sessions: bool = False,
):
    mode = _runtime_mode()
    owner_key = _owner_key_for_request(request)
    visible_session_ids = await chat_scope.get_visible_session_ids(request)
    index = _load_workspace_index()
    files = _iter_area_files("uploads")
    if include_sessions:
        files.extend(_iter_area_files("sessions"))

    items: list[dict[str, Any]] = []
    q_lower = q.strip().lower()
    source_lower = source.strip().lower()
    file_type_lower = file_type.strip().lower()

    for file_path in files:
        try:
            rel_path = to_web_v2_public_path(file_path)
        except Exception:
            continue
        item = _build_workspace_item(file_path, index.get(rel_path, {}))
        if mode == "online":
            if item["bucket"] == "chat_session":
                sid = _session_id_from_relative(rel_path)
                if not sid or sid not in visible_session_ids:
                    continue
            else:
                meta_owner = str(index.get(rel_path, {}).get("owner_key", ""))
                if meta_owner != owner_key:
                    continue
        if q_lower and q_lower not in item["display_name"].lower() and q_lower not in item["storage_path"].lower():
            continue
        if source_lower and source_lower != str(item["source"]).lower():
            continue
        if file_type_lower and file_type_lower != str(item["category"]).lower():
            continue
        items.append(item)

    if sort == "name_asc":
        items.sort(key=lambda x: str(x["display_name"]).lower())
    elif sort == "size_desc":
        items.sort(key=lambda x: int(x["size"]), reverse=True)
    else:
        items.sort(key=lambda x: str(x["created_at"]), reverse=True)

    total = len(items)
    start = (page - 1) * page_size
    end = start + page_size
    return {
        "items": items[start:end],
        "total": total,
        "page": page,
        "page_size": page_size,
        "mode": mode,
        "enabled": True,
    }


@router.post("/upload")
async def upload_workspace_file(request: Request, file: UploadFile = File(...)):
    owner_key = _owner_key_for_request(request)

    filename = os.path.basename(file.filename or "workspace-upload.bin")
    suffix = Path(filename).suffix.lower()
    if suffix and suffix not in _ALLOWED_UPLOAD_SUFFIX:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    run_id = build_run_id_utc()
    upload_dir = get_web_v2_area_dir("uploads", tool="workspace", run_id=run_id)
    stored_name = make_web_v2_upload_name(1, filename)
    dst = upload_dir / stored_name
    content = await file.read()
    dst.write_bytes(content)

    rel_path = to_web_v2_public_path(dst)
    index = _load_workspace_index()
    index[rel_path] = {
        "id": rel_path,
        "display_name": filename,
        "tags": [],
        "category": _guess_category(suffix),
        "ttl_policy": "default",
        "owner_key": owner_key,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_workspace_index(index)

    create_run_manifest(
        run_id=run_id,
        tool="workspace",
        status="uploaded",
        inputs=[
            {
                "path": str(dst),
                "name": filename,
                "size": len(content),
            }
        ],
        outputs=[],
    )

    return {
        "file": _build_workspace_item(dst, index.get(rel_path, {})),
        "run_id": run_id,
    }


@router.post("/register")
async def register_workspace_path(request: Request, storage_path: str = Query(..., min_length=1)):
    owner_key = _owner_key_for_request(request)
    try:
        resolved = resolve_web_v2_client_path(storage_path, allowed_areas=("uploads", "sessions"))
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Access denied.") from exc

    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail="File not found.")

    rel_path = to_web_v2_public_path(resolved)
    if _runtime_mode() == "online" and rel_path.startswith("sessions/"):
        sid = _session_id_from_relative(rel_path)
        visible = await chat_scope.get_visible_session_ids(request)
        if not sid or sid not in visible:
            raise HTTPException(status_code=403, detail="You do not have access to this session file.")

    index = _load_workspace_index()
    existing = index.get(rel_path, {})
    index[rel_path] = {
        "id": str(existing.get("id") or rel_path),
        "display_name": str(existing.get("display_name") or resolved.name),
        "tags": existing.get("tags") if isinstance(existing.get("tags"), list) else [],
        "category": str(existing.get("category") or _guess_category(resolved.suffix.lower())),
        "ttl_policy": str(existing.get("ttl_policy") or "default"),
        "owner_key": owner_key,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_workspace_index(index)
    return {"file": _build_workspace_item(resolved, index[rel_path])}


@router.post("/read-text")
async def read_workspace_text_file(
    request: Request,
    storage_path: str = Query(..., min_length=1),
    max_lines: int = Query(default=5000, ge=1, le=20000),
    max_chars: int = Query(default=200000, ge=256, le=1000000),
):
    try:
        resolved = resolve_web_v2_client_path(storage_path, allowed_areas=("uploads", "sessions"))
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Access denied.") from exc

    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail="File not found.")

    suffix = resolved.suffix.lower()
    if suffix not in _READABLE_TEXT_SUFFIX:
        raise HTTPException(status_code=400, detail=f"Unsupported text file type: {suffix}")

    rel_path = to_web_v2_public_path(resolved)
    index = _load_workspace_index()
    await _assert_online_workspace_access(request, rel_path, _bucket_from_relative(rel_path), index)

    raw_text = resolved.read_text(encoding="utf-8", errors="ignore")
    trimmed_by_chars = False
    if len(raw_text) > max_chars:
        raw_text = raw_text[:max_chars]
        trimmed_by_chars = True

    lines = raw_text.splitlines()
    trimmed_by_lines = False
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        trimmed_by_lines = True
    content = "\n".join(lines)
    if content and not content.endswith("\n"):
        content += "\n"
    return {
        "storage_path": rel_path,
        "content": content,
        "line_count": len(lines),
        "truncated": trimmed_by_chars or trimmed_by_lines,
    }


@router.patch("/file")
async def replace_workspace_file(
    request: Request,
    storage_path: str = Query(..., min_length=1),
    file: UploadFile = File(...),
):
    try:
        resolved = resolve_web_v2_client_path(storage_path, allowed_areas=("uploads", "sessions"))
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Access denied.") from exc

    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail="File not found.")

    rel_path = to_web_v2_public_path(resolved)
    _assert_user_upload_bucket(rel_path)
    index = _load_workspace_index()
    await _assert_online_workspace_access(request, rel_path, _bucket_from_relative(rel_path), index)

    filename = os.path.basename(file.filename or resolved.name)
    new_suffix = Path(filename).suffix.lower()
    effective_suffix = new_suffix or resolved.suffix.lower()
    if effective_suffix and effective_suffix not in _ALLOWED_UPLOAD_SUFFIX:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {effective_suffix}")

    content = await file.read()
    target_path = resolved
    target_rel_path = rel_path

    # Keep replacement in place when suffix matches; rotate file path when suffix changes.
    if effective_suffix != resolved.suffix.lower():
        target_path = resolved.parent / make_web_v2_upload_name(1, filename)
        target_path.write_bytes(content)
        resolved.unlink(missing_ok=True)
        target_rel_path = to_web_v2_public_path(target_path)
    else:
        resolved.write_bytes(content)

    previous = index.pop(rel_path, {})
    index[target_rel_path] = {
        "id": str(previous.get("id") or target_rel_path),
        "display_name": filename,
        "tags": previous.get("tags") if isinstance(previous.get("tags"), list) else [],
        "category": _guess_category(effective_suffix),
        "ttl_policy": str(previous.get("ttl_policy") or "default"),
        "owner_key": str(previous.get("owner_key") or _owner_key_for_request(request)),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_workspace_index(index)
    return {"file": _build_workspace_item(target_path, index[target_rel_path])}


@router.delete("/file")
async def delete_workspace_file(request: Request, storage_path: str = Query(..., min_length=1)):
    try:
        resolved = resolve_web_v2_client_path(storage_path, allowed_areas=("uploads", "sessions"))
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Access denied.") from exc

    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail="File not found.")

    rel_path = to_web_v2_public_path(resolved)
    _assert_user_upload_bucket(rel_path)
    index = _load_workspace_index()
    await _assert_online_workspace_access(request, rel_path, _bucket_from_relative(rel_path), index)

    resolved.unlink(missing_ok=True)
    index.pop(rel_path, None)
    _save_workspace_index(index)
    return {"ok": True, "deleted_path": rel_path}
