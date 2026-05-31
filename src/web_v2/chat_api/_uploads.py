"""Attachment normalization + conversation archive helpers.

Split from ``_shared`` so the SSE stream helpers (``_stream``) and the
attachments router can share them without pulling in the larger access-control
and quota surface.
"""
import asyncio
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

from web.utils.common_utils import (
    make_web_v2_upload_name,
    resolve_web_v2_client_path,
    to_web_v2_public_path,
)
from web_v2.analytics_store import analytics_store
from web_v2.feedback_webhook import dispatch_webhook

from web_v2.chat_api._hooks_runtime import _cfg, _logger, _session_store
from web_v2.chat_api._shared import _redact_obj, _runtime_mode


async def _archive_conversation(state: dict[str, Any]) -> None:
    if not _cfg.feedback.collect_conversations:
        return
    try:
        history = state.get("history", [])
        if not history:
            return
        session_id = state.get("session_id", "")
        model_name = getattr(state.get("llm"), "model_name", "")
        messages_json = json.dumps(
            _redact_obj(list(history)), ensure_ascii=False, default=str
        )
        analytics_store.record_conversation(
            ts=datetime.now(UTC).isoformat(),
            session_id=session_id,
            model_name=model_name,
            messages=messages_json,
            message_count=len(history),
            owner_key=str(state.get("owner_key", "")),
            ip=str(state.get("client_ip", "")),
        )
        asyncio.create_task(dispatch_webhook("conversation_archived", {
            "session_id": session_id,
            "model_name": model_name,
            "message_count": len(history),
            "messages": _redact_obj(list(history)),
        }))
    except Exception:
        _logger.debug("Failed to archive conversation", exc_info=True)


async def _normalize_uploaded_file(
    src_path: str,
    agent_session_dir: str,
    temp_files: list[str],
    owner_key: str,
) -> Optional[str]:
    if not src_path:
        return None
    src_file = Path(src_path)
    if not src_file.is_file():
        try:
            src_file = resolve_web_v2_client_path(src_path, allowed_areas=("uploads", "sessions"))
        except ValueError:
            return None
    if not src_file.is_file():
        return None
    if _runtime_mode() == "online":
        try:
            rel_path = to_web_v2_public_path(src_file)
        except Exception:
            rel_path = ""
        if rel_path.startswith("sessions/"):
            parts = [p for p in rel_path.split("/") if p]
            source_session_id = parts[1] if len(parts) > 1 else ""
            if source_session_id:
                # Use peek_owner to avoid rebuilding the source session's
                # runtime (LLM, memory, ...) just to check ownership.
                peek_owner = getattr(_session_store, "peek_owner", None)
                src_owner: Optional[str] = None
                used_peek = False
                if peek_owner is not None:
                    try:
                        src_owner = await peek_owner(source_session_id)
                        used_peek = True
                    except Exception:
                        _logger.debug(
                            "peek_owner failed; falling back to full session load",
                            exc_info=True,
                        )
                if used_peek:
                    if src_owner is None or src_owner != owner_key:
                        return None
                else:
                    # TODO: remove this fallback once SessionStore.peek_owner is guaranteed.
                    source_session = await _session_store.get(source_session_id)
                    if not source_session or str(source_session.get("owner_key", "")) != owner_key:
                        return None
    os.makedirs(agent_session_dir, exist_ok=True)
    existing = len([p for p in Path(agent_session_dir).glob("u_*__*") if p.is_file()])
    dst_name = make_web_v2_upload_name(existing + 1, src_file.name)
    dst = os.path.join(agent_session_dir, dst_name)
    shutil.copy2(str(src_file), dst)
    normalized = dst.replace("\\", "/")
    temp_files.append(normalized)
    return normalized
