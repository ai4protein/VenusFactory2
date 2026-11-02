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

from mcp_tools.common.common_utils import (
    make_web_v2_upload_name,
    resolve_web_v2_client_path,
    to_web_v2_public_path,
)
from backend.web_v2.analytics_store import analytics_store
from backend.web_v2.feedback_webhook import dispatch_webhook

from backend.web_v2.chat_api._hooks_runtime import _cfg, _logger, _session_store
from backend.web_v2.chat_api._shared import _redact_obj, _runtime_mode


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
    from mcp_tools.common.common_utils import get_web_v2_root_dir
    from utils.scoped_files import copy_file
    from backend.web_v2.upload_limits import upload_limit
    # The attachment endpoint has already authorized and stored the file in
    # this conversation. A client-supplied host path is never an upload grant.
    root = Path(agent_session_dir).resolve()
    raw = Path(src_path)
    src_file = raw if raw.is_absolute() else get_web_v2_root_dir().resolve() / raw
    try:
        relative = src_file.relative_to(root).as_posix()
        existing = len(list(root.glob("u_*__*")))
        dst_name = make_web_v2_upload_name(existing + 1, src_file.name)
        copy_file(root, relative, root, dst_name, max_bytes=upload_limit() or 1024**4)
    except (OSError, ValueError):
        return None
    normalized = (root / dst_name).as_posix()
    temp_files.append(normalized)
    return normalized
