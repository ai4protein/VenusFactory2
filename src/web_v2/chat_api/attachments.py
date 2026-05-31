"""Attachment upload endpoint for chat sessions."""
import os
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, Request, UploadFile

from web.utils.common_utils import (
    make_web_v2_upload_name,
    to_web_v2_public_path,
)

from web_v2.chat_api._hooks_runtime import _session_store
from web_v2.chat_api._shared import (
    _assert_session_access,
    _get_session_or_404,
    _record_access_event,
)

router = APIRouter()


@router.post("/sessions/{session_id}/attachments")
async def upload_attachments(
    session_id: str,
    request: Request,
    files: List[UploadFile] = File(default_factory=list),
):
    _record_access_event(request, "/api/chat/sessions/{id}/attachments")
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)
    target_dir = state.get("agent_session_dir")
    os.makedirs(target_dir, exist_ok=True)
    stored = []
    for f in files:
        filename = os.path.basename(f.filename or f"upload-{uuid.uuid4().hex}")
        existing = len([p for p in Path(target_dir).glob("u_*__*") if p.is_file()])
        dst_name = make_web_v2_upload_name(existing + 1, filename)
        dst = os.path.join(target_dir, dst_name)
        content = await f.read()
        with open(dst, "wb") as out:
            out.write(content)
        normalized = dst.replace("\\", "/")
        state.setdefault("temp_files", []).append(normalized)
        stored.append(
            {
                "name": filename,
                "stored_name": dst_name,
                "path": to_web_v2_public_path(normalized),
                "size": len(content),
            }
        )
    await _session_store.save(session_id)
    return {"files": stored}
