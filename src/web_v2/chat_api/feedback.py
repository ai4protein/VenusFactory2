"""Per-message thumbs-up/down feedback endpoint."""
import asyncio
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Request

from web_v2.analytics_store import analytics_store
from web_v2.feedback_webhook import dispatch_webhook

from web_v2.chat_api._models import FeedbackRequest
from web_v2.chat_api._shared import (
    _assert_session_access,
    _extract_client_ip,
    _get_session_or_404,
    _record_access_event,
    _session_owner_key_for_request,
)

router = APIRouter()


@router.post("/sessions/{session_id}/feedback")
async def submit_feedback(
    session_id: str,
    payload: FeedbackRequest,
    request: Request,
):
    _record_access_event(request, "/api/chat/sessions/{id}/feedback")
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)

    history = state.get("history", [])
    if payload.message_index >= len(history):
        raise HTTPException(status_code=400, detail="Invalid message index.")
    msg = history[payload.message_index]
    if msg.get("role") == "user":
        raise HTTPException(status_code=400, detail="Cannot rate user messages.")

    model_name = getattr(state.get("llm"), "model_name", "")
    ip = _extract_client_ip(request)
    owner_key = _session_owner_key_for_request(request)

    analytics_store.record_feedback(
        ts=datetime.now(UTC).isoformat(),
        session_id=session_id,
        message_index=payload.message_index,
        rating=payload.rating,
        comment=payload.comment,
        owner_key=owner_key,
        ip=ip,
        model_name=model_name,
    )

    webhook_data = {
        "session_id": session_id,
        "message_index": payload.message_index,
        "rating": payload.rating,
        "comment": payload.comment,
        "message_content": msg.get("content", "")[:500],
        "model_name": model_name,
        "owner_key": owner_key,
    }
    asyncio.create_task(dispatch_webhook("feedback_submitted", webhook_data))

    return {"success": True, "session_id": session_id, "message_index": payload.message_index}
