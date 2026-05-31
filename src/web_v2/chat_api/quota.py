"""Online chat quota endpoint."""
from fastapi import APIRouter, Request

from web_v2.chat_api._shared import (
    _get_online_chat_quota_status,
    _record_access_event,
)

router = APIRouter()


@router.get("/quota")
async def get_chat_quota(request: Request):
    _record_access_event(request, "/api/chat/quota")
    return await _get_online_chat_quota_status(request)
