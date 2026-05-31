"""chat_api package: assembles sub-routers into a single APIRouter named ``router``.

External entry point. Imported by:

* ``src/api_server.py`` -> ``from web_v2.chat_api import router``
* ``src/api_server.py`` -> ``getattr(chat_api_module, "_session_store")`` /
  ``_SESSION_IDLE_TTL_HOURS`` (process-startup TTL cleanup task)
* ``src/web_v2/workspace_api.py`` -> ``chat_scope.get_visible_session_ids(...)``

Backwards-compatibility note: the symbols re-exported below MUST keep their
names so the call sites above continue to work without modification.
"""
from fastapi import APIRouter

from web_v2.chat_api._hooks_runtime import (
    _HOOK_BG_TASKS,
    _SESSION_CANCEL_FLAGS,
    _SESSION_DB_PATH,
    _SESSION_IDLE_TTL_HOURS,
    _SESSION_LOCKS,
    _SESSIONS_GUARD,
    _dispatch_hook,
    _session_store,
)
from web_v2.chat_api._shared import get_visible_session_ids
from web_v2.chat_api.attachments import router as attachments_router
from web_v2.chat_api.export import router as export_router
from web_v2.chat_api.feedback import router as feedback_router
from web_v2.chat_api.interactive import router as interactive_router
from web_v2.chat_api.messages import router as messages_router
from web_v2.chat_api.quota import router as quota_router
from web_v2.chat_api.sessions import router as sessions_router

router = APIRouter(prefix="/api/chat", tags=["chat-v2"])
for _sub in (
    sessions_router,
    attachments_router,
    messages_router,
    interactive_router,
    quota_router,
    feedback_router,
    export_router,
):
    router.include_router(_sub)

__all__ = [
    "router",
    # Re-exported for api_server.py / workspace_api.py / external compatibility.
    "_session_store",
    "_SESSION_DB_PATH",
    "_SESSION_IDLE_TTL_HOURS",
    "_SESSION_LOCKS",
    "_SESSION_CANCEL_FLAGS",
    "_SESSIONS_GUARD",
    "_HOOK_BG_TASKS",
    "_dispatch_hook",
    "get_visible_session_ids",
]
