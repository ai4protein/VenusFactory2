"""Process-local runtime singletons for chat_api sub-routers.

Holds the shared SessionStore, asyncio locks, cancel flags, fire-and-forget hook
tasks, and the cached compiled agent graph. Kept dependency-free of ``_shared``
to avoid circular imports.
"""
import asyncio
import functools
import os
import secrets
from pathlib import Path

from agent.chat_agent import ensure_runtime_state
from agent.chat_graph import create_agent_graph
from agent.session_store import SessionStore, SqliteBackend
from config import get_config
from logger import get_logger

_logger = get_logger("web_v2.chat_api")
_cfg = get_config()

_BUILTIN_MODEL_LABELS = {"Gemini-2.5-Pro", "ChatGPT-4o", "Claude-3.7", "DeepSeek-R1"}

# Strong references for fire-and-forget hook dispatch tasks.
# Without this, asyncio.create_task tasks can be garbage-collected mid-flight.
_HOOK_BG_TASKS: set[asyncio.Task] = set()


def _dispatch_hook(coro) -> None:
    """Schedule a hook coroutine as a fire-and-forget background task."""
    try:
        task = asyncio.create_task(coro)
        _HOOK_BG_TASKS.add(task)
        task.add_done_callback(_HOOK_BG_TASKS.discard)
    except Exception:
        _logger.debug("hook dispatch failed", exc_info=True)


_SESSION_DB_PATH = os.getenv("VENUSFACTORY_SESSION_DB_PATH") or str(
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "chat_sessions.db"
)
_SESSION_IDLE_TTL_HOURS = float(os.getenv("VENUSFACTORY_SESSION_IDLE_TTL_HOURS", "168"))

_session_store = SessionStore(
    backend=SqliteBackend(_SESSION_DB_PATH),
    ensure_runtime_fn=ensure_runtime_state,
)

# Process-local concurrency primitives: locks & cancel flags don't need persistence.
# _SESSIONS_GUARD keeps its name for historical reasons but now only guards
# _SESSION_LOCKS / _SESSION_CANCEL_FLAGS (the session state itself lives in _session_store).
_SESSION_LOCKS: dict[str, asyncio.Lock] = {}
_SESSION_CANCEL_FLAGS: dict[str, bool] = {}
_SESSIONS_GUARD = asyncio.Lock()

_ONLINE_DAILY_CHAT_LIMIT = _cfg.online_limits.daily_chat_limit
_SESSION_TOKEN_TTL_HOURS = _cfg.online_limits.session_token_ttl_hours
_SESSION_TOKEN_SECRET = os.getenv("WEBUI_V2_SESSION_TOKEN_SECRET", "").strip() or secrets.token_hex(32)


@functools.cache
def _get_compiled_graph():
    """Return the compiled agent graph, cached at module level.

    StateGraph.compile() is expensive; the resulting graph is stateless and
    safe to share across requests, so we build it once and reuse it.
    """
    return create_agent_graph()


async def _get_lock(session_id: str) -> asyncio.Lock:
    async with _SESSIONS_GUARD:
        if session_id not in _SESSION_LOCKS:
            _SESSION_LOCKS[session_id] = asyncio.Lock()
        return _SESSION_LOCKS[session_id]


async def _is_cancelled(session_id: str) -> bool:
    async with _SESSIONS_GUARD:
        return _SESSION_CANCEL_FLAGS.get(session_id, False)


async def _set_cancel(session_id: str, value: bool) -> None:
    async with _SESSIONS_GUARD:
        _SESSION_CANCEL_FLAGS[session_id] = value
