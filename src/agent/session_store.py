"""High-level session storage for VenusFactory chat sessions.

Wraps an in-memory cache + SQLite persistence backend. Stores the entire
chat session state dict per session_id, persisting only JSON-serializable
("persistable") fields. Runtime fields (LLM, chains, tool_cache) are lazily
rebuilt via _ensure_runtime() on load.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from logger import get_logger

_logger = get_logger("agent.session_store")


# Keys that should be persisted to disk. All others (chains, llm, memory, tool_cache,
# all_tools, workers, etc.) are runtime-only and must be rebuilt on load.
_PERSISTABLE_KEYS: frozenset[str] = frozenset({
    # identity / paths
    "session_id", "agent_session_dir", "created_at",
    # auth
    "session_token_hash", "token_expires_at", "owner_key", "client_ip",
    # user input echo
    "last_user_text", "last_attachment_paths", "has_prior_research",
    # state machine
    "pi_report", "pi_suggest_steps", "plan", "current_step_index", "step_results",
    "clarification_questions", "clarification_answers",
    "waiting_for", "status", "error",
    "execution_failed", "failed_step", "failed_reason",
    "ui_lang", "sub_report_rewrite_comment", "auto_execute", "skipped_steps",
    # research
    "research_sections", "research_idx", "search_idx",
    "current_search_results", "research_sub_reports",
    # conversation
    "dialogue_memory", "history", "conversation_log", "tool_executions",
    "temp_files", "latest_tool_output_file",
    # defaults (so we can recreate llm)
    "default_llm_api_key", "default_llm_base_url", "default_llm_model_name",
    # custom model
    "active_custom_model_id",
    # kimi-code engine: 1:1 mapping to a kimi server session
    "kimi_session_id",
    # Security audit trail: kimi tool-call approvals our policy refused
    "security_events",
})

# Special: protein_context is serialized via ProteinContextManager.serialize()
_PROTEIN_CONTEXT_KEY = "protein_context"


def persistable_state(state: dict[str, Any]) -> dict[str, Any]:
    """Extract only persistable fields from state. Serializes ProteinContextManager."""
    out: dict[str, Any] = {}
    for k in _PERSISTABLE_KEYS:
        if k in state:
            v = state[k]
            if isinstance(v, datetime):
                v = v.isoformat()
            out[k] = v
    pc = state.get(_PROTEIN_CONTEXT_KEY)
    if pc is not None and hasattr(pc, "serialize"):
        out[_PROTEIN_CONTEXT_KEY] = pc.serialize()
    return out


class SqliteBackend:
    """Thread-safe SQLite K-V backend for session persistence.

    Uses one SQLite connection per OS thread (via ``threading.local``) so writes
    no longer serialize behind a single global RLock. WAL journal mode +
    ``busy_timeout`` handle concurrent writers safely.

    Schema: sessions(session_id TEXT PRIMARY KEY, owner_key TEXT, created_at REAL,
                     last_accessed REAL, data TEXT)
    """

    def __init__(self, db_path: str):
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._tls = threading.local()
        # Retain a long-lived "primary" connection + lock for backward-compat
        # (older tests / callers reach into backend._conn / backend._lock).
        # New code paths must use _get_conn() instead.
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._init_schema(self._conn)

    def _init_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                owner_key TEXT,
                created_at REAL,
                last_accessed REAL,
                data TEXT
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_owner_recent "
            "ON sessions(owner_key, last_accessed DESC)"
        )
        conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        """Return a per-thread connection (created lazily). Auto-commit mode."""
        conn = getattr(self._tls, "conn", None)
        if conn is None:
            conn = sqlite3.connect(
                self._db_path, check_same_thread=False, isolation_level=None
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            self._tls.conn = conn
        return conn

    def upsert(self, session_id: str, owner_key: str, payload: dict) -> None:
        now = time.time()
        data_json = json.dumps(payload, default=str, ensure_ascii=False)
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO sessions(session_id, owner_key, created_at, last_accessed, data)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                owner_key = excluded.owner_key,
                last_accessed = excluded.last_accessed,
                data = excluded.data
            """,
            (session_id, owner_key, now, now, data_json),
        )

    def load(self, session_id: str) -> Optional[dict]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT data FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if not row:
            return None
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            _logger.warning("Corrupt session row for %s, ignoring", session_id)
            return None

    def delete(self, session_id: str) -> None:
        conn = self._get_conn()
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

    def list_ids(self, owner_key: Optional[str] = None) -> list[str]:
        conn = self._get_conn()
        if owner_key is None:
            rows = conn.execute(
                "SELECT session_id FROM sessions ORDER BY last_accessed DESC"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT session_id FROM sessions WHERE owner_key = ? ORDER BY last_accessed DESC",
                (owner_key,),
            ).fetchall()
        return [r[0] for r in rows]

    def cleanup_older_than(self, idle_seconds: float) -> int:
        cutoff = time.time() - idle_seconds
        conn = self._get_conn()
        cur = conn.execute(
            "DELETE FROM sessions WHERE last_accessed < ?", (cutoff,)
        )
        return cur.rowcount or 0

    # ------------------------------------------------------------------ new API
    def peek_owner(self, session_id: str) -> Optional[str]:
        """Return owner_key for a session without rebuilding runtime state."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT owner_key FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return row[0] if row else None

    def list_summaries(self, owner_key: Optional[str] = None) -> list[dict]:
        """Lightweight summary listing. Parses minimal fields from data JSON
        without rebuilding any runtime state. Sorted newest-first.
        """
        conn = self._get_conn()
        if owner_key is None:
            rows = conn.execute(
                "SELECT session_id, owner_key, created_at, last_accessed, data "
                "FROM sessions ORDER BY last_accessed DESC"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT session_id, owner_key, created_at, last_accessed, data "
                "FROM sessions WHERE owner_key = ? ORDER BY last_accessed DESC",
                (owner_key,),
            ).fetchall()
        summaries: list[dict] = []
        for sid, owner, created, accessed, data_json in rows:
            try:
                d = json.loads(data_json) if data_json else {}
            except json.JSONDecodeError:
                d = {}
            history = d.get("history") or []
            summaries.append({
                "session_id": sid,
                "owner_key": owner,
                "created_at": d.get("created_at") or created,
                "last_accessed": accessed,
                "status": d.get("status") or "",
                "history_size": len(history) if isinstance(history, list) else 0,
                "model_name": d.get("default_llm_model_name") or "",
            })
        return summaries


class SessionStore:
    """In-memory cache + SQLite persistence wrapper for chat session states."""

    def __init__(
        self,
        backend: SqliteBackend,
        ensure_runtime_fn: Callable[[dict[str, Any]], dict[str, Any]],
    ):
        self._backend = backend
        self._mem: dict[str, dict[str, Any]] = {}
        # Tracks the most recent time each mem entry was touched (created/get/save).
        # Used by cleanup_loop to evict idle mem entries -- created_at alone is wrong
        # because reloaded sessions store created_at as an ISO string, not datetime.
        self._mem_last_accessed: dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._ensure_runtime = ensure_runtime_fn
        # Strong references to background tasks so the loop doesn't drop them mid-flight.
        self._background_tasks: set[asyncio.Task] = set()

    def _track(self, task: asyncio.Task) -> None:
        """Hold a strong reference to a background task; drop it when done."""
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def _touch(self, sid: str) -> None:
        self._mem_last_accessed[sid] = time.time()

    async def create(self, state: dict[str, Any]) -> None:
        sid = state["session_id"]
        async with self._lock:
            self._mem[sid] = state
            self._touch(sid)
        owner = str(state.get("owner_key", ""))
        await asyncio.to_thread(self._backend.upsert, sid, owner, persistable_state(state))

    async def get(self, session_id: str) -> Optional[dict[str, Any]]:
        async with self._lock:
            cached = self._mem.get(session_id)
            if cached is not None:
                self._touch(session_id)
        if cached is not None:
            return cached
        payload = await asyncio.to_thread(self._backend.load, session_id)
        if payload is None:
            return None
        # Restore int keys for step_results (JSON coerces dict keys to str).
        if "step_results" in payload and isinstance(payload["step_results"], dict):
            fixed: dict[Any, Any] = {}
            for k, v in payload["step_results"].items():
                try:
                    fixed[int(k)] = v
                except (TypeError, ValueError):
                    fixed[k] = v
            payload["step_results"] = fixed
        # Restore protein_context
        pc_data = payload.get(_PROTEIN_CONTEXT_KEY)
        if pc_data is not None:
            from agent.chat_agent import ProteinContextManager
            payload[_PROTEIN_CONTEXT_KEY] = ProteinContextManager.deserialize(pc_data)
        # Lazily rebuild runtime (llm, chains, etc.)
        full = self._ensure_runtime(payload)
        async with self._lock:
            self._mem[session_id] = full
            self._touch(session_id)
        return full

    async def save(self, session_id: str) -> None:
        async with self._lock:
            state = self._mem.get(session_id)
            if state is not None:
                self._touch(session_id)
        if state is None:
            return
        owner = str(state.get("owner_key", ""))
        await asyncio.to_thread(self._backend.upsert, session_id, owner, persistable_state(state))

    async def delete(self, session_id: str) -> None:
        async with self._lock:
            self._mem.pop(session_id, None)
            self._mem_last_accessed.pop(session_id, None)
        await asyncio.to_thread(self._backend.delete, session_id)

    async def list_ids(self, owner_key: Optional[str] = None) -> list[str]:
        # DB is the source of truth for ownership; mem is a hot cache. We rely on
        # SqliteBackend.list_ids to filter by owner. Then we union in any mem-only
        # sessions (e.g. just created, not yet persisted) that match the filter.
        db_ids = await asyncio.to_thread(self._backend.list_ids, owner_key)
        ordered: list[str] = list(db_ids)
        seen = set(ordered)
        async with self._lock:
            mem_snapshot = list(self._mem.items())
        for sid, state in mem_snapshot:
            if sid in seen:
                continue
            if owner_key is not None and str(state.get("owner_key", "")) != owner_key:
                continue
            ordered.append(sid)
            seen.add(sid)
        return ordered

    # ------------------------------------------------------------------ new API
    async def peek_owner(self, session_id: str) -> Optional[str]:
        """Resolve the owner_key for a session without rebuilding runtime state.

        Checks the in-memory cache first, then falls back to the SQLite backend.
        Returns None if the session is unknown or has no owner_key.
        """
        async with self._lock:
            s = self._mem.get(session_id)
        if s is not None:
            owner = str(s.get("owner_key", ""))
            return owner or None
        return await asyncio.to_thread(self._backend.peek_owner, session_id)

    async def list_summaries(self, owner_key: Optional[str] = None) -> list[dict]:
        """Return lightweight per-session summaries (no runtime rebuild).

        Includes mem-only sessions that have not been persisted yet so callers
        see them in the listing immediately after create().
        """
        db_summaries = await asyncio.to_thread(self._backend.list_summaries, owner_key)
        seen = {s["session_id"] for s in db_summaries}
        async with self._lock:
            mem_snapshot = list(self._mem.items())
        for sid, state in mem_snapshot:
            if sid in seen:
                continue
            if owner_key is not None and str(state.get("owner_key", "")) != owner_key:
                continue
            history = state.get("history") or []
            llm = state.get("llm")
            model_name = (
                getattr(llm, "model_name", "")
                or state.get("default_llm_model_name", "")
                or ""
            )
            db_summaries.append({
                "session_id": sid,
                "owner_key": str(state.get("owner_key", "")),
                "created_at": str(state.get("created_at", "")),
                "last_accessed": None,
                "status": state.get("status", ""),
                "history_size": len(history) if isinstance(history, list) else 0,
                "model_name": model_name,
            })
        return db_summaries

    def spawn_cleanup_loop(
        self,
        interval_sec: float = 600,
        idle_ttl_sec: float = 86400,
    ) -> asyncio.Task:
        """Start ``cleanup_loop`` as a tracked background task.

        Holds a strong reference so the task isn't garbage-collected mid-flight.
        """
        task = asyncio.create_task(self.cleanup_loop(interval_sec, idle_ttl_sec))
        self._track(task)
        return task

    async def cleanup_loop(self, interval_sec: float = 600, idle_ttl_sec: float = 86400) -> None:
        """Background TTL cleanup."""
        while True:
            try:
                removed = await asyncio.to_thread(self._backend.cleanup_older_than, idle_ttl_sec)
                if removed:
                    _logger.info("Session TTL cleanup removed %d entries", removed)
                async with self._lock:
                    now = time.time()
                    expired_mem = [
                        sid for sid, last in self._mem_last_accessed.items()
                        if (now - last) > idle_ttl_sec
                    ]
                    for sid in expired_mem:
                        self._mem.pop(sid, None)
                        self._mem_last_accessed.pop(sid, None)
                    # Also drop any orphan mem entries whose last-access record was lost
                    # (shouldn't happen, but keep the two dicts aligned).
                    for sid in list(self._mem.keys()):
                        if sid not in self._mem_last_accessed:
                            self._mem_last_accessed[sid] = now
            except Exception:
                _logger.exception("Session cleanup iteration failed")
            await asyncio.sleep(interval_sec)
