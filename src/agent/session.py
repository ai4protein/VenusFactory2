"""
Session persistence for VenusFactory.

Follows the OpenAI Agents SDK's dual Protocol + ABC pattern:
  - ``Session`` Protocol — structural typing for third-party implementations
  - ``SessionABC`` — abstract base for internal implementations
  - ``InMemorySession`` — current behavior (fast, no persistence)
  - ``FileSession`` — JSON-file-backed persistence across restarts
  - ``SessionManager`` — lifecycle management with TTL cleanup
"""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Session Protocol — structural typing for third-party implementations
# ---------------------------------------------------------------------------

@runtime_checkable
class Session(Protocol):
    """Structural interface for session storage.

    Any class with matching methods works — no inheritance required.
    """

    @property
    def session_id(self) -> str: ...

    @property
    def created_at(self) -> float: ...

    @property
    def last_accessed(self) -> float: ...

    def get(self, key: str, default: Any = None) -> Any: ...

    def set(self, key: str, value: Any) -> None: ...

    def delete(self, key: str) -> None: ...

    def get_all(self) -> dict[str, Any]: ...

    def clear(self) -> None: ...

    def touch(self) -> None: ...


# ---------------------------------------------------------------------------
# Session ABC — base for internal implementations
# ---------------------------------------------------------------------------

class SessionABC(ABC):
    """Abstract base with shared behavior for internal session implementations."""

    def __init__(self, session_id: str):
        self._session_id = session_id
        self._created_at = time.time()
        self._last_accessed = self._created_at

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def created_at(self) -> float:
        return self._created_at

    @property
    def last_accessed(self) -> float:
        return self._last_accessed

    def touch(self) -> None:
        self._last_accessed = time.time()

    @property
    def age_seconds(self) -> float:
        return time.time() - self._created_at

    @property
    def idle_seconds(self) -> float:
        return time.time() - self._last_accessed

    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any: ...

    @abstractmethod
    def set(self, key: str, value: Any) -> None: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...

    @abstractmethod
    def get_all(self) -> dict[str, Any]: ...

    @abstractmethod
    def clear(self) -> None: ...


# ---------------------------------------------------------------------------
# InMemorySession — current behavior
# ---------------------------------------------------------------------------

class InMemorySession(SessionABC):
    """Fast in-memory session. Data lost on restart."""

    def __init__(self, session_id: str):
        super().__init__(session_id)
        self._data: dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        self.touch()
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.touch()
        self._data[key] = value

    def delete(self, key: str) -> None:
        self.touch()
        self._data.pop(key, None)

    def get_all(self) -> dict[str, Any]:
        self.touch()
        return dict(self._data)

    def clear(self) -> None:
        self._data.clear()


# ---------------------------------------------------------------------------
# FileSession — JSON-file-backed persistence
# ---------------------------------------------------------------------------

class FileSession(SessionABC):
    """Session backed by a JSON file for persistence across restarts."""

    def __init__(self, session_id: str, storage_dir: str | Path):
        super().__init__(session_id)
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._file_path = self._storage_dir / f"{session_id}.json"
        self._data: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if self._file_path.exists():
            try:
                raw = self._file_path.read_text(encoding="utf-8")
                envelope = json.loads(raw)
                self._data = envelope.get("data", {})
                self._created_at = envelope.get("created_at", self._created_at)
                self._last_accessed = envelope.get("last_accessed", self._last_accessed)
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def _save(self) -> None:
        envelope = {
            "session_id": self._session_id,
            "created_at": self._created_at,
            "last_accessed": self._last_accessed,
            "data": self._data,
        }
        tmp_path = self._file_path.with_suffix(".tmp")
        try:
            tmp_path.write_text(
                json.dumps(envelope, default=str, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp_path.replace(self._file_path)
        except OSError:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def get(self, key: str, default: Any = None) -> Any:
        self.touch()
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.touch()
        self._data[key] = value
        self._save()

    def delete(self, key: str) -> None:
        self.touch()
        self._data.pop(key, None)
        self._save()

    def get_all(self) -> dict[str, Any]:
        self.touch()
        return dict(self._data)

    def clear(self) -> None:
        self._data.clear()
        self._save()

    def destroy(self) -> None:
        """Remove the backing file permanently."""
        try:
            self._file_path.unlink(missing_ok=True)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# SessionManager — lifecycle with TTL
# ---------------------------------------------------------------------------

@dataclass
class SessionManagerConfig:
    """Configuration for session lifecycle management."""

    ttl_seconds: float = 86400.0  # 24 hours
    idle_ttl_seconds: float = 3600.0  # 1 hour idle timeout
    max_sessions: int = 1000
    storage_dir: str = ""
    use_file_sessions: bool = False


class SessionManager:
    """Manages session lifecycle with TTL-based cleanup."""

    def __init__(self, config: SessionManagerConfig | None = None):
        self._config = config or SessionManagerConfig()
        self._sessions: dict[str, SessionABC] = {}

    def create(self, session_id: str) -> SessionABC:
        """Create a new session."""
        if len(self._sessions) >= self._config.max_sessions:
            self._evict_oldest()

        if self._config.use_file_sessions and self._config.storage_dir:
            session = FileSession(session_id, self._config.storage_dir)
        else:
            session = InMemorySession(session_id)

        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> SessionABC | None:
        """Retrieve a session if it exists and hasn't expired."""
        session = self._sessions.get(session_id)
        if session is None:
            return None

        if self._is_expired(session):
            self._remove(session_id)
            return None

        session.touch()
        return session

    def delete(self, session_id: str) -> None:
        """Delete a session and clean up resources."""
        self._remove(session_id)

    def list_sessions(self) -> list[str]:
        """Return all active (non-expired) session IDs."""
        self.cleanup_expired()
        return list(self._sessions.keys())

    def cleanup_expired(self) -> int:
        """Remove expired sessions. Returns count of removed sessions."""
        expired = [
            sid for sid, s in self._sessions.items() if self._is_expired(s)
        ]
        for sid in expired:
            self._remove(sid)
        return len(expired)

    @property
    def count(self) -> int:
        return len(self._sessions)

    def _is_expired(self, session: SessionABC) -> bool:
        if session.age_seconds > self._config.ttl_seconds:
            return True
        if session.idle_seconds > self._config.idle_ttl_seconds:
            return True
        return False

    def _evict_oldest(self) -> None:
        if not self._sessions:
            return
        oldest_id = min(
            self._sessions, key=lambda sid: self._sessions[sid].last_accessed
        )
        self._remove(oldest_id)

    def _remove(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return
        session.clear()
        if isinstance(session, FileSession):
            session.destroy()
