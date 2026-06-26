"""Per-chat-session kimi-code daemon pool (online mode).

In ONLINE mode every VenusFactory chat session gets its own isolated
kimi-code daemon, spawned via `systemd-run --scope --property=DynamicUser=yes`
through the `venus-spawn-kimi` setuid helper. Each daemon:

  - runs as a freshly-allocated unprivileged UID (released on stop)
  - sees its session_dir bind-mounted at /workspace
  - has /home tmpfs'd away
  - cannot escape via mount/swap/reboot syscalls
  - listens on a port from the [58800..58899] pool

The pool enforces a hard concurrency cap (default 5) — the 6th
concurrent session triggers `PoolExhaustedError` which the chat router
maps to HTTP 503. Idle instances are GC'd after `idle_ttl` seconds
(default 600 = 10 min) of no acquire activity.

LOCAL mode (mode=="local") bypasses the pool entirely — see
`kimi_daemon.start_daemon()`. The chat router picks between them.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx

from logger import get_logger

_logger = get_logger("agent.kimi_session_pool")

# ── Tunables (env-overridable) ────────────────────────────────────────────
DEFAULT_POOL_CAP = int(os.environ.get("KIMI_POOL_MAX_SESSIONS", "5"))
DEFAULT_IDLE_TTL = float(os.environ.get("KIMI_POOL_IDLE_TTL_S", "600"))   # 10 min
DEFAULT_PORT_RANGE = (
    int(os.environ.get("KIMI_POOL_PORT_MIN", "58800")),
    int(os.environ.get("KIMI_POOL_PORT_MAX", "58899")),
)
HEALTHZ_TIMEOUT_S = 25.0
GC_TICK_S = 30.0   # how often the GC loop wakes

SPAWN_HELPER_PATHS = (
    "/usr/local/libexec/venus-spawn-kimi",
    # Fallback to repo path for dev/testing without sudo install.
    str(Path(__file__).resolve().parent.parent.parent
        / "scripts" / "install" / "venus-spawn-kimi"),
)


class PoolExhaustedError(RuntimeError):
    """Raised when all pool slots are in use. Map to HTTP 503."""


class SpawnError(RuntimeError):
    """venus-spawn-kimi exited non-zero or didn't report uid/scope."""


@dataclass
class KimiInstance:
    session_id: str           # VenusFactory chat session UUID
    uid: int                  # systemd DynamicUser allocated UID
    port: int                 # 58800..58899
    session_dir: str          # absolute host path bind-mounted to /workspace
    scope_name: str           # systemd scope unit name
    base_url: str             # http://127.0.0.1:<port>
    started_at: float         # monotonic seconds
    last_used: float          # monotonic seconds — refreshed on each acquire


@dataclass
class _PortPool:
    """Thread-unsafe (caller holds the pool lock)."""
    available: list[int] = field(default_factory=list)

    @classmethod
    def from_range(cls, lo: int, hi: int, cap: int) -> "_PortPool":
        # Only allocate as many ports as the concurrency cap — extra ports
        # are wasted slots in `available` and would let a misconfigured cap
        # let through more than expected.
        return cls(available=list(range(lo, lo + cap)))

    def alloc(self) -> int:
        if not self.available:
            raise PoolExhaustedError("port pool empty")
        return self.available.pop(0)

    def free(self, port: int) -> None:
        if port not in self.available:
            self.available.append(port)
            self.available.sort()


# ── Helpers ───────────────────────────────────────────────────────────────


def _find_spawn_helper() -> str:
    for p in SPAWN_HELPER_PATHS:
        if os.access(p, os.X_OK):
            return p
    raise SpawnError(
        "venus-spawn-kimi helper not found. Install with "
        "`sudo bash scripts/install/setup-online-mode.sh`."
    )


async def _spawn_via_systemd_run(
    session_id: str, port: int, session_dir: str,
) -> tuple[int, str]:
    """Invoke the setuid helper via sudo. Returns (uid, scope_name).

    Helper prints two lines: `uid=<n>` and `scope=<name>`. Anything else on
    stdout is treated as an error.
    """
    helper = _find_spawn_helper()
    # `sudo -n` = non-interactive; fails fast if NOPASSWD isn't set up.
    cmd = ["sudo", "-n", helper, session_id, str(port), session_dir]
    _logger.info("spawning kimi via: %s", " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise SpawnError(
            f"venus-spawn-kimi exited {proc.returncode}: "
            f"stderr={stderr.decode('utf-8', errors='replace')[:400]}"
        )
    uid_val: Optional[str] = None
    scope_val: Optional[str] = None
    for line in stdout.decode("utf-8", errors="replace").splitlines():
        if line.startswith("uid="):
            uid_val = line[4:].strip()
        elif line.startswith("scope="):
            scope_val = line[6:].strip()
    if not scope_val:
        raise SpawnError(f"helper did not report scope name: stdout={stdout!r}")
    try:
        uid = int(uid_val) if uid_val and uid_val.isdigit() else -1
    except ValueError:
        uid = -1
    return uid, scope_val


async def _stop_scope(scope_name: str) -> None:
    """Best-effort `sudo systemctl stop <scope>`. Logged but never raises."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "sudo", "-n", "/bin/systemctl", "stop", scope_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            _logger.warning(
                "systemctl stop %s exited %s: %s",
                scope_name, proc.returncode,
                stderr.decode("utf-8", errors="replace")[:300],
            )
    except Exception:
        _logger.exception("systemctl stop %s failed", scope_name)


async def _wait_healthz(port: int, timeout: float = HEALTHZ_TIMEOUT_S) -> None:
    url = f"http://127.0.0.1:{port}/api/v1/healthz"
    deadline = asyncio.get_event_loop().time() + timeout
    last_err: Exception | None = None
    async with httpx.AsyncClient(timeout=2.0) as c:
        while asyncio.get_event_loop().time() < deadline:
            try:
                r = await c.get(url)
                if r.status_code == 200 and (r.json().get("data") or {}).get("ok"):
                    return
            except Exception as exc:
                last_err = exc
            await asyncio.sleep(0.3)
    raise SpawnError(
        f"kimi on port {port} didn't become healthy in {timeout}s "
        f"(last error: {last_err})"
    )


def _project_session_dir(session_id: str) -> str:
    """The host path we bind into /workspace inside the sandbox."""
    root = Path(__file__).resolve().parent.parent.parent / "temp_outputs" / "web_v2" / "sessions"
    return str((root / session_id).resolve())


# ── The pool ──────────────────────────────────────────────────────────────


class KimiSessionPool:
    def __init__(
        self,
        *,
        max_sessions: int = DEFAULT_POOL_CAP,
        idle_ttl: float = DEFAULT_IDLE_TTL,
        port_range: tuple[int, int] = DEFAULT_PORT_RANGE,
    ) -> None:
        self._max = max_sessions
        self._idle_ttl = idle_ttl
        self._instances: dict[str, KimiInstance] = {}
        self._ports = _PortPool.from_range(port_range[0], port_range[1], max_sessions)
        self._lock = asyncio.Lock()
        self._gc_task: Optional[asyncio.Task[None]] = None
        self._stopped = False

    @property
    def active_count(self) -> int:
        return len(self._instances)

    async def acquire(self, session_id: str) -> KimiInstance:
        """Get the kimi instance for this chat session, spawning if needed.

        Raises `PoolExhaustedError` if a NEW spawn would exceed `max_sessions`.
        Existing sessions are always returned (refreshes `last_used`).
        """
        async with self._lock:
            inst = self._instances.get(session_id)
            if inst is not None:
                inst.last_used = time.monotonic()
                return inst
            if len(self._instances) >= self._max:
                raise PoolExhaustedError(
                    f"online pool full: {self._max} sessions active "
                    f"(idle TTL is {int(self._idle_ttl)}s)"
                )
            # Allocate port up-front so failure path doesn't leak it.
            try:
                port = self._ports.alloc()
            except PoolExhaustedError:
                raise

        # Spawn outside the lock — DynamicUser allocation can take 100ms+
        # and we don't want a slow spawn to block acquires for other sessions.
        try:
            session_dir = _project_session_dir(session_id)
            os.makedirs(session_dir, mode=0o2770, exist_ok=True)
            uid, scope = await _spawn_via_systemd_run(session_id, port, session_dir)
            await _wait_healthz(port)
        except Exception:
            # Rollback the port allocation on any failure.
            async with self._lock:
                self._ports.free(port)
            raise

        now = time.monotonic()
        inst = KimiInstance(
            session_id=session_id,
            uid=uid,
            port=port,
            session_dir=session_dir,
            scope_name=scope,
            base_url=f"http://127.0.0.1:{port}",
            started_at=now,
            last_used=now,
        )
        async with self._lock:
            # Race: another acquire for the same session might have spawned
            # while we were outside the lock. Prefer the one already there.
            if session_id in self._instances:
                _logger.warning(
                    "race: two spawns for session %s; stopping the duplicate %s",
                    session_id, scope,
                )
                # Stop our just-spawned dup and release its port.
                self._ports.free(port)
                # Fire-and-forget stop; the existing instance wins.
                asyncio.create_task(_stop_scope(scope))
                return self._instances[session_id]
            self._instances[session_id] = inst
        _logger.info(
            "kimi pool: spawned session=%s uid=%s port=%s scope=%s (active=%d/%d)",
            session_id[:8], uid, port, scope, len(self._instances), self._max,
        )
        return inst

    async def release(self, session_id: str) -> None:
        """Mark this session idle. Doesn't stop the kimi instance — GC does
        that after `idle_ttl` seconds of no acquire."""
        async with self._lock:
            inst = self._instances.get(session_id)
            if inst is not None:
                inst.last_used = time.monotonic()

    async def stop_session(self, session_id: str) -> bool:
        """Explicitly stop and remove the kimi instance for a session."""
        async with self._lock:
            inst = self._instances.pop(session_id, None)
            if inst is not None:
                self._ports.free(inst.port)
        if inst is None:
            return False
        await _stop_scope(inst.scope_name)
        _logger.info(
            "kimi pool: stopped session=%s scope=%s (active=%d/%d)",
            session_id[:8], inst.scope_name, len(self._instances), self._max,
        )
        return True

    async def start_gc(self) -> None:
        if self._gc_task is not None:
            return
        self._gc_task = asyncio.create_task(self._gc_loop())

    async def stop_gc(self) -> None:
        self._stopped = True
        if self._gc_task is not None:
            self._gc_task.cancel()
            try:
                await self._gc_task
            except (asyncio.CancelledError, Exception):
                pass
            self._gc_task = None

    async def shutdown_all(self) -> None:
        """Stop every active scope. Called from fastapi lifespan shutdown."""
        await self.stop_gc()
        async with self._lock:
            snap = list(self._instances.values())
            self._instances.clear()
            for inst in snap:
                self._ports.free(inst.port)
        for inst in snap:
            await _stop_scope(inst.scope_name)
        _logger.info("kimi pool: shutdown complete (%d instances stopped)", len(snap))

    async def _gc_loop(self) -> None:
        while not self._stopped:
            try:
                await asyncio.sleep(GC_TICK_S)
                now = time.monotonic()
                async with self._lock:
                    expired = [
                        sid for sid, inst in self._instances.items()
                        if now - inst.last_used > self._idle_ttl
                    ]
                for sid in expired:
                    _logger.info(
                        "kimi pool: GC stopping session=%s (idle > %ds)",
                        sid[:8], int(self._idle_ttl),
                    )
                    await self.stop_session(sid)
            except asyncio.CancelledError:
                break
            except Exception:
                _logger.exception("kimi pool GC tick failed")


# ── Module-level singleton ────────────────────────────────────────────────


_singleton: Optional[KimiSessionPool] = None


def get_pool() -> KimiSessionPool:
    global _singleton
    if _singleton is None:
        _singleton = KimiSessionPool()
    return _singleton


def reset_pool_for_tests() -> None:
    """Test helper: drop the singleton without touching its state. Caller
    must have already shut down any running instances."""
    global _singleton
    _singleton = None
