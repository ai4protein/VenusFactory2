"""Per-chat-session kimi-code daemon pool (online mode).

In ONLINE mode every VenusFactory chat session gets its own kimi-code
daemon on a port from the [58800..58899] pool.

Spawn modes (`KIMI_ONLINE_SPAWN_MODE`):

  - ``sandbox`` — `systemd-run` + `DynamicUser` via `venus-spawn-kimi`
    (requires passwordless sudo / setup-online-mode.sh). Strongest OS isolation.
  - ``bwrap`` — bubblewrap user-namespace sandbox (no sudo). Hides host
    `/home`, bind-mounts only the session dir at `/workspace`, read-only
    system roots. Preferred on cluster nodes without sudo.
  - ``user`` — same-user child process (no OS sandbox). App-level
    `kimi_security` only; last resort.
  - ``auto`` (default) — ``sandbox`` if passwordless sudo works, else
    ``bwrap`` if available, else ``user``.

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
import shutil
import signal
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

_USERPROC_PREFIX = "userproc:"
_BWRAP_PREFIX = "bwrap:"
_sandbox_available_cache: Optional[bool] = None
_bwrap_available_cache: Optional[bool] = None
_resolved_spawn_mode: Optional[str] = None


class PoolExhaustedError(RuntimeError):
    """Raised when all pool slots are in use. Map to HTTP 503."""


class SpawnError(RuntimeError):
    """venus-spawn-kimi exited non-zero or didn't report uid/scope."""


def _find_spawn_helper() -> str:
    for p in SPAWN_HELPER_PATHS:
        if os.access(p, os.X_OK):
            return p
    raise SpawnError(
        "venus-spawn-kimi helper not found. Install with "
        "`sudo bash scripts/install/setup-online-mode.sh`, "
        "or use KIMI_ONLINE_SPAWN_MODE=bwrap (no sudo)."
    )


def _sandbox_available() -> bool:
    """True when passwordless sudo can invoke the spawn helper."""
    global _sandbox_available_cache
    if _sandbox_available_cache is not None:
        return _sandbox_available_cache
    try:
        helper = _find_spawn_helper()
    except SpawnError:
        _sandbox_available_cache = False
        return False
    try:
        # No args → helper exits 2 after arg validation if sudo auth succeeded.
        proc = subprocess.run(
            ["sudo", "-n", helper],
            capture_output=True,
            text=True,
            timeout=3,
        )
        stderr = (proc.stderr or "").lower()
        if "password is required" in stderr or "a terminal is required" in stderr:
            _sandbox_available_cache = False
        elif proc.returncode == 2:
            _sandbox_available_cache = True
        else:
            # Unexpected but treat non-auth failures as unavailable.
            _sandbox_available_cache = False
            _logger.warning(
                "sandbox probe: sudo helper rc=%s stderr=%s",
                proc.returncode, (proc.stderr or "")[:200],
            )
    except Exception as exc:
        _logger.info("sandbox probe failed (%s)", exc)
        _sandbox_available_cache = False
    return _sandbox_available_cache


def _bwrap_available() -> bool:
    """True when bubblewrap can create an unprivileged user namespace."""
    global _bwrap_available_cache
    if _bwrap_available_cache is not None:
        return _bwrap_available_cache
    bwrap = shutil.which("bwrap")
    if not bwrap:
        _bwrap_available_cache = False
        return False
    # Minimal filesystem so /bin/true + its dynamic linker resolve.
    # On Debian/Ubuntu /bin and /lib* are symlinks into /usr; binding them
    # (not just --symlink) is required for the ELF interpreter path.
    probe_cmd = [
        bwrap,
        "--unshare-user",
        "--die-with-parent",
        "--ro-bind", "/usr", "/usr",
        "--ro-bind", "/bin", "/bin",
        "--ro-bind", "/lib", "/lib",
        "--ro-bind", "/lib64", "/lib64",
        "--ro-bind", "/etc", "/etc",
        "--dev", "/dev",
        "--",
        "/bin/true",
    ]
    try:
        proc = subprocess.run(
            probe_cmd,
            capture_output=True,
            text=True,
            timeout=3,
        )
        _bwrap_available_cache = proc.returncode == 0
        if not _bwrap_available_cache:
            _logger.warning(
                "bwrap probe failed rc=%s stderr=%s",
                proc.returncode, (proc.stderr or "")[:200],
            )
    except Exception as exc:
        _logger.info("bwrap probe failed (%s)", exc)
        _bwrap_available_cache = False
    return _bwrap_available_cache


def _pick_auto_spawn_mode() -> str:
    if _sandbox_available():
        return "sandbox"
    if _bwrap_available():
        return "bwrap"
    return "user"


def resolve_spawn_mode() -> str:
    """Return effective spawn mode: ``sandbox``, ``bwrap``, or ``user``."""
    global _resolved_spawn_mode
    if _resolved_spawn_mode is not None:
        return _resolved_spawn_mode
    raw = (os.environ.get("KIMI_ONLINE_SPAWN_MODE") or "auto").strip().lower()
    if raw in ("sandbox", "bwrap", "user"):
        mode = raw
    elif raw == "auto":
        mode = _pick_auto_spawn_mode()
    else:
        _logger.warning("Unknown KIMI_ONLINE_SPAWN_MODE=%r; using auto", raw)
        mode = _pick_auto_spawn_mode()

    if mode == "sandbox" and not _sandbox_available():
        fallback = "bwrap" if _bwrap_available() else "user"
        _logger.warning(
            "KIMI_ONLINE_SPAWN_MODE=sandbox unavailable (no passwordless sudo); "
            "falling back to %s",
            fallback,
        )
        mode = fallback
    if mode == "bwrap" and not _bwrap_available():
        _logger.warning(
            "KIMI_ONLINE_SPAWN_MODE=bwrap unavailable; falling back to user "
            "(no OS isolation)"
        )
        mode = "user"

    if mode == "user":
        _logger.warning(
            "kimi online spawn mode=user (no OS sandbox). "
            "Prefer KIMI_ONLINE_SPAWN_MODE=bwrap or setup-online-mode.sh."
        )
    else:
        _logger.info("kimi online spawn mode=%s", mode)
    _resolved_spawn_mode = mode
    return mode


def _kimi_bin() -> str:
    from agent.kimi_daemon import _kimi_bin as _daemon_kimi_bin
    return _daemon_kimi_bin()


def _seed_kimi_config(session_dir: str) -> None:
    """Copy host kimi auth/MCP config into the per-session HOME tree."""
    src_root = Path(
        os.environ.get("KIMI_CONFIG_DIR") or (Path.home() / ".kimi-code")
    )
    dst_root = Path(session_dir) / ".kimi-code"
    if not src_root.is_dir():
        _logger.warning("kimi config source missing: %s", src_root)
        return
    dst_root.mkdir(parents=True, exist_ok=True)
    (dst_root / "sessions").mkdir(exist_ok=True)
    for item in ("config.toml", "mcp.json", "providers", "credentials.json", "auth.json"):
        src = src_root / item
        if not src.exists():
            continue
        dst = dst_root / item
        try:
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        except Exception:
            _logger.exception("failed to seed kimi config item %s", item)


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


async def _drain_proc(
    proc: asyncio.subprocess.Process, session_id: str, label: str,
) -> None:
    if proc.stdout is None:
        return
    prefix = session_id[:8]
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        _logger.info(
            "[kimi-%s:%s] %s",
            label,
            prefix,
            line.decode("utf-8", errors="replace").rstrip(),
        )


async def _await_health_or_kill(
    proc: asyncio.subprocess.Process, port: int,
) -> None:
    try:
        await _wait_healthz(port)
    except Exception:
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except Exception:
                pass
        raise


def _kimi_install_root(kimi_bin: str) -> Path:
    """Directory that should be mounted RO at /opt/kimi-code inside bwrap."""
    custom = os.environ.get("KIMI_CONFIG_DIR")
    if custom:
        return Path(custom).expanduser().resolve()
    bin_path = Path(kimi_bin).expanduser().resolve()
    # Typical layout: ~/.kimi-code/bin/kimi
    if bin_path.parent.name == "bin":
        return bin_path.parent.parent
    return Path.home() / ".kimi-code"


def _runtime_env_root() -> Optional[Path]:
    runtime_bin = os.environ.get(
        "KIMI_RUNTIME_PYTHON_BIN",
        "/home/tanyang/miniconda3/envs/agent/bin",
    )
    path = Path(runtime_bin).expanduser()
    if path.is_dir():
        # .../envs/agent/bin → .../envs/agent
        return path.resolve().parent if path.name == "bin" else path.resolve()
    return None


def _bwrap_ensure_parent_dirs(cmd: list[str], abs_path: str) -> None:
    """Append ``--dir`` for every parent of ``abs_path`` (sandbox mkdir -p)."""
    parts = Path(abs_path).resolve().parts  # ('/', 'home', ...)
    cur = Path("/")
    # Skip the final component — it will be created by the bind mount.
    for part in parts[1:-1]:
        cur = cur / part
        cmd.extend(["--dir", str(cur)])


def _build_bwrap_cmd(
    *, port: int, session_dir: str, kimi_bin: str,
) -> list[str]:
    """Assemble bubblewrap argv for an isolated kimi server."""
    bwrap = shutil.which("bwrap") or "bwrap"
    install_root = _kimi_install_root(kimi_bin)
    if not (install_root / "bin" / "kimi").is_file() and not Path(kimi_bin).is_file():
        raise SpawnError(f"kimi binary not found under {install_root} or {kimi_bin!r}")

    session_abs = str(Path(session_dir).resolve())

    cmd: list[str] = [
        bwrap,
        "--unshare-user",
        "--uid", str(os.getuid()),
        "--gid", str(os.getgid()),
        "--unshare-pid",
        "--die-with-parent",
        "--new-session",
        # Keep host network so kimi can reach LLM APIs + local MCP.
        "--ro-bind", "/usr", "/usr",
        "--ro-bind", "/etc", "/etc",
    ]
    for rel in ("bin", "sbin", "lib", "lib64"):
        host = Path("/") / rel
        if host.exists():
            cmd.extend(["--ro-bind", str(host), str(host)])

    cmd.extend([
        "--dev", "/dev",
        "--proc", "/proc",
        "--tmpfs", "/tmp",
        "--tmpfs", "/home",
        "--dir", "/opt",
        "--ro-bind", str(install_root), "/opt/kimi-code",
    ])
    # MCP tools run on the host and return host-absolute paths under
    # session_dir (e.g. .../temp_outputs/web_v2/sessions/<id>/.../x.fasta).
    # With only /workspace mounted, `Read` of those host paths fails because
    # --tmpfs /home hides them. Bind the same directory at BOTH /workspace
    # and its real host path so either form works inside the sandbox.
    _bwrap_ensure_parent_dirs(cmd, session_abs)
    cmd.extend([
        "--bind", session_abs, "/workspace",
        "--bind", session_abs, session_abs,
        "--chdir", "/workspace",
        "--clearenv",
        "--setenv", "HOME", "/workspace",
        "--setenv", "KIMI_PORT", str(port),
        "--setenv", "LANG", os.environ.get("LANG") or "C.UTF-8",
        "--setenv", "TERM", "xterm-256color",
    ])
    # /etc/resolv.conf is often a symlink into systemd-resolved; without this
    # bind, DNS fails inside the sandbox and kimi cannot reach LLM APIs.
    resolve_dir = Path("/run/systemd/resolve")
    if resolve_dir.is_dir():
        cmd.extend([
            "--dir", "/run",
            "--dir", "/run/systemd",
            "--ro-bind", str(resolve_dir), str(resolve_dir),
        ])

    path_entries = ["/opt/kimi-code/bin"]
    runtime_root = _runtime_env_root()
    if runtime_root is not None and runtime_root.is_dir():
        cmd.extend(["--ro-bind", str(runtime_root), "/opt/venus-runtime"])
        path_entries.append("/opt/venus-runtime/bin")
    path_entries.extend(["/usr/local/bin", "/usr/bin", "/bin"])
    cmd.extend(["--setenv", "PATH", ":".join(path_entries)])

    cmd.extend([
        "--",
        "/opt/kimi-code/bin/kimi",
        "server", "run",
        "--port", str(port),
        "--foreground",
        "--log-level", os.environ.get("KIMI_LOG_LEVEL", "info"),
    ])
    return cmd


async def _spawn_via_bwrap(
    session_id: str, port: int, session_dir: str,
) -> tuple[int, str]:
    """Spawn kimi inside bubblewrap (no sudo). Returns (uid, bwrap:<pid>)."""
    os.makedirs(session_dir, exist_ok=True)
    _seed_kimi_config(session_dir)
    kimi_bin = _kimi_bin()
    cmd = _build_bwrap_cmd(port=port, session_dir=session_dir, kimi_bin=kimi_bin)
    _logger.info(
        "spawning bwrap kimi: port=%s session=%s",
        port, session_id[:8],
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except FileNotFoundError as exc:
        raise SpawnError(
            "bwrap or kimi binary not found. Install bubblewrap / kimi-code, "
            "or set KIMI_BIN."
        ) from exc
    asyncio.create_task(_drain_proc(proc, session_id, "bwrap"))
    await _await_health_or_kill(proc, port)
    if proc.pid is None:
        raise SpawnError("bwrap kimi spawn produced no pid")
    return os.getuid(), f"{_BWRAP_PREFIX}{proc.pid}"


async def _spawn_user_process(
    session_id: str, port: int, session_dir: str,
) -> tuple[int, str]:
    """Spawn kimi as the current user (no OS sandbox).

    Returns (uid, scope_name) where scope_name is ``userproc:<pid>`` so
    `_stop_scope` can SIGTERM the child without systemctl.
    """
    os.makedirs(session_dir, exist_ok=True)
    _seed_kimi_config(session_dir)
    kimi_bin = _kimi_bin()
    cmd = [
        kimi_bin, "server", "run",
        "--port", str(port),
        "--foreground",
        "--log-level", os.environ.get("KIMI_LOG_LEVEL", "info"),
    ]
    env = os.environ.copy()
    env["HOME"] = session_dir
    env["KIMI_PORT"] = str(port)
    runtime_root = _runtime_env_root()
    if runtime_root is not None:
        runtime_bin = runtime_root / "bin"
        if runtime_bin.is_dir():
            env["PATH"] = str(runtime_bin) + os.pathsep + env.get("PATH", "")
    _logger.info(
        "spawning user-level kimi: port=%s session=%s bin=%s",
        port, session_id[:8], kimi_bin,
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            cwd=session_dir,
        )
    except FileNotFoundError as exc:
        raise SpawnError(
            f"kimi binary not found at {kimi_bin!r}. "
            "Install kimi-code or set KIMI_BIN."
        ) from exc
    asyncio.create_task(_drain_proc(proc, session_id, "user"))
    await _await_health_or_kill(proc, port)
    if proc.pid is None:
        raise SpawnError("user-level kimi spawn produced no pid")
    return os.getuid(), f"{_USERPROC_PREFIX}{proc.pid}"


async def _stop_pid(pid: int) -> None:
    """SIGTERM/SIGKILL a tracked child pid. Never raises."""
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        _logger.exception("SIGTERM pid=%s failed", pid)
        return
    for _ in range(20):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        await asyncio.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


async def _stop_scope(scope_name: str) -> None:
    """Best-effort stop of sandbox / bwrap / userproc. Never raises."""
    for prefix in (_USERPROC_PREFIX, _BWRAP_PREFIX):
        if scope_name.startswith(prefix):
            try:
                pid = int(scope_name[len(prefix):])
            except ValueError:
                _logger.warning("bad pid scope name: %s", scope_name)
                return
            await _stop_pid(pid)
            return
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

    def snapshot_instances(self) -> list[KimiInstance]:
        """Return a shallow copy of currently active instances (no lock wait).

        Used by the online approval worker to poll each sandboxed kimi.
        Callers must treat the list as a point-in-time snapshot — instances
        may be GC'd immediately afterwards.
        """
        return list(self._instances.values())

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

        # Spawn outside the lock — DynamicUser / process start can take 100ms+
        # and we don't want a slow spawn to block acquires for other sessions.
        try:
            session_dir = _project_session_dir(session_id)
            os.makedirs(session_dir, mode=0o2770, exist_ok=True)
            spawn_mode = resolve_spawn_mode()
            if spawn_mode == "sandbox":
                uid, scope = await _spawn_via_systemd_run(
                    session_id, port, session_dir,
                )
            elif spawn_mode == "bwrap":
                uid, scope = await _spawn_via_bwrap(
                    session_id, port, session_dir,
                )
            else:
                uid, scope = await _spawn_user_process(
                    session_id, port, session_dir,
                )
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
    global _singleton, _sandbox_available_cache, _bwrap_available_cache
    global _resolved_spawn_mode
    _singleton = None
    _sandbox_available_cache = None
    _bwrap_available_cache = None
    _resolved_spawn_mode = None
