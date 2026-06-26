"""Manage the kimi-code server child process from the FastAPI lifespan.

Default: spawn `kimi server run --port $KIMI_PORT --foreground` and wait until
`/api/v1/healthz` returns ok. Stop with SIGTERM then SIGKILL fallback.

Escape hatch: set `KIMI_EXTERNAL=1` to skip spawning entirely. The caller is
expected to have a kimi server already running, and `KIMI_BASE_URL` may be set
to point elsewhere (defaults to http://127.0.0.1:$KIMI_PORT).
"""
from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path
from typing import Optional

import httpx

from logger import get_logger

_logger = get_logger("agent.kimi_daemon")

DEFAULT_KIMI_BIN = "/home/tanyang/.kimi-code/bin/kimi"
DEFAULT_PORT = 58627


def _kimi_bin() -> str:
    custom = os.environ.get("KIMI_BIN")
    if custom:
        return custom
    if Path(DEFAULT_KIMI_BIN).is_file():
        return DEFAULT_KIMI_BIN
    return "kimi"  # rely on PATH


def kimi_port() -> int:
    raw = os.environ.get("KIMI_PORT")
    try:
        return int(raw) if raw else DEFAULT_PORT
    except ValueError:
        return DEFAULT_PORT


def is_external() -> bool:
    return os.environ.get("KIMI_EXTERNAL", "").strip().lower() in ("1", "true", "yes")


def base_url() -> str:
    custom = os.environ.get("KIMI_BASE_URL")
    if custom:
        return custom.rstrip("/")
    return f"http://127.0.0.1:{kimi_port()}"


class KimiDaemon:
    def __init__(self) -> None:
        self._proc: Optional[asyncio.subprocess.Process] = None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def start(self, timeout: float = 20.0) -> None:
        if is_external():
            _logger.info("KIMI_EXTERNAL=1; not spawning kimi server. Probing %s", base_url())
            await self._wait_healthy(timeout=timeout)
            return

        if self.running:
            _logger.info("kimi daemon already running (pid=%s)", self._proc.pid)  # type: ignore[union-attr]
            return

        cmd = [_kimi_bin(), "server", "run",
               "--port", str(kimi_port()),
               "--foreground",
               "--log-level", os.environ.get("KIMI_LOG_LEVEL", "info")]
        # When kimi shells out (Bash tool, scripts), the spawned shell uses
        # PATH to resolve `python`/`pip`. On this host /opt/ADFRsuite/bin
        # ships first and points at a broken Python 2.7, so `import Bio` /
        # `from pymol import cmd` fail. Prepend our agent conda env so kimi
        # picks up the working interpreter that already has BioPython and
        # pymol-open-source installed. Override via KIMI_RUNTIME_PYTHON_BIN.
        env = os.environ.copy()
        runtime_bin = os.environ.get(
            "KIMI_RUNTIME_PYTHON_BIN",
            "/home/tanyang/miniconda3/envs/agent/bin",
        )
        if runtime_bin and Path(runtime_bin).is_dir():
            env["PATH"] = runtime_bin + os.pathsep + env.get("PATH", "")
        _logger.info("Starting kimi daemon: %s (PATH head=%s)", " ".join(cmd), runtime_bin)
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"kimi binary not found at {_kimi_bin()!r}. "
                "Install with: curl -fsSL https://code.kimi.com/kimi-code/install.sh | bash, "
                "or set KIMI_BIN / KIMI_EXTERNAL=1."
            ) from exc

        # Drain stdout to logger in background so the pipe doesn't fill up.
        asyncio.create_task(self._drain_output())

        try:
            await self._wait_healthy(timeout=timeout)
        except Exception:
            await self.stop()
            raise

    async def _drain_output(self) -> None:
        if self._proc is None or self._proc.stdout is None:
            return
        while True:
            line = await self._proc.stdout.readline()
            if not line:
                break
            _logger.info("[kimi] %s", line.decode("utf-8", errors="replace").rstrip())

    async def _wait_healthy(self, timeout: float) -> None:
        url = f"{base_url()}/api/v1/healthz"
        deadline = asyncio.get_event_loop().time() + timeout
        last_err: Exception | None = None
        async with httpx.AsyncClient(timeout=2.0) as client:
            while asyncio.get_event_loop().time() < deadline:
                try:
                    r = await client.get(url)
                    if r.status_code == 200 and (r.json().get("data") or {}).get("ok"):
                        _logger.info("kimi server healthy at %s", url)
                        return
                except Exception as exc:
                    last_err = exc
                await asyncio.sleep(0.3)
        raise RuntimeError(f"kimi server did not become healthy at {url} within {timeout}s; last error: {last_err}")

    async def stop(self) -> None:
        if not self.running:
            return
        proc = self._proc
        assert proc is not None
        _logger.info("Stopping kimi daemon (pid=%s)", proc.pid)
        try:
            proc.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            self._proc = None
            return
        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            _logger.warning("kimi daemon did not exit on SIGTERM; sending SIGKILL")
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                _logger.error("kimi daemon ignored SIGKILL")
        self._proc = None


_singleton: KimiDaemon = KimiDaemon()


async def start_daemon(timeout: float = 20.0) -> None:
    await _singleton.start(timeout=timeout)


async def stop_daemon() -> None:
    await _singleton.stop()
