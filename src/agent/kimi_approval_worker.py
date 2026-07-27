"""Long-running background worker that drains kimi tool-call approvals.

The per-stream auto-approver in `_stream_kimi._auto_approve_all` only runs
while a chat SSE stream is open, and historically only when certain WS
events arrived. MCP tools request approval *before* `tool.call.started`,
so without a poller they sit for 60s and expire with
"authorizeToolExecution hook failed".

This worker is the safety net: every `interval` seconds it drains pending
approvals through `kimi_security.decide`.

  - LOCAL mode: one shared kimi daemon (`kimi_daemon.base_url()`).
  - ONLINE mode: every live instance in `KimiSessionPool` (each has its
    own port / sandbox). Host `session_dir` is used for path policy, not
    the in-sandbox cwd (`/workspace`).

Lifecycle: started/stopped from the FastAPI lifespan in `api_server.py`.
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional

from agent.kimi_client import KimiAPIError, KimiClient
from agent.kimi_daemon import base_url
from agent.kimi_security import decide as security_decide
from config import get_config
from logger import get_logger

_logger = get_logger("agent.kimi_approval_worker")

_DEFAULT_INTERVAL = 2.0  # seconds — keep well under kimi's 60s approval TTL


class KimiApprovalWorker:
    def __init__(self, *, interval: float = _DEFAULT_INTERVAL) -> None:
        self.interval = interval
        self._task: asyncio.Task | None = None
        self._stop_event: asyncio.Event | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        if self.running:
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name="kimi-approval-worker")
        _logger.info("kimi approval worker started (interval=%ss)", self.interval)

    async def stop(self) -> None:
        if not self._task or not self._stop_event:
            return
        self._stop_event.set()
        try:
            await asyncio.wait_for(self._task, timeout=5.0)
        except asyncio.TimeoutError:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        except (asyncio.CancelledError, Exception):
            pass
        _logger.info("kimi approval worker stopped")

    async def _run(self) -> None:
        consecutive_failures = 0
        while self._stop_event and not self._stop_event.is_set():
            try:
                await self._tick_all()
                consecutive_failures = 0
            except Exception as exc:  # noqa: BLE001
                consecutive_failures += 1
                if consecutive_failures <= 2:
                    _logger.warning("approval worker tick failed: %s", exc)
                elif consecutive_failures == 3:
                    _logger.exception(
                        "approval worker tick repeatedly failing; "
                        "suppressing further stack traces and backing off"
                    )
            wait_s = self.interval * min(2 ** max(0, consecutive_failures - 1), 8)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=wait_s)
            except asyncio.TimeoutError:
                pass

    async def _tick_all(self) -> None:
        mode = get_config().server.mode or "local"
        if mode == "online":
            await self._tick_online_pool(mode)
        else:
            await self._tick_shared_daemon(mode)

    async def _tick_shared_daemon(self, mode: str) -> None:
        client = KimiClient(base_url=base_url())
        try:
            await self._drain_client(client, mode=mode, session_dir_override=None)
        finally:
            await client.aclose()

    async def _tick_online_pool(self, mode: str) -> None:
        try:
            from agent.kimi_session_pool import get_pool
            instances = get_pool().snapshot_instances()
        except Exception:
            _logger.debug("online pool unavailable for approval tick", exc_info=True)
            return
        for inst in instances:
            client = KimiClient(base_url=inst.base_url)
            try:
                await self._drain_client(
                    client,
                    mode=mode,
                    session_dir_override=inst.session_dir,
                )
            except Exception:  # noqa: BLE001
                _logger.debug(
                    "approval tick failed for pool instance port=%s",
                    inst.port, exc_info=True,
                )
            finally:
                await client.aclose()

    async def _drain_client(
        self,
        client: KimiClient,
        *,
        mode: str,
        session_dir_override: Optional[str],
    ) -> None:
        try:
            sessions = await client.list_sessions()
        except KimiAPIError as exc:
            if exc.code is not None:
                _logger.debug("list_sessions skipped (kimi code=%s)", exc.code)
            return
        for s in sessions:
            sid = s.get("id")
            if not sid:
                continue
            cwd = session_dir_override or str(
                (s.get("metadata") or {}).get("cwd") or "/tmp"
            )
            try:
                pending = await client.list_pending_approvals(sid)
            except KimiAPIError:
                continue
            except Exception:  # noqa: BLE001
                _logger.exception("list_pending_approvals(%s) failed", sid[:12])
                continue
            await self._process_pending(client, sid, cwd, mode, pending)

    async def _process_pending(
        self,
        client: KimiClient,
        sid: str,
        cwd: str,
        mode: str,
        pending: list[dict[str, Any]],
    ) -> None:
        for ap in pending:
            aid = ap.get("approval_id")
            if not aid:
                continue
            tool = str(ap.get("tool_name") or "")
            action = str(ap.get("action") or "")
            decision = security_decide(ap, session_dir=cwd, mode=mode)
            # ExitPlanMode (and any future needs_user tools) must reach the
            # chat UI — do not race the stream poller by deciding here.
            if decision.needs_user:
                _logger.debug(
                    "worker SKIP needs_user tool=%s sid=%s aid=%s",
                    tool, sid[:12], aid[:12],
                )
                continue
            if decision.allowed:
                try:
                    await client.approve(sid, aid, scope="session")
                    _logger.info(
                        "worker ALLOW tool=%s action=%s sid=%s aid=%s",
                        tool, action, sid[:12], aid[:12],
                    )
                except KimiAPIError as exc:
                    _logger.debug("worker approve %s noop: %s", aid[:12], exc)
                except Exception:  # noqa: BLE001
                    _logger.exception("worker approve POST failed for %s", aid)
            else:
                try:
                    await client.reject(
                        sid, aid,
                        feedback=f"Blocked by VenusFactory security policy: {decision.reason}",
                    )
                except KimiAPIError as exc:
                    _logger.debug("worker reject %s noop: %s", aid[:12], exc)
                except Exception:  # noqa: BLE001
                    _logger.exception("worker reject POST failed for %s", aid)
                _logger.warning(
                    "worker DENY tool=%s action=%s reason=%s sid=%s",
                    tool, action, decision.reason, sid[:12],
                )


_singleton = KimiApprovalWorker()


async def start_worker() -> None:
    await _singleton.start()


async def stop_worker() -> None:
    await _singleton.stop()
