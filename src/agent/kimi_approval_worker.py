"""Long-running background worker that drains kimi tool-call approvals.

The per-stream auto-approver in `_stream_kimi._auto_approve_all` only runs
while a chat SSE stream is open. If the backend restarts, the user closes
the tab, or the agent goes through a >60 s thinking/tool pause without any
new event, kimi's approval requests time out and the agent stalls with
"authorizeToolExecution hook failed".

This worker is the safety net: every `interval` seconds it lists EVERY kimi
session, GETs its pending approvals, and runs each through `kimi_security.
decide`. Approved ones go via `POST /approvals/{id}` (scope=session, so the
same tool only needs one approval per session). Denied ones are rejected
with a human-readable reason as `feedback` so the agent learns what got
blocked and can plan around it.

Lifecycle: started/stopped from the FastAPI lifespan in `api_server.py`.
"""
from __future__ import annotations

import asyncio
from typing import Any

from agent.kimi_client import KimiAPIError, KimiClient
from agent.kimi_daemon import base_url
from agent.kimi_security import decide as security_decide
from config import get_config
from logger import get_logger

_logger = get_logger("agent.kimi_approval_worker")

_DEFAULT_INTERVAL = 4.0  # seconds


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
        client = KimiClient(base_url=base_url())
        consecutive_failures = 0
        try:
            while self._stop_event and not self._stop_event.is_set():
                try:
                    await self._tick(client)
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
                # Exponential backoff after repeated failures (max 8× the
                # base interval) so a dead kimi server doesn't hammer the
                # logs. wait_for + stop_event keeps cancellation responsive.
                wait_s = self.interval * min(2 ** max(0, consecutive_failures - 1), 8)
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=wait_s)
                except asyncio.TimeoutError:
                    pass
        finally:
            await client.aclose()

    async def _tick(self, client: KimiClient) -> None:
        mode = get_config().server.mode or "local"
        try:
            sessions = await client.list_sessions()
        except KimiAPIError as exc:
            # Common: kimi just started and has no provider configured.
            # Don't spam the log every tick.
            if exc.code is not None:
                _logger.debug("list_sessions skipped (kimi code=%s)", exc.code)
            return
        for s in sessions:
            sid = s.get("id")
            if not sid:
                continue
            # `metadata.cwd` is the kimi session's working dir == our
            # agent_session_dir; the security policy needs it to validate
            # path-scoped operations.
            cwd = str((s.get("metadata") or {}).get("cwd") or "/tmp")
            try:
                pending = await client.list_pending_approvals(sid)
            except KimiAPIError:
                # Session may have been deleted between list and pending
                # query, or kimi internal state is partially recovered after
                # a restart. Both are benign — skip silently.
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
            if decision.allowed:
                try:
                    await client.approve(sid, aid, scope="session")
                    _logger.info(
                        "worker ALLOW tool=%s action=%s sid=%s aid=%s",
                        tool, action, sid[:12], aid[:12],
                    )
                except KimiAPIError as exc:
                    # session.not_found / approval already resolved / etc.
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
