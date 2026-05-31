"""Shared trace + token-streaming primitives used by graph nodes."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from typing import Any

from langgraph.config import get_stream_writer

from agent.tracing import NoOpTrace, Scope, create_trace
from logger import get_logger

_logger = get_logger("agent.graph")


@contextmanager
def _ensure_trace(session_id: str = ""):
    """Ensure a Trace is active on the current asyncio context.

    If a real trace is already current (e.g. started by an outer caller), this
    is a no-op and yields the existing trace. Otherwise it starts a fresh trace
    from the global ``TracingProvider`` — which returns ``NoOpTrace`` (zero
    overhead) when no processor is registered.

    Wrapping the top-level graph nodes (plan/execute) in this guarantees that
    spans opened inside the node tree have a parent trace to attach to without
    forcing every caller (chat_api.py, tests, scripts) to manage tracing.
    """
    current = Scope.get_current_trace()
    if not isinstance(current, NoOpTrace):
        yield current
        return
    trace = create_trace(session_id=session_id)
    with trace as t:
        yield t


async def _stream_chain(chain, inputs: dict[str, Any], role_id: str = "assistant") -> str:
    """Invoke a chain with token-level streaming via get_stream_writer().

    Runs chain.stream() in a thread, pipes tokens through a queue to the
    async stream writer so they reach the SSE connection in real-time.
    Falls back to non-streaming invoke on error.
    """
    import queue as _queue

    writer = get_stream_writer()
    writer({"type": "stream_start", "role_id": role_id})
    token_q: _queue.Queue[str | None] = _queue.Queue()

    def _produce() -> str:
        full = ""
        try:
            for chunk in chain.stream(inputs):
                text = chunk if isinstance(chunk, str) else getattr(chunk, "content", str(chunk))
                if text:
                    full += text
                    token_q.put(text)
        except Exception:
            _logger.debug("stream fallback to invoke for chain")
            full = chain.invoke(inputs)
            if full:
                token_q.put(full)
        finally:
            token_q.put(None)
        return full

    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(None, _produce)

    while True:
        try:
            token = token_q.get_nowait()
        except _queue.Empty:
            if future.done():
                break
            await asyncio.sleep(0.02)
            continue
        if token is None:
            break
        writer({"type": "token", "content": token, "role_id": role_id})

    return await future
