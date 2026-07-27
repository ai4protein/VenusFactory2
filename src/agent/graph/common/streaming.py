"""Shared trace + token-streaming primitives used by graph nodes."""
from __future__ import annotations

import asyncio
import contextvars
from contextlib import contextmanager
from typing import Any, Optional

from langgraph.config import get_stream_writer

from agent.tracing import NoOpTrace, Scope, create_trace
from logger import get_logger

_logger = get_logger("agent.graph")

# Side-channel so SSE can flush tokens immediately even if LangGraph buffers
# StreamWriter custom events until the node returns.
_sse_queue_var: contextvars.ContextVar[Optional[asyncio.Queue]] = contextvars.ContextVar(
    "vf_graph_sse_queue", default=None
)
_sse_loop_var: contextvars.ContextVar[Optional[asyncio.AbstractEventLoop]] = contextvars.ContextVar(
    "vf_graph_sse_loop", default=None
)

# Also stashed on LangGraph runnable config (more reliable across node tasks).
_CONFIG_SSE_QUEUE_KEY = "vf_sse_queue"
_CONFIG_SSE_LOOP_KEY = "vf_sse_loop"


def bind_sse_queue(
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop | None = None,
) -> tuple[contextvars.Token, contextvars.Token]:
    """Bind an asyncio.Queue that chat_api will drain for live SSE tokens."""
    q_token = _sse_queue_var.set(queue)
    l_token = _sse_loop_var.set(loop or asyncio.get_running_loop())
    return q_token, l_token


def reset_sse_queue(tokens: tuple[contextvars.Token, contextvars.Token]) -> None:
    _sse_queue_var.reset(tokens[0])
    _sse_loop_var.reset(tokens[1])


def sse_config_keys(queue: asyncio.Queue, loop: asyncio.AbstractEventLoop | None = None) -> dict[str, Any]:
    """Values to merge into graph ``configurable`` for reliable token fan-out."""
    return {
        _CONFIG_SSE_QUEUE_KEY: queue,
        _CONFIG_SSE_LOOP_KEY: loop or asyncio.get_running_loop(),
    }


_STATE_SSE_QUEUE_KEY = "_vf_sse_queue"
_STATE_SSE_LOOP_KEY = "_vf_sse_loop"


def attach_sse_to_session_state(
    state: dict[str, Any],
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop | None = None,
) -> None:
    """Stash the live SSE queue on the session ``chains`` dict.

    Graph nodes receive ``configurable["chains"]`` as this same object, so
    token emitters can resolve the queue even when ContextVars are not copied
    into LangGraph's node tasks.
    """
    state[_STATE_SSE_QUEUE_KEY] = queue
    state[_STATE_SSE_LOOP_KEY] = loop or asyncio.get_running_loop()


def detach_sse_from_session_state(state: dict[str, Any]) -> None:
    state.pop(_STATE_SSE_QUEUE_KEY, None)
    state.pop(_STATE_SSE_LOOP_KEY, None)


def _resolve_sse_target() -> tuple[Optional[asyncio.Queue], Optional[asyncio.AbstractEventLoop]]:
    q = _sse_queue_var.get()
    loop = _sse_loop_var.get()
    if q is not None and loop is not None:
        return q, loop
    try:
        from langgraph.config import get_config

        conf = (get_config() or {}).get("configurable") or {}
        q = conf.get(_CONFIG_SSE_QUEUE_KEY) or q
        loop = conf.get(_CONFIG_SSE_LOOP_KEY) or loop
        if q is not None and loop is not None:
            return q, loop
        chains = conf.get("chains")
        if isinstance(chains, dict):
            q = chains.get(_STATE_SSE_QUEUE_KEY) or q
            loop = chains.get(_STATE_SSE_LOOP_KEY) or loop
    except Exception:
        pass
    return q, loop


def _emit_custom(event: dict[str, Any]) -> None:
    """Fan-out a custom SSE payload to StreamWriter + side-channel queue."""
    try:
        get_stream_writer()(event)
    except Exception:
        pass
    q, loop = _resolve_sse_target()
    if q is None or loop is None:
        return

    def _put() -> None:
        try:
            q.put_nowait(event)
        except Exception:
            pass

    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is loop:
        _put()
    else:
        try:
            loop.call_soon_threadsafe(_put)
        except Exception:
            _put()


def _chunk_content(value: Any) -> str:
    """Normalize LLM / parser chunk payloads to plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content") or ""
                if text:
                    parts.append(str(text))
            else:
                text = getattr(block, "text", None)
                if text:
                    parts.append(str(text))
        return "".join(parts)
    return str(content) if content is not None else ""


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


def _chunk_text(text: str, size: int = 24) -> list[str]:
    """Split prepared text into small chunks for progressive SSE reveal."""
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    buf = ""
    for ch in text:
        buf += ch
        # Prefer breaking on whitespace / newlines so markdown stays readable.
        if len(buf) >= size and (ch.isspace() or ch in "，。；、！？,.!?;:"):
            chunks.append(buf)
            buf = ""
    if buf:
        chunks.append(buf)
    return chunks


async def _stream_text(text: str, role_id: str = "assistant", *, delay_s: float = 0.012) -> str:
    """Emit prepared text as token SSE events (typewriter-style progressive reveal).

    Used when the LLM output is structured (JSON plan / clarification) and we
    still want the user-facing prose to appear gradually before interactive
    forms mount.
    """
    if not text:
        return ""
    _emit_custom({"type": "stream_start", "role_id": role_id})
    for chunk in _chunk_text(text):
        _emit_custom({"type": "token", "content": chunk, "role_id": role_id})
        # Yield so chat_api can flush SSE between chunks.
        await asyncio.sleep(delay_s if delay_s > 0 else 0)
    return text


async def _emit_token_pieces(text: str, role_id: str) -> None:
    """Emit token text, splitting large bursts so the UI can paint progressively.

    Some providers (or proxies) deliver a whole completion as one SSE delta.
    Without splitting, Expert looks like a one-shot dump even though the
    plumbing is "streaming".
    """
    if not text:
        return
    pieces = _chunk_text(text, size=18) if len(text) > 40 else [text]
    for i, piece in enumerate(pieces):
        _emit_custom({"type": "token", "content": piece, "role_id": role_id})
        # Yield to the event loop so ``_drain_graph_astream`` can flush SSE.
        # Use a tiny delay for multi-piece bursts so browsers/proxies don't
        # coalesce every frame into one paint.
        await asyncio.sleep(0.012 if len(pieces) > 1 and i + 1 < len(pieces) else 0)


async def _stream_chain(chain, inputs: dict[str, Any], role_id: str = "assistant") -> str:
    """Invoke a chain with token-level streaming via SSE side-channel.

    Prefers ``chain.astream`` so each token awaits the event loop (allowing
    ``_stream_graph`` to flush SSE). Falls back to threaded ``stream`` / ``invoke``
    with chunked emission.
    """
    _emit_custom({"type": "stream_start", "role_id": role_id})
    full = ""

    # Primary path: native async streaming (lets the event loop flush SSE).
    astream = getattr(chain, "astream", None)
    if callable(astream):
        try:
            async for chunk in astream(inputs):
                text = _chunk_content(chunk)
                if text:
                    full += text
                    await _emit_token_pieces(text, role_id)
            if full:
                return full
        except Exception:
            _logger.debug("astream failed; falling back to threaded stream", exc_info=True)
            full = ""

    # Fallback: threaded sync stream / invoke, then progressive emit.
    import queue as _queue

    token_q: _queue.Queue[str | None] = _queue.Queue()

    def _produce() -> str:
        produced = ""
        try:
            for chunk in chain.stream(inputs):
                text = _chunk_content(chunk)
                if text:
                    produced += text
                    token_q.put(text)
        except Exception:
            _logger.debug("stream fallback to invoke for chain")
            produced = str(chain.invoke(inputs) or "")
            for piece in _chunk_text(produced):
                token_q.put(piece)
        finally:
            token_q.put(None)
        return produced

    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(None, _produce)

    while True:
        try:
            token = token_q.get_nowait()
        except _queue.Empty:
            if future.done():
                while True:
                    try:
                        token = token_q.get_nowait()
                    except _queue.Empty:
                        token = None
                        break
                    if token is None:
                        return await future
                    await _emit_token_pieces(token, role_id)
                break
            await asyncio.sleep(0.02)
            continue
        if token is None:
            break
        await _emit_token_pieces(token, role_id)

    return await future
