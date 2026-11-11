"""Bound synchronous compatibility workflows away from the ASGI event loop.

These adapters still return the legacy result contract. Their existing inference
runners retain queue/resource admission; this limiter only bounds HTTP waits and
file postprocessing, independently of the threads used by metadata endpoints.
"""
from functools import partial, wraps
import inspect

from anyio import CapacityLimiter, to_thread
from anyio.lowlevel import RunVar


_LIMITER = RunVar("venus_http_workflow_limiter")
# Compatibility exporters use shared timestamp-based scratch paths. Preserve
# their serial execution while letting unrelated HTTP traffic continue.
MAX_CONCURRENT_WORKFLOWS = 1


def _limiter():
    # One limiter per event loop, including separately created test applications.
    try:
        return _LIMITER.get()
    except LookupError:
        limiter = CapacityLimiter(MAX_CONCURRENT_WORKFLOWS)
        _LIMITER.set(limiter)
        return limiter


def offload_workflow(function):
    if inspect.iscoroutinefunction(function) or inspect.isasyncgenfunction(function):
        raise TypeError("offload_workflow requires a synchronous workflow")

    @wraps(function)
    async def wrapped(*args, **kwargs):
        # AnyIO copies the request context (visitor, ownership, quota reservation)
        # and keeps the slot until the synchronous call actually exits.
        return await to_thread.run_sync(partial(function, *args, **kwargs), limiter=_limiter())

    wrapped.__signature__ = inspect.signature(function, eval_str=True)
    return wrapped
