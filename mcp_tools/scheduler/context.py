"""Host-bound MCP session identity across queued worker execution."""

from contextlib import contextmanager
from contextvars import ContextVar

_queued_session = ContextVar("queued_mcp_session", default=None)


@contextmanager
def bind_mcp_session(session_id):
    """Bind only the private job specification, never a model/tool argument."""
    if session_id is not None and (
        not isinstance(session_id, str) or not session_id.strip() or len(session_id) > 256
    ):
        raise ValueError("Invalid queued MCP session")
    token = _queued_session.set(session_id)
    try:
        yield
    finally:
        _queued_session.reset(token)


def mcp_session_id(transport_session_id):
    """Use the originating session for a job, or the live direct transport."""
    return _queued_session.get() or transport_session_id
