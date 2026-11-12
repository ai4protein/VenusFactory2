"""Private Researcher task links; replay owned results without a second submit.

A saved link is not authority: the queue rechecks owner, tool, original parent
and the argument hash recorded by the scheduler at submission time.
"""

import copy
import hashlib
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar

from mcp_tools.scheduler.receipt_binding import value_digest

_ACTIVE = ContextVar("research_tool_receipts", default=None)


class ReceiptUnavailable(RuntimeError):
    def __init__(self, code):
        self.code = code
        super().__init__("Stored tool result unavailable: " + code)


@contextmanager
def receipt_scope(state, publish, slot=None):
    token = _ACTIVE.set((state, publish, slot))
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def owner_digest(owner):
    return hashlib.sha256(owner.encode()).hexdigest()


def validate_link(link, name, arguments, owner):
    if (
        link.get("version") != 1
        or link.get("tool") != name
        or link.get("arguments_sha256") != value_digest(arguments)
        or link.get("owner_sha256") != owner_digest(owner)
        or not link.get("parent_job")
        or not link.get("task_id")
    ):
        raise ReceiptUnavailable("binding_mismatch")


def validate_job(link, job):
    if any(
        job.get(key) != link.get(other)
        for key, other in (
            ("id", "task_id"),
            ("tool", "tool"),
            ("parent_job", "parent_job"),
            ("arguments_sha256", "arguments_sha256"),
        )
    ):
        raise ReceiptUnavailable("queue_binding_mismatch")


async def recover_result(link, name, arguments):
    """One owned read only. Pending/failed/missing legacy receipts never resubmit."""
    from mcp_tools.scheduler.client import rpc

    owner = os.environ.get("VENUS_QUEUE_OWNER", "")
    if not owner:
        raise ReceiptUnavailable("missing_owner")
    validate_link(link, name, arguments, owner)
    try:
        job = await rpc("tool_job", {"identity": link["task_id"], "owner": owner})
        validate_job(link, job)
        result = job.get("result", {})
        if job.get("status") != "completed" or result.get("is_error") or "value" not in result:
            raise ReceiptUnavailable("result_not_successful")
        value = result["value"]
        digest = value_digest(value)
        if link.get("result_sha256") and digest != link["result_sha256"]:
            raise ReceiptUnavailable("result_changed")
        return value
    except ReceiptUnavailable:
        raise
    except Exception as exc:
        raise ReceiptUnavailable("receipt_read_failed") from exc


async def invoke(name, arguments, *, owner, parent_job=None, on_status=None, slot=None):
    from mcp_tools.scheduler import client

    active = _ACTIVE.get()
    selected = slot or (active[2] if active else None)
    if not active or not selected or not parent_job:
        return await client.invoke(name, arguments, owner=owner, parent_job=parent_job, on_status=on_status)
    state, publish, _ = active
    links = state.setdefault("tool_receipts", {})
    link = links.get(selected)
    if link is not None:
        value = await recover_result(link, name, arguments)
        link.update(result_sha256=value_digest(value), recovered_at=time.time())
        publish()
        return value
    link = {
        "version": 1,
        "tool": name,
        "arguments": copy.deepcopy(arguments),
        "arguments_sha256": value_digest(arguments),
        "owner_sha256": owner_digest(owner),
        "parent_job": parent_job,
        "status": "submitting",
        "created": time.time(),
    }
    # Persist intent before submitting. A lost acknowledgement must not cause
    # a speculative repeat, even though this means manual recovery is required.
    links[selected] = link
    publish()

    def progress(job):
        if (
            job.get("arguments_sha256") != link["arguments_sha256"]
            or job.get("tool") != name
            or job.get("parent_job") != parent_job
        ):
            raise ReceiptUnavailable("queue_binding_mismatch")
        if link.get("task_id") and link["task_id"] != job.get("id"):
            raise ReceiptUnavailable("task_changed")
        link.update(task_id=job["id"], job_number=job.get("job_number"), status=job["status"])
        publish()
        if on_status:
            on_status(job)

    value = await client.invoke(name, arguments, owner=owner, parent_job=parent_job, on_status=progress)
    if not link.get("task_id"):
        raise ReceiptUnavailable("missing_task_acknowledgement")
    link.update(result_sha256=value_digest(value), delivered_at=time.time())
    publish()
    return value
