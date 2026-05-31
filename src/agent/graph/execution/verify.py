"""Post-execution semantic checks.

Two distinct verifiers, both extracted verbatim from the original implementation:

* :func:`run_mls_post_check` — wraps ``_run_mls_post_step_verify`` (L2139-2169).
* :func:`run_cb_post_check`  — wraps ``_cb_post_step_check`` plus tool-success
  heuristic (L2171-2220).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from logger import get_logger

from agent.chat_agent_utils import (
    _cb_post_step_check,
    _get_output_file_path_from_raw,
    _read_output_file_preview,
    _run_mls_post_step_verify,
)
from agent.graph.execution.context import ExecutionContext

_logger = get_logger("agent.graph")


@dataclass
class VerifyResult:
    ok: bool
    # When ok=False: the new raw_output JSON the orchestrator should record if
    # no retries are possible.
    raw_output: Optional[str] = None
    failure_reason: str = ""
    # If the verifier produced a corrective retry suggestion, this is the dict
    # to merge into invoke_input on the next attempt.
    retry_input: Optional[dict] = None
    failure_type: str = "unknown"


async def run_mls_post_check(
    ctx: ExecutionContext, raw_output: Any
) -> VerifyResult:
    """MLS post-step verifier (L2139-2169).

    Updates ``ctx.history`` / ``ctx.log_entries`` in place to match the legacy
    side effects.
    """
    session_state_for_check = {
        "mls_debug_executor": ctx.chains.get("mls_debug_executor"),
        "llm": ctx.chains.get("llm"),
        "history": ctx.history,
        "conversation_log": ctx.log_entries,
    }
    status_ok, post_retry_input, post_report_for_cb = await _run_mls_post_step_verify(
        session_state_for_check,
        ctx.step_num,
        ctx.task_desc,
        ctx.tool_name,
        ctx.invoke_input,
        raw_output,
    )
    ctx.history = session_state_for_check.get("history", ctx.history)
    ctx.log_entries = session_state_for_check.get("conversation_log", ctx.log_entries)

    if status_ok:
        return VerifyResult(ok=True)

    post_reason = post_report_for_cb or (
        "步骤后置校验失败。" if ctx.ui_lang == "zh" else "Post-step verification failed."
    )
    retry_input = post_retry_input if isinstance(post_retry_input, dict) else None
    new_raw = json.dumps(
        {"status": "error", "error": {"type": "PostStepCheckFailed", "message": post_reason}},
        ensure_ascii=False,
    )
    return VerifyResult(
        ok=False,
        raw_output=new_raw,
        failure_reason=post_reason,
        retry_input=retry_input,
        failure_type="post_step",
    )


async def run_cb_post_check(
    ctx: ExecutionContext, raw_output: Any
) -> VerifyResult:
    """CB post-step verifier (L2171-2220).

    Returns ok=True both when CB agrees with the output AND when CB disagrees
    but the tool itself reported success (legacy tolerant behaviour).
    """
    if not ctx.chains.get("llm"):
        return VerifyResult(ok=True)

    output_file_path = _get_output_file_path_from_raw(raw_output, ctx.tool_name)
    file_preview = _read_output_file_preview(output_file_path) if output_file_path else None
    cb_match, cb_note = await _cb_post_step_check(
        ctx.chains["llm"],
        ctx.step_num,
        ctx.task_desc,
        ctx.tool_name,
        raw_output,
        output_file_path=output_file_path,
        file_preview=file_preview,
    )
    if cb_match:
        return VerifyResult(ok=True)

    # Soften: if the tool itself returned success, downgrade to a log warning.
    tool_itself_succeeded = False
    try:
        _parsed = json.loads(str(raw_output)) if isinstance(raw_output, str) else raw_output
        if isinstance(_parsed, dict) and (
            _parsed.get("success") is True or _parsed.get("status") == "success"
        ):
            tool_itself_succeeded = True
    except Exception:
        pass

    if tool_itself_succeeded:
        _logger.info(
            "CB post-step mismatch for step %s (%s) but tool returned success — "
            "treating as non-fatal (empty results). CB note: %s",
            ctx.step_num,
            ctx.tool_name,
            cb_note,
        )
        return VerifyResult(ok=True)

    failure_reason = cb_note or (
        "CB 后置校验不一致。" if ctx.ui_lang == "zh" else "CB post-step check mismatch."
    )
    new_raw = json.dumps(
        {"status": "error", "error": {"type": "CBPostStepMismatch", "message": failure_reason}},
        ensure_ascii=False,
    )
    return VerifyResult(
        ok=False,
        raw_output=new_raw,
        failure_reason=failure_reason,
        failure_type="cb_mismatch",
    )
