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


def _tool_self_reports_success(raw_output: Any) -> bool:
    """Return True when the tool's own payload looks substantively successful.

    Two acceptance modes (any one is enough):

    A. **Explicit-success mode**: dict has ``success: True`` OR
       ``status: 'success'``, AND carries at least one substantive content
       field (file_info, output_files, data, content, results, sequence,
       file_path, entries, config_path, model_info, predictions, metrics,
       fasta_path, score, ...).

    B. **Implicit-success mode** (new): dict has no explicit failure signal
       (no ``success: False`` / ``status: 'error'`` / ``error`` field /
       ``error_msg`` field), AND total payload size suggests real content
       (>=400 chars of useful keys), AND a substantive content field is
       present. This catches tools that just return their data without
       wrapping it in a ``success: True`` envelope — e.g. some download
       tools that return ``{"file_path": "...", "file_size": 12345,
       "file_name": "..."}``.

    Used by ``run_mls_post_check`` to downgrade verifier rejections to soft
    warnings when the tool clearly produced real output. The verifier's
    rejection is preserved in history as a warning so the user still sees
    its concern, but the orchestrator is told to proceed.
    """
    try:
        parsed = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
    except Exception:
        return False
    if not isinstance(parsed, dict):
        return False

    SUBSTANTIVE_KEYS = (
        "file_info", "output_files", "data", "content", "content_preview",
        "results", "sequence", "sequences", "file_path", "entries",
        # Tool-specific substantive fields:
        "config_path", "dataset_info", "model_info", "model_path",
        "predictions", "metrics", "fasta_path", "score", "scores",
        # Additional fields for tools that don't use success: envelopes
        "biological_metadata", "file_name", "file_size",
    )

    def _has_substantive_content() -> bool:
        for key in SUBSTANTIVE_KEYS:
            val = parsed.get(key)
            if val is None:
                continue
            if isinstance(val, str) and not val.strip():
                continue
            if isinstance(val, (list, dict, tuple, set)) and len(val) == 0:
                continue
            return True
        return False

    # Mode A: explicit success envelope
    if (parsed.get("success") is True or parsed.get("status") == "success") and _has_substantive_content():
        return True

    # Mode B: no explicit failure AND substantive content
    has_explicit_failure = (
        parsed.get("success") is False
        or parsed.get("status") in ("error", "fail", "failed")
        or bool(parsed.get("error"))
        or bool(parsed.get("error_msg"))
        or bool(parsed.get("err"))
    )
    if not has_explicit_failure and _has_substantive_content():
        # Final sanity: payload should be at least mildly large (avoid
        # treating ``{"file_path": null}`` as a real success).
        try:
            payload_size = len(json.dumps(parsed, ensure_ascii=False))
        except Exception:
            payload_size = 0
        if payload_size >= 80:
            return True
    return False


async def run_mls_post_check(
    ctx: ExecutionContext, raw_output: Any
) -> VerifyResult:
    """MLS post-step verifier (L2139-2169).

    Updates ``ctx.history`` / ``ctx.log_entries`` in place to match the legacy
    side effects.

    Tolerant softening: when the underlying tool's payload self-reports success
    with substantive content (see ``_tool_self_reports_success``), a verifier
    rejection is downgraded to a soft warning. The verifier is still invoked
    (so its diagnostic side effects — history note, logs — remain), but the
    orchestrator is told to proceed. This matches the spirit of the CB tolerant
    branch in :func:`run_cb_post_check` and prevents real tool successes (e.g.
    ``download_uniprot_seq_by_id`` returning a FASTA, ``predict_protein_function``
    writing a CSV) from being blocked by an over-strict semantic verifier.
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

    # Soften: when the tool itself returned a success envelope with substantive
    # content, treat the verifier rejection as a non-fatal warning and let the
    # plan continue. We still surface the verifier's opinion in history so the
    # user can see why the verifier was unhappy.
    if _tool_self_reports_success(raw_output):
        _logger.warning(
            "MLS post-step verifier flagged step %s (%s) but tool reports success "
            "with substantive content — treating as soft warning. Verifier note: %s",
            ctx.step_num,
            ctx.tool_name,
            post_reason,
        )
        try:
            note = (
                f"⚠️ **MLS self-check (post-step):** 校验器对工具输出有疑问，但工具自报成功，跳过该警告继续执行。\n\n校验意见：{post_reason}"
                if ctx.ui_lang == "zh"
                else f"⚠️ **MLS self-check (post-step):** Verifier flagged the output but the tool reported success — continuing.\n\nVerifier note: {post_reason}"
            )
            ctx.history.append(
                {"role": "assistant", "content": note, "role_id": "machine_learning_specialist"}
            )
            ctx.log_entries.append(
                f"MLS post-step verifier downgraded to warning for step {ctx.step_num} ({ctx.tool_name}): {post_reason}"
            )
        except Exception:
            # Never let history bookkeeping break the success path.
            pass
        return VerifyResult(ok=True)

    retry_input = post_retry_input if isinstance(post_retry_input, dict) else None
    # Preserve the actual tool output in the failure envelope so the user
    # (and downstream debug tools) can see what the verifier was unhappy
    # about — not just an opaque "Post-step verification failed."
    try:
        raw_preview = (
            raw_output if isinstance(raw_output, str)
            else json.dumps(raw_output, ensure_ascii=False, default=str)
        )
    except Exception:
        raw_preview = str(raw_output)
    if len(raw_preview) > 1500:
        raw_preview = raw_preview[:1500] + "\n...(truncated)"
    new_raw = json.dumps(
        {
            "status": "error",
            "error": {
                "type": "PostStepCheckFailed",
                "message": post_reason,
                "verifier_note": post_reason,
                "actual_tool_output_preview": raw_preview,
            },
        },
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
