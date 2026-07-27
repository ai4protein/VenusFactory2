"""Post-execution semantic checks.

Two distinct verifiers, both extracted verbatim from the original implementation:

* :func:`run_mls_post_check` — wraps ``_run_mls_post_step_verify`` (L2139-2169).
* :func:`run_cb_post_check`  — wraps ``_cb_post_step_check`` plus tool-success
  heuristic (L2171-2220).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional


def _now_iso() -> str:
    return datetime.now().isoformat()

from logger import get_logger

from agent.chat_agent_utils import (
    _BINARY_EXTENSIONS,
    _cb_post_step_check,
    _get_output_file_path_from_raw,
    _read_output_file_preview,
    _run_mls_post_step_verify,
    _tool_output_indicates_failure,
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
        "results", "result", "output", "response",
        "sequence", "sequences", "file_path", "entries",
        # Tool-specific substantive fields:
        "config_path", "dataset_info", "model_info", "model_path",
        "predictions", "metrics", "fasta_path", "fasta_file",
        "pdb_path", "pdb_file", "structure_file", "generated_code_path",
        "csv_path", "download_path", "score", "scores",
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


def _artifact_file_is_empty(raw_output: Any, tool_name: str) -> bool:
    """True when a declared output file path is missing or empty.

    Binary artifacts (png/pdf/…) have no text preview by design — treat a
    positive file size as non-empty so success paths can skip dual verify.
    """
    import os as _os

    output_file_path = _get_output_file_path_from_raw(raw_output, tool_name)
    if not output_file_path:
        return False
    if not _os.path.exists(output_file_path) or _os.path.getsize(output_file_path) == 0:
        return True
    ext = _os.path.splitext(output_file_path)[1].lower()
    if ext in _BINARY_EXTENSIONS:
        return False
    file_preview = _read_output_file_preview(output_file_path)
    return not (file_preview or "").strip()


def _should_skip_dual_verify(raw_output: Any, tool_name: str) -> bool:
    """Skip MLS/CB LLM post-checks when the tool already succeeded with content.

    Runs full dual verify only on failure / empty artifact / ambiguous output.
    """
    is_failure, _ = _tool_output_indicates_failure(raw_output)
    if is_failure:
        return False
    if _artifact_file_is_empty(raw_output, tool_name):
        return False
    if _tool_self_reports_success(raw_output):
        return True
    # Explicit success envelope without matching SUBSTANTIVE_KEYS, but a
    # real on-disk artifact exists → still safe to skip the LLM round-trips.
    try:
        parsed = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
    except Exception:
        return False
    if not isinstance(parsed, dict):
        return False
    explicit_ok = parsed.get("success") is True or parsed.get("status") == "success"
    if not explicit_ok:
        return False
    return bool(_get_output_file_path_from_raw(raw_output, tool_name))


async def run_mls_post_check(
    ctx: ExecutionContext, raw_output: Any
) -> VerifyResult:
    """MLS post-step verifier (L2139-2169).

    Updates ``ctx.history`` / ``ctx.log_entries`` in place to match the legacy
    side effects.

    Fast path: when the tool envelope already reports success with substantive
    non-empty content, skip the MLS LLM verifier entirely (no debug agent).
    Real failures still run the full verifier + retry/debug path.
    """
    # Skip expensive MLS LLM round-trip when the tool already succeeded.
    if _should_skip_dual_verify(raw_output, ctx.tool_name):
        _logger.debug(
            "MLS post-check skipped for step %s (%s): tool success + non-empty artifact",
            ctx.step_num,
            ctx.tool_name,
        )
        return VerifyResult(ok=True)

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

    # Synthesize a useful failure reason. Order of preference:
    #   1. Verifier's own ``report_for_cb`` text
    #   2. A compact summary of the corrective ``retry_input``
    #   3. Generic fallback (kept only for legacy compatibility — with the
    #      parse fix in chat_agent_utils, status_ok is True whenever the
    #      verifier produced no actionable complaint, so this is unreachable
    #      in normal operation).
    if post_report_for_cb and str(post_report_for_cb).strip():
        post_reason = str(post_report_for_cb).strip()
    elif isinstance(post_retry_input, dict) and post_retry_input:
        try:
            keys_preview = ", ".join(sorted(post_retry_input.keys())[:5])
        except Exception:
            keys_preview = ""
        post_reason = (
            f"校验器要求用不同参数重试（{keys_preview}）。" if ctx.ui_lang == "zh"
            else f"Verifier requested a retry with different parameters ({keys_preview})."
        )
    else:
        post_reason = (
            "步骤后置校验失败。" if ctx.ui_lang == "zh" else "Post-step verification failed."
        )

    # Soften: when the tool genuinely produced real content (success envelope
    # + substantive field + non-empty artifact file when applicable), downgrade
    # the verifier rejection to a soft warning. But escalate to a hard failure
    # when the artifact is empty/missing OR the verifier explicitly flags
    # emptiness — those are the cases where "tool says success" is a lie and
    # silently propagating that into downstream steps is what produces
    # convincingly-formatted but empty plots / reports.
    if _tool_self_reports_success(raw_output):
        file_is_empty = _artifact_file_is_empty(raw_output, ctx.tool_name)
        _reason_lower = (post_reason or "").lower()
        verifier_flags_emptiness = any(
            keyword in _reason_lower
            for keyword in (
                "no evidence", "no preview", "no output", "no content",
                "empty result", "empty results", "no sequences", "no records",
                "no data", "no hits", "no file", "0 sequences", "zero sequences",
                "missing file", "missing data", "data missing", "数据缺失",
                "空表", "为空", "缺失",
            )
        )
        if not file_is_empty and not verifier_flags_emptiness:
            _logger.warning(
                "MLS post-step verifier flagged step %s (%s) but tool reports success "
                "with substantive content — treating as soft warning. Verifier note: %s",
                ctx.step_num, ctx.tool_name, post_reason,
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
                ctx.log_entries.append({
                    "role": "assistant",
                    "content": (
                        f"MLS post-step verifier downgraded to warning for step "
                        f"{ctx.step_num} ({ctx.tool_name}): {post_reason}"
                    ),
                    "role_id": "machine_learning_specialist",
                    "timestamp": _now_iso(),
                })
            except Exception:
                pass
            return VerifyResult(ok=True)

        _logger.warning(
            "MLS post-step HARD-FAIL for step %s (%s): tool envelope claimed success "
            "but file_is_empty=%s verifier_flags_emptiness=%s. Verifier note: %s",
            ctx.step_num, ctx.tool_name,
            file_is_empty, verifier_flags_emptiness, post_reason,
        )

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

    Returns ok=True when CB agrees, OR when CB disagrees but the tool's claimed
    success is backed by real content. When the tool claims success but produced
    no real output (empty file, missing file, CB explicitly flags emptiness),
    the rejection is escalated to a hard failure so the orchestrator can retry
    or surface the failure to the user instead of silently propagating empty
    intermediate state into downstream steps.

    Fast path: skip the CB LLM call when the tool already reports success with
    a non-empty artifact.
    """
    if not ctx.chains.get("llm"):
        return VerifyResult(ok=True)

    # Skip CB LLM when tool success + non-empty result (no extra debug loop).
    if _should_skip_dual_verify(raw_output, ctx.tool_name):
        _logger.debug(
            "CB post-check skipped for step %s (%s): tool success + non-empty artifact",
            ctx.step_num,
            ctx.tool_name,
        )
        return VerifyResult(ok=True)

    output_file_path = _get_output_file_path_from_raw(raw_output, ctx.tool_name)
    file_preview = _read_output_file_preview(output_file_path) if output_file_path else None
    file_is_empty = _artifact_file_is_empty(raw_output, ctx.tool_name)

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

    # CB rejected. Decide: soft warning (tool genuinely succeeded, CB is fussy)
    # vs hard failure (tool says success but the artifact is empty/missing).
    #
    # The legacy code only checked the tool envelope's `success: true` flag,
    # which masks the most common failure mode: a tool returns success=true with
    # a valid-looking file_path but the file is empty / has no records.
    # Use `_tool_self_reports_success` (which also requires a SUBSTANTIVE_KEYS
    # field) AND additionally require, when a file_path is declared, that the
    # file actually exists with non-empty preview.
    real_success = _tool_self_reports_success(raw_output)

    # Heuristic: CB note explicitly mentions emptiness / missing evidence.
    _note_lower = (cb_note or "").lower()
    cb_flags_emptiness = any(
        keyword in _note_lower
        for keyword in (
            "no evidence", "no preview", "no output", "no content",
            "empty result", "empty results", "no sequences",
            "no records", "no data", "no hits", "no file",
            "0 sequences", "zero sequences", "missing file",
        )
    )

    if real_success and not file_is_empty and not cb_flags_emptiness:
        _logger.info(
            "CB post-step mismatch for step %s (%s) but tool returned success "
            "and artifact is non-empty — treating as non-fatal warning. CB note: %s",
            ctx.step_num,
            ctx.tool_name,
            cb_note,
        )
        return VerifyResult(ok=True)

    # Hard failure: build an explicit error envelope so the orchestrator's
    # retry/skip logic sees this as a real failure (not a tool success).
    _logger.warning(
        "CB post-step HARD-FAIL for step %s (%s): tool envelope claimed success "
        "but file_is_empty=%s real_success=%s cb_flags_emptiness=%s. CB note: %s",
        ctx.step_num, ctx.tool_name,
        file_is_empty, real_success, cb_flags_emptiness, cb_note,
    )
    try:
        raw_preview = (
            raw_output if isinstance(raw_output, str)
            else json.dumps(raw_output, ensure_ascii=False, default=str)
        )
    except Exception:
        raw_preview = str(raw_output)
    if len(raw_preview) > 1500:
        raw_preview = raw_preview[:1500] + "\n...(truncated)"
    failure_reason = (
        f"CB verifier rejected step {ctx.step_num} ({ctx.tool_name}): tool reported "
        f"success but the artifact is empty/missing. "
        f"file_path={output_file_path or '<none>'}, "
        f"file_is_empty={file_is_empty}. CB note: {cb_note or '(no note)'}"
    )
    new_raw = json.dumps(
        {
            "status": "error",
            "error": {
                "type": "CBPostCheckFailed",
                "message": failure_reason,
                "cb_note": cb_note,
                "file_path": output_file_path,
                "file_is_empty": file_is_empty,
                "actual_tool_output_preview": raw_preview,
            },
        },
        ensure_ascii=False,
    )
    return VerifyResult(
        ok=False,
        raw_output=new_raw,
        failure_reason=failure_reason,
        failure_type="cb_mismatch",
    )
