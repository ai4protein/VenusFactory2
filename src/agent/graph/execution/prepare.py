"""Pre-checks and invoke-input preparation.

Covers the original L1819-1868 region of ``_execute_node_impl``:

* read_skill precondition for code tools
* ``disabled_tool_names`` filtering
* missing-tool error
* ``_sanitize_tool_invoke_input``
* ``_normalize_output_paths``
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from agent.chat_agent_utils import _tool_output_indicates_failure
from agent.graph.execution.context import ExecutionContext, FailureType
from logger import get_logger

_logger = get_logger("agent.graph")


@dataclass
class PreCheckResult:
    ok: bool
    failure_type: FailureType = "unknown"
    reason: str = ""
    raw_output: str | None = None  # Pre-rendered error JSON to surface as last_output


def run_pre_checks(ctx: ExecutionContext) -> PreCheckResult:
    """Run code-tool, disabled-tool, and tool-existence pre-checks.

    Mirrors the order/format of the original implementation so failure
    payloads (``raw_output`` JSON strings) match byte-for-byte.
    """
    from agent.graph.helpers.plan_helpers import _normalize_step_number
    from agent.graph.helpers.tool_io import _get_step_raw_output

    # 1. Hard precondition: code tools require at least one successful read_skill step beforehand.
    if ctx.tool_name in {"run_session_python", "run_session_shell"}:
        has_successful_skill = False
        for prev in ctx.plan[: ctx.idx]:
            if str(prev.get("tool_name") or "").strip() != "read_skill":
                continue
            prev_step = _normalize_step_number(prev.get("step"), 0)
            prev_raw = _get_step_raw_output(ctx.step_results, prev_step)
            if prev_raw is None:
                continue
            prev_failed, _prev_reason = _tool_output_indicates_failure(prev_raw)
            if not prev_failed:
                has_successful_skill = True
                break
        if not has_successful_skill:
            failure_reason = (
                "代码执行步骤前缺少成功的 read_skill 步骤；请先由 CB 规划并执行 read_skill。"
                if ctx.ui_lang == "zh"
                else "Missing a successful read_skill step before code execution; CB must plan and run read_skill first."
            )
            raw_output = json.dumps(
                {
                    "status": "error",
                    "error": {"type": "SkillPreconditionFailed", "message": failure_reason},
                },
                ensure_ascii=False,
            )
            return PreCheckResult(
                ok=False,
                failure_type="skill_precondition",
                reason=failure_reason,
                raw_output=raw_output,
            )

    # 2. Disabled tool gate.
    if ctx.tool_name in ctx.disabled_tool_names:
        raw_output = json.dumps(
            {
                "success": False,
                "error": f"Tool `{ctx.tool_name}` is disabled in online mode.",
                "detail": "Training and protein-discovery tools are unavailable in online mode.",
            },
            ensure_ascii=False,
        )
        return PreCheckResult(
            ok=False,
            failure_type="disabled",
            reason=f"Tool `{ctx.tool_name}` is disabled in online mode.",
            raw_output=raw_output,
        )

    # 3. Tool not registered.
    if ctx.tool is None:
        raw_output = json.dumps({"success": False, "error": f"Unknown tool: {ctx.tool_name}"})
        return PreCheckResult(
            ok=False,
            failure_type="tool",
            reason=f"Unknown tool: {ctx.tool_name}",
            raw_output=raw_output,
        )

    return PreCheckResult(ok=True)


def prepare_invoke_input(ctx: ExecutionContext) -> dict[str, Any]:
    """Run ``_sanitize_tool_invoke_input`` + ``_normalize_output_paths`` once.

    Returns the new invoke_input dict (caller assigns back to ``ctx.invoke_input``).
    """
    from agent.graph.helpers.tool_io import (
        _collect_output_fields,
        _normalize_output_paths,
        _sanitize_tool_invoke_input,
    )

    if ctx.tool is None:
        return dict(ctx.merged_tool_input)

    invoke_input = _sanitize_tool_invoke_input(
        ctx.tool_name,
        ctx.tool,
        ctx.merged_tool_input,
        ctx.agent_session_dir,
        ctx.step_results,
    )
    raw_output_fields = _collect_output_fields(invoke_input)
    if ctx.agent_session_dir:
        invoke_input = _normalize_output_paths(ctx.tool_name, ctx.tool, invoke_input, ctx.agent_session_dir)
    normalized_output_fields = _collect_output_fields(invoke_input)
    if raw_output_fields or normalized_output_fields:
        _logger.debug(
            "Output normalize: tool=%s, raw=%s -> normalized=%s",
            ctx.tool_name,
            json.dumps(raw_output_fields, ensure_ascii=False),
            json.dumps(normalized_output_fields, ensure_ascii=False),
        )
    return invoke_input


def merge_retry_input(
    ctx: ExecutionContext,
    base_input: dict[str, Any],
    retry_input: dict[str, Any],
) -> dict[str, Any]:
    """Merge retry suggestions with current input, then re-sanitize.

    Mirror of the inner ``_merge_retry_input`` closure from the original
    implementation. Re-runs ``_sanitize`` and ``_normalize_output_paths`` so
    debug-supplied paths stay session-rooted.
    """
    from agent.graph.helpers.tool_io import (
        _normalize_output_paths,
        _sanitize_tool_invoke_input,
    )

    candidate = dict(base_input)
    candidate.update(retry_input)
    if ctx.tool is not None:
        candidate = _sanitize_tool_invoke_input(
            ctx.tool_name,
            ctx.tool,
            candidate,
            ctx.agent_session_dir,
            ctx.step_results,
        )
        if ctx.agent_session_dir:
            candidate = _normalize_output_paths(ctx.tool_name, ctx.tool, candidate, ctx.agent_session_dir)
    return candidate


async def materialize_deferred_input(ctx: ExecutionContext) -> dict[str, Any]:
    """Resolve plugin-declared runtime arguments before cache keys are built.

    Ordinary tools retain their existing behavior. A declared deferred tool
    still needs an explicitly installed materializer and a typed schema.
    No fallback or unvalidated parameter merge is allowed.
    """
    if ctx.tool is None or not (getattr(ctx.tool, "metadata", None) or {}).get("deferred_arguments"):
        return dict(ctx.merged_tool_input)
    schema = getattr(ctx.tool, "args_schema", None)
    fields = getattr(schema, "model_fields", None)
    if not isinstance(fields, dict):
        raise ValueError("deferred tool requires a typed argument schema")
    provided = {k: v for k, v in ctx.merged_tool_input.items() if k in fields}
    required = {k for k, field in fields.items() if field.is_required()}
    if all(k in provided and provided[k] not in (None, "") for k in required):
        try:
            return schema.model_validate(provided).model_dump()
        except ValueError:
            # A dependency can be an entire result object; let the registered
            # materializer select the typed value from the actual tool record.
            pass
    materializer = ctx.chains.get("materialize_tool_arguments")
    if not callable(materializer):
        raise ValueError("deferred tool argument materializer unavailable")
    generated = await materializer(ctx)
    if not isinstance(generated, dict) or set(generated) - set(fields):
        raise ValueError("materializer returned undeclared tool arguments")
    validated = schema.model_validate(generated).model_dump()
    if any(validated[k] in (None, "") for k in required):
        raise ValueError("materializer left required arguments empty")
    return validated
