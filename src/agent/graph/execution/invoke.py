"""Tool invocation with per-tool timeout + retry, plus guardrails and caching.

Wraps the original L1909-2052 region of ``_execute_node_impl``.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Optional

from logger import get_logger

from agent.chat_agent import get_cached_tool_result, save_cached_tool_result
from agent.guardrails import DEFAULT_INPUT_GUARDRAILS, run_input_guardrails
from agent.retry import TOOL_RETRY, retry_async
from agent.tracing import ToolSpanData, start_span
from exceptions import InputGuardrailTripped

from agent.graph.execution.context import ExecutionContext, FailureType

_logger = get_logger("agent.graph")


@dataclass
class InvokeOutcome:
    ok: bool
    raw_output: Any
    failure_type: FailureType = "unknown"
    failure_reason: str = ""
    is_timeout: bool = False
    is_guardrail: bool = False


def lookup_cache(ctx: ExecutionContext) -> Optional[Any]:
    """Return cached output for the current invoke_input if present, else None."""
    cached = get_cached_tool_result(
        {"tool_cache": ctx.state.get("tool_cache", {})}, ctx.tool_name, ctx.invoke_input
    )
    if cached:
        return cached["outputs"]
    return None


def save_cache(ctx: ExecutionContext, raw_output: Any) -> None:
    save_cached_tool_result(ctx.state, ctx.tool_name, ctx.invoke_input, raw_output)


async def invoke_tool(ctx: ExecutionContext) -> InvokeOutcome:
    """One full attempt: input guardrails + tool.invoke with timeout + retry_async.

    Mirrors L1985-2045 verbatim in semantics. The orchestrator decides whether
    to retry based on the resulting :class:`InvokeOutcome`.
    """
    # Per-tool timeout map and execution timeout live on the package root;
    # pull at call time to avoid import cycles at module load.
    from agent.chat_agent_utils import TOOL_EXECUTION_TIMEOUT
    from agent.graph import _TOOL_TIMEOUTS

    tool = ctx.tool
    tool_name = ctx.tool_name
    invoke_input = ctx.invoke_input

    # --- Input guardrails ---
    try:
        await run_input_guardrails(DEFAULT_INPUT_GUARDRAILS, tool_name, invoke_input)
    except InputGuardrailTripped as gt:
        raw_output = json.dumps(
            {"success": False, "error": f"Input guardrail blocked: {gt}"},
            ensure_ascii=False,
        )
        _logger.warning("Guardrail tripped: tool=%s, guardrail=%s", tool_name, gt.guardrail_name)
        return InvokeOutcome(
            ok=False,
            raw_output=raw_output,
            failure_type="guardrail",
            failure_reason=str(gt),
            is_guardrail=True,
        )

    # --- Tool invocation with per-tool timeout and retry ---
    tool_timeout = _TOOL_TIMEOUTS.get(tool_name, TOOL_EXECUTION_TIMEOUT)
    inputs_str = json.dumps(invoke_input, ensure_ascii=False, sort_keys=True)
    if len(inputs_str) > 500:
        inputs_str = inputs_str[:500] + "..."
    _logger.info(
        "Execute: tool=%s, timeout=%ss, input=%s", tool_name, tool_timeout, inputs_str
    )

    async def _invoke_with_timeout():
        return await asyncio.wait_for(
            asyncio.to_thread(tool.invoke, invoke_input),
            timeout=tool_timeout,
        )

    with start_span(
        f"tool.{tool_name}",
        ToolSpanData(tool_name=tool_name, tool_input=invoke_input),
    ) as _tool_span:
        retry_result = await retry_async(_invoke_with_timeout, policy=TOOL_RETRY)

        if retry_result.success:
            out = retry_result.value
            raw_output = out if isinstance(out, (str, dict)) else str(out)
            out_preview = str(raw_output)[:300] + ("..." if len(str(raw_output)) > 300 else "")
            _logger.info(
                "Result: tool=%s, output_preview=%s (attempts=%d)",
                tool_name,
                out_preview,
                retry_result.attempts,
            )
            try:
                _tool_span.data.success = True
                _tool_span.data.tool_output = out_preview
            except Exception:
                pass
            return InvokeOutcome(ok=True, raw_output=raw_output)

        err = retry_result.last_error
        is_timeout = isinstance(err, (TimeoutError, asyncio.TimeoutError))
        if is_timeout:
            raw_output = json.dumps(
                {
                    "success": False,
                    "error": (
                        f"Tool execution timed out ({tool_timeout}s) after "
                        f"{retry_result.attempts} attempts"
                    ),
                },
                ensure_ascii=False,
            )
            _logger.warning(
                "Result: tool=%s, timeout after %ss (%d attempts)",
                tool_name,
                tool_timeout,
                retry_result.attempts,
            )
            failure_type: FailureType = "timeout"
        else:
            raw_output = json.dumps({"success": False, "error": str(err)})
            _logger.error(
                "Result: tool=%s, failed after %d attempts, error=%s",
                tool_name,
                retry_result.attempts,
                err,
            )
            failure_type = "tool"
        try:
            _tool_span.data.success = False
            _tool_span.data.error_message = str(retry_result.last_error)[:500]
        except Exception:
            pass
        return InvokeOutcome(
            ok=False,
            raw_output=raw_output,
            failure_type=failure_type,
            failure_reason=str(err),
            is_timeout=is_timeout,
        )
