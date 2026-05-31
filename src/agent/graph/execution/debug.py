"""MLS debug helper used by the retry orchestrator after a failure / CB mismatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from agent.chat_agent_utils import _run_mls_debug_with_tools
from agent.graph.execution.context import ExecutionContext


@dataclass
class DebugResult:
    retry_input: Optional[dict] = None
    report_for_cb: str = ""


async def debug_via_mls(
    ctx: ExecutionContext, failure_reason: str
) -> DebugResult:
    """Invoke ``_run_mls_debug_with_tools`` and mirror history side effects."""
    session_state_for_check = {
        "mls_debug_executor": ctx.chains.get("mls_debug_executor"),
        "llm": ctx.chains.get("llm"),
        "history": ctx.history,
        "conversation_log": ctx.log_entries,
    }
    debug_retry_input, debug_report = await _run_mls_debug_with_tools(
        session_state_for_check,
        ctx.step_num,
        ctx.task_desc,
        ctx.tool_name,
        ctx.invoke_input,
        failure_reason,
    )
    ctx.history = session_state_for_check.get("history", ctx.history)
    ctx.log_entries = session_state_for_check.get("conversation_log", ctx.log_entries)
    return DebugResult(
        retry_input=debug_retry_input if isinstance(debug_retry_input, dict) else None,
        report_for_cb=debug_report or "",
    )
