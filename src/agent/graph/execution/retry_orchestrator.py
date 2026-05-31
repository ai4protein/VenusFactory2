"""Per-reason retry orchestration.

Replaces the legacy single ``step_retry`` counter with independent budgets keyed
by :class:`RetryReason`. Behaviour notes:

* Each ``RetryReason`` has its own budget; running out for one reason does not
  consume slots for others. With the default budgets the worst-case attempt
  count for a step is ~6 (vs. legacy ~3), but typical paths still complete in
  1–2 attempts.
* Failure semantics (raw_output JSON shape, logging) match the legacy code so
  downstream consumers (UI, CB, hooks) see identical payloads.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from logger import get_logger

from agent.chat_agent_utils import _tool_output_indicates_failure
from agent.graph.execution.context import ExecutionContext, ExecutionResult
from agent.graph.execution.debug import debug_via_mls
from agent.graph.execution.invoke import invoke_tool
from agent.graph.execution.path_repair import rebind_file_not_found
from agent.graph.execution.prepare import merge_retry_input
from agent.graph.execution.verify import run_cb_post_check, run_mls_post_check

_logger = get_logger("agent.graph")


class RetryReason(str, Enum):
    FILE_NOT_FOUND = "file_not_found"
    POST_STEP_VERIFY = "post_step_verify"
    CB_MISMATCH = "cb_mismatch"
    MLS_DEBUG = "mls_debug"


@dataclass
class RetryBudget:
    """Per-reason retry budgets.

    Defaults intentionally err on the generous side (vs. the legacy shared cap
    of 2) so that a step running into multiple distinct failure categories
    can recover. Worst case = sum(values) attempts.
    """

    file_not_found: int = 1
    post_step_verify: int = 1
    cb_mismatch: int = 1
    mls_debug: int = 1


@dataclass
class RetryOrchestrator:
    ctx: ExecutionContext
    budget: RetryBudget = field(default_factory=RetryBudget)
    attempts_by_reason: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    async def run(self) -> ExecutionResult:
        """Drive invoke + verify chain with per-reason retry budgets."""
        ctx = self.ctx
        last_raw_output = None
        last_failure_reason = ""
        last_failure_type = "unknown"

        while True:
            invoke_outcome = await invoke_tool(ctx)
            raw_output = invoke_outcome.raw_output

            if not invoke_outcome.ok:
                # Guardrails short-circuit immediately (legacy behaviour: no retry).
                if invoke_outcome.is_guardrail:
                    return ExecutionResult(
                        success=False,
                        raw_output=raw_output,
                        failure_reason=invoke_outcome.failure_reason,
                        failure_type="guardrail",
                        attempts_by_reason=dict(self.attempts_by_reason),
                    )

                last_raw_output = raw_output
                last_failure_reason = invoke_outcome.failure_reason
                last_failure_type = invoke_outcome.failure_type

                # FileNotFound path rebinding has its own slot — try first.
                rebound = rebind_file_not_found(ctx, raw_output, last_failure_reason)
                if rebound is not None and self._claim(RetryReason.FILE_NOT_FOUND):
                    ctx.invoke_input = rebound
                    _logger.info(
                        "FileNotFound retry: tool=%s, step=%s, retry=%s",
                        ctx.tool_name,
                        ctx.step_num,
                        self.attempts_by_reason[RetryReason.FILE_NOT_FOUND.value],
                    )
                    continue

                # Otherwise fall through to MLS debug.
                if self._claim(RetryReason.MLS_DEBUG):
                    debug_result = await debug_via_mls(
                        ctx, last_failure_reason or str(raw_output)
                    )
                    if debug_result.retry_input is not None:
                        ctx.invoke_input = merge_retry_input(
                            ctx, ctx.invoke_input, debug_result.retry_input
                        )
                        _logger.info(
                            "Failure retry: tool=%s, step=%s, retry=%s",
                            ctx.tool_name,
                            ctx.step_num,
                            self.attempts_by_reason[RetryReason.MLS_DEBUG.value],
                        )
                        continue
                    if debug_result.report_for_cb:
                        last_failure_reason = debug_result.report_for_cb
                    # Roll back the speculative claim — we never actually retried.
                    self.attempts_by_reason[RetryReason.MLS_DEBUG.value] -= 1

                return ExecutionResult(
                    success=False,
                    raw_output=raw_output,
                    failure_reason=last_failure_reason,
                    failure_type=last_failure_type,
                    attempts_by_reason=dict(self.attempts_by_reason),
                )

            # Tool call succeeded — but its body might still report failure.
            is_failure, derived_reason = _tool_output_indicates_failure(raw_output)

            # First post-check: MLS semantic/format verifier.
            mls_result = await run_mls_post_check(ctx, raw_output)
            if not mls_result.ok:
                if mls_result.retry_input is not None and self._claim(RetryReason.POST_STEP_VERIFY):
                    ctx.invoke_input = merge_retry_input(
                        ctx, ctx.invoke_input, mls_result.retry_input
                    )
                    _logger.info(
                        "Post-step retry: tool=%s, step=%s, retry=%s",
                        ctx.tool_name,
                        ctx.step_num,
                        self.attempts_by_reason[RetryReason.POST_STEP_VERIFY.value],
                    )
                    continue
                # Fall through: treat as failure (raw_output rewritten).
                raw_output = mls_result.raw_output or raw_output
                is_failure = True
                derived_reason = mls_result.failure_reason
                last_failure_type = "post_step"

            # Second post-check: CB content match (only when not already failed).
            if not is_failure:
                cb_result = await run_cb_post_check(ctx, raw_output)
                if not cb_result.ok:
                    # CB mismatch path: ask MLS debug for new input first.
                    if self._claim(RetryReason.CB_MISMATCH):
                        debug_result = await debug_via_mls(ctx, cb_result.failure_reason)
                        if debug_result.retry_input is not None:
                            ctx.invoke_input = merge_retry_input(
                                ctx, ctx.invoke_input, debug_result.retry_input
                            )
                            _logger.info(
                                "CB retry: tool=%s, step=%s, retry=%s",
                                ctx.tool_name,
                                ctx.step_num,
                                self.attempts_by_reason[RetryReason.CB_MISMATCH.value],
                            )
                            continue
                        # No new input — roll back the claim.
                        self.attempts_by_reason[RetryReason.CB_MISMATCH.value] -= 1
                    raw_output = cb_result.raw_output or raw_output
                    is_failure = True
                    derived_reason = cb_result.failure_reason
                    last_failure_type = "cb_mismatch"

            # Tool body says failure but mls passed: try MLS debug rescue.
            if is_failure:
                if self._claim(RetryReason.MLS_DEBUG):
                    debug_result = await debug_via_mls(
                        ctx, derived_reason or str(raw_output)
                    )
                    if debug_result.retry_input is not None:
                        ctx.invoke_input = merge_retry_input(
                            ctx, ctx.invoke_input, debug_result.retry_input
                        )
                        _logger.info(
                            "Failure retry: tool=%s, step=%s, retry=%s",
                            ctx.tool_name,
                            ctx.step_num,
                            self.attempts_by_reason[RetryReason.MLS_DEBUG.value],
                        )
                        continue
                    if debug_result.report_for_cb:
                        derived_reason = debug_result.report_for_cb
                    self.attempts_by_reason[RetryReason.MLS_DEBUG.value] -= 1

                return ExecutionResult(
                    success=False,
                    raw_output=raw_output,
                    failure_reason=derived_reason or last_failure_reason,
                    failure_type=last_failure_type if last_failure_type != "unknown" else "tool",
                    attempts_by_reason=dict(self.attempts_by_reason),
                )

            # Success!
            return ExecutionResult(
                success=True,
                raw_output=raw_output,
                attempts_by_reason=dict(self.attempts_by_reason),
            )

    def _claim(self, reason: RetryReason) -> bool:
        """Increment-and-check: returns True if a retry slot was available."""
        key = reason.value
        budget = getattr(self.budget, key, 0)
        if self.attempts_by_reason[key] >= budget:
            return False
        self.attempts_by_reason[key] += 1
        return True
