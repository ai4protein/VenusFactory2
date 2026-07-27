"""Thin orchestrator for ``execute_node``.

Mirrors the legacy ``_execute_node_impl`` signature; the wrapper ``execute_node``
in :mod:`agent.chat_graph` still owns trace/span management and simply delegates
here.
"""

from __future__ import annotations

import json
import time
from typing import Any

from langchain_core.runnables import RunnableConfig

from logger import get_logger

from agent.chat_agent_utils import _tool_output_indicates_failure
from agent.graph.execution.artifacts import register_artifacts_to_context
from agent.graph.execution.context import ExecutionContext, ExecutionResult
from agent.graph.execution.deps import resolve_dependencies
from agent.graph.execution.feedback import (
    build_feedback_content,
    dispatch_hook_on_tool_end,
    record_execution_entry,
    schedule_artifact_uploads,
)
from agent.graph.execution.invoke import lookup_cache, save_cache
from agent.graph.execution.path_repair import (
    PathRepairScope,
    repair_missing_file_paths,
)
from agent.graph.execution.prepare import prepare_invoke_input, run_pre_checks
from agent.graph.execution.retry_orchestrator import RetryOrchestrator

_logger = get_logger("agent.graph")


def _failure_payload_to_result(
    raw_output: str, failure_reason: str, failure_type: str
) -> ExecutionResult:
    return ExecutionResult(
        success=False,
        raw_output=raw_output,
        failure_reason=failure_reason,
        failure_type=failure_type,  # type: ignore[arg-type]
    )


def _normalize_step_number_for_state(ctx: ExecutionContext) -> str:
    return str(ctx.step_num)


async def execute_node_impl(state: dict, config: RunnableConfig) -> dict:
    """Run the current plan step end-to-end and return the updated state dict.

    Behavioural parity goal vs. legacy ``_execute_node_impl``: identical
    ``return`` dict keys, identical raw_output strings on failure, identical
    history/log/executions side-effects.
    """
    ctx = ExecutionContext.from_state(state, config)
    execute_started = time.time()

    last_output: Any = None
    failure_reason: str = ""
    cached_flag = False

    # ---- 1. Resolve dependencies ----
    deps_result = resolve_dependencies(ctx)
    if not deps_result.ok:
        failure_reason = deps_result.reason or "Dependency resolution failed."
        last_output = json.dumps(
            {
                "status": "error",
                "error": {"type": "DependencyResolutionError", "message": failure_reason},
                "file_info": None,
            },
            ensure_ascii=False,
        )
    else:
        ctx.merged_tool_input = deps_result.invoke_input

        # ---- 2. Pre-checks (skill precondition, disabled, missing tool) ----
        pre_check = run_pre_checks(ctx)
        if not pre_check.ok:
            failure_reason = pre_check.reason
            last_output = pre_check.raw_output
        else:
            # ---- 3. Sanitize / normalize / repair file inputs ----
            ctx.invoke_input = prepare_invoke_input(ctx)
            ctx.invoke_input = repair_missing_file_paths(
                ctx, scope=PathRepairScope.SESSION_AND_PREV_STEPS
            )

            # ---- 4. Cache lookup ----
            cached = lookup_cache(ctx)
            if cached is not None:
                last_output = cached
                cached_flag = True
            else:
                # ---- 5. Invoke + retry orchestration ----
                orchestrator = RetryOrchestrator(ctx)
                result = await orchestrator.run()
                last_output = result.raw_output
                if not result.success:
                    failure_reason = result.failure_reason
                else:
                    save_cache(ctx, result.raw_output)

    # ---- 6. Artifact registration to protein_ctx ----
    if last_output is not None:
        try:
            out_failed, _ = _tool_output_indicates_failure(last_output)
            if not out_failed:
                register_artifacts_to_context(ctx, last_output)
        except Exception as e:
            _logger.warning("Artifact register skipped: %s", e)

    # Record result
    ctx.protein_ctx.add_tool_call(
        ctx.step_num,
        ctx.tool_name,
        ctx.merged_tool_input,
        last_output,
        cached=cached_flag,
    )
    ctx.step_results[ctx.step_num] = {"raw_output": last_output}

    # ---- 7. Feedback rendering + background uploads ----
    is_failure, parsed_failure_reason = _tool_output_indicates_failure(last_output)
    if not failure_reason:
        failure_reason = parsed_failure_reason

    feedback_content, _out_data, out_file, sse_writer = build_feedback_content(
        ctx, last_output, is_failure
    )
    feedback_content += schedule_artifact_uploads(ctx, last_output, out_file, sse_writer)

    record_execution_entry(ctx, last_output, oss_url=None)

    # Tool-call observability hooks (analytics, webhooks).
    from agent.graph import _BG_TASKS

    dispatch_hook_on_tool_end(
        ctx,
        last_output,
        is_failure,
        failure_reason,
        execute_started,
        _BG_TASKS,
    )

    # ---- 8. Decide skip vs. pause vs. continue ----
    skipped_steps = list(ctx.state.get("skipped_steps", []))
    can_skip_failure = False
    if is_failure:
        failed_step_str = str(ctx.step_num)
        has_downstream_dep = False
        for future_step in ctx.plan[ctx.idx + 1 :]:
            future_input = future_step.get("tool_input", {})
            if isinstance(future_input, dict):
                for _fv in future_input.values():
                    if isinstance(_fv, str) and _fv.startswith("dependency:"):
                        dep_token = (
                            _fv.split(":")[1].replace("step_", "").replace("step", "").strip()
                        )
                        if dep_token == failed_step_str:
                            has_downstream_dep = True
                            break
            if has_downstream_dep:
                break
        can_skip_failure = not has_downstream_dep

    from agent.graph.common.ui_text import _ui_text
    from agent.graph.helpers.plan_helpers import _normalize_step_number

    if is_failure and can_skip_failure:
        feedback_content += _ui_text(ctx.ui_lang, "step_skipped", step_num=ctx.step_num)
        skipped_steps.append(ctx.step_num)
    elif is_failure:
        feedback_content += _ui_text(ctx.ui_lang, "pipeline_paused")

    ctx.history.append(
        {
            "role": "assistant",
            "content": feedback_content,
            "role_id": "machine_learning_specialist",
        }
    )

    next_idx = ctx.idx + 1
    has_more_steps = next_idx < len(ctx.plan)
    # Expert default: auto-run remaining steps unless explicitly disabled.
    auto_execute = ctx.state.get("auto_execute") is not False

    if is_failure and not can_skip_failure:
        status = "execution_failed"
        waiting_for_val = None
    elif has_more_steps and not auto_execute:
        next_step = ctx.plan[next_idx]
        next_step_num = _normalize_step_number(next_step.get("step"), next_idx + 1)
        next_desc = next_step.get("task_description", "…")
        ctx.history.append(
            {
                "role": "assistant",
                "content": _ui_text(
                    ctx.ui_lang,
                    "step_checkpoint",
                    step_num=ctx.step_num,
                    next_step=next_step_num,
                    next_desc=next_desc,
                ),
                "role_id": "machine_learning_specialist",
            }
        )
        status = "waiting_for_step_review"
        waiting_for_val = "step_review"
    else:
        status = "executing"
        waiting_for_val = None

    return {
        "current_step_index": next_idx,
        "step_results": ctx.step_results,
        "history": ctx.history,
        "conversation_log": ctx.log_entries,
        "tool_executions": ctx.executions,
        "tool_cache": ctx.state.get("tool_cache", {}),
        "status": status,
        "execution_failed": is_failure and not can_skip_failure,
        "failed_step": ctx.step_num if is_failure and not can_skip_failure else None,
        "failed_reason": failure_reason if is_failure and not can_skip_failure else None,
        "skipped_steps": skipped_steps,
        "waiting_for": waiting_for_val,
        "ui_lang": ctx.ui_lang,
    }
