"""SC finalizer nodes: summarize the run and emit iteration prompt."""
from __future__ import annotations

import json

from langchain_core.runnables import RunnableConfig

from agent.chat_agent_utils import _tool_output_indicates_failure
from agent.graph.common.lang import _detect_ui_lang
from agent.graph.common.streaming import _stream_chain
from agent.graph.common.ui_text import _ui_text
from agent.graph.state import AgentState
from logger import get_logger

_logger = get_logger("agent.graph")


async def finalize_start_node(state: AgentState, config: RunnableConfig):
    """Show 'Summarizing' so UI updates before LLM runs."""
    history = list(state.get("history", []))
    ui_lang = state.get("ui_lang") or _detect_ui_lang(state["messages"][-1].content)
    history.append({
        "role": "assistant",
        "content": _ui_text(ui_lang, "summarizing"),
        "role_id": "scientific_critic",
        "phase": "summarizing",
    })
    return {"history": history, "ui_lang": ui_lang}


async def finalize_node(state: AgentState, config: RunnableConfig):
    """Finalizer node: generates final summary using tool execution history."""
    chains = config.get("configurable", {}).get("chains", {})
    history = list(state.get("history", []))
    tool_executions = state.get("tool_executions", [])
    protein_ctx = state["protein_context"]
    user_input = state["messages"][-1].content
    ui_lang = state.get("ui_lang") or _detect_ui_lang(user_input)

    # Build the full run record for the finalizer
    analysis_log = []
    for i, entry in enumerate(tool_executions, 1):
        step = entry.get("step", i)
        tool_name = entry.get("tool_name", "unknown")
        inputs = entry.get("inputs", {})
        outputs = entry.get("outputs", "")
        oss_url = entry.get("oss_url")
        analysis_log.append(
            f"Step {step}: {tool_name}\n"
            f"  Input: {json.dumps(inputs, ensure_ascii=False)}\n"
            f"  Output: {str(outputs)[:2000]}\n"
            + (f"  Cloud Download: {oss_url}" if oss_url else "")
        )

    # Include PI research report and sub-reports so SC has full context
    pi_report = state.get("pi_report", "")
    sub_reports = state.get("research_sub_reports", [])
    sub_reports_text = "\n\n".join(sub_reports) if sub_reports else ""

    record_parts = [f"User request: {user_input}"]
    record_parts.append(f"Protein context: {protein_ctx.get_context_summary()}")
    if pi_report:
        record_parts.append(f"Principal Investigator research report:\n{pi_report}")
    if sub_reports_text:
        record_parts.append(f"Research sub-reports:\n{sub_reports_text}")
    if analysis_log:
        record_parts.append("Tool executions:\n" + "\n".join(analysis_log))
    else:
        record_parts.append("No tools executed.")
    # Include skipped steps info
    skipped_steps = state.get("skipped_steps", [])
    if skipped_steps:
        record_parts.append(
            f"Skipped steps (failed but no downstream dependencies): {skipped_steps}\n"
            f"  Note: These steps failed but execution continued because no subsequent "
            f"steps depended on their output. Please note which steps were skipped and "
            f"their impact in your report."
        )
    # Include failure context if pipeline failed
    if state.get("execution_failed"):
        failed_step = state.get("failed_step")
        failed_reason = state.get("failed_reason", "Unknown error")
        record_parts.append(
            f"Pipeline failure:\n"
            f"  Failed at step: {failed_step}\n"
            f"  Failure reason: {failed_reason}\n"
            f"  Note: Downstream steps were skipped. Please analyze the failure cause "
            f"and suggest how to resolve it in your report."
        )
    full_run_record = "\n\n".join(record_parts)

    if history and (
        history[-1].get("phase") == "summarizing"
        or "Summarizing" in history[-1].get("content", "")
        or "正在总结" in history[-1].get("content", "")
        or "汇总" in history[-1].get("content", "")
    ):
        history.pop()

    try:
        summary = await _stream_chain(
            chains["finalizer"],
            {
                "input": user_input,
                "full_run_record": full_run_record,
                "original_input": user_input,
                "analysis_log": "\n".join(analysis_log) if analysis_log else "No analysis log available.",
                "references": "",
            },
            role_id="scientific_critic",
        )
        history.append({"role": "assistant", "content": summary, "role_id": "scientific_critic"})
    except Exception as e:
        _logger.warning("Finalizer failed: %s", e)
        # Fallback: derive status directly from recorded executions.
        if tool_executions:
            summary_parts = [_ui_text(ui_lang, "final_summary_title")]
            if state.get("execution_failed"):
                summary_parts.append(
                    _ui_text(
                        ui_lang,
                        "pipeline_paused_at",
                        step=state.get("failed_step"),
                        reason=state.get("failed_reason") or ("未知错误" if ui_lang == "zh" else "Unknown error"),
                    )
                )
            else:
                summary_parts.append(_ui_text(ui_lang, "task_completed"))
            for entry in tool_executions:
                output_text = entry.get("outputs", "")
                failed, reason = _tool_output_indicates_failure(output_text)
                if failed:
                    summary_parts.append(
                        _ui_text(
                            ui_lang,
                            "tool_failed",
                            tool=entry.get("tool_name", "Tool"),
                            reason=(reason or ("错误" if ui_lang == "zh" else "error"))[:120],
                        )
                    )
                else:
                    summary_parts.append(_ui_text(ui_lang, "tool_executed", tool=entry.get("tool_name", "Tool")))
            summary = "\n".join(summary_parts)
        else:
            summary = _ui_text(ui_lang, "task_ended")
        history.append({"role": "assistant", "content": summary, "role_id": "scientific_critic"})

    history.append({
        "role": "assistant",
        "content": _ui_text(ui_lang, "iteration_prompt"),
        "role_id": "scientific_critic",
    })
    return {"history": history, "status": "waiting_for_iteration", "waiting_for": "iteration"}
