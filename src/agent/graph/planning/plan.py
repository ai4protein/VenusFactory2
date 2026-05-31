"""CB planning nodes: produce normalized pipeline JSON from PI inputs."""
from __future__ import annotations

import json
from datetime import datetime

from langchain_core.runnables import RunnableConfig

from agent.chat_agent_utils import PI_SEARCH_TOOL_NAMES, _parse_cb_plan
from agent.graph.common.lang import _detect_ui_lang
from agent.graph.common.streaming import _ensure_trace
from agent.graph.common.ui_text import _ui_text
from agent.graph.helpers.plan_helpers import (
    _enforce_skill_first_plan,
    _looks_like_execution_request,
    _normalize_step_number,
    _retry_plan_for_model_compat,
)
from agent.graph.helpers.tool_io import _normalize_tool_input
from agent.graph.state import AgentState
from agent.skills import get_skills_metadata_string
from agent.tracing import AgentSpanData, start_span
from logger import get_logger
from web.utils.common_utils import to_project_relative_path

_logger = get_logger("agent.graph")


async def plan_start_node(state: AgentState, config: RunnableConfig):
    """Show 'CB is designing pipeline' so UI updates before LLM runs."""
    history = list(state.get("history", []))
    ui_lang = state.get("ui_lang") or _detect_ui_lang(state["messages"][-1].content)
    history.append({
        "role": "assistant",
        "content": "📋 **Computational Biologist** 正在设计流程 …"
        if ui_lang == "zh" else
        "📋 **Computational Biologist** is designing the pipeline …",
        "role_id": "computational_biologist",
    })
    return {"history": history}


async def plan_node(state: AgentState, config: RunnableConfig):
    """CB planning node: PI report -> pipeline (JSON list)."""
    with _ensure_trace(session_id=state.get("session_id", "")), \
            start_span("cb.plan", AgentSpanData(agent_name="CB", phase="planning")):
        return await _plan_node_impl(state, config)


async def _plan_node_impl(state: AgentState, config: RunnableConfig):
    chains = config.get("configurable", {}).get("chains", {})
    pi_report = state.get("pi_report", "")
    pi_suggest_steps = state.get("pi_suggest_steps", "")
    protein_ctx = state["protein_context"]
    history = list(state.get("history", []))
    log_entries = list(state.get("conversation_log", []))
    ui_lang = state.get("ui_lang") or _detect_ui_lang(state["messages"][-1].content)

    # When research is skipped, use user's original input as the "PI report"
    if not pi_report:
        user_input = state["messages"][-1].content
        if ui_lang == "zh":
            pi_report = f"用户请求：{user_input}\n\n无需文献检索，直接进入工具执行。"
            pi_suggest_steps = pi_suggest_steps or "执行适当工具以完成用户请求。"
        else:
            pi_report = f"User request: {user_input}\n\nNo literature research needed. Proceed directly with tool execution."
            pi_suggest_steps = pi_suggest_steps or "Execute the appropriate tool to fulfill the user's request."

    context_parts = [
        f"蛋白上下文：{protein_ctx.get_context_summary()}"
        if ui_lang == "zh"
        else f"Protein context: {protein_ctx.get_context_summary()}"
    ]
    if state.get("agent_session_dir"):
        context_parts.append(
            f"默认输出目录：{to_project_relative_path(state['agent_session_dir'])}"
            if ui_lang == "zh"
            else f"Default output directory: {to_project_relative_path(state['agent_session_dir'])}"
        )
    protein_context_summary = "; ".join(context_parts)

    recent_tool_calls = protein_ctx.get_tool_records(limit=10)
    tool_outputs_summary = json.dumps(recent_tool_calls, ensure_ascii=False)

    # Pass tools and skills explicitly so CB always sees the full list
    # (session_state may be the only source).
    tools_description = chains.get("tools_description") or ""
    skills_metadata = chains.get("skills_metadata") or get_skills_metadata_string()
    available_tools_list = chains.get("available_tools_list") or ""
    if not available_tools_list and chains.get("workers"):
        available_tools_list = ", ".join(chains["workers"].keys())

    cb_planner_inputs = {
        "pi_report": pi_report,
        "pi_suggest_steps": pi_suggest_steps,
        "protein_context_summary": protein_context_summary,
        "tool_outputs": tool_outputs_summary,
        "tools_description": tools_description,
        "skills_metadata": skills_metadata,
        "available_tools_list": available_tools_list,
    }
    _logger.info(
        "CB planner inputs summary: user_len=%d pi_report_len=%d suggest_len=%d tools_count=%d",
        len(str(state["messages"][-1].content or "")),
        len(str(pi_report or "")),
        len(str(pi_suggest_steps or "")),
        len([x for x in str(available_tools_list or "").split(",") if x.strip()]),
    )

    try:
        raw_msg = await chains["cb_planner_raw"].ainvoke(cb_planner_inputs)
        content = getattr(raw_msg, "content", None) or str(raw_msg) or ""
        _logger.info("CB planner raw output (first 1200 chars): %s", content[:1200])
        plan = _parse_cb_plan(content)
        _logger.info("CB planner parsed steps count: %d", len(plan) if isinstance(plan, list) else 0)
        if (not plan) and _looks_like_execution_request(state["messages"][-1].content):
            _logger.info("CB planner returned empty plan for execution-style request; starting compat retry.")
            plan = await _retry_plan_for_model_compat(
                llm=chains.get("llm"),
                user_text=state["messages"][-1].content,
                pi_report=pi_report,
                pi_suggest_steps=pi_suggest_steps,
                protein_context_summary=protein_context_summary,
                available_tools_list=available_tools_list,
            )
    except Exception as e:
        _logger.warning("CB planner failed: %s", e)
        plan = []

    # Filter and normalize plan
    normalized_plan = []
    dropped_non_dict = 0
    dropped_empty_or_missing_tool = 0
    dropped_pi_search_tool = 0
    for i, p in enumerate(plan):
        if not isinstance(p, dict):
            dropped_non_dict += 1
            continue
        tname = p.get("tool_name") or p.get("tool") or ""
        if not tname:
            dropped_empty_or_missing_tool += 1
            continue
        if tname in PI_SEARCH_TOOL_NAMES:
            dropped_pi_search_tool += 1
            continue
        normalized_plan.append({
            "step": _normalize_step_number(p.get("step"), i + 1),
            "task_description": p.get("task_description") or p.get("task") or "",
            "tool_name": tname.strip(),
            "tool_input": _normalize_tool_input(p.get("tool_input") or p.get("input")),
        })
    _logger.info(
        "CB plan normalization: kept=%d dropped_non_dict=%d dropped_no_tool=%d dropped_pi_search=%d available_tools=%s",
        len(normalized_plan),
        dropped_non_dict,
        dropped_empty_or_missing_tool,
        dropped_pi_search_tool,
        available_tools_list,
    )

    normalized_plan = _enforce_skill_first_plan(normalized_plan, available_tools_list, skills_metadata, ui_lang)

    if not normalized_plan:
        # Safety fallback: CB planner produced no usable steps (typically because
        # the LLM call timed out or returned malformed JSON). DON'T silently route
        # to chat_mode and let the LLM hallucinate fake tool execution — instead
        # surface an honest error so the user can retry or pick a different model.
        failure_msg = (
            "⚠️ **规划失败**：模型未能生成可执行的工具计划（可能因为请求超时或输出格式异常）。"
            "请尝试：(1) 用更具体的描述重试；(2) 切换更稳定的模型；(3) 拆分为更小的子任务。"
            "**注意**：当前无法执行任何工具，下面没有真实结果。"
            if ui_lang == "zh" else
            "⚠️ **Planning failed**: the model did not produce an executable tool plan "
            "(possibly due to a request timeout or malformed output). Try: (1) rephrasing "
            "more specifically; (2) switching to a more stable model; (3) breaking the task "
            "into smaller subtasks. **Note**: no tools were executed; there are no real "
            "results to show."
        )
        history.append({
            "role": "assistant",
            "content": failure_msg,
            "role_id": "computational_biologist",
        })
        return {
            "plan": [],
            "history": history,
            "status": "planning_failed",
            "error": "CB planner returned no usable steps",
        }

    step_lines = [
        (
            f"**第 {p['step']} 步。** {p['task_description']}"
            if ui_lang == "zh"
            else f"**Step {p['step']}.** {p['task_description']}"
        )
        for p in normalized_plan
    ]
    plan_text = _ui_text(ui_lang, "plan_confirmation_title", steps="\n\n".join(step_lines))
    history.append({"role": "assistant", "content": plan_text, "role_id": "computational_biologist"})
    log_entries.append({
        "role": "assistant",
        "content": plan_text,
        "role_id": "computational_biologist",
        "timestamp": datetime.now().isoformat(),
    })

    return {
        "plan": normalized_plan,
        "current_step_index": 0,
        "step_results": {},
        "history": history,
        "conversation_log": log_entries,
        "execution_failed": False,
        "failed_step": None,
        "failed_reason": None,
        "skipped_steps": [],
        "ui_lang": ui_lang,
        "waiting_for": "plan_confirmation",
        "status": "waiting_for_plan_confirmation",
    }
