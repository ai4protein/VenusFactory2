"""CB planning nodes: produce normalized pipeline JSON from PI inputs."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Iterable

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


import re as _re_dep_lint

# Legal dependency token: dependency:step_N[:field...] (deps.py:32-53).
# The N digits are required; "step_" prefix is optional.
_LEGAL_DEP_TOKEN_RE = _re_dep_lint.compile(r"^dependency:step_?\d+(:[\w./-]+)*$")


def _sanitize_dependency_tokens(value: Any, *, step_no: int, tool_name: str) -> Any:
    """Walk ``value`` and strip the ``dependency:`` prefix from malformed tokens.

    The CB planner LLM occasionally hallucinates non-DSL placeholders such as
    ``dependency:system:session_dir/foo`` or ``dependency:user_input/bar``.
    Without sanitization, these slip into ``tool_input`` and ``deps.py`` later
    raises ``Invalid dependency step token`` — which cascades through every
    downstream step that depends on this one. The fix:

    * Tokens matching ``dependency:step_N[:field...]`` are left alone (valid).
    * Anything else starting with ``dependency:`` has its prefix stripped, so
      the residue is treated as a literal path/string and the tool either
      consumes it directly or surfaces a real validation error (instead of an
      opaque dependency-resolution error that masks the planner mistake).
    """
    if isinstance(value, str):
        if value.startswith("dependency:") and not _LEGAL_DEP_TOKEN_RE.match(value):
            stripped = value[len("dependency:"):]
            _logger.warning(
                "Planner lint: step %s (%s) had malformed dependency token %r; "
                "stripped 'dependency:' prefix → %r",
                step_no, tool_name, value, stripped,
            )
            return stripped
        return value
    if isinstance(value, list):
        return [_sanitize_dependency_tokens(v, step_no=step_no, tool_name=tool_name) for v in value]
    if isinstance(value, dict):
        return {k: _sanitize_dependency_tokens(v, step_no=step_no, tool_name=tool_name) for k, v in value.items()}
    return value


def _filter_unparameterized_steps(
    plan: list[dict[str, Any]],
    all_tools: Iterable[Any] | None,
    ui_lang: str,
) -> tuple[list[dict[str, Any]], list[tuple[Any, str, list[str]]]]:
    """Drop plan steps whose tool requires fields but tool_input is empty.

    Conservative filter: only removes a step when (a) the tool is known,
    (b) tool_input is an empty dict, and (c) the tool's args_schema declares
    at least one required field. Steps with partial tool_input are kept so
    the MLS debug pass can still attempt to recover defaults. Returns
    (kept_plan, dropped_steps) where each dropped entry is
    (step_no, tool_name, sorted_required_field_names).
    """
    tool_by_name: dict[str, Any] = {}
    for t in all_tools or []:
        name = getattr(t, "name", None)
        if isinstance(name, str) and name:
            tool_by_name[name] = t

    kept: list[dict[str, Any]] = []
    dropped: list[tuple[Any, str, list[str]]] = []
    for step in plan:
        tname = step.get("tool_name") or ""
        tool = tool_by_name.get(tname)
        ti = step.get("tool_input") or {}
        # Unknown tool or non-empty input → keep (let downstream handle).
        if tool is None or ti:
            kept.append(step)
            continue
        schema = getattr(tool, "args_schema", None)
        required: set[str] = set()
        if schema is not None:
            model_fields = getattr(schema, "model_fields", None)  # pydantic v2
            if isinstance(model_fields, dict) and model_fields:
                for fname, finfo in model_fields.items():
                    try:
                        if finfo.is_required():
                            required.add(fname)
                    except Exception:
                        # Fallback: treat fields with no default as required.
                        if getattr(finfo, "default", None) is None and \
                                getattr(finfo, "default_factory", None) is None:
                            required.add(fname)
            else:
                v1_fields = getattr(schema, "__fields__", None)  # pydantic v1
                if isinstance(v1_fields, dict):
                    for fname, finfo in v1_fields.items():
                        if getattr(finfo, "required", False):
                            required.add(fname)
        if required:
            dropped.append((step.get("step"), tname, sorted(required)))
            continue
        kept.append(step)
    return kept, dropped


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
        content = getattr(raw_msg, "content", None) or ""
        # Reasoning-style models (DeepSeek-V4-Pro, GLM-4.6) sometimes return
        # ``content=''`` and put the actual JSON inside
        # ``additional_kwargs.reasoning_content``. Fall back to it before
        # giving up — the parser is tolerant enough to pick the JSON block.
        if not str(content).strip():
            extra = getattr(raw_msg, "additional_kwargs", None) or {}
            reasoning = ""
            if isinstance(extra, dict):
                reasoning = str(extra.get("reasoning_content") or "")
            if reasoning.strip():
                _logger.info("CB planner: content empty, falling back to reasoning_content (len=%d)", len(reasoning))
                content = reasoning
        if not str(content).strip():
            # Last resort: stringify the message so downstream debugging shows what came back
            content = str(raw_msg) or ""
        _logger.info("CB planner raw output (first 1200 chars): %s", str(content)[:1200])
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
        _tool_input = _normalize_tool_input(p.get("tool_input") or p.get("input"))
        _tool_input = _sanitize_dependency_tokens(_tool_input, step_no=i + 1, tool_name=tname.strip())
        normalized_plan.append({
            "step": _normalize_step_number(p.get("step"), i + 1),
            "task_description": p.get("task_description") or p.get("task") or "",
            "tool_name": tname.strip(),
            "tool_input": _tool_input,
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

    # Drop steps whose tool_input is empty but whose tool requires fields:
    # those will otherwise fail Pydantic validation at execute time. Keep
    # partially-parameterized steps so MLS debug can still recover them.
    normalized_plan, dropped_unparam = _filter_unparameterized_steps(
        normalized_plan, chains.get("all_tools"), ui_lang
    )
    if dropped_unparam:
        for step_no, tool_name, reqs in dropped_unparam:
            _logger.warning(
                "Dropped plan step %s (%s): tool_input empty but requires %s",
                step_no, tool_name, reqs,
            )
        if ui_lang == "zh":
            drop_lines = [
                f"  - 第 {sn} 步 `{tn}` 缺少必填参数：{', '.join(reqs)}"
                for sn, tn, reqs in dropped_unparam
            ]
            drop_msg = (
                "⚠️ 已自动跳过以下计划步骤（参数为空但工具需要必填项）：\n"
                + "\n".join(drop_lines)
                + "\n建议重新生成 plan 或手动编辑。"
            )
        else:
            drop_lines = [
                f"  - Step {sn} `{tn}` requires: {', '.join(reqs)}"
                for sn, tn, reqs in dropped_unparam
            ]
            drop_msg = (
                "⚠️ Auto-dropped plan steps (empty params but tool requires inputs):\n"
                + "\n".join(drop_lines)
                + "\nConsider rerunning the plan or editing manually."
            )
        history.append({
            "role": "assistant",
            "content": drop_msg,
            "role_id": "computational_biologist",
        })

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
