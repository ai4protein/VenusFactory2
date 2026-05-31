"""LangGraph orchestration for the VenusFactory2 Agent — backward-compat shim.

Historically this module was a 1977-line monolith hosting every node, helper
and routing function used by the agent ``StateGraph``. The implementation now
lives in :mod:`agent.graph`:

- ``agent.graph.state``      → :class:`AgentState`
- ``agent.graph.routing``    → ``router_node`` + ``should_continue*`` + ``_route_from_*``
- ``agent.graph.chat``       → ``chat_start_node`` / ``chat_node``
- ``agent.graph.research``   → PI clarify/search/sub_report/report nodes
- ``agent.graph.planning``   → CB plan nodes
- ``agent.graph.execution``  → MLS ``execute_start_node`` + ``execute_node_impl``
- ``agent.graph.finalize``   → SC finalize nodes
- ``agent.graph.common``     → ``_ui_text`` / ``_stream_chain`` / ``_ensure_trace`` /
  ``_detect_ui_lang`` / ``_extract_usage_from_output``
- ``agent.graph.helpers``    → planner, chat-history, and tool I/O helpers

This module only re-exports those names so that downstream callers using
``from agent.chat_graph import X`` keep working without modification, and
hosts the small :func:`execute_node` wrapper that owns trace/span management.
"""
from __future__ import annotations

from langchain_core.runnables import RunnableConfig

# Shared package-level state (single instance lives on ``agent.graph``).
from agent.graph import _BG_TASKS, _TOOL_TIMEOUTS
from agent.graph.chat import chat_node, chat_start_node
from agent.graph.common import (
    _detect_ui_lang,
    _ensure_trace,
    _extract_usage_from_output,
    _stream_chain,
    _ui_text,
)
from agent.graph.compile import create_agent_graph
from agent.graph.execution import execute_start_node
from agent.graph.finalize import finalize_node, finalize_start_node
from agent.graph.helpers import (
    _canonicalize_tool_name,
    _collect_output_fields,
    _enforce_skill_first_plan,
    _extract_skill_ids_from_metadata,
    _format_clarification_answers,
    _format_conversation_history,
    _get_chat_history_messages,
    _get_step_raw_output,
    _get_tool_allowed_param_names,
    _is_write_like_tool,
    _looks_like_execution_request,
    _normalize_output_paths,
    _normalize_step_number,
    _normalize_tool_input,
    _parse_clarification_questions,
    _parse_sections,
    _pick_skill_for_code_step,
    _repair_plan_json_with_llm,
    _retry_plan_for_model_compat,
    _sanitize_tool_invoke_input,
)
from agent.graph.planning import _plan_node_impl, plan_node, plan_start_node
from agent.graph.research import (
    clarification_node,
    clarification_start_node,
    research_plan_node,
    research_plan_start_node,
    research_report_node,
    research_report_start_node,
    research_search_node,
    research_search_start_node,
    research_sub_report_node,
    research_sub_report_start_node,
)
from agent.graph.routing import (
    _route_from_clarification,
    _route_from_plan,
    _route_from_research_plan,
    _route_from_router,
    router_node,
    should_continue,
    should_continue_research,
)
from agent.graph.state import AgentState
from agent.tracing import AgentSpanData, start_span


async def execute_node(state: AgentState, config: RunnableConfig):
    """MLS execution node: executes current step in plan.

    Owns trace/span management around the orchestrator implementation
    (lives in :mod:`agent.graph.execution`). Kept on this module so the
    ``langgraph`` workflow keeps the original module-qualified callable
    identity (``agent.chat_graph.execute_node``) for any external graph
    visualisers / checkpoints that recorded the original path.
    """
    plan = state.get("plan", [])
    idx = state.get("current_step_index", 0)
    step_num_for_span = _normalize_step_number(
        plan[idx].get("step") if (plan and idx < len(plan)) else None,
        idx + 1,
    )
    with _ensure_trace(session_id=state.get("session_id", "")), \
            start_span(
                "mls.execute",
                AgentSpanData(agent_name="MLS", phase=f"step_{step_num_for_span}"),
            ):
        return await _execute_node_impl(state, config)


async def _execute_node_impl(state: AgentState, config: RunnableConfig):
    """Thin delegator: implementation lives in ``agent.graph.execution``.

    Kept as a module-level function so the wrapper :func:`execute_node`
    (which still owns trace/span management) can call it via the original
    name and so external ``from agent.chat_graph import _execute_node_impl``
    imports keep working.
    """
    # Lazy import avoids an import cycle: the execution package itself imports
    # a few helpers (e.g. ``_normalize_step_number``) from this module.
    from agent.graph.execution import execute_node_impl as _impl
    return await _impl(state, config)


__all__ = [
    # state + graph assembly
    "AgentState",
    "create_agent_graph",
    # execution wrapper (lives here for trace/span management)
    "execute_node",
    "execute_start_node",
    "_execute_node_impl",
    # routing
    "router_node",
    "should_continue",
    "should_continue_research",
    "_route_from_router",
    "_route_from_clarification",
    "_route_from_research_plan",
    "_route_from_plan",
    # chat
    "chat_node",
    "chat_start_node",
    # research
    "research_plan_node",
    "research_plan_start_node",
    "clarification_node",
    "clarification_start_node",
    "research_search_node",
    "research_search_start_node",
    "research_sub_report_node",
    "research_sub_report_start_node",
    "research_report_node",
    "research_report_start_node",
    # planning
    "plan_node",
    "plan_start_node",
    "_plan_node_impl",
    # finalize
    "finalize_node",
    "finalize_start_node",
    # common
    "_detect_ui_lang",
    "_ensure_trace",
    "_extract_usage_from_output",
    "_stream_chain",
    "_ui_text",
    # helpers
    "_canonicalize_tool_name",
    "_collect_output_fields",
    "_enforce_skill_first_plan",
    "_extract_skill_ids_from_metadata",
    "_format_clarification_answers",
    "_format_conversation_history",
    "_get_chat_history_messages",
    "_get_step_raw_output",
    "_get_tool_allowed_param_names",
    "_is_write_like_tool",
    "_looks_like_execution_request",
    "_normalize_output_paths",
    "_normalize_step_number",
    "_normalize_tool_input",
    "_parse_clarification_questions",
    "_parse_sections",
    "_pick_skill_for_code_step",
    "_repair_plan_json_with_llm",
    "_retry_plan_for_model_compat",
    "_sanitize_tool_invoke_input",
    # shared state (kept here for backward import)
    "_BG_TASKS",
    "_TOOL_TIMEOUTS",
]
