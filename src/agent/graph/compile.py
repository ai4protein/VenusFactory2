"""Assemble the agent ``StateGraph`` from the modular node functions."""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent.graph.chat import chat_node, chat_start_node
from agent.graph.execution import execute_start_node
from agent.graph.finalize import finalize_node, finalize_start_node
from agent.graph.planning import plan_node, plan_start_node
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


def create_agent_graph():
    # ``execute_node`` is the trace/span wrapper that still lives on
    # ``agent.chat_graph`` (it owns the legacy delegator + import-cycle break).
    # Import it lazily here to avoid a cycle at module load: the chat_graph
    # shim re-exports ``create_agent_graph`` from this module.
    from agent.chat_graph import execute_node

    workflow = StateGraph(AgentState)

    workflow.add_node("router_node", router_node)
    workflow.add_node("research_plan_start_node", research_plan_start_node)
    workflow.add_node("research_plan_node", research_plan_node)
    workflow.add_node("clarification_start_node", clarification_start_node)
    workflow.add_node("clarification_node", clarification_node)
    workflow.add_node("chat_start_node", chat_start_node)
    workflow.add_node("chat_node", chat_node)
    workflow.add_node("research_search_start_node", research_search_start_node)
    workflow.add_node("research_search_node", research_search_node)
    workflow.add_node("research_sub_report_start_node", research_sub_report_start_node)
    workflow.add_node("research_sub_report_node", research_sub_report_node)
    workflow.add_node("research_report_start_node", research_report_start_node)
    workflow.add_node("research_report_node", research_report_node)
    workflow.add_node("plan_start_node", plan_start_node)
    workflow.add_node("plan_node", plan_node)
    workflow.add_node("execute_start_node", execute_start_node)
    workflow.add_node("execute_node", execute_node)
    workflow.add_node("finalize_start_node", finalize_start_node)
    workflow.add_node("finalize_node", finalize_node)

    workflow.add_edge(START, "router_node")
    workflow.add_conditional_edges("router_node", _route_from_router)

    workflow.add_edge("research_plan_start_node", "research_plan_node")
    workflow.add_conditional_edges(
        "research_plan_node",
        _route_from_research_plan,
    )

    workflow.add_edge("clarification_start_node", "clarification_node")
    workflow.add_conditional_edges("clarification_node", _route_from_clarification)

    workflow.add_edge("chat_start_node", "chat_node")
    workflow.add_edge("chat_node", END)

    workflow.add_edge("research_search_start_node", "research_search_node")
    workflow.add_conditional_edges("research_search_node", should_continue_research)
    workflow.add_edge("research_sub_report_start_node", "research_sub_report_node")
    workflow.add_conditional_edges(
        "research_sub_report_node",
        lambda s: END if s.get("status") == "waiting_for_sub_report_review" else "research_search_start_node",
    )
    workflow.add_edge("research_report_start_node", "research_report_node")
    workflow.add_edge("research_report_node", "plan_start_node")

    workflow.add_edge("plan_start_node", "plan_node")
    workflow.add_conditional_edges("plan_node", _route_from_plan)

    workflow.add_edge("execute_start_node", "execute_node")
    workflow.add_conditional_edges("execute_node", should_continue)
    workflow.add_edge("finalize_start_node", "finalize_node")
    workflow.add_edge("finalize_node", END)

    return workflow.compile()
