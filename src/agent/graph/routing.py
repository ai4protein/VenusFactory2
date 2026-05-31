"""Top-level routing nodes and conditional edge functions for the agent graph."""
from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END

from agent.graph.state import AgentState


async def router_node(state: AgentState, config: RunnableConfig):
    """Entry node: routes to the appropriate phase based on resume state."""
    waiting_for = state.get("waiting_for")
    if waiting_for == "clarification_answered":
        return {"status": "resume_research"}
    elif waiting_for == "plan_confirmed":
        return {"status": "resume_execution"}
    elif waiting_for == "iteration_rerun":
        return {"status": "resume_execution"}
    elif waiting_for == "step_continue":
        return {"status": "resume_execution"}
    elif waiting_for == "step_abort":
        return {"status": "resume_finalize"}
    elif waiting_for == "sub_report_continue":
        return {"status": "resume_research"}
    elif waiting_for == "sub_report_skip":
        return {"status": "resume_report"}
    elif waiting_for == "sub_report_rewrite":
        return {"status": "resume_sub_report_rewrite"}
    elif waiting_for == "skip_to_plan":
        return {"status": "resume_plan"}
    return {"status": "new_request"}


def should_continue_research(state: AgentState):
    if state.get("status") == "planning_failed" or state.get("error"):
        return END

    research_idx = state.get("research_idx", 0)
    search_idx = state.get("search_idx", 0)
    sections = state.get("research_sections", [])

    if research_idx < len(sections):
        section = sections[research_idx]
        queries = section.get("search_queries", [])
        if search_idx < len(queries):
            return "research_search_start_node"
        else:
            return "research_sub_report_start_node"
    return "research_report_start_node"


def should_continue(state: AgentState):
    if state.get("status") == "planning_failed" or state.get("error"):
        return END
    if state.get("execution_failed"):
        return "finalize_start_node"
    if state.get("status") == "waiting_for_step_review":
        return END

    plan = state.get("plan", [])
    current_idx = state.get("current_step_index", 0)

    if current_idx < len(plan):
        return "execute_start_node"
    return "finalize_start_node"


def _route_from_router(state: AgentState):
    status = state.get("status", "")
    if status == "resume_research":
        return "research_search_start_node"
    if status == "resume_execution":
        return "execute_start_node"
    if status == "resume_finalize":
        return "finalize_start_node"
    if status == "resume_report":
        return "research_report_start_node"
    if status == "resume_plan":
        return "plan_start_node"
    if status == "resume_sub_report_rewrite":
        return "research_sub_report_start_node"
    return "research_plan_start_node"


def _route_from_clarification(state: AgentState):
    if state.get("status") == "resume_research":
        return "research_search_start_node"
    return END


def _route_from_research_plan(state: AgentState):
    status = state.get("status", "")
    if status == "chat_mode":
        return "chat_start_node"
    if status == "resume_plan":
        return "plan_start_node"
    return "clarification_start_node"


def _route_from_plan(state: AgentState):
    status = state.get("status", "")
    if status == "chat_mode":
        return "chat_start_node"
    if status == "planning_failed" or not state.get("plan"):
        return "finalize_start_node"
    return END
