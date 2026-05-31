"""MLS execution preamble: emit 'executing Step N' UI line before tool runs."""
from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from agent.graph.common.lang import _detect_ui_lang
from agent.graph.common.ui_text import _ui_text
from agent.graph.helpers.plan_helpers import _normalize_step_number
from agent.graph.state import AgentState


async def execute_start_node(state: AgentState, config: RunnableConfig):
    """Show 'MLS is executing step N' so UI updates before tool runs."""
    plan = state.get("plan", [])
    idx = state.get("current_step_index", 0)
    history = list(state.get("history", []))
    ui_lang = state.get("ui_lang") or _detect_ui_lang(state["messages"][-1].content)
    if idx >= len(plan):
        return {}
    step = plan[idx]
    step_num = _normalize_step_number(step.get("step"), idx + 1)
    task_desc = step.get("task_description", "…")
    history.append({
        "role": "assistant",
        "content": _ui_text(ui_lang, "executing_step", step_num=step_num, task_desc=task_desc),
        "role_id": "machine_learning_specialist",
    })
    return {"history": history}
