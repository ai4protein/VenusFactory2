"""Execution context and result dataclasses.

These collect the 17 local variables that used to live as scratch state inside
``_execute_node_impl`` into a single object so the helper functions in this
package can share them without 17-argument signatures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from langchain_core.runnables import RunnableConfig

from agent.chat_agent import ProteinContextManager, _merge_tool_parameters_with_context


FailureType = Literal[
    "dependency",
    "guardrail",
    "tool",
    "timeout",
    "post_step",
    "cb_mismatch",
    "skill_precondition",
    "disabled",
    "unknown",
]


@dataclass
class ExecutionContext:
    """Bundle of all per-step execution state used across the execution package."""

    # ----- Inputs from LangGraph state / config -----
    state: dict
    config: RunnableConfig
    chains: dict

    plan: list[dict]
    idx: int
    step: dict
    step_num: int
    task_desc: str
    tool_name: str
    tool_input: dict
    tool: Optional[Any]

    protein_ctx: ProteinContextManager
    agent_session_dir: str
    ui_lang: str

    # Mutable lists/dicts copied from state (so callers mutate locally before
    # returning a merged update dict).
    history: list[dict]
    log_entries: list[dict]
    executions: list[dict]
    step_results: dict
    disabled_tool_names: set

    # The merged invocation input — populated by ``deps`` first, then mutated
    # in-place by ``prepare`` / ``path_repair`` / retry handlers.
    merged_tool_input: dict = field(default_factory=dict)
    invoke_input: dict = field(default_factory=dict)

    @classmethod
    def from_state(cls, state: dict, config: RunnableConfig) -> "ExecutionContext":
        """Build an ``ExecutionContext`` from raw LangGraph ``state``/``config``."""
        # Local imports avoid an import cycle (chat_graph imports this package
        # at module load to wire ``_execute_node_impl``).
        from agent.graph.common.lang import _detect_ui_lang
        from agent.graph.helpers.plan_helpers import _normalize_step_number

        chains = config.get("configurable", {}).get("chains", {})
        plan = state["plan"]
        idx = state["current_step_index"]
        step = plan[idx]
        protein_ctx = state["protein_context"]
        history = list(state.get("history", []))
        log_entries = list(state.get("conversation_log", []))
        executions = list(state.get("tool_executions", []))
        step_results = dict(state.get("step_results", {}))
        ui_lang = state.get("ui_lang") or _detect_ui_lang(state["messages"][-1].content)
        disabled_tool_names = set(chains.get("disabled_tool_names") or [])

        step_num = _normalize_step_number(step.get("step"), idx + 1)
        task_desc = step["task_description"]
        tool_name = step["tool_name"]
        tool_input = step["tool_input"]

        tool = next((t for t in chains.get("all_tools", []) if t.name == tool_name), None)
        agent_session_dir = state.get("agent_session_dir") or ""

        merged_tool_input = _merge_tool_parameters_with_context(protein_ctx, tool_input)

        return cls(
            state=state,
            config=config,
            chains=chains,
            plan=plan,
            idx=idx,
            step=step,
            step_num=step_num,
            task_desc=task_desc,
            tool_name=tool_name,
            tool_input=tool_input,
            tool=tool,
            protein_ctx=protein_ctx,
            agent_session_dir=agent_session_dir,
            ui_lang=ui_lang,
            history=history,
            log_entries=log_entries,
            executions=executions,
            step_results=step_results,
            disabled_tool_names=disabled_tool_names,
            merged_tool_input=merged_tool_input,
            invoke_input=dict(merged_tool_input),
        )


@dataclass
class ExecutionResult:
    """The outcome of executing one plan step (post retry orchestration)."""

    success: bool
    raw_output: Any = None
    failure_reason: str = ""
    failure_type: FailureType = "unknown"
    cached: bool = False
    attempts_by_reason: dict[str, int] = field(default_factory=dict)
    total_latency_ms: int = 0
