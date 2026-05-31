"""Typed state schema for the LangGraph agent graph."""
from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.messages import BaseMessage

from agent.chat_agent import ProteinContextManager


class AgentState(TypedDict):
    # Core state
    messages: list[BaseMessage]
    protein_context: ProteinContextManager
    session_id: str
    agent_session_dir: str

    # Process management
    pi_report: str
    pi_suggest_steps: str
    plan: list[dict[str, Any]]
    current_step_index: int
    step_results: dict[int, Any]

    # UI compatibility (for yielding partial updates)
    history: list[dict[str, Any]]
    conversation_log: list[dict[str, Any]]
    dialogue_memory: list[dict[str, Any]]
    tool_executions: list[dict[str, Any]]
    tool_cache: dict[str, Any]

    # Internal research state
    research_sections: list[dict[str, Any]]
    research_idx: int
    search_idx: int
    current_search_results: list[str]
    research_sub_reports: list[str]

    # Clarification state
    clarification_questions: list[dict[str, Any]]
    clarification_answers: list[dict[str, Any]]

    # Interactive flow control
    waiting_for: str | None

    # Control flags
    status: str
    error: str | None
    execution_failed: bool
    failed_step: int | None
    failed_reason: str | None
    ui_lang: str | None
    sub_report_rewrite_comment: str
    auto_execute: bool
    skipped_steps: list[int]
