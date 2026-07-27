"""Pydantic request/response models for chat_api endpoints.

Kept in a leaf module so any sub-router can import models without pulling in
the heavier ``_shared`` helpers (which depend on the SessionStore singleton).
"""
from typing import Any, Optional

from pydantic import BaseModel, Field


class CreateSessionResponse(BaseModel):
    session_id: str
    created_at: str
    model_name: str
    session_access_token: str = ""
    token_expires_at: str = ""


class SessionStateResponse(BaseModel):
    session_id: str
    model_name: str
    created_at: str
    history: list[dict[str, Any]]
    conversation_log: list[dict[str, Any]]
    tool_executions: list[dict[str, Any]]
    status: str = ""
    clarification_questions: list[dict[str, Any]] = Field(default_factory=list)
    plan: list[dict[str, Any]] = Field(default_factory=list)
    waiting_for: str = ""
    engine: str = ""
    chat_mode: str = ""
    kimi_pending_question: Optional[dict[str, Any]] = None
    kimi_pending_approval: Optional[dict[str, Any]] = None
    approval_prompt: str = ""
    plan_markdown: str = ""


class ChatStreamRequest(BaseModel):
    text: str = Field(default="")
    model: Optional[str] = Field(default=None)
    attachment_paths: list[str] = Field(default_factory=list)
    custom_model_config: dict[str, str] = Field(default_factory=dict)
    custom_model_id: str = Field(default="")
    # "graph" = legacy LangGraph PI/CB/MLS/SC pipeline (default for back-compat).
    # "kimi-code" = route the turn through the local kimi-code daemon.
    # The model-registry entry for the selected model may override this
    # (see messages.py: any model with engine="kimi-code" forces the kimi path).
    engine: Optional[str] = Field(default=None)
    # Dual-chat mode (preferred by new clients). Maps to engine:
    #   "science_agent"  → kimi-code
    #   "science_expert" → graph (LangGraph PI/CB/MLS/SC)
    # Registry engine=kimi-code still forces science_agent. Old clients that
    # only send model/engine remain supported when this is omitted.
    chat_mode: Optional[str] = Field(default=None)
    # UI locale ("en" | "zh"). When set, the chat router forces the model to
    # respond in this language regardless of the input language. Persisted on
    # the session so retries inherit it. None → fall back to model defaults.
    lang: Optional[str] = Field(default=None)


class ClarificationAnswer(BaseModel):
    question_index: int = 0
    selected_options: list[int] = Field(default_factory=list)
    custom_text: str = ""


class ClarificationResponseRequest(BaseModel):
    answers: list[ClarificationAnswer] = Field(default_factory=list)


class ApprovalDecideRequest(BaseModel):
    """Science Agent Approve/Reject decision (ExitPlanMode / dangerous tools)."""
    decision: str = Field(default="approved")  # approved | rejected
    selected_label: str = ""
    feedback: str = ""


class PlanConfirmRequest(BaseModel):
    plan: list[dict[str, Any]] = Field(default_factory=list)
    # Expert default: auto-run steps after plan confirmation (no per-step gate).
    auto_execute: bool = Field(default=True)


class IterationDecideRequest(BaseModel):
    action: str = Field(default="satisfied")


class StepDecideRequest(BaseModel):
    action: str = Field(default="continue")


class SubReportDecideRequest(BaseModel):
    action: str = Field(default="continue")
    comment: str = Field(default="")


class FeedbackRequest(BaseModel):
    message_index: int = Field(ge=0)
    rating: str = Field(pattern="^(like|dislike)$")
    comment: str = Field(default="")
