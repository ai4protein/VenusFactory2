"""
Unified exception hierarchy for VenusFactory.

Inspired by OpenAI Agents SDK: every exception carries diagnostic context
so callers can inspect run state without re-fetching it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ErrorContext:
    session_id: str | None = None
    tool_name: str | None = None
    step_index: int | None = None
    input_data: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class VenusFactoryError(Exception):
    """Base exception for all VenusFactory errors."""

    def __init__(self, message: str, *, context: ErrorContext | None = None):
        super().__init__(message)
        self.context = context or ErrorContext()


# ---------------------------------------------------------------------------
# Agent / orchestration errors
# ---------------------------------------------------------------------------

class MaxTurnsExceeded(VenusFactoryError):
    """Agent exceeded the maximum number of turns."""

    def __init__(
        self,
        message: str = "Maximum turns exceeded",
        *,
        max_turns: int = 0,
        context: ErrorContext | None = None,
    ):
        super().__init__(message, context=context)
        self.max_turns = max_turns


class MaxToolCallsExceeded(VenusFactoryError):
    """Agent exceeded the maximum number of tool calls."""

    def __init__(
        self,
        message: str = "Maximum tool calls exceeded",
        *,
        max_tool_calls: int = 0,
        context: ErrorContext | None = None,
    ):
        super().__init__(message, context=context)
        self.max_tool_calls = max_tool_calls


# ---------------------------------------------------------------------------
# Tool errors
# ---------------------------------------------------------------------------

class ToolExecutionError(VenusFactoryError):
    """A tool invocation failed."""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str = "",
        raw_output: Any = None,
        context: ErrorContext | None = None,
    ):
        super().__init__(message, context=context)
        self.tool_name = tool_name
        self.raw_output = raw_output


class ToolTimeoutError(ToolExecutionError):
    """Tool invocation exceeded its timeout."""

    def __init__(
        self,
        message: str = "Tool execution timed out",
        *,
        tool_name: str = "",
        timeout_seconds: float = 0,
        context: ErrorContext | None = None,
    ):
        super().__init__(message, tool_name=tool_name, context=context)
        self.timeout_seconds = timeout_seconds


class ToolValidationError(ToolExecutionError):
    """Tool input failed validation."""


# ---------------------------------------------------------------------------
# Model / LLM errors
# ---------------------------------------------------------------------------

class ModelError(VenusFactoryError):
    """Error communicating with the LLM backend."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        model_name: str = "",
        context: ErrorContext | None = None,
    ):
        super().__init__(message, context=context)
        self.status_code = status_code
        self.model_name = model_name


class ModelResponseParseError(ModelError):
    """Failed to parse the model's response."""


# ---------------------------------------------------------------------------
# Configuration errors
# ---------------------------------------------------------------------------

class ConfigurationError(VenusFactoryError):
    """Invalid or missing configuration."""


# ---------------------------------------------------------------------------
# Session / auth errors
# ---------------------------------------------------------------------------

class SessionError(VenusFactoryError):
    """Session-related error (not found, expired, etc.)."""


class SessionNotFoundError(SessionError):
    """Requested session does not exist."""


class SessionExpiredError(SessionError):
    """Session token has expired."""


class RateLimitError(VenusFactoryError):
    """Rate limit exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        *,
        retry_after: float | None = None,
        context: ErrorContext | None = None,
    ):
        super().__init__(message, context=context)
        self.retry_after = retry_after


# ---------------------------------------------------------------------------
# Guardrail errors
# ---------------------------------------------------------------------------

class GuardrailTripped(VenusFactoryError):
    """An input or output guardrail was triggered."""

    def __init__(
        self,
        message: str,
        *,
        guardrail_name: str = "",
        context: ErrorContext | None = None,
    ):
        super().__init__(message, context=context)
        self.guardrail_name = guardrail_name


class InputGuardrailTripped(GuardrailTripped):
    """Input guardrail tripped."""


class OutputGuardrailTripped(GuardrailTripped):
    """Output guardrail tripped."""


# ---------------------------------------------------------------------------
# Pipeline errors
# ---------------------------------------------------------------------------

class PipelineError(VenusFactoryError):
    """Error in the execution pipeline (step failed, cascade, etc.)."""

    def __init__(
        self,
        message: str,
        *,
        step_index: int | None = None,
        step_name: str = "",
        context: ErrorContext | None = None,
    ):
        super().__init__(message, context=context)
        self.step_index = step_index
        self.step_name = step_name
