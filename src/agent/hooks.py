"""
Lifecycle hooks for VenusFactory agent execution.

Two levels of hooks (following OpenAI Agents SDK):
  - ``RunHooks``: global callbacks across an entire agent run.
  - ``AgentHooks``: per-agent callbacks scoped to a single agent instance.

All methods default to no-ops so users only override what they need.
Both sync and async callbacks are supported via MaybeAwaitable pattern.
"""
from __future__ import annotations

import inspect
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Any, TypeVar, Union

T = TypeVar("T")
MaybeAwaitable = Union[Awaitable[T], T]


async def _resolve(value: MaybeAwaitable[T]) -> T:
    if inspect.isawaitable(value):
        return await value  # type: ignore[return-value]
    return value  # type: ignore[return-value]


@dataclass
class ToolCallInfo:
    tool_name: str = ""
    tool_input: dict[str, Any] = field(default_factory=dict)
    step_index: int | None = None


@dataclass
class ToolResultInfo:
    tool_name: str = ""
    tool_input: dict[str, Any] = field(default_factory=dict)
    raw_output: Any = None
    success: bool = True
    error_message: str = ""
    duration_seconds: float = 0.0
    step_index: int | None = None


class RunHooks:
    """Global lifecycle callbacks for an entire agent run.

    Override any method to hook into the agent execution lifecycle.
    All methods are no-ops by default.
    """

    def on_run_start(self, *, session_id: str, user_message: str, **kwargs: Any) -> MaybeAwaitable[None]:
        pass

    def on_run_end(
        self,
        *,
        session_id: str,
        success: bool,
        total_steps: int = 0,
        total_tool_calls: int = 0,
        **kwargs: Any,
    ) -> MaybeAwaitable[None]:
        pass

    def on_run_error(
        self, *, session_id: str, error: Exception, **kwargs: Any
    ) -> MaybeAwaitable[None]:
        pass

    def on_tool_start(self, *, info: ToolCallInfo, **kwargs: Any) -> MaybeAwaitable[None]:
        pass

    def on_tool_end(self, *, info: ToolResultInfo, **kwargs: Any) -> MaybeAwaitable[None]:
        pass

    def on_step_start(
        self, *, step_index: int, step_description: str = "", **kwargs: Any
    ) -> MaybeAwaitable[None]:
        pass

    def on_step_end(
        self,
        *,
        step_index: int,
        success: bool,
        error_message: str = "",
        **kwargs: Any,
    ) -> MaybeAwaitable[None]:
        pass

    def on_llm_call(
        self,
        *,
        model_name: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        **kwargs: Any,
    ) -> MaybeAwaitable[None]:
        pass


class AgentHooks:
    """Per-agent lifecycle callbacks.

    These are scoped to a single agent instance (e.g., PI, CB, MLS)
    and fire in addition to RunHooks.
    """

    def on_agent_start(self, *, agent_name: str, **kwargs: Any) -> MaybeAwaitable[None]:
        pass

    def on_agent_end(
        self, *, agent_name: str, success: bool, **kwargs: Any
    ) -> MaybeAwaitable[None]:
        pass


class CompositeRunHooks(RunHooks):
    """Chains multiple RunHooks instances – fires each in order."""

    def __init__(self, hooks: list[RunHooks] | None = None):
        self._hooks: list[RunHooks] = list(hooks or [])

    def add(self, hook: RunHooks) -> None:
        self._hooks.append(hook)

    async def on_run_start(self, **kwargs: Any) -> None:
        for h in self._hooks:
            try:
                await _resolve(h.on_run_start(**kwargs))
            except Exception:
                pass

    async def on_run_end(self, **kwargs: Any) -> None:
        for h in self._hooks:
            try:
                await _resolve(h.on_run_end(**kwargs))
            except Exception:
                pass

    async def on_run_error(self, **kwargs: Any) -> None:
        for h in self._hooks:
            try:
                await _resolve(h.on_run_error(**kwargs))
            except Exception:
                pass

    async def on_tool_start(self, **kwargs: Any) -> None:
        for h in self._hooks:
            try:
                await _resolve(h.on_tool_start(**kwargs))
            except Exception:
                pass

    async def on_tool_end(self, **kwargs: Any) -> None:
        for h in self._hooks:
            try:
                await _resolve(h.on_tool_end(**kwargs))
            except Exception:
                pass

    async def on_step_start(self, **kwargs: Any) -> None:
        for h in self._hooks:
            try:
                await _resolve(h.on_step_start(**kwargs))
            except Exception:
                pass

    async def on_step_end(self, **kwargs: Any) -> None:
        for h in self._hooks:
            try:
                await _resolve(h.on_step_end(**kwargs))
            except Exception:
                pass

    async def on_llm_call(self, **kwargs: Any) -> None:
        for h in self._hooks:
            try:
                await _resolve(h.on_llm_call(**kwargs))
            except Exception:
                pass


class LoggingRunHooks(RunHooks):
    """Example hook that logs lifecycle events to the venus logger."""

    def __init__(self) -> None:
        from logger import get_logger
        self._logger = get_logger("hooks")

    def on_run_start(self, *, session_id: str, user_message: str, **kwargs: Any) -> None:
        self._logger.info("Run started: session=%s", session_id)

    def on_run_end(self, *, session_id: str, success: bool, **kwargs: Any) -> None:
        status = "success" if success else "failed"
        self._logger.info("Run ended: session=%s status=%s", session_id, status)

    def on_run_error(self, *, session_id: str, error: Exception, **kwargs: Any) -> None:
        self._logger.error("Run error: session=%s error=%s", session_id, error)

    def on_tool_start(self, *, info: ToolCallInfo, **kwargs: Any) -> None:
        self._logger.info("Tool start: %s (step=%s)", info.tool_name, info.step_index)

    def on_tool_end(self, *, info: ToolResultInfo, **kwargs: Any) -> None:
        status = "ok" if info.success else f"failed: {info.error_message[:100]}"
        self._logger.info(
            "Tool end: %s (%.1fs) %s", info.tool_name, info.duration_seconds, status
        )
