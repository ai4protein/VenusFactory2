"""
Structured tracing system for VenusFactory agent execution.

Inspired by OpenAI Agents SDK's tracing architecture:
  - ``Trace`` / ``Span`` hierarchy with typed span data
  - ``NoOpTrace`` / ``NoOpSpan`` — Null Object pattern eliminates branching
  - ``Scope`` — ``contextvars``-based async-safe context propagation
  - ``TracingProcessor`` — observer interface for pluggable exporters
"""
from __future__ import annotations

import contextvars
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from logger import get_logger

_logger = get_logger("tracing")

TSpanData = TypeVar("TSpanData", bound="SpanData")


# ---------------------------------------------------------------------------
# Span data types
# ---------------------------------------------------------------------------

class SpanData(ABC):
    """Base class for typed span payloads."""

    @abstractmethod
    def export(self) -> dict[str, Any]:
        ...


@dataclass
class AgentSpanData(SpanData):
    agent_name: str = ""
    phase: str = ""

    def export(self) -> dict[str, Any]:
        return {"type": "agent", "agent_name": self.agent_name, "phase": self.phase}


@dataclass
class ToolSpanData(SpanData):
    tool_name: str = ""
    tool_input: dict[str, Any] = field(default_factory=dict)
    tool_output: Any = None
    success: bool = True
    error_message: str = ""

    def export(self) -> dict[str, Any]:
        return {
            "type": "tool",
            "tool_name": self.tool_name,
            "success": self.success,
            "error_message": self.error_message,
        }


@dataclass
class LLMSpanData(SpanData):
    model_name: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def export(self) -> dict[str, Any]:
        return {
            "type": "llm",
            "model_name": self.model_name,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class StepSpanData(SpanData):
    step_index: int = 0
    step_description: str = ""
    success: bool = True

    def export(self) -> dict[str, Any]:
        return {
            "type": "step",
            "step_index": self.step_index,
            "step_description": self.step_description,
            "success": self.success,
        }


@dataclass
class GuardrailSpanData(SpanData):
    guardrail_name: str = ""
    triggered: bool = False

    def export(self) -> dict[str, Any]:
        return {
            "type": "guardrail",
            "guardrail_name": self.guardrail_name,
            "triggered": self.triggered,
        }


# ---------------------------------------------------------------------------
# Processor interface
# ---------------------------------------------------------------------------

class TracingProcessor(ABC):
    """Observer for trace/span lifecycle events."""

    @abstractmethod
    def on_trace_start(self, trace: Trace) -> None: ...

    @abstractmethod
    def on_trace_end(self, trace: Trace) -> None: ...

    @abstractmethod
    def on_span_start(self, span: Span[Any]) -> None: ...

    @abstractmethod
    def on_span_end(self, span: Span[Any]) -> None: ...

    def shutdown(self) -> None:
        pass

    def force_flush(self) -> None:
        pass


class ConsoleTracingProcessor(TracingProcessor):
    """Prints trace/span events to the logger."""

    def on_trace_start(self, trace: Trace) -> None:
        _logger.info("[trace:start] %s session=%s", trace.trace_id, trace.session_id)

    def on_trace_end(self, trace: Trace) -> None:
        _logger.info("[trace:end] %s duration=%.2fs", trace.trace_id, trace.duration)

    def on_span_start(self, span: Span[Any]) -> None:
        _logger.info("[span:start] %s %s", span.span_id, type(span.data).__name__)

    def on_span_end(self, span: Span[Any]) -> None:
        _logger.info("[span:end] %s duration=%.3fs", span.span_id, span.duration)


# ---------------------------------------------------------------------------
# Trace / Span abstractions
# ---------------------------------------------------------------------------

class Trace(ABC):
    """An end-to-end agent execution trace."""

    @property
    @abstractmethod
    def trace_id(self) -> str: ...

    @property
    @abstractmethod
    def session_id(self) -> str: ...

    @property
    @abstractmethod
    def duration(self) -> float: ...

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def end(self, error: Exception | None = None) -> None: ...

    @abstractmethod
    def create_span(self, name: str, data: TSpanData) -> Span[TSpanData]: ...

    def __enter__(self) -> Trace:
        self.start()
        Scope.set_current_trace(self)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.end(error=exc_val if isinstance(exc_val, Exception) else None)
        Scope.reset_current_trace()


class Span(ABC, Generic[TSpanData]):
    """A single operation within a trace."""

    @property
    @abstractmethod
    def span_id(self) -> str: ...

    @property
    @abstractmethod
    def data(self) -> TSpanData: ...

    @property
    @abstractmethod
    def duration(self) -> float: ...

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def end(self, error: Exception | None = None) -> None: ...

    def __enter__(self) -> Span[TSpanData]:
        self.start()
        Scope.set_current_span(self)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.end(error=exc_val if isinstance(exc_val, Exception) else None)
        Scope.reset_current_span()


# ---------------------------------------------------------------------------
# Real implementations
# ---------------------------------------------------------------------------

class TraceImpl(Trace):
    def __init__(
        self,
        session_id: str = "",
        metadata: dict[str, Any] | None = None,
        processors: list[TracingProcessor] | None = None,
    ):
        self._trace_id = str(uuid.uuid4())
        self._session_id = session_id
        self._metadata = metadata or {}
        self._processors = processors or []
        self._start_time: float = 0
        self._end_time: float = 0
        self._error: Exception | None = None

    @property
    def trace_id(self) -> str:
        return self._trace_id

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def duration(self) -> float:
        if self._end_time:
            return self._end_time - self._start_time
        if self._start_time:
            return time.monotonic() - self._start_time
        return 0.0

    def start(self) -> None:
        self._start_time = time.monotonic()
        for p in self._processors:
            try:
                p.on_trace_start(self)
            except Exception as e:
                _logger.error("Tracing processor error on trace start: %s", e)

    def end(self, error: Exception | None = None) -> None:
        self._end_time = time.monotonic()
        self._error = error
        for p in self._processors:
            try:
                p.on_trace_end(self)
            except Exception as e:
                _logger.error("Tracing processor error on trace end: %s", e)

    def create_span(self, name: str, data: TSpanData) -> Span[TSpanData]:
        return SpanImpl(
            name=name,
            data=data,
            trace_id=self._trace_id,
            processors=self._processors,
        )


class SpanImpl(Span[TSpanData]):
    def __init__(
        self,
        name: str,
        data: TSpanData,
        trace_id: str = "",
        processors: list[TracingProcessor] | None = None,
    ):
        self._span_id = str(uuid.uuid4())
        self._name = name
        self._data = data
        self._trace_id = trace_id
        self._processors = processors or []
        self._start_time: float = 0
        self._end_time: float = 0
        self._error: Exception | None = None

    @property
    def span_id(self) -> str:
        return self._span_id

    @property
    def data(self) -> TSpanData:
        return self._data

    @property
    def duration(self) -> float:
        if self._end_time:
            return self._end_time - self._start_time
        if self._start_time:
            return time.monotonic() - self._start_time
        return 0.0

    def start(self) -> None:
        self._start_time = time.monotonic()
        for p in self._processors:
            try:
                p.on_span_start(self)
            except Exception as e:
                _logger.error("Tracing processor error on span start: %s", e)

    def end(self, error: Exception | None = None) -> None:
        self._end_time = time.monotonic()
        self._error = error
        for p in self._processors:
            try:
                p.on_span_end(self)
            except Exception as e:
                _logger.error("Tracing processor error on span end: %s", e)


# ---------------------------------------------------------------------------
# NoOp implementations — Null Object pattern
# ---------------------------------------------------------------------------

class NoOpTrace(Trace):
    """No-op trace that does nothing. Eliminates branching on 'is tracing enabled'."""

    @property
    def trace_id(self) -> str:
        return ""

    @property
    def session_id(self) -> str:
        return ""

    @property
    def duration(self) -> float:
        return 0.0

    def start(self) -> None:
        pass

    def end(self, error: Exception | None = None) -> None:
        pass

    def create_span(self, name: str, data: TSpanData) -> Span[TSpanData]:
        return NoOpSpan(data=data)


class NoOpSpan(Span[TSpanData]):
    """No-op span that does nothing."""

    def __init__(self, data: TSpanData | None = None):
        self._data = data  # type: ignore[assignment]

    @property
    def span_id(self) -> str:
        return ""

    @property
    def data(self) -> TSpanData:
        return self._data  # type: ignore[return-value]

    @property
    def duration(self) -> float:
        return 0.0

    def start(self) -> None:
        pass

    def end(self, error: Exception | None = None) -> None:
        pass


# ---------------------------------------------------------------------------
# Scope — contextvars-based async-safe context propagation
# ---------------------------------------------------------------------------

_current_trace: contextvars.ContextVar[Trace | None] = contextvars.ContextVar(
    "current_trace", default=None
)
_current_span: contextvars.ContextVar[Span[Any] | None] = contextvars.ContextVar(
    "current_span", default=None
)


class Scope:
    """Static accessor for the current trace/span in an async-safe manner.

    Uses ``contextvars`` so that concurrent coroutines each see their own trace.
    """

    @staticmethod
    def get_current_trace() -> Trace:
        return _current_trace.get() or NoOpTrace()

    @staticmethod
    def set_current_trace(trace: Trace) -> None:
        _current_trace.set(trace)

    @staticmethod
    def reset_current_trace() -> None:
        _current_trace.set(None)

    @staticmethod
    def get_current_span() -> Span[Any]:
        return _current_span.get() or NoOpSpan()

    @staticmethod
    def set_current_span(span: Span[Any]) -> None:
        _current_span.set(span)

    @staticmethod
    def reset_current_span() -> None:
        _current_span.set(None)


# ---------------------------------------------------------------------------
# Tracing provider — global registry
# ---------------------------------------------------------------------------

class TracingProvider:
    """Registry for tracing processors and factory for traces."""

    def __init__(self) -> None:
        self._processors: list[TracingProcessor] = []
        self._enabled: bool = True

    def add_processor(self, processor: TracingProcessor) -> None:
        self._processors.append(processor)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def create_trace(self, session_id: str = "", metadata: dict[str, Any] | None = None) -> Trace:
        if not self._enabled or not self._processors:
            return NoOpTrace()
        return TraceImpl(
            session_id=session_id,
            metadata=metadata,
            processors=list(self._processors),
        )

    def shutdown(self) -> None:
        for p in self._processors:
            try:
                p.shutdown()
            except Exception as e:
                _logger.error("Error shutting down tracing processor: %s", e)


_provider = TracingProvider()


def get_tracing_provider() -> TracingProvider:
    return _provider


def set_tracing_provider(provider: TracingProvider) -> None:
    global _provider
    _provider = provider


def create_trace(session_id: str = "", **kwargs: Any) -> Trace:
    """Convenience: create a trace from the global provider."""
    return _provider.create_trace(session_id=session_id, metadata=kwargs)
