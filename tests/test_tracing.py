"""Tests for the tracing system."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from agent.tracing import (
    AgentSpanData,
    ConsoleTracingProcessor,
    LLMSpanData,
    NoOpSpan,
    NoOpTrace,
    Scope,
    SpanImpl,
    StepSpanData,
    ToolSpanData,
    Trace,
    TraceImpl,
    TracingProcessor,
    TracingProvider,
    create_trace,
    get_tracing_provider,
)


class TestSpanData:
    def test_agent_span_data(self):
        d = AgentSpanData(agent_name="PI", phase="research")
        exported = d.export()
        assert exported["type"] == "agent"
        assert exported["agent_name"] == "PI"

    def test_tool_span_data(self):
        d = ToolSpanData(tool_name="blast", success=False, error_message="timeout")
        exported = d.export()
        assert exported["type"] == "tool"
        assert not exported["success"]

    def test_llm_span_data(self):
        d = LLMSpanData(model_name="gpt-4", prompt_tokens=100, completion_tokens=50, total_tokens=150)
        exported = d.export()
        assert exported["total_tokens"] == 150

    def test_step_span_data(self):
        d = StepSpanData(step_index=3, step_description="Run BLAST")
        exported = d.export()
        assert exported["step_index"] == 3


class TestNoOpTrace:
    def test_noop_properties(self):
        t = NoOpTrace()
        assert t.trace_id == ""
        assert t.session_id == ""
        assert t.duration == 0.0

    def test_noop_context_manager(self):
        t = NoOpTrace()
        with t:
            span = t.create_span("test", ToolSpanData())
            assert isinstance(span, NoOpSpan)

    def test_noop_span(self):
        s = NoOpSpan()
        assert s.span_id == ""
        assert s.duration == 0.0
        with s:
            pass


class _TrackingProcessor(TracingProcessor):
    def __init__(self):
        self.events = []

    def on_trace_start(self, trace):
        self.events.append(("trace_start", trace.trace_id))

    def on_trace_end(self, trace):
        self.events.append(("trace_end", trace.trace_id))

    def on_span_start(self, span):
        self.events.append(("span_start", span.span_id))

    def on_span_end(self, span):
        self.events.append(("span_end", span.span_id))


class TestTraceImpl:
    def test_basic_trace(self):
        p = _TrackingProcessor()
        t = TraceImpl(session_id="s1", processors=[p])
        with t:
            assert t.trace_id != ""
            assert t.session_id == "s1"
        assert t.duration > 0
        assert ("trace_start", t.trace_id) in p.events
        assert ("trace_end", t.trace_id) in p.events

    def test_span_lifecycle(self):
        p = _TrackingProcessor()
        t = TraceImpl(processors=[p])
        with t:
            span = t.create_span("tool_call", ToolSpanData(tool_name="blast"))
            with span:
                assert span.span_id != ""
                assert span.data.tool_name == "blast"
        assert len(p.events) == 4  # trace_start, span_start, span_end, trace_end

    def test_processor_error_does_not_propagate(self):
        class _BadProcessor(TracingProcessor):
            def on_trace_start(self, trace):
                raise RuntimeError("boom")
            def on_trace_end(self, trace):
                pass
            def on_span_start(self, span):
                pass
            def on_span_end(self, span):
                pass

        t = TraceImpl(processors=[_BadProcessor()])
        with t:
            pass


class TestScope:
    def test_default_noop(self):
        assert isinstance(Scope.get_current_trace(), NoOpTrace)
        assert isinstance(Scope.get_current_span(), NoOpSpan)

    def test_set_and_reset(self):
        t = TraceImpl(session_id="test")
        Scope.set_current_trace(t)
        assert Scope.get_current_trace() is t
        Scope.reset_current_trace()
        assert isinstance(Scope.get_current_trace(), NoOpTrace)


class TestTracingProvider:
    def test_creates_noop_when_disabled(self):
        provider = TracingProvider()
        provider.enabled = False
        trace = provider.create_trace()
        assert isinstance(trace, NoOpTrace)

    def test_creates_noop_when_no_processors(self):
        provider = TracingProvider()
        trace = provider.create_trace()
        assert isinstance(trace, NoOpTrace)

    def test_creates_real_trace_with_processor(self):
        provider = TracingProvider()
        provider.add_processor(_TrackingProcessor())
        trace = provider.create_trace(session_id="s1")
        assert isinstance(trace, TraceImpl)

    def test_convenience_function(self):
        p = get_tracing_provider()
        p.add_processor(_TrackingProcessor())
        trace = create_trace(session_id="s2")
        assert isinstance(trace, TraceImpl)
        p._processors.clear()
