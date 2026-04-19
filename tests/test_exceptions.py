"""Tests for the unified exception hierarchy."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from exceptions import (
    ErrorContext,
    VenusFactoryError,
    MaxTurnsExceeded,
    MaxToolCallsExceeded,
    ToolExecutionError,
    ToolTimeoutError,
    ToolValidationError,
    ModelError,
    ModelResponseParseError,
    ConfigurationError,
    SessionError,
    SessionNotFoundError,
    SessionExpiredError,
    RateLimitError,
    GuardrailTripped,
    InputGuardrailTripped,
    OutputGuardrailTripped,
    PipelineError,
)


class TestErrorContext:
    def test_defaults(self):
        ctx = ErrorContext()
        assert ctx.session_id is None
        assert ctx.tool_name is None
        assert ctx.extra == {}

    def test_with_values(self):
        ctx = ErrorContext(session_id="s1", tool_name="blast", step_index=3)
        assert ctx.session_id == "s1"
        assert ctx.tool_name == "blast"
        assert ctx.step_index == 3


class TestVenusFactoryError:
    def test_base_message(self):
        err = VenusFactoryError("something broke")
        assert str(err) == "something broke"
        assert isinstance(err.context, ErrorContext)

    def test_with_context(self):
        ctx = ErrorContext(session_id="abc")
        err = VenusFactoryError("fail", context=ctx)
        assert err.context.session_id == "abc"


class TestInheritanceHierarchy:
    def test_max_turns_is_venus_error(self):
        err = MaxTurnsExceeded(max_turns=50)
        assert isinstance(err, VenusFactoryError)
        assert err.max_turns == 50

    def test_tool_timeout_chain(self):
        err = ToolTimeoutError(tool_name="blast", timeout_seconds=300)
        assert isinstance(err, ToolExecutionError)
        assert isinstance(err, VenusFactoryError)
        assert err.timeout_seconds == 300
        assert err.tool_name == "blast"

    def test_model_parse_error_chain(self):
        err = ModelResponseParseError("bad json", model_name="gpt-4")
        assert isinstance(err, ModelError)
        assert isinstance(err, VenusFactoryError)
        assert err.model_name == "gpt-4"

    def test_session_not_found(self):
        err = SessionNotFoundError("no such session")
        assert isinstance(err, SessionError)
        assert isinstance(err, VenusFactoryError)

    def test_rate_limit(self):
        err = RateLimitError(retry_after=60.0)
        assert isinstance(err, VenusFactoryError)
        assert err.retry_after == 60.0

    def test_guardrail_subtypes(self):
        assert issubclass(InputGuardrailTripped, GuardrailTripped)
        assert issubclass(OutputGuardrailTripped, GuardrailTripped)

    def test_pipeline_error(self):
        err = PipelineError("step blew up", step_index=2, step_name="blast")
        assert isinstance(err, VenusFactoryError)
        assert err.step_index == 2
        assert err.step_name == "blast"


class TestCatchPatterns:
    def test_catch_all_venus(self):
        errors = [
            MaxTurnsExceeded(),
            ToolExecutionError("fail"),
            ModelError("fail"),
            ConfigurationError("fail"),
            SessionNotFoundError("fail"),
            RateLimitError(),
            PipelineError("fail"),
        ]
        for err in errors:
            try:
                raise err
            except VenusFactoryError:
                pass

    def test_catch_tool_errors(self):
        errors = [
            ToolExecutionError("fail"),
            ToolTimeoutError(),
            ToolValidationError("bad input"),
        ]
        for err in errors:
            try:
                raise err
            except ToolExecutionError:
                pass
