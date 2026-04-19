"""Tests for model provider abstraction."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from agent.models import (
    Model,
    ModelProvider,
    ModelResponse,
    ModelRetryAdvice,
    ModelTracing,
    ModelUsage,
    MultiProvider,
    OpenAICompatibleModel,
    OpenAICompatibleModelConfig,
    OpenAICompatibleProvider,
)


class TestModelUsage:
    def test_defaults(self):
        u = ModelUsage()
        assert u.prompt_tokens == 0
        assert u.total_tokens == 0

    def test_with_values(self):
        u = ModelUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        assert u.total_tokens == 150


class TestModelResponse:
    def test_defaults(self):
        r = ModelResponse()
        assert r.content == ""
        assert r.tool_calls == []
        assert r.usage is None

    def test_with_data(self):
        r = ModelResponse(
            content="hello",
            tool_calls=[{"id": "1", "name": "test", "args": {}}],
            usage=ModelUsage(prompt_tokens=10),
            model_name="gpt-4",
        )
        assert r.content == "hello"
        assert len(r.tool_calls) == 1
        assert r.usage.prompt_tokens == 10


class TestModelRetryAdvice:
    def test_defaults(self):
        a = ModelRetryAdvice()
        assert not a.should_retry
        assert a.is_safe_to_replay

    def test_retry(self):
        a = ModelRetryAdvice(should_retry=True, retry_after_seconds=5.0)
        assert a.should_retry
        assert a.retry_after_seconds == 5.0


class TestModelTracing:
    def test_values(self):
        assert ModelTracing.DISABLED.value == "disabled"
        assert ModelTracing.ENABLED_WITHOUT_DATA.value == "enabled_without_data"


class TestModelProtocol:
    def test_protocol_check(self):
        class _FakeModel:
            @property
            def model_name(self) -> str:
                return "fake"

            async def get_response(self, messages, **kwargs):
                return ModelResponse(content="fake")

            def get_retry_advice(self, error):
                return None

            async def close(self):
                pass

        m = _FakeModel()
        assert isinstance(m, Model)


class TestOpenAICompatibleModel:
    def test_model_name(self):
        cfg = OpenAICompatibleModelConfig(default_model="gpt-4o")
        model = OpenAICompatibleModel(cfg)
        assert model.model_name == "gpt-4o"

    def test_retry_advice_429(self):
        from exceptions import ModelError

        cfg = OpenAICompatibleModelConfig()
        model = OpenAICompatibleModel(cfg)
        advice = model.get_retry_advice(ModelError("rate limit", status_code=429))
        assert advice is not None
        assert advice.should_retry

    def test_retry_advice_connection_error(self):
        cfg = OpenAICompatibleModelConfig()
        model = OpenAICompatibleModel(cfg)
        advice = model.get_retry_advice(ConnectionError("refused"))
        assert advice is not None
        assert advice.should_retry

    def test_retry_advice_unknown_error(self):
        cfg = OpenAICompatibleModelConfig()
        model = OpenAICompatibleModel(cfg)
        advice = model.get_retry_advice(ValueError("bad"))
        assert advice is None


class TestOpenAICompatibleProvider:
    def test_get_default_model(self):
        cfg = OpenAICompatibleModelConfig(default_model="gpt-4o")
        provider = OpenAICompatibleProvider(cfg)
        model = provider.get_model()
        assert model.model_name == "gpt-4o"

    def test_get_named_model(self):
        cfg = OpenAICompatibleModelConfig(default_model="gpt-4o")
        provider = OpenAICompatibleProvider(cfg)
        model = provider.get_model("claude-3-sonnet")
        assert model.model_name == "claude-3-sonnet"


class TestMultiProvider:
    def test_prefix_routing(self):
        cfg1 = OpenAICompatibleModelConfig(default_model="gpt-4o", base_url="https://api.openai.com/v1")
        cfg2 = OpenAICompatibleModelConfig(default_model="gemini-pro", base_url="https://gemini.api/v1")

        multi = MultiProvider()
        multi.register("openai/", OpenAICompatibleProvider(cfg1))
        multi.register("gemini/", OpenAICompatibleProvider(cfg2))

        model1 = multi.get_model("openai/gpt-4o")
        assert model1.model_name == "gpt-4o"

        model2 = multi.get_model("gemini/gemini-2.0-flash")
        assert model2.model_name == "gemini-2.0-flash"

    def test_default_fallback(self):
        cfg = OpenAICompatibleModelConfig(default_model="default-model")
        multi = MultiProvider()
        multi.set_default(OpenAICompatibleProvider(cfg))
        model = multi.get_model("unknown-model")
        assert model.model_name == "unknown-model"

    def test_no_provider_raises(self):
        multi = MultiProvider()
        with pytest.raises(ValueError, match="No provider"):
            multi.get_model("test")
