"""
Model provider abstraction layer for VenusFactory.

Provides a pluggable LLM backend system:
  - ``Model`` Protocol for structural typing
  - ``ModelProvider`` ABC for provider factories
  - ``ModelTracing`` for privacy-aware tracing control
  - Concrete ``OpenAICompatibleModel`` for the current Chat_LLM use case
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Tracing control
# ---------------------------------------------------------------------------

class ModelTracing(Enum):
    """Controls what gets traced during model calls."""

    DISABLED = "disabled"
    ENABLED = "enabled"
    ENABLED_WITHOUT_DATA = "enabled_without_data"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ModelResponse:
    """Structured LLM response."""

    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: ModelUsage | None = None
    model_name: str = ""
    raw: Any = None


@dataclass
class ModelUsage:
    """Token usage from a model call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @property
    def cost_estimate(self) -> float:
        return 0.0


@dataclass
class ModelRetryAdvice:
    """Provider-specific retry guidance."""

    should_retry: bool = False
    retry_after_seconds: float | None = None
    is_safe_to_replay: bool = True


# ---------------------------------------------------------------------------
# Model Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Model(Protocol):
    """Structural interface for LLM backends.

    Third-party implementations can satisfy this protocol without inheriting.
    """

    @property
    def model_name(self) -> str: ...

    async def get_response(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 8192,
        **kwargs: Any,
    ) -> ModelResponse: ...

    def get_retry_advice(self, error: Exception) -> ModelRetryAdvice | None:
        return None

    async def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Model Provider ABC
# ---------------------------------------------------------------------------

class ModelProvider(ABC):
    """Abstract factory for creating Model instances."""

    @abstractmethod
    def get_model(self, model_name: str | None = None) -> Model: ...

    async def aclose(self) -> None:
        pass


class MultiProvider(ModelProvider):
    """Composite provider that delegates to sub-providers by prefix.

    Example::

        multi = MultiProvider()
        multi.register("openai/", openai_provider)
        multi.register("gemini/", gemini_provider)
        model = multi.get_model("openai/gpt-4o")  # delegates to openai_provider
    """

    def __init__(self) -> None:
        self._providers: list[tuple[str, ModelProvider]] = []
        self._default: ModelProvider | None = None

    def register(self, prefix: str, provider: ModelProvider) -> None:
        self._providers.append((prefix, provider))

    def set_default(self, provider: ModelProvider) -> None:
        self._default = provider

    def get_model(self, model_name: str | None = None) -> Model:
        if model_name:
            for prefix, provider in self._providers:
                if model_name.startswith(prefix):
                    return provider.get_model(model_name[len(prefix):])
        if self._default:
            return self._default.get_model(model_name)
        raise ValueError(f"No provider found for model '{model_name}'")

    async def aclose(self) -> None:
        for _, provider in self._providers:
            await provider.aclose()
        if self._default:
            await self._default.aclose()


# ---------------------------------------------------------------------------
# OpenAI-compatible implementation
# ---------------------------------------------------------------------------

@dataclass
class OpenAICompatibleModelConfig:
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    default_model: str = "gpt-4o"
    temperature: float = 0.2
    max_tokens: int = 8192
    timeout_seconds: int = 120


class OpenAICompatibleModel:
    """Model implementation for OpenAI-compatible APIs.

    Works with OpenAI, Azure, DeepSeek, dmxapi, and other
    providers that follow the /chat/completions API spec.
    """

    def __init__(self, config: OpenAICompatibleModelConfig):
        self._config = config
        self._model_name = config.default_model

    @property
    def model_name(self) -> str:
        return self._model_name

    async def get_response(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        import aiohttp

        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "temperature": temperature if temperature is not None else self._config.temperature,
            "max_tokens": max_tokens or self._config.max_tokens,
        }
        if tools:
            payload["tools"] = tools

        url = f"{self._config.base_url}/chat/completions"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self._config.timeout_seconds),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    from exceptions import ModelError
                    raise ModelError(
                        f"API error {resp.status}: {text[:500]}",
                        status_code=resp.status,
                        model_name=self._model_name,
                    )

                data = await resp.json()

        choice = data["choices"][0]
        message = choice["message"]

        tool_calls_raw = message.get("tool_calls") or []
        tool_calls = []
        for tc in tool_calls_raw:
            import json
            tool_calls.append({
                "id": tc.get("id", ""),
                "name": tc.get("function", {}).get("name", ""),
                "args": json.loads(tc.get("function", {}).get("arguments", "{}") or "{}"),
            })

        usage_raw = data.get("usage", {})
        usage = ModelUsage(
            prompt_tokens=usage_raw.get("prompt_tokens", 0),
            completion_tokens=usage_raw.get("completion_tokens", 0),
            total_tokens=usage_raw.get("total_tokens", 0),
        )

        return ModelResponse(
            content=message.get("content", "") or "",
            tool_calls=tool_calls,
            usage=usage,
            model_name=self._model_name,
            raw=data,
        )

    def get_retry_advice(self, error: Exception) -> ModelRetryAdvice | None:
        from exceptions import ModelError
        if isinstance(error, ModelError) and error.status_code == 429:
            return ModelRetryAdvice(should_retry=True, retry_after_seconds=5.0)
        if isinstance(error, (ConnectionError, TimeoutError)):
            return ModelRetryAdvice(should_retry=True, retry_after_seconds=2.0)
        return None

    async def close(self) -> None:
        pass


class OpenAICompatibleProvider(ModelProvider):
    """Provider that creates OpenAICompatibleModel instances."""

    def __init__(self, config: OpenAICompatibleModelConfig):
        self._config = config

    def get_model(self, model_name: str | None = None) -> OpenAICompatibleModel:
        config = self._config
        if model_name:
            config = OpenAICompatibleModelConfig(
                api_key=config.api_key,
                base_url=config.base_url,
                default_model=model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout_seconds=config.timeout_seconds,
            )
        return OpenAICompatibleModel(config)
