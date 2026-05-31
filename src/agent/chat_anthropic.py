"""Anthropic /v1/messages adapter — implements BaseChatModel interface
to integrate with existing LangChain chains transparently.

This is the direct-to-Anthropic counterpart of ``Chat_LLM`` (which only speaks
OpenAI-style /chat/completions). It is selected by ``chat_agent.make_llm``
whenever the resolved endpoint's ``api_compatible`` field is ``anthropic``.

Schema differences vs. OpenAI we have to translate:
  - System messages: separate top-level ``system`` field, NOT inside ``messages``.
  - Tool schema: ``{"name", "description", "input_schema": <JSON Schema>}``
    (not OpenAI's ``{"type":"function","function":{"name","parameters"}}``).
  - Tool calls: ``content`` array contains ``{"type":"tool_use","id","name","input"}``
    blocks (no top-level ``function_call`` / ``tool_calls`` field).
  - Tool results: a ``user`` turn whose content is
    ``[{"type":"tool_result","tool_use_id":...,"content":...}]``.
  - Usage fields: ``input_tokens`` / ``output_tokens`` / ``cache_*_input_tokens``.
  - Auth header: ``x-api-key`` + ``anthropic-version`` (no ``Authorization: Bearer``).
"""
from __future__ import annotations

from copy import copy
from typing import Any, Sequence

import aiohttp
from langchain_classic.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult

from agent.model_registry import get_default_model_id, resolve_endpoint
from logger import get_logger

_logger = get_logger("agent.chat_anthropic")

# Anthropic API version header value. Bump when adopting newer message-API features.
_ANTHROPIC_VERSION = "2023-06-01"

# Minimum system-prompt length (chars) before we attach cache_control: ephemeral.
# Anthropic's documented minimum cacheable size is ~1024 tokens; using chars as
# a cheap proxy keeps us safely above the floor for English/JSON-heavy prompts.
_CACHE_CONTROL_MIN_CHARS = 1024


def _tools_to_anthropic_schema(tools: Sequence) -> list[dict[str, Any]]:
    """Convert LangChain tools to Anthropic tool schema.

    Anthropic shape (note ``input_schema``, not ``parameters``):
        {"name": ..., "description": ..., "input_schema": <JSON Schema>}
    """
    # Imported here to avoid a circular import at module load time
    # (chat_agent imports this module via make_llm).
    from agent.chat_agent import _prune_titles

    out: list[dict[str, Any]] = []
    for t in tools:
        name = getattr(t, "name", "") or ""
        desc = getattr(t, "description", "") or ""
        schema_cls = getattr(t, "args_schema", None)
        if schema_cls is not None and hasattr(schema_cls, "model_json_schema"):
            input_schema = schema_cls.model_json_schema()
            _prune_titles(input_schema)
            input_schema.setdefault("type", "object")
            input_schema.setdefault("properties", {})
        else:
            input_schema = {"type": "object", "properties": {}, "required": []}
        out.append({"name": name, "description": desc, "input_schema": input_schema})
    return out


class ChatAnthropicLLM(BaseChatModel):
    """BaseChatModel implementation for Anthropic's /v1/messages endpoint.

    Mirrors the public surface of ``Chat_LLM`` (bind_tools, _generate, _agenerate,
    _llm_type) so existing LangChain chains and ``LangGraphAgentExecutorWrapper``
    can use either implementation interchangeably.
    """

    api_key: str = ""
    base_url: str = "https://api.anthropic.com/v1"
    model_name: str = "claude-3-7-sonnet-20250219"
    temperature: float = 0.2
    max_tokens: int = 8192
    _bound_tools: list | None = None

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        explicit_model = kwargs.get("model_name")
        explicit_base_url = kwargs.get("base_url")
        explicit_api_key = kwargs.get("api_key")

        model_name = explicit_model or get_default_model_id()

        # Honor a fully-specified (base_url + api_key) pair verbatim — this is
        # the "custom Anthropic-compat endpoint" path and must not be overridden
        # by registry resolution.
        if explicit_base_url and explicit_api_key:
            self.api_key = explicit_api_key
            self.base_url = explicit_base_url
            self.model_name = model_name
        else:
            resolved = resolve_endpoint(model_name, api_key=explicit_api_key or "")
            self.api_key = explicit_api_key or resolved.api_key
            self.base_url = explicit_base_url or resolved.base_url
            self.model_name = resolved.model_id or model_name

    # ------------------------------------------------------------------
    # Tool binding
    # ------------------------------------------------------------------

    def bind_tools(self, tools: Sequence, **kwargs: Any) -> "ChatAnthropicLLM":
        """Return a copy with tools bound (matches Chat_LLM.bind_tools semantics)."""
        obj = copy(self)
        obj._bound_tools = list(tools) if tools else None
        return obj

    # ------------------------------------------------------------------
    # Payload / response translation
    # ------------------------------------------------------------------

    def _build_payload(self, messages: list[BaseMessage]) -> dict[str, Any]:
        """Convert LangChain messages to an Anthropic /messages request payload."""
        system_chunks: list[str] = []
        msg_list: list[dict[str, Any]] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                if msg.content:
                    system_chunks.append(msg.content)
            elif isinstance(msg, HumanMessage):
                msg_list.append({"role": "user", "content": msg.content or ""})
            elif isinstance(msg, AIMessage):
                content_blocks: list[dict[str, Any]] = []
                text = msg.content or ""
                if text:
                    content_blocks.append({"type": "text", "text": text})
                # AIMessage.tool_calls is the canonical place; fall back to additional_kwargs.
                tool_calls = getattr(msg, "tool_calls", None)
                if not tool_calls and hasattr(msg, "additional_kwargs"):
                    tool_calls = msg.additional_kwargs.get("tool_calls") or []
                for tc in tool_calls or []:
                    if isinstance(tc, dict):
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tc.get("id", "") or "",
                            "name": tc.get("name", "") or "",
                            "input": tc.get("args", {}) or {},
                        })
                if content_blocks:
                    msg_list.append({"role": "assistant", "content": content_blocks})
                else:
                    # Empty assistant turn → send empty string to keep the API happy.
                    msg_list.append({"role": "assistant", "content": text})
            elif type(msg).__name__ == "ToolMessage":
                # Tool results are conveyed as a user turn with a tool_result block.
                msg_list.append({"role": "user", "content": [{
                    "type": "tool_result",
                    "tool_use_id": getattr(msg, "tool_call_id", "") or "",
                    "content": getattr(msg, "content", "") or "",
                }]})
            else:
                # Unknown message type — best-effort coercion to user text.
                msg_list.append({"role": "user", "content": str(getattr(msg, "content", msg))})

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": msg_list,
            "max_tokens": self.max_tokens,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature

        # Coalesce all SystemMessages into Anthropic's separate ``system`` field.
        system_text = "\n\n".join(s for s in system_chunks if s)
        if system_text:
            if len(system_text) > _CACHE_CONTROL_MIN_CHARS:
                payload["system"] = [{
                    "type": "text",
                    "text": system_text,
                    "cache_control": {"type": "ephemeral"},
                }]
            else:
                payload["system"] = system_text

        if self._bound_tools:
            payload["tools"] = _tools_to_anthropic_schema(self._bound_tools)

        return payload

    def _parse_response(self, result: dict[str, Any]) -> AIMessage:
        """Translate an Anthropic /messages response into an AIMessage."""
        blocks = result.get("content") or []
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in blocks:
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text") or "")
            elif btype == "tool_use":
                tool_calls.append({
                    "id": block.get("id", "") or "",
                    "name": block.get("name", "") or "",
                    "args": block.get("input", {}) or {},
                })
        content = "".join(text_parts)
        usage_raw = result.get("usage") or {}
        additional: dict[str, Any] = {"usage": usage_raw}
        if tool_calls:
            additional["tool_calls"] = tool_calls
        msg = AIMessage(content=content, additional_kwargs=additional)
        if tool_calls:
            try:
                msg.tool_calls = tool_calls
            except Exception:
                # tool_calls is a computed attribute on some langchain versions;
                # additional_kwargs already carries the data for downstream code.
                pass
        return msg

    def _build_headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }

    def _build_generation(self, ai_msg: AIMessage, usage_raw: dict[str, Any]) -> ChatGeneration:
        return ChatGeneration(
            message=ai_msg,
            generation_info={
                "prompt_tokens": usage_raw.get("input_tokens", 0) or 0,
                "completion_tokens": usage_raw.get("output_tokens", 0) or 0,
                "cache_read_tokens": usage_raw.get("cache_read_input_tokens", 0) or 0,
                "cache_creation_tokens": usage_raw.get("cache_creation_input_tokens", 0) or 0,
                "model": self.model_name,
            },
        )

    # ------------------------------------------------------------------
    # BaseChatModel interface
    # ------------------------------------------------------------------

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not self.api_key:
            raise ValueError("Anthropic API key is not configured.")
        import requests

        payload = self._build_payload(messages)
        # Allow caller-supplied kwargs to override safe payload fields. Never let
        # caller smuggle in identity/auth fields.
        safe_extra = {k: v for k, v in kwargs.items() if k not in ("model", "api_key", "anthropic-version")}
        payload.update(safe_extra)

        url = f"{self.base_url.rstrip('/')}/messages"
        response = requests.post(url, headers=self._build_headers(), json=payload, timeout=300)
        if response.status_code != 200:
            raise RuntimeError(f"Anthropic API error: {response.status_code} - {response.text}")
        result = response.json()

        ai_msg = self._parse_response(result)
        usage_raw = result.get("usage") or {}
        return ChatResult(
            generations=[self._build_generation(ai_msg, usage_raw)],
            llm_output={"token_usage": usage_raw, "model_name": self.model_name},
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not self.api_key:
            raise ValueError("Anthropic API key is not configured.")

        payload = self._build_payload(messages)
        safe_extra = {k: v for k, v in kwargs.items() if k not in ("model", "api_key", "anthropic-version")}
        payload.update(safe_extra)

        url = f"{self.base_url.rstrip('/')}/messages"
        timeout = aiohttp.ClientTimeout(total=300)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=self._build_headers(), json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Anthropic API error: {resp.status} - {text}")
                result = await resp.json()

        ai_msg = self._parse_response(result)
        usage_raw = result.get("usage") or {}
        return ChatResult(
            generations=[self._build_generation(ai_msg, usage_raw)],
            llm_output={"token_usage": usage_raw, "model_name": self.model_name},
        )

    @property
    def _llm_type(self) -> str:
        return "anthropic-messages"


__all__ = ["ChatAnthropicLLM"]
