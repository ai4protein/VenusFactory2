"""Tests for the direct Anthropic /v1/messages adapter (ChatAnthropicLLM)
and the make_llm factory dispatch."""
from __future__ import annotations

from typing import Literal

import pytest
from langchain_classic.schema import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agent.chat_anthropic import ChatAnthropicLLM, _tools_to_anthropic_schema


# ---------------------------------------------------------------------------
# _build_payload
# ---------------------------------------------------------------------------


def test_build_payload_separates_system():
    llm = ChatAnthropicLLM(api_key="k", base_url="https://x", model_name="claude-3-7-sonnet-20250219")
    payload = llm._build_payload([
        SystemMessage(content="You are X"),
        HumanMessage(content="hi"),
    ])
    assert "system" in payload
    # Anthropic's messages array must NEVER contain a system entry.
    assert all(m["role"] != "system" for m in payload["messages"])
    assert payload["system"] == "You are X"


def test_build_payload_long_system_gets_cache_control():
    llm = ChatAnthropicLLM(api_key="k", base_url="https://x", model_name="claude-3-7-sonnet-20250219")
    long_sys = "x " * 600  # >1024 chars
    payload = llm._build_payload([
        SystemMessage(content=long_sys),
        HumanMessage(content="hi"),
    ])
    assert isinstance(payload["system"], list)
    assert payload["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert payload["system"][0]["type"] == "text"


def test_build_payload_coalesces_multiple_system_messages():
    llm = ChatAnthropicLLM(api_key="k", base_url="https://x", model_name="claude-3-7-sonnet-20250219")
    payload = llm._build_payload([
        SystemMessage(content="A"),
        SystemMessage(content="B"),
        HumanMessage(content="hi"),
    ])
    assert payload["system"] == "A\n\nB"


def test_build_payload_tool_result_becomes_user_turn():
    from langchain_core.messages import ToolMessage

    llm = ChatAnthropicLLM(api_key="k", base_url="https://x", model_name="claude-3-7-sonnet-20250219")
    payload = llm._build_payload([
        HumanMessage(content="hi"),
        AIMessage(content="", additional_kwargs={"tool_calls": [{"id": "t1", "name": "search", "args": {"q": "X"}}]}),
        ToolMessage(content="result-text", tool_call_id="t1"),
    ])
    # Assistant turn should have a tool_use content block.
    assistant_msg = next(m for m in payload["messages"] if m["role"] == "assistant")
    assert any(b.get("type") == "tool_use" and b["id"] == "t1" for b in assistant_msg["content"])
    # Tool result is conveyed as a user turn with a tool_result block.
    user_turns = [m for m in payload["messages"] if m["role"] == "user"]
    tool_result_turn = next(t for t in user_turns if isinstance(t["content"], list))
    assert tool_result_turn["content"][0]["type"] == "tool_result"
    assert tool_result_turn["content"][0]["tool_use_id"] == "t1"
    assert tool_result_turn["content"][0]["content"] == "result-text"


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------


def test_parse_response_extracts_tool_use():
    llm = ChatAnthropicLLM(api_key="k", base_url="https://x", model_name="claude-3-7-sonnet-20250219")
    fake = {
        "content": [
            {"type": "text", "text": "Let me check"},
            {"type": "tool_use", "id": "toolu_01", "name": "search", "input": {"query": "X"}},
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    msg = llm._parse_response(fake)
    assert msg.content == "Let me check"
    expected = [{"id": "toolu_01", "name": "search", "args": {"query": "X"}}]
    # Tool calls land either on .tool_calls or in additional_kwargs depending on langchain version.
    tool_calls = getattr(msg, "tool_calls", None) or msg.additional_kwargs.get("tool_calls")
    assert tool_calls == expected


def test_parse_response_text_only():
    llm = ChatAnthropicLLM(api_key="k", base_url="https://x", model_name="claude-3-7-sonnet-20250219")
    fake = {"content": [{"type": "text", "text": "hello"}], "usage": {"input_tokens": 3, "output_tokens": 1}}
    msg = llm._parse_response(fake)
    assert msg.content == "hello"
    # No tool_use blocks → tool_calls in additional_kwargs is unset.
    assert "tool_calls" not in msg.additional_kwargs


# ---------------------------------------------------------------------------
# _tools_to_anthropic_schema
# ---------------------------------------------------------------------------


class _FakeSchema(BaseModel):
    uniprot_id: str = Field(description="UniProt ID")
    format: Literal["pdb", "cif"] = Field(default="pdb")


class _FakeTool:
    name = "download_alphafold"
    description = "Download structure"
    args_schema = _FakeSchema


def test_tools_to_anthropic_schema_uses_input_schema_key():
    out = _tools_to_anthropic_schema([_FakeTool()])
    assert out[0]["name"] == "download_alphafold"
    assert out[0]["description"] == "Download structure"
    # Anthropic uses 'input_schema', NOT OpenAI's 'parameters'.
    assert "input_schema" in out[0]
    assert "parameters" not in out[0]
    assert out[0]["input_schema"]["properties"]["format"]["enum"] == ["pdb", "cif"]
    # title should have been pruned recursively.
    assert "title" not in out[0]["input_schema"]


def test_tools_to_anthropic_schema_handles_no_args_schema():
    class _BareTool:
        name = "noop"
        description = ""
        args_schema = None

    out = _tools_to_anthropic_schema([_BareTool()])
    assert out[0]["input_schema"] == {"type": "object", "properties": {}, "required": []}


# ---------------------------------------------------------------------------
# make_llm factory
# ---------------------------------------------------------------------------


def test_make_llm_picks_anthropic_for_claude():
    from agent.chat_agent import make_llm

    llm = make_llm("claude-3-7-sonnet-20250219", api_key="dummy")
    assert isinstance(llm, ChatAnthropicLLM)
    assert llm.model_name == "claude-3-7-sonnet-20250219"


def test_make_llm_picks_openai_for_others():
    from agent.chat_agent import Chat_LLM, make_llm

    llm = make_llm("gpt-4o", api_key="dummy")
    assert isinstance(llm, Chat_LLM)


def test_make_llm_claude_stays_anthropic_without_gateway(monkeypatch):
    """Claude always uses the official Anthropic Messages adapter; third-party
    OpenAI-compat gateways (e.g. former DMX) are no longer supported."""
    from agent.chat_agent import make_llm
    from agent.chat_anthropic import ChatAnthropicLLM
    from agent.model_registry import set_active_gateway

    monkeypatch.setenv("CHAT_FORCE_GATEWAY", "dmx")  # removed gateway → ignored
    try:
        llm = make_llm("claude-3-7-sonnet-20250219", api_key="dummy")
        assert isinstance(llm, ChatAnthropicLLM)
        assert getattr(llm, "base_url", "").startswith("https://api.anthropic.com")
        try:
            set_active_gateway("dmx")
            assert False, "expected ValueError for unknown gateway"
        except ValueError:
            pass
    finally:
        monkeypatch.delenv("CHAT_FORCE_GATEWAY", raising=False)
        set_active_gateway(None)


# ---------------------------------------------------------------------------
# Sanity: BaseChatModel interface contract
# ---------------------------------------------------------------------------


def test_anthropic_llm_type_is_anthropic_messages():
    llm = ChatAnthropicLLM(api_key="k", base_url="https://x", model_name="claude-3-7-sonnet-20250219")
    assert llm._llm_type == "anthropic-messages"


def test_bind_tools_returns_copy_with_tools():
    llm = ChatAnthropicLLM(api_key="k", base_url="https://x", model_name="claude-3-7-sonnet-20250219")
    bound = llm.bind_tools([_FakeTool()])
    assert bound is not llm
    assert bound._bound_tools is not None
    assert llm._bound_tools is None  # original unaffected


def test_generate_raises_without_api_key():
    llm = ChatAnthropicLLM(api_key="", base_url="https://x", model_name="claude-3-7-sonnet-20250219")
    with pytest.raises(ValueError, match="Anthropic API key"):
        llm._generate([HumanMessage(content="hi")])
