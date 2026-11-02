from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from agent.graph import finalize


@pytest.mark.asyncio
async def test_evidence_report_does_not_trigger_manuscript_expansion(monkeypatch):
    captured = {}

    async def stream(chain, inputs, **kwargs):
        captured.update(inputs)
        return "Candidate source submitted; independent acceptance is pending."

    async def forbidden(*args, **kwargs):
        raise AssertionError("short evidence report must not trigger manuscript retry")

    monkeypatch.setattr(finalize, "_stream_chain", stream)
    monkeypatch.setattr(finalize, "_maybe_retry_truncated_summary", forbidden)
    state = {
        "messages": [HumanMessage(content="build a model")],
        "history": [],
        "tool_executions": [],
        "protein_context": SimpleNamespace(get_context_summary=lambda: "task data only"),
        "ui_lang": "en",
    }
    result = await finalize.finalize_node(
        state, {"configurable": {"chains": {"report_policy": "evidence_summary", "finalizer": object()}}}
    )
    assert result["finalizer_succeeded"] is True
    assert "at most 350 words" in captured["full_run_record"]
    assert "5000" not in captured["full_run_record"]


@pytest.mark.asyncio
async def test_deterministic_fallback_is_not_counted_as_critic_model_completion(monkeypatch):
    async def failed(*args, **kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(finalize, "_stream_chain", failed)
    state = {
        "messages": [HumanMessage(content="build a model")],
        "history": [],
        "tool_executions": [],
        "protein_context": SimpleNamespace(get_context_summary=lambda: "task data only"),
        "ui_lang": "en",
    }
    result = await finalize.finalize_node(
        state, {"configurable": {"chains": {"report_policy": "evidence_summary", "finalizer": object()}}}
    )
    assert result["finalizer_succeeded"] is False
