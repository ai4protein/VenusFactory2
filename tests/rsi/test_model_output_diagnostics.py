"""Model completion metadata identifies truncation without recording payloads."""

import json
import time
from types import SimpleNamespace

from agent.rsi.scientist_runtime import ModelAudit


def test_output_shape_reports_truncation_and_invalid_arguments_without_content(tmp_path):
    path = tmp_path / "model_calls.jsonl"
    audit = ModelAudit(path, time.monotonic() + 60)
    message = SimpleNamespace(
        content="PRIVATE_PAYLOAD",
        tool_calls=[{"args": "SECRET_ARGUMENT"}],
        invalid_tool_calls=[{"error": "SECRET_ERROR"}],
        response_metadata={"finish_reason": "length"},
    )
    generation = SimpleNamespace(message=message, generation_info={})
    response = SimpleNamespace(llm_output={"token_usage": {"completion_tokens": 12288}}, generations=[[generation]])
    audit.on_llm_end(response, run_id="test")
    raw = path.read_text()
    record = json.loads(raw)
    assert record["generations"] == [
        {"finish_reason": "length", "content_characters": 15, "valid_tool_calls": 1, "invalid_tool_calls": 1}
    ]
    assert "PRIVATE_PAYLOAD" not in raw and "SECRET" not in raw
    assert audit.usage == [{"completion_tokens": 12288}]


def test_unknown_provider_finish_text_is_not_logged(tmp_path):
    path = tmp_path / "model_calls.jsonl"
    response = SimpleNamespace(
        llm_output=None,
        generations=[[SimpleNamespace(message=None, generation_info={"finish_reason": "PRIVATE_PROVIDER_TEXT"})]],
    )
    ModelAudit(path, time.monotonic() + 60).on_llm_end(response)
    raw = path.read_text()
    assert json.loads(raw)["generations"][0]["finish_reason"] == "unknown"
    assert "PRIVATE_PROVIDER_TEXT" not in raw
