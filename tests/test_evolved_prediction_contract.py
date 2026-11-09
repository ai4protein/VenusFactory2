"""The selected tool contract must agree across discovery, queues and MCP."""
import copy
import json

import pytest
from fastmcp import Client
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from agent.harness.web_native_worker import mcp_input_schema
from mcp_tools.common.tool_arguments import ToolArgumentError, validate_tool_arguments
from mcp_tools.models.evolved.contract import PredictionArguments
from mcp_tools.models.evolved.tools_agent import predict_with_evolved_model_tool as prediction
from mcp_tools.models.evolved.tools_mcp import mcp


BASE = {"entry_id": "a" * 64, "inputs": [{"id": "sample", "values": [0, 1.0]}]}
REF = {"scope_ref": "b" * 64}


@pytest.mark.asyncio
async def test_all_public_schemas_keep_conditional_requirements_and_no_host_fields():
    async with Client(mcp) as client:
        exposed = next(t for t in await client.list_tools() if t.name == prediction.name)
    expected = PredictionArguments.model_json_schema()
    for schema in (exposed.inputSchema, prediction.tool_call_schema.model_json_schema(), mcp_input_schema(prediction)):
        assert schema == expected
        Draft202012Validator.check_schema(schema)
        assert set(schema["properties"]) == {"entry_id", "inputs", "scope_ref", "scope"}


@pytest.mark.parametrize("binding", [REF, {"scope": {}}, {**REF, "scope": {}}])
def test_contract_preserves_both_supported_call_forms_and_input_representation(binding):
    payload = copy.deepcopy({**BASE, **binding})
    original = json.dumps(payload)
    assert Draft202012Validator(mcp_input_schema(prediction)).is_valid(payload)
    validate_tool_arguments(prediction, payload)
    assert json.dumps(payload) == original


@pytest.mark.parametrize("change", [
    {}, {"scope_ref": None}, {"scope": None}, {"scope_ref": None, "scope": None},
    {"scope_ref": "not-a-reference"}, {"scope_ref": 123}, {"scope": "not-an-object"},
    {**REF, "inputs": []}, {**REF, "inputs": [{}] * 257},
    {**REF, "session_id": "private-session-marker"},
    {**REF, "entry_id": "not-an-entry"}, {**REF, "entry_id": None},
])
def test_disclosed_schema_and_prequeue_validation_reject_the_same_invalid_calls(change):
    payload = copy.deepcopy({**BASE, **change})
    assert not Draft202012Validator(mcp_input_schema(prediction)).is_valid(payload)
    with pytest.raises(ValidationError):
        PredictionArguments.model_validate(payload)
    with pytest.raises(ToolArgumentError):
        validate_tool_arguments(prediction, payload)


def test_missing_binding_feedback_names_alternatives_without_echoing_data():
    payload = {**BASE, "inputs": [{"value": "private-record-marker"}]}
    with pytest.raises(ToolArgumentError) as error:
        validate_tool_arguments(prediction, payload)
    message = str(error.value)
    assert "scope_ref or scope" in message
    assert "private-record-marker" not in message


@pytest.mark.asyncio
async def test_mcp_missing_binding_returns_the_same_safe_contract_error(monkeypatch):
    from mcp_tools.models.evolved import service
    from mcp_tools.tools import runtime_tool_policy

    monkeypatch.setattr(service, "assert_tool_allowed", lambda *a: None)
    monkeypatch.setattr(runtime_tool_policy, "assert_tool_call_allowed", lambda *a: None)
    monkeypatch.setattr(service, "configured_library", lambda: object())
    async with Client(mcp) as client:
        result = (await client.call_tool(prediction.name, BASE)).data
    assert not result["success"] and result["error_code"] == "invalid_model_arguments"
    assert "scope_ref or scope" in result["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", [BASE, REF, {**BASE, "scope_ref": "private-invalid-marker"}])
async def test_invalid_prediction_never_queues_or_consumes_a_worker(tmp_path, monkeypatch, arguments):
    from backend.web_v2.job_queue import JobQueue
    from mcp_tools.scheduler import jobs, mcp_service

    monkeypatch.setattr(jobs, "native_tools", lambda: {prediction.name: prediction})
    monkeypatch.setattr(jobs, "assert_tool_call_allowed", lambda *a: None)
    monkeypatch.setattr(jobs, "launch", lambda *a, **k: pytest.fail("Invalid prediction was launched"))
    scheduler = jobs.ToolJobs(JobQueue(tmp_path / "queue"))
    result = await mcp_service.operation(scheduler.submit, prediction.name, arguments, owner="fixture")
    assert result["error"]["code"] == 422
    assert "private-invalid-marker" not in json.dumps(result)
    assert not scheduler.queue.list("fixture")
    assert not list((tmp_path / "queue" / "mcp-inputs").glob("*.json"))
