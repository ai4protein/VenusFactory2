import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, field_validator

from mcp_tools.common.tool_arguments import ToolArgumentError, validate_tool_arguments


class PredictionInput(BaseModel):
    entry_id: str
    inputs: list[dict]
    scope: dict
    limit: int = Field(default=10, ge=1)


def prediction_tool():
    return SimpleNamespace(get_input_schema=lambda: PredictionInput)


def test_missing_scope_names_the_field_without_copying_input():
    payload = {"entry_id": "fixture", "inputs": [{"private": "private-input-marker"}]}
    original = json.dumps(payload)
    with pytest.raises(ToolArgumentError, match="Missing required arguments: scope") as error:
        validate_tool_arguments(prediction_tool(), payload)
    assert "private-input-marker" not in str(error.value)
    assert json.dumps(payload) == original


def test_validation_preserves_records_and_does_not_insert_defaults():
    payload = {"entry_id": "fixture", "inputs": [{"values": [0, 1.0]}], "scope": {}}
    original = json.dumps(payload)
    validate_tool_arguments(prediction_tool(), payload)
    assert json.dumps(payload) == original and "limit" not in payload


def test_nested_and_custom_validator_errors_do_not_echo_secrets():
    class Inputs(BaseModel):
        samples: dict[str, int]
        query: str

        @field_validator("query")
        @classmethod
        def reject(cls, value):
            raise ValueError("private-validator-marker " + value)

    tool = SimpleNamespace(get_input_schema=lambda: Inputs)
    with pytest.raises(ToolArgumentError) as error:
        validate_tool_arguments(tool, {"samples": {"private-key-marker": "private-value-marker"},
                                       "query": "/home/private/credentials"})
    message = str(error.value)
    assert "samples" in message and "query" in message
    assert "private" not in message and "/home" not in message


@pytest.mark.parametrize("relative", [False, True])
def test_session_binding_precedes_required_output_validation(tmp_path, relative):
    from agent.harness.web_native_worker import prepare_tool_arguments

    def download(record_id: str, out_path: str):
        """Download a record."""
        return out_path

    tool = StructuredTool.from_function(download, name="download_record",
                                       metadata={"workspace_relative_paths": relative})
    if relative:
        with pytest.raises(ToolArgumentError, match="out_path"):
            prepare_tool_arguments(tool.name, tool, {"record_id": "P00918"}, tmp_path, lambda x: x)
    else:
        args = prepare_tool_arguments(tool.name, tool, {"record_id": "P00918"}, tmp_path, lambda x: x)
        assert Path(args["out_path"]).is_relative_to(tmp_path)
        assert args["record_id"] == "P00918"


@pytest.mark.asyncio
async def test_scheduler_returns_safe_feedback_without_launching_a_job(tmp_path, monkeypatch):
    from backend.web_v2.job_queue import JobQueue
    from mcp_tools.scheduler import jobs, mcp_service

    monkeypatch.setattr(jobs, "native_tools", lambda: {"predict_fixture": prediction_tool()})
    monkeypatch.setattr(jobs, "assert_tool_call_allowed", lambda *a: None)
    def must_not_launch(*args, **kwargs):
        raise AssertionError("Invalid arguments reached compute admission")
    monkeypatch.setattr(jobs, "launch", must_not_launch)
    scheduler = jobs.ToolJobs(JobQueue(tmp_path / "queue"))
    result = await mcp_service.operation(scheduler.submit, "predict_fixture",
                                        {"entry_id": "fixture", "inputs": [{"private": "private-marker"}]},
                                        owner="fixture")
    assert result["error"]["code"] == 422
    assert "scope" in result["error"]["message"]
    assert "private-marker" not in json.dumps(result)
    assert not list((tmp_path / "queue" / "mcp-inputs").glob("*.json"))
    assert not scheduler.queue.list("fixture")


def test_root_error_reports_only_declared_safe_alternative_names():
    from pydantic import ConfigDict, model_validator

    class Inputs(BaseModel):
        model_config = ConfigDict(json_schema_extra={"anyOf": [
            {"required": ["left"]}, {"required": ["right"]},
            {"required": ["private-undeclared-marker"]},
        ]})
        left: str | None = None
        right: str | None = None

        @model_validator(mode="after")
        def reject(self):
            raise ValueError("private-validator-marker")

    with pytest.raises(ToolArgumentError) as error:
        validate_tool_arguments(SimpleNamespace(get_input_schema=lambda: Inputs), {})
    assert "left or right" in str(error.value)
    assert "private" not in str(error.value)
