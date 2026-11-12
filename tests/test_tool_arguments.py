import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator

from mcp_tools.common.tool_arguments import (
    ToolArgumentError,
    validate_argument_fields,
    validate_tool_arguments,
)


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
        validate_tool_arguments(
            tool, {"samples": {"private-key-marker": "private-value-marker"}, "query": "/home/private/credentials"}
        )
    message = str(error.value)
    assert "samples" in message and "query" in message
    assert "private" not in message and "/home" not in message


@pytest.mark.parametrize("relative", [False, True])
def test_session_binding_precedes_required_output_validation(tmp_path, relative):
    from agent.harness.session_boundary import prepare_tool_arguments

    def download(record_id: str, out_path: str):
        """Download a record."""
        return out_path

    tool = StructuredTool.from_function(
        download, name="download_record", metadata={"workspace_relative_paths": relative}
    )
    if relative:
        with pytest.raises(ToolArgumentError, match="out_path"):
            prepare_tool_arguments(tool.name, tool, {"record_id": "P00918"}, tmp_path)
    else:
        args = prepare_tool_arguments(tool.name, tool, {"record_id": "P00918"}, tmp_path)
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
    result = await mcp_service.operation(
        scheduler.submit,
        "predict_fixture",
        {"entry_id": "fixture", "inputs": [{"private": "private-marker"}]},
        owner="fixture",
    )
    assert result["error"]["code"] == 422
    assert "scope" in result["error"]["message"]
    assert "private-marker" not in json.dumps(result)
    assert not list((tmp_path / "queue" / "mcp-inputs").glob("*.json"))
    assert not scheduler.queue.list("fixture")


def test_root_error_reports_only_declared_safe_alternative_names():
    from pydantic import ConfigDict, model_validator

    class Inputs(BaseModel):
        model_config = ConfigDict(
            json_schema_extra={
                "anyOf": [
                    {"required": ["left"]},
                    {"required": ["right"]},
                    {"required": ["private-undeclared-marker"]},
                ]
            }
        )
        left: str | None = None
        right: str | None = None

        @model_validator(mode="after")
        def reject(self):
            raise ValueError("private-validator-marker")

    with pytest.raises(ToolArgumentError) as error:
        validate_tool_arguments(SimpleNamespace(get_input_schema=lambda: Inputs), {})
    assert "left or right" in str(error.value)
    assert "private" not in str(error.value)


class WindowOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fields: list[str]
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class WindowInput(BaseModel):
    query: str
    options: WindowOptions | None = None


@pytest.mark.parametrize(
    "bad_options",
    [
        {"fields": ["private-value"], "private_valid_identifier": "private-value"},
        {"fields": ["private-value"], "view": "private-value"},
        {"fields": ["private-value"], "limit": -1},
        {"offset": 4},
    ],
)
def test_nested_contract_feedback_names_only_declared_object_fields(bad_options):
    tool = SimpleNamespace(get_input_schema=lambda: WindowInput)
    payload = {"query": "private-query", "options": bad_options}
    original = json.dumps(payload)
    with pytest.raises(ToolArgumentError) as error:
        validate_tool_arguments(tool, payload)
    message = str(error.value)
    assert "Invalid arguments: options." in message
    assert "Declared object fields for options: fields, limit, offset." in message
    assert "Missing required arguments: options" not in message
    assert "view" not in message and "private" not in message
    assert json.dumps(payload) == original


def test_nested_feedback_correction_preserves_payload_without_inserting_defaults():
    tool = SimpleNamespace(get_input_schema=lambda: WindowInput)
    payload = {"query": "fixture", "options": {"fields": ["score"]}}
    original = json.dumps(payload)
    validate_tool_arguments(tool, payload)
    assert json.dumps(payload) == original


@pytest.mark.parametrize("payload", [None, ["private-input"], "private-input", 1])
def test_closed_surface_requires_an_object_without_echoing_input(payload):
    with pytest.raises(ToolArgumentError, match="Use a JSON object") as error:
        validate_argument_fields(WindowInput.model_json_schema(), payload)
    assert "private" not in str(error.value)


def test_closed_surface_never_echoes_undeclared_names_even_when_valid_identifiers():
    contract = WindowInput.model_json_schema()
    payload = {"query": "private-query", "private_token_identifier": "private-value", 42: "private-value"}
    original = dict(payload)
    with pytest.raises(ToolArgumentError) as error:
        validate_argument_fields(contract, payload)
    assert "Allowed fields: options, query." in str(error.value)
    assert "private" not in str(error.value) and "42" not in str(error.value)
    assert payload == original
    validate_argument_fields(contract, {"query": "fixture"})


def test_feedback_is_bounded_for_wide_nested_models():
    fields = {"field_" + str(i) + "_" * 48: (int, 1) for i in range(80)}
    options = create_model("WideOptions", __config__=ConfigDict(extra="forbid"), **fields)
    model = create_model("WideInput", **{"option_" + str(i): (options, ...) for i in range(8)})
    tool = SimpleNamespace(get_input_schema=lambda: model)
    payload = {"option_" + str(i): {"private_key": "private-value"} for i in range(8)}
    with pytest.raises(ToolArgumentError) as error:
        validate_tool_arguments(tool, payload)
    assert 0 < len(str(error.value)) <= 1500
    assert "Declared object fields for option_0 include:" in str(error.value)
    assert "private" not in str(error.value)


def test_recursive_schema_feedback_does_not_expand_child_models():
    class RecursiveOptions(BaseModel):
        model_config = ConfigDict(extra="forbid")
        child: "RecursiveOptions | None" = None
        limit: int = 1

    model = create_model("RecursiveInput", options=(RecursiveOptions | None, None))
    tool = SimpleNamespace(get_input_schema=lambda: model)
    with pytest.raises(ToolArgumentError) as error:
        validate_tool_arguments(tool, {"options": {"private_key": "private-value"}})
    assert "Declared object fields for options: child, limit." in str(error.value)
    assert "private" not in str(error.value)


@pytest.mark.asyncio
async def test_nested_argument_error_is_rejected_before_queue_admission(tmp_path, monkeypatch):
    from backend.web_v2.job_queue import JobQueue
    from mcp_tools.scheduler import jobs, mcp_service

    tool = SimpleNamespace(get_input_schema=lambda: WindowInput)
    monkeypatch.setattr(jobs, "native_tools", lambda: {"read_records": tool})
    monkeypatch.setattr(jobs, "assert_tool_call_allowed", lambda *args: None)

    def must_not_launch(*args, **kwargs):
        raise AssertionError("Invalid nested arguments reached compute admission")

    monkeypatch.setattr(jobs, "launch", must_not_launch)
    scheduler = jobs.ToolJobs(JobQueue(tmp_path / "queue"))
    result = await mcp_service.operation(
        scheduler.submit,
        "read_records",
        {
            "query": "private-query",
            "options": {"fields": ["score"], "private_key": 1},
        },
        owner="fixture",
    )
    assert result["error"]["code"] == 422
    assert "Declared object fields for options: fields, limit, offset." in result["error"]["message"]
    assert "private" not in json.dumps(result)
    assert not scheduler.queue.list("fixture")
    assert not list((tmp_path / "queue" / "mcp-inputs").glob("*.json"))


def test_direct_tool_wrapper_feedback_does_not_unwrap_or_echo_payload():
    contract = WindowInput.model_json_schema()
    payload = {"arguments": {"query": "private-query", "files": ["private-file"]}}
    original = json.dumps(payload)
    with pytest.raises(ToolArgumentError, match="Remove the outer arguments object") as error:
        validate_argument_fields(contract, payload)
    assert "call_tool" in str(error.value) and "Allowed fields: options, query" in str(error.value)
    assert "private" not in str(error.value) and "files" not in str(error.value)
    assert json.dumps(payload) == original
    # A tool that genuinely declares arguments keeps that contract.
    validate_argument_fields({"properties": {"arguments": {"type": "object"}}}, payload)
