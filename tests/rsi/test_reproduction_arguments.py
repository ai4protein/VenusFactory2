import time
from types import SimpleNamespace

import pytest

from agent.graph.execution.prepare import materialize_deferred_input
from agent.graph.planning.plan import _filter_unparameterized_steps
from agent.rsi.langgraph_tools import build_tools


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,fields,provided,expected",
    [
        (
            "submit_reproduction",
            ["plugin_id", "parameters_json", "idempotency_key"],
            {"plugin_id": "phl7.hrex_validation.v1", "parameters": {"backend": "cuda"}, "idempotency_key": "stable"},
            {
                "plugin_id": "phl7.hrex_validation.v1",
                "parameters_json": '{"backend": "cuda"}',
                "idempotency_key": "stable",
            },
        ),
        ("inspect_reproduction_job", ["job_id"], {"job_id": {"job_id": "case_actual"}}, {"job_id": "case_actual"}),
        ("submit_evidence_review", ["claims_json"], {"claims": {"replicas": 8}}, {"claims_json": '{"replicas": 8}'}),
    ],
)
async def test_actual_reproduction_tools_use_schema_before_execution(tmp_path, name, fields, provided, expected):
    class Client:
        async def list_tools(self):
            return [
                SimpleNamespace(
                    name=name,
                    description="Bound scientific operation",
                    inputSchema={
                        "type": "object",
                        "properties": {k: {"type": "string"} for k in fields},
                        "required": fields,
                    },
                )
            ]

        async def call_tool(self, *args, **kwargs):
            raise AssertionError("No job should run during argument construction")

    tools = await build_tools(
        Client(),
        allowed_tools={name},
        workspace=tmp_path,
        capability=tmp_path,
        deadline=time.monotonic() + 30,
        max_tool_calls=1,
        result={"tool_calls": 0, "tool_errors": 0},
        event=lambda value: None,
        validate_results=False,
    )
    tool = tools[0]
    assert tool.metadata["deferred_arguments"] is True
    assert callable(tool.metadata["result_validator"])
    step = {"step": 1, "tool_name": name, "tool_input": provided}
    kept, _ = _filter_unparameterized_steps([step], tools, "en", allow_deferred_arguments=True)
    assert kept == [step]
    called = []

    async def fill(context):
        called.append(context.merged_tool_input)
        return expected

    ctx = SimpleNamespace(tool=tool, merged_tool_input=provided, chains={"materialize_tool_arguments": fill})
    assert await materialize_deferred_input(ctx) == expected
    assert called == [provided]
    # A second validation of complete inputs does not consume another model call.
    ctx.merged_tool_input = expected
    assert await materialize_deferred_input(ctx) == expected
    assert len(called) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name", ["discover_reproduction_plugins", "inspect_registered_reproduction", "inspect_case_evidence"]
)
async def test_no_argument_reads_do_not_request_argument_generation(tmp_path, name):
    class Client:
        async def list_tools(self):
            return [
                SimpleNamespace(
                    name=name, description="Read pinned data", inputSchema={"type": "object", "properties": {}}
                )
            ]

    tools = await build_tools(
        Client(),
        allowed_tools={name},
        workspace=tmp_path,
        capability=tmp_path,
        deadline=time.monotonic() + 30,
        max_tool_calls=1,
        result={"tool_calls": 0, "tool_errors": 0},
        event=lambda value: None,
        validate_results=False,
    )
    assert tools[0].metadata["deferred_arguments"] is False
