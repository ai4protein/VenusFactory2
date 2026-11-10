"""Public mutation adapters expose scientific inputs, not obsolete host options."""
import json

import pytest
from pydantic import ValidationError

from agent.harness.session_boundary import mcp_input_schema, prepare_tool_arguments
from mcp_tools.common.tool_arguments import ToolArgumentError
from mcp_tools.models.pretrained.mutation import tools_agent, tools_api, tools_mcp


@pytest.mark.parametrize("schema,inputs", [
    (tools_agent.ZeroShotSequenceInput, {"sequence": "ACDE"}),
    (tools_agent.ZeroShotStructureInput, {"structure_file": "input.pdb"}),
    (tools_api.ZeroShotSequenceBody, {"sequence": "ACDE"}),
    (tools_api.ZeroShotStructureBody, {"structure_file": "input.pdb"}),
])
@pytest.mark.parametrize("field,value", [("api_key", "fixture-key"), ("backend", "pjlab")])
def test_public_contract_refuses_obsolete_host_options(schema, inputs, field, value):
    assert field not in schema.model_json_schema()["properties"]
    with pytest.raises(ValidationError):
        schema.model_validate({**inputs, field: value})
    assert schema.model_validate(inputs).model_dump()[next(iter(inputs))] == next(iter(inputs.values()))


@pytest.mark.parametrize("kind,inputs", [
    ("sequence", {"sequence": "ACDE", "model_name": "ESM-1v"}),
    ("structure", {"structure_file": "input.pdb", "model_name": "ProSST-2048"}),
])
def test_agent_execution_keeps_model_and_session_output_binding(tmp_path, monkeypatch, kind, inputs):
    (tmp_path / "input.pdb").write_text("END\n")
    calls = []
    def predict(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": "success"}
    name = "zero_shot_mutation_" + kind + "_prediction"
    monkeypatch.setattr(tools_agent, name, predict)
    tool = getattr(tools_agent, name + "_tool")
    schema = mcp_input_schema(tool)
    assert set(schema["properties"]) == set(inputs) | {"out_dir"} | ({"fasta_file"} if kind == "sequence" else set())
    for field, value in [("api_key", "fixture-key"), ("backend", "pjlab")]:
        with pytest.raises(ToolArgumentError):
            prepare_tool_arguments(name, tool, {**inputs, field: value}, tmp_path)
    assert calls == []
    bound = prepare_tool_arguments(name, tool, inputs, tmp_path)
    assert json.loads(tool.invoke(bound))["status"] == "success"
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert kwargs["model_name"] == inputs["model_name"]
    assert "api_key" not in kwargs and "backend" not in kwargs
    from pathlib import Path
    assert Path(kwargs["out_dir"]).is_relative_to(tmp_path)
    if kind == "structure":
        assert args == (str(tmp_path / "input.pdb"),)
    else:
        assert kwargs["sequence"] == "ACDE"


@pytest.mark.asyncio
async def test_native_mcp_exposes_and_calls_only_scientific_arguments(monkeypatch):
    from fastmcp import Client
    calls = []
    def predict(**kwargs):
        calls.append(kwargs)
        return {"status": "success"}
    for kind in ("sequence", "structure"):
        monkeypatch.setattr(tools_mcp, "zero_shot_mutation_" + kind + "_prediction", predict)
    async with Client(tools_mcp.mcp) as client:
        listed = {t.name: t.inputSchema for t in await client.list_tools()}
        for kind, inputs in [("sequence", {"sequence": "ACDE", "model_name": "ESM-1v"}),
                             ("structure", {"structure_file": "input.pdb", "model_name": "ProSST-2048"})]:
            name = "zero_shot_mutation_" + kind + "_prediction"
            assert not {"api_key", "backend"} & listed[name]["properties"].keys()
            await client.call_tool(name, inputs)
            assert calls[-1]["model_name"] == inputs["model_name"]
            assert not {"api_key", "backend"} & calls[-1].keys()
    assert len(calls) == 2
