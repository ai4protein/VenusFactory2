"""Exercise disclosed model contracts and structured planning at the provider boundary."""
import json
import pytest

from agent.harness.autonomous_research import parse_plan, method_changes
from agent.harness.scientist_role_output import specification, request_fields, output_content
from agent.harness.scientist_gateway_policy import ScientistGatewayPolicy, ScientistResponsesPolicy
from agent.harness.research_workspace import ResearchWorkspace
from agent.harness.research_data_memory import prepared_dataset_context


def response(spec, body):
    call = {"name": spec["name"], "arguments": json.dumps(body)}
    if spec["protocol"] == "responses":
        return {"model": "fixture", "status": "completed", "output": [
            {"type": "function_call", "status": "completed", **call}]}
    return {"model": "fixture", "choices": [{"finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "plan", "type": "function", "function": call}]}}]}


def plan(approach):
    return {"summary": "Develop a measured predictor", "deliverable": "model", "needs_input": [],
            "steps": [{"id": "fit", "title": "Fit and evaluate", "acceptance": "Evaluate frozen groups",
                       "verification": "model", "approach": approach}]}


@pytest.mark.parametrize("protocol", ["chat_completions", "responses"])
def test_adaptive_plan_crosses_provider_schema_and_preserves_host_criteria(protocol):
    spec = specification("CB", protocol)
    policy = (ScientistResponsesPolicy(model="fixture", scientific_tools=[]) if protocol == "responses"
              else ScientistGatewayPolicy(model="fixture", mode="science_agent", scientific_tools=[]))
    payload = {"model": "fixture", **request_fields(spec)}
    payload["input" if protocol == "responses" else "messages"] = []
    with policy.role_output(spec):
        sent, _ = policy.apply(payload)
    assert sent["tools"] == [spec["tool"]]
    old = parse_plan(json.dumps(plan("Training-only preprocessing and a linear baseline")))
    raw = output_content(response(spec, plan("Revise features from validation residuals")), spec, "fixture")
    current = parse_plan(raw, old, objective="Develop a model")
    assert current["steps"][0]["status"] == "pending"
    assert current["steps"][0]["acceptance"] == old["steps"][0]["acceptance"]
    changes = method_changes(old, current, 2)
    assert len(changes) == 1 and changes[0]["approach"] == "Revise features from validation residuals"


@pytest.mark.parametrize("protocol", ["chat_completions", "responses"])
@pytest.mark.parametrize("bad", [None, {}, [], 0, "", "x" * 2001])
def test_provider_schema_rejects_invalid_method_without_execution(protocol, bad):
    spec = specification("CB", protocol)
    with pytest.raises(ValueError):
        output_content(response(spec, plan(bad)), spec, "fixture")


@pytest.mark.asyncio
async def test_tool_preparation_and_resumed_context_share_shape_without_values(tmp_path):
    episodes = tmp_path / "episodes"
    service = ResearchWorkspace(tmp_path / "work", episodes / "cycle_000" / "tools", {"steps": []},
                                development_root=tmp_path / "development")
    text = "id,group,sequence,condition,target\n" + "".join(
        f"row{i},group{i},PRIVATESEQ{i},37,{i / 17}\n" for i in range(40))
    await service.write_research_file("measurements.csv", text)
    prepared = await service.prepare_research_dataset(
        "measurements.csv", "target", ["sequence", "condition"], "group",
        "Synthetic interface fixture", "Independent fixture groups", id_column="id")
    restored = prepared_dataset_context(service.workspace, episodes)
    assert restored["model_interface"] == prepared["model_interface"]
    assert restored["training_profile"] == prepared["training_profile"]
    contract = prepared["model_interface"]
    shape = contract["shape_example"]
    assert shape["training_row"]["inputs"] == shape["prediction_inputs"]
    assert "inputs" not in shape["prediction_inputs"]
    assert "placeholder" in shape["kind"]
    assert contract["input_fields"] == {"sequence": "original CSV string", "condition": "original CSV string"}
    assert "ONE" in contract["predict"]["inputs"]
    assert contract["predict"]["returns"] == {"value": "finite number"}
    encoded = json.dumps(contract)
    assert "PRIVATESEQ" not in encoded and "row0" not in encoded and str(tmp_path) not in encoded
    tool = next(t for t in service.tools() if t.name == "train_research_model")
    assert "not a batch" in tool.description and "numeric feature vectors" in tool.description
    assert 'X=[r["inputs"]' not in tool.description
