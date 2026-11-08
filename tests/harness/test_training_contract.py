"""Reject accidental data-as-code and unfinished model bodies before worker allocation."""
import pytest

from agent.harness.research_training import validate_algorithm
from agent.harness.research_workspace import ResearchWorkspace


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["data.json", "measurements.csv", "inputs.txt"])
async def test_data_file_source_reports_the_contract_before_loading_data(tmp_path, source):
    service = ResearchWorkspace(tmp_path / "workspace", tmp_path / "control", {"steps": []},
                                development_root=tmp_path / "development")

    async def forbidden(*args, **kwargs):
        raise AssertionError("Invalid source must never allocate a training sandbox")

    service.runner = forbidden
    await service.write_research_file(source, "{}")
    with pytest.raises(ValueError, match="source_file must be the saved .py implementation"):
        await service.train_research_model(
            "example", source, "absent-evaluation.json", "Test the input contract", variant="baseline")
    assert not service.development.records()
    assert not list(service.development.root.glob("training/*/receipt.json"))


@pytest.mark.parametrize("body", ["pass", "...", "return", "return None",
                                   "raise NotImplementedError", "raise NotImplementedError()"])
@pytest.mark.parametrize("missing", ["fit", "predict"])
def test_unfinished_algorithm_is_rejected_before_execution(body, missing):
    fit_body = body if missing == "fit" else "return {'mean': 1.0}"
    predict_body = body if missing == "predict" else "return {'value': model['mean']}"
    source = f"def fit(training, seed):\n    {fit_body}\n\ndef predict(model, inputs):\n    {predict_body}\n"
    with pytest.raises(ValueError, match=missing + " is an unfinished placeholder"):
        validate_algorithm(source)


def test_helper_delegation_and_valid_pass_branches_remain_supported():
    source = """def fit(training, seed):
    return fit_helper(training["train"])

def fit_helper(rows):
    return {"mean": sum(row["target"] for row in rows) / len(rows)}

def predict(model, inputs):
    if inputs:
        pass
    return {"value": model["mean"]}
"""
    assert len(validate_algorithm(source)) == 64


def test_discovered_training_schema_explains_code_and_data_arguments(tmp_path):
    service = ResearchWorkspace(tmp_path / "workspace", tmp_path / "control", {"steps": []},
                                development_root=tmp_path / "development")
    tool = next(item for item in service.tools() if item.name == "train_research_model")
    properties = tool.args_schema.model_json_schema()["properties"]
    assert ".py" in properties["source_file"]["description"]
    assert "not the dataset" in properties["source_file"]["description"]
    assert "prepare_research_dataset" in properties["evaluation_file"]["description"]
    assert "parent experiment" in properties["evaluation_file"]["description"]
