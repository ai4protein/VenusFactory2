"""Exercise default baseline and child selection through the public research tools."""
import json
import pytest
from agent.harness.research_workspace import ResearchWorkspace

MEAN = """def fit(training, seed):
    return {"mean": sum(row["target"] for row in training["train"]) / len(training["train"])}
def predict(model, inputs):
    return {"value": model["mean"]}
"""
LINEAR = """def fit(training, seed):
    rows = training["train"]
    x = [row["inputs"]["x"] for row in rows]
    y = [row["target"] for row in rows]
    mx, my = sum(x)/len(x), sum(y)/len(y)
    slope = sum((a-mx)*(b-my) for a,b in zip(x,y)) / sum((a-mx)**2 for a in x)
    return {"slope": slope, "intercept": my-slope*mx}
def predict(model, inputs):
    return {"value": model["slope"]*inputs["x"]+model["intercept"]}
"""

async def workspace(tmp_path):
    service = ResearchWorkspace(tmp_path/"work", tmp_path/"control", {"steps":[]},
                                development_root=tmp_path/"development")
    rows = lambda start, label: [{"id":label+str(i),"group":label+str(i),
                                 "inputs":{"x":i},"target":2*i+1} for i in range(start,start+8)]
    data = {"task":"regression", "provenance":"Synthetic interface test only",
            "grouping":"Independent software fixture rows",
            "splits":{"train":rows(1,"train"),"validation":rows(11,"validation"),"test":rows(21,"test")}}
    await service.write_research_file("evaluation.json",json.dumps(data))
    return service

@pytest.mark.asyncio
async def test_training_defaults_fit_baseline_then_measure_child_on_frozen_splits(tmp_path):
    service = await workspace(tmp_path)
    await service.write_research_file("algorithm.py",MEAN)
    first = await service.train_research_model("Mean","algorithm.py","evaluation.json","Fit a reference")
    await service.write_research_file("algorithm.py",LINEAR)
    child = await service.train_research_model("Linear","algorithm.py","","Fit a linear relation",
                                              parent_experiment_id=first["experiment"]["id"])
    baseline, revision = first["experiment"], child["experiment"]
    assert baseline["variant"] == "baseline" and baseline["parent"] is None
    assert revision["variant"] == "candidate" and revision["parent"] == baseline["id"]
    assert revision["evaluation_sha256"] == baseline["evaluation_sha256"]
    assert child["training"]["validation_and_test_mounted"] is False
    assert revision["validation"]["score"] < baseline["validation"]["score"]
    assert not revision["test_used_for_feedback"]

@pytest.mark.asyncio
async def test_existing_model_defaults_preserve_parent_lineage(tmp_path):
    service = await workspace(tmp_path)
    await service.write_research_file("predict.py","def run(inputs): return {'value': 0.0}")
    baseline = await service.evaluate_research_model("Constant","predict.py",[],"evaluation.json",
                                                     "value","Reference prediction","supervised")
    await service.write_research_file("predict.py","def run(inputs): return {'value': 2*inputs['x']+1}")
    child = await service.evaluate_research_model("Fixture relation","predict.py",[],"","value",
                                                   "Synthetic test relationship","supervised",
                                                   parent_experiment_id=baseline["id"])
    assert baseline["variant"] == "baseline" and child["variant"] == "candidate"
    assert child["parent"] == baseline["id"]
    assert child["evaluation_sha256"] == baseline["evaluation_sha256"]

@pytest.mark.asyncio
async def test_explicit_candidate_still_requires_a_measured_parent(tmp_path):
    service = await workspace(tmp_path)
    await service.write_research_file("algorithm.py",MEAN)
    with pytest.raises(ValueError,match="baseline"):
        await service.train_research_model("Invalid","algorithm.py","evaluation.json",
                                           "Candidate without parent",variant="candidate")
    assert not list(service.development.root.glob("training/*/receipt.json"))
    assert not service.development.records()

def test_tool_schemas_expose_valid_automatic_default(tmp_path):
    service = ResearchWorkspace(tmp_path/"work",tmp_path/"control",{"steps":[]})
    for tool in service.tools():
        if tool.name not in {"train_research_model","evaluate_research_model"}:
            continue
        field = tool.args_schema.model_json_schema()["properties"]["variant"]
        assert field["default"] == "auto"
        assert set(field["enum"]) == {"auto","baseline","candidate","ablation"}
        assert "parent_experiment_id" in field["description"]


@pytest.mark.parametrize("source,line", [
    ("def fit(training, seed)" + chr(10) + "    return {}", 1),
    ("def fit(training, seed):" + chr(10) + "    return {'x': [1, 2}", 2),
])
def test_syntax_errors_identify_the_saved_source_line(source,line):
    from agent.harness.research_training import validate_algorithm
    with pytest.raises(ValueError) as error:
        validate_algorithm(source)
    assert "not valid Python" in str(error.value)
    assert f"line {line}," in str(error.value)
    assert "Repair the saved .py source" in str(error.value)
