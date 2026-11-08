"""Sandboxed generated training: software fixtures, not biological model results."""
import copy
import json
from pathlib import Path

import pytest

from agent.harness.research_workspace import ResearchWorkspace, run_python
from agent.harness.research_training import validate_algorithm
from agent.harness.capability_review import model_evaluation
from agent.harness.tool_disclosure import ToolDisclosure


BASELINE = """from pathlib import Path
def fit(training, seed):
    assert set(training) == {"task","strategy","train","unlabeled"}
    assert not Path("evaluation.json").exists()
    assert not Path("private-other-session.txt").exists()
    assert "holdout" not in str(training)
    return {"mean":sum(row["target"] for row in training["train"])/len(training["train"])}
def predict(model, inputs):
    if set(inputs)!={"x"} or type(inputs["x"]) not in (int,float):
        raise ValueError("Provide numeric x")
    return {"value":model["mean"]}
"""

LINEAR = """def fit(training, seed):
    rows=training["train"]
    x=[row["inputs"]["x"] for row in rows];y=[row["target"] for row in rows]
    mx=sum(x)/len(x);my=sum(y)/len(y)
    slope=sum((a-mx)*(b-my) for a,b in zip(x,y))/sum((a-mx)**2 for a in x)
    return {"slope":slope,"intercept":my-slope*mx}
def predict(model, inputs):
    if set(inputs)!={"x"} or type(inputs["x"]) not in (int,float):
        raise ValueError("Provide numeric x")
    return {"value":model["slope"]*inputs["x"]+model["intercept"]}
"""


def dataset():
    def rows(start, count, prefix):
        return [{"id":prefix+str(i),"group":prefix+str(i),
                 "inputs":{"x":i},"target":2*i+1} for i in range(start,start+count)]
    return {"task":"regression","provenance":"Synthetic software fixture, not protein data",
            "grouping":"Independent synthetic observations",
            "splits":{"train":rows(1,8,"train"),"validation":rows(9,3,"holdout-v"),
                      "test":rows(12,3,"holdout-t")}}


async def service_at(tmp_path):
    service=ResearchWorkspace(tmp_path/"workspace",tmp_path/"control",{"steps":[]},
                              development_root=tmp_path/"development")
    await service.write_research_file("evaluation.json",json.dumps(dataset()))
    await service.write_research_file("algorithm.py",BASELINE)
    await service.write_research_file("private-other-session.txt","not a training input")
    return service


@pytest.mark.asyncio
async def test_fit_revision_reproduction_and_training_only_mount(tmp_path):
    service=await service_at(tmp_path)
    with pytest.raises(ValueError, match="first model with variant=baseline"):
        await service.train_research_model("mean","algorithm.py","evaluation.json","Missing parent",variant="candidate")
    assert not list((service.development.root/"training").glob("*/receipt.json"))
    baseline=await service.train_research_model("mean","algorithm.py","evaluation.json","Fit a reference",variant="baseline")
    b=baseline["experiment"];bt=baseline["training"]
    assert b["validation"]["score"]>0 and not b["test_used_for_feedback"]
    diagnostics=b["validation"]["diagnostics"]
    assert diagnostics["constant_predictions"] and diagnostics["matches_training_reference"]
    assert diagnostics["split"]=="validation" and diagnostics["prediction_count"]==3
    assert "Check feature variation" in baseline["next_step"]
    assert service.development.summary()["generations"][0]["validation"]["diagnostics"]==diagnostics
    assert bt["validation_and_test_mounted"] is False
    assert bt["training_rows"]==8 and len(service.development.records())==1
    for variant in ("candidate", "ablation"):
        with pytest.raises(ValueError, match=b["id"]) as error:
            await service.train_research_model("revision","algorithm.py","evaluation.json","Missing parent",variant=variant)
        assert "same split" in str(error.value)
    assert len(list((service.development.root/"training").glob("*/receipt.json"))) == 1
    await service.write_research_file("algorithm.py",LINEAR)
    child=await service.train_research_model("linear","algorithm.py","","Test a linear relation",
                    parent_experiment_id=b["id"],strategy="few_shot")
    e=child["experiment"];t=child["training"]
    assert e["validation"]["score"]==0 and e["parent"]==b["id"]
    assert e["evaluation_sha256"]==b["evaluation_sha256"]
    assert e["comparison"]["improved_on_validation"] and e["fresh_process_replay"]
    assert not e["validation"]["diagnostics"]["constant_predictions"]
    assert e["validation"]["diagnostics"]["warnings"]==[]
    folder=service.development.root/"experiments"/e["id"]/"runtime"
    assert not list(folder.rglob("training.json")) and not list(folder.rglob("evaluation.json"))
    assert not service.proposals
    spec={"input_schema":{"type":"object","properties":{"x":{"type":"number"}},"required":["x"]},
          "output_schema":{"type":"object","properties":{"value":{"type":"number"}},"required":["value"]},
          "evaluation_file":t["evaluation_file"],"prediction_key":"value",
          "reproduction_script":t["reproduction_script"],"fitted_files":t["fitted_files"]}
    manifest={"source":t["source"],"files":e["files"],"review_spec":spec,
              "evidence_files":{name:"private" for name in t["evidence_files"]}}
    review=await model_evaluation(service.workspace,manifest,tmp_path/"review",run_python)
    assert review["training_reproduced"] and review["score"]==0
    assert review["diagnostics"]["split"]=="test"
    assert not review["diagnostics"]["constant_predictions"]
    await service.write_research_file("same.py",LINEAR+"# unchanged algorithm")
    with pytest.raises(ValueError,match="algorithm or fitted state"):
        await service.train_research_model("renamed","same.py","","No actual revision",parent_experiment_id=e["id"])
    assert len(service.development.records())==2

@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["overlap","invalid_source","outside_source","unlabeled_targets"])
async def test_invalid_training_inputs_never_start_a_process(tmp_path, damage):
    service=await service_at(tmp_path)
    source="algorithm.py";data=dataset()
    if damage=="overlap":
        data["splits"]["test"][0]["group"]=data["splits"]["train"][0]["group"]
    elif damage=="invalid_source":
        await service.write_research_file(source,"def fit(x): return {}")
    elif damage=="outside_source": source="../outside.py"
    else:
        data["unlabeled"]=[{"id":"unlabeled","inputs":{"x":100},"target":20}]
    await service.write_research_file("evaluation.json",json.dumps(data))
    async def forbidden(*args,**kwargs):
        raise AssertionError("Invalid inputs must not start a worker")
    service.runner=forbidden
    with pytest.raises(ValueError):
        await service.train_research_model("invalid",source,"evaluation.json","Invalid fixture",variant="baseline")
    assert not service.development.records()


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["bad_state","nonfinite","tamper"])
async def test_failed_fit_keeps_evidence_and_cannot_become_a_generation(tmp_path, mutation):
    service=await service_at(tmp_path)
    bad={"bad_state":"return [row['target'] for row in training['train']]",
         "nonfinite":"return {'x':sum(row['target'] for row in training['train'])*float('nan')}",
         "tamper":"Path('training.json').write_text('{}'); return {'mean':sum(row['target'] for row in training['train'])}"}[mutation]
    source="from pathlib import Path\ndef fit(training,seed):\n    "+bad+"\ndef predict(model,inputs):\n    return {'value':0}\n"
    await service.write_research_file("algorithm.py",source)
    with pytest.raises(ValueError):
        await service.train_research_model("invalid","algorithm.py","evaluation.json","Test failure",variant="baseline")
    records=[json.loads(p.read_text()) for p in (service.development.root/"training").glob("*/receipt.json")]
    assert len(records)==1 and records[0]["status"]=="failed"
    assert not service.development.records()


def test_training_tool_is_discoverable_with_an_explicit_interface(tmp_path):
    service=ResearchWorkspace(tmp_path/"workspace",tmp_path/"control",{"steps":[]})
    tools={tool.name:tool for tool in service.tools()}
    tool=tools["train_research_model"]
    assert tool.metadata["discovery_category"]=="training"
    properties=tool.args_schema.model_json_schema()["properties"]
    assert {"source_file","evaluation_file","parent_experiment_id","variant","seed"}<=set(properties)


@pytest.mark.parametrize("source",[
    "def fit(x): return {}\ndef predict(m,i): return {}",
    "def fit(x,s): return {}\ndef predict(m,i): return {}\ndef run(x): return {}",
    "def fit(x,s):\n _vf_state={}\n return {}\ndef predict(m,i): return {}",
])
def test_unsupported_algorithm_interfaces_are_rejected(source):
    with pytest.raises(ValueError):
        validate_algorithm(source)


@pytest.mark.asyncio
async def test_failed_preflight_keeps_source_and_diagnostics_without_starting_worker(tmp_path):
    service = await service_at(tmp_path)
    broken = "pipeline=[('a',object())('b',object())]\ndef fit(training,seed):\n return pipeline\ndef predict(m,i):\n return {'value':0}\n"
    await service.write_research_file("algorithm.py", broken)
    executions = []

    async def forbidden(*args, **kwargs):
        executions.append(args)
        raise AssertionError("Static rejection must not acquire a sandbox")

    service.runner = forbidden
    with pytest.raises(ValueError, match="fit does not read"):
        await service.train_research_model("broken", "algorithm.py", "evaluation.json", "Fixture")
    paths = list((service.development.root/"training").glob("*/receipt.json"))
    assert len(paths) == 1 and not executions
    receipt = json.loads(paths[0].read_text())
    assert receipt["status"] == "failed" and receipt["stage"] == "preflight"
    assert receipt["source_check"]["valid"] is False and receipt["source_snapshot"] == "algorithm.py"
    assert "fit does not read" in receipt["error"]
    assert "execution" not in receipt and "experiment_id" not in receipt
    assert (paths[0].parent/"algorithm.py").read_text() == broken
    training = json.loads((paths[0].parent/"workspace/training.json").read_text())
    assert "validation" not in training and "test" not in training
    assert receipt["validation_and_test_mounted"] is False
    await service.write_research_file("algorithm.py", LINEAR)
    assert (paths[0].parent/"algorithm.py").read_text() == broken
    assert not service.development.records()


@pytest.mark.asyncio
async def test_preflight_failures_count_toward_training_attempt_budget(tmp_path):
    service = await service_at(tmp_path)
    await service.write_research_file("algorithm.py", "def fit(training,seed):return {}")
    training = service.development.root/"training"
    for i in range(63):
        folder = training/str(i)
        folder.mkdir(parents=True)
        (folder/"receipt.json").write_text('{"status":"failed","stage":"preflight"}')
    with pytest.raises(ValueError, match="source check failed"):
        await service.train_research_model("broken", "algorithm.py", "evaluation.json", "Fixture")
    assert len(list(training.glob("*/receipt.json"))) == 64
    with pytest.raises(ValueError, match="Training attempt budget reached"):
        await service.train_research_model("broken", "algorithm.py", "evaluation.json", "Fixture")
    assert len(list(training.glob("*/receipt.json"))) == 64
