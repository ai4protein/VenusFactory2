"""Fixed evaluation identity across real, isolated Scientist generations."""
import copy
import json
import os
import pytest
from agent.harness.research_workspace import ResearchWorkspace
from agent.harness.contracts import digest
from test_research_development import SOURCE,data


@pytest.fixture
async def baseline(tmp_path):
    service=ResearchWorkspace(tmp_path/"workspace",tmp_path/"control",{"steps":[],"deliverable":"model"},
                              development_root=tmp_path/"development")
    await service.write_research_file("predict.py",SOURCE)
    await service.write_research_file("eval.json",json.dumps(data()))
    await service.write_research_file("weights.json",'{"coefficients":[0]}')
    result=await service.evaluate_research_model("constant","predict.py",["weights.json"],
        "eval.json","value","Constant fitted baseline","supervised",variant="baseline")
    return service,result


@pytest.mark.asyncio
async def test_child_reuses_frozen_targets_and_exports_without_replacing_working_data(baseline):
    service,parent=baseline
    altered=data();altered["splits"]["validation"][0]["target"]+=100
    await service.write_research_file("eval.json",json.dumps(altered))
    await service.write_research_file("weights.json",'{"coefficients":[1]}')
    with pytest.raises(ValueError,match="identical data"):
        await service.evaluate_research_model("changed labels","predict.py",["weights.json"],
            "eval.json","value","Rejected target transformation","supervised",parent["id"])
    result=await service.evaluate_research_model("revision","predict.py",["weights.json"],
        "","value","Revised fitted constant","supervised",parent["id"])
    assert result["evaluation_sha256"]==parent["evaluation_sha256"]
    assert result["evaluation_origin"]=={"kind":"parent_snapshot","experiment_id":parent["id"]}
    assert result["validation"]["score"]==pytest.approx(sum(abs(1-r["target"]) for r in data()["splits"]["validation"])/3)
    assert len(list((service.development.root/"evaluations").glob("*.json")))==1
    tools={tool.name:tool for tool in service.tools()}
    restored=await tools["restore_research_evaluation"].ainvoke({"experiment_id":result["id"]})
    frozen=json.loads((service.workspace/restored["path"]).read_text())
    assert frozen["splits"]==data()["splits"] and "claimed_metric" not in frozen
    assert json.loads((service.workspace/"eval.json").read_text())==altered
    assert (await service.restore_research_evaluation(result["id"]))==restored
    with pytest.raises(ValueError,match="different data"):
        await service.restore_research_evaluation(result["id"],"eval.json")


@pytest.mark.asyncio
async def test_corrupt_control_snapshot_is_rejected_before_model_execution(baseline):
    service,parent=baseline
    snapshot=service.development.root/"evaluations"/(parent["evaluation_snapshot_id"]+".json")
    snapshot.write_text('{"tampered":true}')
    async def forbidden(*args,**kwargs):
        raise AssertionError("A corrupt benchmark must not reach model execution")
    service.runner=forbidden
    with pytest.raises(ValueError,match="snapshot changed"):
        await service.evaluate_research_model("revision","predict.py",["weights.json"],
            "","value","New fitted model","supervised",parent["id"])


@pytest.mark.asyncio
async def test_historic_receipt_requires_matching_explicit_data_without_rewriting_it(baseline):
    service,parent=baseline
    receipt=service.development.root/"experiments"/parent["id"]/"receipt.json"
    legacy={k:v for k,v in parent.items() if k not in {"evaluation_snapshot_id","evaluation_snapshot_sha256","evaluation_origin","receipt_sha256"}}
    legacy["receipt_sha256"]=digest(legacy);receipt.write_text(json.dumps(legacy))
    original=receipt.read_bytes()
    await service.write_research_file("weights.json",'{"coefficients":[1]}')
    with pytest.raises(ValueError,match="historic experiment"):
        await service.evaluate_research_model("revision","predict.py",["weights.json"],
            "","value","New model","supervised",parent["id"])
    child=await service.evaluate_research_model("revision","predict.py",["weights.json"],
        "eval.json","value","New model","supervised",parent["id"])
    assert child["evaluation_snapshot_id"]==parent["evaluation_snapshot_id"]
    assert receipt.read_bytes()==original


@pytest.mark.asyncio
async def test_frozen_copy_still_rejects_traversal_and_links(baseline,tmp_path):
    service,parent=baseline
    private=tmp_path/"private";private.write_text("outside")
    (service.workspace/"linked").symlink_to(private)
    os.link(private,service.workspace/"hardlink")
    for path in ("../private","linked","hardlink","/etc/passwd",".env"):
        with pytest.raises((ValueError,OSError)):
            await service.restore_research_evaluation(parent["id"],path)
    assert private.read_text()=="outside"


@pytest.mark.asyncio
async def test_first_baseline_cannot_guess_an_evaluation(tmp_path):
    service=ResearchWorkspace(tmp_path/"workspace",tmp_path/"control",{"steps":[]})
    with pytest.raises(ValueError,match="first baseline"):
        await service.evaluate_research_model("baseline","model.py",[],
            "","value","Test a baseline","supervised",variant="baseline")
