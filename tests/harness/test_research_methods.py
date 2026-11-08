import copy
import json

import pytest

from agent.harness.autonomous_research import parse_plan, method_changes
from agent.harness.session_sandbox import execute


def plan():
    return {"summary":"Develop the predictor","needs_input":[],"deliverable":"model",
            "steps":[{"id":"fit","title":"Fit and evaluate a predictor","verification":"model",
                      "acceptance":"Measured performance on the fixed biological group split",
                      "approach":"Sequence composition with a ridge baseline","status":"pending","evidence":[]}]}


def test_unfinished_method_can_change_without_weakening_scientific_acceptance():
    previous=plan();candidate=copy.deepcopy(previous)
    candidate["steps"][0].update(approach="Position-aware encoding and random forest, motivated by validation errors",
                                 acceptance="No evaluation required",status="completed",evidence=["invented"])
    current=parse_plan(json.dumps(candidate),previous,objective="Develop a model")
    row=current["steps"][0]
    assert row["acceptance"]==previous["steps"][0]["acceptance"]
    assert row["approach"]==candidate["steps"][0]["approach"]
    assert row["status"]=="pending" and row["evidence"]==[]
    changes=method_changes(previous,current,2)
    assert len(changes)==1 and changes[0]["episode"]==2
    assert changes[0]["previous"]==previous["steps"][0]["approach"]


def test_completed_methods_and_evidence_are_immutable():
    previous=plan();previous["steps"][0].update(status="completed",evidence=["host-receipt"])
    proposed=copy.deepcopy(previous);proposed["steps"][0]["approach"]="Different experiment"
    result=parse_plan(json.dumps(proposed),previous,objective="Develop a model")
    assert result["steps"][0]==previous["steps"][0]
    assert method_changes(previous,result,3)==[]


@pytest.mark.parametrize("approach",[None,[],{},42,"","x"*2001])
def test_invalid_method_metadata_rejected(approach):
    value=plan();value["steps"][0]["approach"]=approach
    with pytest.raises(ValueError,match="approach"):
        parse_plan(json.dumps(value),objective="Develop a model")


def test_omitted_method_does_not_erase_the_existing_choice():
    previous=plan();proposed=copy.deepcopy(previous);del proposed["steps"][0]["approach"]
    result=parse_plan(json.dumps(proposed),previous,objective="Develop a model")
    assert result["steps"][0]["approach"]==previous["steps"][0]["approach"]
    assert method_changes(previous,result,1)==[]


@pytest.mark.asyncio
async def test_real_sandbox_fails_fast_for_missing_hub_weights(tmp_path,monkeypatch):
    monkeypatch.setenv("VENUS_SESSION_SANDBOX","bwrap")
    monkeypatch.setenv("VENUS_SANDBOX_CONTROL_ROOT",str(tmp_path/"admission"))
    monkeypatch.setenv("VENUS_SANDBOX_LOCAL_MEMORY_MB","512")
    monkeypatch.delenv("VENUS_QUEUE_JOB_ID",raising=False)
    monkeypatch.delenv("VENUS_QUEUE_OWNER",raising=False)
    monkeypatch.setenv("OPENAI_API_KEY","fixture-secret")
    work=tmp_path/"work";work.mkdir()
    (work/"run.py").write_text("""import os,time
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError,OfflineModeIsEnabled
assert all(os.environ.get(k)=="1" for k in ["HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE","HF_DATASETS_OFFLINE"])
assert "OPENAI_API_KEY" not in os.environ
start=time.monotonic()
try:hf_hub_download("offline-test/never-download","config.json")
except (LocalEntryNotFoundError,OfflineModeIsEnabled):pass
else:raise AssertionError("Unexpected model data")
assert time.monotonic()-start<5
print("offline cache miss")
""")
    result=await execute(work,tmp_path/"control","python","run.py",15)
    assert result["exit_code"]==0,result
    assert "offline cache miss" in result["stdout"]
    assert not list((tmp_path/"control").glob("sandbox-*"))


@pytest.mark.asyncio
async def test_goal_persists_method_revision_before_replacing_the_previous_plan(tmp_path):
    from agent.harness import Budget, ModelBinding, Profile
    from agent.harness.autonomous_research import AutonomousResearch, create
    profile=Profile("science_expert","expert-deepseek-harness",
        ModelBinding("dmx","glm-4-flash","https://www.dmxapi.cn/v1"),
        Budget(model_calls=12,tool_calls=6,wall_seconds=30,output_tokens=4096,thinking="disabled"))
    store=create(tmp_path/"goal",profile,"Inspect protein measurements",
                 dict(max_cycles=2,model_calls=24,tool_calls=12,wall_seconds=90))
    approaches=["Direct calculation","Vectorized calculation"]
    planned=[]
    async def consult(role,context):
        if role=="PI":return "Inspect the actual measurements."
        if role=="CB":
            approach=approaches[len(planned)];planned.append(approach)
            return json.dumps({"summary":"Inspect data","needs_input":[],
                "steps":[{"id":"inspect","title":"Inspect measurements",
                          "acceptance":"Measured summary","approach":approach}]})
        if role=="SC":
            return json.dumps({"decision":"needs_revision","summary":"Further work remains","issues":[]})
        raise AssertionError(role)
    async def executor(request,service,events):
        await service.write_research_file("notes.md","Software fixture: partial investigation.")
        await service.submit_research_checkpoint("Partial investigation",[],["notes.md"],[])
        events.append({"kind":"completed","text":"Partial work saved","event_audit":{"passed":True}})
    result=await AutonomousResearch(store,"fixture",{},consult=consult,executor=executor,tools=[]).run()
    assert planned==approaches and len(result["cycles"])==2
    revisions=result["progress"]["method_revisions"]
    assert len(revisions)==1
    assert revisions[0]["previous"]==approaches[0]
    assert approaches[1] in revisions[0].values()
    assert result["progress"]["plan"]["steps"][0]["acceptance"]=="Measured summary"
