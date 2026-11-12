"""Plan evidence checks use synthetic software fixtures, not biological results."""
import copy
import json
import pytest
from agent.harness import Budget, ModelBinding, Profile
from agent.harness import autonomous_research as ar
from agent.harness.contracts import file_hash
from agent.harness.research_development import Development
from agent.harness.research_workspace import ResearchWorkspace
from agent.harness.research_plan_evidence import checkpoint_steps, requirements
from agent.harness.scientist_role_output import specification, validate_output

def plan(kind="model"):
    return {"summary":"Develop a predictor","deliverable":"model","needs_input":[],
            "steps":[{"id":"fit","title":"Fit the predictor","acceptance":"Measured fitted predictor",
                      "verification":kind}]}

@pytest.mark.asyncio
async def test_stub_execution_and_critic_approval_cannot_complete_model_step(tmp_path,monkeypatch):
    profile=Profile("science_expert","expert-kimi-code",
                    ModelBinding("dmx","glm-4-flash","https://www.dmxapi.cn/v1"),
                    Budget(model_calls=12,tool_calls=8,wall_seconds=60,output_tokens=2048))
    store=ar.create(tmp_path/"goal",profile,"Develop a protein model from supplied measurements",
                    dict(max_cycles=1,model_calls=12,tool_calls=8,wall_seconds=90))
    async def consult(role,context):
        if role=="PI":return "Build and evaluate the predictor."
        if role=="CB":
            value=plan()
            value["steps"].insert(0,{"id":"notes","title":"Document data","acceptance":"Notes saved","verification":"artifact"})
            return json.dumps(value)
        assert context["execution"]["step_checks"][1]["passed"] is False
        return json.dumps({"decision":"ready_for_host_validation","summary":"Training completed.","issues":[]})
    async def executor(request,service,events):
        await service.write_research_file("model.py","def fit(training, seed):\n    pass\ndef predict(model, inputs):\n    pass\n")
        await service.write_research_file("README.md","Observed input columns.")
        assert (await service.run_research_python("model.py",10))["exit_code"]==0
        await service.submit_research_checkpoint("Everything completed",["notes","fit"],["model.py","README.md"],[])
        events.append({"kind":"completed","text":"Training completed.","event_audit":{"passed":True}})
    monkeypatch.setattr(ar,"audit_requests",lambda *a:{"passed":True,"model_calls":3})
    state=await ar.AutonomousResearch(store,"fixture",{},consult=consult,executor=executor,tools=[]).run()
    steps={s["id"]:s for s in state["progress"]["plan"]["steps"]}
    assert steps["notes"]["status"]=="completed"
    assert steps["fit"]["status"]=="pending"
    assert state["progress"]["host_review"]["passed"] is False
    assert state["progress"]["host_review"]["execution_passed"] is True
    assert "measured generation" in state["progress"]["last_review"]["summary"]
    assert not state["progress"]["research_completed"]


@pytest.mark.asyncio
async def test_measured_model_is_bound_to_checkpointed_unchanged_runtime(tmp_path):
    service=ResearchWorkspace(tmp_path/"work",tmp_path/"control",{"steps":[]},development_root=tmp_path/"development")
    rows=lambda start,count:[{"id":str(i),"group":str(i),"inputs":{"x":i},"target":2*i} for i in range(start,start+count)]
    data={"task":"regression","provenance":"Synthetic software fixture","grouping":"Independent synthetic observations",
          "splits":{"train":rows(1,6),"validation":rows(7,3),"test":rows(10,3)}}
    await service.write_research_file("data.json",json.dumps(data))
    await service.write_research_file("fit.py","def fit(training, seed):\n    return {'mean':sum(r['target'] for r in training['train'])/len(training['train'])}\ndef predict(model, inputs):\n    return {'value':model['mean']}\n")
    await service.validate_research_evaluation("data.json")
    data_cp={"completed_steps":["fit"],"artifacts":[{"path":"data.json","sha256":file_hash(service.workspace/"data.json")}]}
    data_kwargs=dict(workspace=service.workspace,receipts=service.receipts,experiments=[],progress={})
    for verification in ("data", "data_partition"):
        check = checkpoint_steps(plan(verification), data_cp, **data_kwargs)[0]
        assert check["passed"] and check["requirements"] == ["data_partition"]
    data_cp["artifacts"][0]["sha256"]="changed"
    for verification in ("data", "data_partition"):
        assert not checkpoint_steps(plan(verification), data_cp, **data_kwargs)[0]["passed"]
    result=await service.train_research_model(name="Fixture",source_file="fit.py",evaluation_file="data.json",hypothesis="Training mean reference",variant="baseline")
    record=result["experiment"]
    cp={"completed_steps":["fit"],"artifacts":[{"path":p,"sha256":h} for p,h in record["files"].items()]}
    kwargs=dict(workspace=service.workspace,receipts=service.receipts,experiments=Development(tmp_path/"development").records(),progress={})
    assert checkpoint_steps(plan(),cp,**kwargs)[0]["passed"]
    for verification in ("data", "data_partition"):
        assert checkpoint_steps(plan(verification), cp, **kwargs)[0]["passed"]
    assert not checkpoint_steps(plan("revision"),cp,**kwargs)[0]["passed"]
    unrelated={"completed_steps":["fit"],"artifacts":[{"path":"fit.py","sha256":file_hash(service.workspace/"fit.py")}]}
    assert not checkpoint_steps(plan(),unrelated,**kwargs)[0]["passed"]
    model_path=next(p for p in record["files"] if p.endswith("model.json"))
    (service.workspace/model_path).write_text("{}")
    assert not checkpoint_steps(plan(),cp,**kwargs)[0]["passed"]

def test_legacy_explicit_training_requirement_cannot_be_downgraded():
    value=plan("artifact")
    value["steps"][0]["acceptance"]="Use train_research_model with variant=baseline."
    rows=checkpoint_steps(value,{"completed_steps":["fit"],"artifacts":[{"path":"stub.py","sha256":"abc"}]},
                          workspace=".",receipts=[],experiments=[],progress={})
    assert not rows[0]["passed"] and "model" in rows[0]["requirements"]


@pytest.mark.parametrize("protocol",["chat_completions","responses"])
def test_plan_verification_is_typed_and_cannot_be_weakened_on_replan(protocol):
    value=plan()
    validate_output(json.dumps(value),specification("CB",protocol))
    previous=ar.parse_plan(json.dumps(value))
    value["steps"][0]["verification"]="artifact"
    revised=ar.parse_plan(json.dumps(value),previous)
    assert revised["steps"][0]["verification"]=="model"
    value["steps"][0]["verification"]="believe_my_claim"
    with pytest.raises(ValueError):ar.parse_plan(json.dumps(value))
    with pytest.raises(ValueError):validate_output(json.dumps(value),specification("CB",protocol))


@pytest.mark.parametrize("name", ["validate_research_evaluation", "prepare_research_dataset"])
def test_named_partition_tools_add_canonical_strict_requirement(name):
    step = {"title": "Prepare measured observations", "acceptance": "Call " + name, "verification": "artifact"}
    assert requirements(step) == ["artifact", "data_partition"]
    value = plan("artifact")
    value["steps"][0].update(step)
    result = checkpoint_steps(value, {"completed_steps": ["fit"], "artifacts": [{"path": "notes.md", "sha256": "abc"}]},
                              workspace=".", receipts=[], experiments=[], progress={})
    assert result[0]["passed"] is False
    assert result[0]["requirements"] == ["artifact", "data_partition"]


@pytest.mark.parametrize("replacement", ["artifact", "execution", "data", "data_partition"])
@pytest.mark.parametrize("status", ["pending", "completed"])
def test_legacy_locked_partition_requirement_is_never_downgraded(replacement, status):
    previous = plan("data")
    previous["steps"][0].update(status=status, evidence=[{"path": "protected-evaluation.json", "sha256": "old"}])
    original = copy.deepcopy(previous)
    proposed = plan(replacement)
    proposed["steps"][0]["acceptance"] = "Reading the FASTA is enough"
    revised = ar.parse_plan(json.dumps(proposed), previous)
    step = revised["steps"][0]
    assert step["verification"] == "data_partition"
    assert step["acceptance"] == original["steps"][0]["acceptance"]
    assert step["status"] == status and step["evidence"] == original["steps"][0]["evidence"]
    assert previous == original


def test_omitted_legacy_completed_step_keeps_partition_contract_without_mutating_history():
    previous = plan("data")
    previous["steps"][0].update(status="completed", evidence=[])
    original = copy.deepcopy(previous)
    proposed = plan("artifact")
    proposed["steps"][0]["id"] = "report"
    revised = ar.parse_plan(json.dumps(proposed), previous)
    retained = next(step for step in revised["steps"] if step["id"] == "fit")
    assert retained["verification"] == "data_partition" and retained["status"] == "completed"
    assert previous == original


@pytest.mark.parametrize("verification", ["data", "data_partition"])
def test_initial_partition_plan_is_canonical_without_claiming_evidence(verification):
    value = plan(verification)
    original = copy.deepcopy(value)
    result = ar.parse_plan(json.dumps(value))
    assert result["steps"][0]["verification"] == "data_partition"
    assert result["steps"][0]["status"] == "pending" and result["steps"][0]["evidence"] == []
    assert value == original
