"""Measured replacement tests; synthetic fixtures are not scientific RSI results."""
import copy
import json
from pathlib import Path

import pytest

from agent.harness.contracts import digest
from agent.harness.research_execution_feedback import feedback_state, repair_source_context
from agent.harness.research_attempt_resolution import TrainingAttemptError, attempt_id, resolve_attempts
from agent.harness.research_workspace import ResearchWorkspace

GOOD = """def fit(training, seed):
    rows = training["train"]
    return {"mean": sum(row["target"] for row in rows) / len(rows)}
def predict(model, inputs):
    return {"value": model["mean"]}
"""
BAD = GOOD.replace('row["target"]', 'row["absent_target"]')
LINEAR = """def fit(training, seed):
    rows = training["train"]
    x = [row["inputs"]["x"] for row in rows]; y = [row["target"] for row in rows]
    mx = sum(x)/len(x); my = sum(y)/len(y)
    slope = sum((a-mx)*(b-my) for a,b in zip(x,y))/sum((a-mx)**2 for a in x)
    return {"slope": slope, "intercept": my-slope*mx}
def predict(model, inputs):
    return {"value": model["slope"]*inputs["x"]+model["intercept"]}
"""


def table(offset=0):
    def rows(start, count, group):
        return [{"id":group+str(i),"group":group+str(i),
                 "inputs":{"x":i},"target":i*2+1+offset}
                for i in range(start,start+count)]
    return {"task":"regression","provenance":"Synthetic software fixture",
            "grouping":"Disjoint synthetic observations",
            "splits":{"train":rows(1,8,"t"),"validation":rows(9,3,"v"),
                      "test":rows(12,3,"q")}}


def failed_call(message, source="src/failed.py", parent=""):
    return [{"kind":"tool_call_start","tool_call_id":"failed",
             "tool_name":"mcp__venusfactory__call_tool",
             "args":{"name":"train_research_model",
                     "arguments":{"source_file":source,"parent_experiment_id":parent}}},
            {"kind":"tool_result","tool_call_id":"failed","is_error":True,
             "output":"TrainingAttemptError: "+message}]


async def service_at(tmp_path):
    service=ResearchWorkspace(tmp_path/"workspace",tmp_path/"control",{"steps":[]},
                              development_root=tmp_path/"development")
    await service.write_research_file("evaluation.json",json.dumps(table()))
    await service.write_research_file("src/failed.py",BAD)
    await service.write_research_file("src/replacement.py",GOOD)
    return service


async def failed_fit(service, parent=""):
    with pytest.raises(TrainingAttemptError, match="absent_target") as exc:
        await service.train_research_model("fit","src/failed.py","" if parent else "evaluation.json",
                                          "Failure fixture",parent_experiment_id=parent)
    return exc.value


def test_error_reference_does_not_accept_paths_or_non_ids():
    identity="a"*32
    assert attempt_id(str(TrainingAttemptError(identity,"KeyError: x")))==identity
    assert attempt_id("Training attempt ../../receipt failed: x") is None
    assert attempt_id("Training attempt draft failed: x") is None


@pytest.mark.asyncio
async def test_actual_fit_in_new_file_resolves_failure_with_unchanged_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("VENUS_JOB_QUEUE_ROOT",str(tmp_path/"queue"))
    service=await service_at(tmp_path)
    exc=await failed_fit(service)
    root=service.development.root
    failed_receipt=root/"training"/exc.attempt_id/"receipt.json"
    original=failed_receipt.read_bytes()
    state=feedback_state(failed_call(str(exc)),service.workspace,development=service.development)
    assert state["pending"][0]["training_attempt_id"]==exc.attempt_id
    assert repair_source_context(service.workspace,state)["path"]=="src/failed.py"
    replacement=await service.train_research_model(
        "repaired","src/replacement.py","evaluation.json","Fit a numeric reference")
    # A different successful call or a model-authored claim alone cannot clear the error.
    assert feedback_state([],service.workspace,state)["errors"]==state["errors"]
    snapshot=copy.deepcopy(state)
    current=feedback_state([],service.workspace,state,development=service.development)
    assert current["errors"]==[] and current["pending"]==[]
    link=current["superseded"][0]
    assert link["failed_attempt_id"]==exc.attempt_id
    assert link["replacement_experiment_id"]==replacement["experiment"]["id"]
    assert failed_receipt.read_bytes()==original and state==snapshot
    assert json.loads(original)["status"]=="failed"
    assert repair_source_context(service.workspace,current) is None
    assert len(list((root/"attempt_resolutions").glob("*.json")))==1
    # Restart/resume is idempotent; a successful replacement is not a new delivery.
    assert feedback_state([],service.workspace,current,development=service.development)==current
    assert resolve_attempts(state,service.development)==current
    assert not service.proposals
    assert not replacement["experiment"]["test_used_for_feedback"]


@pytest.mark.asyncio
async def test_other_data_and_older_success_do_not_resolve_a_failed_fit(tmp_path,monkeypatch):
    monkeypatch.setenv("VENUS_JOB_QUEUE_ROOT",str(tmp_path/"queue"))
    service=await service_at(tmp_path)
    await service.train_research_model("older","src/replacement.py","evaluation.json","Earlier fit")
    exc=await failed_fit(service)
    state=feedback_state(failed_call(str(exc)),service.workspace)
    await service.write_research_file("other.json",json.dumps(table(100)))
    await service.train_research_model("unrelated","src/replacement.py","other.json","Different endpoint")
    assert feedback_state([],service.workspace,state,development=service.development)["errors"]==state["errors"]
    assert not (service.development.root/"attempt_resolutions").exists()


@pytest.mark.asyncio
async def test_different_parent_and_ablation_do_not_resolve_candidate_failure(tmp_path,monkeypatch):
    monkeypatch.setenv("VENUS_JOB_QUEUE_ROOT",str(tmp_path/"queue"))
    service=await service_at(tmp_path)
    baseline=await service.train_research_model("baseline","src/replacement.py","evaluation.json","Reference")
    parent=baseline["experiment"]["id"]
    exc=await failed_fit(service,parent)
    state=feedback_state(failed_call(str(exc),parent=parent),service.workspace)
    await service.write_research_file("src/revision.py",LINEAR)
    await service.train_research_model("ablation","src/revision.py","","Controlled removal",
                                      parent_experiment_id=parent,variant="ablation")
    assert feedback_state([],service.workspace,state,development=service.development)["pending"]
    sibling=await service.train_research_model("sibling","src/replacement.py","evaluation.json","Another baseline")
    await service.train_research_model("other branch","src/revision.py","","Different parent",
                                      parent_experiment_id=sibling["experiment"]["id"])
    assert feedback_state([],service.workspace,state,development=service.development)["pending"]
    candidate=await service.train_research_model("revision","src/revision.py","","Measured replacement",
                                                parent_experiment_id=parent)
    result=feedback_state([],service.workspace,state,development=service.development)
    assert not result["pending"]
    assert result["superseded"][0]["replacement_experiment_id"]==candidate["experiment"]["id"]


@pytest.mark.asyncio
@pytest.mark.parametrize("damage",["failed_receipt","failed_source","success_receipt",
                                    "success_runtime","evaluation","wrong_failure_key","fake_id","legacy"])
async def test_tampered_or_unbound_receipts_never_resolve_failure(tmp_path,monkeypatch,damage):
    monkeypatch.setenv("VENUS_JOB_QUEUE_ROOT",str(tmp_path/"queue"))
    service=await service_at(tmp_path)
    exc=await failed_fit(service)
    state=feedback_state(failed_call(str(exc)),service.workspace)
    good=await service.train_research_model("good","src/replacement.py","evaluation.json","Reference")
    root=service.development.root
    bad_folder=root/"training"/exc.attempt_id
    good_folder=root/"training"/good["training"]["id"]
    if damage=="failed_receipt":
        path=bad_folder/"receipt.json"; row=json.loads(path.read_text());row["status"]="completed"
        path.write_text(json.dumps(row))
    elif damage=="failed_source":
        (bad_folder/"algorithm.py").write_text(GOOD)
    elif damage=="success_receipt":
        path=good_folder/"receipt.json";row=json.loads(path.read_text());row["prediction_key"]="other"
        path.write_text(json.dumps(row))
    elif damage=="success_runtime":
        path=root/"experiments"/good["experiment"]["id"]/"runtime"/good["training"]["source"]
        path.write_text("# changed")
    elif damage=="evaluation":
        (root/"evaluations"/(good["experiment"]["evaluation_snapshot_id"]+".json")).write_text("{}")
    elif damage=="wrong_failure_key":
        state["pending"][0]["key"]="not the failed invocation"
    elif damage=="fake_id":
        state["pending"][0]["training_attempt_id"]="f"*32
    else:
        path=bad_folder/"receipt.json";row=json.loads(path.read_text());row.pop("attempt_protocol")
        row["receipt_sha256"]=digest({k:v for k,v in row.items() if k!="receipt_sha256"})
        path.write_text(json.dumps(row))
    result=feedback_state([],service.workspace,state,development=service.development)
    assert result["errors"]==state["errors"] and not result.get("superseded")


@pytest.mark.asyncio
async def test_coordinator_reaches_independent_review_after_measured_replacement(tmp_path,monkeypatch):
    from agent.harness import Budget, ModelBinding, Profile
    from agent.harness import autonomous_research as ar
    from agent.harness.research_data_store import store_for_goal

    monkeypatch.setenv("VENUS_JOB_QUEUE_ROOT",str(tmp_path/"queue"))
    profile=Profile("science_expert","expert-deepseek-harness",
                    ModelBinding("dmx","unisound-u2-flash","https://www.dmxapi.cn/v1"),
                    Budget(model_calls=16,tool_calls=10,wall_seconds=120))
    store=ar.create(tmp_path/"goal",profile,"Develop a predictive model",
                    {"max_cycles":2,"model_calls":32,"tool_calls":20,"wall_seconds":240})
    workspace=store.root/"workspace"
    (workspace/"README.md").write_text("Synthetic fixture; not biological evidence.")
    store_for_goal(store.root).write_evaluation("evaluation.json",json.dumps(table()).encode())
    seen=[];reviewed=[];iterations=[]
    async def consult(role, context):
        if role=="PI":return "Develop and measure a reference and revision."
        if role=="CB":
            return json.dumps({"summary":"Develop a model","deliverable":"model","needs_input":[],
                "steps":[{"id":"fit","title":"Fit and compare","acceptance":"Measured revision","verification":"revision"},
                         {"id":"review","title":"Independent review","acceptance":"Review receipt","verification":"review"}]})
        if role=="SC":
            seen.append(copy.deepcopy(context["execution"]))
            return json.dumps({"decision":"ready_for_host_validation","summary":"Measured work inspected","issues":[]})
        raise AssertionError(role)
    async def executor(request,service,events):
        iterations.append(True)
        if len(iterations)==1:
            await service.write_research_file("src/failed.py",BAD)
            exc=await failed_fit(service)
            events.extend(failed_call(str(exc)))
            await service.submit_research_checkpoint("Failed fit retained",[],["README.md"],[])
        else:
            await service.write_research_file("src/replacement.py",GOOD)
            baseline=await service.train_research_model("reference","src/replacement.py","evaluation.json","Reference")
            await service.write_research_file("src/revision.py",LINEAR)
            candidate=await service.train_research_model("revision","src/revision.py","","Test linear relation",
                                                        parent_experiment_id=baseline["experiment"]["id"])
            await service.submit_research_checkpoint("Measured a revision",["fit"],
                ["README.md","experiments/"+candidate["experiment"]["id"]+".json"],[])
            bundle=store.root/"candidate";bundle.mkdir()
            (bundle/"manifest.json").write_text(json.dumps({"id":"fixture","name":"Fixture"}))
            service.proposals.append(bundle)
        events.append({"kind":"completed","text":"Evidence saved","event_audit":{"passed":True}})
    async def review(bundle,**kwargs):
        reviewed.append(bundle.name)
        receipt={"status":"needs_revision","issues":["Fixture review; no real model approval"]}
        (bundle/"admission.json").write_text(json.dumps(receipt))
        return receipt
    monkeypatch.setattr(ar,"audit_requests",lambda *a:{"passed":True,"model_calls":3})
    monkeypatch.setattr(ar,"review_proposal",review)
    state=await ar.AutonomousResearch(store,"fixture",{},consult=consult,executor=executor,tools=[]).run()
    assert len(iterations)==2 and reviewed==["candidate"]
    assert seen[0]["failures"] and seen[1]["failures"]==[]
    assert len(seen[1]["superseded_attempts"])==1
    assert state["progress"]["execution_feedback"]["errors"]==[]
    assert state["progress"]["publication_reviews"][0]["status"]=="needs_revision"
    assert state["status"]!="completed" and not state["progress"].get("capabilities")
