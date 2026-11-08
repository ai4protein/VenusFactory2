"""Execution failures remain failures even if the tool transport and critic succeed."""
import json
from pathlib import Path
import pytest

from agent.harness import Budget, ModelBinding, Profile
from agent.harness import autonomous_research as ar
from agent.harness.research_workspace import ResearchWorkspace
from agent.harness.execution_outcome import execution_error


@pytest.mark.parametrize("wrapped", [False,True])
def test_legacy_python_failure_is_carried_into_next_iteration(tmp_path,wrapped):
    body={"exit_code":1,"stderr":"KeyError: missing_target","stdout":"private training data"}
    output=json.dumps(body)+" Execution budget: 4 calls remain." if wrapped else body
    events=[{"kind":"tool_call_start","tool_call_id":"x","tool_name":"mcp__venusfactory__run_session_python",
             "args":{"path":"train.py"}},
            {"kind":"tool_result","tool_call_id":"x","is_error":False,"output":output}]
    rows=ar.execution_feedback(events,tmp_path)
    assert len(rows)==1 and "missing_target" in rows[0]["error"]
    assert "private training data" not in str(rows)
    retry=[{**events[0],"tool_call_id":"y"},
           {"kind":"tool_result","tool_call_id":"y","is_error":False,"output":{"exit_code":0}}]
    assert ar.execution_feedback(events+retry,tmp_path)==[]


@pytest.mark.parametrize("result",[{"exit_code":0},{"value":1},{"exit_code":False},"not execution JSON"])
def test_successful_or_unrelated_output_is_not_reclassified(result):
    assert execution_error(result) is None


@pytest.mark.asyncio
async def test_python_failure_is_recorded_before_it_is_raised(tmp_path):
    service=ResearchWorkspace(tmp_path/"work",tmp_path/"control",{"steps":[]})
    await service.write_research_file("bad.py","raise ValueError('observed training failure')")
    with pytest.raises(ValueError,match="observed training failure"):
        await service.run_research_python("bad.py",10)
    row=service.receipts[-1]
    assert row["tool"]=="run_research_python" and row["result"]["exit_code"]!=0
    assert (service.control/"receipts.json").exists()


@pytest.mark.asyncio
async def test_critic_approval_cannot_complete_a_failed_training_checkpoint(tmp_path,monkeypatch):
    profile=Profile("science_expert","expert-kimi-code",
                    ModelBinding("dmx","glm-4-flash","https://www.dmxapi.cn/v1"),
                    Budget(model_calls=12,tool_calls=8,wall_seconds=60,output_tokens=2048))
    store=ar.create(tmp_path/"goal",profile,"Analyze the supplied observations",
                    dict(max_cycles=1,model_calls=12,tool_calls=8,wall_seconds=90))
    async def consult(role,context):
        if role=="PI": return "Execute the supplied analysis."
        if role=="CB":
            return json.dumps({"summary":"Analyze","steps":[{"id":"compute","title":"Analyze",
                               "acceptance":"Successful computed output"}],"needs_input":[],"deliverable":"analysis"})
        assert context["execution"]["failures"]
        return json.dumps({"decision":"ready_for_host_validation","summary":"Analysis completed.","issues":[]})
    async def executor(request,service,events):
        await service.write_research_file("bad.py","raise ValueError('training failed')")
        with pytest.raises(ValueError):
            await service.run_research_python("bad.py",10)
        result=service.receipts[-1]["result"]
        await service.submit_research_checkpoint("Everything completed",["compute"],["bad.py"],[])
        events.extend([
            {"kind":"tool_call_start","tool_call_id":"run","tool_name":"mcp__venusfactory__run_session_python","args":{"path":"bad.py"}},
            {"kind":"tool_result","tool_call_id":"run","is_error":False,"output":json.dumps(result)},
            {"kind":"completed","text":"Everything completed.","event_audit":{"passed":True}}])
    monkeypatch.setattr(ar,"audit_requests",lambda *a:{"passed":True,"model_calls":3})
    state=await ar.AutonomousResearch(store,"fixture",{},consult=consult,executor=executor,tools=[]).run()
    assert state["status"]!="completed" and not state["progress"]["research_completed"]
    assert state["progress"]["plan"]["steps"][0]["status"]=="pending"
    assert state["progress"]["last_review"]["decision"]=="needs_revision"
    assert "training failed" in state["cycles"][0]["summary"]
    episode=store.root/"episodes/cycle_000"
    assert json.loads((episode/"review.json").read_text())["decision"]=="ready_for_host_validation"
    assert json.loads((episode/"host_review.json").read_text())["passed"] is False

def test_inspection_only_iteration_does_not_erase_previous_failure(tmp_path):
    previous={"iteration":1,"errors":[{"tool":"run_session_python","error":"missing target"}]}
    read=[{"kind":"tool_call_start","tool_name":"mcp__venusfactory__read_session_file","tool_call_id":"read","args":{"path":"data.csv"}},
          {"kind":"tool_result","tool_call_id":"read","is_error":False,"output":"header"}]
    assert ar.iteration_feedback(read,tmp_path,previous)==previous["errors"]
    run=[{"kind":"tool_call_start","tool_name":"mcp__venusfactory__run_session_python","tool_call_id":"retry","args":{"path":"train.py"}},
         {"kind":"tool_result","tool_call_id":"retry","is_error":False,"output":{"exit_code":0}}]
    assert ar.iteration_feedback(run,tmp_path,previous)==[]
