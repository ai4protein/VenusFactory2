import json
from types import SimpleNamespace
import pytest
from agent.harness.research_execution_feedback import feedback_state, execution_feedback
from agent.harness.scientist_role_output import specification, validate_output, RoleOutputError
from agent.harness.autonomous_research import AutonomousResearch, parse_plan

def event(identity,path,columns,error=False):
    return [{"kind":"tool_call_start","tool_call_id":identity,"tool_name":"mcp__venusfactory__prepare_research_dataset",
             "args":{"data_file":path,"input_columns":columns}},
            {"kind":"tool_result","tool_call_id":identity,"is_error":error,"output":"Target included" if error else "Prepared"}]

def test_corrected_arguments_resolve_only_the_same_dataset():
    bad=event("bad","private-a.csv",["sequence","target"],True)
    other=event("other","private-b.csv",["sequence"])
    assert len(execution_feedback(bad+other,"."))==1
    retry=event("retry","private-a.csv",["sequence"])
    assert execution_feedback(bad+other+retry,".")==[]
    state=feedback_state(bad,".")
    assert "private-a.csv" not in json.dumps(state)
    assert feedback_state([],".",state)["errors"]==state["errors"]
    assert feedback_state(retry[:1],".",state)["errors"]==state["errors"]
    assert feedback_state(retry,".",state)["errors"]==[]

def value():
    return {"summary":"Fit","deliverable":"model","needs_input":[],
            "steps":[{"id":"fit","title":"Fit model","acceptance":"Measured model","verification":"model"}]}

def test_echoed_completion_metadata_is_discarded_before_schema_validation():
    plan=value();plan["steps"][0].update(status="completed",evidence=[{"path":"invented"}])
    raw=validate_output(json.dumps(plan),specification("CB","chat_completions"))
    parsed=parse_plan(raw)
    assert parsed["steps"][0]["status"]=="pending" and parsed["steps"][0]["evidence"]==[]
    assert "invented" not in raw

@pytest.mark.asyncio
async def test_transport_schema_error_receives_one_bounded_format_repair(tmp_path):
    relay=SimpleNamespace(calls=0)
    profile=SimpleNamespace(budget=SimpleNamespace(model_calls=12))
    seen=[]
    async def consult(role,context):
        relay.calls+=1;seen.append(context)
        if relay.calls==1:
            bad=value();bad["steps"][0]["verification"]="training"
            return validate_output(json.dumps(bad),specification("CB","chat_completions"))
        return json.dumps(value())
    driver=AutonomousResearch(None,"",{},consult=consult)
    parsed=await driver.structured("CB",{"objective":"Fit the supplied real data"},profile,relay,tmp_path,parse_plan)
    assert parsed["steps"][0]["verification"]=="model" and relay.calls==2
    assert "Allowed values" in seen[1]["format_repair"]["validation_error"]
    assert seen[1]["objective"]==seen[0]["objective"]

@pytest.mark.asyncio
async def test_wrong_model_transport_error_is_not_treated_as_format_repair(tmp_path):
    calls=[]
    async def consult(*args):
        calls.append(True)
        raise ValueError("Scientific role response identity mismatch")
    driver=AutonomousResearch(None,"",{},consult=consult)
    with pytest.raises(ValueError,match="identity mismatch"):
        await driver.structured("CB",{},SimpleNamespace(budget=SimpleNamespace(model_calls=12)),
                                SimpleNamespace(calls=1),tmp_path,parse_plan)
    assert len(calls)==1
