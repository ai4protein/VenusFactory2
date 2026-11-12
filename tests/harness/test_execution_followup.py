import json
from pathlib import Path

import pytest

from agent.harness.contracts import Profile
from agent.harness.execution_followup import execution_followup


def profile():
    return Profile.from_dict({"mode":"science_expert","harness":"expert-deepseek-harness",
        "model":{"provider":"dmx","model":"glm-4-flash","base_url":"https://www.dmxapi.cn/v1"},
        "budget":{"model_calls":12,"tool_calls":12,"wall_seconds":120,"output_tokens":4096,"thinking":"disabled"}})


def discovery_events():
    return [{"kind":"tool_usage","tool_calls":0,"discovery_calls":4},
            {"kind":"completed","text":"read_session_file\n{\"path\":\"inputs.csv\"}",
             "event_audit":{"passed":True}}]


def test_followup_uses_remaining_budget_and_does_not_execute_response_text():
    p=profile()
    request={"text":json.dumps({"objective":"Inspect supplied data"}),"control":"old"}
    result=execution_followup(request,p,discovery_events(),checkpoint=None,remaining_calls=5,remaining_seconds=30)
    assert result["budget"]["model_calls"]==5 and result["budget"]["wall_seconds"]==30
    context=json.loads(result["text"])
    assert context["objective"]=="Inspect supplied data"
    assert context["execution_followup"]["previous_response"].startswith("read_session_file")
    assert "structured MCP tool calls" in context["execution_followup"]["next_step"]
    assert "execution_followup" not in json.loads(request["text"])


@pytest.mark.parametrize("reason",["checkpoint","interaction","execution","checkpoints","bad_audit","calls","time"])
def test_followup_never_duplicates_work_or_overrides_a_user_question_or_budget(reason):
    events=discovery_events();checkpoint=None;calls=5;seconds=30
    if reason=="checkpoint":checkpoint={}
    if reason=="interaction":events.append({"kind":"interaction"})
    if reason=="execution":events[0]["tool_calls"]=profile().budget.tool_calls
    if reason=="checkpoints":events[0]["checkpoint_calls"]=2
    if reason=="bad_audit":events[-1]["event_audit"]["passed"]=False
    if reason=="calls":calls=2
    if reason=="time":seconds=9
    assert execution_followup({"text":"{}"},profile(),events,checkpoint=checkpoint,
                              remaining_calls=calls,remaining_seconds=seconds) is None


@pytest.mark.asyncio
async def test_scientist_performs_one_correction_then_reviews_the_actual_work(tmp_path,monkeypatch):
    from agent.harness import autonomous_research as module
    class Relay:
        calls=0;base_url="http://unused";token="fixture"
        def __init__(self,**kwargs):Path(kwargs["output"]).touch()
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
    monkeypatch.setattr(module,"EvidenceGateway",Relay)
    monkeypatch.setattr(module,"audit_requests",lambda _:{"passed":True,"model_calls":4})
    store=module.create(tmp_path/"goal",profile(),"Compute lengths of the supplied peptides",
                        dict(max_cycles=1,model_calls=12,tool_calls=12,wall_seconds=120))
    attempts=[]
    async def consult(role,context):
        if role=="PI":return "Compute the supplied peptide lengths."
        if role=="CB":return json.dumps({"summary":"Calculate lengths",
            "steps":[{"id":"compute","title":"Calculate lengths","acceptance":"Executed code saves lengths.json"}],
            "needs_input":[]})
        assert "report" not in context["execution"]  # SC sees measurements, not the executor endorsement.
        assert context["execution"]["step_checks"][0]["passed"]
        preview=next(row for row in context["artifact_previews"] if row["path"]=="lengths.json")
        assert json.loads(preview["content"])=={"lengths":[4,2]}
        return json.dumps({"decision":"ready_for_host_validation","summary":"Verified lengths","issues":[]})
    async def executor(request,service,events):
        attempts.append(request)
        if len(attempts)==1:
            events.extend(discovery_events())
            return
        assert "execution_followup" in json.loads(request["text"])
        await service.write_research_file("calculate.py","import json\njson.dump({'lengths':[4,2]},open('lengths.json','w'))\n")
        run=await service.run_research_python("calculate.py",10)
        assert run["exit_code"]==0
        await service.submit_research_checkpoint("Measured lengths",["compute"],["lengths.json"],[])
        events.extend([{"kind":"tool_usage","tool_calls":3},
                       {"kind":"completed","text":"Measured lengths saved","event_audit":{"passed":True}}])
    state=await module.AutonomousResearch(store,"fixture",{},consult=consult,executor=executor,tools=[]).run()
    assert len(attempts)==2 and len(state["cycles"])==1
    assert state["status"]=="completed"
    assert (store.root/"episodes/cycle_000/execution_retry/reason.json").is_file()
