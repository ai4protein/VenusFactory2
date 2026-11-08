"""Continue real unfinished work without resetting native execution budgets."""
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from agent.harness.contracts import Profile, Budget, ModelBinding
from agent.harness.execution_followup import execution_followup, execution_usage


def profile():
    return Profile.from_dict({"mode":"science_expert","harness":"expert-deepseek-harness",
        "model":{"provider":"dmx","model":"glm-4-flash","base_url":"https://www.dmxapi.cn/v1"},
        "budget":{"model_calls":12,"tool_calls":12,"wall_seconds":120,"output_tokens":4096,"thinking":"disabled"}})


@pytest.mark.parametrize("executed",[1,4,11])
def test_read_or_preparation_can_continue_with_only_remaining_allowances(executed):
    events=[{"kind":"tool_usage","tool_calls":executed,"discovery_calls":7,"checkpoint_calls":1},
            {"kind":"completed","text":"Data prepared; next write the model","event_audit":{"passed":True}}]
    request={"text":json.dumps({"objective":"Train a model"})}
    followup=execution_followup(request,profile(),events,checkpoint=None,remaining_calls=5,remaining_seconds=25)
    assert followup["budget"]["tool_calls"]==12-executed
    assert followup["research_allowances"]=={"discovery":17,"checkpoint":1}
    assert followup["budget"]["model_calls"]==5
    assert "do not repeat completed work" in json.loads(followup["text"])["execution_followup"]["next_step"]
    assert execution_followup(followup,profile(),events,checkpoint=None,remaining_calls=5,remaining_seconds=25) is None


def test_usage_sums_native_turns_and_keeps_an_interrupted_final_turn():
    events=[{"kind":"tool_usage","tool_calls":1,"discovery_calls":2},
            {"kind":"tool_usage","tool_calls":4,"discovery_calls":3},
            {"kind":"completed","tool_calls":4,"discovery_calls":3},
            {"kind":"tool_usage","tool_calls":2,"checkpoint_calls":1},
            {"kind":"completed","tool_calls":2,"checkpoint_calls":1},
            {"kind":"tool_usage","tool_calls":1}]
    assert execution_usage(events)=={"tool_calls":7,"discovery_calls":3,"checkpoint_calls":1}
    assert execution_usage([{"kind":"tool_call_start"},{"kind":"completed"},{"kind":"tool_call_start"}])["tool_calls"]==2


@pytest.mark.asyncio
@pytest.mark.parametrize("harness",["kimi-code","deepseek-harness","codex","claude-code"])
async def test_native_bridge_obeys_remaining_discovery_and_checkpoint_allowances(tmp_path,monkeypatch,harness):
    from agent.harness import web_native_worker as worker
    from agent.harness.research_workspace import ResearchWorkspace
    service=ResearchWorkspace(tmp_path/"work",tmp_path/"control"/"tools",{"steps":[]})
    events=[]
    async def adapter(ctx):
        connection=json.loads((ctx.task.capability/"connection.json").read_text())
        assert ctx.profile.budget.tool_calls==3  # two executions, zero lookups, one checkpoint
        async with httpx.AsyncClient(trust_env=False,timeout=20) as client:
            async def invoke(name,arguments):
                result=await client.post(connection["url"],headers={"Authorization":"Bearer "+connection["token"]},
                    json={"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":name,"arguments":arguments}})
                return result.json()["result"]
            denied=await invoke("search_tools",{})
            assert denied["isError"] and "Discovery budget reached" in denied["content"][0]["text"]
            result=await invoke("list_session_files",{})
            assert not result["isError"]
            saved=await invoke("submit_research_checkpoint",{"summary":"Partial work saved","completed_steps":[],"artifacts":[],"needs_input":[]})
            assert not saved["isError"]
        return {"status":"completed","text":"Partial work checkpointed"}
    monkeypatch.setattr(worker,"builtin_registry",lambda:SimpleNamespace(resolve=lambda *a:(SimpleNamespace(run=adapter),None)))
    p=Profile("science_agent",harness,ModelBinding("fixture","unused","http://127.0.0.1:1"),Budget(tool_calls=2))
    (tmp_path/"native").mkdir()
    await worker.run({"profile":p.public(),"workspace":str(service.workspace),"control":str(tmp_path/"native"),
        "session_id":"fixture","runtimes":{},"base_url":"http://127.0.0.1:1","token":"fixture","text":"Continue",
        "research_allowances":{"discovery":0,"checkpoint":1}},tools=service.tools(),event_sink=events.append)
    usage=execution_usage(events)
    assert usage=={"tool_calls":1,"discovery_calls":0,"checkpoint_calls":1}


@pytest.mark.asyncio
async def test_prepared_data_refreshed_in_same_iteration_without_repeating_preparation(tmp_path,monkeypatch):
    from agent.harness import autonomous_research as module
    class Relay:
        calls=0;base_url="http://unused";token="fixture"
        def __init__(self,**kwargs):Path(kwargs["output"]).touch()
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
    monkeypatch.setattr(module,"EvidenceGateway",Relay)
    monkeypatch.setattr(module,"audit_requests",lambda _:{"passed":True,"model_calls":5})
    store=module.create(tmp_path/"goal",profile(),"Develop a predictor from supplied measurements",
                        dict(max_cycles=1,model_calls=12,tool_calls=12,wall_seconds=120))
    data = "id,group,x,target\n" + "".join(f"{i},g{i},{i},{2*i+1}\n" for i in range(40))
    source = store.root / "workspace/inputs/data.csv"
    source.parent.mkdir()
    source.write_text(data)
    item = {"path": "inputs/data.csv", "name": "data.csv",
            "sha256": hashlib.sha256(data.encode()).hexdigest()}
    store.mutate(lambda state: state["progress"].update(inputs=[item]))
    attempts=[]
    async def consult(role,context):
        if role=="PI":return "Prepare data and train the predictor."
        if role=="CB":return json.dumps({"summary":"Train from measurements","deliverable":"model","needs_input":[],
            "steps":[{"id":"data","title":"Prepare measured data","acceptance":"Verified split","verification":"data"},
                     {"id":"fit","title":"Fit a model","acceptance":"Host-measured predictor","verification":"model"}]})
        return json.dumps({"decision":"needs_revision","summary":"Dataset prepared; fitting remains.","issues":[]})
    async def executor(request,service,events):
        context=json.loads(request["text"]);attempts.append(context)
        if len(attempts)==1:
            await service.read_research_file("inputs/data.csv")
            await service.prepare_research_dataset("inputs/data.csv","target",["x"],"group","Synthetic fixture","Independent groups",id_column="id")
            events.extend([{"kind":"tool_usage","tool_calls":2,"discovery_calls":1},
                           {"kind":"completed","text":"Data prepared; train next","event_audit":{"passed":True}}])
        else:
            assert request["budget"]["tool_calls"]==10
            assert context["model_development"]["prepared_dataset"]["status"]=="verified"
            assert context["model_development"]["selected_tools"]==["get_research_baseline","train_research_model"]
            assert any(r["tool"]=="prepare_research_dataset" for r in context["execution_receipts"])
            artifact=context["model_development"]["prepared_dataset"]["evaluation_file"]
            await service.write_research_file("README.md","Prepared a synthetic dataset; no model has been fitted.")
            await service.submit_research_checkpoint("Data prepared",["data"],[artifact,"README.md"],[])
            events.extend([{"kind":"tool_usage","tool_calls":1,"checkpoint_calls":1},
                           {"kind":"completed","text":"Prepared data saved","event_audit":{"passed":True}}])
    result=await module.AutonomousResearch(store,"fixture",{},consult=consult,executor=executor,tools=[]).run()
    assert len(attempts)==2 and len(result["cycles"])==1
    assert result["cycles"][0]["usage"]["tool_calls"]==3
    assert result["status"]!="completed"


@pytest.mark.asyncio
async def test_training_row_misuse_returns_original_error_and_exact_contract(tmp_path):
    from agent.harness.research_workspace import ResearchWorkspace
    service=ResearchWorkspace(tmp_path/"work",tmp_path/"control",{"steps":[]},
                              development_root=tmp_path/"development")
    def rows(start,count):
        return [{"id":str(i),"group":str(i),"inputs":{"x":i},"target":2*i+1} for i in range(start,start+count)]
    data={"task":"regression","provenance":"Synthetic software fixture","grouping":"Independent groups",
          "splits":{"train":rows(1,8),"validation":rows(9,3),"test":rows(12,3)}}
    await service.write_research_file("evaluation.json",json.dumps(data))
    await service.write_research_file("bad.py","def fit(training,seed):\n return {'x':[row['inputs'] for row in training]}\ndef predict(model,inputs):\n return {'value':0}\n")
    with pytest.raises(ValueError) as error:
        await service.train_research_model("fixture","bad.py","evaluation.json","Test input diagnostics")
    message=str(error.value)
    assert "TypeError" in message and "string indices" in message
    assert "rows=training['train']" in message and "encode sequence/categorical features" in message
    assert not service.development.records()


def describe_events():
    return [
        {"kind":"tool_call_start","tool_call_id":"desc","tool_name":"mcp__venusfactory__describe_tool",
         "args":{"name":"train_research_model"}},
        {"kind":"tool_result","tool_call_id":"desc","is_error":False,"output":"schema"},
        {"kind":"tool_call_start","tool_call_id":"read","tool_name":"mcp__venusfactory__read_session_file",
         "args":{"path":"data.csv"}},
        {"kind":"tool_result","tool_call_id":"read","is_error":False,"output":"data"},
        {"kind":"tool_usage","tool_calls":1,"discovery_calls":1},
        {"kind":"completed","text":"Ready to train","event_audit":{"passed":True}},
    ]


@pytest.mark.parametrize("deliverable", ["model","analysis"])
def test_read_only_continuation_targets_action_without_repeating_discovery(deliverable):
    request={"text":json.dumps({"plan":{"deliverable":deliverable}})}
    followup=execution_followup(request,profile(),describe_events(),checkpoint=None,
                                remaining_calls=5,remaining_seconds=30)
    context=json.loads(followup["text"])["execution_followup"]
    assert context["discovery_only"]
    assert context["described_tools"]==["train_research_model"]
    assert "Do not repeat unchanged tool descriptions" in context["next_step"]
    assert ("fit it" in context["next_step"]) is (deliverable=="model")
    assert followup["budget"]["tool_calls"]==11


@pytest.mark.parametrize("name,args", [
    ("write_session_file",{"path":"src/model.py","content":"# source"}),
    ("get_research_baseline",{"reference":"aac_ridge","output_file":"src/reference.py"}),
    ("call_tool",{"name":"get_research_baseline","arguments":{"reference":"aac_ridge","output_file":"src/reference.py"}}),
    ("train_research_model",{"source_file":"src/model.py"}),
])
def test_mutating_calls_are_never_classified_as_discovery_only(name,args):
    from agent.harness.execution_followup import discovery_only_observation
    events=describe_events()+[{"kind":"tool_call_start","tool_call_id":"action",
                               "tool_name":"mcp__venusfactory__"+name,"args":args}]
    assert discovery_only_observation(events) is None
