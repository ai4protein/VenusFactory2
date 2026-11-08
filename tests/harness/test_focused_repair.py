import hashlib
import json
from pathlib import Path
import pytest

from agent.harness.research_execution_feedback import feedback_state, repair_source_context

SOURCE = "def fit(training, seed):\n    pass\ndef predict(model, inputs):\n    return {'value': 0.0}\n"


def failure(path="src/model.py", failed=True, wrapped=False):
    name="mcp__venusfactory__train_research_model"
    args={"source_file":path}
    if wrapped:
        name="mcp__venusfactory__call_tool"
        args={"name":"train_research_model","arguments":args}
    return [{"kind":"tool_call_start","tool_call_id":"test","tool_name":name,"args":args},
            {"kind":"tool_result","tool_call_id":"test","is_error":failed,
             "output":"ValueError: fit is an unfinished placeholder" if failed else "{}"}]


@pytest.mark.parametrize("wrapped",[False,True])
def test_current_source_is_disclosed_only_for_its_unresolved_failure(tmp_path, wrapped):
    (tmp_path/"src").mkdir()
    (tmp_path/"src/model.py").write_text(SOURCE)
    (tmp_path/"unrelated.py").write_text("PRIVATE_UNRELATED_CONTENT")
    state=feedback_state(failure(wrapped=wrapped),tmp_path)
    context=repair_source_context(tmp_path,state)
    assert context["path"]=="src/model.py" and context["code"]==SOURCE
    assert context["sha256"]==hashlib.sha256(SOURCE.encode()).hexdigest()
    assert "PRIVATE_UNRELATED_CONTENT" not in json.dumps(context)
    assert not context["redacted"]
    resolved=feedback_state(failure(failed=False),tmp_path,state)
    assert repair_source_context(tmp_path,resolved) is None
    assert repair_source_context(tmp_path,{"errors":state["errors"]}) is None


@pytest.mark.parametrize("path",["../other.py","/tmp/other.py",".secret.py","src/.hidden.py","src/../other.py","x.py/../../other.py"])
def test_invalid_source_arguments_are_never_read_or_retained(tmp_path,path):
    state=feedback_state(failure(path),tmp_path)
    assert "source_file" not in state["pending"][0]
    assert repair_source_context(tmp_path,state) is None


@pytest.mark.parametrize("kind",["symlink","oversized","non_utf8","missing"])
def test_unsafe_or_unbounded_sources_stay_out_of_context(tmp_path,kind):
    (tmp_path/"src").mkdir();path=tmp_path/"src/model.py"
    if kind=="symlink":
        outside=tmp_path.parent/"private.py";outside.write_text("PRIVATE_EXTERNAL_CONTENT")
        path.symlink_to(outside)
    elif kind=="oversized":path.write_text("x"*12001)
    elif kind=="non_utf8":path.write_bytes(bytes([255,254]))
    state=feedback_state(failure(),tmp_path)
    assert repair_source_context(tmp_path,state) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("within_iteration",[False,True])
async def test_repair_context_reaches_executor_after_failed_fit_without_budget_reset(tmp_path,monkeypatch,within_iteration):
    from agent.harness import autonomous_research as module
    from agent.harness.contracts import Profile
    class Relay:
        calls=0;base_url="http://unused";token="fixture"
        def __init__(self,**kwargs):Path(kwargs["output"]).touch()
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
    monkeypatch.setattr(module,"EvidenceGateway",Relay)
    monkeypatch.setattr(module,"audit_requests",lambda _ : {"passed":True,"model_calls":5})
    profile=Profile.from_dict({"mode":"science_expert","harness":"expert-deepseek-harness",
        "model":{"provider":"dmx","model":"glm-4-flash","base_url":"https://www.dmxapi.cn/v1"},
        "budget":{"model_calls":12,"tool_calls":12,"wall_seconds":120,"output_tokens":4096,"thinking":"disabled"}})
    cycles=1 if within_iteration else 2
    store=module.create(tmp_path/"goal",profile,"Develop a predictor from supplied measurements",
                        dict(max_cycles=cycles,model_calls=12*cycles,tool_calls=12*cycles,wall_seconds=120*cycles))
    data = "id,group,x,target\n" + "".join(f"{i},g{i},{i},{2*i+1}\n" for i in range(40))
    source = store.root / "workspace/inputs/data.csv"
    source.parent.mkdir()
    source.write_text(data)
    item = {"path": "inputs/data.csv", "name": "data.csv",
            "sha256": hashlib.sha256(data.encode()).hexdigest()}
    store.mutate(lambda state: state["progress"].update(inputs=[item]))
    attempts=[];planning=[]
    async def consult(role,context):
        if role in ["PI","CB"]:
            planning.append(json.loads(json.dumps(context)))
        if role=="PI":return "Fit and evaluate a predictor."
        if role=="CB":return json.dumps({"summary":"Develop a model","deliverable":"model","needs_input":[],
            "steps":[{"id":"fit","title":"Fit a predictor","acceptance":"Measured fitted model","verification":"model"}]})
        return json.dumps({"decision":"needs_revision","summary":"Training is incomplete","issues":[]})
    async def executor(request,service,events):
        context=json.loads(request["text"]);attempts.append(context)
        if len(attempts)==1:
            await service.read_research_file("inputs/data.csv")
            prepared=await service.prepare_research_dataset(
                "inputs/data.csv","target",["x"],"group","Synthetic fixture","Independent fixture groups")
            await service.write_research_file("src/model.py",SOURCE)
            try:
                await service.train_research_model("Fixture","src/model.py",prepared["evaluation_file"],"Interface fixture")
            except ValueError as exc:
                error=str(exc)
            else:raise AssertionError("Unimplemented source must not train")
            events.extend(failure())
            events[-1]["output"]=error
            if not within_iteration:
                await service.submit_research_checkpoint("Partial source saved",[],["src/model.py"],[])
            events.extend([{"kind":"tool_usage","tool_calls":4,"checkpoint_calls":0 if within_iteration else 1},
                {"kind":"completed","text":"Fitting requires repair","event_audit":{"passed":True}}])
        else:
            repair=context["model_development"]["repair_source"]
            assert repair["code"]==SOURCE
            assert "unfinished placeholder" in repair["error"]
            assert request["budget"]["tool_calls"]==(8 if within_iteration else 12)
            await service.write_research_file("README.md","This fixture has no trained model.")
            await service.submit_research_checkpoint("Partial work saved",[],["README.md"],[])
            events.extend([{"kind":"tool_usage","tool_calls":1,"checkpoint_calls":1},
                {"kind":"completed","text":"Incomplete model retained","event_audit":{"passed":True}}])
    state=await module.AutonomousResearch(store,"fixture",{},consult=consult,executor=executor,tools=[]).run()
    assert len(attempts)==2
    assert all("repair_source" not in x["model_development"] for x in planning)
    assert state["status"]!="completed"
    assert len(state["cycles"])==cycles
    assert sum(x["usage"]["tool_calls"] for x in state["cycles"])==5


@pytest.mark.parametrize("separator",["","\n"," \n "])
def test_stale_host_budget_footer_is_removed_even_when_harness_concatenates_it(tmp_path,separator):
    events=failure()
    events[-1]["output"]=("ValueError: fit is incomplete"+separator+
        "Execution budget: 6 tool calls remain; 13 read-only discovery calls remain. "
        "Save a checkpoint when the current step is ready. "
        "Checkpoint saves have a separate allowance: 2 remain.")
    state=feedback_state(events,tmp_path)
    assert state["errors"][0]["error"]=="ValueError: fit is incomplete"
    events[-1]["output"]="DisclosureError: Execution budget reached."
    assert feedback_state(events,tmp_path)["errors"][0]["error"]==events[-1]["output"]


def test_metadata_only_failure_stays_a_failure_without_stale_budget_text(tmp_path):
    events=failure()
    events[-1]["output"]=("Execution budget: 6 tool calls remain; 13 read-only discovery calls remain. "
        "Save a checkpoint when the current step is ready. "
        "Checkpoint saves have a separate allowance: 2 remain.")
    assert feedback_state(events,tmp_path)["errors"][0]["error"]=="Tool failed"


def test_credentials_in_current_source_are_redacted_with_original_hash_retained(tmp_path):
    secret=("s"+"k-or-v1-")+("a"*64)
    source="token = "+repr(secret)+"\n"+SOURCE
    (tmp_path/"src").mkdir()
    (tmp_path/"src/model.py").write_text(source)
    context=repair_source_context(tmp_path,feedback_state(failure(),tmp_path))
    assert secret not in context["code"]
    assert context["redacted"] is True
    assert context["sha256"]==hashlib.sha256(source.encode()).hexdigest()
