import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.harness import Profile
from agent.harness.autonomous_research import AutonomousResearch, create, parse_plan, verified_checkpoint
from agent.harness.research_workspace import ResearchWorkspace, run_python, safe_file, publish_proposals, run_capability, describe_capability

def profile():
    return Profile.from_dict({'mode':'science_expert','harness':'expert-deepseek-harness','model':{'provider':'dmx','model':'glm-4-flash','base_url':'https://www.dmxapi.cn/v1'},'budget':{'model_calls':12,'tool_calls':12,'wall_seconds':120,'output_tokens':4096,'thinking':'disabled'}})

def plan():
    return {'summary':'Evaluate peptide lengths','steps':[{'id':'compute','title':'Compute sequence lengths','acceptance':'Run code on the supplied sequences and save JSON'}],'needs_input':[]}

def test_plan_cannot_replace_the_target_steps_or_self_declare_success():
    first=parse_plan(json.dumps(plan()))
    changed=plan();changed['steps'][0]['id']='unrelated'
    with pytest.raises(ValueError):parse_plan(json.dumps(changed),first)
    changed=plan();changed['steps'][0]['status']='completed'
    # Echoed checkpoint metadata is accepted syntactically but never trusted.
    clean=parse_plan(json.dumps(changed))
    assert clean['steps'][0]['status']=='pending'
    assert clean['steps'][0]['evidence']==[]
    changed=plan();changed['steps'][0]['acceptance']='Just say done'
    assert parse_plan(json.dumps(changed),first)['steps'][0]['acceptance']==first['steps'][0]['acceptance']

def test_workspace_rejects_paths_outside_the_goal(tmp_path):
    outside=tmp_path/'secret';outside.write_text('private');work=tmp_path/'work';work.mkdir()
    (work/'link').symlink_to(outside)
    for path in ['../secret',str(outside),'link','.env']:
        with pytest.raises(ValueError):safe_file(work,path)

@pytest.mark.asyncio
async def test_code_replay_publication_and_fresh_agent_invocation(tmp_path):
    service=ResearchWorkspace(tmp_path/'work',tmp_path/'control',parse_plan(json.dumps(plan())))
    code='def run(inputs):\n    return {"lengths": [len(s) for s in inputs["sequences"]]}\n'
    await service.write_research_file('lengths.py',code)
    result=await service.propose_research_capability('Peptide lengths','Sequence length calculation','lengths.py',[],[{'sequences':['AAAA','GG']}])
    library=tmp_path/'library'
    assert not library.exists()  # Proposal is not a publication.
    with pytest.raises((ValueError,OSError)):
        publish_proposals(service.proposals,'goal-fixture',library)
    with pytest.raises((ValueError,OSError)):
        await run_capability(result['id'],{'sequences':['MKT','GGGGG']},library=library)


@pytest.mark.asyncio
async def test_python_has_no_host_credentials_or_network(tmp_path,monkeypatch):
    monkeypatch.setenv('PRIVATE_TEST_KEY','must-not-be-visible')
    work=tmp_path/'work';work.mkdir()
    (work/'probe.py').write_text('import os,socket,json\nassert "PRIVATE_TEST_KEY" not in os.environ\nassert not os.path.exists('+repr(str(tmp_path/'private'))+')\ns=socket.socket()\ns.settimeout(1)\ntry:\n s.connect(("1.1.1.1",443))\n raise AssertionError("network available")\nexcept OSError: pass\njson.dump({"isolated":True},open("result.json","w"))\n')
    (tmp_path/'private').write_text('private')
    result=await run_python(work,tmp_path/'runs','probe.py',10)
    assert result['exit_code']==0,result
    assert json.loads((work/'result.json').read_text())=={'isolated':True}

@pytest.mark.asyncio
@pytest.mark.parametrize("catalog_kind", ["empty", "categories"])
async def test_goal_executes_reviews_and_revises_until_evidence_passes(tmp_path,monkeypatch,catalog_kind):
    import agent.harness.autonomous_research as module
    class Relay:
        calls=0;base_url='http://unused';token='fixture'
        def __init__(self,**kwargs):Path(kwargs['output']).touch()
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
    monkeypatch.setattr(module,'EvidenceGateway',Relay)
    monkeypatch.setattr(module,'audit_requests',lambda path:{'passed':True,'model_calls':4})
    store=create(tmp_path/'goal',profile(),'Compute lengths for the supplied peptide sequences',dict(max_cycles=3,model_calls=36,tool_calls=36,wall_seconds=360))
    roles=[];attempts=[]
    async def consult(role,context):
        roles.append(role)
        if catalog_kind == "categories":
            assert {row["id"] for row in context["available_tools"]["categories"]} == {"compute", "databases"}
            assert context["available_tools"]["preview"] == []
        if role=='PI':return 'Preserve the supplied peptide sequences.'
        if role=='CB':
            proposed=plan();proposed['needs_input']=['Do you prefer a CSV or JSON report?']
            return json.dumps(proposed)
        return json.dumps({'decision':'needs_revision' if len(attempts)==1 else 'ready_for_host_validation','summary':'Repair the calculation' if len(attempts)==1 else 'Computed lengths checked','issues':[]})
    async def executor(request,service,events):
        attempts.append(1)
        await service.write_research_file('calculate.py','import json\njson.dump({"lengths":[4,2]},open("result.json","w"))\n')
        result=await service.run_research_python('calculate.py',10)
        assert result['exit_code']==0
        await service.submit_research_checkpoint('Calculated lengths',['compute'],['result.json'],[])
        events.extend([{'kind':'tool_call_start'},{'kind':'completed','text':'Ran calculate.py and wrote result.json','event_audit':{'passed':True}}])
    catalog = [] if catalog_kind == "empty" else [
        SimpleNamespace(name="read_record", description="Look up a record.", metadata={"discovery_category":"databases"}),
        SimpleNamespace(name="compute_feature", description="Compute a feature.", metadata={"discovery_category":"compute"})]
    runner=AutonomousResearch(store,'fixture',{},consult=consult,executor=executor,tools=catalog)
    state=await runner.run()
    assert state['status']=='completed',state
    assert len(state['cycles'])==2 and roles==['PI','CB','SC']*2
    assert state['cycles'][0]['status']=='needs_revision'
    (store.root/'workspace/result.json').write_text('{}')
    assert not runner.evaluate(state)['criteria']['research_plan_executed']['passed']

@pytest.mark.asyncio
async def test_missing_input_stops_without_substituting_a_demo(tmp_path,monkeypatch):
    import agent.harness.autonomous_research as module
    class Relay:
        calls=0;base_url='http://unused';token='fixture'
        def __init__(self,**kwargs):Path(kwargs['output']).touch()
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
    monkeypatch.setattr(module,'EvidenceGateway',Relay)
    monkeypatch.setattr(module,'audit_requests',lambda path:{'passed':True,'model_calls':4})
    async def consult(role,context):
        if role=='PI':return 'Need the target sequence.'
        if role=='SC':return json.dumps({'decision':'needs_revision','summary':'Target sequence is required','issues':['No user target provided']})
        value=plan();value['needs_input']=['Please provide the target protein sequence.'];return json.dumps(value)
    async def executor(request,service,events):
        assert request['text'] and not (await service.list_research_files())['files']
        await service.submit_research_checkpoint('Cannot identify the user target',[],[],['Please provide the target protein sequence.'])
        events.append({'kind':'completed','text':'No target was supplied; no sequence invented','event_audit':{'passed':True}})
    store=create(tmp_path/'goal',profile(),'Improve solubility of my protein',dict(max_cycles=3,model_calls=36,tool_calls=36,wall_seconds=360))
    state=await AutonomousResearch(store,'fixture',{},consult=consult,executor=executor,tools=[]).run()
    assert state['status']=='waiting_for_input'
    assert state['progress']['questions']==['Please provide the target protein sequence.']
    assert len(state['cycles'])==1 and not list((store.root/'workspace').iterdir())

def test_empty_checkpoint_cannot_complete_work(tmp_path):
    assert not verified_checkpoint(tmp_path,None,[])
    assert not verified_checkpoint(tmp_path,{'completed_steps':['compute'],'artifacts':[]},[{'tool':'write'}])


def test_material_driven_model_plan_cannot_be_downgraded_to_analysis():
    proposed=plan();proposed['deliverable']='model'
    previous=parse_plan(json.dumps(proposed),objective='Extend the supplied paper using these measurements')
    revised=plan();revised['deliverable']='analysis'
    assert parse_plan(json.dumps(revised),previous)['deliverable']=='model'
    assert parse_plan(json.dumps(revised),objective='用这些湿实验数据开发一个预测模型')['deliverable']=='model'
    proposed['deliverable']='tool'
    previous=parse_plan(json.dumps(proposed))
    assert parse_plan(json.dumps(revised),previous)['deliverable']=='tool'


@pytest.mark.parametrize('deliverable',['model','tool'])
def test_material_inspection_is_not_capability_delivery(tmp_path,deliverable):
    store=create(tmp_path/'goal',profile(),'Extend the supplied paper using these measurements',dict(max_cycles=3,model_calls=36,tool_calls=36,wall_seconds=360))
    proposed=plan();proposed['deliverable']=deliverable
    accepted=parse_plan(json.dumps(proposed));accepted['steps'][0]['status']='completed'
    state=store.read();state['progress'].update(plan=accepted,reviewed_episode=0)
    result=AutonomousResearch(store,'fixture',{},tools=[]).evaluate(state)
    assert result['delivery_pending']
    assert not any(row['passed'] for row in result['criteria'].values())


@pytest.mark.parametrize('deliverable,kind',[('model','method'),('tool','model')])
def test_delivery_kind_must_match_the_planned_research(tmp_path,deliverable,kind):
    store=create(tmp_path/'goal',profile(),'Extend the supplied paper using these measurements',dict(max_cycles=3,model_calls=36,tool_calls=36,wall_seconds=360))
    proposed=plan();proposed['deliverable']=deliverable
    accepted=parse_plan(json.dumps(proposed));accepted['steps'][0]['status']='completed'
    state=store.read();state['progress'].update(plan=accepted,reviewed_episode=0,
        capabilities=[{'id':'fixture'}],deliveries=[{'candidate_id':'fixture','kind':kind}])
    result=AutonomousResearch(store,'fixture',{},tools=[]).evaluate(state)
    assert result['delivery_pending']
    assert not any(row['passed'] for row in result['criteria'].values())


@pytest.mark.asyncio
async def test_native_path_adapter_supports_existing_files_and_live_python_progress(tmp_path):
    import asyncio
    from agent.harness.web_native_worker import prepare_tool_arguments
    service = ResearchWorkspace(tmp_path / "work", tmp_path / "control", parse_plan(json.dumps(plan())))
    tools = {t.name: t for t in service.tools()}
    async def invoke(name, args):
        tool = tools[name]
        prepared = prepare_tool_arguments(name, tool, args, service.workspace, lambda value: value)
        return await tool.ainvoke(prepared)
    await invoke("write_session_file", {"path": "helper.py", "content": "value=1\n"})
    await invoke("write_session_file", {"path": "helper.py", "content": "value=2\n"})
    assert (await invoke("read_session_file", {"path": "helper.py"}))["text"] == "value=2\n"
    await invoke("write_session_file", {"path": "main.py", "content": 'from helper import value\nimport json,time\nassert value==2\nprint(json.dumps({"event":"progress","step":1,"total":2,"metrics":{"loss":0.5}}),flush=True)\ntime.sleep(2)\nprint(json.dumps({"event":"progress","step":2,"total":2}),flush=True)\n'})
    pending = asyncio.create_task(invoke("run_session_python", {"path": "main.py", "timeout_seconds": 10}))
    observed = False
    while not pending.done():
        await asyncio.sleep(0.1)
        path = service.control / "execution.json"
        if path.exists():
            progress = json.loads(path.read_text())
            if progress.get("status") == "running" and progress.get("progress", {}).get("step") == 1:
                observed = True
    result = await pending
    assert result["exit_code"] == 0, result
    assert observed, "Progress must be observable before process completion"
    assert json.loads((service.control / "execution.json").read_text())["progress"]["step"] == 2
    with pytest.raises(ValueError):
        await invoke("read_session_file", {"path": "../private"})


def test_plan_accepts_a_single_json_object_with_prose_but_rejects_ambiguity():
    raw=json.dumps(plan())
    assert parse_plan("Here is the research plan:\n```json\n"+raw+"\n```\nReady to execute.")["steps"][0]["id"]=="compute"
    with pytest.raises(ValueError):parse_plan(raw+"\nAlternative:\n"+raw)
    with pytest.raises(ValueError):parse_plan("[]")

@pytest.mark.asyncio
async def test_planner_format_repair_does_not_spend_a_research_iteration(tmp_path):
    calls=[]
    async def consult(role,context):
        calls.append(context)
        return '# Plan\nCompute sequence lengths' if len(calls)==1 else json.dumps(plan())
    runner=AutonomousResearch(None,'fixture',{},consult=consult,tools=[])
    value=await runner.structured('CB',{'objective':'Compute supplied lengths'},profile(),SimpleNamespace(calls=1),tmp_path,parse_plan)
    assert value['steps'][0]['id']=='compute' and len(calls)==2
    assert calls[1]['objective']==calls[0]['objective']
    assert 'format_repair' in calls[1]

@pytest.mark.asyncio
async def test_format_repair_is_bounded_and_does_not_fabricate_a_plan(tmp_path):
    calls=[]
    async def consult(role,context):calls.append(1);return 'not a plan'
    runner=AutonomousResearch(None,'fixture',{},consult=consult,tools=[])
    with pytest.raises(ValueError):await runner.structured('CB',{},profile(),SimpleNamespace(calls=1),tmp_path,parse_plan)
    assert len(calls)==2
