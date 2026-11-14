"""Unattended confirmations never create human input or expand execution authority."""
import asyncio
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from agent.harness.autonomous_research import AutonomousResearch, create
from agent.harness.contracts import Budget, ModelBinding, Profile
from agent.harness.interactions import interaction_tools
from agent.harness.unattended_research import POLICY
from test_autonomous_research import profile, plan
from test_research_interaction import task_at

LIMITS = dict(max_cycles=2, model_calls=24, tool_calls=24, wall_seconds=240)


def test_policy_is_frozen_in_goal_not_inferred_from_research_text(tmp_path):
    ordinary = create(tmp_path/'ordinary', profile(), POLICY, LIMITS)
    assert not AutonomousResearch(ordinary, 'fixture', {}, tools=[]).unattended
    store = create(tmp_path/'auto', profile(), 'Count supplied records', LIMITS, unattended=True)
    state = store.read()
    assert state['definition']['admission'] == {'execution_policy': POLICY}
    assert AutonomousResearch(store, 'fixture', {}, tools=[]).unattended
    state['definition']['admission']['execution_policy'] = 'unknown'
    (store.root/'goal.json').write_text(json.dumps(state))
    with pytest.raises(ValueError, match='Goal definition changed'):
        _ = AutonomousResearch(store, 'fixture', {}, tools=[]).unattended
    with pytest.raises(ValueError, match='boolean'):
        create(tmp_path/'bad', profile(), 'test', LIMITS, unattended='true')


@pytest.mark.asyncio
async def test_confirmation_is_policy_audit_not_fake_human_answer():
    pending, events = [], []
    tools = {t.name:t for t in interaction_tools(pending.append, unattended=True, on_automatic=events.append)}
    assert set(tools) == {'request_confirmation'}
    result = await tools['request_confirmation'].ainvoke({'title':'Run the next trial', 'message':'Continue the admitted plan'})
    assert result['status'] == 'approved' and result['human_response'] is False
    assert events[0]['kind'] == 'automatic_approval' and events[0]['policy'] == POLICY
    assert events[0]['decision'] == 'approve' and not pending
    with pytest.raises(ValueError, match='audit sink'):
        interaction_tools(pending.append, unattended=True)
    ordinary = {t.name:t for t in interaction_tools(pending.append)}
    result = await ordinary['request_confirmation'].ainvoke({'title':'Question', 'message':'Confirm'})
    assert result['status'] == 'waiting_for_input' and pending


def test_native_contract_stays_web_scoped_and_preserves_budget(tmp_path):
    from agent.harness.adapters import task_message
    original = task_at(tmp_path)
    task = replace(original, interaction='research_unattended')
    assert task.phase == original.phase and task.tools == original.tools
    assert task.binding()['interaction'] == 'research_unattended'
    assert 'pre-authorized' in task_message(task) and 'independent SC/Review' in task_message(task)
    for change in ({'phase':'develop'}, {'mcp_entry':'generated_tools.py'}, {'tools':frozenset()}):
        with pytest.raises(ValueError): replace(task, **change)


@pytest.mark.asyncio
@pytest.mark.parametrize('harness', ['kimi-code','deepseek-harness','codex','claude-code'])
async def test_all_native_bridges_continue_after_auto_confirmation(tmp_path,monkeypatch,harness):
    from agent.harness import web_native_worker as worker
    from agent.harness.research_workspace import ResearchWorkspace
    work,control=tmp_path/'work',tmp_path/'control'
    work.mkdir();control.mkdir()
    service=ResearchWorkspace(work,control/'tools',{'steps':[]})
    events=[]
    async def adapter(ctx):
        assert ctx.task.interaction_mode=='research_unattended'
        assert 'ask_user' not in ctx.task.tools
        connection=json.loads((ctx.task.capability/'connection.json').read_text())
        trace=ctx.task.output/'events.jsonl';trace.parent.mkdir(parents=True,exist_ok=True);trace.touch()
        async with httpx.AsyncClient(trust_env=False,timeout=5,headers={'Authorization':'Bearer '+connection['token']}) as client:
            async def rpc(name,args):
                response=await client.post(connection['url'],json={'jsonrpc':'2.0','id':1,'method':'tools/call','params':{'name':name,'arguments':args}})
                return response.json()
            for index,(name,args) in enumerate([
                ('request_confirmation',{'title':'Continue', 'message':'Write the analysis file'}),
                ('write_session_file',{'path':'result.txt','content':'recorded after automatic approval'}),
            ]):
                response=await rpc(name,args)
                assert not response.get('error') and not response['result'].get('isError'),response
                with trace.open('a') as out:
                    out.write(json.dumps({'kind':'tool_call_start','tool_call_id':str(index),'tool_name':'mcp__venusfactory__'+name,'args':args})+'\n')
                    out.write(json.dumps({'kind':'tool_result','tool_call_id':str(index),'is_error':False,'output':json.dumps(response['result'])})+'\n')
            denied=await rpc('Bash',{'command':'pwd'})
            assert 'error' in denied
            denied=await rpc('ask_user',{'questions':[{'id':'q','question':'Continue?'}]})
            assert 'error' in denied
        return {'status':'completed','text':'Saved the result','tool_calls':2}
    monkeypatch.setattr(worker,'builtin_registry',lambda:SimpleNamespace(resolve=lambda *a:(SimpleNamespace(run=adapter),None)))
    p=Profile('science_agent',harness,ModelBinding('fixture','unused','http://127.0.0.1:1'),Budget(tool_calls=3))
    await asyncio.wait_for(worker.run({'profile':p.public(),'workspace':str(work),'control':str(control),
        'session_id':'auto','runtimes':{},'base_url':'http://127.0.0.1:1','token':'fixture','text':'Count records',
        'execution_policy':POLICY,'audit_required':True}, tools=service.tools(),event_sink=events.append),20)
    assert (work/'result.txt').read_text()=='recorded after automatic approval'
    assert len([e for e in events if e['kind']=='automatic_approval'])==1
    assert not any(e['kind']=='interaction' for e in events)
    assert next(e for e in events if e['kind']=='completed')['event_audit']['passed']
    assert not (control/'native/capability/connection.json').exists()


@pytest.mark.asyncio
async def test_host_policy_cannot_enable_unattended_in_agent_chat(tmp_path):
    from agent.harness.web_native_worker import run
    with pytest.raises(ValueError,match='audited Scientist'):
        await run({'workspace':str(tmp_path),'control':str(tmp_path/'control'),'execution_policy':POLICY},tools=[])


@pytest.mark.asyncio
async def test_missing_scientific_input_is_retained_without_fake_success(tmp_path,monkeypatch):
    import agent.harness.autonomous_research as module
    class Relay:
        calls=0;base_url='http://unused';token='fixture'
        def __init__(self,**kw):Path(kw['output']).touch()
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
    monkeypatch.setattr(module,'EvidenceGateway',Relay)
    monkeypatch.setattr(module,'audit_requests',lambda path:{'passed':True,'model_calls':3})
    async def consult(role,context):
        if role=='PI':return 'Inspect the supplied records.'
        assert role=='CB' # Missing native input must not masquerade as SC acceptance.
        return json.dumps(plan())
    async def executor(request,service,events):
        assert request['execution_policy']==POLICY
        assert 'ask_user' not in json.loads(request['text'])['tools']
        events.append({'kind':'research_input_required','questions':['Missing the actual input file'],'research_complete':False})
    store=create(tmp_path/'goal',profile(),'Count the supplied records',LIMITS,unattended=True)
    state=await AutonomousResearch(store,'fixture',{},consult=consult,executor=executor,tools=[]).run()
    assert state['status']=='waiting_for_input' and len(state['cycles'])==1
    assert state['progress']['questions']==['Missing the actual input file']
    assert not state['progress']['research_completed']
    assert not state['progress'].get('pending_interaction')


@pytest.mark.asyncio
async def test_native_question_is_scientific_blocker_not_runtime_crash(tmp_path):
    from agent.rsi.engine.runtime import run_task
    # Reuse the Kimi protocol fixture without depending on pytest collection order.
    import importlib.util
    spec = importlib.util.spec_from_file_location("unattended_kimi_fixture", Path(__file__).parents[1]/"rsi/test_runtime.py")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    FakeClient = fixture.FakeClient
    class QuestionClient(FakeClient):
        async def list_pending_questions(self,sid):
            await asyncio.sleep(.01)
            return [{'question':'Missing measured data'}]
    client=QuestionClient([],stall=True)
    result=await run_task(client,prompt='task',system_prompt='policy',model='m',output_dir=tmp_path,interaction='research_unattended')
    assert result['status']=='needs_input' and client.aborted
    assert result['questions']==[{'question':'Missing measured data'}]


@pytest.mark.asyncio
async def test_raw_native_missing_input_preserves_binding_and_cleanup(tmp_path,monkeypatch):
    from agent.harness import web_native_worker as worker
    from agent.harness.research_workspace import ResearchWorkspace
    work,control=tmp_path/'work',tmp_path/'control'
    work.mkdir();control.mkdir()
    service=ResearchWorkspace(work,control/'tools',{'steps':[]})
    events,bindings=[],[]
    async def adapter(ctx):
        return {'status':'needs_input','session_id':'native-missing','questions':[{'question':'Supply the actual measurements'}]}
    monkeypatch.setattr(worker,'builtin_registry',lambda:SimpleNamespace(resolve=lambda *a:(SimpleNamespace(run=adapter),None)))
    p=Profile('science_agent','kimi-code',ModelBinding('fixture','unused','http://127.0.0.1:1'),Budget(tool_calls=3))
    await worker.run({'profile':p.public(),'workspace':str(work),'control':str(control),
        'session_id':'auto','runtimes':{},'base_url':'http://127.0.0.1:1','token':'fixture','text':'Count records',
        'execution_policy':POLICY,'audit_required':True}, tools=service.tools(),event_sink=events.append,
        gateway_calls=lambda:1,native_observer=bindings.append)
    assert any(e['kind']=='research_input_required' and e['research_complete'] is False for e in events)
    assert not any(e['kind'] in {'interaction','completed'} for e in events)
    assert not (control/'native/capability/connection.json').exists()
    result=json.loads((control/'native/run/native_result.json').read_text())
    assert result['status']=='needs_input'
    binding=json.loads((control/'native/run/native_binding.json').read_text())
    from agent.harness.contracts import digest
    assert binding['native_result_sha256']==digest(result)
    assert binding['session_id']=='native-missing' and bindings==[binding]


def test_unattended_execution_followup_keeps_policy_and_records_missing_input():
    from agent.harness.execution_followup import execution_followup
    p=profile()
    request={'text':json.dumps({'objective':'Count actual records'}),'execution_policy':POLICY}
    retry=execution_followup(request,p,[{'kind':'completed','tool_calls':1,'event_audit':{'passed':True}}],
        checkpoint=None,remaining_calls=8,remaining_seconds=90)
    assert retry['execution_policy']==POLICY
    assert 'ask_user' not in json.loads(retry['text'])['execution_followup']['next_step']
    assert 'checkpoint needs_input' in json.loads(retry['text'])['execution_followup']['next_step']


@pytest.mark.parametrize('field,value', [('execution_policy',POLICY), ('unattended',True), ('admission',{'execution_policy':POLICY})])
def test_browser_creation_rejects_host_only_execution_policy(field,value):
    from backend.web_v2.chat_api.science_goals import CreateGoal
    from pydantic import ValidationError
    body={'request_id':'fixture-123','objective':'Count the supplied records'}
    assert CreateGoal.model_validate(body).objective==body['objective']
    with pytest.raises(ValidationError) as rejected:
        CreateGoal.model_validate({**body,field:value})
    assert any(error['loc']==(field,) and error['type']=='extra_forbidden' for error in rejected.value.errors())
