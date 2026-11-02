import asyncio
import json
from types import SimpleNamespace

import pytest

from agent.harness.interactions import AskInput, Interaction, InteractionResponse, response_text, interaction_tools


def question(**overrides):
    return Interaction(id='a'*32, kind='question', questions=[{'id':'target','question':'Which protein?', **overrides}], created_at=1).model_dump()


def response(**overrides):
    return InteractionResponse(interaction_id='a'*32, decision='answer', **overrides)


def test_text_and_multiple_choices_and_explicit_decisions():
    gate=question()
    assert 'PETase' in response_text(gate,response(answers=[{'question_id':'target','text':'PETase'}]))
    gate=question(options=['Solubility','Stability'],multiple=True)
    text=response_text(gate,response(answers=[{'question_id':'target','selected':[0,1]}]))
    assert 'Solubility' in text and 'Stability' in text
    with pytest.raises(ValueError):response_text(gate,response(answers=[]))
    with pytest.raises(ValueError):response_text(gate,response(answers=[{'question_id':'target','selected':[12]}]))
    with pytest.raises(ValueError):response_text(question(options=['A','B']),response(answers=[{'question_id':'target','selected':[0,1]}]))
    with pytest.raises(ValueError):response_text(question(allow_text=False,options=['A']),response(answers=[{'question_id':'target','text':'B'}]))
    with pytest.raises(ValueError):AskInput(questions=[{'id':'q','question':'A'},{'id':'q','question':'B'}])
    with pytest.raises(ValueError):InteractionResponse(interaction_id='a'*32)  # No default approval.
    confirmation=Interaction(id='a'*32,kind='plan',title='Proposed study',message='Train on supplied public data',created_at=1).model_dump()
    for decision in ['approve','reject','cancel','revise']:
        answer=InteractionResponse(interaction_id='a'*32,decision=decision,feedback='Use group split')
        assert f'"decision": "{decision}"' in response_text(confirmation,answer)
    with pytest.raises(ValueError):response_text(confirmation,response(answers=[]))
    with pytest.raises(ValueError):response_text(question(),InteractionResponse(interaction_id='b'*32,decision='cancel'))


def test_question_tool_is_not_displayed_as_cancelled_work():
    from agent.harness.interactions import settle_interaction_tools
    tools=[{'turn_id':'now','status':'running','tool_name':'mcp__venusfactory__ask_user'},
           {'turn_id':'now','status':'running','tool_name':'train_model'},
           {'turn_id':'old','status':'running','tool_name':'ask_user'}]
    settle_interaction_tools(tools,'now',question())
    assert tools[0]['status']=='ok' and tools[0]['output']['status']=='waiting_for_input'
    assert tools[1]['status']==tools[2]['status']=='running'


@pytest.mark.parametrize('harness',['kimi-code','codex','claude-code','deepseek-harness'])
@pytest.mark.asyncio
async def test_shared_mcp_pauses_every_native_adapter_and_blocks_later_actions(harness,tmp_path,monkeypatch):
    from aiohttp import ClientSession
    from langchain_core.tools import StructuredTool
    from agent.harness import web_native_worker as worker
    from agent.harness.contracts import Profile,Budget,ModelBinding
    from mcp_tools.scheduler import catalog
    from mcp_tools.tools import runtime_tool_policy
    import agent.harness.session_sandbox as sandbox
    workspace=tmp_path/'workspace';workspace.mkdir()
    control=tmp_path/'control';control.mkdir()
    cancelled=[];ran=[];events=[]
    async def analyze():
        ran.append(True)
        return 'should not execute'
    tool=StructuredTool.from_function(coroutine=analyze,name='analyze_fixture',description='Test fixture')
    async def fake_run(ctx):
        try:
            config=json.loads((control/'native/capability/connection.json').read_text())
            async with ClientSession(headers={'Authorization':'Bearer '+config['token']}) as client:
                async def rpc(method,params):
                    async with client.post(config['url'],json={'jsonrpc':'2.0','id':1,'method':method,'params':params}) as resp:return await resp.json()
                listing=await rpc('tools/list',{})
                names={t['name'] for t in listing['result']['tools']}
                assert {'ask_user','request_confirmation'} <= names
                result=await rpc('tools/call',{'name':'ask_user','arguments':{'questions':[{'id':'target','question':'Which protein?'}]}})
                assert 'waiting_for_input' in json.dumps(result)
                denied=await rpc('tools/call',{'name':'analyze_fixture','arguments':{}})
                assert 'error' in denied
                await asyncio.Event().wait()
        finally:cancelled.append(True)
    monkeypatch.setattr(worker,'builtin_registry',lambda:SimpleNamespace(resolve=lambda *a:(SimpleNamespace(run=fake_run),None)))
    monkeypatch.setattr(catalog,'native_tools',lambda:{})
    monkeypatch.setattr(sandbox,'check_workspace',lambda *a:None)
    monkeypatch.setattr(runtime_tool_policy,'assert_tool_allowed',lambda *a:None)
    profile=Profile('science_agent',harness,ModelBinding('dmx','glm-4-flash','https://www.dmxapi.cn/v1',protocol='responses' if harness=='codex' else 'chat_completions'),Budget())
    request={'profile':profile.public(),'workspace':str(workspace),'control':str(control),'session_id':'fixture','runtimes':{},'base_url':'http://localhost','token':'fixture','text':'Ask which protein'}
    await asyncio.wait_for(worker.run(request,tools=[tool],event_sink=events.append),20)
    assert cancelled and not ran
    gate=next(e['interaction'] for e in events if e['kind']=='interaction')
    assert gate['questions'][0]['question']=='Which protein?'
    assert not any(e['kind']=='completed' for e in events)
    assert not (control/'native/capability/connection.json').exists()


@pytest.mark.asyncio
async def test_pending_question_and_recorded_work_survive_store_restart(tmp_path):
    from agent.session_store import SessionStore,SqliteBackend
    from backend.web_v2.agent_conversation import continuation_messages
    path=str(tmp_path/'session.sqlite3')
    store=SessionStore(SqliteBackend(path),lambda x:x)
    gate=question()
    await store.create({'session_id':'s','status':'waiting_for_input','pending_interaction':gate,
        'agent_messages':[], 'interaction_history':[], 'agent_pending_turn':{'text':'Analyze protein','interaction':gate,'observations':[{'role':'assistant','content':'Saved result.csv'}]}})
    restored=await SessionStore(SqliteBackend(path),lambda x:x).get('s')
    assert restored['pending_interaction']==gate
    messages=continuation_messages(restored,harness='codex')
    assert 'Saved result.csv' in json.dumps(messages) and 'Execution paused' in json.dumps(messages)


def test_scientist_answer_is_idempotent_and_preserves_research(tmp_path):
    from agent.science_goal import GoalStore,GoalSpec
    from agent.harness import Profile
    from agent.harness.research_updates import queue_revision,apply_revisions
    profile=Profile.from_dict({'mode':'science_expert','harness':'expert-kimi-code','model':{'provider':'dmx','model':'glm-4-flash','base_url':'https://www.dmxapi.cn/v1'},'budget':{'model_calls':12,'tool_calls':12,'wall_seconds':120,'output_tokens':4096}})
    store=GoalStore.create(tmp_path/'goal',GoalSpec('Develop a model',('accepted',),max_cycles=3,model_calls=36,tool_calls=36,wall_seconds=360),profile)
    plan={'steps':[{'id':'data','status':'completed','evidence':[{'path':'data.csv'}]}]}
    store.mutate(lambda s:s.update(status='waiting_for_input',progress={'plan':plan,'pending_interaction':question(),'reviewed_episode':0}))
    answer=response(answers=[{'question_id':'target','text':'PETase'}]).model_dump()
    with pytest.raises(ValueError):queue_revision(store,'Bypass pending interaction')
    queue_revision(store,interaction=answer);queue_revision(store,interaction=answer)
    assert len(store.read()['pending_research_updates'])==1
    state=apply_revisions(store)
    assert state['progress']['plan']==plan and 'pending_interaction' not in state['progress']
    assert 'PETase' in state['progress']['user_updates'][0]
    assert 'reviewed_episode' not in state['progress']
    changed={**answer,'decision':'cancel','answers':[]}
    with pytest.raises(ValueError):queue_revision(store,interaction=changed)


@pytest.mark.asyncio
async def test_http_answer_enqueues_once_and_enforces_session_access(tmp_path,monkeypatch,durable_chat_driver):
    import httpx
    from fastapi import FastAPI,HTTPException
    from agent.session_store import SessionStore,SqliteBackend
    from backend.web_v2.chat_api import messages,_shared
    from backend.web_v2 import agent_execution
    from backend.web_v2.job_queue import JobQueue
    from backend.web_v2.science_goals import service
    from agent.harness.contracts import Profile,Budget,ModelBinding
    monkeypatch.setenv('WEBUI_V2_MODE','local')
    monkeypatch.setattr(service,'has_running_goal',lambda sid:False)
    profile=Profile('science_agent','kimi-code',ModelBinding('dmx','glm-4-flash','https://www.dmxapi.cn/v1'),Budget())
    monkeypatch.setattr(agent_execution,'admit',lambda *a:({'profile':profile.public()},'fixture'))
    store=SessionStore(SqliteBackend(str(tmp_path/'sessions.db')),lambda x:x)
    calls=[]
    async def stream(state,text,paths,*,retry=False,interaction_response=None):
        assert interaction_response['decision']=='answer'
        assert 'PETase' in text and state['pending_interaction']['id']=='a'*32
        calls.append(text);state.pop('pending_interaction');state['status']='completed'
        yield 'event: state\ndata: {"status":"completed"}\n\n'
    durable_chat_driver(store,stream)
    monkeypatch.setattr(_shared,'_session_store',store)
    await store.create({'session_id':'interaction-test','chat_mode':'science_agent','pending_interaction':question(),'history':[],'owner_key':'local'})
    app=FastAPI();app.include_router(messages.router)
    answer=response(answers=[{'question_id':'target','text':'PETase'}]).model_dump()
    url='/sessions/interaction-test/interactions/respond/stream'
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
        invalid=await client.post(url,json={**answer,'answers':[]})
        assert invalid.status_code==422
        first=await client.post(url,json=answer)
        assert first.status_code==200 and 'event: done' in first.text
        store._mem.clear()
        duplicate=await client.post(url,json=answer)
        assert duplicate.status_code==200 and len(calls)==1
        conflict=await client.post(url,json={**answer,'decision':'cancel','answers':[]})
        assert conflict.status_code==409
        stale=await client.post(url,json={**answer,'interaction_id':'b'*32})
        assert stale.status_code==409
        def forbidden(*a):raise HTTPException(403,'Not your conversation')
        monkeypatch.setattr(messages,'_assert_session_access',forbidden)
        assert (await client.post(url,json=answer)).status_code==403
    with JobQueue().connection() as c:assert c.execute('select count(*) from jobs').fetchone()[0]==1


@pytest.mark.asyncio
async def test_scientist_shared_request_pauses_before_review_and_keeps_plan(tmp_path,monkeypatch):
    from pathlib import Path
    from agent.harness import autonomous_research as module
    from agent.harness.contracts import Profile
    class Relay:
        calls=0;base_url='http://unused';token='fixture'
        def __init__(self,**kwargs):Path(kwargs['output']).touch()
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
    monkeypatch.setattr(module,'EvidenceGateway',Relay)
    monkeypatch.setattr(module,'audit_requests',lambda path:{'passed':True,'model_calls':2})
    roles=[]
    async def consult(role,context):
        roles.append(role)
        if role=='PI':return 'Identify the protein first.'
        assert role=='CB'
        return json.dumps({'summary':'Study protein','steps':[{'id':'inspect','title':'Inspect supplied protein','acceptance':'Sequence identified'}],'needs_input':[]})
    async def executor(request,service,events):events.append({'kind':'interaction','interaction':question()})
    profile=Profile.from_dict({'mode':'science_expert','harness':'expert-deepseek-harness','model':{'provider':'dmx','model':'glm-4-flash','base_url':'https://www.dmxapi.cn/v1'},'budget':{'model_calls':12,'tool_calls':12,'wall_seconds':120,'output_tokens':4096}})
    store=module.create(tmp_path/'goal',profile,'Study my protein',dict(max_cycles=3,model_calls=36,tool_calls=36,wall_seconds=360))
    result=await module.AutonomousResearch(store,'fixture',{},consult=consult,executor=executor,tools=[]).run()
    assert result['status']=='waiting_for_input'
    assert result['progress']['pending_interaction']['id']=='a'*32
    assert result['progress']['plan']['steps'][0]['status']=='pending'
    assert not result['progress']['research_completed']
    assert roles==['PI','CB']
