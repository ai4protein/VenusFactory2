import asyncio
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import sqlite3
import time
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend.web_v2.usage_store import UsageStore, Observation, execution_context, normalize_usage


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv('VENUS_USAGE_DB',str(tmp_path/'usage.sqlite3'))
    monkeypatch.setenv('VENUS_JOB_QUEUE_ROOT',str(tmp_path/'queue'))
    monkeypatch.setenv('VENUSFACTORY_SESSION_DB_PATH',str(tmp_path/'sessions.sqlite3'))
    monkeypatch.setenv('VENUS_ACCESS_DB',str(tmp_path/'access.sqlite3'))
    monkeypatch.setenv('WEBUI_V2_MODE','local')
    monkeypatch.delenv('VENUS_QUEUE_JOB_ID',raising=False)
    return UsageStore()


def record(store,identity='one',**kw):
    values=dict(identity=identity,kind='model',name='free-model',status='completed',context={'mode':'agent','session_id':'a','run_id':'run-a'},usage={'prompt_tokens':100,'completion_tokens':20,'total_tokens':120})
    values.update(kw);store.record(**values)


@pytest.mark.parametrize('usage,expected',[(None,(None,None,None)),({},(None,None,None)),({'prompt_tokens':0,'completion_tokens':0},(0,0,0)),({'input_tokens':3,'output_tokens':2},(3,2,5)),({'prompt_tokens':-1,'completion_tokens':True},(None,None,None)),({'prompt_tokens':10,'completion_tokens':5,'total_tokens':15,'reasoning_tokens':4,'cached_tokens':8},(10,5,15))])
def test_usage_never_invents_counts_or_adds_subtotals(usage,expected):
    assert normalize_usage(usage)==expected


def test_upsert_failures_missing_usage_and_filters(store):
    record(store,status='running',usage=None)
    record(store)
    record(store)
    record(store,'two',status='failed',usage=None)
    record(store,'tool',kind='tool',name='sequence_properties',usage=None)
    record(store,'science',context={'mode':'scientist','session_id':'b'},usage={'input_tokens':50,'output_tokens':5})
    result=store.overview()
    assert result['summary']['total_tokens']==175
    assert result['summary']['model_calls']==3
    assert result['summary']['unreported_calls']==1
    assert result['summary']['tool_calls']==1
    assert store.overview(mode='agent')['summary']['total_tokens']==120
    assert store.events(session_id='b')['total']==1
    assert store.events(session_id="' OR 1=1 --")['total']==0
    assert len(store.events(limit=2,offset=1)['items'])==2
    with store.connection() as c:c.execute('UPDATE usage_events SET created=? WHERE id=?',(time.time()-10*86400,'science'))
    assert store.overview(days=7)['summary']['total_tokens']==120


def _process_write(payload):
    path,index=payload
    record(UsageStore(path),str(index))


def test_multiple_processes_persist_without_double_counting(store):
    with ProcessPoolExecutor(max_workers=4) as pool:
        list(pool.map(_process_write,[(str(store.path),i) for i in range(20)]))
    assert UsageStore(store.path).overview()['summary']['model_calls']==20


def test_modes_have_separate_default_paths(tmp_path,monkeypatch):
    monkeypatch.delenv('VENUS_USAGE_DB',raising=False)
    monkeypatch.setenv('VENUS_RUNTIME_ROOT',str(tmp_path))
    monkeypatch.setenv('WEBUI_V2_MODE','local');local=UsageStore();record(local)
    monkeypatch.setenv('WEBUI_V2_MODE','online');online=UsageStore()
    assert local.path!=online.path
    assert online.events()['total']==0


def queue_job(kind,metadata,owner='test'):
    from backend.web_v2.job_queue import JobQueue
    q=JobQueue();identity=q.submit(owner,kind,metadata=metadata)
    return q,identity


def test_parent_attribution_and_worker_interruption(store,monkeypatch):
    q,parent=queue_job('protein-research',{'mode':'researcher','report_id':'report1'})
    _,child=queue_job('mcp:protein',{'parent_job':parent,'tool':'blast'})
    monkeypatch.setenv('VENUS_QUEUE_JOB_ID',child)
    assert execution_context()=={'run_id':child,'session_id':'report1','mode':'researcher','harness':''}
    observation=Observation('tool','blast')
    q.finish(child,'cancelled','Cancelled')
    event=store.events()['items'][0]
    assert event['id']==observation.id and event['status']=='interrupted'
    assert event['total_tokens'] is None


def test_session_summary_includes_three_modes_without_private_payloads(store):
    now=time.time()
    with sqlite3.connect(os.environ['VENUSFACTORY_SESSION_DB_PATH']) as c:
        c.execute('CREATE TABLE sessions(session_id TEXT,owner_key TEXT,created_at REAL,last_accessed REAL,data TEXT)')
        c.execute('INSERT INTO sessions VALUES(?,?,?,?,?)',('a','private-owner',now,now,json.dumps({'chat_mode':'science_agent','status':'completed','history':[{'role':'user','content':'Protein question'}],'api_key':'MUST_NOT_APPEAR'})))
    record(store)
    queue_job('scientist',{'mode':'scientist','session_id':'b','harness':'codex'})
    queue_job('protein-research',{'mode':'researcher','report_id':'c','title':'Hemoglobin'})
    data=store.sessions()
    assert {r['mode'] for r in data['items']}=={'agent','scientist','researcher'}
    assert data['total']==3 and data['active']==2
    assert next(r for r in data['items'] if r['session_id']=='a')['total_tokens']==120
    assert 'MUST_NOT_APPEAR' not in json.dumps(data) and 'private-owner' not in json.dumps(data)
    assert store.sessions(mode='researcher')['total']==1


def test_admin_authentication_pagination_validation_and_no_store(store):
    from backend.web_v2.access_api import router
    from backend.web_v2.online_access import AccessStore
    access=AccessStore();access.bootstrap('test@example.org','temporary-test-password')
    app=FastAPI();app.include_router(router);client=TestClient(app)
    for path in ['/api/admin/usage','/api/admin/usage/events','/api/admin/usage/sessions','/api/admin/usage/jobs']:
        assert client.get(path).status_code==401
    assert client.post('/api/admin/login',json={'email':'test@example.org','password':'temporary-test-password'},headers={'X-Venus-Access':'1'}).status_code==200
    record(store)
    response=client.get('/api/admin/usage')
    assert response.status_code==200 and response.headers['cache-control']=='no-store'
    assert response.json()['summary']['total_tokens']==120
    assert client.get('/api/admin/usage?days=999').status_code==422
    assert client.get('/api/admin/usage/events?offset=-1').status_code==422
    assert client.get('/api/admin/usage/events?kind=invalid').status_code==422
    assert client.post('/api/admin/logout',json={},headers={'X-Venus-Access':'1'}).status_code==200
    assert client.get('/api/admin/usage/events').status_code==401


def test_queue_history_includes_training_and_parent_scoped_jobs(store):
    q,parent=queue_job('scientist',{'mode':'scientist','session_id':'science1','api_key':'MUST_NOT_APPEAR'})
    _,child=queue_job('custom:training',{'parent_job':parent,'command':'MUST_NOT_APPEAR'})
    queue_job('platform:predict',{'session_id':'plaza1'})
    rows=store.jobs(mode='scientist',session_id='science1')
    assert rows['total']==2
    training=next(r for r in rows['items'] if r['id']==child)
    assert training['kind']=='custom:training' and training['status']=='queued'
    assert 'MUST_NOT_APPEAR' not in json.dumps(rows)
    assert store.jobs(mode='tools')['total']==1
    assert len(store.jobs(limit=1,offset=1)['items'])==1


def test_live_gateway_records_actual_response_and_transport_failure(store,tmp_path):
    from agent.harness.gateway import EvidenceGateway
    from agent.rsi.engine.model_gateway import GatewayPolicy
    async def run():
        calls=0
        def upstream(request):
            nonlocal calls
            calls+=1
            if calls==2:return httpx.Response(403,json={'error':'denied'})
            return httpx.Response(200,json={'id':'mock','model':'offline-model','choices':[{'index':0,'message':{'role':'assistant','content':'OK'},'finish_reason':'stop'}],'usage':{'prompt_tokens':15,'completion_tokens':2,'total_tokens':17}})
        policy=GatewayPolicy(model='offline-model',mode='science_agent',scientific_tools=set(),max_calls=2)
        async with EvidenceGateway(policy=policy,upstream_url='https://offline.invalid/v1',api_key='DO_NOT_LOG',output=tmp_path/'gateway.jsonl',transport=httpx.MockTransport(upstream)) as gateway:
            async with httpx.AsyncClient() as client:
                for _ in range(2):await client.post(gateway.base_url+'/chat/completions',headers={'Authorization':'Bearer '+gateway.token},json={'model':'offline-model','messages':[{'role':'user','content':'DO_NOT_LOG_PROMPT'}],'stream':False})
    asyncio.run(run())
    result=store.overview()['summary']
    assert result['model_calls']==2 and result['total_tokens']==17 and result['failed']==1
    assert 'DO_NOT_LOG' not in json.dumps(store.events())


def test_researcher_complete_records_usage_without_persisting_text(store,monkeypatch):
    from backend.web_v2 import research_report,job_queue
    from agent.harness import free_catalog
    from agent.rsi import model_access
    binding=SimpleNamespace(model='offline-model',base_url='https://offline.invalid/v1',protocol='chat_completions')
    monkeypatch.setattr(research_report,'model_option',lambda _: (SimpleNamespace(binding=binding,inference_controls={},max_output_tokens=0), 'DO_NOT_LOG'))
    monkeypatch.setattr(free_catalog,'free_routes',lambda: ([{'model':binding.model,'base_url':binding.base_url,'protocol':binding.protocol,'max_output_tokens':100}],None))
    monkeypatch.setattr(model_access,'authorize_model',lambda **_:None)
    @asynccontextmanager
    async def slot(*args,**kwargs):yield
    monkeypatch.setattr(job_queue,'resource_slot',slot)
    client=httpx.AsyncClient
    transport=httpx.MockTransport(lambda _: httpx.Response(200,json={'choices':[{'message':{'content':'DO_NOT_LOG_REPORT'},'finish_reason':'stop'}],'usage':{'prompt_tokens':7,'completion_tokens':3,'total_tokens':10}}))
    monkeypatch.setattr(research_report.httpx,'AsyncClient',lambda **kw: client(**kw,transport=transport))
    assert asyncio.run(research_report.complete('offline',[{'role':'user','content':'DO_NOT_LOG_PROMPT'}]))[0]=='DO_NOT_LOG_REPORT'
    assert store.overview()['summary']['total_tokens']==10
    assert 'DO_NOT_LOG' not in json.dumps(store.events())


def test_waiting_requests_are_not_missing_reports_and_daily_uses_quota_timezone(store, monkeypatch):
    from datetime import datetime, timezone
    monkeypatch.setenv('VENUS_QUOTA_TIMEZONE','Asia/Shanghai')
    instant = datetime(2026,11,3,18,0,tzinfo=timezone.utc).timestamp()
    monkeypatch.setattr('backend.web_v2.usage_store.time.time', lambda:instant)
    record(store,'waiting',status='running',usage=None)
    record(store,'missing',status='completed',usage=None)
    record(store,'reported')
    result=store.overview()
    assert result['summary']['pending_calls']==1
    assert result['summary']['unreported_calls']==1
    assert result['summary']['reported_calls']==1
    assert result['summary']['last_reported_at']==instant
    assert result['daily'][0]['day']=='2026-11-04'
    assert result['timezone']=='Asia/Shanghai'


def test_provider_counters_reject_overflow_without_losing_other_fields(store):
    record(store,usage={'input_tokens':2**100,'output_tokens':3,'total_tokens':2**100})
    row=store.events()['items'][0]
    assert row['input_tokens'] is None and row['total_tokens'] is None
    assert row['output_tokens']==3


@pytest.mark.parametrize('protocol', ['chat','responses'])
def test_interrupted_gateway_keeps_reported_usage_without_success_or_retries(store,tmp_path,protocol):
    from agent.harness.gateway import EvidenceGateway
    from agent.harness.responses_gateway import ResponsesGateway, ResponsesPolicy
    from agent.rsi.engine.model_gateway import GatewayPolicy
    async def run():
        counts={'input_tokens':15,'output_tokens':2,'total_tokens':17}
        event={'usage':counts} if protocol=='chat' else {'type':'response.completed','response':{'usage':counts}}
        raw=('data: '+json.dumps(event)+'\n\n').encode()  # provider sent usage, but no valid completed response
        def upstream(request):
            return httpx.Response(200,content=raw,headers={'content-type':'text/event-stream'})
        if protocol=='chat':
            cls=EvidenceGateway;route='/chat/completions'
            policy=GatewayPolicy(model='offline-model',mode='science_agent',scientific_tools=set(),max_calls=1)
            payload={'model':'offline-model','messages':[{'role':'user','content':'DO_NOT_LOG'}],'stream':True}
        else:
            cls=ResponsesGateway;route='/responses'
            policy=ResponsesPolicy(model='offline-model',scientific_tools=set(),max_calls=1,max_output_tokens=100)
            payload={'model':'offline-model','input':'DO_NOT_LOG','stream':True}
        async with cls(policy=policy,upstream_url='https://offline.invalid/v1',api_key='DO_NOT_LOG',output=tmp_path/'gateway.jsonl',transport=httpx.MockTransport(upstream)) as gateway:
            async with httpx.AsyncClient() as client:
                response=await client.post(gateway.base_url+route,headers={'Authorization':'Bearer '+gateway.token},json=payload)
                assert response.status_code==502
    asyncio.run(run())
    result=store.overview()['summary']
    assert result['model_calls']==1 and result['failed']==1
    assert result['total_tokens']==17 and result['unreported_calls']==0
    assert 'DO_NOT_LOG' not in json.dumps(store.events())
