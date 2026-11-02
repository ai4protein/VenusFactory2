"""No provider requests: admission races, identity, credentials and HTTP boundary."""
import concurrent.futures
import os
import time
from pathlib import Path
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request
from backend.web_v2.online_access import AccessStore, VisitorMiddleware, client_ip, fingerprint, password_matches
from backend.web_v2.access_api import router


@pytest.fixture
def store(tmp_path,monkeypatch):
    monkeypatch.setenv('VENUS_ACCESS_DB',str(tmp_path/'access.sqlite3'))
    monkeypatch.setenv('WEBUI_V2_ONLINE_DAILY_CHAT_LIMIT','3')
    monkeypatch.setenv('VENUS_QUOTA_TIMEZONE','Asia/Shanghai')
    return AccessStore()


def actor(store):
    assert store.bootstrap('admin@example.org','test-password-123')
    return store.login('admin@example.org','test-password-123','ip')[1]


def grant(store,who,extra=2):
    return store.save_grant(who,dict(email='person@example.org',extra=extra,active=True))


def test_modes_share_three_and_existing_conversation_stays_admitted(store):
    for i,mode in enumerate(['agent','scientist','researcher']):
        assert store.admit(str(i),mode,'ip','browser')
    assert store.status('ip','browser')['remaining']==0
    assert not store.admit('0','agent','ip','browser')
    with pytest.raises(HTTPException) as error: store.admit('fourth','agent','ip','browser')
    assert error.value.status_code==429
    assert error.value.detail['code']=='CHAT_DAILY_LIMIT_REACHED'
    assert error.value.detail['quota']['remaining']==0


def test_atomic_concurrency_and_cross_process_database(store):
    def submit(i):
        try: return AccessStore(store.path).admit(str(i),'agent','ip','browser')
        except HTTPException: return False
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        assert sum(pool.map(submit,range(30)))==3
    assert AccessStore(store.path).status('ip','browser')['used']==3


def test_duplicate_admission_is_atomic(store):
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        assert sum(pool.map(lambda _:store.admit('same','agent','ip','browser'),range(20)))==1
    assert store.status('ip','browser')['used']==1


def test_ip_and_browser_each_limit_evasion(store):
    for i in range(3): store.admit(str(i),'agent','ip','browser')
    for ip,browser in [('ip','cleared-cookies'),('another-ip','browser')]:
        with pytest.raises(HTTPException):store.admit('new:'+ip,'agent',ip,browser)


def test_key_extra_shared_across_devices_and_revocation(store):
    who=actor(store);key=grant(store,who)
    for i in range(5):store.admit(str(i),'agent','ip'+str(i),'visitor'+str(i),key['email'],key['key'])
    assert store.status('fresh','fresh',key['email'],key['key'])['remaining']==0
    with pytest.raises(HTTPException):store.admit('six','agent','fresh','fresh',key['email'],key['key'])
    store.delete_grant(who,key['id'])
    with pytest.raises(HTTPException) as error:store.status('ip','browser',key['email'],key['key'])
    assert error.value.status_code==401


def test_key_can_add_bonus_after_shared_ip_is_full(store):
    who=actor(store);key=grant(store,who,2)
    for i in range(3):store.admit(str(i),'agent','institution-ip','anon')
    assert store.status('institution-ip','person',key['email'],key['key'])['remaining']==2
    for i in range(2):store.admit('bonus'+str(i),'scientist','institution-ip','person',key['email'],key['key'])
    with pytest.raises(HTTPException):store.admit('too-many','agent','institution-ip','other',key['email'],key['key'])


def test_rotation_and_expiration(store):
    who=actor(store);one=grant(store,who)
    two=store.save_grant(who,dict(email=one['email'],extra=2,active=True),one['id'],True)
    with pytest.raises(HTTPException):store.status('ip','v',one['email'],one['key'])
    assert store.status('ip','v',two['email'],two['key'])['limit']==5
    store.save_grant(who,dict(email=one['email'],extra=2,active=True,expires=time.time()-1),one['id'])
    with pytest.raises(HTTPException):store.status('ip','v',two['email'],two['key'])
    assert all('key_hash' not in row and 'key' not in row for row in store.grants())


def test_day_reset_and_no_refund_by_deleting_session(store,monkeypatch):
    store.admit('chat:1','agent','ip','visitor')
    monkeypatch.setattr(AccessStore,'day',staticmethod(lambda:('2099-01-01','2099-01-02T00:00:00+08:00')))
    assert store.status('ip','visitor')['remaining']==3
    assert not store.admit('chat:1','agent','ip','visitor')
    assert store.status('ip','visitor')['remaining']==3


def test_admin_credentials_and_session_revocation(store):
    who=actor(store)
    assert not store.bootstrap('another@example.org','another-password')
    token,_=store.login(who['email'],'test-password-123','ip')
    assert store.authenticate(token)['id']==who['id']
    with store.connect() as c:
        encoded=c.execute('SELECT password_hash FROM admins').fetchone()[0]
    assert 'test-password' not in encoded and password_matches('test-password-123',encoded)
    store.save_admin(who,dict(email=who['email'],password='changed-password',active=True),who['id'])
    with pytest.raises(HTTPException):store.authenticate(token)
    with pytest.raises(ValueError):store.delete_admin(who,who['id'])
    other=store.save_admin(who,dict(email='other@example.org',password='other-password',active=True))
    store.delete_admin(who,other['id'])
    assert len(store.admins())==1


def test_admin_http_boundary_and_csrf(store):
    actor(store)
    app=FastAPI();app.add_middleware(VisitorMiddleware);app.include_router(router)
    client=TestClient(app)
    assert client.get('/api/admin/grants').status_code==401
    body={'email':'admin@example.org','password':'test-password-123'}
    assert client.post('/api/admin/login',json=body).status_code==403
    headers={'X-Venus-Access':'1'}
    assert client.post('/api/admin/login',json=body,headers={**headers,'Origin':'https://evil.example'}).status_code==403
    r=client.post('/api/admin/login',json=body,headers=headers)
    assert r.status_code==200 and 'httponly' in r.headers['set-cookie'].lower()
    assert client.get('/api/admin/admins').status_code==200
    assert client.post('/api/admin/grants',headers=headers,json={'email':'person@example.org','extra':7}).status_code==200
    assert 'key' not in client.get('/api/admin/grants').json()['items'][0]
    assert client.post('/api/admin/logout',headers=headers).status_code==200
    assert client.get('/api/admin/grants').status_code==401


def test_brute_force_limit(store):
    actor(store)
    for _ in range(8):
        with pytest.raises(HTTPException):store.login('missing@example.org','wrong-password','bad-ip')
    with pytest.raises(HTTPException) as error:store.login('missing@example.org','wrong-password','bad-ip')
    assert error.value.status_code==429


def test_forwarded_headers_only_from_explicit_proxies(monkeypatch):
    def request(peer,xff):return Request({'type':'http','client':(peer,123),'headers':[(b'x-forwarded-for',xff.encode())]})
    monkeypatch.delenv('VENUS_TRUSTED_PROXIES',raising=False)
    assert client_ip(request('192.0.2.1','198.51.100.1'))=='192.0.2.1'
    monkeypatch.setenv('VENUS_TRUSTED_PROXIES','127.0.0.1/32')
    assert client_ip(request('127.0.0.1','198.51.100.1,192.0.2.2'))=='192.0.2.2'
    assert client_ip(request('2001:db8::1234',''))=='2001:db8::/64'


def test_scientist_one_active_per_ip_or_browser(store):
    store.science_slot('one','ip','visitor')
    store.science_slot('one','ip','visitor')
    with pytest.raises(HTTPException):store.science_slot('two','ip','other')
    with pytest.raises(HTTPException):store.science_slot('two','other','visitor')
    store.release_science('one')
    store.science_slot('two','ip','visitor')


def test_real_mode_routes_share_admission_and_retries(tmp_path,store,monkeypatch,durable_chat_driver):
    """Exercise real FastAPI route wiring; replace only computation, never admission."""
    import asyncio
    from types import SimpleNamespace
    from backend.web_v2 import online_access, report_api, research_jobs
    from backend.web_v2.chat_api import messages, science_goals, sessions
    monkeypatch.setattr(online_access,'online',lambda:True)
    monkeypatch.setattr(messages,'_runtime_mode',lambda:'online')
    monkeypatch.setattr(science_goals,'_runtime_mode',lambda:'online')
    states={}
    async def state(sid):return states.setdefault(sid,{'session_id':sid,'history':[],'last_user_text':'protein'})
    async def noop(*args):pass
    async def stream(*args,**kwargs):yield 'event: done\ndata: {}\n\n'
    monkeypatch.setattr(messages,'_get_session_or_404',state)
    monkeypatch.setattr(messages,'_assert_session_access',lambda *_:None)
    monkeypatch.setattr(messages,'_assert_no_goal',noop)
    monkeypatch.setattr(messages,'_set_cancel',noop)
    monkeypatch.setattr(messages,'_stream_kimi',stream)
    durable_chat_driver(SimpleNamespace(get=state, save=noop), stream)
    from backend.web_v2 import agent_execution
    monkeypatch.setenv('DMXAPI_API_KEY', 'offline-placeholder')
    real_catalog = agent_execution.catalog
    def catalog():
        cat=real_catalog()
        cat.registry=SimpleNamespace(specs=cat.registry.specs,preflight=lambda *a,**kw: {'ready':True,'reasons':[]})
        return cat
    monkeypatch.setattr(agent_execution,'catalog',catalog)
    monkeypatch.setattr(messages,'_resolve_engine',lambda *_:'kimi-code')
    monkeypatch.setattr(messages,'_record_access_event',lambda *_:None)
    monkeypatch.setattr(science_goals,'_get_session_or_404',state)
    monkeypatch.setattr(science_goals,'_assert_session_access',lambda *_:None)
    monkeypatch.setattr(science_goals,'_session_store',SimpleNamespace(save=noop))
    def create_goal(sid,spec,rid,*,before_start,online_owner):
        before_start();return {'id':rid,'status':'active'}
    monkeypatch.setattr(science_goals.service,'create',create_goal)
    monkeypatch.setenv('VENUS_REPORT_ROOT',str(tmp_path/'reports'))
    monkeypatch.setenv('VENUS_JOB_QUEUE_ROOT',str(tmp_path/'queue'))
    monkeypatch.setattr(research_jobs,'model_option',lambda *_:None)
    def launch(*args,**kwargs):
        q=kwargs['queue'];ident=q.submit(kwargs['owner'],'protein-research','research');q.bind(ident)
        return ident,None
    monkeypatch.setattr(research_jobs,'launch',launch)
    app=FastAPI();app.add_middleware(VisitorMiddleware)
    app.include_router(messages.router,prefix='/api/chat');app.include_router(science_goals.router,prefix='/api/chat');app.include_router(sessions.router,prefix='/api/chat');app.include_router(report_api.router)
    client=TestClient(app)
    assert client.post('/api/chat/sessions/agent-one/messages/stream',json={'text':'Explain this protein'}).status_code==200
    assert client.post('/api/chat/sessions/science-one/science-goals',json={'request_id':'goal-12345','objective':'Analyze my protein','budget':online_access.SCIENCE_LIMITS}).status_code==200
    report={'request_id':'report-12345','model_id':'free','accession':'P69905','analyses':['properties']}
    created=client.post('/api/report/jobs',json=report)
    assert created.status_code==202,created.text
    again=client.post('/api/report/generate',json=report)
    assert again.status_code==202 and again.json()['id']==created.json()['id']
    exceeded=client.post('/api/chat/sessions/agent-two/messages/stream',json={'text':'A new protein'})
    assert exceeded.status_code==429 and exceeded.json()['detail']['code']=='CHAT_DAILY_LIMIT_REACHED'
    assert client.post('/api/chat/sessions/agent-one/messages/retry/stream',json={}).status_code==200
    assert client.post('/api/report/jobs',json={**report,'request_id':'report-67890'}).status_code==429
    assert client.post('/api/report/generate/stream',json={**report,'request_id':'report-stream'}).status_code==429
    assert client.post('/api/chat/sessions',json={}).status_code==429
    assert client.post('/api/chat/sessions/legacy/messages/stream',json={'text':'Bypass','chat_mode':'science_expert'}).status_code==409


def test_scientist_server_budget_rejects_unbounded_specs(tmp_path,monkeypatch):
    from backend.web_v2 import online_access
    from backend.web_v2.science_goals import ScienceGoalService
    monkeypatch.setattr(online_access,'online',lambda:True)
    service=ScienceGoalService(state_root=tmp_path/'goals')
    base=dict(driver='autonomous_research.v1',parameters={},objective='Train a protein model',budget=dict(online_access.SCIENCE_LIMITS))
    with pytest.raises(ValueError):service.create('s',{**base,'driver':'paper_campaign.v1'},'request-1234')
    for key in online_access.SCIENCE_LIMITS:
        with pytest.raises(ValueError):service.create('s',{**base,'budget':{**base['budget'],key:base['budget'][key]+1}},'request-1234')
    assert not (tmp_path/'goals').exists()


def test_online_scientist_can_train_small_cpu_model_in_queue(tmp_path,monkeypatch):
    import asyncio, json, shutil
    if not shutil.which('bwrap'):pytest.skip('Bubblewrap is required on the research host')
    from agent.harness.research_workspace import run_python
    monkeypatch.setenv('VENUS_SCIENTIST_ONLINE','1')
    monkeypatch.setenv('VENUS_JOB_QUEUE_ROOT',str(tmp_path/'queue'))
    monkeypatch.delenv('VENUS_QUEUE_JOB_ID',raising=False)
    work=tmp_path/'workspace';work.mkdir()
    (work/'train.py').write_text('''import os,resource,json
import numpy as np
from sklearn.linear_model import Ridge
x=np.arange(24).reshape(-1,1);y=x[:,0]*2+1
model=Ridge(alpha=.01).fit(x,y)
assert abs(model.predict([[25]])[0]-51)<.1
assert resource.getrlimit(resource.RLIMIT_AS)[0]==2*1024**3
assert not os.path.exists('/dev/nvidia0')
assert not os.path.exists('/home/tanyang/workspace/VenusFactory2/.env')
assert not os.getenv('DMXAPI_API_KEY')
print(json.dumps({'trained':True,'prediction':float(model.predict([[25]])[0]),'memory_gib':2}))
''')
    result=asyncio.run(run_python(work,tmp_path/'control','train.py',120))
    assert result['exit_code']==0,result['stderr']
    assert json.loads(result['stdout'])['trained'] is True
