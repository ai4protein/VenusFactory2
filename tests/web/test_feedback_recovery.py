import asyncio
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request
from backend.web_v2 import quota_feedback as feedback
from backend.web_v2.online_access import AccessStore, fingerprint, VisitorMiddleware
from backend.web_v2.compute_access import ComputeStore
from backend.web_v2.chat_api import _shared as chat
from agent.session_store import SessionStore, SqliteBackend

@pytest.fixture
def stores(tmp_path, monkeypatch):
    monkeypatch.setenv('VENUS_ACCESS_DB',str(tmp_path/'access.db'))
    monkeypatch.setenv('VENUS_USAGE_DB',str(tmp_path/'usage.db'))
    monkeypatch.setenv('VENUS_ONLINE_IP_STARTS_PER_MINUTE','1000')
    monkeypatch.setenv('VENUS_ONLINE_IP_COMPUTE_STARTS_PER_MINUTE','1000')
    monkeypatch.setattr(chat,'_runtime_mode',lambda:'online')
    monkeypatch.setattr('backend.web_v2.online_visibility.OnlineVisibility.is_hidden',lambda *a:False)
    return AccessStore(),ComputeStore()

def request(cookie, ip='10.0.0.1', ua='browser'):
    headers=[(b'user-agent',ua.encode()),(b'cookie',('vf_online_visitor='+cookie).encode())]
    return Request({'type':'http','method':'GET','path':'/api/chat/sessions','headers':headers,'client':(ip,1)})

def test_cookie_recovers_new_and_legacy_sessions_after_ip_change_and_token_expiry(stores):
    access,_=stores; req=request('a'*64)
    state={'session_id':'new'}
    chat._issue_session_access_token(state,req)
    state['token_expires_at']='2000-01-01T00:00:00+00:00'
    chat._assert_session_access(state,request('a'*64,'10.0.0.2','new-browser-version'))
    with pytest.raises(HTTPException):chat._assert_session_access(state,request('b'*64))
    legacy={'session_id':'old','owner_key':'old-ip-hash','token_expires_at':'2000-01-01'}
    access.admit('chat:old','agent','ip',fingerprint('a'*64))
    chat._assert_session_access(legacy,request('a'*64,'10.0.0.9'))
    with pytest.raises(HTTPException):chat._assert_session_access(legacy,request('b'*64))
    assert chat._historical_session_ids(request('a'*64))=={'old'}

def test_recovered_sidebar_is_owner_scoped_and_survives_new_store(tmp_path):
    db=tmp_path/'sessions.db'
    store=SessionStore(SqliteBackend(db),lambda s:s)
    async def scenario():
        for sid,owner in [('old','legacy'),('new','visitor:a'),('other','visitor:b')]:
            await store.create({'session_id':sid,'owner_key':owner,'history':[{'role':'user','content':sid}]})
        restored=SessionStore(SqliteBackend(db),lambda s:s)
        rows=await restored.list_summaries('visitor:a',extra_session_ids={'old'})
        assert {r['session_id'] for r in rows}=={'old','new'}
    asyncio.run(scenario())

def test_reset_restores_all_resources_without_erasing_identity_history_or_idempotency(stores):
    access,compute=stores
    for i,mode in enumerate(['agent','scientist','researcher']):
        access.admit('chat:'+str(i),mode,'ip','visitor')
    compute.bind('owner',('ip','visitor'))
    compute.admit('job','owner',{'prediction':20})
    assert access.status('ip','visitor')['remaining']==0
    feedback.reset(access,{'email':'admin@example.org'})
    assert access.status('ip','visitor')['remaining']==3
    assert compute.status('owner')['prediction']['remaining']==20
    assert access.owned_chat_ids('visitor')=={'0','1','2'}
    assert access.admit('chat:0','agent','ip','visitor') is False
    assert compute.admit('job','owner',{'prediction':20}) is False
    stats=feedback.overview(compute)
    assert stats['modes']==dict(agent=1,scientist=1,researcher=1)
    assert stats['conversations_used']==stats['predictions_used']==0

def test_prediction_grant_shared_across_devices_and_revoked_on_rotation(stores):
    access,compute=stores; actor={'email':'admin@example.org'}
    data=dict(email='user@example.org',extra=4,extra_predictions=10,active=True)
    grant=access.save_grant(actor,data)
    for visitor in ['a','b']:
        feedback.bind(access,visitor,data['email'],grant['key'])
        compute.bind(visitor,('ip',visitor))
    assert compute.status('a')['prediction']['limit']==30
    compute.admit('batch','a',{'prediction':30})
    assert compute.status('b')['prediction']['remaining']==0
    with pytest.raises(HTTPException):compute.admit('excess','b',{'prediction':1})
    access.save_grant(actor,data,grant['id'],True)
    assert compute.status('b')['prediction']['limit']==20
    assert access.grants()[0]['extra_predictions']==10

def test_html_establishes_cookie_feedback_is_email_only_and_reset_requires_validation(stores):
    from backend.web_v2.access_api import router
    access,compute=stores
    app=FastAPI();app.add_middleware(VisitorMiddleware);app.include_router(router)
    @app.get('/')
    @app.get('/api/chat/sessions')
    def root():return {'ok':True}
    with TestClient(app) as client:
        page=client.get('/',headers={'accept':'text/html'})
        assert 'vf_online_visitor=' in page.headers['set-cookie']
        assert page.headers['cache-control'] == 'private, no-store'
        history = client.get('/api/chat/sessions')
        assert history.headers['cache-control'] == 'private, no-store'
        assert 'cookie' in history.headers['vary'].lower()
        payload=dict(email='person@example.org',name='A',institution='Lab',contact='email',feedback='Prediction upload needs clearer units',conversations=5,predictions=40)
        assert client.post('/api/quota/feedback',json=payload,headers={'X-Venus-Access':'1'}).status_code==404
        with access.connect() as c:
            assert not c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='quota_feedback'").fetchone()
        assert client.post('/api/admin/allowances/reset',json={'confirm':'RESET'},headers={'X-Venus-Access':'1'}).status_code==401
        access.bootstrap('admin@example.org','fixture-password-123')
        assert client.post('/api/admin/login',json={'email':'admin@example.org','password':'fixture-password-123'},headers={'X-Venus-Access':'1'}).status_code==200
        assert set(client.get('/api/admin/allowances').json()) == {'overview'}
        assert client.post('/api/admin/allowances/reset',json={'confirm':'WRONG'},headers={'X-Venus-Access':'1'}).status_code==422
        assert client.post('/api/admin/allowances/reset',json={'confirm':'RESET'},headers={'X-Venus-Access':'1'}).status_code==200

def test_visit_and_mode_counts_are_not_reset_with_allowances(stores):
    from backend.web_v2.usage_store import UsageStore
    access, compute = stores
    feedback.visit(access, 'a')
    feedback.visit(access, 'a', document=False)
    feedback.visit(access, 'b')
    usage = UsageStore()
    for mode in ['agent', 'scientist', 'researcher']:
        usage.record(identity=mode,kind='model',name='fixture',status='succeeded',
                     context={'mode':mode},usage={'total_tokens':7})
    stats = feedback.overview(compute)
    assert stats['visitors_today'] == stats['visits_today'] == 2
    assert all(stats['mode_calls'][mode]['tokens'] == 7 for mode in ['agent','scientist','researcher'])
    feedback.reset(access, {'email':'admin@example.org'})
    assert feedback.overview(compute)['mode_calls'] == stats['mode_calls']


def test_bound_chat_and_prediction_grant_remains_after_storage_loss(stores):
    access, compute = stores
    grant = access.save_grant({'email':'admin@example.org'},
                             dict(email='user@example.org',extra=4,extra_predictions=10,active=True))
    feedback.bind(access,'visitor','user@example.org',grant['key'])
    assert access.status('ip','visitor')['limit'] == 7
    assert compute.status(actor=('ip','visitor'))['prediction']['limit'] == 30
    feedback.unbind(access,'visitor')
    assert access.status('ip','visitor')['limit'] == 3
    assert compute.status(actor=('ip','visitor'))['prediction']['limit'] == 20
