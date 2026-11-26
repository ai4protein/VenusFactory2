"""Online selection uses real admission and quotas without making provider calls."""
import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.web_v2 import agent_execution, online_access
from backend.web_v2.chat_api import messages
from backend.web_v2.online_access import AccessStore, VisitorMiddleware


@pytest.fixture
def client(tmp_path, monkeypatch, durable_chat_driver):
    monkeypatch.setenv('WEBUI_V2_MODE', 'online')
    monkeypatch.setenv('WEBUI_V2_ONLINE_DAILY_CHAT_LIMIT', '3')
    # A test run may cross midnight; quota admission here stays on one day.
    monkeypatch.setattr(AccessStore, 'day', staticmethod(lambda: ('2030-01-01', '2030-01-02T00:00:00+00:00')))
    monkeypatch.setenv('VENUS_ACCESS_DB', str(tmp_path / 'access.sqlite3'))
    monkeypatch.setenv('DMXAPI_API_KEY', 'test-placeholder')
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-placeholder')
    monkeypatch.setattr(messages, '_runtime_mode', lambda: 'online')
    monkeypatch.setattr(online_access, 'online', lambda: True)
    states, locks = {}, {}
    async def state(sid):
        return states.setdefault(sid, {'session_id': sid, 'history': []})
    async def lock(sid):
        return locks.setdefault(sid, asyncio.Lock())
    async def noop(*args): pass
    async def stream(state, text, paths, **kwargs):
        state['last_user_text'] = text
        state['last_attachment_paths'] = paths
        yield 'event: done\ndata: {}\n\n'
    # The executable/runtime is irrelevant to HTTP admission; model resolution,
    # credential checks, selected-profile freezing and quota transactions are real.
    real_catalog = agent_execution.catalog
    def catalog():
        cat = real_catalog()
        cat.registry = SimpleNamespace(
            specs=cat.registry.specs,
            preflight=lambda *args, **kwargs: {'ready': True, 'reasons': []},
        )
        return cat
    monkeypatch.setattr(agent_execution, 'catalog', catalog)
    monkeypatch.setattr(messages, '_get_session_or_404', state)
    monkeypatch.setattr(messages, '_get_lock', lock)
    monkeypatch.setattr(messages, '_assert_session_access', lambda *_: None)
    monkeypatch.setattr(messages, '_assert_no_goal', noop)
    monkeypatch.setattr(messages, '_set_cancel', noop)
    monkeypatch.setattr(messages, '_record_access_event', lambda *_: None)
    monkeypatch.setattr(messages, '_stream_agent', stream)
    durable_chat_driver(SimpleNamespace(get=state, save=noop))
    app = FastAPI()
    app.add_middleware(VisitorMiddleware)
    app.include_router(messages.router, prefix='/api/chat')
    from backend.web_v2.access_api import router
    app.include_router(router)
    with TestClient(app) as http:
        yield http, states


@pytest.mark.parametrize('harness', ['kimi-code', 'deepseek-harness', 'codex', 'claude-code'])
def test_online_selection_is_used_and_shares_quota(client, harness):
    http, states = client
    options = http.get('/api/chat/agent-execution-options').json()
    assert next(h for h in options['harnesses'] if h['id'] == harness)['ready']
    free_models = {m['id'] for m in options['models'] if m['free']}
    model = next(c['model_id'] for c in options['combinations']
                 if c['harness'] == harness and c['ready'] and c['model_id'] in free_models)
    selection = {'harness': harness, 'model_id': model}
    url = '/api/chat/sessions/one/messages/stream'
    assert http.post(url, json={'text': ' ', 'execution': selection}).status_code == 400
    assert http.get('/api/quota').json()['used'] == 0
    payload = {'text': 'Describe protein solubility.', 'execution': selection}
    assert http.post(url, json=payload).status_code == 200
    assert states['one']['agent_execution']['selection'] == selection
    assert states['one']['agent_execution']['profile']['model']['model']
    assert http.post(url, json=payload).status_code == 200
    assert http.post('/api/chat/sessions/one/messages/retry/stream').status_code == 200
    assert http.get('/api/quota').json()['used'] == 1
    for sid in ['two', 'three']:
        assert http.post(f'/api/chat/sessions/{sid}/messages/stream', json=payload).status_code == 200
    assert http.post('/api/chat/sessions/four/messages/stream', json=payload).status_code == 429
    assert http.post(url, json=payload).status_code == 200


def test_online_rejects_paid_unregistered_and_unconfigured_models(client, monkeypatch):
    http, states = client
    monkeypatch.setenv('OPENAI_API_KEY', 'test-paid-key-is-not-enough')
    endpoint = '/api/chat/sessions/rejected/messages/stream'
    for mid in ['openai-official', 'unregistered-model']:
        r = http.post(endpoint, json={'text': 'hello', 'execution': {'harness': 'deepseek-harness', 'model_id': mid}})
        assert r.status_code == 409
    monkeypatch.delenv('DMXAPI_API_KEY')
    r = http.post(endpoint, json={'text': 'hello', 'execution': {'harness': 'deepseek-harness', 'model_id': 'dmx-u2-free'}})
    assert r.status_code == 409
    assert http.get('/api/quota').json()['used'] == 0
    assert 'agent_execution' not in states['rejected']
