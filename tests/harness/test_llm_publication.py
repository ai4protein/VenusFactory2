"""LLM publishing tests: isolated state and mock transports only."""
import asyncio
import json
from dataclasses import replace
import httpx
import pytest
from backend.web_v2 import agent_execution, llm_admin
from agent.harness import provider_config
from agent.harness.llm_policy import listed, route_key
from agent.harness.free_catalog import free_routes
from agent.rsi.engine.model_access import ModelAccessDenied, authorize_model
from mcp_tools.tools.capability_policy import CapabilityStore


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv('VENUS_RUNTIME_ROOT', str(tmp_path/'runtime'))
    monkeypatch.setenv('VENUS_ACCESS_DB', str(tmp_path/'access.sqlite3'))
    monkeypatch.setenv('WEBUI_V2_MODE', 'local')
    monkeypatch.setattr('mcp_tools.tools.runtime_tool_policy.is_online_mode', provider_config.online)
    for key in provider_config.configuration_keys():
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('DMXAPI_API_KEY', 'test-key-only')
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-key-only')


def row(identity='dmx-u2-free'):
    return next(r for r in llm_admin.inventory()['items'] if r['id']==identity)


def test_unpublish_all_modes_and_stale_catalog_then_republish():
    cat = agent_execution.catalog()
    r = row()
    b = cat.models[r['id']].binding
    llm_admin.publish(r['key'], False, 0, 'admin@example.org')
    assert r['id'] not in agent_execution.catalog().models
    from backend.web_v2.science_goals import ScienceGoalService
    assert r['id'] not in ScienceGoalService().execution_catalog().models
    from backend.web_v2.research_report import model_option
    with pytest.raises(ValueError): model_option(r['id'])
    with pytest.raises(ValueError, match='unpublished'):
        cat.select(dict(harness='deepseek-harness', model_id=r['id']))
    with pytest.raises(ModelAccessDenied, match='unpublished'):
        authorize_model(model=b.model, base_url=b.base_url, protocol=b.protocol, output_tokens=100)
    after=row()
    assert after['free'] and not after['enabled'] and after['revision']==1
    with pytest.raises(ValueError, match='changed'): llm_admin.publish(r['key'], True, 0, 'admin@example.org')
    llm_admin.publish(r['key'], True, 1, 'admin@example.org')
    assert r['id'] in agent_execution.catalog().models


def test_mode_isolation_and_endpoint_identity(monkeypatch):
    r=row(); b=agent_execution.catalog().models[r['id']].binding
    llm_admin.publish(r['key'], False, 0, 'admin@example.org')
    assert not listed(b.model,b.base_url)
    assert listed(b.model,'https://another.example/v1')
    monkeypatch.setenv('WEBUI_V2_MODE','online')
    assert row()['enabled'] and listed(b.model,b.base_url)


def test_free_marker_survives_unpublishing_and_can_publish_catalog_disabled_model():
    rows=llm_admin.inventory()['items']
    disabled=next(r for r in rows if r['free'] and not r['enabled'])
    assert disabled['can_publish']
    llm_admin.publish(disabled['key'], True, 0, 'admin@example.org')
    assert disabled['id'] in agent_execution.catalog().models
    assert any(r['model']==disabled['model'] for r in free_routes()[0])


def test_no_paid_online_publication_or_credential_bypass(monkeypatch):
    monkeypatch.setenv('ANTHROPIC_MODEL','claude-test')
    monkeypatch.setenv('ANTHROPIC_API_KEY','test-claude-key')
    official=next(r for r in llm_admin.inventory()['items'] if r['provider']=='anthropic')
    assert not official['free'] and official['can_publish']
    monkeypatch.setenv('WEBUI_V2_MODE','online')
    official=next(r for r in llm_admin.inventory()['items'] if r['provider']=='anthropic')
    assert official['reason']=='free_model_required' and not official['can_publish']
    with pytest.raises(PermissionError): llm_admin.publish(official['key'], True, 0, 'test')
    monkeypatch.delenv('DMXAPI_API_KEY')
    with pytest.raises(PermissionError): llm_admin.publish(row()['key'], True, 0, 'test')
    with pytest.raises(LookupError): llm_admin.publish('llm:unknown', True, 0, 'test')
    assert 'test-claude-key' not in json.dumps(llm_admin.inventory())


@pytest.mark.parametrize("mode", ["local", "online"])
def test_disabling_all_free_routes_leaves_no_implicit_paid_fallback(monkeypatch, mode):
    monkeypatch.setenv("WEBUI_V2_MODE", mode)
    monkeypatch.setenv('OPENAI_MODEL','test-paid')
    monkeypatch.setenv('OPENAI_API_KEY','test-paid-key')
    for r in llm_admin.inventory()['items']:
        if r['free'] and r['enabled']: llm_admin.publish(r['key'], False, 0, 'test')
    assert agent_execution.catalog().default_model==''
    if mode == 'online':
        assert agent_execution.catalog().public({})['models']==[]


@pytest.mark.parametrize('protocol', ['chat_completions','responses'])
def test_running_gateway_rechecks_before_next_provider_request(protocol,tmp_path):
    from agent.harness.gateway import EvidenceGateway
    from agent.harness.responses_gateway import ResponsesGateway, ResponsesPolicy
    from agent.rsi.engine.model_gateway import GatewayPolicy
    identity='dmx-u2-free' if protocol=='chat_completions' else 'or-nvidia-nemotron-3-5-lightning-free'
    option=agent_execution.catalog().models[identity]; b=option.binding
    seen=[]
    def upstream(request):
        seen.append(request)
        raise AssertionError('Unpublished model must not reach provider')
    class MockProvider(httpx.MockTransport):
        pass  # Retain live admission checks while keeping HTTP entirely in memory.
    async def run():
        policy=GatewayPolicy(model=b.model,mode='science_agent',scientific_tools=set(),max_tokens=100,thinking=None) if protocol=='chat_completions' else ResponsesPolicy(model=b.model,scientific_tools=set(),max_output_tokens=100)
        cls=EvidenceGateway if protocol=='chat_completions' else ResponsesGateway
        async with cls(policy=policy,upstream_url=b.base_url,api_key='test-key-only',output=tmp_path/'audit.jsonl',transport=MockProvider(upstream)) as gateway:
            r=row(identity);llm_admin.publish(r['key'],False,0,'test')
            async with httpx.AsyncClient() as client:
                route='/chat/completions' if protocol=='chat_completions' else '/responses'
                body={'model':b.model,'messages':[{'role':'user','content':'hello'}]} if protocol=='chat_completions' else {'model':b.model,'input':'hello'}
                response=await client.post(gateway.base_url+route,headers={'Authorization':'Bearer '+gateway.token},json=body)
                assert response.status_code==400
    asyncio.run(run());assert seen==[]


def test_admin_auth_csrf_and_revision(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.web_v2.access_api import router
    from backend.web_v2.online_access import AccessStore
    AccessStore().bootstrap('admin@example.org','test-password-123')
    app=FastAPI();app.include_router(router);client=TestClient(app)
    assert client.get('/api/admin/llms').status_code==401
    headers={'X-Venus-Access':'1'}
    assert client.post('/api/admin/login',headers=headers,json={'email':'admin@example.org','password':'test-password-123'}).status_code==200
    result=client.get('/api/admin/llms')
    assert result.status_code==200 and 'test-key-only' not in result.text
    assert result.headers['cache-control']=='no-store'
    r=next(r for r in result.json()['items'] if r['id']=='dmx-u2-free')
    payload={'key':r['key'],'enabled':False,'revision':0}
    assert client.put('/api/admin/llms',json=payload).status_code==403
    assert client.put('/api/admin/llms',headers={**headers,'Origin':'https://foreign.example'},json=payload).status_code==403
    assert client.put('/api/admin/llms',headers=headers,json=payload).status_code==200
    assert client.put('/api/admin/llms',headers=headers,json=payload).status_code==409
    assert client.put('/api/admin/llms',headers=headers,json={**payload,'key':'tool:search_tools'}).status_code==404


def test_provider_keys_stay_masked_while_token_limits_are_editable(tmp_path,monkeypatch):
    from backend.web_v2 import access_api, settings_api
    env=tmp_path/'.env';env.write_text('ANTHROPIC_API_KEY=test-secret-claude\nANTHROPIC_CONTEXT_TOKENS=131072\nANTHROPIC_MAX_OUTPUT_TOKENS=4096\nGLM_API_KEY=test-secret-glm\nWEBUI_V2_SESSION_TOKEN_SECRET=test-signing-secret\n')
    monkeypatch.setattr(settings_api,'_ENV_PATH',env)
    monkeypatch.setattr(settings_api,'_ENV_EXAMPLE_PATH',env)
    data=asyncio.run(access_api.admin_env(user={'email':'test@example.org'}))
    values={r['key']:r['value'] for r in data['entries']}
    assert values['ANTHROPIC_CONTEXT_TOKENS']=='131072'
    assert values['ANTHROPIC_MAX_OUTPUT_TOKENS']=='4096'
    assert 'test-secret' not in json.dumps(data) and 'test-signing-secret' not in json.dumps(data)
