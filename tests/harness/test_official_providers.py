"""Official provider paths are tested without real keys or paid API calls."""
import asyncio
import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from agent.harness import provider_config as pc
from agent.harness.catalog import ExecutionCatalog, resolve_credential
from agent.harness.contracts import Profile
from agent.rsi.engine.model_access import ModelAccessDenied, authorize_model

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    for key in pc.configuration_keys():
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('WEBUI_V2_MODE', 'local')


def configure(monkeypatch, prefix='OPENAI', model='test-model'):
    monkeypatch.setenv(prefix + '_API_KEY', 'test-secret-' + prefix)
    monkeypatch.setenv(prefix + '_MODEL', model)
    return next(r for r in pc.configured_models() if r['binding'].model == model)


def catalog(mode='science_agent', online=False):
    profile = Profile.from_dict(json.loads((ROOT / 'configs/harness/free_science_goal.json').read_text()))
    return ExecutionCatalog(ROOT, replace(profile, mode=mode),
                            path=ROOT / 'configs/harness/free_execution_catalog.json',
                            free_only=online, include_configured=True)


@pytest.mark.parametrize('provider', pc.PROVIDERS, ids=lambda p: p.id)
def test_provider_binding_admission_and_no_secret_in_public_catalog(monkeypatch, provider):
    row = configure(monkeypatch, provider.prefix)
    binding = row['binding']
    assert binding.base_url == provider.base_url
    assert resolve_credential(binding) == 'test-secret-' + provider.prefix
    result = authorize_model(model=binding.model, base_url=binding.base_url, protocol='chat_completions', output_tokens=100)
    assert result['scope'] == 'local_configured_model'
    for mode in ('science_agent', 'science_expert'):
        cat = catalog(mode)
        assert row['id'] in cat.models
        assert 'test-secret' not in json.dumps(cat.public({}))
    from backend.web_v2.research_report import model_option
    assert model_option(row['id'])[0].binding == binding


def test_key_alone_does_not_enable_models_or_paid_fallback(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'test-secret')
    assert not pc.configured_models()
    assert all(m.binding.provider != 'openai' for m in catalog().models.values())
    with pytest.raises(ModelAccessDenied):
        authorize_model(model='gpt-4o', base_url='https://api.openai.com/v1', protocol='chat_completions', output_tokens=100)


def test_multiple_models_stable_ids_and_key_aliases(monkeypatch):
    monkeypatch.setenv('GLM_MODELS', 'glm-a,glm-b,glm-a')
    monkeypatch.setenv('ZHIPU_API_KEY', 'test-alias')
    before = {r['binding'].model: r['id'] for r in pc.configured_models()}
    monkeypatch.setenv('GLM_MODELS', 'glm-b,glm-a')
    assert {r['binding'].model: r['id'] for r in pc.configured_models()} == before
    assert all(resolve_credential(r['binding']) == 'test-alias' for r in pc.configured_models())


def test_online_hides_and_rejects_official_models_even_with_credentials(monkeypatch):
    row = configure(monkeypatch)
    monkeypatch.setenv('WEBUI_V2_MODE', 'online')
    assert row['id'] not in catalog(online=True).models
    assert all(m['free'] for m in catalog(online=True).public({})['models'])
    with pytest.raises(ModelAccessDenied):
        authorize_model(model='test-model', base_url='https://api.openai.com/v1', protocol='chat_completions', output_tokens=100)
    from backend.web_v2.research_report import model_option
    with pytest.raises(ValueError):
        model_option(row['id'])


@pytest.mark.parametrize('base', ['http://api.example.com/v1', 'https://key@api.example.com/v1', 'https://api.example.com/v1?key=secret', 'https://api.example.com:8443/v1'])
def test_invalid_endpoints_rejected(monkeypatch, base):
    configure(monkeypatch)
    monkeypatch.setenv('OPENAI_BASE_URL', base)
    with pytest.raises(ValueError): pc.configured_models()


def test_exact_endpoint_model_limits_and_missing_key_fail_closed(monkeypatch):
    configure(monkeypatch)
    for model, base, tokens in [('other-model', 'https://api.openai.com/v1', 100),
                                ('test-model', 'https://other.example/v1', 100),
                                ('test-model', 'https://api.openai.com/v1', 4097)]:
        with pytest.raises(ModelAccessDenied):
            authorize_model(model=model, base_url=base, protocol='chat_completions', output_tokens=tokens)
    monkeypatch.delenv('OPENAI_API_KEY')
    with pytest.raises(ModelAccessDenied):
        authorize_model(model='test-model', base_url='https://api.openai.com/v1', protocol='chat_completions', output_tokens=100)


def test_codex_uses_responses_other_harnesses_use_chat(monkeypatch, tmp_path):
    row = configure(monkeypatch)
    cat = catalog()
    binary = tmp_path / 'runtime'; binary.write_text('test')
    runtimes = {k: str(binary) for k in ['kimi_bin', 'dsh_bin', 'codex_bin', 'claude_bin', 'python_bin']}
    for harness in ['kimi-code', 'deepseek-harness', 'claude-code', 'codex']:
        profile, _ = cat.select(dict(harness=harness, model_id=row['id']))
        assert profile.model.protocol == ('responses' if harness == 'codex' else 'chat_completions')
        assert cat.registry.preflight(profile, runtimes)['ready']


def test_workspace_header_only_reaches_anthropic(monkeypatch):
    monkeypatch.setenv('ANTHROPIC_WORKSPACE_ID', 'workspace-test')
    assert pc.request_headers('https://api.anthropic.com/v1', 'key')['anthropic-workspace-id'] == 'workspace-test'
    assert 'anthropic-workspace-id' not in pc.request_headers('https://api.x.ai/v1', 'key')


@pytest.mark.parametrize('prefix', ['OPENAI', 'ANTHROPIC', 'XAI', 'GLM', 'MINIMAX'])
@pytest.mark.parametrize('stream', [False, True])
def test_gateway_real_http_contract_with_mock_provider(monkeypatch, tmp_path, prefix, stream):
    from agent.harness.gateway import EvidenceGateway
    from agent.rsi.engine.model_gateway import GatewayPolicy
    row = configure(monkeypatch, prefix)
    b = row['binding']
    seen = []
    def upstream(request):
        data = json.loads(request.content); seen.append(data)
        assert str(request.url) == b.base_url + '/chat/completions'
        assert request.headers['authorization'] == 'Bearer test-secret-' + prefix
        assert 'thinking' not in data and 'temperature' not in data
        assert data.get('max_completion_tokens' if prefix == 'OPENAI' else 'max_tokens') == 128
        assert data['tools'][0]['function']['name'] == 'mcp__venusfactory__search_tools'
        choice = {'index': 0, 'finish_reason': 'stop'}
        if stream:
            chunks = [{'choices': [{'index': 0, 'delta': {'content': 'hello'}, 'finish_reason': None}]},
                      {'choices': [{**choice, 'delta': {}}], 'usage': {'total_tokens': 5}}]
            body = ''.join('data: ' + json.dumps(c) + '\n\n' for c in chunks) + 'data: [DONE]\n\n'
            return httpx.Response(200, text=body, headers={'content-type': 'text/event-stream'})
        return httpx.Response(200, json={'choices': [{**choice, 'message': {'role': 'assistant', 'content': 'hello'}}], 'usage': {'total_tokens': 5}})
    async def run():
        policy = GatewayPolicy(model=b.model, mode='science_agent', scientific_tools={'search_tools'}, max_tokens=128, thinking=None)
        async with EvidenceGateway(policy=policy, upstream_url=b.base_url, api_key='test-secret-' + prefix,
                                   output=tmp_path / 'requests.jsonl', transport=httpx.MockTransport(upstream)) as relay:
            async with httpx.AsyncClient() as client:
                response = await client.post(relay.base_url + '/chat/completions', headers={'Authorization': 'Bearer ' + relay.token},
                    json={'model': b.model, 'stream': stream, 'messages': [{'role': 'user', 'content': 'hello'}],
                          'tools': [{'type': 'function', 'function': {'name': 'mcp__venusfactory__search_tools', 'parameters': {'type': 'object'}}}]})
                assert response.status_code == 200
                assert 'hello' in response.text
    asyncio.run(run())
    assert len(seen) == 1
    assert 'test-secret' not in (tmp_path / 'requests.jsonl').read_text()


def test_legacy_config_never_uses_another_providers_key(monkeypatch):
    from agent.model_config import ModelRuntimeConfig
    monkeypatch.setenv('OPENAI_API_KEY', 'test-openai-secret')
    monkeypatch.setenv('DMXAPI_API_KEY', 'test-dmx-secret')
    cfg = ModelRuntimeConfig(provider='anthropic', api_key_env='ANTHROPIC_API_KEY')
    assert cfg.resolve_api_key() == ''


def test_scientist_worker_gets_selected_credential_and_provider_configuration(monkeypatch):
    row = configure(monkeypatch, 'ANTHROPIC')
    configure(monkeypatch, 'OPENAI', 'another-model')
    monkeypatch.setenv('ANTHROPIC_WORKSPACE_ID', 'workspace-test')
    from backend.web_v2.science_goals import ScienceGoalService
    service = ScienceGoalService(root=ROOT)
    profile, _ = service.execution_catalog().select(dict(harness='expert-deepseek-harness', model_id=row['id']))
    env = service.environment(profile)
    assert env['ANTHROPIC_API_KEY'] == 'test-secret-ANTHROPIC'
    assert env['ANTHROPIC_MODEL'] == 'test-model'
    assert env['ANTHROPIC_WORKSPACE_ID'] == 'workspace-test'
    assert 'OPENAI_API_KEY' not in env


@pytest.mark.parametrize('prefix', ['OPENAI', 'ANTHROPIC', 'MINIMAX'])
def test_researcher_sends_configured_provider_request(monkeypatch, prefix):
    from backend.web_v2 import research_report
    row = configure(monkeypatch, prefix)
    real_client = httpx.AsyncClient
    seen = []
    def upstream(request):
        body = json.loads(request.content); seen.append(body)
        assert 'thinking' not in body and 'temperature' not in body
        assert body.get('max_completion_tokens' if prefix == 'OPENAI' else 'max_tokens') == 100
        return httpx.Response(200, json={'choices': [{'message': {'content': 'Protein report'}, 'finish_reason': 'stop'}]})
    monkeypatch.setattr(research_report.httpx, 'AsyncClient',
                        lambda **kwargs: real_client(**kwargs, transport=httpx.MockTransport(upstream)))
    assert asyncio.run(research_report.complete(row['id'], [{'role': 'user', 'content': 'test'}], 100)) == ('Protein report', False)
    assert len(seen) == 1


def test_settings_template_contains_canonical_provider_fields():
    text = (ROOT / '.env.example').read_text()
    for provider in pc.PROVIDERS:
        for suffix in ('_API_KEY', '_MODEL', '_MODELS', '_BASE_URL', '_CONTEXT_TOKENS', '_MAX_OUTPUT_TOKENS'):
            assert provider.prefix + suffix + '=' in text

def test_official_alias_can_resolve_to_its_dated_snapshot_only(monkeypatch):
    configure(monkeypatch, 'OPENAI', 'gpt-4.1')
    assert pc.matches_snapshot('gpt-4.1', 'gpt-4.1-2025-04-14', 'https://api.openai.com/v1')
    assert not pc.matches_snapshot('gpt-4.1', 'gpt-4.1-mini-2025-04-14')
    assert not pc.matches_snapshot('gpt-4.1', 'different-model')
    assert not pc.matches_snapshot('gpt-4.1', 'gpt-4.1-2025-04-14', 'https://another.example/v1')
    monkeypatch.setenv('WEBUI_V2_MODE', 'online')
    assert not pc.matches_snapshot('gpt-4.1', 'gpt-4.1-2025-04-14')

def test_native_claude_bridge_roundtrips_official_tool_call(monkeypatch, tmp_path):
    from agent.harness.messages_bridge import MessagesBridge
    from agent.rsi.engine.model_gateway import GatewayPolicy
    configure(monkeypatch, 'ANTHROPIC', 'claude-test')
    name = 'mcp__venusfactory__search_tools'
    def upstream(request):
        payload = json.loads(request.content)
        assert payload['model'] == 'claude-test' and 'thinking' not in payload
        assert payload['tools'][0]['function']['name'] == name
        return httpx.Response(200, json={
            'id': 'chatcmpl-test', 'model': 'claude-test',
            'choices': [{'index': 0, 'finish_reason': 'tool_calls', 'message': {
                'role': 'assistant', 'content': None,
                'tool_calls': [{'id': 'call-test', 'type': 'function', 'function': {'name': name, 'arguments': '{"query":"solubility"}'}}]}}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 8, 'total_tokens': 18}})
    async def run():
        policy = GatewayPolicy(model='claude-test', mode='science_agent', scientific_tools={'search_tools'}, max_tokens=128, thinking=None)
        async with MessagesBridge(policy=policy, upstream_url='https://api.anthropic.com/v1',
                                  api_key='test-secret-ANTHROPIC', output=tmp_path/'bridge.jsonl',
                                  transport=httpx.MockTransport(upstream)) as relay:
            async with httpx.AsyncClient() as client:
                response = await client.post(relay.base_url+'/messages', headers={'x-api-key':relay.token}, json={
                    'model':'claude-test','max_tokens':128, 'messages':[{'role':'user','content':'find tools'}],
                    'tools':[{'name':name,'input_schema':{'type':'object','properties':{'query':{'type':'string'}}}}]})
                assert response.status_code == 200, response.text
                assert response.json()['stop_reason'] == 'tool_use'
                assert response.json()['content'][0]['input'] == {'query':'solubility'}
    asyncio.run(run())

def test_implicit_migration_does_not_fallback_to_paid_model(monkeypatch):
    from backend.web_v2 import agent_execution
    row = configure(monkeypatch)
    monkeypatch.setattr(agent_execution, 'options', lambda: {
        'default': {'harness':'deepseek-harness','model_id':'missing-free'},
        'models':[{'id':row['id'],'free':False}],
        'combinations':[{'harness':'deepseek-harness','model_id':row['id'],'ready':True}]})
    with pytest.raises(ValueError, match='Choose a configured model'):
        agent_execution.admit({})
