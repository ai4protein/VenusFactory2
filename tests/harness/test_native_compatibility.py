import json
from dataclasses import replace

import httpx
import pytest

from agent.harness.codex_adapter import CodexEvents
from agent.harness.claude_events import ClaudeEvents
from agent.harness.gateway import EvidenceGateway
from agent.harness.messages_bridge import MessagesBridge, chat_to_message
from agent.harness.catalog import ExecutionCatalog
from agent.harness.contracts import Profile, ModelBinding, Budget
from agent.rsi.engine.model_gateway import GatewayPolicy


@pytest.mark.parametrize('content', [
    '# README\nThe experiment produced a model.',
    '{"saved": true}\nExecution budget: 3 tool calls remain.',
    '[1, 2, 3]',
    '',
])
def test_web_codex_accepts_text_and_multiblock_mcp_results(content):
    from test_codex_adapter import initialized, tool_event
    state=initialized();state.consume(tool_event())
    parts=[{'type':'text','text':part}for part in content.split('\n')]
    event=state.consume(tool_event('item.completed', result={'content':parts}))[0]
    assert event['output']==content and not event['is_error']
    state.consume({'type':'turn.completed'})
    assert state.turn_completed


@pytest.mark.parametrize('content', ['# README\nA protein model.', '{"saved": true}\nBudget: 2 calls'])
def test_web_claude_accepts_text_and_multiblock_mcp_results(content):
    from test_claude_events import init,call,result,terminal
    state=ClaudeEvents({'protein_probe'},1)
    state.consume(init());state.consume(call());event=result()
    event['message']['content'][0]['content']=[{'type':'text','text':part}for part in content.split('\n')]
    assert state.consume(event)[0]['output']==content
    state.consume(terminal());assert state.completed


def test_typed_research_receipts_still_require_valid_json_objects():
    from test_codex_adapter import initialized,tool_event
    state=initialized();state.validator=lambda name,data:None;state.consume(tool_event())
    with pytest.raises(ValueError):
        state.consume(tool_event('item.completed',result={'content':[{'type':'text','text':'not a scientific receipt'}]}))


def test_typed_receipt_keeps_its_canonical_serialization():
    from test_codex_adapter import initialized, tool_event
    state = initialized()
    state.validator = lambda name, data: None
    state.consume(tool_event())
    result = state.consume(tool_event('item.completed', result={
        'content': [{'type': 'text', 'text': '{\n  "value": 1\n}'}],
    }))[0]
    assert result['output'] == '{"value": 1}'


@pytest.mark.asyncio
async def test_deepseek_local_plugin_metadata_is_not_sent_to_openrouter(tmp_path):
    model='nvidia/nemotron-3.5-lightning:free';requests=[]
    async def upstream(request):
        body=json.loads(request.content);requests.append(body)
        assert 'dsh_plugin_packages' not in body
        assert body['provider']['max_price']=={'prompt':0,'completion':0,'request':0,'image':0}
        assert body['provider']['allow_fallbacks'] is False
        return httpx.Response(200,json={'model':model.removesuffix(':free'),'choices':[{'index':0,'message':{'role':'assistant','content':'OK'},'finish_reason':'stop'}], 'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2,'cost':0}})
    async with EvidenceGateway(policy=GatewayPolicy(model=model,mode='science_agent',scientific_tools=set(),thinking='disabled'),upstream_url='https://openrouter.ai/api/v1',api_key='fixture',output=tmp_path/'requests.jsonl',transport=httpx.MockTransport(upstream))as gateway:
        async with httpx.AsyncClient(trust_env=False)as client:
            body={'model':model,'messages':[], 'dsh_plugin_packages':{'private_local_inventory':'not provider input'}}
            response=await client.post(gateway.base_url+'/chat/completions',json=body,headers={'Authorization':'Bearer '+gateway.token})
            assert response.status_code==200,response.text
            response=await client.post(gateway.base_url+'/chat/completions',json={**body,'models':['paid-fallback']},headers={'Authorization':'Bearer '+gateway.token})
            assert response.status_code==400
    assert len(requests)==1
    assert 'private_local_inventory' not in (tmp_path/'requests.jsonl').read_text()


def test_claude_accepts_only_the_same_openrouter_free_model_alias():
    from test_messages_bridge import answer,TOOL
    model='nvidia/nemotron-3.5-lightning:free';raw=answer();raw['model']=model.removesuffix(':free')
    kwargs=dict(model=model,allowed_tools={TOOL},max_tokens=4096,upstream_url='https://openrouter.ai/api/v1')
    message,_=chat_to_message(json.dumps(raw).encode(),**kwargs)
    assert message['model']==model
    with pytest.raises(ValueError,match='identity mismatch'):
        chat_to_message(json.dumps(raw).encode(),**{**kwargs,'upstream_url':'https://unrelated.invalid/v1'})
    raw['model']='another/model'
    with pytest.raises(ValueError,match='identity mismatch'):
        chat_to_message(json.dumps(raw).encode(),**kwargs)


@pytest.mark.asyncio
async def test_claude_normalizes_only_advertised_tool_names(tmp_path):
    from test_messages_bridge import policy,request,answer,TOOL
    raw=answer();raw['choices'][0]['message']['tool_calls'][0]['function']['name']='protein_probe'
    async def upstream(request):return httpx.Response(200,json=raw)
    async with MessagesBridge(policy=policy(),upstream_url='https://fixture.invalid/v1',api_key='fixture',output=tmp_path/'requests.jsonl',transport=httpx.MockTransport(upstream))as gateway:
        async with httpx.AsyncClient(trust_env=False)as client:
            body=request();body['stream']=False
            response=await client.post(gateway.base_url+'/messages',json=body,headers={'Authorization':'Bearer '+gateway.token})
            assert response.status_code==200,response.text
            assert response.json()['content'][0]['name']==TOOL


def test_catalog_enforces_output_limit_and_declared_protocols(tmp_path):
    binding=ModelBinding('openrouter','fixture/model:free','https://openrouter.ai/api/v1')
    path=tmp_path/'catalog.json';path.write_text(json.dumps({'models':[{'id':'fixture','label':'Fixture','binding':binding.__dict__,'max_output_tokens':1024,'protocols':['chat_completions']}]}))
    cat=ExecutionCatalog(tmp_path,Profile('science_agent','deepseek-harness',binding,Budget(output_tokens=8192)),path=path)
    assert cat.select({'harness':'kimi-code','model_id':'fixture'})[0].budget.output_tokens==1024
    with pytest.raises(ValueError,match='protocol not enabled'):
        cat.select({'harness':'codex','model_id':'fixture'})


@pytest.mark.asyncio
@pytest.mark.parametrize('live, expected', [(True, 400), (False, 429)])
async def test_live_rate_limit_stops_sdk_retries_but_keeps_upstream_evidence(tmp_path, live, expected):
    async def upstream(request):
        return httpx.Response(429, json={'error': {'message': 'Provider temporarily busy'}})

    async def preview(*args):
        pass

    path = tmp_path / 'requests.jsonl'
    async with EvidenceGateway(
        policy=GatewayPolicy(model='fixture', mode='science_agent', scientific_tools=set()),
        upstream_url='https://fixture.invalid/v1', api_key='fixture', output=path,
        on_text=preview if live else None, transport=httpx.MockTransport(upstream),
    ) as gateway:
        async with httpx.AsyncClient(trust_env=False) as client:
            result = await client.post(gateway.base_url + '/chat/completions',
                json={'model': 'fixture', 'messages': []},
                headers={'Authorization': 'Bearer ' + gateway.token})
            await client.post(gateway.base_url + '/chat/completions',
                json={'model': 'fixture', 'messages': []},
                headers={'Authorization': 'Bearer ' + gateway.token})
        assert gateway.calls == (1 if live else 2)
        assert result.status_code == expected
        assert result.json()['error'] == 'upstream_http_429'
    evidence = [json.loads(line) for line in path.read_text().splitlines()]
    assert next(row for row in evidence if row['event'] == 'response')['status'] == 429


@pytest.mark.parametrize('status,error,code', [
    (429, None, 'upstream_rate_limited'),
    (403, None, 'model_access_403'),
    (404, None, 'model_unavailable'),
    (200, 'TimeoutError', 'upstream_timeout'),
    (200, 'ValueError', 'model_response_invalid'),
])
def test_failure_hints_do_not_expose_provider_bodies(tmp_path, status, error, code):
    from agent.harness.failure_summary import public_failure
    path = tmp_path / 'requests.jsonl'
    path.write_text(json.dumps({'event': 'response', 'completed': False, 'status': status,
                               'error_type': error, 'body': 'private-provider-text'}) + '\n')
    result = public_failure(path)
    assert result['code'] == code
    assert 'private-provider-text' not in json.dumps(result)


def test_old_recovered_failures_are_not_reported_as_current_failures(tmp_path):
    from agent.harness.failure_summary import public_failure
    path = tmp_path / 'requests.jsonl'
    path.write_text('\n'.join(json.dumps(row) for row in [
        {'event': 'response', 'status': 429, 'completed': False},
        {'event': 'response', 'status': 200, 'completed': True},
    ]))
    assert public_failure(path)['code'] == 'runtime_error'
