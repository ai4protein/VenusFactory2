import asyncio
from contextlib import asynccontextmanager
import json
import logging
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastmcp import Client

from mcp_tools.tools.search import web_provider as web
from mcp_tools.tools.search.tools_mcp import mcp
from mcp_tools.tools.search.tools_agent import query_web_tool, query_serpapi_tool
from backend.web_v2 import research_report


@pytest.fixture(autouse=True)
def search_environment(monkeypatch):
    for name in ('TAVILY_API_KEY', 'SERP_API_KEY', 'SERPAPI_API_KEY', 'VENUS_SEARCH_PROVIDER', 'VENUS_REPORT_SEARCH_PROVIDER'):
        monkeypatch.delenv(name, raising=False)
    @asynccontextmanager
    async def slot(*args, **kwargs):
        yield 'offline-test'
    monkeypatch.setattr('backend.web_v2.job_queue.resource_slot', slot)


@pytest.fixture
def mock_http(monkeypatch):
    real = httpx.AsyncClient
    def install(handler):
        monkeypatch.setattr(web.httpx, 'AsyncClient', lambda **kwargs: real(transport=httpx.MockTransport(handler), **kwargs))
    return install


def serp_results():
    return {'search_metadata': {'status': 'Success'}, 'organic_results': [
        {'title': '<b>Protein engineering</b>', 'link': 'https://example.org/paper', 'snippet': 'Measured protein stability.'},
        {'title': 'Duplicate', 'link': 'https://example.org/paper'},
        {'title': 'Unsafe', 'link': 'javascript:alert(1)'},
        {'title': 'Credentials', 'link': 'https://user:pass@example.org/paper'},
        {'title': 'Second', 'link': 'https://example.org/dataset', 'snippet': 'Public dataset.'},
    ]}


def unpack(text):
    envelope = json.loads(text)
    assert envelope['status'] == 'success', envelope
    return json.loads(envelope['content'])


def test_key_aliases_and_explicit_provider_order(monkeypatch):
    monkeypatch.setenv('SERP_API_KEY', 'alias-test')
    assert web.credential('serpapi') == 'alias-test'
    assert web.provider_order() == ['serpapi']
    monkeypatch.setenv('SERPAPI_API_KEY', 'canonical-test')
    monkeypatch.setenv('TAVILY_API_KEY', 'tavily-test')
    assert web.credential('serpapi') == 'canonical-test'
    assert web.provider_order() == ['tavily', 'serpapi']
    monkeypatch.setenv('VENUS_SEARCH_PROVIDER', 'serpapi')
    assert web.provider_order() == ['serpapi']
    assert web.provider_order('tavily') == ['tavily']
    assert 'canonical-test' not in json.dumps(web.search_options())


@pytest.mark.asyncio
async def test_serpapi_parameters_normalization_and_log_redaction(monkeypatch, mock_http, caplog):
    monkeypatch.setenv('SERP_API_KEY', 'private-test-key')
    def handle(request):
        assert request.url.host == 'serpapi.com'
        assert request.url.params['api_key'] == 'private-test-key'
        assert request.url.params['engine'] == 'google_scholar'
        assert request.url.params['q'] == 'PET hydrolase'
        assert request.url.params['num'] == '2'
        return httpx.Response(200, json=serp_results())
    mock_http(handle)
    with caplog.at_level(logging.INFO, logger='httpx'):
        result = await web.search_web('PET hydrolase', 2, 'serpapi', 'google_scholar')
    assert result['count'] == 2
    assert result['results'][0]['title'] == 'Protein engineering'
    assert result['results'][0]['text'] == 'Measured protein stability.'
    assert result['results'][0]['evidence_level'] == 'search_snippet'
    assert 'private-test-key' not in caplog.text + json.dumps(result)
    assert '[redacted]' in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize('first_status', [429, 200])
async def test_auto_falls_back_on_error_or_empty(monkeypatch, mock_http, first_status):
    monkeypatch.setenv('TAVILY_API_KEY', 'tavily-test')
    monkeypatch.setenv('SERP_API_KEY', 'serp-test')
    calls = []
    def handle(request):
        calls.append(request.url.host)
        if request.url.host == 'api.tavily.com':
            return httpx.Response(first_status, json={'results': []})
        return httpx.Response(200, json=serp_results())
    mock_http(handle)
    result = await web.search_web('protein')
    assert calls == ['api.tavily.com', 'serpapi.com']
    assert result['provider'] == 'serpapi'
    assert len(result['attempts']) == 2


@pytest.mark.asyncio
async def test_explicit_provider_does_not_switch_or_echo_failure(monkeypatch, mock_http):
    monkeypatch.setenv('TAVILY_API_KEY', 'tavily-test')
    monkeypatch.setenv('SERP_API_KEY', 'private-test-key')
    seen = []
    def handle(request):
        seen.append(request.url.host)
        return httpx.Response(200, json={'error': 'Do not expose private-test-key'})
    mock_http(handle)
    value = json.loads(await web.query_web_async('protein', provider='serpapi'))
    assert value['status'] == 'error'
    assert 'private-test-key' not in json.dumps(value)
    assert seen == ['serpapi.com']


@pytest.mark.asyncio
async def test_invalid_inputs_do_not_make_requests(mock_http):
    def handle(request): raise AssertionError('Must validate before HTTP')
    mock_http(handle)
    for kwargs in ({'query': ''}, {'query': 'x'*501}, {'query': 'protein', 'max_results': 11},
                   {'query': 'protein', 'engine': 'google_gemini'}, {'query': 'protein', 'provider': 'arbitrary'}):
        with pytest.raises(web.SearchError): await web.search_web(**kwargs)


@pytest.mark.asyncio
async def test_large_response_and_redirect_are_errors(monkeypatch, mock_http):
    monkeypatch.setenv('SERP_API_KEY', 'test-only')
    for response in (httpx.Response(302, headers={'location': 'https://example.org'}), httpx.Response(200, content=b'x'*1_000_001)):
        mock_http(lambda request: response)
        with pytest.raises(web.SearchError): await web.search_web('protein', provider='serpapi')


@pytest.mark.asyncio
async def test_cancel_is_not_treated_as_provider_failure(monkeypatch):
    monkeypatch.setenv('SERP_API_KEY', 'test-only')
    with patch.object(web, '_provider_search', new=AsyncMock(side_effect=asyncio.CancelledError)):
        with pytest.raises(asyncio.CancelledError): await web.search_web('protein')


@pytest.mark.asyncio
async def test_agent_mcp_scientist_and_researcher_use_same_serpapi(monkeypatch, mock_http):
    monkeypatch.setenv('SERP_API_KEY', 'test-only')
    monkeypatch.setenv('VENUS_SEARCH_PROVIDER', 'serpapi')
    seen = []
    def handle(request):
        seen.append(request.url.host)
        return httpx.Response(200, json=serp_results())
    mock_http(handle)
    async with Client(mcp) as client:
        catalog = {t.name: t for t in await client.list_tools()}
        assert {'query_web', 'query_serpapi', 'query_tavily'} <= catalog.keys()
        assert 'api_key' not in catalog['query_serpapi'].inputSchema['properties']
        response = await client.call_tool('query_serpapi', {'query': 'protein', 'max_results': 2})
        agent = unpack(response.content[0].text)
    scientist = unpack(await query_serpapi_tool.ainvoke({'query': 'protein', 'max_results': 2}))
    researcher = await research_report.web_search('protein')
    assert agent['provider'] == scientist['provider'] == researcher[0]['provider'] == 'serpapi'
    assert seen == ['serpapi.com']*3
    assert agent['results'][0]['url'] == scientist['results'][0]['url'] == researcher[0]['url']


def test_hub_and_discovery_expose_new_tools_in_online_mode(monkeypatch):
    monkeypatch.setattr('mcp_tools.tools.runtime_tool_policy.is_online_mode', lambda: True)
    from mcp_tools.tools.tools_agent_hub import get_tools, PI_SEARCH_TOOL_NAMES
    from mcp_tools.tools.runtime_tool_policy import filter_tools
    from mcp_tools.tools.discovery.service import search
    tools = filter_tools(get_tools())[0]
    names = [t.name for t in tools]
    assert len(names) == len(set(names))
    assert {'query_web', 'query_serpapi'} <= set(names)
    assert {'query_web', 'query_serpapi'} <= PI_SEARCH_TOOL_NAMES
    value = search('serpapi', kind='tools')
    assert 'query_serpapi' in json.dumps(value)


def test_harness_guidance_names_only_registered_search_tools():
    from agent.harness.capability_guidance import discovery_instructions
    text = discovery_instructions(['search_research_tools', 'mcp__venusfactory__query_web', 'mcp__venusfactory__query_serpapi'])
    assert 'mcp__venusfactory__query_serpapi' in text
    assert 'google_scholar' in text


@pytest.mark.asyncio
async def test_report_override_and_shared_selection(monkeypatch, mock_http):
    monkeypatch.setenv('SERP_API_KEY', 'test-only')
    monkeypatch.setenv('TAVILY_API_KEY', 'tavily-test')
    monkeypatch.setenv('VENUS_REPORT_SEARCH_PROVIDER', 'serpapi')
    assert research_report.search_provider() == 'serpapi'
    mock_http(lambda request: httpx.Response(200, json=serp_results()))
    assert (await research_report.web_search('protein'))[0]['provider'] == 'serpapi'
