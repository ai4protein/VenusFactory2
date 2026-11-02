"""Shared, bounded web retrieval for MCP, Scientist tools and Researcher reports."""
from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import re
import time
from urllib.parse import urlsplit, urlencode, urljoin

import httpx

class SearchError(ValueError):
    """Safe public error: never includes provider payloads or credential URLs."""


class _RedactSearchKey(logging.Filter):
    def filter(self, record):
        message = record.getMessage()
        if 'api_key=' in message:
            record.msg = re.sub(r'(?i)(api_key=)[^&\s\'"<>]+', r'\1[redacted]', message)
            record.args = ()
        return True


# SerpAPI requires a query parameter. httpx logs request URLs at INFO, so redact
# before records reach any configured handlers (including the preview log).
for _name in ('httpx', 'httpcore.http11', 'httpcore.http2', 'httpcore.connection', 'httpcore.proxy'):
    _logger = logging.getLogger(_name)
    if not any(isinstance(f, _RedactSearchKey) for f in _logger.filters):
        _logger.addFilter(_RedactSearchKey())


def credential(provider):
    names = ('SERPAPI_API_KEY', 'SERP_API_KEY') if provider == 'serpapi' else ('TAVILY_API_KEY',)
    return next((os.getenv(n, '').strip() for n in names if os.getenv(n, '').strip()), '')


def provider_order(provider='auto'):
    selected = (provider or 'auto').strip().lower()
    if selected == 'auto':
        selected = os.getenv('VENUS_SEARCH_PROVIDER', 'auto').strip().lower() or 'auto'
    if selected not in {'auto', 'tavily', 'serpapi', 'bing'}:
        raise SearchError('Unknown web search provider')
    if selected != 'auto':
        return [selected]
    return [p for p in ('tavily', 'serpapi') if credential(p)] or ['bing']


def search_options(provider='auto'):
    return {'provider': provider_order(provider)[0], 'routes': provider_order(provider),
            'tavily_configured': bool(credential('tavily')), 'serpapi_configured': bool(credential('serpapi')),
            'tools': ['query_web', 'query_serpapi', 'query_tavily']}


def plain(value, limit):
    return re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', ' ', str(value or '')))).strip()[:limit]


def safe_url(value):
    if not isinstance(value, str) or len(value) > 2500:
        return ''
    try:
        target = urlsplit(value)
        return value if target.scheme in {'http', 'https'} and target.hostname and not target.username and not target.password else ''
    except ValueError:
        return ''


async def _read(response):
    if response.status_code != 200:
        raise SearchError('Search provider returned HTTP ' + str(response.status_code))
    raw = bytearray()
    async for chunk in response.aiter_bytes():
        raw.extend(chunk)
        if len(raw) > 1_000_000:
            raise SearchError('Search response is too large')
    return bytes(raw)


async def _provider_search(provider, query, count, engine):
    from backend.web_v2.job_queue import resource_slot
    if provider != 'bing' and not credential(provider):
        raise SearchError(provider + ' is not configured')
    host = {'tavily': 'api.tavily.com', 'serpapi': 'serpapi.com', 'bing': 'www.bing.com'}[provider]
    async with resource_slot('api:' + host, kind='web-search'):
        async with httpx.AsyncClient(timeout=25, follow_redirects=False) as client:
            if provider == 'tavily':
                async with client.stream('POST', 'https://api.tavily.com/search',
                        headers={'Authorization': 'Bearer ' + credential(provider)},
                        json={'query': query, 'max_results': count, 'search_depth': 'basic', 'topic': 'general',
                              'include_answer': False, 'include_raw_content': False, 'auto_parameters': False}) as response:
                    data = json.loads(await _read(response))
                rows = data.get('results', [])
            elif provider == 'serpapi':
                async with client.stream('GET', 'https://serpapi.com/search.json', params={
                        'engine': engine, 'q': query, 'num': count, 'hl': 'en', 'output': 'json',
                        'api_key': credential(provider)}) as response:
                    data = json.loads(await _read(response))
                if data.get('error') or data.get('search_metadata', {}).get('status') == 'Error':
                    raise SearchError('SerpAPI could not complete this search')
                rows = data.get('organic_results', [])
            else:
                import xml.etree.ElementTree as ET
                url = 'https://www.bing.com/search?' + urlencode({'format': 'rss', 'q': query, 'setlang': 'en-us', 'cc': 'US'})
                for _ in range(3):
                    target = urlsplit(url)
                    if target.scheme != 'https' or target.hostname not in {'www.bing.com', 'cn.bing.com'} or target.path != '/search' or target.username or target.password:
                        raise SearchError('Unsupported search redirect')
                    async with client.stream('GET', url) as response:
                        if response.status_code in {301, 302, 303, 307, 308}:
                            url = urljoin(url, response.headers.get('location', '')); continue
                        raw = await _read(response)
                    rows = [{'title': x.findtext('title'), 'link': x.findtext('link'), 'snippet': x.findtext('description')}
                            for x in ET.fromstring(raw).findall('.//item')]
                    break
                else:
                    raise SearchError('Search redirect limit reached')
    result, seen = [], set()
    for row in rows:
        if not isinstance(row, dict): continue
        url = safe_url(row.get('url') or row.get('link'))
        if not url or url in seen: continue
        seen.add(url)
        text = plain(row.get('content') or row.get('snippet'), 3000)
        result.append({'title': plain(row.get('title'), 240), 'url': url, 'text': text, 'snippet': text,
                       'kind': 'web', 'source': provider, 'provider': provider,
                       'evidence_level': 'search_extract' if provider == 'tavily' else 'search_snippet'})
        if len(result) >= count: break
    return result


async def search_web(query, max_results=5, provider='auto', engine='google'):
    if not isinstance(query, str) or not 1 <= len(query.strip()) <= 500:
        raise SearchError('Use a search query of 1–500 characters')
    if type(max_results) is not int or not 1 <= max_results <= 10:
        raise SearchError('Use max_results between 1 and 10')
    if engine not in {'google', 'google_scholar'}:
        raise SearchError('Choose google or google_scholar')
    from mcp_tools.tools.runtime_tool_policy import assert_tool_allowed, is_tool_allowed
    assert_tool_allowed('query_web')
    providers = [p for p in provider_order(provider) if is_tool_allowed('query_' + p)]
    if engine == 'google_scholar':
        if provider not in {'auto', 'serpapi'} or not credential('serpapi'):
            raise SearchError('Google Scholar requires configured SerpAPI')
        assert_tool_allowed('query_serpapi')
        providers = ['serpapi']
    if not providers: raise SearchError('Web search is unavailable')
    query = query.strip()
    attempts = []
    for current in providers:
        try:
            rows = await asyncio.wait_for(_provider_search(current, query, max_results, engine), timeout=35)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            message = str(exc) if isinstance(exc, SearchError) else 'Search provider temporarily unavailable'
            attempts.append({'provider': current, 'status': 'unavailable', 'message': message})
            continue
        attempts.append({'provider': current, 'status': 'completed' if rows else 'empty'})
        if rows or current == providers[-1]:
            return {'query': query, 'provider': current, 'engine': engine if current == 'serpapi' else current,
                    'count': len(rows), 'results': rows, 'attempts': attempts}
    raise SearchError('; '.join(a['provider'] + ': ' + a.get('message', 'No results') for a in attempts))


async def query_web_async(query, max_results=5, provider='auto', engine='google'):
    from .deepsearch.response_utils import query_success_response, error_response
    started = time.perf_counter()
    try:
        value = await search_web(query, max_results, provider, engine)
        return query_success_response(json.dumps(value, ensure_ascii=False),
            query_time_ms=int((time.perf_counter()-started)*1000), source=value['provider'])
    except SearchError as exc:
        return error_response('SearchError', str(exc))


def query_web(query, max_results=5, provider='auto', engine='google'):
    return asyncio.run(query_web_async(query, max_results, provider, engine))
