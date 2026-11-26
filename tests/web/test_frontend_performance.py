import asyncio
from types import SimpleNamespace
import httpx
from starlette.applications import Starlette
from starlette.responses import StreamingResponse
from starlette.routing import Mount, Route
from backend.web_v2.frontend_assets import frontend_assets

def test_assets_compress_and_cache_but_api_streams_are_untouched(tmp_path):
    body = 'const test = "' + 'abcdef' * 5000 + '";'
    (tmp_path / 'index-abc12345.js').write_text(body)
    (tmp_path / 'plain.js').write_text(body)
    async def stream(request):
        async def events():
            yield 'data: first\n\n'
            yield 'data: second\n\n'
        return StreamingResponse(events(), media_type='text/event-stream')
    app = Starlette(routes=[Mount('/assets', app=frontend_assets(tmp_path)), Route('/events', stream)])
    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            result = await client.get('/assets/index-abc12345.js', headers={'accept-encoding': 'gzip'})
            assert result.status_code == 200 and result.text == body
            assert result.headers['content-encoding'] == 'gzip'
            assert 'immutable' in result.headers['cache-control']
            plain = await client.get('/assets/plain.js')
            assert plain.headers['cache-control'] == 'no-cache'
            identity = await client.get('/assets/index-abc12345.js', headers={'accept-encoding': 'identity'})
            assert 'content-encoding' not in identity.headers
            assert identity.text == body
            cached = await client.get('/assets/index-abc12345.js', headers={'if-none-match': identity.headers['etag']})
            assert cached.status_code == 304 and 'immutable' in cached.headers['cache-control']
            stream = await client.get('/events', headers={'accept-encoding': 'gzip'})
            assert 'content-encoding' not in stream.headers
            assert stream.text == 'data: first\n\ndata: second\n\n'
    asyncio.run(check())

def test_composer_catalog_skips_legacy_evidence_checks(monkeypatch):
    from backend.web_v2 import science_goals as goals
    service = object.__new__(goals.ScienceGoalService)
    calls = []
    def driver(identity, cases):
        return SimpleNamespace(id=identity, title=identity, criteria=(), cases=cases,
                               case_details=lambda profile: calls.append(identity) or {})
    service.drivers = {'new': driver('autonomous_research.v1', ()), 'legacy': driver('legacy', ('case',))}
    service.execution_catalog = lambda: SimpleNamespace(public=lambda runtime: {'combinations': [{'ready': True}]})
    service.runtime_paths = lambda: {}
    service.profile = lambda: SimpleNamespace(model=SimpleNamespace(model='free-model'))
    monkeypatch.setattr(goals, 'driver_limits', lambda driver: {})
    compact = service.catalog(False)
    assert compact['available'] and len(compact['drivers']) == 1
    assert calls == ['autonomous_research.v1']
    calls.clear()
    assert len(service.catalog()['drivers']) == 2
    assert calls == ['autonomous_research.v1', 'legacy']
