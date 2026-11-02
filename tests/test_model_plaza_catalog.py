import time
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend.web_v2.model_platform import Platform, write
from backend.web_v2 import model_platform_api as api


@pytest.fixture
def setup(tmp_path):
    p = Platform(tmp_path)
    owner = p.session(None)[1]
    p._cache, p._cache_time = (p.official(), []), time.monotonic()
    return p, owner


def test_full_catalog_includes_tools_models_and_backbones(setup):
    p, owner = setup
    result = p.plaza_catalog(owner, limit=100000)
    ids = [m['id'] for m in result['items']]
    assert len(ids) == len(set(ids)) == result['total']
    assert 'tool:query_pubmed' in ids and 'tool:read_fasta' in ids
    assert any(m['adapter'] == 'foundation' for m in result['items'])
    assert sum(result['categories'].values()) == result['total']
    a = p.plaza_catalog(owner, limit=10)
    b = p.plaza_catalog(owner, offset=10, limit=10)
    assert not {m['id'] for m in a['items']} & {m['id'] for m in b['items']}
    assert a['total'] == b['total'] == result['total']


def test_function_source_type_and_search_filters(setup):
    p, owner = setup
    tools = p.plaza_catalog(owner, category='processing', kind='tool')
    assert tools['items'] and all(m['category'] == 'processing' and m['node_kind'] == 'tool' for m in tools['items'])
    result = p.plaza_catalog(owner, query='PubMed', source='platform', kind='tool')
    assert [m['id'] for m in result['items']] == ['tool:query_pubmed']
    assert sum(result['categories'].values()) == 1
    assert not p.plaza_catalog(owner, query='PubMed', kind='model')['items']


def test_private_models_stay_owner_scoped_and_mine_excludes_tools(setup):
    p, owner = setup
    identity = 'a' * 32
    write(p.path('models', identity), dict(id=identity, owner=owner, name='Private activity', description='Fixture',
                                         category='function', visibility='private'))
    assert p.plaza_catalog(owner, mine=True)['items'][0]['id'] == 'user:' + identity
    assert p.plaza_catalog(owner, mine=True)['total'] == 1
    assert not p.plaza_catalog('other', mine=True)['items']
    assert not p.plaza_catalog('other', query='Private activity')['items']
    assert not p.plaza_catalog(owner, mine=True, kind='tool')['items']


def test_legacy_modes_return_the_same_complete_catalog(setup):
    p, owner = setup
    quick = p.catalog(owner, mode='quick', limit=100000)
    full = p.catalog(owner, mode='advanced', limit=100000)
    assert quick == full
    assert any(m['adapter'] == 'foundation' for m in quick['items'])


def test_http_catalog_and_tool_drawer_execution(setup, monkeypatch):
    p, _ = setup
    monkeypatch.setattr(api, 'service', p)
    start = p.start
    monkeypatch.setattr(p, 'start', lambda owner, kind, body: start(owner, kind, body, launch=False))
    app = FastAPI(); app.include_router(api.router)
    with TestClient(app) as client:
        result = client.get('/api/model-platform/catalog?category=processing&kind=tool')
        assert result.status_code == 200 and result.json()['items']
        assert client.get('/api/model-platform/catalog?kind=unknown').status_code == 422
        assert client.get('/api/model-platform/catalog?limit=101').status_code == 422
        d = client.post('/api/model-platform/datasets', files={'file': ('example.fasta', b'>fixture\nACDEFGHIK')}).json()
        graph = {'nodes': [
            {'id': 'input', 'kind': 'data', 'dataset_id': d['id']},
            {'id': 'tool', 'kind': 'tool', 'model_id': 'tool:read_fasta', 'config': {'parameters': {}, 'sequence_column': 'sequence'}},
            {'id': 'result', 'kind': 'output'},
        ], 'edges': [{'source': 'input', 'target': 'tool'}, {'source': 'tool', 'target': 'result'}]}
        response = client.post('/api/model-platform/workflows/run', json={'graph': graph})
        assert response.status_code == 200, response.text
        job = response.json(); p.execute(job['id'])
        done = client.get('/api/model-platform/jobs/' + job['id']).json()
        assert done['status'] == 'completed', done
        assert done['result']['preview'][0]['sequence'] == 'ACDEFGHIK'
