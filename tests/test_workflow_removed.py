"""Removed graph workflows cannot be submitted or replayed in either mode."""
import time
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend.web_v2.model_platform import Platform, write, read
from backend.web_v2 import model_platform_api as api
from backend.web_v2 import compute_access


@pytest.fixture(params=['online', 'local'])
def platform(request, tmp_path, monkeypatch):
    monkeypatch.setenv('WEBUI_V2_MODE', request.param)
    monkeypatch.setenv('VENUS_RUNTIME_ROOT', str(tmp_path / 'runtime'))
    monkeypatch.setenv('VENUS_ACCESS_DB', str(tmp_path / 'access.sqlite'))
    monkeypatch.setenv('VENUS_WORKER_AUTOSTART', '0')
    value = Platform(tmp_path / 'platform')
    monkeypatch.setattr(api, 'service', value)
    return value, value.session(None)[1]


def test_graph_routes_are_absent_and_cannot_spend_quota(platform):
    value, owner = platform
    app = FastAPI(); app.include_router(api.router)
    routes = ['/workflows', '/workflows/run', '/workflows/' + 'a'*32,
              '/workflow-components', '/workflow-components/tool:read_fasta']
    with TestClient(app) as client:
        for route in routes:
            assert client.get('/api/model-platform' + route).status_code == 404
            assert client.post('/api/model-platform' + route, json={'graph': {}}).status_code == 404
    assert not list((value.root / 'jobs').glob('*.json'))
    with pytest.raises(ValueError, match='Unknown job'):
        value.start(owner, 'workflow', {'graph': {}}, launch=False)
    with pytest.raises(ValueError, match='no longer supported'):
        compute_access.enqueue(value.queue, owner, 'platform:workflow', 'cpu', {'command': ['true']}, 'test')
    assert not value.queue.list(owner)
    assert 'workflow' not in compute_access.ComputeStore().status(owner)
    for name in ('save_workflow', 'workflows', 'workflow', 'validate_graph', 'workflow_components'):
        assert not hasattr(value, name)


def test_historical_files_remain_readable_but_cannot_run_or_retry(platform, monkeypatch):
    value, owner = platform
    identity = 'a' * 32
    job = dict(id=identity, owner=owner, kind='workflow', payload={'graph': {'nodes': []}},
               status='completed', progress=100, message='Completed', created=time.time(), steps=[])
    write(value.path('jobs', identity), job)
    write(value.root / 'results' / (identity + '.json'), {'outputs': {'saved': {'rows': [{'x': 1}]}}})
    value.register_job(job)
    assert value.job(owner, identity)['status'] == 'completed'
    assert value.jobs(owner) == []
    with pytest.raises(FileNotFoundError): value.job('another-owner', identity)
    saved = (value.root / 'results' / (identity + '.json')).read_bytes()
    monkeypatch.setattr(value, 'predict', lambda *a: pytest.fail('A retired workflow may not execute a model'))
    value.execute(identity)
    assert value.job(owner, identity)['status'] == 'failed'
    assert 'no longer supported' in value.job(owner, identity)['message']
    with pytest.raises(ValueError, match='Unknown job'): value.retry_job(owner, identity)
    assert (value.root / 'results' / (identity + '.json')).read_bytes() == saved
