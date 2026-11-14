from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from backend.web_v2 import compute_access as access
from backend.web_v2.job_queue import JobQueue
from backend.web_v2.online_access import VisitorMiddleware


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv('VENUS_ACCESS_DB', str(tmp_path/'access.sqlite3'))
    monkeypatch.setenv('VENUS_WORKER_AUTOSTART', '0')
    monkeypatch.setenv('VENUS_ONLINE_IP_COMPUTE_STARTS_PER_MINUTE', '1000')
    monkeypatch.setattr(access, 'online', lambda: True)
    value = access.ComputeStore()
    value.bind('a', ('same-network', 'visitor-a'))
    value.bind('b', ('same-network', 'visitor-b'))
    return value


def test_daily_predictions_are_per_visitor_not_shared_ip(store):
    for i in range(20):
        store.admit('a'+str(i), 'a', {'prediction': 1})
    with pytest.raises(HTTPException) as error:
        store.admit('a-twenty-one', 'a', {'prediction': 1})
    assert error.value.status_code == 429
    assert store.status('a')['prediction']['remaining'] == 0
    assert store.status('b')['prediction']['remaining'] == 20
    store.admit('b-first', 'b', {'prediction': 1})
    assert store.status('b')['prediction']['remaining'] == 19


def test_concurrent_requests_cannot_exceed_twenty_predictions(store):
    def reserve(i):
        try:
            store.admit('parallel-'+str(i), 'a', {'prediction': 1})
            return True
        except HTTPException as error:
            assert error.status_code == 429
            return False
    with ThreadPoolExecutor(max_workers=8) as pool:
        accepted = list(pool.map(reserve, range(40)))
    assert sum(accepted) == 20
    assert store.status('a')['prediction'] == {'limit': 20, 'used': 20, 'remaining': 0}
    store.bind('another-entry-for-a', ('same-network', 'visitor-a'))
    with pytest.raises(HTTPException):
        store.admit('switch-entry', 'another-entry-for-a', {'prediction': 1})


def test_prediction_batch_reservation_is_atomic_and_old_workspaces_keep_identity(store):
    store.admit('nineteen', 'a', {'prediction': 19})
    with pytest.raises(HTTPException):
        store.admit('too-large', 'a', {'prediction': 2})
    assert store.status('a')['prediction']['used'] == 19
    store.bind('a', ('new-network', 'changed-cookie'))
    assert store.status('a')['prediction']['remaining'] == 1
    store.admit('fits', 'a', {'prediction': 1})
    assert not store.admit('fits', 'a', {'prediction': 1})


def test_day_rollover_and_network_burst_are_separate(store, monkeypatch):
    store.admit('old-day', 'a', {'prediction': 20})
    monkeypatch.setattr(store, 'day', lambda: ('2099-01-01', '2099-01-02T00:00:00+08:00'))
    assert store.status('a')['prediction']['remaining'] == 20
    monkeypatch.setenv('VENUS_ONLINE_IP_COMPUTE_STARTS_PER_MINUTE', '1')
    with pytest.raises(HTTPException) as error:
        store.admit('network-burst', 'b', {'prediction': 1})
    assert error.value.detail['code'] == 'COMPUTE_START_RATE_LIMIT'
    assert store.status('b')['prediction']['used'] == 0


def queued(q, owner, kind, metadata=None, key=None):
    return access.enqueue(q, owner, kind, 'cpu', {'command': ['true']}, 'test', metadata=metadata,
                          request_key=key, request_hash='a'*64 if key else None)


def test_web_and_mcp_share_quota_without_double_charging_reserved_children(store, tmp_path):
    q = JobQueue(tmp_path/'queue')
    parent, _ = queued(q, 'a', 'platform:tool', {'compute_demands': {'prediction': 2}})
    queued(q, 'a', 'mcp:model_prediction', {'tool': 'predict_protein_function', 'parent_job': parent})
    queued(q, 'a', 'mcp:molecular_docking', {'tool': 'vina_dock', 'parent_job': parent})
    assert store.status('a')['prediction']['used'] == 2
    queued(q, 'a', 'mcp:model_prediction', {'tool': 'predict_residue_function'})
    access.inline_prediction('a', 'predict_s4pred', 'sequence_analysis')
    assert store.status('a')['prediction']['used'] == 4


def test_idempotency_queue_failure_and_unowned_compute(store, tmp_path, monkeypatch):
    q = JobQueue(tmp_path/'queue')
    first = queued(q, 'a', 'platform:predict', key='same-request')
    again = queued(q, 'a', 'platform:predict', key='same-request')
    assert first[0] == again[0] and first[1] and not again[1]
    assert store.status('a')['prediction']['used'] == 1
    with pytest.raises(HTTPException) as error:
        queued(q, 'unbound', 'mcp:model_prediction')
    assert error.value.status_code == 403
    def unavailable(*args, **kwargs):
        raise ValueError('Job queue is full')
    monkeypatch.setattr(q, 'enqueue', unavailable)
    with pytest.raises(ValueError):
        queued(q, 'a', 'platform:predict')
    assert store.status('a')['prediction']['used'] == 1


def test_local_mode_has_no_daily_limit(store, tmp_path, monkeypatch):
    monkeypatch.setattr(access, 'online', lambda: False)
    q = JobQueue(tmp_path/'queue')
    queued(q, 'unbound-local', 'platform:predict')
    assert access.allowance() == {'mode': 'local', 'enforced': False}
    assert store.status('a')['prediction']['used'] == 0


def test_mcp_submission_enforces_shared_limit_before_any_model_worker_starts(store, tmp_path, monkeypatch):
    from pydantic import BaseModel
    from mcp_tools.scheduler import jobs
    class Input(BaseModel):
        sequence: str
    class Tool:
        metadata = None
        @staticmethod
        def get_input_schema():
            return Input
    monkeypatch.setattr(jobs, 'native_tools', lambda: {'predict_s4pred': Tool()})
    monkeypatch.setattr(jobs, 'assert_tool_call_allowed', lambda *args: None)
    q = JobQueue(tmp_path/'queue')
    scheduler = jobs.ToolJobs(q)
    with pytest.raises(ValueError):
        scheduler.submit('predict_s4pred', {}, owner='a')
    assert store.status('a')['prediction']['used'] == 0
    store.admit('used-in-model-plaza', 'a', {'prediction': 19})
    job = scheduler.submit('predict_s4pred', {'sequence':'ACDEFGHIK'}, owner='a')
    assert job['status'] == 'queued'
    assert store.status('a')['prediction']['used'] == 20
    with pytest.raises(HTTPException) as error:
        scheduler.submit('predict_s4pred', {'sequence':'ACDEFGHIK'}, owner='a')
    assert error.value.status_code == 429
    assert len(q.list('a')) == 1
    assert len(list((q.root/'mcp-inputs').glob('*.json'))) == 1


@pytest.mark.parametrize('value', ['', 'not-a-number', '0', '-1'])
def test_invalid_configuration_keeps_safe_defaults(store, monkeypatch, value):
    monkeypatch.setenv('WEBUI_V2_ONLINE_DAILY_PREDICTION_LIMIT', value)
    assert store.status('a')['prediction']['limit'] == 20


def test_legacy_sync_and_streaming_endpoints_share_twenty_without_double_count(store, monkeypatch):
    from pydantic import BaseModel
    from starlette.responses import StreamingResponse
    monkeypatch.delenv('VENUS_QUEUE_JOB_ID', raising=False)
    monkeypatch.delenv('VENUS_QUEUE_OWNER', raising=False)
    class Input(BaseModel):
        sequence: str
    app = FastAPI(); app.add_middleware(VisitorMiddleware)
    @app.post('/api/test/predict')
    @access.metered_prediction
    def predict(body: Input):
        return {'ok': True}
    @app.post('/api/test/stream')
    @access.metered_prediction
    async def stream(body: Input):
        async def output():
            predict(body); predict(body)
            yield 'data: completed\n\n'
        return StreamingResponse(output(), media_type='text/event-stream')
    with TestClient(app) as a, TestClient(app) as b:
        assert a.post('/api/test/predict', json={}).status_code == 422
        for _ in range(19):
            assert a.post('/api/test/predict', json={'sequence':'ACDEF'}).status_code == 200
        assert a.post('/api/test/stream', json={'sequence':'ACDEF'}).status_code == 200
        assert a.post('/api/test/predict', json={'sequence':'ACDEF'}).status_code == 429
        assert b.post('/api/test/predict', json={'sequence':'ACDEF'}).status_code == 200


def test_legacy_adapter_reuses_queue_reservation(store, tmp_path, monkeypatch):
    q = JobQueue(tmp_path/'queue'); monkeypatch.setenv('VENUS_JOB_QUEUE_ROOT', str(q.root))
    identity, _ = queued(q, 'a', 'platform:predict')
    monkeypatch.setenv('VENUS_QUEUE_OWNER', 'a'); monkeypatch.setenv('VENUS_QUEUE_JOB_ID', identity)
    @access.metered_prediction
    def predict():
        return 'done'
    assert predict() == 'done'
    assert store.status('a')['prediction']['used'] == 1


def test_researcher_children_share_prediction_quota_with_other_entries(store, tmp_path):
    q = JobQueue(tmp_path/'queue')
    store.admit('other-model-entry', 'a', {'prediction': 19})
    parent, _ = queued(q, 'a', 'protein-research')
    assert store.status('a')['prediction']['used'] == 19
    child, _ = queued(q, 'a', 'mcp:model_prediction', {
        'tool': 'predict_protein_function', 'parent_job': parent})
    assert q.get(child, 'a')['metadata']['parent_job'] == parent
    assert store.status('a')['prediction']['used'] == 20
    with pytest.raises(HTTPException) as error:
        queued(q, 'a', 'mcp:model_prediction', {
            'tool': 'predict_residue_function', 'parent_job': parent})
    assert error.value.detail['code'] == 'COMPUTE_DAILY_LIMIT_REACHED'
    assert len(q.list('a')) == 2
    assert store.status('b')['prediction']['remaining'] == 20


def test_standalone_tool_http_shares_prediction_quota_and_idempotency(store, tmp_path, monkeypatch):
    from backend.web_v2 import model_platform_api as api, plaza_tools
    from backend.web_v2.model_platform import Platform
    from mcp_tools.scheduler import catalog
    from types import SimpleNamespace
    monkeypatch.setenv('WEBUI_V2_MODE', 'online')
    service = Platform(tmp_path / 'platform'); monkeypatch.setattr(api, 'service', service)
    monkeypatch.setattr(plaza_tools, 'card', lambda name: {
        'id':'tool:' + name, 'name':'Fixture prediction', 'adapter':'registered-tool',
        'available':True, 'input_kind':'table', 'contract':{'properties':{}}})
    monkeypatch.setattr(catalog, 'native_tools', lambda: {'predict_s4pred':SimpleNamespace(metadata=None)})
    app = FastAPI(); app.add_middleware(VisitorMiddleware); app.include_router(api.router)
    with TestClient(app) as a, TestClient(app) as b:
        a.get('/api/model-platform/session'); b.get('/api/model-platform/session')
        data = a.post('/api/model-platform/datasets', files={'file':('protein.fasta', b'>x\nACDEFGHIK')}).json()
        payload = {'tool_id':'tool:predict_s4pred', 'dataset_id':data['id']}
        headers = {'Idempotency-Key':'same-tool-job'}
        first = a.post('/api/model-platform/tools/run', json=payload, headers=headers)
        assert first.status_code == 200, first.text
        again = a.post('/api/model-platform/tools/run', json=payload, headers=headers)
        assert again.status_code == 200 and again.json()['id'] == first.json()['id']
        for _ in range(19):
            response = a.post('/api/model-platform/tools/run', json=payload)
            assert response.status_code == 200, response.text
        assert a.post('/api/model-platform/tools/run', json=payload).status_code == 429
        assert a.get('/api/model-platform/allowance').json()['prediction']['remaining'] == 0
        assert b.get('/api/model-platform/allowance').json()['prediction']['remaining'] == 20
        assert b.get('/api/model-platform/jobs/' + first.json()['id']).status_code == 404
        assert b.post('/api/model-platform/tools/run', json=payload).status_code == 404
