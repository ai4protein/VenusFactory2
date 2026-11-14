import json
import time
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend.web_v2.model_platform import Platform, read
from backend.web_v2 import model_platform_api as api
from backend.web_v2 import plaza_tools


@pytest.fixture
def setup(tmp_path):
    p = Platform(tmp_path)
    owner = p.session(None)[1]
    p._cache, p._cache_time = (p.official(), []), time.monotonic()
    d = p.upload(owner, 'train.csv', ('x,target,sequence\n' + '\n'.join(f'{i},{i*2+3},ACDEFGHIK' for i in range(40))).encode())
    predict = p.upload(owner, 'predict.csv', b'x\n61\n62\n63')
    return p, owner, d, predict


def test_registered_tool_reads_managed_input_and_computes_properties(setup):
    p,owner,d,_=setup
    data=p.owned('datasets',d['id'],owner);data={**data,'rows':data['rows'][:2]}
    result=plaza_tools.run('calculate_physchem_from_fasta',data,{},p.root/'work')
    assert len(result['rows'])==2
    assert result['rows'][0]['x']=='0'
    assert 'file_info' not in result['rows'][0]
    assert any(isinstance(v,(int,float)) for v in result['rows'][0].values())


@pytest.mark.parametrize('config',[{'parameters':{'file_path':'/etc/passwd'}},{'columns':{'output_file':'x'}},{'parameters':{'api_key':'not-a-real-key'}}])
def test_managed_tool_parameters_cannot_be_overridden(setup,config):
    p,owner,d,_=setup
    with pytest.raises(ValueError,match='documented tool parameters'):
        plaza_tools.run('read_fasta',p.owned('datasets',d['id'],owner),config,p.root/'work')


def test_tool_error_not_reported_as_success():
    with pytest.raises(ValueError,match='missing'):
        plaza_tools.result_rows(json.dumps({'status':'error','error':{'message':'missing'}}))
    with pytest.raises(ValueError):plaza_tools.card('agent_generated_code')
    with pytest.raises(ValueError):plaza_tools.card('query_tavily')


@pytest.mark.parametrize('mode', ['online', 'local'])
def test_standalone_tool_submission_and_private_results(setup, monkeypatch, mode):
    p, owner, d, _ = setup
    monkeypatch.setenv('WEBUI_V2_MODE', mode)
    p._cache = (p.official(), [])
    job = p.start(owner, 'tool', {'tool_id': 'tool:read_fasta', 'dataset_id': d['id']}, launch=False)
    p.execute(job['id'])
    done = p.job(owner, job['id'])
    assert done['status'] == 'completed', done
    assert done['result']['rows'] == 40
    assert done['result']['preview'][0]['sequence'] == 'ACDEFGHIK'
    with pytest.raises(FileNotFoundError): p.job('other', job['id'])
    assert p.compute_demands(owner, 'tool', {'tool_id':'tool:read_fasta'}) == {'prediction':0}
    assert p.compute_demands(owner, 'tool', {'tool_id':'tool:predict_s4pred'}) == {'prediction':1}


def test_tool_endpoint_and_validation_before_enqueue(setup, monkeypatch):
    p, owner, data, _ = setup
    monkeypatch.setattr(api, 'service', p)
    app = FastAPI(); app.include_router(api.router)
    with TestClient(app) as client:
        card = client.get('/api/model-platform/tools/tool:query_pubmed')
        assert card.status_code == 200 and 'query' in card.json()['contract']['properties']
        assert client.get('/api/model-platform/tools/tool:query_tavily').status_code == 422
        assert client.post('/api/model-platform/tools/run', json={'tool_id':'tool:read_fasta', 'dataset_id':data['id']}).status_code == 404
    for payload in (
        {'tool_id':'tool:agent_generated_code'},
        {'tool_id':'tool:read_fasta'},
        {'tool_id':'tool:read_fasta', 'dataset_id':data['id'], 'config':{'parameters':{'file_path':'/etc/passwd'}}},
    ):
        with pytest.raises(ValueError): p.start(owner, 'tool', payload, launch=False)
    assert not list((p.root / 'jobs').glob('*.json'))


def test_tool_policy_rechecked_after_enqueue(setup, monkeypatch):
    from mcp_tools.tools.capability_policy import CapabilityStore
    p, owner, data, _ = setup
    monkeypatch.setenv('VENUS_RUNTIME_ROOT', str(p.root / 'runtime'))
    job = p.start(owner, 'tool', {'tool_id':'tool:read_fasta', 'dataset_id':data['id']}, launch=False)
    CapabilityStore().set('tool:read_fasta', False, 0, {}, 'fixture')
    p.execute(job['id'])
    assert p.job(owner, job['id'])['status'] == 'failed'
    assert not (p.root / 'results' / (job['id'] + '.json')).exists()


def test_standalone_tool_runs_in_separate_worker(tmp_path, monkeypatch):
    monkeypatch.setenv('WEBUI_V2_MODE', 'local')
    monkeypatch.setenv('VENUS_WORKER_AUTOSTART', '1')
    monkeypatch.setenv('VENUS_RUNTIME_ROOT', str(tmp_path / 'runtime'))
    p = Platform(tmp_path / 'platform'); monkeypatch.setattr(api, 'service', p)
    app = FastAPI(); app.include_router(api.router)
    with TestClient(app) as client:
        data = client.post('/api/model-platform/datasets', files={'file':('proteins.fasta', b'>test\nACDEFGHIK')}).json()
        response = client.post('/api/model-platform/tools/run', json={'tool_id':'tool:read_fasta', 'dataset_id':data['id']})
        assert response.status_code == 200, response.text
        job = response.json(); deadline = time.monotonic() + 45
        while job['status'] in {'queued', 'running'} and time.monotonic() < deadline:
            time.sleep(.2); job = client.get('/api/model-platform/jobs/' + job['id']).json()
        assert job['status'] == 'completed', job
        assert job['result']['preview'][0]['sequence'] == 'ACDEFGHIK'
        result = client.get(job['result']['download'])
        assert result.status_code == 200 and result.json()['rows'][0]['sequence'] == 'ACDEFGHIK'
