import hashlib
import os
from backend.web_v2.job_access import cookie_name
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.testclient import TestClient
from backend.web_v2.job_queue import JobQueue
from backend.web_v2.job_records import JobRecords


def test_job_numbers_unique_persistent_and_owner_scoped(tmp_path):
    queue = JobQueue(tmp_path/'queue')
    records = JobRecords(queue)
    def register(i):
        return records.register('platform', f'{i:032x}', 'alice', 'predict', 'Solubility', 1793318400)['job_number']
    with ThreadPoolExecutor(max_workers=8) as workers:
        numbers = list(workers.map(register, range(30)))
    assert len(set(numbers)) == 30
    assert register(3) == numbers[3]
    recreated = JobRecords(JobQueue(tmp_path/'queue'))
    assert recreated.resolve('platform', numbers[3].lower(), 'alice') == f'{3:032x}'
    with pytest.raises(FileNotFoundError): recreated.resolve('platform', numbers[3], 'bob')
    with pytest.raises(FileNotFoundError): recreated.resolve('custom', numbers[3], 'alice')


def test_artifact_ownership_includes_aliases_and_denies_unclaimed_private_files(tmp_path):
    records = JobRecords(JobQueue(tmp_path/'queue'))
    private = tmp_path/'private'; private.mkdir()
    records.protect(private); records.grant(private/'alice', 'platform', 'alice')
    (private/'alice').mkdir(); file = private/'alice'/'results.csv'; file.write_text('secret')
    records.check_artifact(file, {'platform':'alice'})
    for owners in [{}, {'platform':'bob'}, {'custom':'alice'}]:
        with pytest.raises(FileNotFoundError): records.check_artifact(file, owners)
    link = tmp_path/'public-link.csv'; link.symlink_to(file)
    with pytest.raises(FileNotFoundError): records.check_artifact(link, {})
    with pytest.raises(FileNotFoundError): records.check_artifact(private/'unclaimed.csv', {'platform':'alice'})
    records.check_artifact(tmp_path/'public-model.pt', {})


@pytest.fixture
def platform_app(tmp_path, monkeypatch):
    from backend.web_v2.model_platform import Platform
    from backend.web_v2 import model_platform_api as api
    service = Platform(tmp_path/'platform'); monkeypatch.setattr(api, 'service', service)
    app = FastAPI(); app.include_router(api.router)
    return app, service


def make_prediction(client, service):
    client.get('/api/model-platform/session')
    owner = hashlib.sha256(client.cookies.get(cookie_name('platform')).encode()).hexdigest()
    data = client.post('/api/model-platform/datasets', files={'file':('input.fa',b'>sample\nACDEFGHIKLMN')}).json()
    model = next(m for m in service.official() if m['adapter']=='descriptors')
    job = service.start(owner, 'predict', {'dataset_id': data['id'], 'model_id': model['id']}, launch=False)
    service.execute(job['id'])
    return service.job(owner, job['id']), owner


def test_prediction_detail_search_download_cancel_retry_are_private(platform_app):
    app, service = platform_app
    with TestClient(app) as alice, TestClient(app) as bob:
        job, owner = make_prediction(alice, service)
        number = job['job_number']
        assert number.startswith('VF-') and job['title']=='Protein descriptors'
        assert alice.get('/api/model-platform/jobs/'+number).json()['id']==job['id']
        assert alice.get('/api/model-platform/jobs/'+number+'/result').status_code==200
        for identity in [number, job['id']]:
            for suffix in ['', '/result']:
                assert bob.get('/api/model-platform/jobs/'+identity+suffix).status_code==404
            for suffix in ['/cancel', '/retry']:
                assert bob.post('/api/model-platform/jobs/'+identity+suffix,json={'owner':owner}).status_code==404
        own = alice.get('/api/model-platform/job-history',params={'query':number}).json()
        assert own['total']==1 and own['items'][0]['job_number']==number
        assert 'result' not in own['items'][0] and 'owner' not in own['items'][0]
        other = bob.get('/api/model-platform/job-history',params={'query':number}).json()
        assert other['items']==[] and other['total']==0 and other['active']==0
        assert 'no-store' in alice.get('/api/model-platform/job-history').headers['cache-control']
        assert bob.post('/api/model-platform/jobs/'+number+'/cancel',headers={'Sec-Fetch-Site':'cross-site'}).status_code==403


def test_job_history_filters_and_pagination(platform_app):
    app, service = platform_app
    with TestClient(app) as client:
        for _ in range(3):make_prediction(client,service)
        page=client.get('/api/model-platform/job-history',params={'limit':2,'offset':1,'kind':'predict','status':'completed'}).json()
        assert page['total']==3 and len(page['items'])==2
        assert client.get('/api/model-platform/job-history',params={'status':'failed'}).json()['total']==0
        assert client.get('/api/model-platform/job-history',params={'kind':'train'}).json()['total']==0


def test_legacy_cookie_migration_preserves_jobs(platform_app):
    app,service=platform_app
    token,owner=service.session(None)
    with TestClient(app) as client:
        client.cookies.set('vf_model_workspace',token,domain='testserver.local',path='/api/model-platform')
        response=client.get('/api/model-platform/job-history')
        assert response.status_code==200
        from backend.web_v2.job_access import cookie_name
        assert client.cookies.get(cookie_name('platform'))==token
        assert [c.path for c in client.cookies.jar if c.name==cookie_name('platform')]==['/api']
        assert service.session(token)[1]==owner


@pytest.mark.parametrize("mode", ["local", "online"])
def test_custom_first_request_stream_cookie_and_artifact_privacy(tmp_path,monkeypatch,mode):
    from backend.web_v2 import custom_model_api as api
    from backend.web_v2.job_access import guard_artifact
    monkeypatch.setenv('WEBUI_V2_MODE',mode)
    monkeypatch.setenv('TEMP_OUTPUTS_DIR',str(tmp_path/'files'))
    monkeypatch.setenv('VENUS_JOB_QUEUE_ROOT',str(tmp_path/'queue'))
    monkeypatch.setattr(api,'_require_training_model',lambda args:None)
    monkeypatch.setattr(api,'_normalize_training_args',lambda args,_:args)
    monkeypatch.setattr(api,'_safe_read_json',lambda *_:{})
    monkeypatch.setattr(api,'_build_exec_cmd',lambda *_:[sys.executable,'-u','-c','print("private training log",flush=True)'])
    monkeypatch.setattr(api,'_ALLOWED_FILE_ROOTS',[tmp_path])
    app=FastAPI();app.include_router(api.router)
    @app.get('/api/test-generic-result')
    def result(request:Request,path:str):
        guard_artifact(path,request)
        return FileResponse(path)
    with TestClient(app) as alice,TestClient(app) as bob:
        # No bootstrap GET: streaming responses must retain both cookie headers.
        response=alice.post('/api/custom-model/training/start',json={'args':{'output_dir':str(tmp_path/'training'),'num_epochs':1}})
        assert response.status_code==200 and 'private training log' in response.text
        assert alice.cookies.get(cookie_name('custom'))
        job=alice.get('/api/custom-model/job-history').json()['items'][0]
        number=job['job_number'];identity=job['id']
        assert alice.get('/api/custom-model/jobs/'+number).json()['id']==identity
        assert bob.get('/api/custom-model/jobs/'+number).status_code==404
        assert bob.get('/api/custom-model/job-history',params={'query':number}).json()['total']==0
        for operation in ['stream','cancel','retry']:
            assert bob.post(f'/api/custom-model/jobs/{number}/{operation}').status_code==404
        queue_job=JobQueue().get(identity);folder=Path(queue_job['metadata']['output_root']);folder.mkdir(parents=True,exist_ok=True)
        artifact=folder/'predictions.csv';artifact.write_text('sequence,score\nACD,0.9\n')
        for route in ['/api/custom-model/artifacts/download','/api/test-generic-result']:
            assert alice.get(route,params={'path':str(artifact)}).status_code==200
            assert bob.get(route,params={'path':str(artifact)}).status_code==404
        replay=alice.post(f'/api/custom-model/jobs/{number}/stream')
        assert replay.status_code==200 and 'private training log' in replay.text
        assert 'no-store' in replay.headers['cache-control']


def test_custom_model_lists_do_not_reveal_other_job_outputs(tmp_path,monkeypatch):
    from backend.web_v2 import custom_model_api as api
    monkeypatch.setenv('WEBUI_V2_MODE','local')
    monkeypatch.setenv('VENUS_JOB_QUEUE_ROOT',str(tmp_path/'queue'))
    monkeypatch.setattr(api,'_ALLOWED_FILE_ROOTS',[tmp_path])
    folder=tmp_path/'models';folder.mkdir()
    (folder/'public.pt').write_text('public')
    private=folder/'private-run';private.mkdir();(private/'private.pt').write_text('private')
    JobRecords().grant(private,'custom','someone-else')
    app=FastAPI();app.include_router(api.router)
    with TestClient(app) as client:
        rows=client.get('/api/custom-model/models',params={'folder':str(folder)}).json()['models']
        assert len(rows)==1 and rows[0]['label']=='public.pt'


def test_real_download_inline_and_structure_handlers_enforce_owner(tmp_path,monkeypatch):
    """Load actual route functions without starting the unrelated model engines."""
    import ast
    from backend.web_v2.job_access import guard_artifact
    monkeypatch.setenv('VENUS_JOB_QUEUE_ROOT',str(tmp_path/'queue'))
    token='c'*64;owner=hashlib.sha256(token.encode()).hexdigest()
    private=tmp_path/'results'/'custom_model'/'run_demo';private.mkdir(parents=True)
    target=private/'protein.pdb';target.write_text('ATOM      1  N   ALA A   1       0.0     0.0     0.0\n')
    JobRecords().grant(private,'custom',owner)
    app=FastAPI()
    source=Path(__file__).resolve().parents[1]/'backend/api_server.py'
    names={'get_inline_file','download_file','get_structure_content'}
    tree=ast.parse(source.read_text())
    functions=ast.Module(body=[node for node in tree.body if isinstance(node,ast.AsyncFunctionDef) and node.name in names],type_ignores=[])
    scope={'app':app,'__file__':str(source),'Request':Request,'Path':Path,'FileResponse':FileResponse,
           'HTTPException':__import__('fastapi').HTTPException,'guard_artifact':guard_artifact,
           'WEB_V2_RESULTS_ROOT':tmp_path/'results','WEB_V2_MANIFESTS_ROOT':tmp_path/'manifests','WEB_V2_UPLOAD_ROOT':tmp_path/'uploads',
           'ensure_within_roots':lambda p,roots:any(p.resolve().is_relative_to(r.resolve()) for r in roots),
           '_INLINE_ALLOWED_EXTS':{'.pdb'},'_INLINE_MEDIA_TYPES':{'.pdb':'text/plain'},'_STRUCTURE_EXTENSIONS':{'.pdb'}}
    exec(compile(functions,str(source),'exec'),scope)
    with TestClient(app) as alice,TestClient(app) as bob:
        alice.cookies.set('vf_custom_jobs',token,domain='testserver.local',path='/api')
        routes=[('/api/download/custom_model/run_demo/protein.pdb',{}),('/api/files/inline',{'path':str(target)}),('/api/structure/content',{'path':str(target)}),('/api/structure/content',{'path':str(target),'download':1})]
        for route,params in routes:
            assert alice.get(route,params=params).status_code==200,route
            assert bob.get(route,params=params).status_code==404,route


def test_custom_meta_does_not_replace_identity_during_first_page_load(tmp_path,monkeypatch):
    from backend.web_v2 import custom_model_api as api
    monkeypatch.setenv('VENUS_JOB_QUEUE_ROOT',str(tmp_path/'queue'))
    app=FastAPI();app.include_router(api.router)
    with TestClient(app) as client:
        response=client.get('/api/custom-model/meta')
        assert 'set-cookie' not in response.headers
        assert client.get('/api/custom-model/session').status_code==200
        cookie=client.cookies.get(cookie_name('custom'))
        assert cookie
        client.get('/api/custom-model/meta')
        assert client.cookies.get(cookie_name('custom'))==cookie

def test_same_host_local_and_online_keep_their_workspaces(tmp_path,monkeypatch):
    from fastapi import Response
    from backend.web_v2.model_platform import Platform
    from backend.web_v2.job_access import browser_owner,request_owners
    local=Platform(tmp_path/"local")
    online=Platform(tmp_path/"online")
    app=FastAPI()
    @app.get("/api/identity")
    def identity(request:Request,response:Response):
        service=local if os.environ["WEBUI_V2_MODE"]=="local" else online
        owner=browser_owner(request,response,"platform",service.session)
        return {"owner":owner,"file_owner":request_owners(request)["platform"]}
    with TestClient(app) as client:
        monkeypatch.setenv("WEBUI_V2_MODE","local")
        local_owner=client.get("/api/identity").json()["owner"]
        monkeypatch.setenv("WEBUI_V2_MODE","online")
        online_owner=client.get("/api/identity").json()["owner"]
        assert local_owner!=online_owner
        monkeypatch.setenv("WEBUI_V2_MODE","local")
        assert client.get("/api/identity").json()=={"owner":local_owner,"file_owner":local_owner}
        monkeypatch.setenv("WEBUI_V2_MODE","online")
        assert client.get("/api/identity").json()=={"owner":online_owner,"file_owner":online_owner}
        assert client.cookies.get("vf_model_workspace_local")!=client.cookies.get("vf_model_workspace")


def test_cookie_namespace_uses_config_when_mode_env_is_absent(monkeypatch):
    from types import SimpleNamespace
    import config
    monkeypatch.delenv("WEBUI_V2_MODE", raising=False)
    monkeypatch.setattr(config, "get_config", lambda: SimpleNamespace(
        server=SimpleNamespace(is_online=True)))
    assert cookie_name("platform") == "vf_model_workspace"
