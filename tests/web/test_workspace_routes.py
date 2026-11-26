import io
import zipfile
from pathlib import Path
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend.web_v2 import workspace_api as api

@pytest.fixture
def workspace(tmp_path,monkeypatch):
    root=tmp_path/'uploads';root.mkdir()
    def resolve(path,allowed_areas=()):
        candidate=(tmp_path/path).resolve()
        if not candidate.is_relative_to(root):raise ValueError('outside')
        return candidate
    monkeypatch.setattr(api,'resolve_web_v2_client_path',resolve)
    monkeypatch.setattr(api,'to_web_v2_public_path',lambda p:str(Path(p).relative_to(tmp_path)))
    monkeypatch.setattr(api,'_load_workspace_index',lambda:{'uploads/sample.csv':{'owner_key':'owner'}})
    monkeypatch.setattr(api,'_runtime_mode',lambda:'local')
    monkeypatch.setattr(api,'_owner_key_for_request',lambda request:'owner')
    (root/'sample.csv').write_text('sequence,value\nACDE,1\n')
    (root/'private.csv').write_text('private\n')
    (root/'.env').write_text('test only')
    app=FastAPI();app.include_router(api.router)
    return TestClient(app)

def test_download_authorized_file(workspace):
    r=workspace.get('/api/workspace/file/content',params={'storage_path':'uploads/sample.csv'})
    assert r.status_code==200 and 'ACDE,1' in r.text
    assert r.headers['content-disposition'].startswith('attachment;')

@pytest.mark.parametrize('path',['../outside.csv','uploads/.env'])
def test_denied_files(workspace,path):
    assert workspace.get('/api/workspace/file/content',params={'storage_path':path}).status_code==403
    assert workspace.post('/api/workspace/archive',json={'paths':[path]}).status_code==403

def test_zip_preserves_contents_and_deduplicates(workspace):
    r=workspace.post('/api/workspace/archive',json={'paths':['uploads/sample.csv']*2})
    assert r.status_code==200
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        assert z.namelist()==['uploads/sample.csv']
        assert z.read(z.namelist()[0])==b'sequence,value\nACDE,1\n'

def test_archive_missing_file_and_limits(workspace,monkeypatch):
    assert workspace.post('/api/workspace/archive',json={'paths':[]}).status_code==422
    assert workspace.post('/api/workspace/archive',json={'paths':['uploads/sample.csv']*201}).status_code==422
    assert workspace.post('/api/workspace/archive',json={'paths':['uploads/missing.csv']}).status_code==404

def test_archive_checks_every_file_before_output(workspace):
    r=workspace.post('/api/workspace/archive',json={'paths':['uploads/sample.csv','uploads/.env']})
    assert r.status_code==403


@pytest.mark.parametrize('method,path,kwargs', [
    ('get','/api/workspace/files',{}),
    ('get','/api/workspace/file/content',{'params':{'storage_path':'uploads/sample.csv'}}),
    ('post','/api/workspace/archive',{'json':{'paths':['uploads/sample.csv']}}),
    ('post','/api/workspace/read-text',{'params':{'storage_path':'uploads/sample.csv'}}),
    ('post','/api/workspace/upload',{'files':{'file':('test.txt',b'test','text/plain')}}),
    ('post','/api/workspace/register',{'params':{'storage_path':'uploads/sample.csv'}}),
    ('patch','/api/workspace/file',{'params':{'storage_path':'uploads/sample.csv'},'files':{'file':('test.csv',b'new','text/csv')}}),
    ('delete','/api/workspace/file',{'params':{'storage_path':'uploads/sample.csv'}}),
])
def test_entire_library_is_local_only(workspace,monkeypatch,method,path,kwargs):
    monkeypatch.setattr(api,'_runtime_mode',lambda:'online')
    response=getattr(workspace,method)(path,**kwargs)
    assert response.status_code==403
    assert 'local mode' in response.json()['detail']


def test_local_library_includes_files_from_other_local_conversations(workspace):
    response=workspace.get('/api/workspace/file/content',params={'storage_path':'uploads/private.csv'})
    assert response.status_code==200
    assert response.text=='private\n'


def test_scientist_artifacts_download_and_archive(workspace, tmp_path, monkeypatch):
    from backend.web_v2 import library_files
    root = tmp_path / 'science'
    monkeypatch.setattr(library_files, 'science_root', lambda: root)
    rel = 'a' * 64 + '/uploads/' + 'b' * 32 + '/observations.csv'
    path = root / rel
    path.parent.mkdir(parents=True)
    path.write_text('value\n1\n')
    key = 'science/' + rel
    response = workspace.get('/api/workspace/file/content', params={'storage_path':key})
    assert response.status_code == 200 and response.text == 'value\n1\n'
    response = workspace.post('/api/workspace/archive', json={'paths':[key]})
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert archive.read(key) == b'value\n1\n'
    denied = 'science/' + 'a' * 64 + '/' + 'c' * 32 + '/runtime.json'
    assert workspace.get('/api/workspace/file/content', params={'storage_path':denied}).status_code == 403
