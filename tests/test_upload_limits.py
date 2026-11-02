import asyncio
import hashlib
import io
from pathlib import Path

import pytest
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile as Uploaded

from backend.web_v2 import upload_limits as limits
from backend.web_v2.research_files import save_inputs, resolve_inputs


def upload(content, *, reported=None):
    return Uploaded(io.BytesIO(content), filename='input.csv', size=reported)


@pytest.mark.parametrize('delta', [-1, 0, 1])
@pytest.mark.parametrize('reported', [None, 1])
def test_online_checks_actual_bytes_and_exact_boundary(monkeypatch, delta, reported):
    monkeypatch.setenv('WEBUI_V2_MODE','online')
    content=b'x'*(limits.ONLINE_FILE_BYTES+delta)
    if delta <= 0:
        assert asyncio.run(limits.read_upload(upload(content,reported=reported))) == content
    else:
        with pytest.raises(HTTPException) as error:
            asyncio.run(limits.read_upload(upload(content,reported=reported)))
        assert error.value.status_code == 413


def test_local_read_does_not_truncate_at_old_20mb_limit(monkeypatch):
    monkeypatch.setenv('WEBUI_V2_MODE','local')
    content=b'x'*(21*1024*1024)+b'END'
    assert asyncio.run(limits.read_upload(upload(content,reported=len(content)))) == content


def test_online_rejects_known_oversize_without_reading(monkeypatch):
    monkeypatch.setenv('WEBUI_V2_MODE','online')
    class File:
        size=limits.ONLINE_FILE_BYTES+1
        async def read(self,count): raise AssertionError('Oversize file should not be read')
    with pytest.raises(HTTPException) as error: asyncio.run(limits.read_upload(File()))
    assert error.value.status_code == 413


def test_streamed_replacement_failure_keeps_existing_file(monkeypatch,tmp_path):
    monkeypatch.setenv('WEBUI_V2_MODE','online')
    destination=tmp_path/'data.csv'; destination.write_bytes(b'original')
    with pytest.raises(HTTPException):
        asyncio.run(limits.save_upload(upload(b'x'*(limits.ONLINE_FILE_BYTES+1)),destination))
    assert destination.read_bytes() == b'original'
    assert list(tmp_path.iterdir()) == [destination]


def test_online_does_not_leave_truncated_new_upload(monkeypatch,tmp_path):
    monkeypatch.setenv('WEBUI_V2_MODE','online')
    with pytest.raises(HTTPException):
        asyncio.run(limits.save_upload(upload(b'x'*(limits.ONLINE_FILE_BYTES+1)),tmp_path/'new.csv'))
    assert list(tmp_path.iterdir()) == []


def test_local_streamed_upload_preserves_entire_file(monkeypatch,tmp_path):
    monkeypatch.setenv('WEBUI_V2_MODE','local')
    data=b'x'*(21*1024*1024)+b'END'; target=tmp_path/'large.csv'
    assert asyncio.run(limits.save_upload(upload(data,reported=len(data)),target)) == len(data)
    assert hashlib.sha256(target.read_bytes()).digest() == hashlib.sha256(data).digest()


def test_scientist_online_limit_is_per_file(monkeypatch,tmp_path):
    monkeypatch.setenv('WEBUI_V2_MODE','online')
    items=save_inputs(tmp_path,[('a.csv',b'a'*(6*1024*1024)),('b.csv',b'b'*(6*1024*1024))])
    assert len(items)==2 and sum(item['size'] for item in items)>limits.ONLINE_FILE_BYTES
    assert len(resolve_inputs(tmp_path,[item['id'] for item in items])) == 2


def test_scientist_online_prevalidates_whole_batch(monkeypatch,tmp_path):
    monkeypatch.setenv('WEBUI_V2_MODE','online')
    with pytest.raises(ValueError): save_inputs(tmp_path,[('small.csv',b'1'),('big.csv',b'x'*(limits.ONLINE_FILE_BYTES+1))])
    assert list(tmp_path.iterdir()) == []


def test_scientist_local_allows_batch_over_old_80mb_limit(monkeypatch,tmp_path):
    monkeypatch.setenv('WEBUI_V2_MODE','local')
    data=b'x'*(27*1024*1024)
    items=save_inputs(tmp_path,[(f'{i}.csv',data) for i in range(3)])
    assert sum(item['size'] for item in items) > 80*1024*1024
    assert len(resolve_inputs(tmp_path,[item['id'] for item in items])) == 3


@pytest.mark.parametrize('mode,length,status', [('online',10*1024*1024,200),('online',10*1024*1024+1,413),('local',21*1024*1024,200)])
def test_http_upload_reports_limit_without_truncation(monkeypatch,tmp_path,mode,length,status):
    monkeypatch.setenv('WEBUI_V2_MODE',mode)
    app=FastAPI();target=tmp_path/'upload.bin'
    @app.post('/upload')
    async def receive(file:UploadFile=File(...)):
        return {'size':await limits.save_upload(file,target)}
    with TestClient(app) as client:
        response=client.post('/upload',files={'file':('input.bin',b'x'*length)})
    assert response.status_code==status
    if status==200:
        assert response.json()['size']==length and target.stat().st_size==length
    else:
        assert '10 MB' in response.json()['detail'] and not target.exists()
