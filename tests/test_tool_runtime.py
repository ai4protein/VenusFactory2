"""MCP transport, queue ownership, cancellation and browser admission contracts."""
import asyncio
import json
import os
from pathlib import Path
import time

import pytest

from mcp_tools.scheduler.catalog import classify, catalog
from mcp_tools.scheduler.jobs import ToolJobs
from backend.web_v2.job_queue import JobQueue, TERMINAL


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv('VENUS_JOB_QUEUE_ROOT', str(tmp_path / 'queue'))
    monkeypatch.setenv('WEBUI_V2_MODE', 'local')
    return ToolJobs()


@pytest.mark.parametrize('name, category, resource', [
    ('query_tavily', 'web_search', 'api:web-search'),
    ('query_serpapi', 'web_search', 'api:web-search'),
    ('download_pubmed_abstracts', 'web_search', 'api:web-search'),
    ('query_ncbi', 'protein_database', 'api:ncbi'),
    ('query_gget', 'protein_database', 'api:gget'),
    ('query_ncbi_blast', 'remote_analysis', 'api:ncbi'),
    ('query_foldseek_remote', 'remote_analysis', 'api:remote-analysis'),
    ('download_clustalo_msa', 'remote_analysis', 'api:remote-analysis'),
    ('calculate_physchem_from_fasta', 'sequence_analysis', 'cpu'),
    ('calculate_sasa_from_pdb', 'structure_analysis', 'cpu'),
    ('proteinmpnn_design', 'model_prediction', 'gpu'),
    ('train_protein_model', 'model_training', 'gpu'),
])
def test_categories(name, category, resource):
    policy = classify(name)
    assert (policy['category'], policy['resource']) == (category, resource)


def test_all_registered_tools_have_category():
    data = catalog()
    assert data['total'] > 100
    assert sum(g['count'] for g in data['categories']) == data['total']
    assert all(t['input_schema'] and t['category'] for t in data['tools'])
    assert classify('search_research_tools')['execution'] == 'inline'


def test_invalid_submission_does_not_create_job(isolated):
    with pytest.raises(ValueError):
        isolated.submit('not_a_tool', {}, owner='alice')
    with pytest.raises(ValueError):
        isolated.submit('query_ncbi', {'limit': 21}, owner='alice')
    assert isolated.list('alice') == []


def reserve_cpu(jobs):
    identity = jobs.queue.submit('blocker', 'test', 'cpu')
    jobs.queue.bind(identity)
    assert jobs.queue.claim(identity)
    return identity


def test_queue_cancel_ownership_and_stable_number(isolated, monkeypatch, tmp_path):
    monkeypatch.setenv('VENUS_QUEUE_CPU_SLOTS', '1')
    blocker = reserve_cpu(isolated)
    fasta = tmp_path / 'test.fasta'; fasta.write_text('>test\nMKWVTFISLLFLFSSAYS\n')
    job = isolated.submit('read_fasta', {'file_path': str(fasta)}, owner='alice')
    try:
        assert job['status'] == 'queued' and job['position'] >= 1
        assert job['job_number'].startswith('VF-')
        assert isolated.get(job['job_number'], 'alice')['id'] == job['id']
        with pytest.raises(FileNotFoundError): isolated.get(job['id'], 'bob')
        with pytest.raises(FileNotFoundError): isolated.cancel(job['id'], 'bob')
        assert isolated.list('bob') == []
        assert isolated.cancel(job['id'], 'alice')['status'] == 'cancelled'
    finally:
        isolated.queue.finish(blocker, 'completed')


def test_worker_executes_real_registered_tool(isolated, tmp_path):
    fasta = tmp_path / 'test.fasta'; fasta.write_text('>test\nMKWVTFISLLFLFSSAYS\n')
    job = isolated.submit('read_fasta', {'file_path': str(fasta)}, owner='alice')
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        final = isolated.get(job['id'], 'alice')
        if final['status'] in TERMINAL: break
        time.sleep(.2)
    assert final['status'] == 'completed', (final, (isolated.queue.files(job['id'])/'worker.log').read_text()[-3000:])
    assert 'MKWVTFISLLFLFSSAYS' in str(final['result']['value'])
    assert not final['result']['is_error']
    assert not list((isolated.queue.root/'mcp-inputs').glob('*.json'))


def test_file_scope_rejects_escape_and_symlink(tmp_path):
    from fastapi import HTTPException
    from backend.web_v2.tool_jobs_api import check_paths
    workspace=tmp_path/'workspace';workspace.mkdir()
    for args, root in [({'pdb_file':'/etc/passwd'},workspace),({'pdb_file':'x.pdb'},None),({'pdb_file':'../escape'},workspace)]:
        with pytest.raises(HTTPException):check_paths(args,root)
    (workspace/'link').symlink_to(tmp_path)
    with pytest.raises(HTTPException):check_paths({'pdb_file':'link/escape'},workspace)
    args={'pdb_file':'protein.pdb'};check_paths(args,workspace)
    assert args['pdb_file'] == str(workspace/'protein.pdb')


def test_browser_async_jobs_do_not_accept_client_owner(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.web_v2 import tool_jobs_api as api
    from mcp_tools.scheduler.client import ToolJobError
    records={}
    async def mock_rpc(name,args):
        owner=args.get('owner')
        if name=='submit_tool':
            records['id']=owner
            return {'id':'id','job_number':'VF-20261031-000001','status':'queued'}
        if name=='list_tool_jobs':return {'items':['id'] if records.get('id')==owner else []}
        if records.get('id')!=owner:raise ToolJobError(404,'Task not found')
        return {'id':'id','status':'cancelled' if name=='cancel_tool_job' else 'queued'}
    monkeypatch.setattr(api,'rpc',mock_rpc)
    app=FastAPI();app.include_router(api.router)
    alice=TestClient(app);bob=TestClient(app)
    body={'name':'query_ncbi','arguments':{'query':'HBA1'}}
    response=alice.post('/api/tools/jobs',json=body)
    assert response.status_code==202
    assert response.headers['location'].endswith('/id')
    assert alice.get('/api/tools/jobs/id').status_code==200
    assert bob.get('/api/tools/jobs/id').status_code==404
    assert bob.get('/api/tools/jobs').json()['items']==[]
    assert bob.post('/api/tools/jobs/id/cancel').status_code==404
    assert alice.post('/api/tools/jobs/id/cancel').json()['status']=='cancelled'
    assert alice.post('/api/tools/jobs',json={**body,'owner':'bob'}).status_code==422
    assert alice.post('/api/tools/jobs',json=body,headers={'Origin':'https://elsewhere.invalid'}).status_code==403
    assert alice.post('/api/tools/jobs',json={'name':'agent_generated_code','arguments':{}}).status_code==422


def test_internal_control_requires_service_credential(monkeypatch):
    from mcp_tools.scheduler import mcp_service as server
    monkeypatch.setattr(server,'service_key',lambda:'secret')
    monkeypatch.setattr(server,'get_http_headers',lambda **kw:{})
    with pytest.raises(PermissionError):server.authorize()
    monkeypatch.setattr(server,'get_http_headers',lambda **kw:{'authorization':'Bearer secret'})
    server.authorize()


def test_stopping_parent_cancels_waiting_child(isolated, monkeypatch, tmp_path):
    monkeypatch.setenv('VENUS_QUEUE_CPU_SLOTS', '1')
    parent=isolated.queue.submit('alice','report','research')
    isolated.queue.bind(parent);assert isolated.queue.claim(parent)
    blocker=reserve_cpu(isolated)
    fasta=tmp_path/'test.fasta';fasta.write_text('>test\nMKWVTFISLLFLFSSAYS\n')
    child=isolated.submit('read_fasta',{'file_path':str(fasta)},owner='alice',parent_job=parent)
    try:
        isolated.queue.cancel(parent,'alice')
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            row=isolated.get(child['id'],'alice')
            if row['status'] in TERMINAL:break
            time.sleep(.2)
        assert row['status']=='cancelled'
        assert 'result' not in row
    finally:
        isolated.queue.finish(parent,'cancelled')
        isolated.queue.finish(blocker,'completed')


def test_legacy_content_blocks_are_preserved():
    from mcp.types import ContentBlock
    from pydantic import TypeAdapter
    from fastmcp.tools.tool import ToolResult
    data=[{'type':'text','text':'{"status":"success"}'}]
    result=ToolResult(content=[TypeAdapter(ContentBlock).validate_python(x) for x in data])
    assert json.loads(result.content[0].text)['status']=='success'


def test_legacy_semantic_errors_fail_the_job():
    from types import SimpleNamespace
    from mcp.types import TextContent
    from mcp_tools.scheduler.worker import mcp_result
    raw=SimpleNamespace(content=[TextContent(type='text',text='{"success":false,"error":"Missing sequence"}')],
                        structured_content=None,is_error=False)
    result=mcp_result(raw)
    assert result['is_error']
    assert json.loads(result['content'][0]['text'])['error']=='Missing sequence'


def test_native_failure_feedback_reaches_harness():
    from mcp_tools.scheduler.client import wait
    value={'success':False,'error':'Missing sequence'}
    job={'id':'x','status':'failed','position':0,'result':{'is_error':True,'value':value}}
    assert asyncio.run(wait(job,'alice'))==value
