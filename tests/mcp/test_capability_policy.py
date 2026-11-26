"""Admission and replay tests with isolated state; never call model providers."""
import asyncio
import sys
import time
from pathlib import Path
import pytest
from mcp_tools.tools.capability_policy import CapabilityStore, CapabilityDisabled, selector, guard_operation
from mcp_tools.tools.runtime_tool_policy import assert_tool_call_allowed, is_tool_allowed


@pytest.fixture(autouse=True)
def isolated(tmp_path,monkeypatch):
    monkeypatch.setenv('VENUS_RUNTIME_ROOT',str(tmp_path/'runtime'))
    monkeypatch.setenv('WEBUI_V2_MODE','local')
    monkeypatch.setenv('VENUS_ACCESS_DB',str(tmp_path/'access.sqlite3'))
    monkeypatch.setattr('mcp_tools.tools.runtime_tool_policy.is_online_mode',lambda:False)


def close(key,spec=None):
    store=CapabilityStore();revision=store.rows().get(key,{}).get('revision',0)
    store.set(key,False,revision,spec or {},'test@example.org')
    return store


def test_mode_isolation_live_change_and_revision(monkeypatch):
    store=close('tool:predict_admet')
    assert not is_tool_allowed('predict_admet')
    with pytest.raises(CapabilityDisabled):assert_tool_call_allowed('predict_admet',{'smiles':['CCO']})
    monkeypatch.setattr('mcp_tools.tools.runtime_tool_policy.is_online_mode',lambda:True)
    assert is_tool_allowed('predict_admet')
    monkeypatch.setattr('mcp_tools.tools.runtime_tool_policy.is_online_mode',lambda:False)
    with pytest.raises(ValueError,match='changed'):store.set('tool:predict_admet',True,0,{},'test')
    store.set('tool:predict_admet',True,1,{},'test')
    assert is_tool_allowed('predict_admet')


def test_model_variant_and_tool_switch_share_guard():
    card={'id':'builtin:test','adapter':'function','task':'Solubility','dataset':'ProtSolM','model':'ESM2-650M'}
    close('model:'+card['id'],selector(card))
    with pytest.raises(CapabilityDisabled):assert_tool_call_allowed('predict_protein_function',{'task':'Solubility','model_name':'ESM2-650M','adapter_path':'ckpt/ProtSolM/esm2'})
    assert_tool_call_allowed('predict_protein_function',{'task':'Stability','model_name':'ESM2-650M'})
    assert_tool_call_allowed('predict_protein_function',{'task':'Solubility','model_name':'Ankh-large'})
    assert_tool_call_allowed('predict_protein_function',{'task':'Solubility','model_name':'ESM2-650M','adapter_path':'ckpt/Other/esm2'})
    close('tool:predict_protein_function')
    with pytest.raises(CapabilityDisabled):assert_tool_call_allowed('predict_protein_function',{'task':'Stability'})


def test_stale_callable_rechecks_and_reenables():
    calls=[]
    @guard_operation('calculate_physchem_from_fasta')
    def compute(value):calls.append(value);return value
    assert compute(1)==1
    store=close('tool:calculate_physchem_from_fasta')
    with pytest.raises(CapabilityDisabled):compute(2)
    assert calls==[1]
    store.set('tool:calculate_physchem_from_fasta',True,1,{},'test')
    assert compute(3)==3


def test_plaza_visible_and_stale_card_denied(tmp_path):
    from backend.web_v2.model_platform import Platform
    p=Platform(tmp_path/'models')
    card=next(c for c in p.official() if c['adapter']=='design')
    close('model:'+card['id'],selector(card))
    assert any(c['id']==card['id'] and not c['available'] for c in p.official())
    with pytest.raises(CapabilityDisabled):p.require_available(card)
    with pytest.raises(CapabilityDisabled):assert_tool_call_allowed('proteinmpnn_sequence_scoring_from_structure',{})


def test_disabled_admet_refuses_direct_execution_before_weights():
    from mcp_tools.models.pretrained.admet.service import run_prediction
    close('model:external:admet_ai',selector({'id':'external:admet_ai','adapter':'external:admet_ai'}))
    assert not is_tool_allowed('predict_admet')
    with pytest.raises(CapabilityDisabled):run_prediction({'kind':'table','rows':[]},{})


def test_native_cached_tool_and_mcp_alias_are_both_guarded(monkeypatch):
    from mcp_tools.tools.tools_agent_hub import ALL_TOOLS
    tool=next(t for t in ALL_TOOLS if t.name=='proteinmpnn_sequence_design_from_structure')
    close('tool:proteinmpnn_sequence_design_from_structure')
    assert not is_tool_allowed('proteinmpnn_design')
    with pytest.raises(CapabilityDisabled):assert_tool_call_allowed('proteinmpnn_design',{})
    with pytest.raises(CapabilityDisabled):tool.invoke({'pdb_path':'missing.pdb'})


def test_public_mcp_hides_closed_tools_and_rejects_stale_calls(monkeypatch):
    monkeypatch.setenv('VENUS_MCP_QUEUE_WORKER','1')
    from fastmcp import Client
    from mcp_tools.mcp_server import mcp
    async def run():
        async with Client(mcp) as client:
            assert 'predict_admet' in {t.name for t in await client.list_tools()}
            store=close('tool:predict_admet')
            assert 'predict_admet' not in {t.name for t in await client.list_tools()}
            value=await client.call_tool('predict_admet',{'smiles':['CCO']},raise_on_error=False)
            assert value.is_error
            store.set('tool:predict_admet',True,1,{},'test')
            assert 'predict_admet' in {t.name for t in await client.list_tools()}
    asyncio.run(run())


def test_old_http_search_cannot_bypass_switch():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from fastapi.responses import JSONResponse
    from mcp_tools.tools.search.tools_api import router
    app=FastAPI();app.include_router(router)
    @app.exception_handler(CapabilityDisabled)
    async def denied(request,exc):return JSONResponse(status_code=403,content={'detail':str(exc)})
    close('tool:query_pubmed')
    response=TestClient(app).post('/api/v1/search/pubmed',json={'query':'test'})
    assert response.status_code==403


@pytest.mark.parametrize('route',['native','mcp','public_service'])
def test_queue_replay_does_not_execute_disabled_tools(route):
    from mcp_tools.scheduler.worker import execute
    close('tool:query_public_protein_service')
    with pytest.raises(CapabilityDisabled):asyncio.run(execute({'name':'query_public_protein_service','arguments':{'service':'blast'},'route':route}))


def test_direct_report_service_guard():
    from mcp_tools.tools.research.services import run_service
    close('tool:query_ncbi_blast')
    with pytest.raises(CapabilityDisabled):asyncio.run(run_service('blast',{}))


def test_queued_custom_training_rechecks_before_process_launch(tmp_path,monkeypatch):
    from backend.web_v2 import custom_model_api as api
    from backend.web_v2.job_queue import JobQueue
    monkeypatch.setenv('VENUS_JOB_QUEUE_ROOT',str(tmp_path/'queue'))
    monkeypatch.setenv('TEMP_OUTPUTS_DIR',str(tmp_path/'outputs'))
    monkeypatch.setenv('VENUS_QUEUE_GPU_SLOTS','1')
    monkeypatch.setenv('VENUS_WORKER_AUTOSTART','1')
    marker=tmp_path/'should-not-execute.txt'
    monkeypatch.setattr(api,'_build_exec_cmd',lambda *a:[sys.executable,'-c',f'from pathlib import Path;Path({str(marker)!r}).write_text("executed")'])
    queue=JobQueue();holder=queue.submit('holder','test','gpu');queue.bind(holder);assert queue.claim(holder)
    identity=None
    try:
        identity=api._create_task('training','src/train.py',{'plm_model':'facebook/esm2_t6_8M_UR50D','output_dir':str(tmp_path/'weights')},'test-owner')
        assert queue.get(identity)['status']=='queued'
        close('tool:train_protein_model')
        queue.finish(holder,'completed')
        deadline=time.monotonic()+25
        while time.monotonic()<deadline and queue.get(identity)['status'] not in {'failed','completed','cancelled'}:time.sleep(.1)
        assert queue.get(identity)['status']=='failed'
        assert not marker.exists()
    finally:
        queue.finish(holder,'completed')
        if identity and queue.get(identity)['status'] not in {'failed','completed','cancelled'}:queue.cancel(identity,'test-owner')


def test_admin_auth_csrf_mutation_and_masked_env(tmp_path,monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.web_v2.access_api import router
    from backend.web_v2.online_access import AccessStore
    from backend.web_v2 import capability_admin,settings_api
    AccessStore().bootstrap('admin@example.org','test-password-123')
    async def inventory():
        return {'mode':'local','items':[{'key':'tool:predict_admet','selector':{}}],'unavailable_sources':[]}
    monkeypatch.setattr(capability_admin,'inventory',inventory)
    path=tmp_path/'.env';path.write_text('TAVILY_API_KEY=secret-value\nVENUS_SEARCH_PROVIDER=auto\n')
    template=tmp_path/'.env.example';template.write_text(path.read_text())
    monkeypatch.setattr(settings_api,'_ENV_PATH',path);monkeypatch.setattr(settings_api,'_ENV_EXAMPLE_PATH',template)
    app=FastAPI();app.include_router(router);client=TestClient(app)
    assert client.get('/api/admin/capabilities').status_code==401
    assert client.get('/api/admin/env').status_code==401
    headers={'X-Venus-Access':'1'}
    login=client.post('/api/admin/login',headers=headers,json={'email':'admin@example.org','password':'test-password-123'})
    assert login.status_code==200
    assert 'vf_admin_session_local=' in login.headers['set-cookie']
    payload={'key':'tool:predict_admet','enabled':False,'revision':0}
    assert client.put('/api/admin/capabilities',json=payload).status_code==403
    assert client.put('/api/admin/capabilities',headers={**headers,'Origin':'https://other.test'},json=payload).status_code==403
    assert client.put('/api/admin/capabilities',headers=headers,json=payload).status_code==200
    assert client.put('/api/admin/capabilities',headers=headers,json=payload).status_code==409
    env=client.get('/api/admin/env').json()
    assert 'secret-value' not in str(env)
    entries=[{'key':'TAVILY_API_KEY','value':'••••••••'},{'key':'VENUS_SEARCH_PROVIDER','value':'tavily'}]
    assert client.post('/api/admin/env',headers=headers,json={'entries':entries,'revision':env['revision']}).status_code==200
    assert 'secret-value' in path.read_text() and 'VENUS_SEARCH_PROVIDER=tavily' in path.read_text()
    assert 'secret-value' not in client.get('/api/admin/audit').text
