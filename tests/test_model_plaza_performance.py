"""Catalog browsing must stay light without caching permissions across requests."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

from backend.web_v2.model_platform import Platform
from mcp_tools.models.catalog_snapshot import catalog_snapshot
from mcp_tools.tools.capability_policy import CapabilityStore, CapabilityDisabled, selector
from mcp_tools.tools import runtime_tool_policy as policy


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv('WEBUI_V2_RUNTIME_ROOT', str(tmp_path / 'runtime'))
    monkeypatch.setenv('VENUS_MODEL_RUNTIMES_CONFIG', str(tmp_path / 'runtimes.json'))
    monkeypatch.setenv('WEBUI_V2_MODE', 'online')
    monkeypatch.setattr(policy, 'is_online_mode', lambda: os.getenv('WEBUI_V2_MODE') == 'online')
    store = CapabilityStore(tmp_path / 'switches.sqlite3')
    initialize = CapabilityStore.__init__
    monkeypatch.setattr(CapabilityStore, '__init__', lambda self, path=None: initialize(self, path or store.path))
    return store


@pytest.mark.parametrize("mode", ["online", "local"])
@pytest.mark.parametrize("listing", ["plaza_catalog"])
def test_fresh_catalog_does_not_import_inference_or_registered_tool_implementations(tmp_path, mode, listing):
    code = """
import json,sys
from backend.web_v2.model_platform import Platform
p = Platform(sys.argv[1])
result = getattr(p, sys.argv[2])('owner', limit=10000)
assert result['items'] and not result['unavailable_sources'], result['unavailable_sources']
assert not result['unavailable_tools']
assert not ({'torch','transformers','sklearn','backend.web_v2.advanced_tools_api'} & set(sys.modules))
assert 'mcp_tools.tools.search.tools_agent' not in sys.modules
assert 'agent.rsi.engine.context_feedback_dispatch' not in sys.modules
assert all('contract' not in c and 'parameter_schema' not in c for c in result['items'])
design = next(c for c in result['items'] if c['adapter']=='design')
fields = p.model('owner',design['id'],detail=True)['parameter_schema']['properties']
assert {'fixed_residues_text','pssm_rules_text','model_family','model_name','temperatures'} <= fields.keys()
print(len(result['items']))
"""
    env = dict(os.environ, WEBUI_V2_MODE=mode, VENUS_MODEL_RUNTIMES_CONFIG=str(tmp_path/'absent.json'))
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path/'platform'), listing], env=env, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr


def test_catalog_reuses_reads_but_next_request_observes_admin_change(isolated, tmp_path, monkeypatch):
    from mcp_tools.models.pretrained.design_docking import service
    monkeypatch.setattr(service, 'runtime', lambda key: {})
    calls = []
    rows = CapabilityStore.rows
    monkeypatch.setattr(CapabilityStore, 'rows', lambda self: (calls.append(1), rows(self))[1])
    p = Platform(tmp_path/'platform')
    first = p.plaza_catalog('owner', query='LigandMPNN')['items'][0]
    assert first['available'] and len(calls)==1
    isolated.set('model:'+first['id'], False, 0, selector(first), 'test')
    assert not p.plaza_catalog('owner', query='LigandMPNN')['items'][0]['available']
    with pytest.raises(CapabilityDisabled): policy.assert_tool_call_allowed('ligandmpnn_design', {})
    isolated.set('model:'+first['id'], True, 1, selector(first), 'test')
    assert p.plaza_catalog('owner', query='LigandMPNN')['items'][0]['available']


def test_profile_snapshot_expires_and_restores_after_exception(tmp_path, monkeypatch):
    from mcp_tools.models.profiles import load_runtime_settings
    cfg=tmp_path/'runtime.json';monkeypatch.setenv('VENUS_MODEL_RUNTIMES_CONFIG',str(cfg))
    cfg.write_text('{"vina":{"enabled":true}}')
    @catalog_snapshot
    def listing():
        a=load_runtime_settings();cfg.write_text('{"vina":{"enabled":false}}')
        a['vina']['enabled']=False
        assert load_runtime_settings()['vina']['enabled'] is True
        raise ValueError('fixture')
    with pytest.raises(ValueError):listing()
    assert load_runtime_settings()['vina']['enabled'] is False


@pytest.mark.parametrize('key,name,resource',[
    ('ligandmpnn','ligandmpnn_design','gpu'),('antibmpnn','antibmpnn_design','gpu'),
    ('vina','vina_dock','cpu'),('diffdock','diffdock_dock','gpu')])
def test_online_models_share_admin_and_resource_guards(key,name,resource,isolated,monkeypatch):
    from mcp_tools.models.pretrained.design_docking import service
    from mcp_tools.scheduler.catalog import classify
    monkeypatch.setattr(service,'runtime',lambda key:{})
    assert service.availability(key)==(True,None)
    policy.assert_tool_call_allowed(name,{})
    from mcp_tools.models.pretrained.design_docking.tools_mcp import EXTERNAL_TOOLS
    tool = next(t for t in EXTERNAL_TOOLS if t.name == name)
    assert classify(name,tool.metadata)['resource']==resource
    isolated.set('tool:'+name,False,0,{},'test')
    assert service.availability(key)==(False,'admin_disabled')
    with pytest.raises(CapabilityDisabled):policy.assert_tool_call_allowed(name,{})
    assert not policy.is_tool_allowed(name)
    assert service.availability('bindcraft')==(False,'local_only')
    assert service.availability('boltzgen')==(False,'local_only')


def test_online_vina_still_queues_and_owns_secondary_input(isolated,tmp_path,monkeypatch):
    from mcp_tools.models.pretrained.design_docking import service
    monkeypatch.setattr(service,'runtime',lambda key:{})
    p=Platform(tmp_path/'platform')
    receptor=p.upload('owner','receptor.pdbqt',b'ATOM  fixture')
    ligand=p.upload('owner','ligand.pdbqt',b'ATOM  ligand')
    body={'model_id':'external:vina','dataset_id':receptor['id'],
          'config':{'parameters':{'center':[0,0,0]},'ligand_dataset_id':ligand['id']}}
    assert p.start('owner','predict',body,launch=False)['status']=='queued'
    with pytest.raises(FileNotFoundError):p.start('other','predict',body,launch=False)
