"""Full installation composition and model-library boundaries (offline)."""
import json
from pathlib import Path

import pytest

from install_full import arguments
from scripts.bootstrap import cli, components
from scripts.bootstrap.common import atomic_json
from scripts.bootstrap import backbones
from mcp_tools.common.installation import gget_python, scientist_library_root


def manifest():
    return json.loads((Path(__file__).resolve().parents[1] / 'configs/bootstrap.json').read_text())


def test_full_entrypoint_has_one_command_default_and_readonly_plan():
    assert arguments([]) == ['install', '--profile', 'full']
    assert arguments(['plan', '--json']) == ['plan', '--profile', 'full', '--json']
    with pytest.raises(ValueError):
        arguments(['--profile', 'minimal'])


def test_full_selects_every_family_even_with_a_tool_override():
    args = cli.parser().parse_args(arguments(['--tools', 'proteinmpnn']))
    stages, tools = cli.selection(args, manifest())
    assert stages == list(cli.STAGES)
    assert tools == ['proteinmpnn']


def test_builtin_database_services_are_not_external_plugins():
    data = manifest()
    assert 'interproscan' not in data['tools']
    assert 'InterProScan' in components.BUILTINS['public_protein']['services']
    assert components.BUILTINS['web_search']['services'] == ['Tavily', 'SerpAPI']


@pytest.mark.parametrize('selection,optional', [('none', []), ('external', ['tools']), ('factory,rsi', ['factory','rsi']), ('harnesses', ['harnesses'])])
def test_optional_choices_always_keep_a_complete_core(selection, optional):
    args = cli.parser().parse_args(arguments(['--components', selection]))
    stages, _ = cli.selection(args, manifest())
    assert {'environment','builtins','frontend'} <= set(stages)
    assert set(stages) - {'environment','builtins','frontend','skills'} == set(optional)


def test_old_full_platform_command_includes_builtin_queries():
    args = cli.parser().parse_args(['install','--dependencies','platform','--stages','environment,frontend'])
    assert cli.selection(args, manifest())[0] == ['environment','builtins','frontend']


def test_conflicting_component_override_is_rejected():
    args = cli.parser().parse_args(arguments(['--components','factory','--harnesses','codex']))
    with pytest.raises(ValueError, match='conflict'):
        cli.selection(args, manifest())


def test_components_choose_actual_external_tools_not_empty_minimal_profile():
    args = cli.parser().parse_args(['install','--components','external'])
    assert cli.selection(args, manifest())[1] == manifest()['profiles']['full']


def test_full_plan_has_all_harnesses_and_factory_backbones(tmp_path,monkeypatch,capsys):
    atomic_json(tmp_path/'configs/bootstrap.json', manifest())
    entries = [{'path':f'DeepSol/{name}/model.pt','size':10,'sha256':'a'*64} for name in manifest()['factory_backbones']]
    entries.append({'path':'demo/demo.pt','size':1,'sha256':'b'*64})
    atomic_json(tmp_path/'ckpt/manifest.json',{'presets':{'predict-all':['**']},'files':entries})
    monkeypatch.setattr(cli,'ROOT',tmp_path)
    before = list(tmp_path.rglob('*'))
    assert cli.main(arguments(['plan','--json'])) == 0
    plan=json.loads(capsys.readouterr().out)
    assert len(plan['harnesses']) == 4 and len(plan['backbones']) == 5
    assert plan['dependencies'] == 'platform'
    assert plan['component_families'] == ['external','factory','rsi','harnesses']
    assert plan['builtins'] and list(tmp_path.rglob('*')) == before


def test_unknown_factory_variant_cannot_silently_omit_required_backbone(tmp_path):
    atomic_json(tmp_path/'ckpt/manifest.json',{'presets':{'predict-all':['**']},'files':[{'path':'New/unknown/model.pt'}]})
    with pytest.raises(ValueError,match='backbone mapping'):
        components.factory_backbones(tmp_path,manifest(),'predict-all')


def test_rsi_default_preserves_existing_research_library(tmp_path,monkeypatch):
    monkeypatch.delenv('VENUS_SCIENTIST_MODEL_LIBRARY',raising=False)
    assert scientist_library_root(tmp_path) == tmp_path/'runtime/local/models/rsi/library'
    legacy=tmp_path/'research_runs/rsi/scientist_model_library_v1';legacy.mkdir(parents=True)
    assert scientist_library_root(tmp_path) == legacy
    chosen=tmp_path/'reviewed';chosen.mkdir()
    atomic_json(tmp_path/'runtime/local/components/rsi.json',{'library_root':'reviewed'})
    assert scientist_library_root(tmp_path) == chosen
    monkeypatch.setenv('VENUS_SCIENTIST_MODEL_LIBRARY',str(legacy))
    assert scientist_library_root(tmp_path) == legacy


def test_rsi_setup_does_not_publish_or_copy_models(tmp_path,monkeypatch):
    monkeypatch.delenv('VENUS_SCIENTIST_MODEL_LIBRARY',raising=False)
    library=tmp_path/'reviewed';library.mkdir();(library/'evidence.json').write_text('preserve')
    monkeypatch.setattr(components,'rsi_health',lambda *args:{'status':'ready'})
    components.install_rsi(tmp_path,Path('/python'),'reviewed')
    config=json.loads((tmp_path/'runtime/local/components/rsi.json').read_text())
    assert config == {'library_root':'reviewed','auto_publish':False}
    assert (library/'evidence.json').read_text() == 'preserve'
    assert len(list(library.iterdir())) == 1


def test_rsi_environment_override_cannot_be_silently_ignored(tmp_path,monkeypatch):
    library=tmp_path/'reviewed';library.mkdir()
    monkeypatch.setenv('VENUS_SCIENTIST_MODEL_LIBRARY',str(tmp_path/'other'))
    with pytest.raises(ValueError,match='overrides'):
        components.install_rsi(tmp_path,Path('/python'),library)
    assert not (tmp_path/'runtime/local/components/rsi.json').exists()


def test_doctor_checks_requested_rsi_library_without_changing_selection(tmp_path,monkeypatch):
    monkeypatch.delenv('VENUS_SCIENTIST_MODEL_LIBRARY',raising=False)
    chosen=tmp_path/'chosen';chosen.mkdir()
    monkeypatch.setattr(components,'capture',lambda *a,**kw:'')
    checked=components.rsi_health(tmp_path,Path('/python'),'chosen')
    assert checked['status']=='ready' and checked['library_root']==str(chosen)
    assert not (tmp_path/'runtime/local/components/rsi.json').exists()


def test_gget_discovery_keeps_legacy_and_explicit_configuration(tmp_path,monkeypatch):
    monkeypatch.delenv('VENUS_GGET_PYTHON',raising=False)
    legacy=tmp_path/'.venvs/public-query/bin/python';legacy.parent.mkdir(parents=True);legacy.touch()
    assert gget_python(tmp_path) == legacy
    managed=tmp_path/'runtime/local/builtin-tools/gget/env/bin/python';managed.parent.mkdir(parents=True);managed.touch()
    assert gget_python(tmp_path) == managed
    monkeypatch.setenv('VENUS_GGET_PYTHON',str(legacy))
    assert gget_python(tmp_path) == legacy


def test_backbone_partial_snapshot_is_not_a_complete_install(tmp_path):
    from types import SimpleNamespace
    model = SimpleNamespace(sha='a'*40,siblings=[SimpleNamespace(rfilename='model.safetensors',size=3,lfs={})])
    with pytest.raises(ValueError,match='incomplete'):
        backbones.provision('example/model',tmp_path/'cache/hub',metadata=lambda *a,**k:model,download=lambda *a,**k:tmp_path/'partial')
    assert not (tmp_path/'cache/venus-backbones.json').exists()


def test_complete_backbone_reuses_receipt_without_contacting_hub(tmp_path):
    from types import SimpleNamespace
    snapshot=tmp_path/'snapshot';snapshot.mkdir();(snapshot/'model.safetensors').write_bytes(b'abc')
    model=SimpleNamespace(sha='b'*40,siblings=[SimpleNamespace(rfilename='model.safetensors',size=3,lfs={})])
    args={'metadata':lambda *a,**k:model,'download':lambda *a,**k:snapshot}
    record=backbones.provision('example/model',tmp_path/'cache/hub',**args)
    assert backbones.snapshot_ready(record)
    def no_network(*a,**k):pytest.fail('A complete verified receipt should be reused')
    assert backbones.provision('example/model',tmp_path/'cache/hub',metadata=no_network,download=no_network)==record


def test_backbone_doctor_detects_deleted_weight(tmp_path):
    cache=tmp_path/'runtime/local/model-runtimes/factory/cache'
    snapshot=cache/'huggingface/hub/snapshot';snapshot.mkdir(parents=True)
    record={'snapshot':str(snapshot),'files':{'model.bin':{'size':3}}}
    atomic_json(cache/'huggingface/venus-backbones.json',{'example/model':record})
    assert backbones.health(tmp_path,['example/model'])['example/model']['status']=='missing'
    (snapshot/'model.bin').write_bytes(b'abc')
    assert backbones.health(tmp_path,['example/model'])['example/model']['status']=='ready'
