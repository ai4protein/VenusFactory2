import hashlib
import json
import time
from pathlib import Path

import pytest

import mcp_tools.models.identity as identity
from backend.web_v2.model_platform import Platform, write


@pytest.fixture(autouse=True)
def profiles(monkeypatch):
    monkeypatch.setattr(identity, 'load_runtime_settings', lambda: {})


def test_task_artifact_is_distinct_from_backbone_and_pretrained_method(tmp_path):
    p=Platform(tmp_path);cards=p.official()
    task=next(c for c in cards if c.get('dataset')=='DeepSol' and c.get('model')=='ESM2-650M')
    expected='builtin:'+hashlib.sha256(json.dumps(['ESM2-650M · DeepSol','function',
        {'task':task['task'],'model':'ESM2-650M','dataset':'DeepSol'}],sort_keys=True).encode()).hexdigest()[:24]
    assert task['id']==expected
    assert task['name']=='DeepSol · ESM2-650M'
    assert task['model_type']=='finetuned' and task['runtime_id']=='finetuned'
    assert task['backbone']['repository']=='facebook/esm2_t33_650M_UR50D'
    assert task['training_dataset']=='DeepSol'
    other=next(c for c in cards if c.get('adapter')=='function' and c.get('model')=='ESM2-650M' and c.get('dataset')!='DeepSol')
    assert other['id']!=task['id'] and other['runtime_id']==task['runtime_id']
    assert other['training_dataset']!=task['training_dataset']
    mpnn=next(c for c in cards if c['name']=='ProteinMPNN')
    assert mpnn['model_type']=='pretrained' and mpnn['runtime_id']=='proteinmpnn'
    assert all(c['model_type']=='foundation' for c in cards if c['adapter']=='foundation')
    assert not {'python','weights','model_path','config_path','owner'} & task.keys()


def test_runtime_disabling_preserves_identity_and_parameters(tmp_path,monkeypatch):
    p=Platform(tmp_path);before=next(c for c in p.official() if c['name']=='ProteinMPNN')
    monkeypatch.setattr(identity,'load_runtime_settings',lambda:{'proteinmpnn':{'enabled':False}})
    after=next(c for c in p.official() if c['name']=='ProteinMPNN')
    assert after['id']==before['id'] and after['parameter_schema']==before['parameter_schema']
    assert not after['available'] and after['disabled_reason']=='runtime_disabled'
    with pytest.raises(ValueError,match='runtime is disabled'):p.require_available(after)


def test_filters_do_not_publish_private_models(tmp_path):
    p=Platform(tmp_path);p._cache,p._cache_time=(p.official(),[]),time.monotonic()
    write(p.path('models','a'*32),{'id':'a'*32,'owner':'alice','name':'Private model','description':'Private',
        'category':'function','visibility':'private'})
    items=p.plaza_catalog('alice',model_type='finetuned',limit=1000)['items']
    assert items and all(c['model_type']=='finetuned' for c in items)
    assert len(p.plaza_catalog('alice',mine=True,model_type='trained')['items'])==1
    assert not p.plaza_catalog('bob',mine=True,model_type='trained')['items']


def test_registry_artifact_lineage_and_legacy_metadata(tmp_path,monkeypatch):
    from mcp_tools.models.factory.training import model_registry as registry
    monkeypatch.setattr(registry,'get_resources',lambda:None)
    monkeypatch.setattr(registry,'_project_root',lambda:tmp_path)
    monkeypatch.setattr(registry,'_stored_path',lambda p:str(Path(p).relative_to(tmp_path)))
    folder=tmp_path/'experiment';folder.mkdir();weights=folder/'best_model.pt';weights.write_bytes(b'test-checkpoint')
    config=folder/'training.json';config.write_text(json.dumps({'plm_model':'ESM2-650M','dataset_custom':'DeepSol',
        'training_method':'freeze','output_dir':str(folder)}))
    out=registry.register_trained_model(str(config),model_id='deepsol_test')
    assert out['success'] and out['status']=='registered'
    model=registry.list_trained_models()['models'][0]
    assert model['training_dataset']=='DeepSol' and model['model_type']=='finetuned'
    assert model['artifact']['sha256']==hashlib.sha256(weights.read_bytes()).hexdigest()
    assert model['version']==model['artifact']['sha256'][:16]
    # Old records gain only known metadata, never a fabricated dataset or validation.
    monkeypatch.setattr(registry,'_load_index',lambda:{'models':[{'model_id':'old','plm_model':'ESM2-650M','status':'candidate'}]})
    old=registry.list_trained_models()['models'][0]
    assert old['status']=='candidate' and old['training_dataset'] is None
    assert old['backbone']['name']=='ESM2-650M' and 'artifact' not in old


def test_research_candidates_remain_unpublished(monkeypatch):
    from mcp_tools.models.factory.training import model_registry as registry
    monkeypatch.setattr(registry,'get_resources',lambda:object())
    monkeypatch.setattr(registry,'_load_index',lambda:{'models':[{'model_id':'forged','status':'registered'}]})
    monkeypatch.setattr(registry,'registry_root',lambda:Path('/tmp/test'))
    monkeypatch.setattr(registry,'_stored_path',str)
    assert registry.list_trained_models()['models']==[]
