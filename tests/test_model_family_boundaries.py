from pathlib import Path
from types import SimpleNamespace
import importlib

import pytest

from mcp_tools.models.identity import enrich, trained_metadata
from mcp_tools.scheduler.catalog import tool_family

@pytest.mark.parametrize('adapter,family', [
    ('evolution','pretrained'),('structure-evolution','pretrained'),('design','pretrained'),
    ('external:ligandmpnn','pretrained'),('external:vina','pretrained'),
    ('function','factory'),('checkpoint','factory'),('personal','factory'),
    ('scientist','rsi'),('capability','rsi'),('foundation','foundation'),('descriptors','utility'),
])
def test_family_is_orthogonal_to_identity_visibility_and_review(adapter,family):
    c={'id':'original-id','adapter':adapter,'model':'ESM2-650M','dataset':'DeepSol',
       'owner':'alice','visibility':'private','review_status':'candidate'}
    result=enrich(c)
    assert result['model_family']==family
    for field in ('id','owner','visibility','review_status'):assert result[field]==c[field]
    if adapter=='external:vina':assert result['model_type']=='tool'

def test_factory_artifact_metadata_carries_origin_without_guessing_dataset():
    result=trained_metadata({'plm_model':'ESM2-650M'})
    assert result['model_family']=='factory' and result['training_dataset'] is None

def test_tool_family_keeps_resource_class_separate():
    stub=SimpleNamespace(func=None,coroutine=None)
    assert tool_family('predict_s4pred',stub)=='pretrained'
    assert tool_family('predict_protein_function',stub)=='factory'
    assert tool_family('predict_with_evolved_model',stub)=='rsi'
    assert tool_family('query_ncbi_blast',stub)=='utility'

def test_vendored_skills_keep_source_and_collision_identity():
    from agent.skills import get_skills_metadata, resolve_skill_path
    items=get_skills_metadata()
    native=[x for x in items if x['source']=='venusfactory']
    assert native
    scientific=[x for x in items if x['source']=='scientific']
    for x in scientific:
        assert '/skills/vendor/scientific-agent-skills/skills/' in x['path'].replace('\\','/')
        assert resolve_skill_path(x['skill_id']).is_file()
        assert resolve_skill_path(x['skill_id'],'../../.env') is None
    ids={x['skill_id'] for x in items}
    if scientific and 'biopython' in ids:assert 'sas_biopython' in ids

def test_prosst_assets_are_runtime_data_not_source():
    # AST inspection avoids loading Torch/PyG in this layout regression.
    root=Path(__file__).resolve().parents[1]
    source=root/'mcp_tools/models/pretrained/prosst/structure/get_sst_seq.py'
    assert 'structure_weight_root()' in source.read_text()
    assert not (source.parent/'static').exists()


def test_registered_s4pred_card_keeps_pretrained_origin():
    c=enrich({'id':'tool:predict_s4pred','adapter':'registered-tool'})
    assert (c['model_family'],c['model_type'],c['runtime_id'])==('pretrained','pretrained','s4pred')
