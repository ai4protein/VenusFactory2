import asyncio
from unittest.mock import AsyncMock
import pytest
from backend.web_v2.model_parameters import parameter_schema,validated_parameters,BOUND,DERIVED
from backend.web_v2.model_platform import Platform


def test_design_schema_covers_all_editable_existing_parameters():
    from backend.web_v2.advanced_tools_api import SequenceDesignBody
    assert set(parameter_schema('design')['properties']) == set(SequenceDesignBody.model_fields)-BOUND-DERIVED


def test_design_values_and_family_are_preserved():
    p=validated_parameters('design', {'parameters':{'model_family':'ca','seed':42,'num_sequences':12,'batch_size':3,'designed_chains':['A'],'temperatures':[.1,.2],'aa_bias_text':'A:0.5'}})
    assert p['ca_only'] and not p['use_soluble_model']
    assert p['seed']==42 and p['batch_size']==3 and p['designed_chains']==['A']
    assert p['temperatures']==[.1,.2] and p['aa_bias_text']=='A:0.5'
    with pytest.raises(ValueError):validated_parameters('design',{'parameters':{'temperatures':[0]}})
    with pytest.raises(ValueError):validated_parameters('design',{'parameters':{'batch_size':3}})
    with pytest.raises(ValueError):validated_parameters('design',{'parameters':{'user_api_key':'not-accepted'}})


def test_plaza_and_workflow_forward_design_parameters(tmp_path,monkeypatch):
    from backend.web_v2 import advanced_tools_api as api
    monkeypatch.setenv('TEMP_OUTPUTS_DIR',str(tmp_path/'files'))
    run=AsyncMock(return_value={'table':[{'sequence':'ACDE'}]})
    monkeypatch.setattr(api,'run_sequence_design',run)
    platform=Platform(tmp_path/'models');model=next(m for m in platform.official() if m['adapter']=='design')
    data={'kind':'structure','structure':'ATOM  input','rows':[]}
    platform.run_existing(model,data,{'parameters':{'seed':21,'num_sequences':4,'designed_chains':['B'],'fixed_chains':['A'],'temperatures':[.2]}})
    body=run.call_args.args[0]
    assert body.seed==21 and body.designed_chains==['B'] and body.fixed_chains==['A']
    assert body.temperatures==[.2] and body.num_sequences==4


def test_discovery_options_and_display_names(tmp_path):
    p=validated_parameters('discovery',{'parameters':{'protect_start':8,'mmseqs_iterations':5}})
    assert p['protect_start']==8 and p['mmseqs_iterations']==5 and p['mmseqs_threads']==4
    cards=Platform(tmp_path/'models').official()
    vaccine=next(m for m in cards if m.get('dataset')=='VenusVaccine_VirusBinary' and m.get('model')=='ESM2-650M')
    assert vaccine['name']=='VenusVaccine_VirusBinary · ESM2-650M'
    venusx=next(m for m in cards if m.get('family')=='VenusX' and m.get('model')=='ESM2-650M')
    assert venusx['name'].startswith('VenusX_') and venusx['name'].endswith(' · ESM2-650M')


def test_checkpoints_follow_installed_model_family(monkeypatch):
    from backend.web_v2 import advanced_tools_api as api
    choices={'soluble':['v_48_020'],'vanilla':['v_48_010','v_48_020'],'ca':['v_48_002']}
    monkeypatch.setattr(api,'_proteinmpnn_model_options',lambda:choices)
    assert parameter_schema('design')['properties']['model_name']['choices_by']['options']==choices
    with pytest.raises(ValueError,match='supported ca'):validated_parameters('design',{'parameters':{'model_family':'ca','model_name':'v_48_020'}})
    assert validated_parameters('design',{'parameters':{'model_family':'ca','model_name':'v_48_002'}})['ca_only']
