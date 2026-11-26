import importlib.util
import json
from pathlib import Path
import pytest

from mcp_tools.models.pretrained.admet.service import ADMETParameters, validate_input, molecule_limit
from backend.web_v2 import external_models as external


def table(rows):return {'kind':'table','name':'molecules.json','rows':rows}


def test_online_bound_cannot_be_overridden(monkeypatch):
    monkeypatch.setattr('mcp_tools.tools.runtime_tool_policy.is_online_mode',lambda:True)
    assert molecule_limit()==32
    with pytest.raises(ValueError,match='32'):validate_input(table([{'smiles':'CCO'}]*33),ADMETParameters().model_dump())
    with pytest.raises(ValueError):ADMETParameters(cpu=100,limit=10000)


def test_column_and_kind_validation():
    with pytest.raises(ValueError,match='SMILES column'):validate_input(table([{'sequence':'CCO'}]),ADMETParameters().model_dump())
    with pytest.raises(ValueError):validate_input({'kind':'structure'},ADMETParameters().model_dump())
    with pytest.raises(ValueError):validate_input(table([{'smiles':123}]),ADMETParameters().model_dump())
    validate_input(table([{'SMILES':'CCO'}]),{'smiles_column':'SMILES'})


def test_reserved_results_do_not_overwrite_input():
    with pytest.raises(ValueError,match='reserved'):
        validate_input(table([{'smiles':'CCO','admet_status':'a measurement'}]),ADMETParameters().model_dump())


def test_molecular_files_bound_online(monkeypatch):
    monkeypatch.setattr('mcp_tools.tools.runtime_tool_policy.is_online_mode',lambda:True)
    with pytest.raises(ValueError,match='32'):validate_input({'kind':'ligand','name':'molecules.sdf','content':'$$$$'*33},ADMETParameters().model_dump())


def test_card_and_cpu_routing(monkeypatch):
    monkeypatch.setattr(external,'runtime',lambda key:{'python':'test'})
    monkeypatch.setattr('mcp_tools.tools.runtime_tool_policy.is_online_mode',lambda:True)
    card=next(c for c in external.cards() if c['id']=='external:admet_ai')
    assert card['available'] and card['category']=='admet'
    assert card['accepted_input_kinds']==['table','ligand']
    from mcp_tools.scheduler.catalog import classify
    assert classify('predict_admet')['resource']=='cpu'
    assert classify('predict_admet')['execution']=='queued'
    from backend.web_v2.model_platform import Platform
    obj=object.__new__(Platform)
    obj.model=lambda owner,identity:card
    assert obj.resource_for('owner','predict',{'model_id':card['id']})=='cpu'


def test_no_external_inputs_for_admet():
    external.validate_inputs(None,'owner','admet_ai',{},table([{'smiles':'CCO'}]))


def test_runtime_requires_verified_receipt(tmp_path,monkeypatch):
    py=tmp_path/'python';py.touch();receipt=tmp_path/'installation.json';receipt.write_text('{}')
    monkeypatch.setattr(external,'settings',lambda:{'admet_ai':{'enabled':True,'python':str(py),'installation':str(receipt)}})
    with pytest.raises(ValueError,match='verified'):external.runtime('admet_ai')
    receipt.write_text(json.dumps({'verified':True,'version':'2.0.1'}));assert external.runtime('admet_ai')['enabled']


def test_mcp_schema_and_native_tool():
    from mcp_tools.models.pretrained.admet.tools_agent import predict_admet_tool
    assert predict_admet_tool.name=='predict_admet'
    assert 'smiles' in predict_admet_tool.get_input_schema().model_json_schema()['properties']
