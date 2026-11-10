import json
from pathlib import Path
import pytest


def test_tools_expose_full_validated_scientific_parameters():
    from mcp_tools.models.pretrained.design_docking.tools_mcp import EXTERNAL_TOOLS
    from mcp_tools.models.pretrained.design_docking.service import MODELS
    from backend.web_v2.external_dispatch import NAMES
    tools = {t.name:t for t in EXTERNAL_TOOLS}
    assert set(tools) == set(NAMES.values())
    for model,name in NAMES.items():
        schema = tools[name].get_input_schema().model_json_schema()
        parameter = schema['properties']['parameters']
        reference = parameter.get('$ref') or next(branch['$ref'] for branch in parameter.get('anyOf', []) if '$ref' in branch)
        nested = reference.split('/')[-1]
        assert set(schema['$defs'][nested]['properties']) == set(MODELS[model][-1].model_fields)
        assert tools[name].metadata['queue_resource'] == ('cpu' if model=='vina' else 'gpu')


def test_web_dispatch_preserves_parameters_and_parent_and_owns_ligand(tmp_path, monkeypatch):
    from backend.web_v2 import external_dispatch as dispatch
    monkeypatch.setenv('TEMP_OUTPUTS_DIR', str(tmp_path))
    monkeypatch.setenv('VENUS_QUEUE_JOB_ID', 'parent-job')
    monkeypatch.setenv('VENUS_QUEUE_OWNER', 'owner')
    seen = {}
    class Inputs:
        def owned(self, kind, identity, owner):
            assert (kind,identity,owner)==('datasets','ligand','owner')
            return {'kind':'ligand','name':'ligand.pdbqt','content':'ATOM ligand'}
    async def invoke(name,args,**kwargs):
        seen.update(name=name,args=args,**kwargs)
        assert Path(args['structure_file']).is_relative_to(tmp_path)
        assert Path(args['ligand_file']).read_text() == 'ATOM ligand'
        return {'kind':'table','rows':[{'affinity':-1}]}
    monkeypatch.setattr(dispatch, 'invoke', invoke)
    result=dispatch.predict(Inputs(),'owner','vina',{'kind':'structure','name':'receptor.pdbqt','structure':'ATOM receptor'},
                            {'parameters':{'center':[1,2,3],'exhaustiveness':16},'ligand_dataset_id':'ligand'})
    assert result['rows'] and seen['name']=='vina_dock'
    assert seen['args']['parameters']['exhaustiveness']==16
    assert seen['parent_job']=='parent-job' and seen['owner']=='owner'


def test_new_tools_obey_online_and_admin_policy(tmp_path, monkeypatch):
    from mcp_tools.tools.runtime_tool_policy import assert_tool_call_allowed
    from mcp_tools.tools.capability_policy import CapabilityStore
    from backend.web_v2.external_dispatch import NAMES
    monkeypatch.setenv('WEBUI_V2_MODE','online')
    from mcp_tools.tools import runtime_tool_policy
    monkeypatch.setattr(runtime_tool_policy,'is_online_mode',lambda:True)
    from backend.web_v2 import runtime_paths
    monkeypatch.setattr(runtime_paths, 'runtime_root', lambda *args: tmp_path)
    for model,name in NAMES.items():
        if model in {'bindcraft','boltzgen'}:
            with pytest.raises(Exception): assert_tool_call_allowed(name, {})
        else:
            assert_tool_call_allowed(name, {})
    CapabilityStore().set('tool:'+NAMES['vina'], False, 0, {}, 'test-admin')
    with pytest.raises(Exception): assert_tool_call_allowed(NAMES['vina'], {})
