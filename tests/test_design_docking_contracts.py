"""Exercise the real Agent and MCP contracts without launching protein runtimes."""
import pytest
from pydantic import ValidationError
from mcp_tools.models.pretrained.design_docking import operations,service
from mcp_tools.models.pretrained.design_docking.tools_agent import EXTERNAL_TOOLS
TOOLS={tool.name:tool for tool in EXTERNAL_TOOLS}
DEFAULTED=['ligandmpnn_design','bindcraft_design','boltzgen_design','diffdock_dock']
STRUCTURE_ONLY=['ligandmpnn_design','antibmpnn_design','bindcraft_design','boltzgen_design']

@pytest.fixture
def execution(monkeypatch):
    calls=[]
    def execute(model,structure,parameters,ligand_file=''):
        values=service.validate(model,{'parameters':parameters})
        result={'model':model,'structure':structure,'parameters':values,'ligand':ligand_file}
        calls.append(result)
        return result
    monkeypatch.setattr(operations,'execute',execute)
    return calls

@pytest.mark.parametrize('name',DEFAULTED)
@pytest.mark.parametrize('settings',['omitted',None,{}])
def test_agent_can_omit_optional_parameter_objects(name,settings,execution):
    tool=TOOLS[name]
    schema=tool.get_input_schema().model_json_schema()
    assert 'parameters' not in schema.get('required',[])
    payload={'structure_file':'input.pdb'}
    if settings != 'omitted':payload['parameters']=settings
    if name=='diffdock_dock':payload['ligand_file']='ligand.sdf'
    result=tool.invoke(payload)
    assert len(execution)==1 and result['parameters']==service.validate(result['model'],{})
    assert result['structure']=='input.pdb'

@pytest.mark.parametrize('name',STRUCTURE_ONLY)
def test_structure_design_never_advertises_an_unused_ligand_file(name):
    assert 'ligand_file' not in TOOLS[name].get_input_schema().model_json_schema()['properties']

@pytest.mark.parametrize('name,payload',[
    ('vina_dock',{'structure_file':'receptor.pdbqt','parameters':{'center':[0,0,0]}}),
    ('vina_dock',{'structure_file':'receptor.pdbqt','ligand_file':'ligand.pdbqt','parameters':{}}),
    ('antibmpnn_design',{'structure_file':'antibody.pdb','parameters':{}}),
])
def test_genuine_required_inputs_fail_before_execution(name,payload,execution):
    with pytest.raises(ValidationError):TOOLS[name].invoke(payload)
    assert execution==[]

def test_agent_keeps_explicit_nested_scientific_parameters(execution):
    result=TOOLS['ligandmpnn_design'].invoke({'structure_file':'input.pdb','parameters':{'chains_to_design':'A','fixed_residues':'A12 A13','temperature':.2,'number_of_batches':4}})
    assert result['parameters']['fixed_residues']=='A12 A13'
    assert result['parameters']['temperature']==.2 and result['parameters']['number_of_batches']==4

@pytest.mark.parametrize('payload',[{'parameters':'settings.json'},{'parameters':{'model_name':'invented'}},{'parameters':{'temperature':float('nan')}}])
def test_parameter_objects_reject_files_unknown_keys_and_nonfinite_values(payload,execution):
    with pytest.raises(ValidationError):TOOLS['ligandmpnn_design'].invoke({'structure_file':'input.pdb',**payload})
    assert execution==[]

@pytest.mark.asyncio
async def test_fastmcp_and_agent_share_required_arguments_and_defaults(execution):
    from fastmcp import Client
    from mcp_tools.models.pretrained.design_docking.tools_mcp import mcp
    async with Client(mcp) as client:
        advertised={tool.name:tool.inputSchema for tool in await client.list_tools()}
        for name,tool in TOOLS.items():
            native=tool.get_input_schema().model_json_schema()
            assert set(advertised[name]['properties'])==set(native['properties'])
            assert set(advertised[name].get('required',[]))==set(native.get('required',[]))
        for name in DEFAULTED:
            assert 'parameters' not in advertised[name].get('required',[])
            payload={'structure_file':'input.pdb'}
            if name=='diffdock_dock':payload['ligand_file']='ligand.sdf'
            result=await client.call_tool(name,payload)
            assert not result.is_error
        assert len(execution)==4

@pytest.mark.parametrize('key',['ligandmpnn','antibmpnn','bindcraft','boltzgen'])
def test_plaza_rejects_unused_secondary_files_before_reading_them(key):
    class Inputs:
        def owned(self,*args):pytest.fail('An irrelevant ligand must never be read')
    parameters={'design_positions':'1 2'} if key=='antibmpnn' else {}
    with pytest.raises(ValueError,match='separate ligand'):
        service.validate_inputs(Inputs(),'owner',key,{'parameters':parameters,'ligand_dataset_id':'unused'})
