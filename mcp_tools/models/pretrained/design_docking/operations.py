"""One explicit tool per upstream model, all using the shared runtime adapters."""
from pathlib import Path
import uuid
from mcp_tools.tools.runtime_tool_policy import assert_tool_call_allowed
from mcp_tools.models.pretrained.design_docking.service import run, LigandParameters, AntibodyParameters, VinaParameters, BindCraftParameters, BoltzGenParameters, DiffDockParameters
from utils.scoped_files import open_absolute, read_bytes, write_bytes

MODEL_TOOLS = {}

def execute(model, structure_file, parameters, ligand_file=''):
    name = MODEL_TOOLS[model]
    assert_tool_call_allowed(name, {'parameters': parameters})
    source = Path(structure_file)
    with open_absolute(source, max_bytes=20*1024**2) as stream:
        data = {'name': source.name, 'structure': stream.read().decode('utf-8')}
    ligand = None
    if ligand_file:
        path = Path(ligand_file)
        with open_absolute(path, max_bytes=20*1024**2) as stream:
            ligand = {'name': path.name, 'structure': stream.read().decode('utf-8')}
    class Inputs:
        def owned(self, kind, identity, owner):
            if kind != 'datasets' or identity != 'ligand' or ligand is None:
                raise ValueError('Ligand input is missing')
            return ligand
    config = {'parameters': parameters}
    if ligand is not None: config['ligand_dataset_id'] = 'ligand'
    result = run(Inputs(), 'session', model, data, config)
    # Place returned files next to the authorized input, preserving nested paths.
    prefix = 'results/' + model + '_' + uuid.uuid4().hex
    copied = []
    for index, filename in enumerate(result.get('artifacts', [])):
        artifact = Path(filename)
        with open_absolute(artifact, max_bytes=100*1024**2) as stream:
            content = stream.read()
        relative = prefix + '/' + str(index) + '_' + artifact.name
        write_bytes(source.parent, relative, content)
        copied.append(str(source.parent / relative))
    return {**result, 'artifacts': copied}

def ligandmpnn_design(structure_file: str, parameters: LigandParameters, ligand_file: str = '') -> dict:
    """Design protein amino-acid sequences conditioned on ligand atoms with LigandMPNN, using a protein-ligand complex PDB. Keep ligand atoms in that PDB; this adapter does not use a separate ligand_file. Parameters can be {} for defaults, or specify designed chains, fixed residues and sampling settings. Returns designed sequences and files."""
    return execute('ligandmpnn', structure_file, parameters.model_dump() if hasattr(parameters, 'model_dump') else parameters, ligand_file)

MODEL_TOOLS['ligandmpnn'] = 'ligandmpnn_design'
def antibmpnn_design(structure_file: str, parameters: AntibodyParameters, ligand_file: str = '') -> dict:
    """Design antibody amino-acid sequences with AntiBMPNN using an antibody structure PDB and design_positions in parameters. Specify chains_to_design when needed; returns designed antibody sequences and files. A separate ligand_file is not used."""
    return execute('antibmpnn', structure_file, parameters.model_dump() if hasattr(parameters, 'model_dump') else parameters, ligand_file)

MODEL_TOOLS['antibmpnn'] = 'antibmpnn_design'
def vina_dock(structure_file: str, parameters: VinaParameters, ligand_file: str = '') -> dict:
    """Dock a small molecule into a protein using AutoDock Vina. Supply the prepared receptor PDBQT as structure_file, the ligand PDBQT as ligand_file, and the docking-box center in parameters; box size and search settings are adjustable. Returns docking poses and scoring-function estimates."""
    return execute('vina', structure_file, parameters.model_dump() if hasattr(parameters, 'model_dump') else parameters, ligand_file)

MODEL_TOOLS['vina'] = 'vina_dock'
def bindcraft_design(structure_file: str, parameters: BindCraftParameters, ligand_file: str = '') -> dict:
    """Design protein binders for a target protein with BindCraft. Supply the target PDB; parameters select target chains, optional hotspot residues, binder lengths and design count. Returns candidate binders and computational evaluation files. A separate ligand_file is not used."""
    return execute('bindcraft', structure_file, parameters.model_dump() if hasattr(parameters, 'model_dump') else parameters, ligand_file)

MODEL_TOOLS['bindcraft'] = 'bindcraft_design'
def boltzgen_design(structure_file: str, parameters: BoltzGenParameters, ligand_file: str = '') -> dict:
    """Generate protein or peptide binders for a target with BoltzGen. Supply a target PDB; parameters select the target chain, binder length, optional binding residues and sampling budget. Returns binder designs and evaluation files. A separate ligand_file is not used."""
    return execute('boltzgen', structure_file, parameters.model_dump() if hasattr(parameters, 'model_dump') else parameters, ligand_file)

MODEL_TOOLS['boltzgen'] = 'boltzgen_design'
def diffdock_dock(structure_file: str, parameters: DiffDockParameters, ligand_file: str = '') -> dict:
    """Predict protein-small-molecule binding poses with DiffDock. Supply a protein PDB and either ligand_file in SDF/MOL format or parameters.ligand_smiles. Parameters also control sample count and diffusion steps; returns poses and confidence scores."""
    return execute('diffdock', structure_file, parameters.model_dump() if hasattr(parameters, 'model_dump') else parameters, ligand_file)

MODEL_TOOLS['diffdock'] = 'diffdock_dock'
