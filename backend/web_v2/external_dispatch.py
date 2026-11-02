"""Submit Model Plaza/workflow design and docking to the shared MCP service."""
import asyncio
import os
from pathlib import Path
import uuid

from mcp_tools.models.pretrained.design_docking.service import validate, validate_inputs
from mcp_tools.scheduler.client import invoke

NAMES = {'ligandmpnn':'ligandmpnn_design','antibmpnn':'antibmpnn_design',
         'vina':'vina_dock','bindcraft':'bindcraft_design','boltzgen':'boltzgen_design','diffdock':'diffdock_dock'}


def predict(platform, owner, model, data, config):
    parameters = validate(model, config)
    ligand = validate_inputs(platform, owner, model, config, data)
    work = Path(os.environ['TEMP_OUTPUTS_DIR']) / ('mcp_' + model + '_' + uuid.uuid4().hex)
    work.mkdir(parents=True, mode=0o700)
    source = work / ('input' + Path(data['name']).suffix.lower())
    source.write_text(data['structure'])
    args = {'structure_file':str(source), 'parameters':parameters}
    if ligand:
        path = work / ('ligand' + Path(ligand['name']).suffix.lower())
        path.write_text(ligand.get('content') or ligand['structure'])
        args['ligand_file'] = str(path)
    parent = os.getenv('VENUS_QUEUE_JOB_ID')
    queue_owner = os.getenv('VENUS_QUEUE_OWNER') or owner
    return asyncio.run(invoke(NAMES[model], args, owner=queue_owner, parent_job=parent))
