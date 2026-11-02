"""Lightweight admission and process boundary; chemistry dependencies stay isolated."""

from mcp_tools.tools.capability_policy import guard_operation
from pathlib import Path
import json
import os
import uuid

from pydantic import BaseModel, ConfigDict, Field


class ADMETParameters(BaseModel):
    model_config = ConfigDict(extra='forbid')
    smiles_column: str = Field('smiles', min_length=1, max_length=200, title='SMILES column')
    include_physchem: bool = Field(True, title='Include physicochemical properties')


def molecule_limit():
    from mcp_tools.tools.runtime_tool_policy import is_online_mode
    return 32 if is_online_mode() else 1000


def validate_input(data, parameters):
    limit = molecule_limit()
    if data['kind'] == 'table':
        rows = data.get('rows', [])
        if not 1 <= len(rows) <= limit:
            raise ValueError(f'Use 1–{limit} molecules per prediction')
        if any(str(key).startswith('admet_') for row in rows for key in row):
            raise ValueError('Input columns beginning with admet_ are reserved for prediction results')
        column = parameters['smiles_column']
        if any(column not in row for row in rows):
            raise ValueError('Choose the SMILES column')
        if any(not isinstance(row[column], str) or len(row[column]) > 4096 for row in rows):
            raise ValueError('SMILES must be text, up to 4096 characters per molecule')
    elif data['kind'] == 'ligand':
        content = data.get('content', '')
        count = content.count('$$$$') or 1
        if count > limit:
            raise ValueError(f'Use at most {limit} molecules per prediction')
        if Path(data.get('name', '')).suffix.lower() not in {'.sdf', '.mol'}:
            raise ValueError('Choose an SDF or MOL file')
    else:
        raise ValueError('Use SMILES, a molecular table, SDF or MOL')


@guard_operation('predict_admet')
def run_prediction(data, parameters):
    from backend.web_v2.external_models import runtime, execute_command
    parameters = ADMETParameters.model_validate(parameters).model_dump()
    validate_input(data, parameters)
    cfg = runtime('admet_ai')
    from mcp_tools.common.common_utils import get_temp_outputs_base_dir
    work = get_temp_outputs_base_dir() / ('admet_' + uuid.uuid4().hex)
    work.mkdir(parents=True, mode=0o700)
    request = work / 'input.json'
    request.write_text(json.dumps({'data': data, 'parameters': parameters, 'limit': molecule_limit()}))
    env = {k:v for k,v in os.environ.items() if not any(s in k.upper() for s in ('API_KEY','TOKEN','SECRET','PASSWORD'))}
    env.pop('PYTHONPATH', None); env.pop('PYTHONHOME', None)
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MPLCONFIGDIR=str(work/'mpl'))
    with (work/'runtime.log').open('ab') as log:
        try:
            execute_command([cfg['python'], '-I', str(Path(__file__).with_name('runner.py')), str(request), str(work)],
                            cwd=work, env=env, log=log, timeout=min(600, int(cfg.get('timeout_seconds', 300))))
        except ValueError:
            if (work/'error.json').exists():raise ValueError(json.loads((work/'error.json').read_text())['message']) from None
            raise
    result = json.loads((work/'result.json').read_text())
    result['artifacts'] = [str(work/'predictions.csv'), str(work/'endpoints.json'), str(work/'result.json')]
    return result


def predict_admet(smiles: list[str], include_physchem: bool = True) -> dict:
    """Entry point shared by Agent and MCP; scheduling happens in the tool gateway."""
    data = {'kind':'table', 'name':'molecules.json', 'rows':[{'id':str(i+1),'smiles':s} for i,s in enumerate(smiles)]}
    result = run_prediction(data, {'smiles_column':'smiles', 'include_physchem':include_physchem})
    # Preserve full output on disk, keep large batches from flooding the LLM context.
    return {'status':'success', 'count':len(result['rows']), 'preview':result['rows'][:10],
            'metadata':result['metadata'], 'file_info':[{'path':p,'name':Path(p).name} for p in result['artifacts']]}
