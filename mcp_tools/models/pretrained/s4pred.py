"""CPU-only single-sequence secondary structure inference with S4PRED."""
from pathlib import Path
import json
import os
import uuid


def runtime():
    from mcp_tools.models.profiles import load_runtime_settings
    cfg=load_runtime_settings().get('s4pred',{})
    if not cfg.get('enabled'):raise ValueError('S4PRED is not configured')
    for name in ['python','root','installation']:
        if not cfg.get(name) or not Path(cfg[name]).exists():raise ValueError('S4PRED runtime is incomplete')
    root=Path(cfg['root'])
    if not (root/'run_model.py').is_file() or any(not (root/'weights'/f'weights_{i}.pt').is_file() for i in range(1,6)):
        raise ValueError('S4PRED model weights are missing')
    if not json.loads(Path(cfg['installation']).read_text()).get('verified'):
        raise ValueError('S4PRED installation has not been verified')
    return cfg


def predict(sequence):
    from mcp_tools.tools.research.services import sequence as validate
    from mcp_tools.tools.runtime_tool_policy import assert_tool_call_allowed,is_online_mode
    from mcp_tools.models.pretrained.design_docking.service import execute_command
    assert_tool_call_allowed('predict_s4pred')
    seq=validate(sequence)
    if is_online_mode() and len(seq)>2000:raise ValueError('Use at most 2,000 residues online')
    cfg=runtime()
    root=Path(os.environ['TEMP_OUTPUTS_DIR'])/('s4pred_'+uuid.uuid4().hex)
    root.mkdir(parents=True,mode=0o700)
    source=root/'input.fasta';source.write_text('>query\n'+seq+'\n')
    env={'PATH':'/usr/bin:/bin','HOME':str(root),'LANG':'C.UTF-8','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'}
    command=[cfg['python'],str(Path(cfg['root'])/'run_model.py'),str(source),'--device','cpu','--threads','2','--save-files','--save-by-idx','--outdir',str(root)]
    with (root/'runtime.log').open('wb') as log:
        execute_command(command,cwd=cfg['root'],env=env,log=log,timeout=180 if is_online_mode() else 600)
    path=root/'s4_out_0.ss2';rows=[]
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith('#'):continue
        pos,aa,ss,pc,ph,pe=line.split()
        rows.append({'position':int(pos),'residue':aa,'secondary_structure':ss,'coil_probability':float(pc),'helix_probability':float(ph),'strand_probability':float(pe)})
    if len(rows)!=len(seq) or ''.join(x['residue'] for x in rows)!=seq:raise ValueError('S4PRED returned incomplete output')
    result={'status':'completed','model':'S4PRED','sequence':seq,'secondary_structure':''.join(x['secondary_structure'] for x in rows),'count':len(rows),'kind':'table','rows':rows,'artifacts':[str(path)]}
    (root/'prediction.json').write_text(json.dumps(result))
    result['artifacts'].append(str(root/'prediction.json'))
    return result
