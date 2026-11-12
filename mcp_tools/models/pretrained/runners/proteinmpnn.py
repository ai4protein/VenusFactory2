"""ProteinMPNN design/scoring share one runtime and variant-aware weight root."""
from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path

from mcp_tools.models.profiles import repository_path
from mcp_tools.models.execution import command, execute, profile

VARIANTS={'vanilla':'vanilla_model_weights','soluble':'soluble_model_weights','ca':'ca_model_weights'}
PATH_ARGUMENTS={'pdb_path','out_folder','path_to_fasta','jsonl_path','chain_id_jsonl','fixed_positions_jsonl',
                'bias_AA_jsonl','bias_by_res_jsonl','omit_AA_jsonl','pssm_jsonl','tied_positions_jsonl'}


def weight_root(settings=None):
    cfg,_=profile('proteinmpnn',settings)
    from src.ckpt_hub import ckpt_root
    root=repository_path(cfg['weights']) if cfg.get('weights') else ckpt_root()/'ProteinMPNN'
    if cfg.get('weights') and not root.is_dir():raise ValueError('ProteinMPNN weights are not installed')
    return root


def prepare(args, settings=None):
    cfg,_=profile('proteinmpnn',settings)
    values=dict(vars(args))
    if values.get('ca_only') and values.get('use_soluble_model'):
        raise ValueError('CA-only and soluble variants cannot be combined')
    name=str(values.get('model_name','v_48_020'))
    if not re.fullmatch(r'v_\d+_\d+',name):raise ValueError('Invalid ProteinMPNN checkpoint name')
    variant='ca' if values.get('ca_only') else 'soluble' if values.get('use_soluble_model') else 'vanilla'
    # A configured runtime weight root takes precedence over legacy callers.
    weights=(weight_root(settings)/VARIANTS[variant] if cfg.get('weights') or not values.get('path_to_model_weights')
             else repository_path(values['path_to_model_weights']))
    if cfg.get('weights') and not (weights/(name+'.pt')).is_file():
        raise ValueError(f'ProteinMPNN {variant} checkpoint is not installed')
    values['path_to_model_weights']=str(weights)
    for key in PATH_ARGUMENTS:
        if values.get(key):values[key]=str(Path(values[key]).absolute())
    return values


def run(args):
    values=prepare(args)
    out=Path(values['out_folder']);out.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.proteinmpnn-',dir=out) as folder:
        request=Path(folder)/'request.json'
        request.write_text(json.dumps(values),encoding='utf-8');request.chmod(0o600)
        return execute(command('proteinmpnn',['-m','mcp_tools.models.pretrained.runners.proteinmpnn',str(request)]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request',type=Path)
    args=parser.parse_args()
    values=json.loads(args.request.read_text())
    from mcp_tools.models.pretrained.design.proteinmpnn.protein_mpnn_run import _proteinmpnn_run_in_process
    _proteinmpnn_run_in_process(argparse.Namespace(**values))


if __name__=='__main__':main()
