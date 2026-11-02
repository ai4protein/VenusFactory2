"""A shared execution environment for many distinct trained task artifacts."""
from __future__ import annotations

import argparse
import functools
import inspect
import json
import tempfile
from pathlib import Path

from mcp_tools.models.profiles import PROJECT
from mcp_tools.models.execution import command, execute, profile

OPERATIONS={'predict_protein_function','predict_residue_function'}
LOADERS={'esm2','ankh','prott5','protbert'}


def run_cli(argv):
    """Only maintained finetuned model scripts may run in this environment."""
    script=Path(argv[1]).absolute()
    allowed={PROJECT/'mcp_tools/models/factory/prediction'/f'{name}.py' for name in LOADERS}
    if script not in allowed:raise ValueError('Unknown finetuned model loader')
    cfg,_=profile('finetuned')
    if cfg.get('weights') or cfg.get('model_id'):
        raise ValueError('Task weights belong to each model artifact, not the shared finetuned runtime')
    return execute(command('finetuned',[str(script),*argv[2:]]))


def in_runtime(fn):
    if fn.__name__ not in OPERATIONS:raise ValueError('Unknown finetuned operation')
    signature=inspect.signature(fn)
    @functools.wraps(fn)
    def invoke(*args,**kwargs):
        values=signature.bind(*args,**kwargs);values.apply_defaults()
        cfg,_=profile('finetuned')
        if cfg.get('weights') or cfg.get('model_id'):
            raise ValueError('Task weights belong to each model artifact, not the shared finetuned runtime')
        with tempfile.TemporaryDirectory(prefix='venus-finetuned-') as folder:
            request=Path(folder)/'request.json';result=Path(folder)/'result.json'
            request.write_text(json.dumps({'operation':fn.__name__,'arguments':values.arguments}))
            request.chmod(0o600)
            execute(command('finetuned',['-m','mcp_tools.models.factory.runners.finetuned',str(request),str(result)]))
            return json.loads(result.read_text())
    invoke.runtime_implementation=fn
    return invoke


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request',type=Path);parser.add_argument('result',type=Path)
    args=parser.parse_args();request=json.loads(args.request.read_text())
    if request['operation'] not in OPERATIONS:raise ValueError('Unknown finetuned operation')
    from mcp_tools.models.factory.prediction import fintuned_operations
    fn=getattr(fintuned_operations,request['operation']).runtime_implementation
    args.result.write_text(json.dumps(fn(**request['arguments'])))


if __name__=='__main__':main()
