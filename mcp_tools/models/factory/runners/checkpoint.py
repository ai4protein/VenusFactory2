"""Execution profile for registered PLM checkpoints; artifacts retain their weights."""
from pathlib import Path
from mcp_tools.models.profiles import PROJECT
from mcp_tools.models.execution import command, profile


def prepare(argv):
    arguments=list(argv[1:])
    if arguments and arguments[0]=='-u':arguments.pop(0)
    if not arguments:raise ValueError('Missing checkpoint loader')
    path=Path(arguments[0])
    path=(path if path.is_absolute() else PROJECT/path).resolve()
    if path not in {PROJECT/'src/predict.py',PROJECT/'src/evaluate.py'}:
        raise ValueError('Unknown checkpoint loader')
    cfg,_=profile('plm_checkpoint')
    if cfg.get('weights') or cfg.get('model_id'):
        raise ValueError('Checkpoint weights belong to the model artifact, not its runtime')
    return command('plm_checkpoint',['-u',str(path),*arguments[1:]])
