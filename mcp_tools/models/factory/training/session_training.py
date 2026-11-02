"""Trusted queue-owned training scope. No workspace is accepted from model arguments."""
import json
import os
from pathlib import Path
import re
import time

ESM8M = 'facebook/esm2_t6_8M_UR50D'


def workspace_root():
    value = os.getenv('VENUS_TOOL_WORKSPACE')
    return Path(value).resolve(strict=True) if value else None


def scoped_path(value):
    from mcp_tools.tools.runtime_tool_policy import is_online_mode
    root = workspace_root()
    if not root:
        if is_online_mode(): raise ValueError('Training requires an authenticated session workspace')
        return value
    from agent.harness.session_boundary import session_path
    return session_path(root, str(value))


def validate_config(config, prediction=False):
    from mcp_tools.tools.runtime_tool_policy import is_online_mode
    config = dict(config)
    root = workspace_root()
    online = is_online_mode()
    if online and not root: raise ValueError('Training requires an authenticated session workspace')
    if not root: return config
    from mcp_tools.models.factory.training.train_operations import PLM_MODELS
    model = PLM_MODELS.get(config.get('plm_model'), config.get('plm_model'))
    if model not in set(PLM_MODELS.values()) | {ESM8M}:
        raise ValueError('Select a configured protein language model')
    if online and model != ESM8M: raise ValueError('Online training supports ESM2-8M only')
    config['plm_model'] = model
    if config.get('dataset_selection', 'Custom') != 'Custom':
        raise ValueError('Use training data from this research workspace')
    for key in ('dataset_custom','train_file','valid_file','test_file','model_path','output_dir'):
        if config.get(key): config[key] = scoped_path(config[key])
    if not config.get('output_dir'): raise ValueError('Specify a workspace output directory')
    name = config.get('output_model_name','model.pt')
    if not re.fullmatch(r'[\w.-]{1,120}', name) or name.startswith('.'):
        raise ValueError('Use a checkpoint filename without directories')
    config['output_model_name'] = name
    config['wandb_enabled'] = False
    if online:
        for key, default, maximum in [('num_epochs',20,20),('batch_size',16,16),('max_seq_len',512,1024),('num_workers',1,2),('gradient_accumulation_steps',1,16),('num_labels',2,1000)]:
            n = config.get(key,default)
            if type(n) is not int or n < (0 if key=='num_workers' else 1) or n > maximum:
                raise ValueError(f'Online {key} must be at most {maximum}')
        if config.get('batch_mode','Batch Size Mode') != 'Batch Size Mode':
            raise ValueError('Online training uses fixed batch size')
    return config


def stage_dataset(train, valid, test, sequence_column, label_column, problem_type):
    import pandas as pd
    from mcp_tools.common.common_utils import get_save_path
    from mcp_tools.models.factory.training.data.split_policy import holdout_indices, assert_disjoint_sequences
    from uuid import uuid4
    frames = {'train':pd.read_csv(scoped_path(train))}
    for name, path in [('validation',valid),('test',test)]:
        if path: frames[name] = pd.read_csv(scoped_path(path))
    missing = [name for name in ('validation','test') if name not in frames]
    if missing:
        df=frames['train']
        labels = df[label_column].tolist() if problem_type == 'single_label_classification' else None
        indices=holdout_indices(df[sequence_column].tolist(), missing, labels)
        frames.update({name:df.iloc[rows] for name,rows in indices.items()})
    assert_disjoint_sequences({name:df[sequence_column].tolist() for name,df in frames.items()})
    folder=get_save_path('training_pipeline','data')/uuid4().hex[:12]
    folder.mkdir()
    for name, df in frames.items(): df.to_csv(folder/(name+'.csv'),index=False)
    return str(folder)


def training_progress(logs, status):
    identity = os.getenv('VENUS_QUEUE_JOB_ID')
    if not identity: return
    from backend.web_v2.job_queue import JobQueue
    from mcp_tools.scheduler.jobs import save_json
    from agent.harness.session_boundary import public_value
    tail='\n'.join(logs[-40:])[-12000:]
    row={'kind':'training','status':status,'updated_at':time.time(),'stdout_tail':tail}
    matches=list(re.finditer(r'(?i)epoch\s*[: ]\s*(\d+)\s*/\s*(\d+)',tail))
    if matches:
        step,total=map(int,matches[-1].groups())
        if 0 <= step <= total and total > 0: row['progress']={'step':step,'total':total,'metrics':{}}
    save_json(JobQueue().files(identity)/'mcp-progress.json',public_value(row,workspace_root() or Path.cwd()))
