"""Shared lifecycle for trusted model processes, separate from model artifacts."""
from __future__ import annotations

import os
import json
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from mcp_tools.models.profiles import PROJECT, load_runtime_settings, repository_path


@dataclass(frozen=True)
class RuntimeCommand:
    argv: list[str]
    env: dict[str, str]
    cwd: Path
    timeout: int
    profile: str


def profile(key, settings=None):
    settings = load_runtime_settings() if settings is None else settings
    cfg = dict(settings.get(key, {}))
    if cfg.get('enabled') is False:
        raise ValueError(f'{key} runtime is disabled')
    if key in {'esm3', 'rfd4_proteina', 'rfantibody', 'esmfold2', 'alphafold2', 'protenix', 'mol1'} or (key == 'esmfold' and cfg.get('installation')):
        receipt_path = cfg.get('installation')
        receipt = {}
        if receipt_path:
            path = repository_path(receipt_path)
            if path.is_file():
                receipt = json.loads(path.read_text())
        if receipt.get('runtime') != key or receipt.get('verified') is not True:
            raise ValueError(f'{key} runtime requires verified inference evidence')
    python = repository_path(cfg['python']) if cfg.get('python') else Path(sys.executable)
    if not python.is_file():
        raise ValueError(f'{key} runtime Python is not installed')
    return cfg, str(python)


def environment(cfg):
    allowed = {'PATH','HOME','USER','LANG','LC_ALL','TMPDIR','TEMP','TMP','SYSTEMROOT',
               'LD_LIBRARY_PATH','CUDA_HOME','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS',
               'HF_HOME','HF_HUB_CACHE','TORCH_HOME','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE',
               'TEMP_OUTPUTS_DIR','VENUS_RUNTIME_ROOT','WEBUI_V2_MODE','VENUS_CKPT_AUTO_DOWNLOAD','VENUS_CKPT_DIR',
               'VENUS_JOB_QUEUE_ROOT','VENUS_CAPABILITY_DB','VENUS_MODEL_RUNTIMES_CONFIG'}
    env = {k:v for k,v in os.environ.items() if k in allowed}
    env.update(PYTHONPATH=os.pathsep.join([str(PROJECT),str(PROJECT/'src')]),
               PYTHONNOUSERSITE='1',PYTHON_DOTENV_DISABLED='1')
    if os.getenv('VENUS_TOOL_WORKSPACE'):
        # Session-authored checkpoints are untrusted tensor data, never pickle code.
        env['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '1'
    if cfg.get('cache'):
        cache = repository_path(cfg['cache'])
        env.update(HF_HOME=str(cache/'huggingface'),HF_HUB_CACHE=str(cache/'huggingface/hub'),TORCH_HOME=str(cache/'torch'))
    if cfg.get('ccd'):
        env['ESMCFOLD_CCD_PATH'] = str(repository_path(cfg['ccd']))
    if 'CUDA_VISIBLE_DEVICES' not in env and cfg.get('cuda_visible_devices') is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(cfg['cuda_visible_devices'])
    return env


def command(key, arguments, *, cwd=None, settings=None):
    cfg, python = profile(key, settings)
    timeout = max(1,min(int(cfg.get('timeout_seconds',3600)),86400))
    env = environment(cfg)
    env['PATH'] = str(Path(python).parent) + os.pathsep + env.get('PATH', '')
    return RuntimeCommand([python,*map(str,arguments)],env,Path(cwd or PROJECT),timeout,key)


def execute(spec):
    """Bound the whole process group; return only a bounded tail of its log."""
    spec.cwd.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryFile(mode='w+b') as log:
        proc=subprocess.Popen(spec.argv,cwd=spec.cwd,env=spec.env,stdout=log,
                              stderr=subprocess.STDOUT,start_new_session=os.name=='posix')
        try:
            code=proc.wait(timeout=spec.timeout)
        except BaseException:
            if proc.poll() is None:
                if os.name=='posix':os.killpg(proc.pid,signal.SIGKILL)
                else:proc.kill()
            proc.wait()
            raise
        log.seek(0,os.SEEK_END);log.seek(max(0,log.tell()-8000))
        tail=log.read().decode('utf-8',errors='replace')
        if code:raise RuntimeError('Prediction runtime failed: '+tail)
        return tail
