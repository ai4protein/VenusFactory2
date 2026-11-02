"""Paths shared by optional installation and runtime discovery (stdlib only)."""
import json
import os
from pathlib import Path


def gget_python(root):
    root = Path(root)
    configured = os.environ.get('VENUS_GGET_PYTHON')
    if configured:
        return Path(configured).expanduser().absolute()
    managed = root / 'runtime/local/builtin-tools/gget/env/bin/python'
    legacy = root / '.venvs/public-query/bin/python'
    return managed if managed.is_file() or not legacy.is_file() else legacy


def scientist_library_root(root):
    root = Path(root)
    if os.environ.get('VENUS_SCIENTIST_MODEL_LIBRARY'):
        return Path(os.environ['VENUS_SCIENTIST_MODEL_LIBRARY']).expanduser().absolute()
    config = root / 'runtime/local/components/rsi.json'
    if config.is_file():
        value = json.loads(config.read_text())['library_root']
        return (root / value).absolute()
    legacy = root / 'research_runs/rsi/scientist_model_library_v1'
    return legacy if legacy.is_dir() else root / 'runtime/local/models/rsi/library'
