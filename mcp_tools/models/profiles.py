"""Portable administrator-owned runtime profiles; paths are repository-relative.

Profiles declare installations, never executable code supplied by a chat user.
An absolute path remains supported for environments installed outside the checkout.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
PATH_FIELDS = frozenset({
    'python', 'executable', 'root', 'weights', 'packing_weights', 'installation',
    'filters', 'advanced', 'model_dir', 'confidence_model_dir', 'config', 'cache', 'structure_weights', 'ccd', 'data_dir', 'bio_bin',
})


def repository_path(value, *, root=None):
    """Resolve independently of the worker's working directory."""
    base = PROJECT if root is None else Path(root)
    path = Path(value).expanduser()
    # Preserve executable symlinks: resolving venv/bin/python to the base Python
    # silently drops the environment and its model dependencies.
    return Path(os.path.abspath(path if path.is_absolute() else base / path))


def load_runtime_settings(*, root=None, env=None):
    base = PROJECT if root is None else Path(root)
    environment = os.environ if env is None else env
    path = repository_path(environment.get('VENUS_MODEL_RUNTIMES_CONFIG') or
                           'configs/model_runtimes.json', root=base)
    from copy import deepcopy
    from mcp_tools.models.catalog_snapshot import cached
    return deepcopy(cached(('runtime_settings', str(base), str(path)), lambda: _read_runtime_settings(path, base)))


def _read_runtime_settings(path, base):
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('Model runtimes must be a JSON object')
    return {name: normalize_runtime_entry(raw, name=name, root=base) for name, raw in data.items()}


def normalize_runtime_entry(raw, *, name, root):
    """Resolve declared installation paths, including explicitly named variants."""
    from copy import deepcopy

    if not isinstance(raw, dict):
        raise ValueError('Each model runtime must be an object: ' + name)
    entry = deepcopy(raw)
    if 'enabled' in entry and not isinstance(entry['enabled'], bool):
        raise ValueError('Runtime enabled must be a boolean: ' + name)
    for key in PATH_FIELDS & entry.keys():
        if entry[key]:
            if not isinstance(entry[key], str):
                raise ValueError('Runtime path must be text: ' + name + '.' + key)
            entry[key] = str(repository_path(entry[key], root=root))
    if 'variants' in entry:
        if not isinstance(entry['variants'], dict):
            raise ValueError('Runtime variants must be an object: ' + name)
        entry['variants'] = {
            variant: normalize_runtime_entry(config, name=name + '.' + variant, root=root)
            for variant, config in entry['variants'].items()
        }
    return entry
