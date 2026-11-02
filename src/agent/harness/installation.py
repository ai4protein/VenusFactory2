"""Discover optional project-local Harness installations without loading SDKs."""
from __future__ import annotations

import json
import os
from pathlib import Path


ENV_KEYS = {'kimi_bin': 'VENUS_KIMI_BIN', 'codex_bin': 'VENUS_CODEX_BIN',
            'claude_bin': 'VENUS_CLAUDE_BIN', 'dsh_bin': 'VENUS_DSH_BIN', 'dsh_site': 'VENUS_DSH_SITE'}


def discover(root, defaults=None):
    root = Path(root).resolve()
    result = dict(defaults or {})
    for name in ('kimi-code', 'deepseek-harness', 'codex', 'claude-code'):
        receipt = root / 'runtime/local/harnesses' / name / 'installation.json'
        if not receipt.is_file():
            continue
        try:
            paths = json.loads(receipt.read_text()).get('paths', {})
            resolved = {}
            for key, value in paths.items():
                if key not in ENV_KEYS or not isinstance(value, str):
                    raise ValueError('Unsupported runtime key')
                path = root / value
                if not path.resolve().is_relative_to(root / 'runtime/local/harnesses') or not path.exists():
                    raise ValueError('Missing or out-of-tree runtime')
                resolved[key] = str(path)
            result.update(resolved)
        except (OSError, ValueError, TypeError):
            continue
    for key, variable in ENV_KEYS.items():
        if os.getenv(variable):
            result[key] = str(Path(os.environ[variable]).expanduser().absolute())
    return result
