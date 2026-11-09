"""Discover optional project-local Harness installations without loading SDKs."""
from __future__ import annotations

import json
import os
import shutil
import sys
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


def default_runtimes(root: Path) -> dict[str, str]:
    """Resolve shared Web/CLI paths without importing services or probe scripts.

    Explicit environment settings override managed installation receipts, then
    existing research installations and PATH. No SDK or credential is loaded.
    """
    root = Path(root).resolve()
    python_root = Path(sys.executable).absolute().parent.parent
    dsh_site = root / "research_runs/rsi/envs/dsh-0.1.5rc1/lib/python3.12/site-packages"
    defaults = {
        "python_root": str(python_root), "python_bin": str(python_root / "bin/python"),
        "kimi_bin": str(Path.home() / ".kimi-code/bin/kimi"),
        "claude_bin": str(root / "research_runs/rsi/envs/claude-code-2.1.312/package/claude"),
        "codex_bin": str(root / "research_runs/rsi/envs/codex-0.163.0/vendor/x86_64-unknown-linux-musl/bin/codex"),
        "dsh_site": str(dsh_site),
        "dsh_bin": str(dsh_site / "deepseek_harness_runtime/runtime/deepseek-harness-sdk-runtime-linux-x64"),
    }
    for key, name in (("kimi_bin", "kimi"), ("codex_bin", "codex"), ("claude_bin", "claude")):
        if not Path(defaults[key]).is_file():
            executable = shutil.which(name)
            if executable:
                defaults[key] = executable
    return discover(root, defaults)
