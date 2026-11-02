#!/usr/bin/env python3
"""Check source-control candidates without importing application dependencies.

Use --staged before a commit. Findings show paths and rules, never secret values.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path, PurePosixPath
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MAX_FILE = 10 * 1024**2
PRIVATE_ROOTS = {'archive', 'third_party', 'epitope_miner', 'insulin_variants', 'runtime', 'research_runs', 'temp_outputs', 'temp_downloads', '.venvs', '.venv', '.kimi-code', '.claude'}
PRIVATE_PARTS = {'node_modules', '__pycache__', '.pytest_cache', '.ruff_cache'}
PRIVATE_FILES = {'model-config.yaml', 'configs/model_runtimes.json', 'configs/rsi/paper_steps.json'}
SECRET_PATTERNS = tuple(re.compile(pattern) for pattern in (
    rb'sk-or-v1-[A-Za-z0-9]{40,}',
    rb'sk-ant-[A-Za-z0-9_-]{40,}',
    rb'olp_[A-Za-z0-9]{25,}',
    rb'\bsk-[A-Za-z0-9]{32,}\b',
    rb'\bsk-proj-[A-Za-z0-9_-]{40,}',
    rb'\bgh[pousr]_[A-Za-z0-9]{30,}',
    rb'\bgithub_pat_[A-Za-z0-9_]{40,}',
    rb'\bAKIA[0-9A-Z]{16}\b',
    rb'\bAIza[0-9A-Za-z_-]{30,}',
    rb'\btvly-[A-Za-z0-9_-]{25,}',
    rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
))


def private_path(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(
        path.parts and path.parts[0] in PRIVATE_ROOTS
        or set(path.parts) & PRIVATE_PARTS
        or name in PRIVATE_FILES
        or name.startswith(('archive/research-notes/', 'configs/rsi/paper_step_definitions/', 'frontend/dist/'))
        or name.startswith('ckpt/') and name not in {'ckpt/README.md', 'ckpt/manifest.json'}
        or path.name.startswith('.env') and path.name != '.env.example'
        or path.suffix in {'.key', '.pem', '.sqlite3', '.db', '.pyc'}
    )


def inspect_bytes(name: str, data: bytes) -> list[str]:
    issues = []
    if private_path(name):
        issues.append(f'{name}: deployment state or credentials do not belong in Git')
    if len(data) > MAX_FILE:
        issues.append(f'{name}: exceeds 10 MiB; use the runtime installer or research data store')
    if any(pattern.search(data) for pattern in SECRET_PATTERNS):
        issues.append(f'{name}: possible credential or private key')
    if name.endswith('.zip'):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as bundle:
                entries = bundle.infolist()
                if len(entries) > 1000 or sum(x.file_size for x in entries) > 100 * 1024**2:
                    return issues + [f'{name}: example archive exceeds inspection budget']
                for entry in entries:
                    member = PurePosixPath(entry.filename)
                    if member.is_absolute() or '..' in member.parts or private_path(entry.filename):
                        issues.append(f'{name}: unsafe or private archive member {entry.filename}')
                    elif not entry.is_dir():
                        if entry.file_size > 20 * 1024**2:
                            issues.append(f'{name}: oversized archive member {entry.filename}')
                        elif any(p.search(bundle.read(entry)) for p in SECRET_PATTERNS):
                            issues.append(f'{name}: possible credential in {entry.filename}')
        except (zipfile.BadZipFile, RuntimeError):
            issues.append(f'{name}: cannot inspect example archive')
    return issues


def check(root: Path = ROOT, *, staged: bool = False) -> list[str]:
    args = ['diff', '--cached', '--name-only', '--diff-filter=ACMR', '-z'] if staged else ['ls-files', '--cached', '--others', '--exclude-standard', '-z']
    names = subprocess.check_output(['git', *args], cwd=root).decode().split('\0')
    issues = []
    for name in sorted(set(filter(None, names))):
        path = root / name
        if not staged and not path.is_file():
            continue  # removed paths and optional Git submodules
        if staged:
            mode = subprocess.check_output(['git', 'ls-files', '--stage', '--', name], cwd=root).decode().split()[0]
            if mode == '160000':
                continue
            data = subprocess.check_output(['git', 'show', ':' + name], cwd=root)
        else:
            data = path.read_bytes()
        issues.extend(inspect_bytes(name, data))
    return issues


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staged', action='store_true')
    args = parser.parse_args()
    findings = check(staged=args.staged)
    if findings:
        raise SystemExit('\n'.join(findings))
    print('Source candidates: no private runtime paths, oversized files or recognized credential patterns')
