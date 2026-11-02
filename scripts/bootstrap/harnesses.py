"""Optional, pinned Harness clients. Installation never invokes an LLM."""
from __future__ import annotations

import os
import platform
import re
from pathlib import Path
from urllib.parse import quote

from .assets import download, extract, fetch_json
from .common import atomic_json, capture, digest, inside, read_json
from .system import node_runtime


RECIPES = {
    'deepseek-harness': {'kind': 'sdk', 'package': 'deepseek-harness-sdk', 'version': '0.1.5rc1', 'key': 'dsh_bin',
                         'source': 'https://github.com/deepseek-ai/deepseek-harness'},
    'codex': {'kind': 'npm', 'package': '@openai/codex', 'version': '0.163.0', 'key': 'codex_bin',
              'source': 'https://developers.openai.com/codex/cli/'},
    'claude-code': {'kind': 'npm', 'package': '@anthropic-ai/claude-code', 'version': '2.1.312', 'key': 'claude_bin',
                    'source': 'https://code.claude.com/docs/en/setup'},
    'kimi-code': {'kind': 'release', 'version': '0.19.2', 'key': 'kimi_bin',
                  'source': 'https://github.com/MoonshotAI/kimi-code'},
}


def names(value):
    result = list(RECIPES) if value == 'all' else list(dict.fromkeys(x.strip() for x in (value or '').split(',') if x.strip()))
    unknown = set(result) - set(RECIPES)
    if unknown:
        raise ValueError('Unknown harness: ' + ','.join(sorted(unknown)))
    return result


def version_matches(output, version):
    normalize = lambda value: re.sub(r'[^a-z0-9]', '', value.lower())
    return normalize(version) in [normalize(v) for v in re.findall(r'\d+(?:\.\d+)+(?:-?rc\.?\d+)?', output)]


def health(root, name):
    base = root / 'runtime/local/harnesses' / name
    try:
        record = read_json(base / 'installation.json', {})
        binary = inside(base, root / record.get('paths', {}).get(RECIPES[name]['key'], '__missing__')) if record else base / '__missing__'
        if not record or not binary.is_file() or digest(binary) != record.get('binary_sha256'):
            return {'status': 'missing', 'id': name}
        text = capture([binary, '--version'], timeout=30)
        if not version_matches(text, RECIPES[name]['version']):
            raise ValueError('Version mismatch')
    except Exception:
        return {'status': 'failed', 'id': name, 'message': 'Executable/version check failed'}
    return {'status': 'ready', 'version': record['version'], 'verification': 'cli_only'}


def install(root, env, name):
    recipe = RECIPES[name]
    base = inside(root, 'runtime/local/harnesses/' + name)
    record_path = base / 'installation.json'
    if record_path.exists():
        checked = health(root, name)
        if checked['status'] != 'ready':
            raise ValueError(f'{name}: managed files changed; inspect the installation before replacing it')
        return {**checked, 'reused': True}
    base.mkdir(parents=True, exist_ok=True)
    paths = {}
    if recipe['kind'] == 'sdk':
        python = env.create(base / 'env', '3.12')
        env.pip(python, recipe['package'] + '==' + recipe['version'])
        env.run([python, '-c', 'from deepseek_harness import DeepSeekHarness'])
        sites = list((base / 'env/lib').glob('python*/site-packages'))
        if len(sites) != 1:
            raise ValueError('Unexpected DeepSeek SDK environment layout')
        site = sites[0]
        arch = {'x86_64': 'x64', 'aarch64': 'arm64'}[platform.machine()]
        binary = site / f'deepseek_harness_runtime/runtime/deepseek-harness-sdk-runtime-linux-{arch}'
        paths['dsh_site'] = str(site.relative_to(root))
    elif recipe['kind'] == 'npm':
        node, npm = node_runtime(root)
        env.run([npm, 'install', '--prefix', base, '--no-audit', '--no-fund', '--save-exact',
                 '--registry=https://registry.npmjs.org', recipe['package'] + '@' + recipe['version']],
                extra_env={'PATH': str(node.parent) + os.pathsep + os.environ.get('PATH', '')})
        # The sandbox mounts a native executable, not an npm shim requiring host Node.
        candidates = []
        for p in (base / 'node_modules').rglob('*'):
            if p.is_file() and not p.is_symlink() and p.name in ('codex', 'claude', 'claude.exe'):
                with p.open('rb') as f:
                    if f.read(4) == b'\x7fELF':
                        candidates.append(p)
        canonical = base / 'node_modules/@anthropic-ai/claude-code/bin/claude.exe'
        if name == 'claude-code' and canonical in candidates:
            candidates = [canonical]
        if len(candidates) != 1:
            raise ValueError(f'{name}: expected one native runtime, found {len(candidates)}')
        binary = candidates[0]
    else:
        tag = '@moonshot-ai/kimi-code@' + recipe['version']
        release = fetch_json('https://api.github.com/repos/MoonshotAI/kimi-code/releases/tags/' + quote(tag, safe=''))
        arch = {'x86_64': 'x64', 'aarch64': 'arm64'}[platform.machine()]
        filename = f'kimi-code-linux-{arch}.zip'
        assets = {a['name']: a for a in release['assets']}
        checksum = download(assets[filename + '.sha256']['browser_download_url'], base / (filename + '.sha256'), max_bytes=4096)
        archive = download(assets[filename]['browser_download_url'], base / filename, expected=checksum.read_text().split()[0])
        if not (base / 'package').exists():
            extract(archive, base / 'package')
        matches = [p for p in (base / 'package').rglob('kimi') if p.is_file()]
        if len(matches) != 1:
            raise ValueError('Unexpected Kimi release layout')
        binary = matches[0]
    binary.chmod(binary.stat().st_mode | 0o111)
    version = capture([binary, '--version'], timeout=30)
    if not version_matches(version, recipe['version']):
        raise ValueError(f'{name}: installed CLI version differs from the pinned recipe')
    paths[recipe['key']] = str(binary.relative_to(root))
    atomic_json(record_path, {'id': name, **recipe, 'paths': paths, 'binary_sha256': digest(binary),
                              'verification': 'cli_only', 'llm_requests': 0})
    return {'status': 'ready', 'version': recipe['version'], 'verification': 'cli_only'}
