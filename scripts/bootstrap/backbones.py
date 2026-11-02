"""Download/check complete PyTorch backbone snapshots, never partial cache hits."""
from __future__ import annotations

import json
import sys
from fnmatch import fnmatch
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.bootstrap.common import atomic_json, digest, inside, read_json

PATTERNS = ('*.json', '*.txt', '*.model', '*.bin', '*.safetensors', 'tokenizer*', 'merges*', 'README*', 'LICENSE*')


def snapshot_ready(record, hashes=False):
    folder = Path(record.get('snapshot', '__missing__'))
    entries = record.get('files', {})
    if not entries or not any(name.endswith(('.bin','.safetensors')) for name in entries):
        return False
    try:
        for name, info in entries.items():
            file = folder / name
            # HF uses symlinks into its sibling blob store; path traversal in a
            # recorded filename is still invalid.
            if Path(name).is_absolute() or '..' in Path(name).parts:
                return False
            if not file.is_file() or file.stat().st_size != info['size']:
                return False
            if hashes and info.get('sha256') and digest(file) != info['sha256']:
                return False
    except (OSError, KeyError, TypeError):
        return False
    return True


def provision(repo, cache, *, metadata=None, download=None):
    cache = Path(cache)
    manifest = cache.parent / 'venus-backbones.json'
    records = read_json(manifest, {})
    old = records.get(repo, {})
    if snapshot_ready(old):
        return old
    if metadata is None or download is None:
        from huggingface_hub import HfApi, snapshot_download
        metadata = HfApi(token=False).model_info
        download = snapshot_download
    info = metadata(repo, revision=old.get('revision','main'), files_metadata=True)
    entries = {}
    for item in info.siblings:
        if not any(fnmatch(item.rfilename, pattern) for pattern in PATTERNS):
            continue
        if item.size is None:
            raise ValueError('Hub did not provide required file sizes')
        lfs = item.lfs or {}
        sha = lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)
        entries[item.rfilename] = {'size': item.size, **({'sha256':sha} if sha else {})}
    folder = download(repo, revision=info.sha, token=False, cache_dir=cache, allow_patterns=list(PATTERNS))
    record = {'revision':info.sha, 'snapshot':str(folder), 'files':entries}
    if not snapshot_ready(record, hashes=True):
        raise ValueError('Backbone download is incomplete or failed verification; no ready receipt written')
    records[repo] = record
    atomic_json(manifest, records)
    return record


def health(root, repos, hashes=False):
    config = read_json(root / 'configs/model_runtimes.json', {}).get('finetuned', {})
    cache = inside(root, config.get('cache','runtime/local/model-runtimes/factory/cache'))
    records = read_json(cache / 'huggingface/venus-backbones.json', {})
    return {repo:{'status':'ready' if snapshot_ready(records.get(repo,{}),hashes) else 'missing'} for repo in repos}


if __name__ == '__main__':
    record = provision(sys.argv[1], sys.argv[2])
    print(json.dumps({'repository':sys.argv[1], 'revision':record['revision'], 'verified_files':len(record['files'])}))
