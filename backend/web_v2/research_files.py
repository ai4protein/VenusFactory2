"""Session-owned research inputs and workspace files."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import uuid
import zipfile
from utils.scoped_files import open_file, chunks, copy_file, read_bytes

from backend.web_v2.upload_limits import upload_limit, validate_upload_size

# Compatibility constant; runtime enforcement is mode-dependent.
MAX_FILE = 10 * 1024 * 1024
MAX_BATCH = 80 * 1024 * 1024
INPUT_EXTENSIONS = set('csv tsv json jsonl fasta fa faa fna pdb ent cif mmcif sdf mol mol2 pdf txt md yaml yml xls xlsx png jpg jpeg webp py r'.split())


def safe_path(root, relative):
    root = Path(root)
    if not isinstance(relative, str) or '\\' in relative or '\x00' in relative:
        raise ValueError('Invalid file path')
    bits = PurePosixPath(relative).parts
    if not bits or PurePosixPath(relative).is_absolute() or any(p.startswith('.') or ':' in p for p in bits):
        raise ValueError('Invalid file path')
    path = root
    if root.is_symlink():
        raise ValueError('Linked roots are not supported')
    for part in bits:
        path = path / part
        if path.is_symlink():
            raise ValueError('Linked files are not supported')
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise FileNotFoundError('File not found')
    with open_file(root, relative): pass
    return path


def save_inputs(base, files):
    if not 1 <= len(files) <= 32:
        raise ValueError('Upload up to 32 files')
    if upload_limit() is not None and sum(len(data) for _, data in files) > MAX_BATCH:
        raise ValueError('Upload up to 80 MB per batch')
    for name, data in files:
        if not name or name.startswith('.') or '/' in name or '\\' in name or ':' in name or len(name) > 180 or Path(name).suffix.lower().lstrip('.') not in INPUT_EXTENSIONS:
            raise ValueError('Unsupported research file')
        if not data:
            raise ValueError('Choose a nonempty file')
        validate_upload_size(len(data))
    result = []
    base = Path(base) / 'uploads'
    for name, data in files:
        identity = uuid.uuid4().hex
        folder = base / identity
        folder.mkdir(parents=True, mode=0o700)
        (folder / name).write_bytes(data)
        record = {'id': identity, 'name': name, 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        (folder / '.metadata.json').write_text(json.dumps(record))
        result.append(record)
    return result


def resolve_inputs(base, identities):
    if not isinstance(identities, list) or len(identities) > 32 or len(set(identities)) != len(identities):
        raise ValueError('Invalid research attachments')
    result = []
    for identity in identities:
        if not isinstance(identity, str) or not re.fullmatch('[a-f0-9]{32}', identity):
            raise ValueError('Invalid attachment ID')
        folder = Path(base) / 'uploads' / identity
        record = json.loads((folder / '.metadata.json').read_text())
        path = safe_path(folder, record['name'])
        validate_upload_size(path.stat().st_size)
        digest = hashlib.sha256()
        with open_file(folder, record['name']) as stream:
            for block in chunks(stream):
                digest.update(block)
        if digest.hexdigest() != record['sha256']:
            raise ValueError('Attachment changed')
        result.append((path, record))
    return result


def stage_inputs(base, workspace, identities, *, data_store=None):
    result = []
    for source, record in resolve_inputs(base, identities):
        relative = 'inputs/' + record['id'] + '/' + record['name']
        if data_store is not None:
            from agent.harness.research_data_store import TABLE_SUFFIXES
            if Path(relative).suffix.lower() in TABLE_SUFFIXES:
                data_store.ingest_file(relative, source.parent, source.name, record['sha256'])
                result.append({**record,'path':relative})
                continue
        destination = Path(workspace) / relative
        if destination.parent.exists() and destination.parent.is_symlink():
            raise ValueError('Invalid input directory')
        if (Path(workspace) / 'inputs').is_symlink():
            raise ValueError('Invalid input directory')
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            checked = safe_path(workspace, relative)
            digest = hashlib.sha256()
            with open_file(workspace, relative) as stream:
                for block in chunks(stream):
                    digest.update(block)
            if digest.hexdigest() != record['sha256']:
                raise ValueError('Input file changed')
        else:
            copy_file(source.parent, source.name, workspace, relative, max_bytes=source.stat().st_size)
        result.append({**record, 'path': relative})
    return result


def session_roots(state, service, sid):
    roots = {}
    if state.get('agent_session_dir'):
        roots['agent'] = Path(state['agent_session_dir'])
    base = service.session_folder(sid)
    if (base / 'uploads').is_dir():
        roots['uploads'] = base / 'uploads'
    for metadata in base.glob('*.json'):
        if not re.fullmatch('[a-f0-9]{32}', metadata.stem):
            continue
        folder, _, _ = service.metadata(sid, metadata.stem)
        if (folder / 'workspace').is_dir():
            roots['research-' + metadata.stem] = folder / 'workspace'
    return roots


def resolve_file(roots, key):
    scope, slash, relative = key.partition('/')
    if not slash or scope not in roots:
        raise FileNotFoundError('File not found')
    return safe_path(roots[scope], relative)


def open_owned_file(roots, key):
    scope, slash, relative = key.partition('/')
    if not slash or scope not in roots:
        raise FileNotFoundError('File not found')
    return open_file(roots[scope], relative)


def list_files(roots, limit=2000):
    files = []
    for scope, root in roots.items():
        if root.is_symlink():
            continue
        for directory, dirs, names in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if d not in {'__pycache__', 'node_modules'} and not d.startswith('.') and not (Path(directory)/d).is_symlink()]
            for name in names:
                if name.startswith('.') or name.endswith(('.pyc', '.pyo')):
                    continue
                p = Path(directory) / name
                rel = p.relative_to(root).as_posix()
                try:
                    with open_file(root, rel) as stream: stat = os.fstat(stream.fileno())
                except (OSError, ValueError):
                    continue
                files.append({'key': scope+'/'+rel, 'name': name, 'rel': rel, 'scope': scope, 'size': stat.st_size, 'mtime': stat.st_mtime, 'ext': p.suffix.lower().lstrip('.')})
    files.sort(key=lambda f: f['mtime'], reverse=True)
    return {'files': files[:limit], 'total': len(files), 'truncated': len(files) > limit}


def archive_files(roots, keys):
    if not isinstance(keys, list) or not 1 <= len(keys) <= 200 or len(set(keys)) != len(keys):
        raise ValueError('Select 1 to 200 files')
    archive = tempfile.SpooledTemporaryFile(max_size=8*1024*1024)
    try:
        total = 0
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as out:
            for key in keys:
                with open_owned_file(roots, key) as src, out.open(key, 'w') as dst:
                    for data in chunks(src):
                        total += len(data)
                        if total > 200*1024**2: raise ValueError('Select at most 200 MB')
                        dst.write(data)
        archive.seek(0)
        return archive
    except BaseException:
        archive.close()
        raise
