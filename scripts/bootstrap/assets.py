from __future__ import annotations

import hashlib
import json
import os
import re
import time
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from .common import atomic_json, capture, digest, inside
from . import progress


def fetch_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "VenusFactory2-initialize"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def download(url, path: Path, *, expected=None, algorithm="sha256", max_bytes=16 * 2**30):
    """Atomic bounded download; checksum-backed partial files survive interruptions."""
    if not url.startswith("https://"):
        raise ValueError("Downloads must use HTTPS")
    if path.exists():
        if path.is_file() and path.stat().st_size and (not expected or digest(path, algorithm) == expected):
            progress.ui.line(f'  {path.name}: already downloaded')
            return path
        raise ValueError(f"Existing asset failed verification: {path}; move it aside and retry")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.part')
    receipt = temp.with_name(temp.name + '.json')
    identity = {'url': url, 'expected': expected, 'algorithm': algorithm}
    if temp.is_symlink() or receipt.is_symlink():
        raise ValueError('Download staging must not be a symbolic link')
    if temp.exists():
        previous = json.loads(receipt.read_text()) if receipt.is_file() else None
        if previous != identity:
            raise ValueError(f'Partial download belongs to another source: {temp}')
        if not expected:
            temp.unlink()
    atomic_json(receipt, identity)
    initial = temp.stat().st_size if temp.exists() else 0
    if initial > max_bytes:
        raise ValueError('Partial download exceeds size limit')
    # A process can exit after receiving the last byte but before atomic rename.
    if initial and expected and digest(temp, algorithm) == expected:
        os.replace(temp, path)
        atomic_json(path.with_name(path.name + '.source.json'), {'url': url, 'bytes': initial, algorithm: expected})
        receipt.unlink(missing_ok=True)
        progress.ui.line(f'  {path.name}: recovered completed download')
        return path
    started = time.monotonic()
    try:
        headers = {'User-Agent': 'VenusFactory2-initialize', 'Accept-Encoding': 'identity'}
        if initial:
            headers['Range'] = f'bytes={initial}-'
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=90) as source:
            source_headers = getattr(source, 'headers', {})
            if getattr(source, 'status', 200) == 206:
                match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', source_headers.get('Content-Range', ''))
                if not match or int(match[1]) != initial or int(match[2]) < initial or int(match[3]) <= int(match[2]):
                    raise ValueError('Invalid partial download response')
                total = int(match[3])
            else:
                initial = 0  # Server ignored Range: restart instead of appending.
                total = int(source_headers.get('Content-Length', 0)) or None
            if total and total > max_bytes:
                raise ValueError('Download exceeded the declared size limit')
            count = initial
            progress.ui.update(path.name, 0, count, total, force=True, initial=initial)
            with temp.open('ab' if initial else 'wb') as target:
                for chunk in iter(lambda: source.read(1024 * 1024), b''):
                    count += len(chunk)
                    if count > max_bytes:
                        raise ValueError('Download exceeded the declared size limit')
                    target.write(chunk)
                    progress.ui.update(path.name, time.monotonic() - started, count, total, initial=initial)
            if total and count != total:
                raise OSError('Incomplete download; rerun the same command to resume')
        checksum = digest(temp, algorithm)
        if not count or (expected and checksum != expected):
            temp.unlink(missing_ok=True)
            raise ValueError(f"Checksum/size verification failed: {path.name}")
        os.replace(temp, path)
        atomic_json(path.with_name(path.name + ".source.json"),
                    {"url": url, "bytes": count, algorithm: checksum})
        progress.ui.update(path.name, time.monotonic() - started, count, count, force=True, initial=initial)
        progress.ui.line(f'  {path.name}: verified ({progress.size(count)})')
        return path
    finally:
        if not expected:
            temp.unlink(missing_ok=True)
        if not temp.exists():
            receipt.unlink(missing_ok=True)


def extract(archive: Path, destination: Path, *, max_bytes=32 * 2**30):
    """Only regular files/directories; reject traversal, links and device nodes."""
    destination.mkdir(parents=True, exist_ok=True)
    # A tar containing NPZ/PyTorch ZIP members can also pass is_zipfile().
    # Recognize the outer tar before inspecting ZIP footer signatures.
    if not tarfile.is_tarfile(archive) and zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as handle:
            entries = handle.infolist()
            for entry in entries:
                inside(destination, entry.filename)
                mode = entry.external_attr >> 16
                if (mode & 0o170000) == 0o120000:
                    raise ValueError("Archive links are not permitted")
            if sum(x.file_size for x in entries) > max_bytes:
                raise ValueError("Archive is too large")
            handle.extractall(destination)
    else:
        with tarfile.open(archive) as handle:
            entries = handle.getmembers()
            for entry in entries:
                inside(destination, entry.name)
                if not entry.isdir() and not entry.isfile():
                    raise ValueError("Archive links/devices are not permitted")
            if sum(x.size for x in entries) > max_bytes:
                raise ValueError("Archive is too large")
            # Members have been checked explicitly, also on bootstrap Python 3.10.
            handle.extractall(destination, members=entries)


def clone(run, url, revision, folder: Path):
    if folder.exists():
        if not (folder / ".git").exists():
            raise ValueError(f"Existing source is not a managed Git checkout: {folder}")
        if capture(["git", "remote", "get-url", "origin"], folder).rstrip("/") != url.rstrip("/"):
            raise ValueError(f"Unexpected upstream at {folder}")
        head = capture(["git", "rev-parse", "HEAD"], folder)
        if len(revision) == 40 and head != revision:
            raise ValueError(f"Upstream revision differs at {folder}; no automatic reset")
        return head
    folder.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".clone-", dir=folder.parent) as temporary:
        target = Path(temporary) / "source"
        # Fetch blobs once. Partial clone causes checkout to open many extra
        # network requests and fails badly behind intermittent academic proxies.
        target.mkdir()
        run(["git", "init", target], timeout=30)
        run(["git", "remote", "add", "origin", url], cwd=target, timeout=30)
        run(["git", "-c", "http.lowSpeedLimit=1024", "-c", "http.lowSpeedTime=30",
             "fetch", "--depth", "1", "origin", revision], cwd=target, timeout=300)
        run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=target, timeout=60)
        head = capture(["git", "rev-parse", "HEAD"], target)
        os.replace(target, folder)
    return head


def github_asset(repo, tag, name, path):
    release = fetch_json(f"https://api.github.com/repos/{repo}/releases/tags/{tag}")
    selected = next((a for a in release["assets"] if a["name"] == name), None)
    if selected is None:
        raise ValueError(f"Upstream release has no asset: {name}")
    checksum = selected.get("digest", "") or ""
    return download(selected["browser_download_url"], path,
                    expected=checksum.removeprefix("sha256:") if checksum.startswith("sha256:") else None,
                    max_bytes=max(selected["size"], 1))
