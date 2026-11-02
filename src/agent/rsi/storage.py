"""Small atomic storage primitives. Experiments use one writer per run."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextlib.contextmanager
def file_lock(path: Path):
    """Interprocess lock for the Linux research runner."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def safe_child(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if not relative or rel.is_absolute() or ".." in rel.parts:
        raise ValueError("expected a nonempty relative path without traversal")
    root = root.resolve()
    result = root / rel
    cursor = root
    for part in rel.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError("symlinks are not allowed in experiment artifacts")
    result.resolve().relative_to(root)
    return result


@contextlib.contextmanager
def case_gpu_lock(path: Path):
    """Reuse a verified controller-owned lock, or acquire it for a direct run.

    The broker passes the open file description with pass_fds. A plain boolean
    environment flag would permit accidentally running without a lock.
    """
    import fcntl

    inherited = os.environ.get("VENUS_RSI_CASE_LOCK_FD")
    if inherited is None:
        with file_lock(path):
            yield
        return
    fd = int(inherited)
    actual, expected = os.fstat(fd), path.stat()
    if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
        raise ValueError("inherited GPU lock does not name the declared lock file")
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Do not unlock the shared open file description. Its lifetime belongs to
    # the broker and remains locked until it finishes collecting this job.
    yield
