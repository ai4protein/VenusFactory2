"""Bounded tracked-only source snapshots for fixed PBS research workers."""
from __future__ import annotations

import hashlib
import json
import os
import re
import selectors
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath

MAX_BYTES = 512 * 1024 * 1024
MAX_FILES = 100000
ARCHIVE_SECONDS = 120


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def _relative(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or str(path) != name or "\\" in name or "\x00" in name:
        raise ValueError("Invalid source snapshot path")
    if ".git" in path.parts:
        raise ValueError("Git metadata cannot enter a source snapshot")
    if path.name == ".env" or (path.name.startswith(".env.") and not path.name.endswith((".example", ".sample", ".template"))):
        raise ValueError("Deployment environment files cannot enter a source snapshot")
    return path


def _git(repository, *args):
    result = subprocess.run(["git", "-C", str(repository), *args], capture_output=True, check=True, timeout=60)
    return result.stdout


def _archive(repository, commit, output):
    """Bound both archive transport bytes and total subprocess time."""
    deadline = time.monotonic() + ARCHIVE_SECONDS
    proc = subprocess.Popen(["git", "-C", str(repository), "archive", "--format=tar", commit],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    total = 0
    try:
        with selectors.DefaultSelector() as ready:
            ready.register(proc.stdout, selectors.EVENT_READ)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not ready.select(remaining):
                    raise TimeoutError("Source archive exceeded its deadline")
                data = os.read(proc.stdout.fileno(), 1024 * 1024)
                if not data:
                    break
                total += len(data)
                if total > MAX_BYTES:
                    raise ValueError("Source archive exceeds 512 MiB")
                output.write(data)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Source archive exceeded its deadline")
        if proc.wait(timeout=remaining):
            raise ValueError("Git source archive failed")
        output.seek(0)
        return deadline
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        proc.stdout.close()


def _exact_archive(repository, commit, output):
    """Ignore release-export attributes in a private object-only repository.

    The alternate object store is read-only from this operation: no refs, config
    or attributes in the source repository are changed, and no fetch occurs.
    """
    common = _git(repository, "rev-parse", "--git-common-dir").decode().strip()
    objects = (Path(repository) / common / "objects").resolve()
    if not objects.is_dir() or any(c in str(objects) for c in "\r\n\x00"):
        raise ValueError("Invalid source Git object directory")
    object_format = _git(repository, "rev-parse", "--show-object-format").decode().strip()
    if object_format not in {"sha1", "sha256"}:
        raise ValueError("Unsupported Git object format")
    with tempfile.TemporaryDirectory(prefix="pbs-source-objects-") as temporary:
        private = Path(temporary)
        _git(private, "init", "--bare", "--quiet", "--template=", "--object-format=" + object_format)
        (private / "objects/info/alternates").write_text(str(objects) + "\n")
        (private / "info").mkdir(exist_ok=True)
        # info/attributes outranks the committed release .gitattributes files.
        (private / "info/attributes").write_text("* -export-ignore -export-subst\n")
        return _archive(private, commit, output)


def freeze_source(repository: Path, destination: Path) -> dict:
    """Export a clean committed tree, without untracked deployment contents."""
    repository, destination = Path(repository).resolve(), Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError("Source snapshot destination already exists")
    if (_git(repository, "status", "--porcelain", "--untracked-files=no", "--ignore-submodules=all").strip()
            or _git(repository, "diff", "--cached", "--name-only", "--ignore-submodules=none").strip()):
        raise ValueError("Commit tracked source changes before freezing a PBS worker")
    commit = _git(repository, "rev-parse", "--verify", "HEAD^{commit}").decode().strip()
    expected, gitlinks = {}, {}
    for item in _git(repository, "ls-tree", "-r", "-z", "--full-tree", commit).split(b"\0"):
        if not item:
            continue
        metadata, name = item.split(b"\t", 1)
        mode, kind, oid = metadata.decode().split()
        relative = str(_relative(name.decode("utf-8")))
        if kind == "commit" and mode == "160000":
            # Gitlinks identify optional external source, not vendored bytes.
            # Never inspect an initialized host checkout or fetch its contents.
            gitlinks[relative] = oid
            continue
        if kind != "blob" or mode not in {"100644", "100755"}:
            raise ValueError("Source snapshots require regular tracked files, not symlinks")
        expected[relative] = (mode, oid)
    if not expected or len(expected) + len(gitlinks) > MAX_FILES:
        raise ValueError("Invalid source snapshot file count")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pbs-source-", dir=destination.parent) as temporary:
        stage = Path(temporary)
        files, total, members = {}, 0, 0
        with tempfile.TemporaryFile() as archive:
            deadline = _exact_archive(repository, commit, archive)
            with tarfile.open(fileobj=archive, mode="r|") as tree:
                for member in tree:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Source extraction exceeded its deadline")
                    members += 1
                    if members > MAX_FILES * 2:
                        raise ValueError("Source archive contains too many members")
                    relative = _relative(member.name.rstrip("/"))
                    if member.isdir():
                        continue
                    if not member.isfile() or str(relative) not in expected or str(relative) in files:
                        raise ValueError("Source archive contains an unexpected member")
                    total += member.size
                    if member.size < 0 or total > MAX_BYTES:
                        raise ValueError("Source snapshot exceeds 512 MiB")
                    target = stage / str(relative)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    content = tree.extractfile(member)
                    if content is None:
                        raise ValueError("Source archive file is unreadable")
                    with content, target.open("xb") as output:
                        shutil.copyfileobj(content, output, length=1024 * 1024)
                    if target.stat().st_size != member.size:
                        raise ValueError("Source archive file size changed")
                    mode, oid = expected[str(relative)]
                    blob = hashlib.new("sha1" if len(oid) == 40 else "sha256")
                    blob.update(f"blob {member.size}\0".encode())
                    with target.open("rb") as content:
                        for chunk in iter(lambda stream=content: stream.read(1024 * 1024), b""):
                            blob.update(chunk)
                    if blob.hexdigest() != oid:
                        raise ValueError("Archived source differs from the committed Git blob")
                    target.chmod(0o500 if mode == "100755" else 0o400)
                    files[str(relative)] = _hash(target)
        if set(files) != set(expected):
            raise ValueError("Git archive omitted tracked source files")
        os.rename(stage, destination)
        snapshot = {"root": str(destination.resolve()), "commit": commit, "files": files}
        identity = {"commit": commit, "files": files}
        if gitlinks:
            snapshot["gitlinks"] = gitlinks
            identity["gitlinks"] = gitlinks
        snapshot["sha256"] = _digest(identity)
        verify_source(snapshot)
        return snapshot


def verify_source(snapshot) -> None:
    """Verify every tracked byte; extra files and links invalidate the snapshot."""
    required = {"root", "commit", "files", "sha256"}
    if not isinstance(snapshot, dict) or not required <= set(snapshot) or set(snapshot) - required - {"gitlinks"}:
        raise ValueError("Invalid source snapshot manifest")
    root = Path(snapshot["root"])
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError("Invalid source snapshot root")
    if not isinstance(snapshot["commit"], str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", snapshot["commit"]):
        raise ValueError("Invalid source commit")
    files = snapshot["files"]
    if not isinstance(files, dict) or not 1 <= len(files) <= MAX_FILES:
        raise ValueError("Invalid source snapshot file count")
    gitlinks = snapshot.get("gitlinks", {})
    if not isinstance(gitlinks, dict) or len(files) + len(gitlinks) > MAX_FILES:
        raise ValueError("Invalid external source references")
    for name, commit in gitlinks.items():
        _relative(name)
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit):
            raise ValueError("Invalid external source commit")
    for name, value in files.items():
        path = _relative(name)
        if any(path == PurePosixPath(link) or path.is_relative_to(PurePosixPath(link)) for link in gitlinks):
            raise ValueError("External submodule contents cannot enter a source snapshot")
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("Invalid source file hash")
    identity = {"commit": snapshot["commit"], "files": files}
    if "gitlinks" in snapshot:
        identity["gitlinks"] = gitlinks
    if snapshot["sha256"] != _digest(identity):
        raise ValueError("Source snapshot manifest changed")
    actual, total = set(), 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Source snapshot contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("Source snapshot contains a special file")
        name = str(_relative(path.relative_to(root).as_posix()))
        actual.add(name)
        total += path.stat().st_size
        if total > MAX_BYTES or name not in files or _hash(path) != files[name]:
            raise ValueError("Source snapshot contents changed")
    if actual != set(files):
        raise ValueError("Source snapshot file inventory changed")
