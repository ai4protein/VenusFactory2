"""Content bindings for a host-audited model and its complete research ancestry.

This module fingerprints evidence; it cannot approve a publication. Numerical
and scientific acceptance remains in ScientistModelLibrary.resolve_many().
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import stat
import sys
import threading
from pathlib import Path

from agent.rsi._compat import relocated_source

from .storage import safe_child

MAX_FILES = 100_000
MAX_BYTES = 4 * 1024**3
_READ_BUFFER = threading.local()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ordinary(path):
    path = Path(path).absolute()
    require(".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
            "Inference evidence cannot traverse symlinks")
    return path


def file_digest(path, *, before=None):
    """Hash actual bytes, rejecting special files and replacement during reading."""
    before = before or path.lstat()
    require(stat.S_ISREG(before.st_mode), "Inference evidence must be a regular file")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb", buffering=0) as stream:
        opened = os.fstat(stream.fileno())
        require((opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino),
                "Inference evidence changed while opening")
        h = hashlib.sha256()
        if not hasattr(_READ_BUFFER, "data"):
            _READ_BUFFER.data = bytearray(1024 * 1024)
        view = memoryview(_READ_BUFFER.data)
        while size := stream.readinto(view):
            h.update(view[:size])
        after = os.fstat(stream.fileno())
    stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    require(stamp(before) == stamp(after) == stamp(path.lstat()),
            "Inference evidence changed while reading")
    return {"bytes": after.st_size, "sha256": h.hexdigest()}


class ScanBudget:
    def __init__(self):
        self.files = self.bytes = 0
        self.lock = threading.Lock()

    def consume(self, size):
        with self.lock:
            self.files += 1
            self.bytes += size
            require(self.files <= MAX_FILES and self.bytes <= MAX_BYTES,
                    "Inference evidence exceeds its scan budget")


def tree_digest(root, *, python_only=False, budget=None):
    """Overlap traversal and bounded batch reads, preserving every content check."""
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    root = ordinary(root)
    require(root.is_dir(), "Missing inference evidence directory")
    directories, files = [], {}
    budget = budget or ScanBudget()
    stack = [(str(root), "", False)]
    batch, pending = [], deque()

    def hash_batch(items):
        records = {}
        for relative, entry in items:
            before = entry.stat(follow_symlinks=False)
            budget.consume(before.st_size)
            records[relative] = file_digest(Path(entry.path), before=before)
        return records

    def submit(pool):
        if batch:
            pending.append(pool.submit(hash_batch, tuple(batch)))
            batch.clear()
        if len(pending) == 8:
            files.update(pending.popleft().result())

    # Four readers share bounded batches, so traversal can continue during I/O.
    # At most 128 files are queued or running; each reader uses a 1 MiB buffer.
    with ThreadPoolExecutor(max_workers=4) as pool:
        while stack:
            parent, prefix, parent_volatile = stack.pop()
            with os.scandir(parent) as entries:
                for entry in sorted(entries, key=lambda item: item.name):
                    relative = prefix + entry.name
                    require(not entry.is_symlink(), "Symlinked inference evidence")
                    volatile = parent_volatile or entry.name == "__pycache__"
                    if entry.is_dir(follow_symlinks=False):
                        stack.append((entry.path, relative + "/", volatile))
                        if not volatile and not python_only:
                            directories.append(relative)
                        continue
                    require(entry.is_file(follow_symlinks=False), "Nonregular inference evidence")
                    if volatile or entry.name.endswith(".lock") or (python_only and not entry.name.endswith(".py")):
                        continue
                    batch.append((relative, entry))
                    if len(batch) == 16:
                        submit(pool)
        submit(pool)
        while pending:
            files.update(pending.popleft().result())
    return {"directories": sorted(directories), "files": files}

def _read(path):
    import json
    ordinary(path)
    require(path.is_file() and path.stat().st_size <= 32 * 1024**2, "Missing or oversized inference dependency")
    return json.loads(path.read_text())


def _bound_files(value, root):
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and "sha256" in value:
            path = Path(value["path"])
            yield ordinary(path if path.is_absolute() else safe_child(root, value["path"]))
        for item in value.values():
            yield from _bound_files(item, root)
    elif isinstance(value, list):
        for item in value:
            yield from _bound_files(item, root)


def dependencies(library, entry_ids):
    """Follow the same declared donor graph as full publication verification."""
    from .scientist_model_library import ScientistModelLibrary

    trees, files, visited, goals = set(), set(), set(), set()
    stack = [(library, entry_id) for entry_id in entry_ids]
    while stack:
        store, identity = stack.pop()
        marker = (str(store.root), identity)
        if marker in visited:
            continue
        require(len(visited) < 256, "Too many inference ancestry dependencies")
        visited.add(marker)
        entry, key, origin, goal = store._binding(identity)
        files.update((
            store.root / "entries" / (identity + ".json"),
            store.root / "index" / (key + ".json"),
            store.root / "origins" / key / (entry["canonical_origin"] + ".json"),
            store.root / "packets" / (entry["packet_id"] + ".json"),
        ))
        if "runtime_migration" in origin:
            trees.add(safe_child(store.root, "migrations/" + origin["runtime_migration"]))
        if goal in goals:
            continue
        goals.add(goal)
        trees.add(goal)
        state, protocol = _read(goal / "goal.json"), _read(goal / "protocol.json")
        files.update(_bound_files(state, goal))
        files.update(_bound_files(protocol, goal))
        tasks = [goal / "model_task.json"] if (goal / "model_task.json").exists() else []
        tasks.extend(safe_child(goal, "targets/" + item["id"] + "/model_task.json")
                     for item in protocol.get("targets", []))
        for path in tasks:
            files.update(_bound_files(_read(path), goal))
        field = ("shared_library_experience" if protocol.get("schema") == "shared_scientist_goal.v1"
                 else "library_experience")
        if protocol.get(field) is not None:
            from .library_experience import read_frozen
            frozen = read_frozen(goal, state, protocol)
            selection = frozen["manifest"]["selection"]
            donor = ScientistModelLibrary(selection["library_root"])
            stack.extend((donor, entry_id) for entry_id in selection["entry_ids"])
    # Goal trees already bind their internal files; avoid hashing each twice.
    outside = [path for path in files if not any(path == root or root in path.parents for root in trees)]
    return {"trees": sorted(map(str, trees)), "files": sorted(map(str, outside))}


def runtime_identity():
    """Bind verification code, model code and numeric package versions."""
    agent_root = Path(__file__).resolve().parents[2]
    sources = {}
    for folder in ("rsi", "harness"):
        for name, record in tree_digest(agent_root / folder, python_only=True)["files"].items():
            sources[folder + "/" + name] = record
    for path in sorted(agent_root.glob("*.py")):
        sources[path.name] = file_digest(path)
    return {
        "sources": sources,
        "python": sys.version,
        "executable": file_digest(Path(sys.executable).resolve()),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn")},
    }


def capture(declared):
    budget = ScanBudget()
    trees = {name: tree_digest(Path(name), budget=budget) for name in declared["trees"]}
    files = {}
    for name in declared["files"]:
        logical_path = ordinary(name)
        path = ordinary(relocated_source(logical_path))
        budget.consume(path.lstat().st_size)
        files[name] = file_digest(path)
        if path != logical_path:
            files[name]["resolved_path"] = str(path)
    return {"dependencies": declared, "trees": trees, "files": files, "runtime": runtime_identity()}
