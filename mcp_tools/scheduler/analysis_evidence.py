"""Worker-owned evidence for successful scoped scientific computations.

The scheduler result, not the model or tool-returned metadata, owns this record.
Files are read through directory descriptors and only from declared arguments.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath

from mcp_tools.scheduler.receipt_binding import value_digest

ANALYSIS_CATEGORIES = frozenset({
    "sequence_analysis", "structure_analysis", "molecular_docking",
    "molecular_admet", "model_prediction", "remote_analysis",
})
MAX_FILES = 100
MAX_BYTES = 100 * 1024**2
MAX_ENTRIES = 1000
MAX_DEPTH = 32
_OUTPUT_KEYS = frozenset({
    "output_file", "output_path", "output_dir", "output_directory",
    "out_file", "out_path", "out_dir",
})
_PATH_KEY = re.compile(r"(?:^|_)(?:path|paths|file|files|dir|directory|folder|root)(?:$|_)", re.I)
_NON_PATH_KEYS = frozenset({"file_type", "file_format", "format", "config_name"})
_UNFINISHED = frozenset({
    "pending", "queued", "running", "submitted", "accepted", "processing",
    "in_progress", "in-progress", "waiting", "failed", "error", "cancelled", "canceled",
})


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _relative(workspace, value):
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise ValueError("Invalid analysis file argument")
    if value == "/workspace":
        value = "."
    elif value.startswith("/workspace/"):
        value = value[len("/workspace/"):]
    path = PurePosixPath(value)
    if path.is_absolute():
        path = path.relative_to(workspace.as_posix())
    if any(part.startswith(".") or ":" in part for part in path.parts):
        raise ValueError("Invalid analysis file argument")
    return path.as_posix()


def _declared_paths(arguments, workspace):
    inputs, outputs = set(), set()

    def visit(value, key=""):
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, child_key)
        elif isinstance(value, list):
            for child in value:
                visit(child, key)
        elif value is not None and value != "" and key not in _NON_PATH_KEYS and _PATH_KEY.search(key):
            target = outputs if key in _OUTPUT_KEYS else inputs
            target.add(_relative(workspace, value))

    visit(arguments)
    return sorted(inputs), sorted(outputs)


def _snapshot(workspace, paths, *, missing=False, budget=None):
    """Hash bounded regular files, with no link traversal or blocking FIFO opens."""
    files, identities = {}, {}
    budget = [0, 0] if budget is None else budget
    entries = 0
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    root_fd = os.open(workspace, flags | os.O_DIRECTORY)
    root_identity = os.fstat(root_fd)

    def scan(parent, name, relative, depth):
        nonlocal entries
        entries += 1
        if entries > MAX_ENTRIES or depth > MAX_DEPTH:
            raise ValueError("Analysis file traversal limit reached")
        try:
            fd = os.open(name, flags, dir_fd=parent)
        except FileNotFoundError:
            if missing:
                return
            raise
        try:
            info = os.fstat(fd)
            if stat.S_ISDIR(info.st_mode):
                for child in sorted(os.listdir(fd)):
                    if child.startswith("."):
                        raise ValueError("Hidden files are not analysis artifacts")
                    scan(fd, child, child if relative == "." else relative + "/" + child, depth + 1)
                if _identity(info) != _identity(os.fstat(fd)):
                    raise ValueError("Analysis directory changed while reading")
            elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                if relative in files:
                    return
                if budget[0] >= MAX_FILES or budget[1] + info.st_size > MAX_BYTES:
                    raise ValueError("Analysis evidence size limit reached")
                budget[0] += 1
                digest, count = hashlib.sha256(), 0
                while block := os.read(fd, min(1024**2, MAX_BYTES - budget[1] + 1)):
                    count += len(block)
                    budget[1] += len(block)
                    if budget[1] > MAX_BYTES:
                        raise ValueError("Analysis evidence size limit reached")
                    digest.update(block)
                if count != info.st_size or _identity(info) != _identity(os.fstat(fd)):
                    raise ValueError("Analysis file changed while reading")
                files[relative] = digest.hexdigest()
                identities[relative] = _identity(info)
            else:
                raise ValueError("Regular analysis files without links are required")
        finally:
            os.close(fd)

    try:
        for relative in paths:
            names = PurePosixPath(relative).parts
            parent = os.dup(root_fd)
            try:
                try:
                    for name in names[:-1]:
                        child = os.open(name, flags | os.O_DIRECTORY, dir_fd=parent)
                        os.close(parent)
                        parent = child
                except FileNotFoundError:
                    if missing:
                        continue
                    raise
                scan(parent, names[-1] if names else ".", relative, 0)
            finally:
                os.close(parent)
        current = os.stat(workspace, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (root_identity.st_dev, root_identity.st_ino):
            raise ValueError("Analysis workspace changed")
    finally:
        os.close(root_fd)
    return files, identities


def _finished(value):
    if value is None or (isinstance(value, str) and not value.strip()):
        return False
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return True
    if value is None or isinstance(value, (dict, list)) and not value:
        return False

    def check(item, depth=0):
        if depth > 32:
            return False
        if isinstance(item, dict):
            if item.get("is_error") is True:
                return False
            for key, child in item.items():
                if key in {"status", "state"} and isinstance(child, str) and child.lower() in _UNFINISHED:
                    return False
                if not check(child, depth + 1):
                    return False
        elif isinstance(item, list):
            return all(check(child, depth + 1) for child in item)
        return True

    return check(value)


class AnalysisCapture:
    """Return evidence or None; evidence capture never changes ordinary tool behavior."""

    def __init__(self, spec, workspace, category):
        self.valid = False
        try:
            workspace = Path(workspace)
            if (spec.get("route") != "native" or category not in ANALYSIS_CATEGORIES
                    or not spec.get("workspace") or not spec.get("session_id")
                    or workspace.is_symlink() or not workspace.is_absolute()
                    or Path(spec["workspace"]).resolve(strict=True) != workspace
                    or workspace.resolve(strict=True) != workspace):
                return
            info = workspace.stat()
            self.workspace_identity = (info.st_dev, info.st_ino)
            self.workspace = workspace
            self.tool = spec["name"]
            self.category = category
            self.arguments_sha256 = value_digest(spec["arguments"])
            self.inputs, self.outputs = _declared_paths(spec["arguments"], workspace)
            budget = [0, 0]
            self.input_files, self.input_identity = _snapshot(workspace, self.inputs, budget=budget)
            _, self.output_identity = _snapshot(workspace, self.outputs, missing=True, budget=budget)
            self.valid = True
        except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError):
            return

    def finish(self, result):
        if not self.valid:
            return None
        self.valid = False
        try:
            if result.get("is_error") is not False or "value" not in result or not _finished(result["value"]):
                return None
            info = self.workspace.stat(follow_symlinks=False)
            if (info.st_dev, info.st_ino) != self.workspace_identity:
                return None
            budget = [0, 0]
            inputs, identities = _snapshot(self.workspace, self.inputs, budget=budget)
            if inputs != self.input_files or identities != self.input_identity:
                return None
            outputs, output_identity = _snapshot(self.workspace, self.outputs, budget=budget)
            # Existing unrelated files in an output directory are not produced artifacts.
            outputs = {path: digest for path, digest in outputs.items()
                       if self.output_identity.get(path) != output_identity[path]}
            if self.outputs and not outputs:
                return None
            info = self.workspace.stat(follow_symlinks=False)
            if (info.st_dev, info.st_ino) != self.workspace_identity:
                return None
            return {
                "version": 1,
                "tool": self.tool,
                "category": self.category,
                "workspace_sha256": hashlib.sha256(str(self.workspace).encode()).hexdigest(),
                "arguments_sha256": self.arguments_sha256,
                "result_sha256": value_digest(result["value"]),
                "input_files": inputs,
                "output_files": outputs,
            }
        except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError):
            return None
