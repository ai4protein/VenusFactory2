"""Explicit local CLI inputs, independent from generated workspace files.

This is an operator CLI helper, never an online upload or model-call surface.
"""
from __future__ import annotations

import hashlib
import os
import time
import uuid
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from utils.scoped_files import open_absolute, parent_fd

from .host import write_json

CHUNK_BYTES = 1024**2
MAX_INPUTS = 32


@dataclass
class PreparedInput:
    name: str
    stream: BinaryIO
    identity: tuple


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _local_only(paths):
    if paths and (os.getenv("WEBUI_V2_MODE", "").strip().lower() == "online"
                  or os.getenv("VENUS_SCIENTIST_ONLINE") == "1"):
        raise ValueError("Local --input paths are unavailable in online mode; use session uploads")


@contextmanager
def prepare_inputs(paths):
    """Validate every source before goal creation; keep descriptors until copied."""
    paths = list(paths)
    _local_only(paths)
    if len(paths) > MAX_INPUTS:
        raise ValueError("Provide at most 32 input files")
    names = set()
    sources = []
    for value in paths:
        source = Path(os.path.abspath(Path(value).expanduser()))
        name = source.name
        if (not name or name.startswith(".") or name != name.strip() or len(name) > 180
                or any(char in name for char in "/\\:") or any(ord(char) < 32 for char in name)):
            raise ValueError("Use a nonhidden, safe input filename of at most 180 characters")
        if name.casefold() in names:
            raise ValueError("Input filenames must be unique; rename duplicate files before creating the goal")
        names.add(name.casefold())
        sources.append((source, name))
    with ExitStack() as stack:
        prepared = []
        for source, name in sources:
            stream = stack.enter_context(open_absolute(source))
            info = os.fstat(stream.fileno())
            if info.st_size == 0:
                raise ValueError("Choose nonempty input files")
            prepared.append(PreparedInput(name, stream, _identity(info)))
        yield prepared


def _copy_input(workspace, item):
    """Exclusive destination creation; retain incomplete copies as failure evidence."""
    if _identity(os.fstat(item.stream.fileno())) != item.identity:
        raise ValueError("An input changed after validation; create a new goal from stable files")
    relative = "inputs/" + item.name
    with parent_fd(workspace, relative, create=True) as (directory, name):
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                     0o600, dir_fd=directory)
        digest, size = hashlib.sha256(), 0
        with os.fdopen(fd, "wb") as target:
            while block := item.stream.read(CHUNK_BYTES):
                size += len(block)
                digest.update(block)
                target.write(block)
            target.flush()
            os.fsync(target.fileno())
        if size != item.identity[2] or _identity(os.fstat(item.stream.fileno())) != item.identity:
            raise ValueError("An input changed during copying; the incomplete preparation is retained")
    return {"id": uuid.uuid4().hex, "name": item.name, "size": size,
            "sha256": digest.hexdigest(), "path": relative, "source_type": "cli"}


def stage_inputs(store, prepared):
    """Register explicit snapshots; apply the existing model-data protection flow."""
    prepared = list(prepared)
    _local_only(prepared)
    state = store.read()
    if state.get("cycles") or state.get("inflight") or state["progress"].get("inputs"):
        raise ValueError("Inputs are registered only during goal creation; later attachments use research revision")
    receipt_path = store.root / "cli-input-preparation.json"
    if receipt_path.exists():
        raise ValueError("Input preparation already exists; create a new goal after reviewing its diagnostics")
    records = []
    receipt = {"schema": "scientist.cli-inputs.v1", "status": "preparing",
               "names": [item.name for item in prepared], "inputs": records}
    write_json(receipt_path, receipt)
    current_name = None
    try:
        for item in prepared:
            current_name = item.name
            records.append(_copy_input(store.root / "workspace", item))
            write_json(receipt_path, receipt)
        from .research_data_store import store_for_goal

        data_store = store_for_goal(store.root)
        if data_store is not None:
            data_store.protect_inputs(records, prior_agent_execution=False)
        store.mutate(lambda current: current["progress"].update(inputs=records))
        receipt["status"] = "completed"
        write_json(receipt_path, receipt)
        return records
    except Exception as exc:
        error_type = type(exc).__name__
        receipt.update(status="failed", failed_input=current_name, error_type=error_type)
        write_json(receipt_path, receipt)

        def failed(current):
            current["status"] = "interrupted"
            current["progress"].update(inputs=records, input_preparation={"status": "failed",
                "receipt": "cli-input-preparation.json", "error_type": error_type})
            current["events"].append({"event": "input_preparation_failed", "at": time.time()})

        store.mutate(failed)
        raise RuntimeError("Input preparation failed; files and diagnostics are retained in the goal directory") from exc
