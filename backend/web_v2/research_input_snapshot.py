"""Immutable, owner-scoped snapshots of inputs actually used by Researcher."""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile

SCHEMA = "venusfactory.research-input.v1"


def _root(folder, *, create=False):
    folder = Path(folder)
    if folder.is_symlink():
        raise ValueError("Research input snapshot directory is invalid")
    if create:
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    return folder


def _bytes(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("Research input snapshot is not a regular file")
        return stream.read()


def save_inputs(folder, context):
    folder = _root(folder, create=True)
    value = {**context, "schema": SCHEMA}
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    path = folder / (digest + ".json")
    with tempfile.NamedTemporaryFile(prefix=".input-", dir=folder) as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
        try:
            os.link(stream.name, path, follow_symlinks=False)
        except FileExistsError:
            if _bytes(path) != raw:
                raise ValueError("Research input snapshot changed")
        directory = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    sequence = value.get("sequence", "")
    structure = value.get("structure_pdb", "").encode("utf-8")
    return {"schema": SCHEMA, "sha256": digest, "bytes": len(raw),
            "sequence_sha256": hashlib.sha256(sequence.encode("utf-8")).hexdigest() if sequence else None,
            "sequence_length": len(sequence),
            "structure_sha256": hashlib.sha256(structure).hexdigest() if structure else None,
            "structure_bytes": len(structure),
            **{k: value.get(k) for k in ("origin", "accession", "pdb_id", "structure_chain",
                                         "source_chain", "recorded_at")}}


def load_inputs(folder, reference):
    if not isinstance(reference, dict) or reference.get("schema") != SCHEMA:
        raise ValueError("Research input snapshot reference is invalid")
    digest = reference.get("sha256")
    if not isinstance(digest, str) or not re.fullmatch("[a-f0-9]{64}", digest):
        raise ValueError("Research input snapshot identifier is invalid")
    try:
        raw = _bytes(_root(folder) / (digest + ".json"))
    except OSError:
        raise ValueError("Research input snapshot is unavailable") from None
    if len(raw) != reference.get("bytes") or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Research input snapshot changed")
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("Research input snapshot schema is invalid")
    return value
