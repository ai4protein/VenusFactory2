"""Host-issued, persistent inference audits; full scientific audits remain explicit.

Only the operator prepares these records, outside agent/session mounts. A signed
record can skip recomputation, never checks of current files, ancestry or runtime.
Missing records use the original full verification path. Invalid records fail
closed until a new complete audit succeeds.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import stat

from agent.harness.contracts import digest

from . import inference_evidence
from .storage import atomic_json, file_lock, safe_child

SCHEMA = "scientist_inference_audit.v1"
MAX_RECORD_BYTES = 32 * 1024**2


def _area(library, *, create=False):
    path = safe_child(library.root, ".inference")
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.exists():
        info = path.stat()
        inference_evidence.require(path.is_dir() and info.st_uid == os.getuid() and not info.st_mode & 0o077,
                                   "Inference control directory must be private to the host")
    return path


def _key(area, *, create=False):
    path = safe_child(area, "signing-key")
    if create and not path.exists():
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(os.urandom(32))
            stream.flush()
            os.fsync(stream.fileno())
    info = path.lstat()
    inference_evidence.require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()
                               and not info.st_mode & 0o077 and info.st_size == 32,
                               "Missing or unsafe host inference signing key")
    return path.read_bytes()


def _read(path):
    inference_evidence.ordinary(path)
    inference_evidence.require(path.is_file() and path.stat().st_size <= MAX_RECORD_BYTES,
                               "Missing or oversized inference audit")
    return json.loads(path.read_text())


def _write(path, value):
    inference_evidence.require(len(json.dumps(value, ensure_ascii=False, indent=2).encode()) + 1 <= MAX_RECORD_BYTES,
                               "Inference audit exceeds its record budget")
    atomic_json(path, value)
    path.chmod(0o400)


def _signature(key, record):
    return hmac.new(key, digest(record).encode("ascii"), hashlib.sha256).hexdigest()


def prepare(library, entry_ids):
    """Perform full training/provenance replay once per origin, then seal evidence."""
    from .scientist_model_library import _identity, _require

    _require(isinstance(entry_ids, list) and 1 <= len(entry_ids) <= 256, "Declare 1..256 model identities")
    groups = {}
    for identity in dict.fromkeys(_identity(value) for value in entry_ids):
        _, _, _, folder = library._binding(identity)
        groups.setdefault(folder, []).append(identity)
    area = _area(library, create=True)
    outcomes = []
    with file_lock(area / ".prepare.lock"):
        key = _key(area, create=True)
        for identities in groups.values():
            before = inference_evidence.capture(inference_evidence.dependencies(library, identities))
            verified = library.resolve_many(identities, replay_training=True)
            after = inference_evidence.capture(inference_evidence.dependencies(library, identities))
            _require(before == after, "Evidence or runtime changed during the full inference audit")
            snapshot_id = digest(after)
            snapshot_path = safe_child(area, "snapshots/" + snapshot_id + ".json")
            if snapshot_path.exists():
                _require(_read(snapshot_path) == after, "Immutable inference snapshot changed")
            else:
                _write(snapshot_path, after)
            for identity in identities:
                record = {"schema": SCHEMA, "library": str(library.root), "entry_id": identity,
                          "snapshot_id": snapshot_id, "verified": verified[identity]}
                path = safe_child(area, "entries/" + identity + ".json")
                _write(path, {"record": record, "signature": _signature(key, record)})
                outcomes.append({"entry_id": identity, "audit_id": digest(record), "snapshot_id": snapshot_id})
    return {"schema": SCHEMA, "models": outcomes, "prepared_entries": len(outcomes),
            "training_replayed": True, "external_model_calls": 0, "new_autonomous_models": 0}


def resolve(library, entry_id):
    """Return a current bound packet, with no baseline fitting on a prepared entry."""
    from .scientist_model_library import _identity, _require

    entry_id = _identity(entry_id)
    area = _area(library)
    path = safe_child(area, "entries/" + entry_id + ".json")
    if not path.exists():
        return library.resolve(entry_id)
    envelope = _read(path)
    _require(set(envelope) == {"record", "signature"}, "Invalid inference audit envelope")
    record = envelope["record"]
    _require(isinstance(record, dict) and isinstance(envelope["signature"], str)
             and hmac.compare_digest(_signature(_key(area), record), envelope["signature"]),
             "Host inference audit signature changed")
    _require(set(record) == {"schema", "library", "entry_id", "snapshot_id", "verified"}
             and record["schema"] == SCHEMA and record["library"] == str(library.root)
             and record["entry_id"] == entry_id, "Inference audit identity changed")
    snapshot_id = _identity(record["snapshot_id"])
    snapshot = _read(safe_child(area, "snapshots/" + snapshot_id + ".json"))
    _require(digest(snapshot) == snapshot_id, "Inference evidence snapshot changed")
    # Recompute content hashes, including donor state and source datasets.
    # No result may be promoted from a timestamp or a model-written flag.
    current = inference_evidence.capture(snapshot["dependencies"])
    _require(current == snapshot, "Inference evidence or runtime changed; a full host audit is required")
    value = library._bind_verified(entry_id, library._binding(entry_id), record["verified"])
    return {**value, "inference_audit_id": digest(record)}
