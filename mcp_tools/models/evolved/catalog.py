"""Short-lived recorded metadata shared by Agent search and Model Plaza.

Only discovery metadata is retained. A private host-authenticated disk record
survives worker restarts. Registry file identities and change times invalidate it;
these are cache invalidators, never scientific or inference verification.
Evidence verification and runtime policy are checked when a model is used.
"""

import errno
import hashlib
import hmac
import json
import os
import stat
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future
from copy import deepcopy

from agent.rsi.engine.storage import atomic_json, file_lock, safe_child

_CACHE = OrderedDict()
_PENDING = {}
_LOCK = threading.Lock()
_TTL_SECONDS = 15
_MAX_LIBRARIES = 8


_SCHEMA = "evolved_discovery_metadata.v1"
_MAX_RECORD_BYTES = 64 * 1024**2
_MAX_REGISTRY_PATHS = 250_000
_AREAS = ("index", "entries", "origins", "packets", "curation")


def _encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _key(library):
    # ctime also changes when contents are overwritten and mtime is restored.
    # Only host registry files participate; usage receipts and inference audits
    # do not alter recorded catalog metadata.
    records = []
    pending = [library.root / name for name in _AREAS]
    while pending:
        path = pending.pop()
        try:
            info = path.lstat()
        except FileNotFoundError:
            records.append((str(path.relative_to(library.root)), None))
            if len(records) > _MAX_REGISTRY_PATHS:
                raise ValueError("Model discovery registry exceeds its path budget") from None
            continue
        if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
            raise ValueError("Symlinked or non-regular model registry path")
        records.append((str(path.relative_to(library.root)), info.st_dev, info.st_ino,
                        info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_mode))
        if len(records) > _MAX_REGISTRY_PATHS:
            raise ValueError("Model discovery registry exceeds its path budget")
        if stat.S_ISDIR(info.st_mode):
            pending.extend(path.iterdir())
    return str(library.root), hashlib.sha256(_encoded(sorted(records))).hexdigest()


def _private(path, *, directory=False):
    info = path.lstat()
    valid_kind = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
    if not valid_kind or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("Discovery cache must be private to the host")
    return info


def _area(library):
    if not hasattr(os, "getuid") or not hasattr(os, "O_NOFOLLOW"):
        return None
    if not library.root.is_dir():
        return None
    area = safe_child(library.root, ".discovery")
    try:
        area.mkdir(mode=0o700, exist_ok=True)
    except OSError:
        # A read-only installation can still discover models via the original
        # immutable-object reader; persistence is an optional optimization.
        if not area.exists():
            return None
        raise
    _private(area, directory=True)
    return area


def _signing_key(area, *, create=False):
    path = safe_child(area, "signing-key")
    if create and not path.exists():
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(os.urandom(32))
            stream.flush()
            os.fsync(stream.fileno())
    info = _private(path)
    if info.st_size != 32 or info.st_nlink != 1:
        raise ValueError("Invalid discovery signing key")
    return path.read_bytes()


def _signature(key, value):
    return hmac.new(key, _encoded(value), hashlib.sha256).hexdigest()


def _read_record(area, key):
    path = safe_child(area, "catalog.json")
    if not path.exists():
        return None
    info = _private(path)
    if info.st_size > _MAX_RECORD_BYTES:
        return None
    try:
        with path.open("rb") as stream:
            raw = stream.read(_MAX_RECORD_BYTES + 1)
        if len(raw) > _MAX_RECORD_BYTES:
            return None
        envelope = json.loads(raw)
        if not isinstance(envelope, dict) or set(envelope) != {"record", "signature"}:
            return None
        record, signature = envelope["record"], envelope["signature"]
        if (not isinstance(record, dict) or set(record) != {"schema", "library", "registry", "metadata"}
                or record["schema"] != _SCHEMA or (record["library"], record["registry"]) != key
                or not isinstance(signature, str)):
            return None
        if not hmac.compare_digest(_signature(_signing_key(area), record), signature):
            return None
        return record["metadata"]
    except (json.JSONDecodeError, UnicodeError, FileNotFoundError):
        return None


def _persistent_metadata(library, key):
    area = _area(library)
    if area is None:
        return library.discovery_metadata()
    value = _read_record(area, key)
    if value is not None:
        return value
    # The process-local Future coalesces threads; the host lock coalesces workers.
    try:
        with file_lock(safe_child(area, ".build.lock")):
            value = _read_record(area, key)
            if value is not None:
                return value
            value = library.discovery_metadata()
            if _key(library) != key:
                raise ValueError("Model catalog changed during discovery")
            record = {"schema": _SCHEMA, "library": key[0], "registry": key[1], "metadata": value}
            envelope = {"record": record, "signature": _signature(_signing_key(area, create=True), record)}
            if len(json.dumps(envelope, ensure_ascii=False, indent=2).encode()) + 1 <= _MAX_RECORD_BYTES:
                target = safe_child(area, "catalog.json")
                atomic_json(target, envelope)
                target.chmod(0o600)
            return value
    except OSError as exc:
        if exc.errno not in {errno.EACCES, errno.EROFS}:
            raise
        return library.discovery_metadata()


def recorded_metadata(library):
    key = _key(library)
    with _LOCK:
        cached = _CACHE.get(key)
        if cached is not None and time.monotonic() < cached[0]:
            _CACHE.move_to_end(key)
            value = cached[1]
            future = None
            owner = False
        else:
            value = None
            future = _PENDING.get(key)
            owner = future is None
            if owner:
                future = Future()
                _PENDING[key] = future
    if value is not None:
        return deepcopy(value)
    if not owner:
        return deepcopy(future.result())
    try:
        value = _persistent_metadata(library, key)
        if _key(library) != key:
            raise ValueError("Model catalog changed during discovery")
        with _LOCK:
            # A large library can take longer than the TTL to read. Its lifetime
            # starts after that work, so concurrent and following callers reuse it.
            _CACHE[key] = (time.monotonic() + _TTL_SECONDS, value)
            _CACHE.move_to_end(key)
            while len(_CACHE) > _MAX_LIBRARIES:
                _CACHE.popitem(last=False)
    except BaseException as exc:
        future.set_exception(exc)
        raise
    else:
        future.set_result(value)
    finally:
        with _LOCK:
            _PENDING.pop(key, None)
    return deepcopy(value)
