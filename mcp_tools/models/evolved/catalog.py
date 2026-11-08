"""Short-lived recorded metadata shared by Agent search and Model Plaza.

Only discovery metadata is retained. Evidence verification and runtime policy
remain with their owning modules and are checked when a model is used.
"""

from collections import OrderedDict
from concurrent.futures import Future
from copy import deepcopy
import threading
import time

_CACHE = OrderedDict()
_PENDING = {}
_LOCK = threading.Lock()
_TTL_SECONDS = 15
_MAX_LIBRARIES = 8


def _key(library):
    index = library.root / "index"
    if index.is_symlink():
        raise ValueError("Symlinked model index")
    try:
        stat = index.stat()
        revision = (stat.st_dev, stat.st_ino, stat.st_mtime_ns)
    except FileNotFoundError:
        revision = None
    return str(library.root), revision


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
        value = library.discovery_metadata()
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
