"""Reuse metadata within one catalog response; never keep it across requests."""
from contextvars import ContextVar
from functools import wraps

_current = ContextVar('model_catalog_snapshot', default=None)


def cached(key, read):
    values = _current.get()
    if values is None:
        return read()
    if key not in values:
        values[key] = read()
    return values[key]


def catalog_snapshot(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        if _current.get() is not None:
            return fn(*args, **kwargs)
        token = _current.set({})
        try:
            return fn(*args, **kwargs)
        finally:
            _current.reset(token)
    return wrapped
