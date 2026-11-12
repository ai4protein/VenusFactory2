"""Canonical, non-secret task-argument binding shared by submitters and readers."""

import hashlib
import json


def value_digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()
