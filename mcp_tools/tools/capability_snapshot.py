"""Read-only capability switches for a frozen standalone research execution."""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

SCHEMA = "venusfactory.capability-policy.v1"
MAX_BYTES = 16 * 1024 * 1024


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def document(rows, mode="local"):
    switches = {}
    for key, row in rows.items():
        if not isinstance(key, str) or len(key) > 1024 or not key.startswith(("tool:", "model:", "llm:")):
            raise ValueError("Invalid capability key")
        enabled, revision = row["enabled"], row["revision"]
        if type(enabled) not in (bool, int) or enabled not in (0, 1) or type(revision) is not int or revision < 0:
            raise ValueError("Invalid capability switch")
        selector = row["selector"]
        if isinstance(selector, str):
            selector = json.loads(selector)
        if not isinstance(selector, dict):
            raise ValueError("Invalid capability selector")
        switches[key] = {"enabled": bool(enabled), "revision": revision, "selector": selector}
    value = {"schema": SCHEMA, "mode": mode, "switches": switches}
    return {**value, "sha256": fingerprint(value)}


def read(path):
    path = Path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError("Execution policy must be a private regular file")
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("Execution policy is too large")
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"schema", "mode", "switches", "sha256"} or value["schema"] != SCHEMA:
        raise ValueError("Invalid execution policy")
    from .capability_policy import mode
    if value["mode"] != mode() or not isinstance(value["switches"], dict):
        raise ValueError("Execution policy belongs to a different mode")
    expected = document(value["switches"], value["mode"])
    if value != expected:
        raise ValueError("Execution policy changed")
    return {key: {"key": key, "enabled": int(row["enabled"]), "revision": row["revision"],
                  "selector": json.dumps(row["selector"])} for key, row in value["switches"].items()}
