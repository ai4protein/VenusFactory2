"""Shared download notices / progress helpers for ckpt & frpc hubs.

Environment:
  VENUS_DOWNLOAD_QUIET=1          suppress user-facing notices (CI)
  HF_HUB_DISABLE_PROGRESS_BARS=1  also treated as quiet for our notices
"""
from __future__ import annotations

import os
import sys
import threading
from collections.abc import Callable
from contextlib import contextmanager
from typing import Iterator

_Listener = Callable[[str], None]
_listeners: list[_Listener] = []
_listeners_guard = threading.Lock()
_tls = threading.local()


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def quiet() -> bool:
    return _env_bool("VENUS_DOWNLOAD_QUIET") or _env_bool("HF_HUB_DISABLE_PROGRESS_BARS")


def format_bytes(num: int | float) -> str:
    n = float(num)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            if unit == "B":
                return f"{int(n)} {unit}"
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def add_listener(callback: _Listener) -> None:
    with _listeners_guard:
        _listeners.append(callback)


def remove_listener(callback: _Listener) -> None:
    with _listeners_guard:
        try:
            _listeners.remove(callback)
        except ValueError:
            pass


@contextmanager
def capture_notices() -> Iterator[list[str]]:
    """Collect announce() messages for the current thread (useful in tests / SSE)."""
    bucket: list[str] = []
    previous = getattr(_tls, "bucket", None)
    _tls.bucket = bucket
    try:
        yield bucket
    finally:
        _tls.bucket = previous


def announce(message: str, *, log=None) -> None:
    """Emit a user-facing download notice (+ optional logger)."""
    text = message.rstrip()
    if not text:
        return
    bucket = getattr(_tls, "bucket", None)
    if isinstance(bucket, list):
        bucket.append(text)
    with _listeners_guard:
        listeners = list(_listeners)
    for cb in listeners:
        try:
            cb(text)
        except Exception:
            pass
    if log is not None:
        try:
            log.info("%s", text)
        except Exception:
            pass
    if quiet():
        return
    try:
        print(text, file=sys.stderr, flush=True)
    except Exception:
        pass


def urllib_reporthook(label: str = "Downloading"):
    """Return a urllib.urlretrieve reporthook that prints coarse percent updates."""
    state = {"last": -1}

    def _hook(block_num: int, block_size: int, total_size: int) -> None:
        if quiet() or total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = min(100, int(downloaded * 100 / total_size))
        # Update every 10% to avoid spam.
        if pct < 100 and pct // 10 <= state["last"] // 10:
            return
        if pct == state["last"]:
            return
        state["last"] = pct
        announce(f"{label}: {pct}% ({format_bytes(min(downloaded, total_size))} / {format_bytes(total_size)})")

    return _hook
