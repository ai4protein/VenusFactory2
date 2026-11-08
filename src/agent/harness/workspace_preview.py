"""Bounded UTF-8 previews through the session's anchored file descriptor."""
from __future__ import annotations

import codecs
import hashlib
import os

from utils.scoped_files import open_file


def text_window(workspace, path, *, offset_bytes=0, max_bytes=20000):
    if type(offset_bytes) is not int or offset_bytes < 0:
        raise ValueError("offset_bytes must be a nonnegative integer")
    if type(max_bytes) is not int or not 256 <= max_bytes <= 200000:
        raise ValueError("max_bytes must be between 256 and 200000")
    with open_file(workspace, path) as stream:
        size = os.fstat(stream.fileno()).st_size
        if offset_bytes > size:
            raise ValueError("offset_bytes is beyond this file")
        stream.seek(offset_bytes)
        raw = stream.read(min(max_bytes, size-offset_bytes))
    if b"\x00" in raw:
        raise ValueError("Binary file: inspect it with an appropriate scientific tool or isolated Python")
    try:
        decoder = codecs.getincrementaldecoder("utf-8")()
        text = decoder.decode(raw, final=offset_bytes+len(raw) >= size)
    except UnicodeDecodeError as exc:
        raise ValueError("Use UTF-8 text and the returned next_offset_bytes; inspect other encodings with isolated Python") from exc
    pending = decoder.getstate()[0]
    used = len(raw)-len(pending)
    end = offset_bytes+used
    truncated = offset_bytes != 0 or end < size
    result = {"path": path, "text": text, "bytes": size,
              "offset_bytes": offset_bytes, "truncated": truncated,
              "next_offset_bytes": end if end < size else None}
    result["slice_sha256" if truncated else "sha256"] = hashlib.sha256(raw[:used]).hexdigest()
    if end < size:
        result["next_step"] = "Read the next window if needed; use isolated Python for full-table analysis."
    return result
