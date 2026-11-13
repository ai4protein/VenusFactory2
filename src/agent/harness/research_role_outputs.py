"""Private bounded role-text evidence and explicit repair completeness.

Only the model's textual role output enters this module, never HTTP envelopes,
headers, API credentials, or request configuration. These files stay in the
host-owned episode control tree and are not workspace or delivery artifacts.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path

from utils.scoped_files import write_bytes

ROLES = frozenset({"PI", "CB", "SC", "REVIEW"})
MAX_SAVED_CHARACTERS = 256000
MAX_SAVED_BYTES = 1024 * 1024
MAX_REPAIR_CHARACTERS = 32000
MAX_REPAIR_BYTES = 128 * 1024
MAX_EXCERPT_CHARACTERS = 12000


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def record_response(folder, role, attempt, text):
    """Save one private response with an explicit, hash-bound completeness record."""
    if role not in ROLES or type(attempt) is not int or attempt not in {0, 1}:
        raise ValueError("Unsupported structured role response identity")
    if not isinstance(text, str):
        raise ValueError("A role response must be text, not a provider envelope")
    received = text.encode("utf-8")
    saved = text[:MAX_SAVED_CHARACTERS].encode("utf-8")[:MAX_SAVED_BYTES]
    # Never leave a cut UTF-8 character at a byte limit.
    saved = saved.decode("utf-8", errors="ignore").encode("utf-8")
    identity = role.lower() + "-" + uuid.uuid4().hex + "-" + str(attempt)
    relative = "role_outputs/" + identity + ".txt"
    metadata = {
        "schema": "scientist.private-role-response.v1",
        "role": role, "attempt": attempt, "created_at": time.time(),
        "path": relative, "encoding": "utf-8",
        "response_sha256": _digest(received), "response_bytes": len(received),
        "response_characters": len(text), "saved_sha256": _digest(saved),
        "saved_bytes": len(saved), "truncated": saved != received,
        "limits": {"characters": MAX_SAVED_CHARACTERS, "bytes": MAX_SAVED_BYTES},
        "scope": "private_episode_control",
    }
    # Anchor at the filesystem root so missing control directories are created
    # with mode 0700 and every ancestor is opened without following symlinks.
    # abspath is lexical: resolve() would silently accept linked ancestors.
    control = Path(os.path.abspath(folder))
    prefix = control.relative_to(control.anchor).as_posix()
    write_bytes(control.anchor, prefix + "/" + relative, saved)
    write_bytes(control.anchor, prefix + "/role_outputs/" + identity + ".json",
                (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode())
    return metadata


def repair_feedback(text, validation_error, instruction):
    """A bounded repair is either complete text or explicitly a non-JSON excerpt."""
    if not isinstance(text, str):
        raise ValueError("A role response must be text")
    raw = text.encode("utf-8")
    complete = len(text) <= MAX_REPAIR_CHARACTERS and len(raw) <= MAX_REPAIR_BYTES
    feedback = {
        "validation_error": str(validation_error)[:400],
        "instruction": instruction,
        "previous_response_complete": complete,
        "previous_response_sha256": _digest(raw),
        "previous_response_characters": len(text),
        "previous_response_bytes": len(raw),
    }
    if complete:
        feedback["previous_response"] = text
    else:
        feedback["previous_response_excerpt"] = text[:MAX_EXCERPT_CHARACTERS]
        feedback["previous_response_omitted_characters"] = len(text) - len(feedback["previous_response_excerpt"])
        feedback["instruction"] += (
            " The previous_response_excerpt is incomplete text, not a complete JSON document. "
            "Return a fresh complete response using the original evidence context and validation_error. "
            "Do not invent missing fields or claims from the excerpt; retain only supported evidence."
        )
    return feedback
