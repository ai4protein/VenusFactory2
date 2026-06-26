"""Output redaction for kimi tool results / chat snapshots.

Two redaction profiles:

- `redact_for_kimi(text, current_session_id)`: aggressive. Used before
  storing tool outputs in session state, so kimi (when re-reading its own
  chat history on later turns) sees `/workspace` instead of the raw host
  session_dir, and `[other-session]` instead of cross-session paths.
  Also strips obvious API-key-shaped tokens.

- `redact_for_frontend(text)`: more lenient. The browser is the legitimate
  owner of its session so the session uuid is fine, but we still strip
  `/home/<user>` (host topology) and any key-shaped tokens that might have
  leaked through a tool's error message.

Both walk arbitrary nested dict/list structures so we can run them on tool
output JSON without losing shape — see `redact_obj_for_*`.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

# Project-local session root we hand kimi. Recomputed lazily so tests can
# monkey-patch the cwd without re-importing.
def _session_root() -> str:
    # Mirror web_v2.chat_api logic: temp_outputs/web_v2/sessions
    # under the project root. Falls back to a literal that won't match.
    try:
        # Walk up from this file: src/web_v2/redact.py → project root
        return str(
            (Path(__file__).resolve().parent.parent.parent
             / "temp_outputs" / "web_v2" / "sessions").resolve()
        )
    except OSError:
        return "/__no_match__/"


_HOST_SESSION_PATH_RE = re.compile(
    # Matches <session_root>/<uuid>(/...)? where uuid is the standard 8-4-4-4-12 form.
    r"(?P<root>/(?:home|root|var|opt|tmp|mnt|data)/[^\s\"',:;]*"
    r"/temp_outputs/web_v2/sessions)/(?P<sid>[0-9a-fA-F-]{36})(?P<rest>/[^\s\"',:;]*)?"
)

_HOME_RE = re.compile(r"/home/[a-z_][a-z0-9_-]*", re.IGNORECASE)

# API-key-value patterns — looks like `OPENAI_API_KEY=sk-abc123...` or
# `"api_key": "sk-..."` in JSON. Substring match is fine; we only want
# to redact the *value*, not the key name.
_KEY_VALUE_RE = re.compile(
    r"(?i)("
    r"(?:api[_-]?key|secret(?:_key)?|access[_-]?token|bearer|"
    r"password|passwd|private[_-]?key|client[_-]?secret|refresh[_-]?token|"
    r"openai[_-]?api[_-]?key|anthropic[_-]?api[_-]?key|deepseek[_-]?api[_-]?key|"
    r"hf[_-]?token|huggingface[_-]?token|aws[_-]?(?:access[_-]?key[_-]?id|secret[_-]?access[_-]?key|session[_-]?token))"
    r"[\"']?\s*[:=]\s*[\"']?)"
    # Value: 16+ chars of base62 / hyphen / underscore / dot, until end-of-token or quote
    r"([A-Za-z0-9_\-.+/=]{16,})"
)

# Bearer tokens / JWTs not preceded by a key= label (e.g. raw `Authorization: Bearer <jwt>`)
_BEARER_RE = re.compile(
    r"(?i)(authorization\s*:\s*bearer\s+)([A-Za-z0-9_\-.+/=]{16,})"
)

# SSH/PGP private key bodies
_PEM_KEY_RE = re.compile(
    r"-----BEGIN\s+(?:OPENSSH|RSA|DSA|EC|PGP)\s+PRIVATE\s+KEY-----.*?"
    r"-----END\s+(?:OPENSSH|RSA|DSA|EC|PGP)\s+PRIVATE\s+KEY-----",
    re.DOTALL | re.IGNORECASE,
)


def _redact_keys(text: str) -> str:
    """Strip API-key-shaped tokens. Order matters: PEM blocks first, then
    `Authorization: Bearer`, then generic key=value pairs."""
    text = _PEM_KEY_RE.sub("[REDACTED-PRIVATE-KEY]", text)
    text = _BEARER_RE.sub(r"\1[REDACTED]", text)
    text = _KEY_VALUE_RE.sub(r"\1[REDACTED]", text)
    return text


def _redact_host_paths(text: str, current_session_id: str = "") -> str:
    """Replace `<host_root>/temp_outputs/web_v2/sessions/<uuid>(/...)?`
    with `/workspace(/...)?` when uuid matches `current_session_id`, and
    with `[other-session]` otherwise."""
    def _sub(m: re.Match[str]) -> str:
        sid = m.group("sid")
        rest = m.group("rest") or ""
        if current_session_id and sid.lower() == current_session_id.lower():
            return "/workspace" + rest
        return "[other-session]"
    return _HOST_SESSION_PATH_RE.sub(_sub, text)


def _redact_home(text: str) -> str:
    return _HOME_RE.sub("[redacted-home]", text)


def redact_for_kimi(text: Any, *, current_session_id: str = "") -> str:
    """Strict redaction: host paths → /workspace, other sessions → marker,
    home dirs → marker, key values → [REDACTED]. Coerces non-str input to
    string. Returns a string (use `redact_obj_for_kimi` for structures)."""
    s = str(text or "")
    if not s:
        return s
    s = _redact_host_paths(s, current_session_id)
    s = _redact_home(s)
    s = _redact_keys(s)
    return s


def redact_for_frontend(text: Any) -> str:
    """Lenient redaction: only strip host topology and key values.
    Session uuid is preserved (frontend owns its session)."""
    s = str(text or "")
    if not s:
        return s
    s = _redact_home(s)
    s = _redact_keys(s)
    return s


# Dict keys whose values must always be redacted, even when the value
# alone wouldn't match _KEY_VALUE_RE (because in nested JSON the key name
# and value are separate strings — the regex only catches them when they
# appear together in one line). Case-insensitive substring match.
_SECRET_KEY_NAME_RE = re.compile(
    r"(?i)(?:^|[_-])(?:"
    r"api[_-]?key|secret(?:[_-]?key)?|access[_-]?token|refresh[_-]?token|"
    r"bearer|password|passwd|private[_-]?key|client[_-]?secret|"
    r"openai[_-]?key|anthropic[_-]?key|deepseek[_-]?key|moonshot[_-]?key|"
    r"hf[_-]?token|huggingface[_-]?token|"
    r"aws[_-]?(?:access[_-]?key[_-]?id|secret[_-]?access[_-]?key|session[_-]?token)|"
    r"github[_-]?token|gh[_-]?token"
    r")(?:[_-]|$)"
)


def _is_secret_key_name(name: str) -> bool:
    return bool(_SECRET_KEY_NAME_RE.search(name or ""))


def _walk(value: Any, leaf_fn) -> Any:
    """Recursively apply `leaf_fn` to every string leaf in dict/list/tuple.
    Dict values whose KEY name looks like a secret-bearing name are forced
    to a constant marker — covers cases where the value alone doesn't look
    key-shaped (short tokens, weird base of the rainbow)."""
    if isinstance(value, str):
        return leaf_fn(value)
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if isinstance(v, str) and isinstance(k, str) and _is_secret_key_name(k):
                out[k] = "[REDACTED]" if v else v
            else:
                out[k] = _walk(v, leaf_fn)
        return out
    if isinstance(value, (list, tuple)):
        return [_walk(v, leaf_fn) for v in value]
    return value


def redact_obj_for_kimi(obj: Any, *, current_session_id: str = "") -> Any:
    """Walk a JSON-ish structure, redacting every string leaf for kimi.
    Dict values keyed by a secret-shaped name are unconditionally redacted."""
    return _walk(obj, lambda s: redact_for_kimi(s, current_session_id=current_session_id))


def redact_obj_for_frontend(obj: Any) -> Any:
    """Walk a JSON-ish structure, redacting every string leaf for the UI.
    Dict values keyed by a secret-shaped name are unconditionally redacted."""
    return _walk(obj, redact_for_frontend)


# Re-export the regex so tests / callers can verify what we catch without
# poking the module internals.
__all__ = [
    "redact_for_kimi",
    "redact_for_frontend",
    "redact_obj_for_kimi",
    "redact_obj_for_frontend",
]
