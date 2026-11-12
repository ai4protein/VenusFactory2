"""Inspect decoded content rather than JSON transport escape sequences.

Tool arguments and results often embed JSON inside message strings. A result
may append a plain-text budget notice after that JSON. Consumers perform their
own disclosure checks on the resulting strings; nothing is removed or rewritten.
"""

from __future__ import annotations

import json


def decoded_strings(value, *, _depth=0):
    if _depth > 40:
        raise ValueError("Outbound content nesting exceeds the inspection limit")
    if isinstance(value, str):
        stripped = value.lstrip()
        if stripped.startswith(("{", "[", '"')):
            try:
                nested, end = json.JSONDecoder().raw_decode(stripped)
            except (ValueError, RecursionError):
                pass
            else:
                if isinstance(nested, (dict, list, str)):
                    yield from decoded_strings(nested, _depth=_depth + 1)
                    suffix = stripped[end:]
                    if suffix.strip():
                        yield from decoded_strings(suffix, _depth=_depth + 1)
                    return
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from decoded_strings(key, _depth=_depth + 1)
            yield from decoded_strings(item, _depth=_depth + 1)
    elif isinstance(value, list):
        for item in value:
            yield from decoded_strings(item, _depth=_depth + 1)
