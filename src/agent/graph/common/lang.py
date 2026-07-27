"""User-facing language detection / resolution."""
from __future__ import annotations

import re
from typing import Any

_CJK_RE = re.compile("[一-鿿]")


def _detect_ui_lang(text: str) -> str:
    """Detect user-facing language from latest user text."""
    if isinstance(text, str) and _CJK_RE.search(text):
        return "zh"
    return "en"


def _resolve_ui_lang(state: Any = None, text: str | None = None) -> str:
    """Prefer explicit session/state lang (``user_lang`` / ``ui_lang``), else detect.

    Order: ``user_lang`` → ``ui_lang`` → CJK detect on ``text`` / last message.
    """
    if state is not None and hasattr(state, "get"):
        for key in ("user_lang", "ui_lang"):
            v = state.get(key)
            if v in ("en", "zh"):
                return str(v)
        if text is None:
            messages = state.get("messages") or []
            if messages:
                content = getattr(messages[-1], "content", None)
                if isinstance(content, str):
                    text = content
    return _detect_ui_lang(text or "")
