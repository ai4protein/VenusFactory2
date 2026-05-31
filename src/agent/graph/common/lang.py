"""User-facing language detection."""
from __future__ import annotations

import re

_CJK_RE = re.compile("[一-鿿]")


def _detect_ui_lang(text: str) -> str:
    """Detect user-facing language from latest user text."""
    if isinstance(text, str) and _CJK_RE.search(text):
        return "zh"
    return "en"
