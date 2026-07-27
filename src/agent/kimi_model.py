"""Map VenusFactory registry model ids → kimi-code ``agent_config.model`` ids.

kimi-code addresses models as ``{provider}/{model}`` (see ``~/.kimi-code/config.toml``),
e.g. ``deepseek/deepseek-v4-pro``. VF registry ids are usually the bare model
name (``deepseek-v4-pro``). Science Agent always routes through kimi-code; in
local mode the UI may pick an underlying LLM that we forward here.
"""
from __future__ import annotations

from agent.model_registry import get_model


def to_kimi_model_id(model_id: str | None) -> str | None:
    """Return kimi model id, or ``None`` to leave kimi on its ``default_model``.

    - empty / ``kimi-code`` → None (daemon default)
    - already ``provider/model`` → pass through
    - registry hit → ``{provider}/{id}``
    - unknown bare id → pass through (kimi may still resolve it)
    """
    mid = (model_id or "").strip()
    if not mid or mid == "kimi-code":
        return None
    if "/" in mid:
        return mid
    spec = get_model(mid)
    if spec is not None and spec.provider:
        return f"{spec.provider}/{spec.id}"
    return mid
