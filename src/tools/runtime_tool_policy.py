"""Online/local capability policy for Agent tools, MCP tools, and heavy APIs.

Local: full tool surface.
Online: refuse training, VenusMine / FoldSeek discovery, and other high-compute
local workloads. Callers should use ``assert_tool_allowed`` / ``assert_local_feature``
for hard denials (not only UI readonly).
"""
from __future__ import annotations

import json
from typing import Any, Iterable, Sequence

from config import get_config

# Agent / MCP tool names disabled when WEBUI_V2_MODE=online.
ONLINE_DISABLED_TOOL_NAMES: frozenset[str] = frozenset({
    # Training + custom model reuse via agent/MCP
    "generate_training_config",
    "train_protein_model",
    "protein_model_predict",
    "register_trained_model",
    "list_trained_models",
    "agent_generated_code",
    # VenusMine / FoldSeek discovery (heavy local compute + large DB)
    # Agent hub name + MCP short name both listed.
    "download_foldseek_results_by_pdb_file",
    "download_foldseek_results",
    "query_foldseek_search_by_pdb_file",
})

# API feature keys for assert_local_feature(...)
ONLINE_DISABLED_FEATURES: frozenset[str] = frozenset({
    "training",
    "venusmine",
    "foldseek",
    "custom_model_training",
    "custom_model_evaluation",
    "custom_model_predict",
})


def is_online_mode() -> bool:
    try:
        return bool(get_config().server.is_online)
    except Exception:
        import os
        return os.getenv("WEBUI_V2_MODE", "local").strip().lower() == "online"


def is_local_mode() -> bool:
    return not is_online_mode()


def is_tool_allowed(tool_name: str) -> bool:
    if not is_online_mode():
        return True
    name = (tool_name or "").strip()
    if not name:
        return True
    if name in ONLINE_DISABLED_TOOL_NAMES:
        return False
    lower = name.lower()
    if "foldseek" in lower or "venusmine" in lower:
        return False
    return True


def online_disabled_reason(tool_name: str = "", *, feature: str = "") -> str:
    """JSON-friendly error message for online-mode denials."""
    label = (tool_name or feature or "this capability").strip()
    return (
        f"Unavailable in online mode: {label}. "
        "Heavy local compute (model training, VenusMine / FoldSeek discovery) "
        "is restricted to local deployments."
    )


def assert_tool_allowed(tool_name: str) -> None:
    if not is_tool_allowed(tool_name):
        raise RuntimeError(online_disabled_reason(tool_name=tool_name))


def assert_local_feature(feature: str) -> None:
    """Hard-deny for HTTP APIs (training, venusmine, …) when online."""
    if not is_online_mode():
        return
    key = (feature or "").strip().lower() or "local feature"
    if key in ONLINE_DISABLED_FEATURES or "train" in key or "venusmine" in key or "foldseek" in key:
        raise RuntimeError(online_disabled_reason(feature=key))


def tool_denied_json(tool_name: str) -> str:
    return json.dumps(
        {"success": False, "error": online_disabled_reason(tool_name=tool_name)},
        ensure_ascii=False,
    )


def filter_tools(tools: Sequence[Any]) -> tuple[list[Any], list[str]]:
    """Filter LangChain tools by runtime mode → (enabled, disabled_names)."""
    if not is_online_mode():
        return list(tools), []
    enabled: list[Any] = []
    disabled: list[str] = []
    for tool in tools:
        name = getattr(tool, "name", "") or ""
        if is_tool_allowed(name):
            enabled.append(tool)
        else:
            disabled.append(name)
    return enabled, sorted(set(disabled))


def iter_online_disabled_tool_names() -> Iterable[str]:
    return sorted(ONLINE_DISABLED_TOOL_NAMES)
