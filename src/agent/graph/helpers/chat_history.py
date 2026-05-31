"""Dialogue-history reconstruction utilities for prompts."""
from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


def _get_chat_history_messages(
    chains: dict[str, Any],
    history: list[dict[str, Any]],
    current_input: str = "",
) -> list[BaseMessage]:
    """Return dialogue context for prompts: previous user turns plus final model reports only."""
    dialogue_memory = chains.get("dialogue_memory") if isinstance(chains, dict) else None
    if isinstance(dialogue_memory, list) and dialogue_memory:
        out: list[BaseMessage] = []
        current = (current_input or "").strip()
        for item in dialogue_memory[-10:]:
            if not isinstance(item, dict):
                continue
            user = str(item.get("user") or item.get("input") or "").strip()
            assistant = str(item.get("assistant") or item.get("output") or "").strip()
            if user and not (current and user == current):
                out.append(HumanMessage(content=user))
            if assistant:
                out.append(AIMessage(content=assistant))
        if out:
            return out

    memory = chains.get("memory") if isinstance(chains, dict) else None
    try:
        messages = list(memory.chat_memory.messages) if memory is not None else []
    except Exception:
        messages = []
    if messages:
        return messages

    current = (current_input or "").strip()
    out: list[BaseMessage] = []
    for item in list(history or [])[-20:]:
        role = str(item.get("role") or "").strip().lower()
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        if role == "user" and current and content == current:
            continue
        if role == "user":
            out.append(HumanMessage(content=content))
    return out


def _format_conversation_history(
    chains: dict[str, Any],
    history: list[dict[str, Any]],
    current_input: str = "",
    limit: int = 10,
) -> str:
    rows = []
    for msg in _get_chat_history_messages(chains, history, current_input)[-limit:]:
        content = str(getattr(msg, "content", "") or "").strip()
        if not content:
            continue
        role = str(getattr(msg, "type", "") or getattr(msg, "role", "") or "").strip().lower()
        if not role:
            name = type(msg).__name__.lower()
            role = "user" if "human" in name else "assistant" if "ai" in name else "message"
        if role == "human":
            role = "user"
        elif role == "ai":
            role = "assistant"
        rows.append(f"{role}: {content}")
    return "\n".join(rows) if rows else "No previous conversation."
