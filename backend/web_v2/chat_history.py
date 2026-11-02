"""Lightweight history metadata; empty drafts remain stored and recoverable."""
import json
import re

from agent.session_store import shorten_session_title


def research_status(folder, live=None):
    """Read only session-owned goal metadata; never scan workspaces or resume work."""
    rows = []
    for meta_path in folder.glob("*.json"):
        if not re.fullmatch(r"[0-9a-f]{32}", meta_path.stem) or meta_path.is_symlink():
            continue
        goal_folder = folder / meta_path.stem
        path = goal_folder / "goal.json"
        if path.is_symlink() or goal_folder.is_symlink():
            continue
        try:
            metadata = json.loads(meta_path.read_text())
            state = json.loads(path.read_text()) if path.is_file() else {}
            status = state.get("status", metadata.get("status", "preparing"))
            if status == "active" and live is not None and not live(goal_folder):
                status = "interrupted"
            rows.append((status in {"active", "preparing", "queued"}, metadata.get("created_at", 0), status))
        except (OSError, ValueError, TypeError, KeyError):
            continue
    return max(rows, default=(False, 0, ""))[-1]


def include_research_content(summary, folder, live=None):
    """Keep goals and uploaded research files even before a chat turn exists."""
    item = dict(summary)
    # Older Scientist goals may predate the history echo / explicit chat mode.
    if any(folder.glob("*.json")):
        item["chat_mode"] = "science_expert"
        item["status"] = research_status(folder, live) or item.get("status", "")
    if item.get("has_content") or item.get("history_size") or item.get("title"):
        item["has_content"] = True
        return item
    # Read only metadata: listing chats must not reconcile or resume workers.
    try:
        for path in sorted(folder.glob("*.json")):
            item["has_content"] = True
            metadata = json.loads(path.read_text())
            objective = (metadata.get("spec") or {}).get("objective", "")
            item["title"] = shorten_session_title(objective)
            if item["title"]:
                return item
        if not item.get("has_content"):
            item["has_content"] = any(path.is_file() for path in folder.rglob("*")
                                      if not path.name.startswith("."))
            if item["has_content"]:
                item["chat_mode"] = "science_expert"
    except (OSError, ValueError, TypeError, AttributeError):
        # Keep an unreadable record visible rather than assuming it is empty.
        item["has_content"] = True
    return item
