"""Resolve private Scientist deliveries from a live, host-owned task lease."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from agent.rsi.artifacts.contract import selected_deliveries
from agent.science_goal import GoalStore
from utils.scoped_files import read_bytes

from .job_queue import JobQueue

PROJECT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ResearchLibrary:
    path: Path
    goal: Path
    deliveries: dict

    def verify(self, identity):
        from agent.harness.scientific_acceptance import verify_delivery

        delivery = self.deliveries.get(identity)
        if not delivery:
            raise ValueError("Model unavailable. Search for an accessible model.")
        if delivery.get("kind") == "model" and not verify_delivery(self.goal, delivery):
            raise ValueError("The model's scientific acceptance evidence changed.")


def owned_sessions():
    """No model-supplied identity, filesystem path or HTTP header grants access."""
    identity = os.getenv("VENUS_QUEUE_JOB_ID", "")
    owner = os.getenv("VENUS_QUEUE_OWNER", "")
    lease = os.getenv("VENUS_QUEUE_LEASE_TOKEN", "")
    if not identity or not owner or not lease:
        return []
    queue = JobQueue()
    try:
        row = queue.get(identity, owner)
        if not secrets.compare_digest(str(row.get("lease") or ""), lease):
            return []
        for _ in range(8):
            if row["status"] != "running" or row["cancel_requested"]:
                return []
            parent = row["metadata"].get("parent_job")
            if not parent:
                break
            row = queue.get(parent, owner)
        else:
            return []
    except (FileNotFoundError, ValueError):
        return []
    if row["kind"] not in {"agent", "scientist", "researcher"}:
        return []
    session = row["metadata"].get("session_id")
    if not isinstance(session, str) or not session:
        return []
    database = Path(os.getenv("VENUSFACTORY_SESSION_DB_PATH") or PROJECT / "data/chat_sessions.db")
    if not database.is_file():
        return []
    try:
        with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True, timeout=2) as connection:
            current = connection.execute("SELECT owner_key FROM sessions WHERE session_id=?", (session,)).fetchone()
            if not current or not current[0]:
                return []
            # Older local Scientist jobs use the hashed session directory as the
            # queue owner. Their persisted session still determines the user.
            local_scientist = (
                os.getenv("WEBUI_V2_MODE", "local") == "local"
                and row["kind"] == "scientist"
                and owner == hashlib.sha256(session.encode()).hexdigest()
            )
            if current[0] != owner and not local_scientist:
                return []
            return [
                item[0]
                for item in connection.execute(
                    "SELECT session_id FROM sessions WHERE owner_key=? ORDER BY session_id", (current[0],)
                )
            ]
    except sqlite3.Error:
        return []


def private_libraries():
    """Only currently owned sessions and selected, reviewed deliveries are visible."""
    state_root = Path(os.getenv("VENUS_SCIENCE_GOAL_ROOT") or PROJECT / "research_runs/rsi/web_goals")
    libraries = []
    for session in owned_sessions():
        prefix = hashlib.sha256(session.encode()).hexdigest()
        folder = state_root / prefix
        if folder.is_symlink() or not folder.is_dir():
            continue
        for metadata in sorted(folder.glob("*.json")):
            if not re.fullmatch(r"[a-f0-9]{32}", metadata.stem):
                continue
            try:
                meta = json.loads(read_bytes(state_root, prefix + "/" + metadata.name, max_bytes=100000))
                if meta.get("session_id") != session or meta.get("spec", {}).get("driver") != "autonomous_research.v1":
                    continue
                goal = folder / metadata.stem
                if goal.is_symlink() or not goal.is_dir():
                    continue
                path = goal / "private_capabilities"
                if path.is_symlink() or not path.is_dir() or not (path / "catalog.sqlite3").is_file():
                    continue
                progress = GoalStore(goal).read()["progress"]
                deliveries = {
                    item["candidate_id"]: item
                    for item in selected_deliveries(progress.get("deliveries", []))
                    if re.fullmatch(r"[a-f0-9]{64}", item.get("candidate_id", ""))
                    and (
                        bool(item.get("scientific_acceptance"))
                        if item.get("kind") == "model"
                        else progress.get("scientific_acceptance", {}).get("passed") is True
                    )
                }
                if deliveries:
                    libraries.append(ResearchLibrary(path, goal, deliveries))
            except (OSError, ValueError, KeyError, TypeError):
                continue
    return libraries
