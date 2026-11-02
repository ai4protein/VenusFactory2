"""Host-only task/cohort views for legacy and explicitly shared model origins."""

from __future__ import annotations

import json
import re
from pathlib import Path

from agent.rsi import model_tasks
from .protein_model import digest
from .storage import safe_child


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    require(
        path.is_file() and not any(p.is_symlink() for p in [path, *path.parents]), "Missing or symlinked origin data"
    )
    value = json.loads(path.read_text())
    digest(value)
    return value


def origin_view(verified):
    """Read after library provenance verification; never expose this view to an LLM."""
    root = Path(verified["origin_goal"])
    protocol = read(root / "protocol.json")
    if protocol.get("schema") == "shared_scientist_goal.v1":
        identity = verified["origin"]["task_id"]
        require(
            isinstance(identity, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,119}", identity),
            "Invalid shared origin target",
        )
        require(
            verified["origin"]["evidence_kind"] == "native_scientist_shared_program_model",
            "Shared origin evidence kind changed",
        )
        matches = [t for t in protocol["targets"] if t["id"] == identity]
        require(len(matches) == 1, "Shared donor task is not uniquely admitted")
        binding = matches[0]
        folder = safe_child(root, "targets/" + identity)
        task = read(folder / "model_task.json")
        require(digest(task) == binding["task_sha256"] and task["id"] == identity, "Shared donor task changed")
        hashes, source = binding["data_hashes"], binding["source"]
    else:
        folder = root
        task = read(root / "model_task.json") if (root / "model_task.json").exists() else None
        hashes, source = protocol["data_hashes"], protocol.get("source")
    data = {phase: read(folder / "data" / (phase + ".json")) for phase in model_tasks.PHASES}
    require(
        all(digest(rows) == hashes[phase] for phase, rows in data.items()), "Donor cohort changed after verification"
    )
    if protocol.get("schema") == "shared_scientist_goal.v1":
        model = verified["packet"]["model"]
        require(
            model_tasks.load_task(task) == data
            and task["scope"] == model["scope"]
            and task["plugin_id"] == model["plugin_id"]
            and digest(data["training"]) == model["training_sha256"],
            "Shared donor model/task binding changed",
        )
    return {"task": task, "data": data, "data_hashes": hashes, "source": source}
