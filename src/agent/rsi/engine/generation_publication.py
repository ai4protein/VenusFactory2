"""Publish verified generation checkpoints without completing their parent goal.

Snapshots preserve the actual checkpoint, including unmet goal criteria. Native
execution and numerical validation remain the model library's responsibility.
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from agent.harness.contracts import digest, file_hash

from .storage import atomic_json, file_lock, safe_child

MANIFEST = "generation_milestone.json"


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    require(path.is_file() and not path.is_symlink(), "Missing or symlinked generation evidence")
    return json.loads(path.read_text())


def files(root):
    result = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "Symlinked generation evidence")
        if path.is_dir():
            continue
        require(path.is_file(), "Nonregular generation evidence")
        relative = path.relative_to(root)
        if path.name.endswith(".lock") or "__pycache__" in relative.parts or relative.as_posix() == MANIFEST:
            continue
        result[relative.as_posix()] = file_hash(path)
    return result


def records_at_checkpoint(state, protocol):
    """Return an entire completed prefix, never discard an unfinished generation."""
    records = state.get("progress", {}).get("generations", [])
    if "evolution" not in protocol or not records or not all(r.get("reuse_receipt") for r in records):
        return []
    require(state["inflight"] is None, "Generation checkpoint is still in flight")
    require(
        state["status"] in {"active", "completed", "paused", "budget_exhausted", "interrupted", "waiting_for_input"},
        "Unsupported generation checkpoint status",
    )
    target = protocol["evolution"]["generations"]
    require(type(target) is int and 1 <= len(records) <= target, "Invalid generation prefix length")
    cycles = state["cycles"]
    require(bool(cycles), "Generation checkpoint lacks episodes")
    final = cycles[-1]
    if final.get("status") != "completed" or final.get("phase") != "reuse":
        return []
    require(final.get("progress") == state["progress"], "Checkpoint progress differs from its reuse episode")
    require(
        records[-1]["reuse_receipt"] == f"episodes/cycle_{len(cycles) - 1:03d}/reuse_receipt.json",
        "Checkpoint reuse is not the last saved episode",
    )
    return records


def checkpoint_key(folder, state, protocol):
    return digest(
        {
            "parent_goal": str(folder),
            "definition": state["definition_sha256"],
            "protocol": digest(protocol),
            "cycles": state["cycles"],
        }
    )


def verify_snapshot(root):
    root = Path(root)
    require(not any(p.is_symlink() for p in (root, *root.parents)), "Symlinked generation snapshot")
    manifest = read(root / MANIFEST)
    require(
        manifest["schema"] == 1
        and manifest["kind"] == "scientist_generation_checkpoint"
        and root.name == digest(manifest),
        "Generation snapshot identity changed",
    )
    require(files(root) == manifest["files"], "Generation snapshot files changed")
    state, protocol = read(root / "goal.json"), read(root / "protocol.json")
    records = records_at_checkpoint(state, protocol)
    require(bool(records), "Generation snapshot contains no completed prefix")
    require(
        manifest["checkpoint_key"] == checkpoint_key(Path(manifest["parent_goal"]), state, protocol)
        and manifest["parent_status"] == state["status"]
        and manifest["requested_generations"] == protocol["evolution"]["generations"]
        and manifest["model_ids"] == [r["model_id"] for r in records],
        "Generation checkpoint binding changed",
    )
    return manifest


def bind_parent(snapshot, parent):
    """Verify a frozen prefix against the actual later parent, not just its path."""
    manifest = verify_snapshot(snapshot)
    require(manifest["parent_goal"] == str(parent), "Generation checkpoint belongs to another goal")
    frozen, current = read(snapshot / "goal.json"), read(parent / "goal.json")
    n = len(frozen["cycles"])
    require(
        frozen["definition"] == current["definition"]
        and frozen["definition_sha256"] == current["definition_sha256"]
        and frozen["cycles"] == current["cycles"][:n]
        and file_hash(snapshot / "protocol.json") == file_hash(parent / "protocol.json"),
        "Parent no longer contains the published generation prefix",
    )
    return manifest


def freeze(library, folder, state):
    folder = library.goal(folder)
    require(read(folder / "goal.json") == state, "Goal changed before generation snapshot")
    require(not (folder / MANIFEST).exists(), "Cannot checkpoint another snapshot")
    protocol = read(folder / "protocol.json")
    records = records_at_checkpoint(state, protocol)
    require(bool(records), "No completed generation prefix")
    before = files(folder)
    manifest = {
        "schema": 1,
        "kind": "scientist_generation_checkpoint",
        "parent_goal": str(folder),
        "parent_status": state["status"],
        "requested_generations": protocol["evolution"]["generations"],
        "model_ids": [r["model_id"] for r in records],
        "checkpoint_key": checkpoint_key(folder, state, protocol),
        "files": before,
    }
    base = library.goal(library.owner / (library.root.name + "_milestones"))
    require(
        base != folder and base not in folder.parents and folder not in base.parents,
        "Milestone storage must be outside its parent goal",
    )
    base.mkdir(parents=True, exist_ok=True)
    destination = safe_child(base, digest(manifest))
    with file_lock(base / ".snapshot.lock"):
        if destination.exists():
            verify_snapshot(destination)
            return destination
        temporary = base / ("pending_" + uuid.uuid4().hex)
        temporary.mkdir()
        # A failed copy is retained for diagnosis, never published or reused.
        for name in before:
            target = safe_child(temporary, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(safe_child(folder, name), target)
        require(
            files(temporary) == before and files(folder) == before, "Goal evidence changed while freezing generation"
        )
        atomic_json(temporary / MANIFEST, manifest)
        temporary.rename(destination)
        for path in destination.rglob("*"):
            if path.is_file():
                path.chmod(0o444)
    verify_snapshot(destination)
    return destination


def publish_checkpoint(library, folder, state):
    """Called under the goal runner lease after its checkpoint is durable."""
    folder = library.goal(folder)
    protocol = read(folder / "protocol.json")
    records = records_at_checkpoint(state, protocol)
    if not records:
        return None
    identity = checkpoint_key(folder, state, protocol)
    pointer = safe_child(library.root, "generation_index/" + identity + ".json")
    if pointer.exists():
        report_id = read(pointer)["report_id"]
        report = library._object("publication_attempts", report_id)
        snapshot = library.goal(safe_child(library.owner, report["origin_goal"]))
        manifest = bind_parent(snapshot, folder)
        require(
            report["goal_sha256"] == file_hash(snapshot / "goal.json")
            and report["parent_goal"] == str(folder.relative_to(library.owner)),
            "Indexed generation report changed",
        )
        require(
            manifest["checkpoint_key"] == identity and report["all_published"], "Generation publication index changed"
        )
        # Resolve again after restart; an indexed receipt is not sufficient proof.
        for outcome in report["outcomes"]:
            library.resolve(outcome["entry_id"])
        return {**report, "report_id": report_id}
    snapshot = freeze(library, folder, state)
    outcomes = []
    for record in records:
        mid = record["model_id"]
        try:
            outcomes.append({"model_id": mid, "status": "published", **library.publish(snapshot, mid)})
        except (ValueError, KeyError, OSError) as exc:
            outcomes.append(
                {"model_id": mid, "status": "rejected", "error_type": type(exc).__name__, "reason": str(exc)}
            )
    report = {
        "schema": 1,
        "kind": "model_generation_publication_attempt",
        "attempt_id": uuid.uuid4().hex,
        "origin_goal": str(snapshot.relative_to(library.owner)),
        "parent_goal": str(folder.relative_to(library.owner)),
        "goal_sha256": file_hash(snapshot / "goal.json"),
        "outcomes": outcomes,
        "all_published": all(o["status"] == "published" for o in outcomes),
        "new_entries": sum(o.get("created", False) for o in outcomes),
        "duplicate_entries": sum(o["status"] == "published" and not o["created"] for o in outcomes),
        "rejected_entries": sum(o["status"] == "rejected" for o in outcomes),
        "new_autonomous_executions": 0,
    }
    with file_lock(library.root / ".library.lock"):
        report_id = library._immutable("publication_attempts", report)
        if report["all_published"]:
            atomic_json(pointer, {"report_id": report_id})
    return {**report, "report_id": report_id}


def publisher(library, folder, record=None):
    def checkpoint(state):
        report = publish_checkpoint(library, folder, state)
        if report is not None:
            if record is not None:
                record(report)
            if not report["all_published"]:
                raise ValueError("Generation publication failed; retained report " + report["report_id"])

    return checkpoint
