"""Owner-scoped immutable references to independently verified paper recipes."""

from __future__ import annotations

import json
from pathlib import Path

from agent.harness.contracts import digest, file_hash
from agent.rsi.engine.storage import atomic_json, file_lock, safe_child


class PaperCapabilityLibrary:
    def __init__(self, root):
        requested = Path(root).absolute()
        if any(p.is_symlink() for p in [requested, *requested.parents]):
            raise ValueError("Capability library cannot traverse symlinks")
        self.root = requested.resolve()
        self.owner = self.root.parent

    def goal(self, folder):
        folder = Path(folder).absolute()
        try:
            name = str(folder.relative_to(self.owner))
        except ValueError as exc:
            raise ValueError("Capability goal is outside the owner namespace") from exc
        if name == "." or folder == self.root or self.root in folder.parents:
            raise ValueError("A dedicated goal in the owner namespace is required")
        return safe_child(self.owner, name)

    def publish(self, folder, case, validate):
        folder = self.goal(folder)
        verified = validate(folder, case)
        recipe = verified["recipe"]
        if "library_origin" in recipe:
            entry_id = recipe["library_origin"]["entry_id"]
            self.resolve(entry_id, recipe["paper"], validate)
            return entry_id
        payload = {
            "schema": 1,
            "kind": "verified_paper_capability",
            "origin_goal": str(folder.relative_to(self.owner)),
            "case": case,
            "paper": recipe["paper"],
            "recipe_version": verified["recipe_version"],
            "recipe_sha256": digest(recipe),
            "goal_sha256": file_hash(folder / "goal.json"),
            "protocol_sha256": file_hash(folder / "protocol.json"),
        }
        identity = digest(payload)
        path = safe_child(self.root, "entries/" + identity + ".json")
        with file_lock(self.root / ".library.lock"):
            if path.exists():
                if json.loads(path.read_text()) != payload:
                    raise ValueError("Capability entry was changed")
            else:
                atomic_json(path, payload)
                path.chmod(0o444)
        return identity

    def resolve(self, identity, paper, validate):
        if not isinstance(identity, str) or len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
            raise ValueError("Invalid capability identity")
        path = safe_child(self.root, "entries/" + identity + ".json")
        value = json.loads(path.read_text())
        if digest(value) != identity or value.get("schema") != 1 or value.get("kind") != "verified_paper_capability":
            raise ValueError("Capability entry identity changed")
        if value["paper"] != paper or value["case"] != paper["case"]:
            raise ValueError("Capability scope is incompatible")
        folder = self.goal(safe_child(self.owner, value["origin_goal"]))
        if (
            file_hash(folder / "goal.json") != value["goal_sha256"]
            or file_hash(folder / "protocol.json") != value["protocol_sha256"]
        ):
            raise ValueError("Capability origin changed")
        verified = validate(folder, value["case"])
        if (
            verified["recipe_version"] != value["recipe_version"]
            or digest(verified["recipe"]) != value["recipe_sha256"]
            or "library_origin" in verified["recipe"]
        ):
            raise ValueError("Capability source recipe changed")
        return {**verified, "origin_goal": folder, "entry_id": identity}

    def find(self, paper, validate):
        # Explicit scope match; neither weaker allocations nor another user's
        # library is searched. Changed matching evidence fails closed.
        for path in sorted((self.root / "entries").glob("*.json")):
            value = json.loads(safe_child(self.root, "entries/" + path.name).read_text())
            if digest(value) != path.stem:
                raise ValueError("Capability entry identity changed")
            if value.get("paper") == paper:
                return self.resolve(path.stem, paper, validate)
        return None
