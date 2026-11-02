"""Content-addressed immutable capability packages and atomic incumbent updates."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from .storage import atomic_json, file_lock, safe_child, sha256_file


class VersionStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        (self.root / "versions").mkdir(parents=True, exist_ok=True)

    def snapshot(self, source: Path, *, parent: str | None = None, metadata: dict | None = None) -> dict:
        entries = {}
        for path in sorted(source.rglob("*")):
            if path.is_symlink():
                raise ValueError("version packages cannot contain symlinks")
            if path.is_file():
                rel = str(path.relative_to(source))
                if rel == "manifest.json":
                    raise ValueError("manifest.json is reserved")
                if any(part.startswith(".") or part == "__pycache__" for part in path.relative_to(source).parts):
                    raise ValueError("hidden files and caches cannot enter version packages")
                entries[rel] = sha256_file(path)
        if not entries:
            raise ValueError("empty version package")
        payload = {"parent": parent, "files": entries, "metadata": metadata or {}}
        vid = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        manifest = {"version_id": vid, **payload}
        target = self.root / "versions" / vid
        with file_lock(self.root / ".versions.lock"):
            if not target.exists():
                staging = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=target.parent))
                try:
                    for name, digest in entries.items():
                        dest = safe_child(staging, name)
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(source / name, dest)
                        if sha256_file(dest) != digest:
                            raise ValueError("source changed while snapshotting")
                        dest.chmod(0o444)
                    atomic_json(staging / "manifest.json", manifest)
                    (staging / "manifest.json").chmod(0o444)
                    for directory in sorted(
                        [p for p in staging.rglob("*") if p.is_dir()], key=lambda x: len(x.parts), reverse=True
                    ):
                        directory.chmod(0o555)
                    staging.chmod(0o555)
                    os.replace(staging, target)
                except BaseException:
                    for item in staging.rglob("*"):
                        if item.is_dir():
                            item.chmod(0o700)
                    staging.chmod(0o700)
                    shutil.rmtree(staging)
                    raise
        self.verify(vid)
        return manifest

    def verify(self, version_id: str) -> dict:
        if len(version_id) != 64 or any(c not in "0123456789abcdef" for c in version_id):
            raise ValueError("invalid version id")
        folder = self.root / "versions" / version_id
        manifest = json.loads((folder / "manifest.json").read_text())
        payload = {k: manifest[k] for k in ("parent", "files", "metadata")}
        actual = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if actual != version_id:
            raise ValueError("manifest content hash mismatch")
        for name, digest in manifest["files"].items():
            if sha256_file(safe_child(folder, name)) != digest:
                raise ValueError(f"version file changed: {name}")
        expected = set(manifest["files"]) | {"manifest.json"}
        actual_files = {str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file() or p.is_symlink()}
        if actual_files != expected:
            raise ValueError("unexpected files in version")
        return manifest

    def accept(self, version_id: str, *, expected_parent: str | None, evidence: dict) -> None:
        manifest = self.verify(version_id)
        if manifest["parent"] != expected_parent:
            raise ValueError("candidate parent mismatch")
        if evidence.get("passed") is not True or not evidence.get("evaluation_id"):
            raise ValueError("external acceptance evidence is required")
        with file_lock(self.root / ".incumbent.lock"):
            ptr = self.root / "incumbent.json"
            old = json.loads(ptr.read_text())["version_id"] if ptr.exists() else None
            if old != expected_parent:
                raise ValueError("incumbent changed; re-evaluate against current version")
            atomic_json(ptr, {"version_id": version_id, "previous": old, "evidence": evidence})
