import json

import pytest

from agent.rsi.engine.storage import atomic_json, safe_child
from agent.rsi.engine.versions import VersionStore


def test_reject_and_stale_accept_leave_incumbent_unchanged(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "executor_prompt.md").write_text("first")
    store = VersionStore(tmp_path / "store")
    first = store.snapshot(source)
    store.accept(first["version_id"], expected_parent=None, evidence={"passed": True, "evaluation_id": "smoke"})
    (source / "executor_prompt.md").write_text("candidate")
    second = store.snapshot(source, parent=first["version_id"])
    pointer = store.root / "incumbent.json"
    before = pointer.read_bytes()
    with pytest.raises(ValueError):
        store.accept(second["version_id"], expected_parent=first["version_id"], evidence={"passed": False})
    assert pointer.read_bytes() == before
    store.accept(
        second["version_id"], expected_parent=first["version_id"], evidence={"passed": True, "evaluation_id": "next"}
    )
    with pytest.raises(ValueError):
        store.accept(
            second["version_id"],
            expected_parent=first["version_id"],
            evidence={"passed": True, "evaluation_id": "stale"},
        )
    assert json.loads(pointer.read_text())["version_id"] == second["version_id"]


def test_snapshot_detects_tampering_and_symlinks(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "prompt.md").write_text("v1")
    store = VersionStore(tmp_path / "store")
    version = store.snapshot(source)
    path = store.root / "versions" / version["version_id"] / "prompt.md"
    path.chmod(0o644)
    path.write_text("corrupt")
    with pytest.raises(ValueError, match="changed"):
        store.verify(version["version_id"])
    (source / "link").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="symlink"):
        store.snapshot(source)


@pytest.mark.parametrize("name", ["../answer.json", "/tmp/answer.json", ""])
def test_artifact_escape_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        safe_child(tmp_path, name)


def test_atomic_json_does_not_replace_valid_file_with_nan(tmp_path):
    path = tmp_path / "value.json"
    atomic_json(path, {"ok": 1})
    with pytest.raises(ValueError):
        atomic_json(path, {"bad": float("nan")})
    assert json.loads(path.read_text()) == {"ok": 1}
