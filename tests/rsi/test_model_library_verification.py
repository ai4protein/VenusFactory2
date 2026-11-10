"""Storage/CLI regression fixtures are never scientific model evidence."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from test_scientist_model_library import storage

from agent.rsi import scientist_model_library as library_module
from agent.rsi.engine.storage import atomic_json


def published(storage):
    store, root, verified, calls = storage
    identity = store.publish(root, verified["packet"]["model_id"])["entry_id"]
    calls.clear()
    return store, root, identity, calls


def test_duplicate_entries_replay_once_without_writing_library(storage):
    store, root, identity, calls = published(storage)
    before = {str(p.relative_to(store.root)): p.read_bytes() for p in store.root.rglob("*") if p.is_file()}
    result = store.resolve_many([identity, identity], replay_training=True)
    assert list(result) == [identity] and calls == [True]
    assert result[identity]["entry_id"] == identity
    after = {str(p.relative_to(store.root)): p.read_bytes() for p in store.root.rglob("*") if p.is_file()}
    assert before == after
    assert not (store.root / "usage").exists()


@pytest.mark.parametrize("ids", [[], ["a" * 64] * 257, "a" * 64, ["invalid"], [None], ("a" * 64,)])
def test_invalid_batch_is_rejected_before_validation(storage, ids):
    store, _, _, calls = published(storage)
    with pytest.raises(ValueError):
        store.resolve_many(ids, replay_training=True)
    assert not calls


def test_ambiguous_replay_option_is_rejected(storage):
    store, _, identity, calls = published(storage)
    with pytest.raises(ValueError, match="boolean"):
        store.resolve_many([identity], replay_training="false")
    assert not calls


@pytest.mark.parametrize("batch", [False, True])
def test_index_change_during_validation_is_rejected(storage, monkeypatch, batch):
    store, _, identity, _ = published(storage)
    original = library_module.validate_model_goal
    index = next((store.root / "index").glob("*.json"))

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        atomic_json(index, {"entry_id": "a" * 64})
        return result

    monkeypatch.setattr(library_module, "validate_model_goal", changed)
    with pytest.raises(ValueError, match="Model index changed"):
        store.resolve_many([identity]) if batch else store.resolve(identity)


def cli():
    path = Path(__file__).resolve().parents[2] / "scripts/rsi/scientist_model_library.py"
    spec = importlib.util.spec_from_file_location("scientist_library_cli_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verification_command_writes_only_after_strict_success(storage, tmp_path, monkeypatch):
    store, _, identity, calls = published(storage)
    module = cli()
    monkeypatch.setattr(module, "ScientistModelLibrary", lambda path: store)
    inputs = tmp_path / "entries.json"
    output = tmp_path / "verified.json"
    inputs.write_text(json.dumps([identity, identity]))
    monkeypatch.setattr(sys, "argv", ["library", "--library", str(store.root), "verify",
                                    "--entry-ids", str(inputs), "--output", str(output)])
    module.main()
    report = json.loads(output.read_text())
    assert calls == [True] and report["verified_entries"] == 1
    assert report["training_replayed"] is True and report["new_autonomous_models"] == 0
    assert report["external_model_calls"] == 0 and report["final_labels_scored"] is False
    original = output.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        module.main()
    assert output.read_bytes() == original and calls == [True]


def test_failed_verification_command_does_not_create_success_report(storage, tmp_path, monkeypatch):
    store, _, _, _ = published(storage)
    module = cli()
    monkeypatch.setattr(module, "ScientistModelLibrary", lambda path: store)
    inputs = tmp_path / "entries.json"
    output = tmp_path / "verified.json"
    inputs.write_text(json.dumps(["a" * 64]))
    monkeypatch.setattr(sys, "argv", ["library", "--library", str(store.root), "verify",
                                    "--entry-ids", str(inputs), "--output", str(output)])
    with pytest.raises(ValueError):
        module.main()
    assert not output.exists()


def test_overlapping_audits_keep_first_completed_report(storage, tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    store, _, identity, _ = published(storage)
    module = cli()
    output = tmp_path / "verified.json"
    inputs = tmp_path / "entries.json"
    inputs.write_text(json.dumps([identity]))
    result = store.resolve_many([identity], replay_training=True)
    gate = Barrier(2)

    class ConcurrentLibrary:
        def resolve_many(self, ids, *, replay_training):
            assert ids == [identity] and replay_training is True
            gate.wait(timeout=10)
            return result

    monkeypatch.setattr(module, "ScientistModelLibrary", lambda path: ConcurrentLibrary())
    monkeypatch.setattr(sys, "argv", ["library", "--library", str(store.root), "verify",
                                    "--entry-ids", str(inputs), "--output", str(output)])

    def run():
        try:
            module.main()
            return "written"
        except ValueError as exc:
            assert "already exists" in str(exc)
            return "preserved"

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run) for _ in range(2)]
        assert sorted(f.result(timeout=20) for f in futures) == ["preserved", "written"]
    assert json.loads(output.read_text())["verified_entries"] == 1


def test_discovery_uses_canonical_origin_without_scanning_retrained_variants(storage, monkeypatch):
    store, root, verified, calls = storage
    first = store.publish(root, verified["packet"]["model_id"])
    verified["packet"]["model"]["recipe"]["seed"] += 1
    verified["packet"]["model"]["model_state"] = {"variant": True}
    store.publish(root, verified["packet"]["model_id"])
    assert store.catalog()["entries"][0]["recorded_origins"] == 2
    reads = []
    original = store._object

    def read(area, identity):
        reads.append((area, identity))
        return original(area, identity)

    monkeypatch.setattr(store, "_object", read)
    calls.clear()
    metadata = store.discovery_metadata()
    assert metadata["indexed_entries"] == 1 and not calls
    assert metadata["models"][0]["entry_id"] == first["entry_id"]
    assert sum(area.startswith("origins/") for area, _ in reads) == 1
    for private in ("source_code", "model_state", "goal_sha256", str(root)):
        assert private not in json.dumps(metadata)


@pytest.mark.parametrize("area", ["entries", "origin", "packets", "index"])
def test_discovery_rejects_modified_registry_objects(storage, area):
    store, _, identity, _ = published(storage)
    entry = store._object("entries", identity)
    index = next((store.root / "index").glob("*.json"))
    key = index.stem
    paths = {
        "entries": store.root / "entries" / f"{identity}.json",
        "origin": store.root / "origins" / key / f"{entry['canonical_origin']}.json",
        "packets": store.root / "packets" / f"{entry['packet_id']}.json",
        "index": index,
    }
    atomic_json(paths[area], {"entry_id": "a" * 64} if area == "index" else {"modified": True})
    with pytest.raises(ValueError):
        store.discovery_metadata()


@pytest.mark.parametrize("area", ["entries", "origin", "packets", "index"])
def test_persistent_discovery_never_masks_registry_tampering(storage, area):
    from mcp_tools.models.evolved import catalog
    store, _, identity, _ = published(storage)
    entry = store._object("entries", identity)
    index = next((store.root / "index").glob("*.json"))
    paths = {
        "entries": store.root / "entries" / f"{identity}.json",
        "origin": store.root / "origins" / index.stem / f"{entry['canonical_origin']}.json",
        "packets": store.root / "packets" / f"{entry['packet_id']}.json",
        "index": index,
    }
    catalog.recorded_metadata(store)
    catalog._CACHE.clear()
    atomic_json(paths[area], {"entry_id": "a" * 64} if area == "index" else {"modified": True})
    with pytest.raises(ValueError):
        catalog.recorded_metadata(store)


def test_same_size_packet_edit_with_restored_mtime_is_rejected_after_cache(storage):
    import os
    from mcp_tools.models.evolved import catalog
    store, _, identity, _ = published(storage)
    entry = store._object("entries", identity)
    packet = store.root / "packets" / (entry["packet_id"] + ".json")
    catalog.recorded_metadata(store)
    info = packet.stat()
    original = packet.read_text()
    changed = original.replace('"model_id"', '"other_id"', 1)
    assert len(changed) == len(original) and changed != original
    packet.chmod(0o600)
    packet.write_text(changed)
    packet.chmod(info.st_mode & 0o777)
    os.utime(packet, ns=(info.st_atime_ns, info.st_mtime_ns))
    with pytest.raises(ValueError, match="identity"):
        catalog.recorded_metadata(store)
