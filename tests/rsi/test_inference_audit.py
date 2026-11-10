"""Serving-audit integrity tests. Synthetic fixtures are not scientific models."""

import copy
import json
import os

import pytest

from test_scientist_model_library import storage  # noqa: F401
from agent.harness.contracts import digest
from agent.rsi.engine import inference_audit as audit, inference_evidence as evidence
from agent.rsi.engine import scientist_model_library as library
from agent.rsi.engine.storage import atomic_json


@pytest.fixture
def publication(storage):
    store, root, verified, calls = storage
    identity = store.publish(root, "b" * 64)["entry_id"]
    return store, root, verified, calls, identity


def test_unprepared_inference_still_uses_full_verification(publication):
    store, _, _, calls, identity = publication
    audit.resolve(store, identity)
    assert calls == [True, False]
    assert not (store.root / ".inference").exists()


def test_full_audit_survives_restart_without_replaying_training(publication, monkeypatch):
    store, _, verified, calls, identity = publication
    result = store.prepare_inference([identity])
    assert calls == [True, True]
    assert result["prepared_entries"] == 1 and result["training_replayed"] is True
    assert result["external_model_calls"] == result["new_autonomous_models"] == 0
    monkeypatch.setattr(library, "validate_model_goal", lambda *a, **kw: pytest.fail("Repeated acquisition replay"))
    reopened = library.ScientistModelLibrary(store.root)
    value = audit.resolve(reopened, identity)
    assert value["packet"] == verified["packet"]
    assert value["inference_audit_id"] == result["models"][0]["audit_id"]
    assert stat_mode(store.root / ".inference") == 0o700
    assert stat_mode(store.root / ".inference/signing-key") == 0o600


def stat_mode(path):
    return path.stat().st_mode & 0o777


@pytest.mark.parametrize("change", ["new_file", "deleted_file", "content_same_mtime", "symlink", "fifo", "retirement"])
def test_changed_goal_evidence_fails_before_prediction(publication, change):
    store, root, _, _, identity = publication
    path = root / "controller/state.json"
    atomic_json(path, {"status": "registered"})
    old = path.stat()
    store.prepare_inference([identity])
    if change == "new_file":
        (root / "unrecorded.txt").write_text("new evidence")
    elif change == "deleted_file":
        path.unlink()
    elif change == "content_same_mtime":
        data = path.read_bytes().replace(b"registered", b"retired___")
        path.write_bytes(data)
        os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))
        assert path.stat().st_size == old.st_size
    elif change == "symlink":
        (root / "alias").symlink_to(path)
    elif change == "fifo":
        os.mkfifo(root / "pipe")
    else:
        atomic_json(path, {"status": "retired"})
    with pytest.raises((ValueError, OSError)):
        audit.resolve(store, identity)


@pytest.mark.parametrize("part", ["entry", "index", "origin", "packet"])
def test_live_registry_bindings_are_not_cached(publication, part):
    store, _, _, _, identity = publication
    store.prepare_inference([identity])
    patterns = {"entry": "entries/*.json", "index": "index/*.json",
                "origin": "origins/*/*.json", "packet": "packets/*.json"}
    atomic_json(next(store.root.glob(patterns[part])), {"forged": True})
    with pytest.raises((ValueError, KeyError)):
        audit.resolve(store, identity)


@pytest.mark.parametrize("part", ["record", "signature", "snapshot", "key_missing", "key_permissions", "area_permissions"])
def test_model_written_or_damaged_audits_are_not_accepted(publication, part):
    store, _, _, _, identity = publication
    store.prepare_inference([identity])
    area = store.root / ".inference"
    pointer = json.loads((area / "entries" / (identity + ".json")).read_text())
    path = area / "audits" / (pointer["audit_id"] + ".json")
    envelope = json.loads(path.read_text())
    if part == "record":
        envelope["record"]["verified"]["skill"] = 999
        # A content digest is not a host signature.
        envelope["signature"] = digest(envelope["record"])
        atomic_json(path, envelope)
    elif part == "signature":
        envelope["signature"] = "0" * 64
        atomic_json(path, envelope)
    elif part == "snapshot":
        atomic_json(area / "snapshots" / (envelope["record"]["snapshot_id"] + ".json"), {})
    elif part == "key_missing":
        (area / "signing-key").unlink()
    elif part == "key_permissions":
        (area / "signing-key").chmod(0o644)
    else:
        area.chmod(0o755)
    with pytest.raises((ValueError, OSError)):
        audit.resolve(store, identity)


def test_external_task_data_is_bound_even_when_outside_goal(publication):
    store, root, _, _, identity = publication
    raw = root.parent / "dataset.json"
    atomic_json(raw, [{"value": 1}])
    atomic_json(root / "model_task.json", {"data": {"training": {"path": str(raw), "sha256": "fixture"}}})
    store.prepare_inference([identity])
    atomic_json(raw, [{"value": 2}])
    with pytest.raises(ValueError, match="changed"):
        audit.resolve(store, identity)


def test_ancestor_retirement_and_external_data_invalidate_child(publication, monkeypatch):
    store, root, _, _, identity = publication
    donor_root = root.parent / "donor"
    donor_root.mkdir()
    atomic_json(donor_root / "goal.json", {})
    atomic_json(donor_root / "protocol.json", {})
    atomic_json(donor_root / "state.json", {"status": "registered"})
    donor = library.ScientistModelLibrary(root.parent / "donor_library")
    donor_model = copy.deepcopy(store.resolve(identity))
    donor_model["origin"].update(goal_sha256=library.file_hash(donor_root / "goal.json"),
                                protocol_sha256=library.file_hash(donor_root / "protocol.json"))
    published = donor._publish_verified(donor_root, donor_model)
    # Preserve the publication anchors while adding a declared history fixture.
    declaration = {"library_experience": {"fixture": True}}
    atomic_json(root / "protocol.json", declaration)
    entry = store._object("entries", identity)
    origin_path = store.root / "origins" / digest(entry["key"]) / (entry["canonical_origin"] + ".json")
    # Recreate a fresh storage fixture publication, bound to the new protocol.
    revised = copy.deepcopy(store._object("packets", entry["packet_id"]))
    verified = {"packet": revised, "key": entry["key"], "skill": 0.5,
                "origin": {k: v for k, v in json.loads(origin_path.read_text()).items()
                           if k not in {"goal", "key_id", "packet_id"}}}
    verified["origin"]["protocol_sha256"] = library.file_hash(root / "protocol.json")
    second = library.ScientistModelLibrary(root.parent / "recipient_library")
    identity = second._publish_verified(root, verified)["entry_id"]
    monkeypatch.setattr(library, "validate_model_goal", lambda *a, **kw: copy.deepcopy(verified))
    monkeypatch.setattr(library.library_experience, "read_frozen",
                        lambda *a: {"manifest": {"selection": {"library_root": str(donor.root),
                                                               "entry_ids": [published["entry_id"]]}}})
    second.prepare_inference([identity])
    atomic_json(donor_root / "state.json", {"status": "retired"})
    with pytest.raises(ValueError, match="changed"):
        audit.resolve(second, identity)


def test_runtime_changes_require_a_new_complete_audit(publication, monkeypatch):
    store, _, _, _, identity = publication
    store.prepare_inference([identity])
    original = evidence.runtime_identity
    monkeypatch.setattr(evidence, "runtime_identity", lambda: {**original(), "python": "different runtime"})
    with pytest.raises(ValueError, match="runtime changed"):
        audit.resolve(store, identity)


def test_changes_during_audit_never_create_a_usable_record(publication, monkeypatch):
    store, root, verified, _, identity = publication

    def mutate(*args, **kwargs):
        (root / "new-during-audit").write_text("changed")
        return copy.deepcopy(verified)

    monkeypatch.setattr(library, "validate_model_goal", mutate)
    with pytest.raises(ValueError, match="during the full"):
        store.prepare_inference([identity])
    assert not (store.root / ".inference/entries" / (identity + ".json")).exists()


def test_failed_reaudit_preserves_the_prior_receipt_without_accepting_changed_data(publication, monkeypatch):
    store, root, _, _, identity = publication
    store.prepare_inference([identity])
    path = store.root / ".inference/entries" / (identity + ".json")
    before = path.read_bytes()
    (root / "changed-data").write_text("changed")
    def reject(*args, **kwargs):
        raise ValueError("Full audit rejected")
    monkeypatch.setattr(library, "validate_model_goal", reject)
    with pytest.raises(ValueError, match="Full audit rejected"):
        store.prepare_inference([identity])
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="changed"):
        audit.resolve(store, identity)


def test_explicit_scientific_verification_always_replays(publication):
    store, _, _, calls, identity = publication
    store.prepare_inference([identity])
    store.resolve(identity, replay_training=True)
    assert calls == [True, True, True]


def test_volatile_regular_locks_and_bytecode_do_not_change_frozen_evidence(publication):
    store, root, _, _, identity = publication
    store.prepare_inference([identity])
    (root / "runtime.lock").touch()
    (root / "__pycache__").mkdir()
    (root / "__pycache__/module.pyc").write_bytes(b"bytecode")
    audit.resolve(store, identity)
    (root / "__pycache__/unsafe").symlink_to(root / "goal.json")
    with pytest.raises(ValueError, match="Symlink"):
        audit.resolve(store, identity)


def test_prepared_prediction_keeps_scope_and_receipt_contract(publication, monkeypatch):
    store, _, verified, _, identity = publication
    store.prepare_inference([identity])
    class Plugin:
        def predict(self, model, rows, scope):
            return [{"id": row["id"], "sequence": row["sequence"], "prediction": 0.25} for row in rows]
    monkeypatch.setattr(library, "get_model_plugin", lambda _: Plugin())
    monkeypatch.setattr(library, "validate_model_goal", lambda *a, **kw: pytest.fail("Unexpected replay"))
    result = store.predict(identity, [{"id": "x", "sequence": "AA", "target": 99}],
                           verified["key"]["scope"], session_id="test-agent")
    assert result["receipt"]["inference_audit_id"]
    assert result["receipt"]["session_id"] == "test-agent"
    assert result["predictions"][0]["prediction"] == 0.25
    with pytest.raises(ValueError, match="applicability"):
        store.predict(identity, [], {"wrong": True}, session_id="test-agent")


def test_copied_audit_cannot_move_to_another_owner_library(publication):
    import shutil
    store, _, _, _, identity = publication
    store.prepare_inference([identity])
    other = store.root.with_name("other-library")
    shutil.copytree(store.root, other)
    with pytest.raises(ValueError, match="identity changed"):
        audit.resolve(library.ScientistModelLibrary(other), identity)


def test_verification_source_content_is_part_of_runtime_identity(tmp_path, monkeypatch):
    base = tmp_path / "agent"
    for folder in ("rsi/engine", "harness"):
        (base / folder).mkdir(parents=True)
    source = base / "harness/validator.py"
    source.write_text("valid = True\n")
    monkeypatch.setattr(evidence, "__file__", str(base / "rsi/engine/inference_evidence.py"))
    before = evidence.runtime_identity()
    source.write_text("valid = False\n")
    assert evidence.runtime_identity() != before


@pytest.mark.parametrize("budget", ["files", "bytes", "record"])
def test_resource_limits_fail_before_publishing_a_record(publication, monkeypatch, budget):
    store, _, _, _, identity = publication
    module, name = ((audit, "MAX_RECORD_BYTES") if budget == "record" else
                    (evidence, "MAX_FILES" if budget == "files" else "MAX_BYTES"))
    monkeypatch.setattr(module, name, 1)
    with pytest.raises(ValueError, match="budget"):
        store.prepare_inference([identity])
    assert not (store.root / ".inference/entries" / (identity + ".json")).exists()


def test_parallel_preparation_uses_one_host_key_and_complete_records(publication):
    from concurrent.futures import ThreadPoolExecutor
    store, _, _, _, identity = publication
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: store.prepare_inference([identity]), range(2)))
    assert results[0] == results[1]
    assert audit.resolve(store, identity)["inference_audit_id"] == results[0]["models"][0]["audit_id"]

def test_parallel_hashing_is_bounded_and_preserves_content(tmp_path, monkeypatch):
    import hashlib
    import threading
    import time
    root = tmp_path / "tree"
    root.mkdir()
    expected = {}
    for index in range(300):
        content = ("file-" + str(index)).encode()
        path = root / str(index)
        path.write_bytes(content)
        expected[path.name] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    original = evidence.file_digest
    lock = threading.Lock()
    current = peak = 0

    def observe(path, *, before=None):
        nonlocal current, peak
        with lock:
            current += 1
            peak = max(peak, current)
        try:
            time.sleep(0.005)
            return original(path, before=before)
        finally:
            with lock:
                current -= 1
    monkeypatch.setattr(evidence, "file_digest", observe)
    assert evidence.tree_digest(root) == {"directories": [], "files": expected}
    assert 1 < peak <= 4


def test_directory_stat_cannot_hide_replacement_before_hashing(tmp_path, monkeypatch):
    path = tmp_path / "model.json"
    path.write_text("original")
    original = evidence.file_digest

    def replace(file_path, *, before=None):
        file_path.unlink()
        file_path.write_text("replacement")
        return original(file_path, before=before)
    monkeypatch.setattr(evidence, "file_digest", replace)
    with pytest.raises(ValueError, match="changed"):
        evidence.tree_digest(tmp_path)


def test_reaudit_preserves_signed_history_referenced_by_old_usage(publication):
    store, root, _, _, identity = publication
    first = store.prepare_inference([identity])["models"][0]
    first_path = store.root / ".inference/audits" / (first["audit_id"] + ".json")
    original = first_path.read_bytes()
    # The fixture host verifier accepts this new non-model evidence file.
    (root / "additional-host-evidence.json").write_text("{}")
    second = store.prepare_inference([identity])["models"][0]
    assert second["audit_id"] != first["audit_id"]
    assert first_path.read_bytes() == original
    assert audit.resolve(store, identity)["inference_audit_id"] == second["audit_id"]


def test_reused_read_buffer_does_not_leak_between_file_sizes(tmp_path):
    import hashlib
    for index, content in enumerate([b"x" * (1024 * 1024 + 19), b"short", b"", b"y" * 125]):
        path = tmp_path / str(index)
        path.write_bytes(content)
        assert evidence.file_digest(path) == {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def replace_signed_fixture_record(store, result, change):
    area = store.root / ".inference"
    previous = result["models"][0]["audit_id"]
    record = json.loads((area / "audits" / (previous + ".json")).read_text())["record"]
    change(record)
    identity = digest(record)
    atomic_json(area / "audits" / (identity + ".json"),
                {"record": record, "signature": audit._signature(audit._key(area), record)})
    atomic_json(area / "entries" / (record["entry_id"] + ".json"), {"audit_id": identity})
    return identity


def test_large_learned_state_is_served_without_duplicating_the_model_packet(storage, monkeypatch):
    store, root, verified, calls = storage
    verified["packet"]["model"]["model_state"]["weights"] = [0.25] * 200_000
    identity = store.publish(root, "b" * 64)["entry_id"]
    monkeypatch.setattr(audit, "MAX_RECORD_BYTES", 512 * 1024)
    packet = next((store.root / "packets").glob("*.json"))
    assert packet.stat().st_size > audit.MAX_RECORD_BYTES
    result = store.prepare_inference([identity])
    path = store.root / ".inference/audits" / (result["models"][0]["audit_id"] + ".json")
    assert path.stat().st_size < audit.MAX_RECORD_BYTES
    resolved = audit.resolve(library.ScientistModelLibrary(store.root), identity)
    assert resolved["packet"] == verified["packet"] and resolved["skill"] == verified["skill"]
    assert calls == [True, True]


def test_prepared_reference_preserves_every_verified_field(publication):
    store, _, _, _, identity = publication
    before = store.resolve(identity)
    store.prepare_inference([identity])
    after = audit.resolve(store, identity)
    assert {key: value for key, value in after.items() if key != "inference_audit_id"} == before


@pytest.mark.parametrize("change", ["foreign_reference", "embedded_packet"])
def test_signed_packet_reference_cannot_override_current_publication(publication, change):
    store, _, verified, _, identity = publication
    result = store.prepare_inference([identity])

    def modify(record):
        if change == "foreign_reference":
            record["packet_id"] = "0" * 64
        else:
            record["verified"]["packet"] = verified["packet"]

    replace_signed_fixture_record(store, result, modify)
    with pytest.raises(ValueError, match="audit packet changed"):
        audit.resolve(store, identity)


def test_legacy_inline_audit_keeps_its_signature_and_serving_contract(publication):
    store, _, verified, calls, identity = publication
    prepared = store.prepare_inference([identity])

    def inline(record):
        record["schema"] = audit.LEGACY_SCHEMA
        record.pop("packet_id")
        record["verified"]["packet"] = copy.deepcopy(verified["packet"])

    previous = replace_signed_fixture_record(store, prepared, inline)
    value = audit.resolve(library.ScientistModelLibrary(store.root), identity)
    assert value["packet"] == verified["packet"] and value["inference_audit_id"] == previous
    assert calls == [True, True]
    store.prepare_inference([identity])
    assert (store.root / ".inference/audits" / (previous + ".json")).is_file()


def test_large_verification_metadata_still_obeys_record_budget(publication, monkeypatch):
    store, _, verified, _, identity = publication
    verified["validation_trace"] = "x" * (1024 * 1024)
    monkeypatch.setattr(audit, "MAX_RECORD_BYTES", 512 * 1024)
    with pytest.raises(ValueError, match="record budget"):
        store.prepare_inference([identity])
    assert not (store.root / ".inference/entries" / (identity + ".json")).exists()


@pytest.mark.parametrize("limit", ["files", "bytes"])
def test_late_scan_budget_failure_cannot_return_a_partial_snapshot(tmp_path, monkeypatch, limit):
    # Enough files to cross the bounded queue, then fail inside a later read.
    for index in range(300):
        (tmp_path / f"{index:04d}.json").write_bytes(b"{}")
    monkeypatch.setattr(evidence, "MAX_FILES" if limit == "files" else "MAX_BYTES", 257)
    with pytest.raises(ValueError, match="scan budget"):
        evidence.tree_digest(tmp_path)


def test_mixed_directory_scan_preserves_exact_content_and_filters(tmp_path):
    import hashlib
    expected = {}
    for folder in ("a", "b", "b/nested"):
        directory = tmp_path / folder
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(101):
            content = f"{folder} / {index}\n".encode()
            name = f"{folder}/{index:03d}.py"
            (tmp_path / name).write_bytes(content)
            expected[name] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    (tmp_path / "a/readme.txt").write_bytes(b"notes")
    (tmp_path / "b/temporary.lock").write_bytes(b"lock")
    (tmp_path / "b/__pycache__").mkdir()
    (tmp_path / "b/__pycache__/cached.py").write_bytes(b"cached")
    assert evidence.tree_digest(tmp_path, python_only=True) == {"directories": [], "files": expected}
    expected["a/readme.txt"] = {"bytes": 5, "sha256": hashlib.sha256(b"notes").hexdigest()}
    assert evidence.tree_digest(tmp_path) == {"directories": ["a", "b", "b/nested"], "files": expected}
