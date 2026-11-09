"""Synthetic numerical migration fixtures; real historical goals are audited separately."""

import copy
import hashlib

import pytest
from test_scientist_model_library import numerical as current_numerical  # noqa: F401

from agent.harness.contracts import digest, file_hash
from agent.rsi import model_runtime_migration as migration
from agent.rsi import scientist_model_library as library
from agent.rsi.engine.model_lifecycle import ModelLifecycle
from agent.rsi.engine.storage import atomic_json
from agent.rsi.engine.versions import VersionStore


@pytest.fixture
def legacy(request, monkeypatch):
    root, current_record, protocol, data, episodes, current = request.getfixturevalue("current_numerical")
    model = current._model(current_record["model_id"])
    report = library._read(current.root / ("assessment_" + current_record["model_id"] + ".json"))
    sources, implementation = [], {}
    for name in migration.LEGACY_FILES:
        relative = "src/agent/rsi/" + name
        path = root / "implementation" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Synthetic archived interface fixture: " + name + "\n")
        sources.append(path.read_bytes())
        implementation[relative] = file_hash(path)
    policy = {**current.policy, "plugin_sha256": hashlib.sha256(b"".join(sources)).hexdigest()}
    controller = current.root.parent / "legacy_controller"
    controller.mkdir()
    staging = controller / "submission"
    atomic_json(staging / "model.json", model)
    metadata = {"session_id": "development", "mode": "science_expert", "policy_sha256": digest(policy)}
    mid = VersionStore(controller / "artifacts").snapshot(staging, metadata=metadata)["version_id"]
    report = {**report, "model_id": mid, "policy_sha256": digest(policy), "plugin_sha256": policy["plugin_sha256"]}
    del report["numerical_evidence"]
    receipt = library._read(root / current_record["reuse_receipt"])
    receipt["model_id"] = mid
    atomic_json(root / current_record["reuse_receipt"], receipt)
    atomic_json(controller / "policy.json", policy)
    atomic_json(controller / f"assessment_{mid}.json", report)
    atomic_json(
        controller / "state.json",
        {
            "schema": 1,
            "candidates": {
                mid: {
                    "status": "registered",
                    "session_id": "development",
                    "mode": "science_expert",
                    "assessment_sha256": digest(report),
                }
            },
            "published": [mid],
            "events": [receipt],
        },
    )
    record = {**current_record, "model_id": mid, "controller": str(controller.relative_to(root))}
    protocol = {**protocol, "implementation": implementation}
    atomic_json(
        root / "goal.json",
        {
            "status": "completed",
            "inflight": None,
            "progress": {"accepted_model": record, "reuse_receipt": record["reuse_receipt"]},
        },
    )
    atomic_json(root / "protocol.json", protocol)
    for phase, rows in data.items():
        atomic_json(root / "data" / (phase + ".json"), rows)

    def validate(folder, model_id, *, replay_training=True, runtime_migration=False):
        assert folder == root and model_id == mid
        result = library._model(root, record, protocol, data, episodes, 0.0, replay_training, runtime_migration)
        return {
            **result,
            "origin": {
                "goal_sha256": file_hash(root / "goal.json"),
                "protocol_sha256": file_hash(root / "protocol.json"),
                "model_id": mid,
                "stage": "activity",
                "evidence_kind": "synthetic_test_only",
            },
        }

    monkeypatch.setattr(migration, "validate_model_goal", validate)
    monkeypatch.setattr(library, "validate_model_goal", validate)
    return library.ScientistModelLibrary(root.parent / "library"), root, mid, controller, protocol, data


def fingerprint(root):
    return {
        str(p.relative_to(root)): file_hash(p) for p in root.rglob("*") if p.is_file() and not p.name.startswith(".")
    }


def test_explicit_migration_replays_numerics_preserves_origin_and_is_idempotent(legacy):
    store, root, mid, controller, protocol, data = legacy
    before = fingerprint(root)
    with pytest.raises(ValueError, match="implementation changed"):
        store.publish(root, mid)
    result = store.migrate(root, mid)
    assert result["created"] and result["new_autonomous_models"] == 0
    verified = store.resolve(result["entry_id"], replay_training=True)
    assert verified["packet"]["assessment"]["numerical_evidence"]
    assert verified["origin"]["model_id"] == mid
    assert verified["origin"]["evidence_kind"] == "runtime_revalidated_native_scientist_model"
    assert verified["packet"]["model_id"] != mid
    assert fingerprint(root) == before
    repeated = store.migrate(root, mid)
    assert repeated["already_migrated"] and not repeated["created"]
    assert repeated["entry_id"] == result["entry_id"]
    assert store.catalog(verify=True)["verified_entries"] == 1
    assert fingerprint(root) == before
    # Migration does not silently weaken the ordinary publisher's old-runtime gate.
    with pytest.raises(ValueError, match="implementation changed"):
        store.publish(root, mid)


@pytest.mark.parametrize(
    "attack",
    [
        "archive",
        "policy",
        "assessment",
        "original_retirement",
        "migrated_retirement",
        "certificate",
        "missing_current_evidence",
    ],
)
def test_changed_or_retired_evidence_cannot_keep_a_migrated_model_available(legacy, attack):
    store, root, mid, controller, protocol, data = legacy
    result = store.migrate(root, mid)
    migrated = store.root / "migrations" / result["migration_id"]
    certificate = library._read(migrated / "certificate.json")
    if attack == "archive":
        (root / "implementation/src/agent/rsi/code_worker.py").write_text("changed")
    elif attack == "policy":
        policy = library._read(controller / "policy.json")
        policy["minimum_skill"] = 0.1
        atomic_json(controller / "policy.json", policy)
    elif attack == "assessment":
        path = controller / f"assessment_{mid}.json"
        value = library._read(path)
        value["metrics"]["rmse"] = 1234
        atomic_json(path, value)
    elif attack == "original_retirement":
        state = library._read(controller / "state.json")
        state["published"] = []
        state["candidates"][mid]["status"] = "retired"
        atomic_json(controller / "state.json", state)
    elif attack == "migrated_retirement":
        current = migrated / "controller"
        ModelLifecycle(current, library._read(current / "policy.json")).retire(
            certificate["model_id"], "Synthetic test"
        )
    elif attack == "certificate":
        certificate["new_autonomous_models"] = 1
        atomic_json(migrated / "certificate.json", certificate)
    else:
        (migrated / "controller/numerical_evidence").rename(migrated / "saved_numeric_fixture")
    with pytest.raises(ValueError):
        store.resolve(result["entry_id"])
    if attack == "missing_current_evidence":
        assert not (migrated / "controller/numerical_evidence").exists()


def test_old_interface_hash_must_bind_archived_sources(legacy):
    store, root, mid, controller, protocol, data = legacy
    policy = library._read(controller / "policy.json")
    policy["plugin_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="supported numerical ABI"):
        migration.read_legacy_model(root, controller, mid, policy, protocol)
    unsupported = copy.deepcopy(policy)
    unsupported["plugin_id"] = "generated_features.v1"
    with pytest.raises(ValueError, match="No migration verifier"):
        migration.read_legacy_model(root, controller, mid, unsupported, protocol)


def test_rejected_migration_retains_failure_without_changing_original(legacy):
    store, root, mid, controller, protocol, data = legacy
    state = library._read(controller / "state.json")
    state["published"] = []
    atomic_json(controller / "state.json", state)
    before = fingerprint(root)
    with pytest.raises(ValueError):
        store.migrate(root, mid)
    reports = list((store.root / "migration_reports").glob("*.json"))
    assert len(reports) == 1
    report = store._object("migration_reports", reports[0].stem)
    assert report["status"] == "failed" and report["new_autonomous_models"] == 0
    assert fingerprint(root) == before


def test_migration_inference_matches_existing_predictions_without_editing_origin(legacy):
    store, root, mid, controller, protocol, data = legacy
    result = store.migrate(root, mid)
    before = fingerprint(root)
    verified = store.resolve(result["entry_id"])
    predicted = store.predict(
        result["entry_id"], data["future"], verified["key"]["scope"], session_id="synthetic-agent-chat"
    )
    assert predicted["predictions"] == library._read(root / "episodes/cycle_001/workspace/predictions.json")
    assert predicted["receipt"]["mode"] == "science_agent"
    assert fingerprint(root) == before


def test_runtime_canonical_replacement_preserves_history_and_count(legacy):
    store, root, mid, controller, protocol, data = legacy
    # Simulate an index originally published when the legacy runtime was current.
    original = migration.validate_model_goal(root, mid, runtime_migration=True)
    previous = store._publish_verified(root, original)
    result = store.migrate(root, mid)
    assert not result["created"] and result["entry_id"] != previous["entry_id"]
    entry = store._object("entries", result["entry_id"])
    assert entry["previous_entry_id"] == previous["entry_id"]
    # A concurrent writer that already validated the same migration is idempotent too.
    verified = store.resolve(result["entry_id"])
    duplicate = store._publish_verified(root, verified, replace_runtime=True)
    assert duplicate["entry_id"] == result["entry_id"] and not duplicate["created"]
    assert store._object("entries", previous["entry_id"])
    assert store.catalog(verify=True)["verified_entries"] == 1
    with pytest.raises(ValueError, match="index changed"):
        store.resolve(previous["entry_id"])


def test_publication_inspection_accepts_certified_runtime_identity_without_replay(legacy, monkeypatch):
    store, root, mid, controller, protocol, data = legacy
    result = store.migrate(root, mid)
    verified = store.resolve(result["entry_id"], replay_training=True)
    before = fingerprint(root)

    def unexpected(*args, **kwargs):
        pytest.fail("Metadata inspection must not replay or decode model data")

    monkeypatch.setattr(migration, "verify_migration", unexpected)
    monkeypatch.setattr(migration, "validate_model_goal", unexpected)
    monkeypatch.setattr(library, "validate_model_goal", unexpected)
    inspected = store.inspect_publication(result["entry_id"])
    for field in ("entry_id", "key_id", "origin_goal", "key", "origin", "packet"):
        assert inspected[field] == verified[field]
    assert inspected["origin"]["model_id"] == mid
    assert inspected["packet"]["model_id"] != mid
    assert fingerprint(root) == before


@pytest.mark.parametrize("attack", ["bytes", "missing", "goal", "original_model_id", "model_id", "packet_sha256"])
def test_publication_inspection_rejects_changed_migration_bindings(legacy, attack):
    store, root, mid, controller, protocol, data = legacy
    result = store.migrate(root, mid)
    certificate_path = store.root / "migrations" / result["migration_id"] / "certificate.json"
    certificate = library._read(certificate_path)
    entry_id = result["entry_id"]
    if attack == "missing":
        certificate_path.unlink()
    elif attack == "bytes":
        certificate["new_autonomous_models"] = 1
        atomic_json(certificate_path, certificate)
    else:
        # Rebind the immutable objects too: matching hashes alone cannot establish
        # that a certificate belongs to this goal and this pair of artifact IDs.
        verified = store.inspect_publication(entry_id)
        certificate[attack] = "another-goal" if attack == "goal" else "0" * 64
        identity = digest(certificate)
        atomic_json(store.root / "migrations" / identity / "certificate.json", certificate)
        verified["origin"] = {**verified["origin"], "runtime_migration": identity}
        entry_id = store._publish_verified(root, verified, replace_runtime=True)["entry_id"]
    with pytest.raises(ValueError, match="Migration|Missing"):
        store.inspect_publication(entry_id)


def test_publication_inspection_requires_migration_certificate_for_different_ids(legacy):
    store, root, mid, controller, protocol, data = legacy
    result = store.migrate(root, mid)
    verified = store.inspect_publication(result["entry_id"])
    verified["origin"].pop("runtime_migration")
    separate = library.ScientistModelLibrary(root.parent / "uncertified_library")
    published = separate._publish_verified(root, verified)
    with pytest.raises(ValueError, match="Published model identity changed"):
        separate.inspect_publication(published["entry_id"])
