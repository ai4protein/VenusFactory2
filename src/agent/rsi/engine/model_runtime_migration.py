"""Explicit numerical revalidation of historical Scientist model artifacts.

Original goals and policies remain unchanged. Migration never invents a model,
weakens an acceptance threshold, executes archived controller code or scores
future labels. The migrated model must equal the original fitted artifact.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import uuid
from pathlib import Path

from agent.harness.contracts import digest, file_hash

from agent.rsi import model_tasks
from .model_lifecycle import ModelLifecycle
from .model_plugins import get_model_plugin
from .scientist_model_library import _identity, _read, _require, _tree, validate_model_goal
from .storage import atomic_json, file_lock, safe_child
from .versions import VersionStore

LEGACY_FILES = ("generated_model.py", "code_sandbox.py", "code_worker.py", "protein_references.py")
KIND = "numerical_runtime_migration.v1"


def read_legacy_model(root, controller, model_id, policy, protocol):
    """Read the known pre-numerical-evidence ABI; current numerical replay follows."""
    _require(policy["plugin_id"] == "generated_python.v1", "No migration verifier for this historical interface")
    _require(
        set(policy)
        == {
            "schema",
            "scope",
            "plugin_id",
            "plugin_sha256",
            "training_sha256",
            "validation_sha256",
            "minimum_skill",
            "max_candidates",
        }
        and policy["schema"] == 1
        and policy["max_candidates"] == 1
        and type(policy["minimum_skill"]) in (int, float)
        and 0 <= policy["minimum_skill"] < 1,
        "Invalid historical assessment policy",
    )
    sources = []
    for name in LEGACY_FILES:
        relative = "src/agent/rsi/" + name
        path = safe_child(root, "implementation/" + relative)
        _require(file_hash(path) == protocol["implementation"][relative], "Historical numerical source changed")
        sources.append(path.read_bytes())
    expected = hashlib.sha256(b"".join(sources)).hexdigest()
    _require(policy["plugin_sha256"] == expected, "Historical policy does not bind the supported numerical ABI")
    _tree(controller)
    _read(safe_child(controller, f"artifacts/versions/{_identity(model_id)}/manifest.json"))
    manifest = VersionStore(controller / "artifacts").verify(model_id)
    state = _read(controller / "state.json")
    _require(
        state.get("schema") == 1
        and isinstance(state.get("candidates"), dict)
        and isinstance(state.get("published"), list)
        and isinstance(state.get("events"), list),
        "Invalid historical lifecycle state",
    )
    candidate = state["candidates"][model_id]
    _require(
        manifest["metadata"]
        == {"session_id": candidate["session_id"], "mode": candidate["mode"], "policy_sha256": digest(policy)},
        "Historical model policy binding changed",
    )
    model = _read(safe_child(controller, f"artifacts/versions/{model_id}/model.json"))
    report = _read(controller / f"assessment_{model_id}.json")
    _require("numerical_evidence" not in report, "This migration supports the pre-evidence numerical ABI only")
    _require(
        report.get("schema") == 1
        and report.get("model_id") == model_id
        and report.get("policy_sha256") == digest(policy)
        and report.get("plugin_sha256") == expected
        and report.get("passed") is True
        and report.get("training_replayed") is True
        and digest(report) == candidate["assessment_sha256"],
        "Historical acceptance binding changed",
    )
    return state, model, report


def original_controller(folder, model_id):
    state = _read(folder / "goal.json")
    records = state["progress"].get("generations") or [state["progress"]["accepted_model"]]
    records = [r for r in records if r["model_id"] == model_id]
    _require(len(records) == 1, "Ambiguous original model")
    return safe_child(folder, records[0]["controller"])


def migration_view(original, packet, identity):
    _require(packet["model"] == original["packet"]["model"], "Migration changed fitted model or source")
    _require(
        packet["assessment"]["metrics"] == original["packet"]["assessment"]["metrics"],
        "Migration changed validation results",
    )
    return {
        **original,
        "packet": packet,
        "origin": {
            **original["origin"],
            "runtime_migration": identity,
            "evidence_kind": "runtime_revalidated_native_scientist_model",
        },
    }


def verify_migration(library, folder, identity, *, replay_training=False):
    folder = library.goal(folder)
    root = safe_child(library.root, "migrations/" + _identity(identity))
    _tree(root)
    certificate = _read(root / "certificate.json")
    _require(digest(certificate) == identity and certificate.get("kind") == KIND, "Migration certificate changed")
    _require(certificate["goal"] == str(folder.relative_to(library.owner)), "Migration belongs to another goal")
    original = validate_model_goal(
        folder, certificate["original_model_id"], replay_training=replay_training, runtime_migration=True
    )
    old_policy = _read(original_controller(folder, certificate["original_model_id"]) / "policy.json")
    plugin = get_model_plugin(old_policy["plugin_id"])
    _require(certificate["plugin_sha256"] == plugin.source_sha256, "Migrated numerical runtime has changed")
    policy = {**old_policy, "plugin_sha256": plugin.source_sha256}
    _require(
        certificate["original_origin_sha256"] == digest(original["origin"])
        and certificate["original_packet_sha256"] == digest(original["packet"])
        and certificate["original_policy_sha256"] == digest(old_policy)
        and certificate["current_policy_sha256"] == digest(policy),
        "Migration origin or policy changed",
    )
    controller = root / "controller"
    mid = _identity(certificate["model_id"])
    for name in ("policy.json", "state.json", f"assessment_{mid}.json", f"artifacts/versions/{mid}/manifest.json"):
        _read(safe_child(controller, name))
    assessment = _read(controller / f"assessment_{mid}.json")
    reference = assessment["numerical_evidence"]
    _read(
        safe_child(controller, "numerical_evidence/versions/" + _identity(reference["version_id"]) + "/manifest.json")
    )
    lc = ModelLifecycle(controller, policy)
    state, model = lc._state(), lc._model(mid)
    _require(mid in state["published"] and state["candidates"][mid]["status"] == "registered", "Migrated model retired")
    report = lc._verified_assessment(mid, state, model)
    packet = {"schema": 1, "model_id": mid, "model": model, "model_sha256": digest(model), "assessment": report}
    _require(digest(packet) == certificate["packet_sha256"], "Migration model packet changed")
    _require(
        report["training_replayed"]
        and report["passed"]
        and report["metrics"][plugin.acceptance_metric] > policy["minimum_skill"],
        "Migration acceptance changed",
    )
    if replay_training:
        train, validation = (_read(folder / "data" / (phase + ".json")) for phase in ("training", "validation"))
        _require(plugin.fit(train, policy["scope"], model["recipe"]) == model, "Migrated training replay differs")
        _require(
            plugin.evaluate(model, validation, policy["scope"]) == report["metrics"],
            "Migrated validation replay differs",
        )
    return migration_view(original, packet, identity)


def migrate(library, folder, model_id):
    """Revalidate explicitly, preserve failed attempts, and publish a compatible view."""
    folder = library.goal(folder)
    report = {
        "kind": KIND,
        "call_id": uuid.uuid4().hex,
        "goal": str(folder.relative_to(library.owner)),
        "original_model_id": _identity(model_id),
        "new_autonomous_models": 0,
    }
    try:
        result = _migrate(library, folder, model_id)
    except Exception as exc:
        with file_lock(library.root / ".migration.lock"):
            library._immutable("migration_reports", {**report, "status": "failed", "error_type": type(exc).__name__})
        raise
    with file_lock(library.root / ".migration.lock"):
        report_id = library._immutable("migration_reports", {**report, "status": "available", "result": result})
    return {**result, "report_id": report_id}


def _migrate(library, folder, model_id):
    folder = library.goal(folder)
    original = validate_model_goal(folder, model_id, replay_training=True, runtime_migration=True)
    old_policy = _read(original_controller(folder, model_id) / "policy.json")
    plugin = get_model_plugin(old_policy["plugin_id"])
    _require(plugin.source_sha256 != old_policy["plugin_sha256"], "Model already uses the current runtime; use publish")
    key_id = digest(original["key"])
    index = safe_child(library.root, f"index/{key_id}.json")
    if index.exists():
        entry_id = _read(index)["entry_id"]
        entry = library._object("entries", entry_id)
        origin = library._object("origins/" + key_id, entry["canonical_origin"])
        _require(
            origin["goal"] == str(folder.relative_to(library.owner)) and origin["model_id"] == model_id,
            "Migrate the existing canonical origin before this supporting model",
        )
        if origin.get("runtime_migration"):
            library.resolve(entry_id, replay_training=True)
            return {
                "entry_id": entry_id,
                "key_id": key_id,
                "created": False,
                "migration_id": origin["runtime_migration"],
                "already_migrated": True,
                "new_autonomous_models": 0,
            }
    attempts = safe_child(library.root, "migration_attempts")
    attempts.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix="attempt_", dir=attempts))
    atomic_json(
        attempt / "attempt.json",
        {
            "kind": KIND,
            "goal": str(folder.relative_to(library.owner)),
            "original_model_id": model_id,
            "status": "revalidating",
        },
    )
    try:
        policy = {**old_policy, "plugin_sha256": plugin.source_sha256}
        lc = ModelLifecycle(attempt / "controller", policy)
        atomic_json(attempt / "original_model.json", original["packet"]["model"])
        session = "runtime_migration_" + digest({"origin": original["origin"], "policy": policy})
        mid = lc.submit(attempt / "original_model.json", session_id=session, mode="science_expert")
        data = {phase: _read(folder / "data" / (phase + ".json")) for phase in model_tasks.PHASES}
        assessment = lc.assess(mid, training=data["training"], validation=data["validation"])
        _require(
            assessment["passed"] and assessment["metrics"] == original["packet"]["assessment"]["metrics"],
            "Current numerical acceptance differs from the original",
        )
        packet = lc.export(mid, attempt / "export")
        certificate = {
            "schema": 1,
            "kind": KIND,
            "goal": str(folder.relative_to(library.owner)),
            "original_model_id": model_id,
            "original_origin_sha256": digest(original["origin"]),
            "original_packet_sha256": digest(original["packet"]),
            "original_policy_sha256": digest(old_policy),
            "current_policy_sha256": digest(policy),
            "plugin_sha256": plugin.source_sha256,
            "model_id": mid,
            "packet_sha256": digest(packet),
            "new_autonomous_models": 0,
            "future_labels_scored": False,
            "migration_verifier_sha256": file_hash(Path(__file__)),
        }
        identity = digest(certificate)
        atomic_json(attempt / "certificate.json", certificate)
        atomic_json(
            attempt / "attempt.json",
            {
                "kind": KIND,
                "status": "numerically_revalidated",
                "migration_id": identity,
                "original_goal_changed": False,
                "new_autonomous_models": 0,
            },
        )
        destination = safe_child(library.root, "migrations/" + identity)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with file_lock(library.root / ".migration.lock"):
            if destination.exists():
                _require(_read(destination / "certificate.json") == certificate, "Migration identity collision")
            else:
                os.replace(attempt, destination)
        verified = verify_migration(library, folder, identity, replay_training=True)
        published = library._publish_verified(folder, verified, replace_runtime=True)
        return {**published, "migration_id": identity, "already_migrated": False, "new_autonomous_models": 0}
    except Exception as exc:
        if attempt.exists():
            atomic_json(
                attempt / "failure.json",
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "original_goal_changed": False,
                    "new_autonomous_models": 0,
                },
            )
        raise
