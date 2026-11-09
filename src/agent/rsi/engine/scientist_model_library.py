"""Host-verified Scientist models shared across goals in one owner namespace.

Publication verifies native expert execution and numerical evidence. Catalog
identity groups a program on one frozen training task; it is not an algorithmic
novelty measure. Inference never writes to the originating goal's history.
"""

from __future__ import annotations

import copy
import json
import math
import uuid
from pathlib import Path

from agent.harness.contracts import Profile, digest, file_hash
from agent.harness.events import audit_requests, audit_trace
from agent.harness.native_expert_audit import audit_native_expert
from agent.harness.native_expert_team import NATIVE_EXPERT_HARNESSES
from agent.research_modes import get_research_mode
from agent.rsi.datasets import model_tasks
from agent.rsi.datasets.source_lineage import program_hash
from agent.science_goal import GoalSpec, GoalStore, validate_assessment

from . import library_experience, training_attempt_memory
from .context_feedback_dispatch import verification_session
from .experience import ModelExperienceStore
from .lineage_promotion import gain_error, validate_evolution
from .model_lifecycle import ModelLifecycle
from .model_plugins import get_model_plugin
from .storage import atomic_json, file_lock, safe_child
from .versions import VersionStore

_COUNTING_RULE = "program + task scope + frozen training data + model interface; not algorithmic novelty"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _identity(value):
    _require(
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
        "Invalid model library identity",
    )
    return value


def _root(path):
    path = Path(path).absolute()
    _require(not any(p.is_symlink() for p in [path, *path.parents]), "Model paths cannot traverse symlinks")
    return path.resolve()


def _read(path):
    _require(path.is_file() and not path.is_symlink(), "Missing or symlinked model evidence")
    return json.loads(path.read_text())


def _tree(root):
    _require(root.is_dir() and not root.is_symlink(), "Missing or symlinked model evidence directory")
    _require(not any(p.is_symlink() for p in root.rglob("*")), "Symlinked model evidence")


def model_key(model, _policy):
    """Ignore names, seeds, comments and docstrings, not semantic program equivalence."""
    return {
        "schema": 1,
        "plugin_id": model["plugin_id"],
        "scope": model["scope"],
        "training_sha256": model["training_sha256"],
        "program_sha256": program_hash(model["recipe"]["source_code"]),
    }


def _episodes(root, state, protocol):
    from .context_feedback_dispatch import verify_bundle, verify_goal

    verify_goal(state, protocol)
    training_attempt_memory.verify_goal(state, protocol)
    base = Profile.from_dict(state["definition"]["profile"])
    _require(base.harness in NATIVE_EXPERT_HARNESSES, "No publication auditor for this Scientist harness yet")
    _require(base.mode == "science_expert", "Scientist execution required")
    _require(
        base.model.provider.lower() not in {"test", "test_fixture", "fixture", "mock"},
        "Fixture execution cannot be published",
    )
    prefix = copy.deepcopy(state)
    prefix["cycles"], prefix["progress"] = [], {}
    counts = {"model_calls": 0, "tool_calls": 0}
    sessions, episodes = set(), {}
    for index, cycle in enumerate(state["cycles"]):
        _require(cycle["index"] == index, "Non-contiguous Scientist episodes")
        folder = safe_child(root, f"episodes/cycle_{index:03d}")
        _tree(folder)
        _require(_read(root / f"cycle_{index:03d}.json") == cycle, "Episode checkpoint changed")
        outcome = _read(folder / "outcome.json")
        _require(
            {k: v for k, v in cycle.items() if k not in {"index", "usage"}}
            == {k: v for k, v in outcome.items() if k != "usage"},
            "Episode outcome changed",
        )
        training_attempt_memory.verify_episode(root, prefix, protocol, cycle)
        library_experience.verify_episode(root, prefix, protocol, cycle)
        run = _read(folder / "execution/run.json")
        lock = _read(folder / "execution/profile.lock.json")
        profile = Profile.from_dict(lock["profile"])
        _require(
            profile.model == base.model
            and profile.mode == base.mode
            and profile.harness == base.harness
            and profile.required_controls == base.required_controls,
            "Episode differs from the goal's model or harness",
        )
        for name in ("output_tokens", "temperature", "thinking", "reasoning_effort"):
            _require(getattr(profile.budget, name) == getattr(base.budget, name), "Episode sampling changed")
        for name in ("model_calls", "tool_calls", "wall_seconds", "request_seconds"):
            _require(getattr(profile.budget, name) <= getattr(base.budget, name), "Episode budget increased")
        _require(lock["sha256"] == digest({k: v for k, v in lock.items() if k != "sha256"}), "Episode lock changed")
        _require(run["profile_sha256"] == lock["sha256"], "Execution binding changed")
        requests = audit_requests(folder / "execution/model_requests.jsonl")
        events = audit_trace(folder / "execution/events.jsonl", lock["task"]["tools"], profile.budget.tool_calls)
        for name, value in (("model_calls", requests["model_calls"]), ("tool_calls", events["tool_calls"])):
            _require(value == cycle["usage"][name], "Episode usage changed")
            counts[name] += value
        if cycle["status"] == "completed":
            versions = folder / "capability/bundles/versions"
            caps = list(versions.iterdir())
            _require(len(caps) == 1, "Ambiguous episode capability")
            VersionStore(versions.parent).verify(caps[0].name)
            verify_bundle(_read(caps[0] / "task.json"), protocol)
            check = audit_native_expert(folder / "execution", folder / "workspace", caps[0])
            _require(check["passed"], "Native Scientist audit failed")
            _require(run["session_id"] not in sessions, "Scientist sessions were not fresh")
            sessions.add(run["session_id"])
            episodes[str(folder.relative_to(root))] = run
        prefix["cycles"].append(cycle)
        if cycle.get("progress") is not None:
            prefix["progress"] = copy.deepcopy(cycle["progress"])
    _require(all(state["usage"][k] == v for k, v in counts.items()), "Cumulative goal usage changed")
    return episodes, counts


def _model(
    root, record, protocol, data, episodes, threshold, replay_training, runtime_migration=False, *, evidence_paths=None
):
    mid = _identity(record["model_id"])
    controller = safe_child(root, record["controller"])
    _tree(controller)
    for name in ("policy.json", "state.json", f"assessment_{mid}.json", f"artifacts/versions/{mid}/manifest.json"):
        _read(safe_child(controller, name))
    assessment = _read(controller / f"assessment_{mid}.json")
    if "numerical_evidence" in assessment:
        reference = assessment["numerical_evidence"]
        _require(isinstance(reference, dict), "Invalid numerical evidence reference")
        version = _identity(reference["version_id"])
        _read(safe_child(controller, f"numerical_evidence/versions/{version}/manifest.json"))
    policy = _read(controller / "policy.json")
    plugin_id = protocol.get("model_plugin_id", "generated_python.v1")
    _require(plugin_id in model_tasks.PLUGINS, "Unsupported generated-model interface")
    plugin = get_model_plugin(plugin_id)
    _require(
        policy["scope"] == protocol["scope"]
        and policy["plugin_id"] == plugin_id
        and policy["training_sha256"] == digest(data["training"])
        and policy["validation_sha256"] == digest(data["validation"])
        and policy["minimum_skill"] == threshold,
        "Model assessment differs from frozen task policy",
    )
    if plugin_id == "generated_context.v1" and policy["plugin_sha256"] != plugin.source_sha256:
        from .context_runtime_compat import read_compatible_context

        state, model, report = read_compatible_context(root, controller, mid, policy, protocol)
        # Exact numerical equivalence is mandatory for every compatible read.
        replay_training = True
    elif runtime_migration and policy["plugin_sha256"] != plugin.source_sha256:
        from .model_runtime_migration import read_legacy_model

        state, model, report = read_legacy_model(root, controller, mid, policy, protocol)
    else:
        lc = ModelLifecycle(controller, policy)
        state, model = lc._state(), lc._model(mid)
        report = lc._verified_assessment(mid, state, model)
    candidate = state["candidates"][mid]
    _require(mid in state["published"] and candidate["status"] == "registered", "Model is unavailable or retired")
    _require(report.get("training_replayed") is True, "Missing independent training evidence")
    _require(
        model["scope"] == protocol["scope"]
        and model["plugin_id"] == plugin_id
        and model["training_sha256"] == digest(data["training"]),
        "Model task binding changed",
    )
    skill = report["metrics"][plugin.acceptance_metric]
    _require(type(skill) in (float, int) and math.isfinite(skill) and skill > threshold, "Acceptance threshold not met")
    if evidence_paths is None:
        develop = controller.parent
        reuse = safe_child(root, record["reuse_receipt"]).parent
        development_workspace, reuse_workspace = develop / "workspace", reuse / "workspace"
        receipt_path = reuse / "reuse_receipt.json"
    else:
        _require(
            set(evidence_paths) == {"develop", "reuse", "development_workspace", "reuse_workspace", "receipt"},
            "Invalid explicit model evidence layout",
        )
        paths = {name: safe_child(root, value) for name, value in evidence_paths.items()}
        develop, reuse = paths["develop"], paths["reuse"]
        development_workspace, reuse_workspace, receipt_path = (
            paths["development_workspace"],
            paths["reuse_workspace"],
            paths["receipt"],
        )
        _require(
            develop in controller.parents
            and develop in development_workspace.parents
            and reuse in reuse_workspace.parents
            and reuse in receipt_path.parents,
            "Model evidence escaped its shared episode",
        )
    dev_run, reuse_run = (episodes.get(str(p.relative_to(root))) for p in (develop, reuse))
    _require(dev_run is not None and reuse_run is not None, "Model lacks completed Scientist episodes")
    _require(dev_run["phase"] == "develop" and reuse_run["phase"] == "reuse", "Invalid model episode phases")
    _require(
        candidate["session_id"] == dev_run["session_id"] != reuse_run["session_id"]
        and candidate["mode"] == "science_expert",
        "Model session binding changed",
    )
    _require(_read(development_workspace / "candidate.json") == model, "Registered model differs from expert artifact")
    provenance = _read(development_workspace / "source_provenance.json")
    _require(provenance["source_sha256"] == model["source_sha256"], "Generated source provenance changed")
    if replay_training:
        _require(plugin.fit(data["training"], protocol["scope"], model["recipe"]) == model, "Training replay differs")
        _require(
            plugin.evaluate(model, data["validation"], protocol["scope"]) == report["metrics"],
            "Validation replay differs",
        )
    inputs = model_tasks.inference_inputs(data["future"], plugin_id)
    if plugin_id in {
        "generated_context.v1",
        "generated_context.v2",
        "generated_context.v3",
        "generated_assay.v1",
        "generated_antibody.v1",
    }:
        from .context_interfaces import module_for

        module_for(plugin_id).require_independent(model, inputs)
    else:
        _require(not set(model["training_sequences"]) & {r["sequence"] for r in inputs}, "Reuse overlaps training")
    predictions = plugin.predict(model, inputs, protocol["scope"])
    _require(predictions == _read(reuse_workspace / "predictions.json"), "Fresh-session inference differs")
    receipt = _read(receipt_path)
    expected = {
        "event": "reused",
        "model_id": mid,
        "session_id": reuse_run["session_id"],
        "mode": "science_expert",
        "input_sha256": digest(inputs),
        "prediction_sha256": digest(predictions),
        "rows": len(inputs),
    }
    _require(receipt == expected and receipt in state["events"], "Fresh-session receipt changed")
    return {
        "key": model_key(model, policy),
        "packet": {"schema": 1, "model_id": mid, "model": model, "model_sha256": digest(model), "assessment": report},
        "skill": skill,
        "reuse_receipt_sha256": digest(receipt),
    }


def _experience(root, identity):
    folder = safe_child(root, "experience")
    _tree(folder)
    _read(safe_child(folder, "catalog.json"))
    _require(safe_child(folder, "entries").is_dir(), "Missing experience entries")
    return ModelExperienceStore(folder)._entry(identity)


@verification_session()
def validate_model_goal(folder, model_id, *, replay_training=True, runtime_migration=False):
    """Validate existing evidence without an LLM call or future-label scoring."""
    root = _root(folder)
    state, protocol = _read(root / "goal.json"), _read(root / "protocol.json")
    if protocol.get("schema") == "shared_scientist_goal.v1":
        _require(not runtime_migration, "Shared model runtime migration requires an explicit policy")
        from .shared_model_publication import validate_shared_goal

        return validate_shared_goal(root, model_id, replay_training=replay_training)
    _require(
        state["schema"] == 1 and digest(state["definition"]) == state["definition_sha256"], "Goal definition changed"
    )
    milestone = None
    if (root / "generation_milestone.json").exists():
        from .generation_publication import verify_snapshot

        milestone = verify_snapshot(root)
    else:
        _require(state["status"] == "completed" and state["inflight"] is None, "Publication requires a completed goal")
    _require(state["definition"].get("condition") == "science_goal", "Only Scientist goals enter this library")
    GoalSpec(**{**state["definition"]["goal"], "criteria": tuple(state["definition"]["goal"]["criteria"])})
    criteria = set(state["definition"]["goal"]["criteria"])
    expected = {"accepted_model", "fresh_session_reuse"} | ({"verified_lineage"} if "evolution" in protocol else set())
    _require(criteria == expected, "Unsupported model-goal criteria")
    for check in state["assessment"]["criteria"].values():
        for item in check.get("evidence", []):
            safe_child(root, item["path"])
    if milestone is None:
        _require(validate_assessment(GoalStore(root), state, state["assessment"]), "Goal criteria do not pass")
    _require(bool(protocol["implementation"]), "Missing frozen implementation")
    for name, sha in protocol["implementation"].items():
        _require(file_hash(safe_child(root, "implementation/" + name)) == sha, "Archived implementation changed")
    data = {name: _read(safe_child(root, f"data/{name}.json")) for name in model_tasks.PHASES}
    _require(
        all(digest(rows) == protocol["data_hashes"][name] for name, rows in data.items()), "Frozen model data changed"
    )
    task = None
    if "model_task" in protocol:
        task = _read(root / "model_task.json")
        _require(
            model_tasks.admission(task) == protocol["model_task"] == state["definition"].get("admission")
            and task["plugin_id"] == protocol["model_plugin_id"]
            and task["scope"] == protocol["scope"]
            and task["minimum_skill"] == protocol["model_task_minimum_skill"]
            and model_tasks.load_task(task) == data,
            "Task admission changed",
        )
    library_experience.verify_goal(root, state, protocol)
    episodes, usage = _episodes(root, state, protocol)
    if "evolution" in protocol:
        validate_evolution(protocol)
        records = state["progress"]["generations"]
        if milestone is None:
            _require(len(records) == protocol["evolution"]["generations"], "Incomplete evolution goal")
        else:
            _require(0 < len(records) <= protocol["evolution"]["generations"], "Invalid generation milestone")
    else:
        records = [{**state["progress"]["accepted_model"], "reuse_receipt": state["progress"]["reuse_receipt"]}]
    _require(len({r["model_id"] for r in records}) == len(records), "Duplicate lineage models")
    _require(model_id in {r["model_id"] for r in records}, "Model is not a verified goal output")
    selected, programs, parents = None, set(), []
    threshold = protocol.get("model_task_minimum_skill", 0.0)
    for index, record in enumerate(records):
        result = _model(root, record, protocol, data, episodes, threshold, replay_training, runtime_migration)
        if "evolution" in protocol:
            _require(
                gain_error(protocol, threshold, result["skill"]) is None,
                "Development gain fails the frozen promotion guard",
            )
            entry = _experience(root, record["experience_id"])
            _require(
                entry["model_id"] == record["model_id"] and entry["parents"] == parents, "Evolution ancestry changed"
            )
            if protocol["evolution"].get("schema", 1) >= 2:
                program = result["key"]["program_sha256"]
                _require(
                    record["generation"] == index + 1
                    and record["program_sha256"] == program
                    and program not in programs
                    and record["source_sha256"] == entry["source_sha256"] == result["packet"]["model"]["source_sha256"],
                    "Evolution program lineage changed",
                )
                programs.add(program)
            parents = (
                [record["experience_id"]]
                if protocol.get("history_policy", "source_and_metrics") == "source_and_metrics"
                else []
            )
        if record["model_id"] == model_id:
            selected = result
        threshold = result["skill"]
    return {
        **selected,
        "origin": {
            "goal_sha256": file_hash(root / "goal.json"),
            "protocol_sha256": file_hash(root / "protocol.json"),
            "model_id": model_id,
            "profile": state["definition"]["profile"],
            "usage": usage,
            "stage": task["stage"] if task else "unspecified",
            "paper": task["paper"] if task else None,
            "task_id": task["id"] if task else protocol["case"],
            "source": protocol["source"],
            "reuse_receipt_sha256": selected["reuse_receipt_sha256"],
            "evidence_kind": "native_scientist_generated_model",
            **(
                {
                    "generation_milestone": {
                        "snapshot_sha256": digest(milestone),
                        "parent_goal": milestone["parent_goal"],
                        "parent_status": milestone["parent_status"],
                        "requested_generations": milestone["requested_generations"],
                        "verified_generations": len(records),
                    }
                }
                if milestone is not None
                else {}
            ),
            "future_labels_scored": False,
        },
    }


class ScientistModelLibrary:
    def __init__(self, root):
        self.root = _root(root)
        self.owner = self.root.parent

    def goal(self, folder):
        folder = _root(folder)
        try:
            name = str(folder.relative_to(self.owner))
        except ValueError as exc:
            raise ValueError("Model goal is outside the owner namespace") from exc
        _require(
            name != "." and folder != self.root and self.root not in folder.parents, "A dedicated goal is required"
        )
        return safe_child(self.owner, name)

    def _immutable(self, area, value):
        identity = digest(value)
        path = safe_child(self.root, f"{area}/{identity}.json")
        if path.exists():
            _require(_read(path) == value, "Immutable library object changed")
        else:
            atomic_json(path, value)
            path.chmod(0o444)
        return identity

    def _object(self, area, identity):
        value = _read(safe_child(self.root, f"{area}/{_identity(identity)}.json"))
        _require(digest(value) == identity, "Library object identity changed")
        return value

    def publish(self, folder, model_id):
        folder = self.goal(folder)
        verified = validate_model_goal(folder, model_id, replay_training=True)
        return self._publish_verified(folder, verified)

    def migrate(self, folder, model_id):
        from .model_runtime_migration import migrate

        return migrate(self, folder, model_id)

    def _publish_verified(self, folder, verified, *, replace_runtime=False):
        folder = self.goal(folder)
        key_id = digest(verified["key"])
        origin = {
            **verified["origin"],
            "goal": str(folder.relative_to(self.owner)),
            "key_id": key_id,
            "packet_id": digest(verified["packet"]),
        }
        with file_lock(self.root / ".library.lock"):
            _require(
                file_hash(folder / "goal.json") == origin["goal_sha256"]
                and file_hash(folder / "protocol.json") == origin["protocol_sha256"],
                "Goal changed during publication",
            )
            packet_id = self._immutable("packets", verified["packet"])
            origin_id = self._immutable("origins/" + key_id, origin)
            index = safe_child(self.root, f"index/{key_id}.json")
            created = not index.exists()
            if created:
                entry_id = self._immutable(
                    "entries",
                    {"schema": 1, "key": verified["key"], "canonical_origin": origin_id, "packet_id": packet_id},
                )
                atomic_json(index, {"entry_id": entry_id})
                index.chmod(0o444)
            else:
                entry_id = _read(index)["entry_id"]
                existing = self._object("entries", entry_id)
                _require(existing["key"] == verified["key"], "Model index changed")
                if replace_runtime and (
                    existing["canonical_origin"] != origin_id or existing["packet_id"] != packet_id
                ):
                    prior = self._object("origins/" + key_id, existing["canonical_origin"])
                    _require(
                        "runtime_migration" in origin
                        and prior["goal"] == origin["goal"]
                        and prior["model_id"] == origin["model_id"],
                        "Runtime migration must preserve canonical provenance",
                    )
                    entry_id = self._immutable(
                        "entries",
                        {
                            "schema": 1,
                            "key": verified["key"],
                            "canonical_origin": origin_id,
                            "packet_id": packet_id,
                            "previous_entry_id": entry_id,
                        },
                    )
                    atomic_json(index, {"entry_id": entry_id})
                    index.chmod(0o444)
        return {"entry_id": entry_id, "key_id": key_id, "origin_id": origin_id, "created": created}

    def publish_goal(self, folder):
        """Ingest a completed goal, retaining failures without editing its history."""
        folder = self.goal(folder)
        state = _read(folder / "goal.json")
        if _read(folder / "protocol.json").get("schema") == "shared_scientist_goal.v1":
            from .shared_model_publication import publish_shared_goal

            return publish_shared_goal(self, folder)
        _require(
            state.get("status") == "completed" and state.get("inflight") is None,
            "Publication requires a completed goal",
        )
        records = state["progress"].get("generations")
        if records is None:
            records = [state["progress"]["accepted_model"]]
        _require(bool(records), "Completed goal contains no accepted models")
        outcomes = []
        for record in records:
            mid = _identity(record["model_id"])
            try:
                published = self.publish(folder, mid)
                outcomes.append({"model_id": mid, "status": "published", **published})
            except (ValueError, KeyError, OSError) as exc:
                outcomes.append(
                    {"model_id": mid, "status": "rejected", "error_type": type(exc).__name__, "reason": str(exc)}
                )
        report = {
            "schema": 1,
            "kind": "model_goal_publication_attempt",
            "attempt_id": uuid.uuid4().hex,
            "origin_goal": str(folder.relative_to(self.owner)),
            "goal_sha256": file_hash(folder / "goal.json"),
            "outcomes": outcomes,
            "all_published": all(o["status"] == "published" for o in outcomes),
            "new_entries": sum(o.get("created", False) for o in outcomes),
            "duplicate_entries": sum(o["status"] == "published" and not o["created"] for o in outcomes),
            "rejected_entries": sum(o["status"] == "rejected" for o in outcomes),
            "new_autonomous_executions": 0,
        }
        with file_lock(self.root / ".library.lock"):
            report_id = self._immutable("publication_attempts", report)
        return {**report, "report_id": report_id}

    def _binding(self, entry_id):
        entry = self._object("entries", entry_id)
        key_id = digest(entry["key"])
        _require(_read(safe_child(self.root, f"index/{key_id}.json")) == {"entry_id": entry_id}, "Model index changed")
        origin = self._object("origins/" + key_id, entry["canonical_origin"])
        folder = self.goal(safe_child(self.owner, origin["goal"]))
        return entry, key_id, origin, folder

    def _bind_verified(self, entry_id, binding, verified):
        entry, key_id, origin, folder = binding
        _require(
            verified["origin"] == {k: v for k, v in origin.items() if k not in {"goal", "key_id", "packet_id"}},
            "Model origin changed",
        )
        _require(
            verified["key"] == entry["key"]
            and origin["key_id"] == key_id
            and digest(verified["packet"]) == entry["packet_id"] == origin["packet_id"],
            "Model binding changed",
        )
        _require(self._object("packets", entry["packet_id"]) == verified["packet"], "Published model changed")
        _require(
            file_hash(folder / "goal.json") == origin["goal_sha256"]
            and file_hash(folder / "protocol.json") == origin["protocol_sha256"],
            "Goal changed during model verification",
        )
        return {**verified, "entry_id": entry_id, "key_id": key_id, "origin_goal": str(folder)}

    def inspect_publication(self, entry_id):
        """Read a bound publication without executing or replaying its model.

        This checks the immutable registry objects and their frozen goal anchors,
        not current numerical or native-execution evidence. Inference still
        requires resolve(); no successful inspection is cached as verification.
        """
        entry, _, origin, folder = self._binding(entry_id)
        packet = self._object("packets", entry["packet_id"])
        _require(model_key(packet["model"], None) == entry["key"], "Published model scope changed")
        if "runtime_migration" in origin:
            from .model_runtime_migration import inspect_migration

            inspect_migration(self, folder, origin, packet)
        else:
            _require(packet["model_id"] == origin["model_id"], "Published model identity changed")
        publication = {
            "key": entry["key"],
            "packet": packet,
            "origin": {k: v for k, v in origin.items() if k not in {"goal", "key_id", "packet_id"}},
        }
        # Rebind after reading so retirement or a changed index cannot race inspection.
        return self._bind_verified(entry_id, self._binding(entry_id), publication)

    def resolve(self, entry_id, *, replay_training=False):
        binding = self._binding(entry_id)
        _, _, origin, folder = binding
        if "runtime_migration" in origin:
            from .model_runtime_migration import verify_migration

            verified = verify_migration(self, folder, origin["runtime_migration"], replay_training=replay_training)
        else:
            verified = validate_model_goal(folder, origin["model_id"], replay_training=replay_training)
        return self._bind_verified(entry_id, self._binding(entry_id), verified)

    def resolve_many(self, entry_ids, *, replay_training=False):
        """Verify a bounded set, sharing terminal goal audits only within this call."""
        from .shared_model_publication import SCHEMA, validate_shared_goal

        _require(isinstance(entry_ids, list) and 1 <= len(entry_ids) <= 256, "Declare 1..256 model identities")
        _require(type(replay_training) is bool, "Training replay must be a boolean")
        identities = list(dict.fromkeys(_identity(value) for value in entry_ids))
        bindings = {entry_id: self._binding(entry_id) for entry_id in identities}
        shared, resolved = {}, {}
        for entry_id, binding in bindings.items():
            _, _, origin, folder = binding
            if "runtime_migration" in origin or _read(folder / "protocol.json").get("schema") != SCHEMA:
                value = self.resolve(entry_id, replay_training=replay_training)
            else:
                if folder not in shared:
                    rows = validate_shared_goal(folder, replay_training=replay_training)
                    shared[folder] = {row["origin"]["model_id"]: row for row in rows}
                    _require(len(shared[folder]) == len(rows), "Repeated shared model identity")
                _require(origin["model_id"] in shared[folder], "Model is not a qualified shared output")
                value = shared[folder][origin["model_id"]]
            resolved[entry_id] = self._bind_verified(entry_id, binding, value)
        # A different target must not change an earlier target's registry binding.
        for entry_id, value in resolved.items():
            self._bind_verified(entry_id, self._binding(entry_id), value)
        return resolved

    def _indexed_entries(self):
        for index in sorted((self.root / "index").glob("*.json")):
            key_id = _identity(index.stem)
            entry_id = _read(safe_child(self.root, "index/" + index.name))["entry_id"]
            entry = self._object("entries", entry_id)
            _require(digest(entry["key"]) == key_id, "Model index scope changed")
            yield key_id, entry_id, entry

    def discovery_metadata(self):
        """Read canonical publication metadata without scanning superseded origins."""
        models = []
        for key_id, entry_id, entry in self._indexed_entries():
            origin = self._object("origins/" + key_id, entry["canonical_origin"])
            packet = self._object("packets", entry["packet_id"])
            _require(
                origin["key_id"] == key_id and origin["packet_id"] == entry["packet_id"],
                "Model binding changed",
            )
            models.append({
                "entry_id": entry_id,
                "key": entry["key"],
                "origin": {
                    k: origin[k] for k in ("stage", "paper", "evidence_kind", "runtime_migration")
                    if k in origin
                },
                "plugin_sha256": packet.get("assessment", {}).get("plugin_sha256"),
            })
        return {
            "models": models,
            "indexed_entries": len(models),
            "counting_rule": _COUNTING_RULE,
        }

    def catalog(self, *, verify=False):
        entries = []
        for key_id, entry_id, entry in self._indexed_entries():
            if verify:
                self.resolve(entry_id, replay_training=True)
            origins = [
                self._object("origins/" + key_id, p.stem)
                for p in sorted(safe_child(self.root, "origins/" + key_id).glob("*.json"))
            ]
            entries.append(
                {
                    "entry_id": entry_id,
                    "key": entry["key"],
                    "verified_now": verify,
                    "recorded_origins": len(origins),
                    "recorded_artifact_variants": len({o["packet_id"] for o in origins}),
                }
            )
        return {
            "schema": 1,
            "entries": entries,
            "indexed_entries": len(entries),
            "verified_entries": len(entries) if verify else None,
            "distinct_programs": len({e["key"]["program_sha256"] for e in entries}),
            "distinct_scopes": len({digest(e["key"]["scope"]) for e in entries}),
            "counting_rule": _COUNTING_RULE,
        }

    def predict(self, entry_id, inputs, scope, *, session_id, mode="science_agent"):
        get_research_mode(mode)
        _require(isinstance(session_id, str) and bool(session_id.strip()), "Inference requires a session identifier")
        verified = self.resolve(entry_id)
        model = verified["packet"]["model"]
        _require(scope == model["scope"], "Model applicability mismatch")
        rows = model_tasks.inference_inputs(inputs, model["plugin_id"])
        predictions = get_model_plugin(model["plugin_id"]).predict(model, rows, scope)
        receipt = {
            "schema": 1,
            "kind": "model_library_inference",
            "call_id": uuid.uuid4().hex,
            "entry_id": entry_id,
            "session_id": session_id,
            "mode": mode,
            "scope_sha256": digest(scope),
            "input_sha256": digest(rows),
            "prediction_sha256": digest(predictions),
            "rows": len(rows),
        }
        with file_lock(self.root / ".library.lock"):
            receipt_id = self._immutable("usage", receipt)
        return {"predictions": predictions, "receipt": receipt, "receipt_id": receipt_id}

    def predict_many(self, requests):
        """Batch inference with request-local provenance verification and distinct sessions."""
        from agent.rsi.campaigns.model_library_batch import predict_many

        return predict_many(self, requests)
