"""Synthetic software checks, never counted as scientific model publications."""

import copy
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from test_generated_models import CODE, SCOPE, TRAIN, VALID

from agent.harness import Budget, ModelBinding, Profile
from agent.harness.contracts import digest, file_hash
from agent.rsi import model_tasks
from agent.rsi import scientist_model_library as library
from agent.rsi.generated_model import fit_candidate, predict
from agent.rsi.model_lifecycle import ModelLifecycle
from agent.rsi.model_plugins import get_model_plugin
from agent.rsi.storage import atomic_json
from agent.science_goal import GoalSpec, GoalStore


def read(path):
    return json.loads(path.read_text())


@pytest.fixture
def numerical(tmp_path):
    root = tmp_path / "goal"
    root.mkdir()
    future = [{"id": "HI", "sequence": "HI", "target": 19.0}, {"id": "IH", "sequence": "IH", "target": 20.0}]
    data = {"training": TRAIN, "validation": VALID, "future": future}
    plugin = get_model_plugin("generated_python.v1")
    policy = {
        "schema": 1,
        "scope": SCOPE,
        "plugin_id": plugin.id,
        "plugin_sha256": plugin.source_sha256,
        "training_sha256": digest(TRAIN),
        "validation_sha256": digest(VALID),
        "minimum_skill": 0.0,
        "max_candidates": 1,
    }
    model = fit_candidate(TRAIN, SCOPE, {"source_code": CODE, "seed": 1234})
    development = root / "episodes/cycle_000"
    reuse = root / "episodes/cycle_001"
    atomic_json(development / "workspace/candidate.json", model)
    atomic_json(development / "workspace/source_provenance.json", {"source_sha256": model["source_sha256"]})
    lc = ModelLifecycle(development / "controller", policy)
    mid = lc.submit(development / "workspace/candidate.json", session_id="development", mode="science_expert")
    assert lc.assess(mid, training=TRAIN, validation=VALID)["passed"]
    inputs = model_tasks.inference_inputs(future, plugin.id)
    predictions = predict(model, inputs, SCOPE)
    atomic_json(reuse / "workspace/predictions.json", predictions)
    receipt = lc.record_reuse(
        mid, session_id="reuse", mode="science_expert", inputs=inputs, predictions=predictions, scope=SCOPE
    )
    atomic_json(reuse / "reuse_receipt.json", receipt)
    record = {
        "model_id": mid,
        "controller": "episodes/cycle_000/controller",
        "reuse_receipt": "episodes/cycle_001/reuse_receipt.json",
    }
    episodes = {
        "episodes/cycle_000": {"phase": "develop", "session_id": "development"},
        "episodes/cycle_001": {"phase": "reuse", "session_id": "reuse"},
    }
    return root, record, {"scope": SCOPE}, data, episodes, lc


def test_replay_actual_model_and_reuse_without_future_scoring(numerical):
    root, record, protocol, data, episodes, lc = numerical
    before = file_hash(lc.root / "state.json")
    result = library._model(root, record, protocol, data, episodes, 0.0, True)
    assert result["skill"] > 0
    assert result["key"]["training_sha256"] == digest(TRAIN)
    assert file_hash(lc.root / "state.json") == before


@pytest.mark.parametrize(
    "attack",
    [
        "retired",
        "prediction",
        "reuse_session",
        "candidate",
        "provenance",
        "threshold",
        "scope",
        "missing_state",
        "missing_numerical_evidence",
        "plugin_version",
        "symlink",
    ],
)
def test_numerical_or_session_tampering_fails(numerical, attack):
    root, record, protocol, data, episodes, lc = numerical
    if attack == "retired":
        lc.retire(record["model_id"], "Synthetic test")
    elif attack == "prediction":
        atomic_json(root / "episodes/cycle_001/workspace/predictions.json", [])
    elif attack == "reuse_session":
        episodes["episodes/cycle_001"]["session_id"] = "development"
    elif attack == "candidate":
        atomic_json(root / "episodes/cycle_000/workspace/candidate.json", {})
    elif attack == "provenance":
        atomic_json(root / "episodes/cycle_000/workspace/source_provenance.json", {"source_sha256": "0" * 64})
    elif attack == "scope":
        protocol["scope"] = {**SCOPE, "target": "unbound task"}
    elif attack == "plugin_version":
        policy = read(lc.root / "policy.json")
        policy["plugin_sha256"] = "0" * 64
        atomic_json(lc.root / "policy.json", policy)
    elif attack == "missing_numerical_evidence":
        (lc.root / "numerical_evidence").rename(lc.root / "saved_numeric_fixture")
    elif attack == "missing_state":
        (lc.root / "state.json").unlink()
    elif attack == "symlink":
        (lc.root / "alias").symlink_to(root / "episodes/cycle_000/workspace/candidate.json")
    with pytest.raises(ValueError):
        library._model(root, record, protocol, data, episodes, 0.2 if attack == "threshold" else 0.0, True)
    if attack == "missing_state":
        assert not (lc.root / "state.json").exists()


@pytest.fixture
def storage(tmp_path, monkeypatch):
    # Storage-only fixture: numerical and native proof gates are tested separately.
    root = tmp_path / "origin"
    root.mkdir()
    atomic_json(root / "goal.json", {"synthetic": True})
    atomic_json(root / "protocol.json", {"synthetic": True})
    model = {
        "plugin_id": "generated_python.v1",
        "scope": SCOPE,
        "training_sha256": digest(TRAIN),
        "recipe": {"source_code": CODE, "seed": 1234},
        "model_state": {"fixture": True},
    }
    verified = {
        "key": library.model_key(model, {"plugin_sha256": "a" * 64}),
        "packet": {"model": model, "model_id": "b" * 64},
        "skill": 0.5,
        "origin": {
            "goal_sha256": file_hash(root / "goal.json"),
            "protocol_sha256": file_hash(root / "protocol.json"),
            "model_id": "b" * 64,
            "evidence_kind": "synthetic_test_only",
        },
    }
    calls = []

    def verify(folder, model_id, *, replay_training):
        calls.append(replay_training)
        return copy.deepcopy(verified)

    monkeypatch.setattr(library, "validate_model_goal", verify)
    return library.ScientistModelLibrary(tmp_path / "library"), root, verified, calls


def test_publication_is_idempotent_across_restart_and_concurrency(storage):
    store, root, verified, calls = storage
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: store.publish(root, "b" * 64), range(8)))
    assert sum(r["created"] for r in results) == 1
    assert len({r["entry_id"] for r in results}) == 1
    reopened = library.ScientistModelLibrary(store.root)
    assert not reopened.publish(root, "b" * 64)["created"]
    cached = reopened.catalog()
    assert cached["indexed_entries"] == 1 and cached["verified_entries"] is None
    assert reopened.catalog(verify=True)["verified_entries"] == 1
    assert all(calls)


def test_seed_retraining_records_an_origin_without_increasing_model_count(storage):
    store, root, verified, _ = storage
    first = store.publish(root, "b" * 64)
    verified["packet"]["model"]["recipe"]["seed"] = 5
    verified["packet"]["model"]["model_state"] = {"another": True}
    second = store.publish(root, "b" * 64)
    assert first["entry_id"] == second["entry_id"]
    assert not second["created"]
    catalog = store.catalog()
    assert catalog["indexed_entries"] == 1
    assert catalog["entries"][0]["recorded_origins"] == 2
    assert catalog["entries"][0]["recorded_artifact_variants"] == 2


def test_comment_and_display_name_do_not_change_identity(storage):
    _, _, verified, _ = storage
    model = verified["packet"]["model"]
    key = library.model_key(model, {"plugin_sha256": "a" * 64})
    model["name"] = "A different display name"
    model["recipe"]["source_code"] = "# cosmetic change\n" + CODE
    model["recipe"]["seed"] += 1
    assert library.model_key(model, {"plugin_sha256": "a" * 64}) == key
    assert library.model_key(model, {"plugin_sha256": "c" * 64}) == key
    model["scope"] = {**SCOPE, "target": "different task"}
    assert library.model_key(model, {"plugin_sha256": "a" * 64}) != key


@pytest.mark.parametrize("attack", ["packet", "entry", "index", "origin", "live_origin", "retired"])
def test_resolve_revalidates_current_origin_and_immutable_objects(storage, monkeypatch, attack):
    store, root, verified, _ = storage
    published = store.publish(root, "b" * 64)
    if attack in {"packet", "entry", "index", "origin"}:
        pattern = {
            "packet": "packets/*.json",
            "entry": "entries/*.json",
            "index": "index/*.json",
            "origin": "origins/*/*.json",
        }[attack]
        p = next(store.root.glob(pattern))
        p.chmod(0o644)
        atomic_json(p, {"changed": True})
    elif attack == "live_origin":
        verified["origin"]["goal_sha256"] = "0" * 64
    else:

        def retired(*args, **kwargs):
            raise ValueError("Model retired")

        monkeypatch.setattr(library, "validate_model_goal", retired)
    with pytest.raises((ValueError, KeyError)):
        store.resolve(published["entry_id"])


def test_owner_namespace_and_symlinks_are_rejected(storage, tmp_path):
    store, root, _, _ = storage
    with pytest.raises(ValueError):
        store.goal(tmp_path.parent / "another_owner")
    with pytest.raises(ValueError):
        store.goal(store.root / "inside_library")
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError):
        store.goal(alias)


def test_library_inference_preserves_original_history_and_strips_labels(storage, monkeypatch):
    store, root, verified, _ = storage
    result = store.publish(root, "b" * 64)
    before = file_hash(root / "goal.json")
    observed = []

    class Plugin:
        def predict(self, model, rows, scope):
            observed.extend(rows)
            return [{"id": row["id"], "sequence": row["sequence"], "prediction": 0.25} for row in rows]

    monkeypatch.setattr(library, "get_model_plugin", lambda name: Plugin())
    output = store.predict(
        result["entry_id"], [{"id": "x", "sequence": "AA", "target": 1e9}], SCOPE, session_id="agent-chat"
    )
    assert "target" not in observed[0]
    assert output["receipt"]["mode"] == "science_agent"
    assert output["receipt"]["kind"] == "model_library_inference"
    assert file_hash(root / "goal.json") == before
    assert store.catalog()["indexed_entries"] == 1
    with pytest.raises(ValueError, match="applicability"):
        store.predict(result["entry_id"], [], {**SCOPE, "unit": "wrong"}, session_id="agent-chat")


@pytest.mark.parametrize(
    "attack", ["nonterminal", "agent_mode", "definition", "criterion", "archived_source", "data", "fake_provider"]
)
def test_completion_flags_alone_never_admit_a_model(tmp_path, attack):
    profile = Profile(
        "science_expert", "expert-kimi-code", ModelBinding("test_fixture", "model", "https://example.org/v1"), Budget()
    )
    store = GoalStore.create(
        tmp_path / "goal", GoalSpec("Synthetic test", ("accepted_model", "fresh_session_reuse")), profile
    )
    root = store.root
    atomic_json(root / "proof.json", {"synthetic": True})
    evidence = [{"path": "proof.json", "sha256": file_hash(root / "proof.json")}]

    def complete(s):
        s["status"] = "completed"
        s["assessment"] = {
            "criteria": {name: {"passed": True, "evidence": evidence} for name in s["definition"]["goal"]["criteria"]}
        }

    state = store.mutate(complete)
    atomic_json(root / "implementation/source.json", {})
    for phase, rows in (("training", TRAIN), ("validation", VALID), ("future", VALID)):
        atomic_json(root / f"data/{phase}.json", rows)
    protocol = {
        "implementation": {"source.json": file_hash(root / "implementation/source.json")},
        "data_hashes": {"training": digest(TRAIN), "validation": digest(VALID), "future": digest(VALID)},
    }
    atomic_json(root / "protocol.json", protocol)
    if attack == "nonterminal":
        state["status"] = "active"
    elif attack == "agent_mode":
        state["definition"]["condition"] = "persistent_agent_ablation"
        state["definition_sha256"] = digest(state["definition"])
    elif attack == "definition":
        state["definition"]["goal"]["objective"] = "changed"
    elif attack == "criterion":
        state["assessment"]["criteria"]["accepted_model"]["passed"] = False
    elif attack == "archived_source":
        atomic_json(root / "implementation/source.json", {"changed": True})
    elif attack == "data":
        atomic_json(root / "data/training.json", [])
    atomic_json(root / "goal.json", state)
    with pytest.raises(ValueError):
        library.validate_model_goal(root, "a" * 64)


def test_whole_goal_publication_retains_failures_and_does_not_claim_new_executions(storage, monkeypatch):
    store, root, _, _ = storage
    state = {
        "status": "completed",
        "inflight": None,
        "progress": {"generations": [{"model_id": c * 64} for c in "abc"]},
    }
    atomic_json(root / "goal.json", state)
    before = file_hash(root / "goal.json")

    def publish(folder, mid):
        if mid == "c" * 64:
            raise ValueError("Frozen model runtime differs")
        return {"created": mid == "a" * 64, "entry_id": mid}

    monkeypatch.setattr(store, "publish", publish)
    report = store.publish_goal(root)
    assert report["new_entries"] == report["duplicate_entries"] == report["rejected_entries"] == 1
    assert not report["all_published"] and report["new_autonomous_executions"] == 0
    assert file_hash(root / "goal.json") == before
    assert store._object("publication_attempts", report["report_id"])["outcomes"] == report["outcomes"]


@pytest.mark.parametrize("status", ["active", "budget_exhausted", "cancelled", "interrupted"])
def test_whole_goal_publication_rejects_noncompleted_goals(storage, status):
    store, root, _, _ = storage
    atomic_json(root / "goal.json", {"status": status, "inflight": None})
    with pytest.raises(ValueError, match="completed"):
        store.publish_goal(root)
    assert not (store.root / "publication_attempts").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("runner_name", ["run_science_goal", "run_science_evolution"])
@pytest.mark.parametrize(
    "action, terminal, reject",
    [
        ("run", "completed", False),
        ("resume", "completed", False),
        ("run", "budget_exhausted", False),
        ("status", "completed", False),
        ("pause", "active", False),
        ("run", "completed", True),
    ],
)
async def test_goal_runner_publishes_only_completed_execution(
    tmp_path, monkeypatch, runner_name, action, terminal, reject
):
    import importlib
    from pathlib import Path
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
    runner = importlib.import_module(runner_name)
    profile = Profile(
        "science_expert", "expert-kimi-code", ModelBinding("test_fixture", "model", "https://example.org/v1"), Budget()
    )
    store = GoalStore.create(tmp_path / "goal", GoalSpec("Synthetic wiring check", ("accepted_model",)), profile)

    def ready(s):
        s["status"] = "paused" if action == "resume" else terminal
        s["progress"] = {"generations": []}

    store.mutate(ready)
    atomic_json(store.root / "future_metrics.json", [])
    calls = []

    class Publication:
        def __init__(self, root):
            self.root = root

        def goal(self, folder):
            assert folder == store.root

        def publish_goal(self, folder):
            calls.append(folder)
            return {"all_published": not reject, "report_id": "a" * 64}

    async def execute(goal_store, cycle, evaluate, **kwargs):
        return goal_store.mutate(lambda s: s.update(status=terminal))

    monkeypatch.setattr(library, "ScientistModelLibrary", Publication)
    monkeypatch.setattr(runner, "credential_from_toml", lambda *args: "synthetic-credential")
    monkeypatch.setattr(runner, "run_goal", execute)
    monkeypatch.setattr(
        runner,
        "GeneratedModelGoal" if runner_name == "run_science_goal" else "EvolvingModelGoal",
        lambda *args: SimpleNamespace(cycle=None, evaluate=None),
    )
    args = SimpleNamespace(
        action=action, output=store.root, model_library=tmp_path / "library", credential_toml=tmp_path / "unused.toml"
    )
    if reject:
        with pytest.raises(ValueError, match="retained report"):
            await runner.main(args)
        assert store.read()["status"] == "completed"
    else:
        await runner.main(args)
    assert len(calls) == int(action in {"run", "resume"} and terminal == "completed")


def test_missing_numeric_evidence_is_not_recreated_by_verification(numerical):
    root, record, protocol, data, episodes, lc = numerical
    (lc.root / "numerical_evidence").rename(lc.root / "saved_numeric_fixture")
    with pytest.raises(ValueError):
        library._model(root, record, protocol, data, episodes, 0.0, False)
    assert not (lc.root / "numerical_evidence").exists()


@pytest.mark.parametrize("missing", ["catalog", "entries"])
def test_missing_experience_is_not_recreated_by_publication(tmp_path, missing):
    folder = tmp_path / "experience"
    folder.mkdir()
    if missing != "catalog":
        atomic_json(folder / "catalog.json", {"schema": 1, "entries": []})
    if missing != "entries":
        (folder / "entries").mkdir()
    before = sorted(str(p.relative_to(folder)) for p in folder.rglob("*"))
    with pytest.raises(ValueError):
        library._experience(tmp_path, "a" * 64)
    assert sorted(str(p.relative_to(folder)) for p in folder.rglob("*")) == before
