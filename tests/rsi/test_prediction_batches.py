"""Per-observation model review; synthetic programs are regression fixtures."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_generated_features import CODE, ROWS, SCOPE, lifecycle

from agent.rsi import generated_features as api
from agent.rsi.engine import prediction_batches as batches
from agent.rsi.engine.model_lifecycle import ModelLifecycle
from agent.rsi.engine.numerical_evidence import verify_assessment
from agent.rsi.engine.protein_model import digest
from agent.rsi.engine.storage import atomic_json

FIELD = "prediction_batch_review"
CONTRACT = {
    "schema": 1,
    "semantics": "independent_rows",
    "rtol": 1e-6,
    "atol": 1e-12,
    "probes": "reverse_sample_single_v1",
}
BATCH_DEPENDENT = CODE.replace(
    "return (np.asarray(features) @ np.asarray(model['weights'])).tolist()",
    "return ((np.asarray(features) @ np.asarray(model['weights'])) + (100 if len(features) == 1 else 0)).tolist()",
)


def assess(root, source, *, reviewed):
    original = lifecycle(root / "legacy-policy")
    policy = deepcopy(original.policy)
    if reviewed:
        policy.update(schema=2, **{FIELD: CONTRACT})
    controller = ModelLifecycle(root / "controller", policy)
    model = api.fit_candidate(ROWS[:60], SCOPE, {"source_code": source, "seed": 11})
    artifact = root / "candidate.json"
    atomic_json(artifact, model)
    identity = controller.submit(artifact, session_id="fixture", mode="science_expert")
    return controller, identity, model, controller.assess(identity, training=ROWS[:60], validation=ROWS[60:80])


def test_legacy_exact_replay_does_not_test_batch_dependence(tmp_path):
    controller, identity, model, report = assess(tmp_path, BATCH_DEPENDENT, reviewed=False)
    assert report["passed"] and report["prediction_replay"]["exact_match"]
    batch = api.predict(model, ROWS[60:80], SCOPE)
    single = api.predict(model, ROWS[60:61], SCOPE)
    assert single[0]["prediction"] - batch[0]["prediction"] == pytest.approx(100)
    assert FIELD not in report


def test_new_policy_rejects_batch_dependent_model_despite_perfect_fit(tmp_path):
    controller, identity, model, report = assess(tmp_path, BATCH_DEPENDENT, reviewed=True)
    assert report["metrics"]["rmse"] < 1e-9
    assert report["prediction_replay"]["exact_match"]
    assert not report["passed"] and not report[FIELD]["passed"]
    assert "prediction_batch_inconsistent" in report["rejection_reasons"]
    with pytest.raises(ValueError, match="validated"):
        controller.export(identity, tmp_path / "unqualified")


def test_new_policy_accepts_real_independent_model_and_exports(tmp_path):
    controller, identity, model, report = assess(tmp_path, CODE, reviewed=True)
    assert report["passed"] and report[FIELD]["passed"]
    packet = controller.export(identity, tmp_path / "qualified")
    assert packet["assessment"][FIELD] == report[FIELD]


def fixture_predictions():
    inputs = [{"id": str(i), "sequence": "A" + "ACDEFGHI"[i]} for i in range(8)]
    predictions = [{**row, "prediction": float(i + 1)} for i, row in enumerate(inputs)]
    return inputs, predictions


def test_probe_plan_is_bounded_and_covers_order_subset_and_single_row():
    assert [p["name"] for p in batches.probe_plan(8)] == ["reversed", "sampled", "single"]
    for count in range(1, 30):
        plan = batches.probe_plan(count)
        assert len(plan) <= 3
        assert len({tuple(p["indices"]) for p in plan}) == len(plan)
        assert all(0 <= i < count for p in plan for i in p["indices"])


def test_numeric_roundoff_is_tolerated_but_probabilities_and_classes_are_reviewed():
    reference = [{"id": "x", "prediction": 20000.0}]
    assert batches._compare(reference, [{"id": "x", "prediction": 20000.0 + 1e-11}], CONTRACT)["passed"]
    assert not batches._compare(reference, [{"id": "x", "prediction": 20100.0}], CONTRACT)["passed"]
    classifier = [{"id": "x", "prediction": 0, "probabilities": [0.6, 0.4]}]
    assert not batches._compare(classifier, [{"id": "x", "prediction": 0, "probabilities": [0.7, 0.3]}], CONTRACT)[
        "passed"
    ]
    assert not batches._compare(classifier, [{"id": "x", "prediction": 1, "probabilities": [0.6, 0.4]}], CONTRACT)[
        "passed"
    ]
    assert not batches._compare(reference, [{"id": "other", "prediction": 20000.0}], CONTRACT)["passed"]
    assert not batches._compare(reference, [{"id": "x", "prediction": True}], CONTRACT)["passed"]


def test_probe_binding_detects_missing_rows_labels_and_changed_plan():
    inputs, predictions = fixture_predictions()
    by_id = {p["id"]: p for p in predictions}
    plugin = SimpleNamespace(predict=lambda model, rows, scope: [by_id[r["id"]] for r in rows])
    payload = batches.collect(plugin, {}, inputs, {})
    assert batches.summarize(predictions, payload, CONTRACT, "generated_python.v1")["passed"]
    for change in ("remove", "indices", "labels", "hash"):
        altered = deepcopy(payload)
        if change == "remove":
            altered["probes"].pop()
        if change == "indices":
            altered["probes"][0]["indices"].reverse()
        if change == "labels":
            altered["inputs"][0]["target"] = 1234
        if change == "hash":
            altered["probes"][0]["inputs_sha256"] = "wrong"
        with pytest.raises(ValueError):
            batches.summarize(predictions, altered, CONTRACT, "generated_python.v1")


def test_probe_errors_and_model_mutation_are_rejections_without_private_messages():
    inputs, predictions = fixture_predictions()
    original = {"weight": 1}

    def mutating(model, rows, scope):
        model["weight"] = 2
        return []

    payload = batches.collect(SimpleNamespace(predict=mutating), original, inputs, {})
    assert original == {"weight": 1}
    assert all(p["error"] == "model_state_changed" for p in payload["probes"])
    assert not batches.summarize(predictions, payload, CONTRACT, "generated_python.v1")["passed"]

    def failing(*args):
        raise ValueError("private training content must not appear in reports")

    payload = batches.collect(SimpleNamespace(predict=failing), original, inputs, {})
    assert "private" not in str(payload)
    assert not batches.summarize(predictions, payload, CONTRACT, "generated_python.v1")["passed"]


@pytest.mark.parametrize("damage", ["file", "summary", "removed_summary", "removed_evidence"])
def test_batch_evidence_cannot_be_tampered_with_or_removed_before_export(tmp_path, damage):
    controller, identity, model, report = assess(tmp_path, CODE, reviewed=True)
    if damage == "file":
        folder = controller.root / "numerical_evidence/versions" / report["numerical_evidence"]["version_id"]
        folder.chmod(0o755)
        path = folder / "batch_predictions.json"
        payload = json.loads(path.read_text())
        payload["probes"][0]["predictions"][0]["prediction"] += 1
        atomic_json(path, payload)
    else:
        if damage == "summary":
            report[FIELD]["probes_sha256"] = "0" * 64
        if damage == "removed_summary":
            report.pop(FIELD)
        if damage == "removed_evidence":
            report.pop("numerical_evidence")
        atomic_json(controller.root / f"assessment_{identity}.json", report)
        state = controller._state()
        state["candidates"][identity]["assessment_sha256"] = digest(report)
        atomic_json(controller.root / "state.json", state)
    with pytest.raises(ValueError):
        controller.export(identity, tmp_path / "tampered")
    assert not (tmp_path / "tampered").exists()


def test_new_policy_is_predeclared_and_cannot_be_claimed_by_legacy_evidence(tmp_path):
    controller, identity, model, report = assess(tmp_path, CODE, reviewed=False)
    report[FIELD] = {"passed": True}
    with pytest.raises(ValueError, match="Legacy assessment"):
        verify_assessment(controller.root, identity, model, controller.policy, report)
    old_policy = controller.policy
    assert batches.apply_policy(old_policy, {}) == old_policy
    protocol = {FIELD: batches.contract()}
    new_policy = batches.apply_policy(old_policy, protocol)
    batches.require_bound_policy(new_policy, protocol)
    with pytest.raises(ValueError):
        batches.require_bound_policy(old_policy, protocol)
    with pytest.raises(ValueError):
        batches.require_bound_policy(new_policy, {})
    with pytest.raises(ValueError):
        batches.validate_contract({**CONTRACT, "rtol": 1.0})
    with pytest.raises(ValueError):
        batches.validate_contract({**CONTRACT, "schema": True})
    batches.verify_goal({"definition": deepcopy(protocol)}, protocol)
    with pytest.raises(ValueError):
        batches.verify_goal({"definition": deepcopy(protocol)}, {})


def test_contract_is_disclosed_only_for_new_development_prompts():
    original = "Frozen original prompt"
    protocol = {FIELD: batches.contract()}
    assert batches.append_prompt(original, {}, "develop") == original
    assert batches.append_prompt(original, protocol, "reuse") == original
    current = batches.append_prompt(original, protocol, "develop")
    assert current.startswith(original) and "independent result per observation" in current
    assert "rtol=1e-6" in current and "same-batch replay must remain exact" in current


def test_review_is_frozen_in_locked_goal_definition_before_execution(tmp_path):
    from test_science_goal import create

    store = create(tmp_path)
    review = batches.freeze_goal(store)
    state = store.read()
    assert state["definition"][FIELD] == review == CONTRACT
    assert state["definition_sha256"] == digest(state["definition"])
    review["rtol"] = 1.0
    assert store.read()["definition"][FIELD] == CONTRACT
    assert batches.freeze_goal(store) == CONTRACT


@pytest.mark.parametrize("change", ["cycle", "inflight", "usage", "paused"])
def test_review_cannot_be_added_after_execution_starts(tmp_path, change):
    from test_science_goal import create

    store = create(tmp_path)

    def start(state):
        if change == "cycle":
            state["cycles"].append({"index": 0})
        elif change == "inflight":
            state["inflight"] = {"index": 0}
        elif change == "usage":
            state["usage"]["model_calls"] = 1
        else:
            state["status"] = "paused"

    store.mutate(start)
    before = (store.root / "goal.json").read_bytes()
    with pytest.raises(ValueError, match="before execution"):
        batches.freeze_goal(store)
    assert (store.root / "goal.json").read_bytes() == before
