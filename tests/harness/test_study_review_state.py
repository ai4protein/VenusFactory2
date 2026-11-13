"""Interim factual consistency is distinct from scientific usefulness.

Synthetic receipts exercise the contract; the coordinator test fits a real
isolated numerical fixture. No external model or biological claim is involved.
"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_autonomous_research import profile
from test_sc_study_evidence import add_measurement, request

from agent.harness import autonomous_research as ar
from agent.harness import scientific_acceptance as sc
from agent.harness import study_review_state as study
from agent.harness.research_development import Development
from agent.harness.scientist_role_output import RoleOutputError, specification, validate_output


def packet(tmp_path):
    development = Development(tmp_path / "development")
    measured, path = add_measurement(development)
    context = sc.evidence_context({"objective": "Develop a useful synthetic predictor"}, [], development)
    return context, measured, path


def review(context, requests):
    return {
        "decision": "needs_revision",
        "summary": "Further research is required.",
        "issues": ["Review the unresolved scientific work."],
        "evidence_ids": [row["id"] for row in context["development_measurements"]["experiments"]],
        "revision_requests": requests,
    }


@pytest.mark.parametrize("requirement", ["data_partitions", "validation_measurement", "measured_baseline"])
@pytest.mark.parametrize("specific", [False, True])
def test_measured_baseline_cannot_be_reported_as_missing(tmp_path, requirement, specific):
    context, measured, path = packet(tmp_path)
    original = path.read_bytes()
    value = review(context, [request(requirement, identity=measured["id"] if specific else "")])
    with pytest.raises(study.StudyReviewInconsistent, match="already recorded|study-wide"):
        sc.parse_study_review(json.dumps(value), context)
    result = sc.assess(value, context)
    assert not result["passed"] and not result["approved_models"]
    assert "review_evidence_inconsistent" in " ".join(result["issues"])
    assert path.read_bytes() == original
    state = context["measurement_state"]["experiments"][0]
    assert state["validation_completed"] and state["partitions_recorded"]
    assert state["validation"]["score"] == 0.5


@pytest.mark.parametrize("requirement", ["measured_revision", "model_submission"])
def test_actual_missing_revision_or_submission_remains_actionable(tmp_path, requirement):
    context, _, _ = packet(tmp_path)
    value = review(context, [request(requirement)])
    assert sc.parse_study_review(json.dumps(value), context) == value
    result = sc.assess(value, context)
    assert not result["passed"] and result["approved_models"] == {}


def test_scientific_rejection_can_question_utility_and_split_quality(tmp_path):
    context, measured, _ = packet(tmp_path)
    concern = request("scientific_value", kind="scientific_concern", identity=measured["id"])
    concern.update(reason="The recorded split does not establish out-of-family generalization.",
                   proposed_action="Design a grouped transfer analysis; preserve the recorded baseline.")
    value = review(context, [concern])
    assert sc.parse_study_review(json.dumps(value), context) == value
    result = sc.assess(value, context)
    assert not result["passed"]
    result.update(receipt="episodes/cycle_001/scientific_acceptance.json", sha256="a" * 64)
    feedback = sc.revision_feedback(result, context, value, 1)
    assert feedback["measurement_state"] == context["measurement_state"]
    assert feedback["reviewer_commentary"]["summary"] == value["summary"]
    assert feedback["summary"] != value["summary"]
    assert feedback["issues"] == ["Address the scientific concern"]
    assert feedback["study_revision_requests"] == [concern]
    assert feedback["requests"] == []  # Not a rejected submitted model.


@pytest.mark.parametrize("protocol", ["chat_completions", "responses"])
def test_new_contract_is_versioned_without_reinterpreting_historical_reviews(tmp_path, protocol):
    context, _, _ = packet(tmp_path)
    old_context = copy.deepcopy(context)
    old_context["scientific_acceptance"].pop("study_review_policy")
    old_context.pop("measurement_state")
    old_review = review(old_context, [])
    old_review.pop("revision_requests")
    assert sc.parse_study_review(json.dumps(old_review), old_context) == old_review
    old_result = sc.assess(old_review, old_context)
    assert old_result == {
        "policy": sc.POLICY, "passed": False,
        "issues": [*old_review["issues"], old_review["summary"]],
        "approved_models": {}, "model_assessments": [],
    }
    with pytest.raises(RoleOutputError):
        sc.parse_study_review(json.dumps(old_review), context)
    legacy = specification("SC", protocol, study_acceptance=True)
    modern = specification("SC", protocol, study_acceptance=True, study_review_policy=study.POLICY)
    assert "revision_requests" not in legacy["schema"]["properties"]
    assert "revision_requests" in modern["schema"]["required"]
    model = specification("SC", protocol, model_acceptance=True, study_review_policy=study.POLICY)
    assert model["schema"] == sc.model_schema()
    value = review(context, [request()])
    assert json.loads(validate_output(json.dumps(value), modern)) == value


@pytest.mark.parametrize("fault", ["unknown_experiment", "wrong_kind", "future_review", "empty_rejection", "ready_with_requests", "unrecorded_failure"])
def test_unsupported_or_inconsistent_requests_never_become_valid_actions(tmp_path, fault):
    context, _, _ = packet(tmp_path)
    value = review(context, [request()])
    if fault == "unknown_experiment":
        value["revision_requests"][0]["experiment_id"] = "f" * 32
    elif fault == "wrong_kind":
        value["revision_requests"][0]["kind"] = "scientific_concern"
    elif fault == "future_review":
        value["revision_requests"][0]["requirement"] = "independent_review"
    elif fault == "empty_rejection":
        value["revision_requests"] = []
    elif fault == "ready_with_requests":
        value.update(decision="ready_for_host_validation", issues=[])
    else:
        value["revision_requests"] = [request("execution_failure", kind="execution_failure")]
    with pytest.raises(ValueError):
        sc.parse_study_review(json.dumps(value), context)


def test_model_claims_cannot_replace_measurement_state_and_bounded_history_is_not_absence(tmp_path):
    context, _, _ = packet(tmp_path)
    context["measurement_state"]["experiments"][0]["validation_completed"] = False
    with pytest.raises(study.StudyReviewInconsistent, match="does not match"):
        sc.parse_study_review(json.dumps(review(context, [request()])), context)
    context["development_measurements"]["omitted_count"] = 1
    context["measurement_state"] = study.measurement_state(context)
    with pytest.raises(study.StudyReviewInconsistent, match="bounded history"):
        sc.parse_study_review(json.dumps(review(context, [request()])), context)


@pytest.mark.asyncio
async def test_inconsistent_review_uses_one_factual_repair_without_forcing_approval(tmp_path):
    context, measured, path = packet(tmp_path)
    original = path.read_bytes()
    calls = []

    async def consult(role, current):
        calls.append(current)
        request_value = request("validation_measurement", identity=measured["id"]) if len(calls) == 1 else request(
            "scientific_value", kind="scientific_concern", identity=measured["id"])
        return json.dumps(review(context, [request_value]))

    runner = ar.AutonomousResearch(SimpleNamespace(root=tmp_path), "unused", {}, consult=consult)
    value = await runner.structured("SC", context, profile(), SimpleNamespace(calls=0),
                                    tmp_path / "review", lambda raw: sc.parse_study_review(raw, context))
    assert len(calls) == 2 and value["decision"] == "needs_revision"
    assert "Correct the typed missing-evidence assertions" in calls[1]["format_repair"]["instruction"]
    assert calls[1]["measurement_state"] == context["measurement_state"]
    assert path.read_bytes() == original
    assert not sc.assess(value, context)["passed"]


@pytest.mark.asyncio
async def test_exhausted_inconsistent_review_preserves_fit_and_does_not_instruct_pi_to_retrain(tmp_path, monkeypatch):
    store = ar.create(tmp_path / "goal", profile(), "Develop a useful numerical model",
                      {"max_cycles": 2, "model_calls": 36, "tool_calls": 20, "wall_seconds": 180})
    from test_training_attempt_resolution import table

    from agent.harness.research_data_store import store_for_goal

    monkeypatch.setenv("VENUS_JOB_QUEUE_ROOT", str(tmp_path / "queue"))
    store_for_goal(store.root).write_evaluation("data.json", json.dumps(table()).encode())
    stale = {"summary": "No earlier measurement", "issues": ["stale-missing-baseline"],
             "requests": [], "study_revision_requests": [request("measured_baseline")]}
    store.mutate(lambda state: state["progress"].update(sc_revision=stale))
    sc_calls = 0
    pi_calls = []
    measured = []

    async def consult(role, context):
        nonlocal sc_calls
        if role == "PI":
            pi_calls.append(copy.deepcopy(context))
            return "Inspect the recorded model and propose a justified improvement."
        if role == "CB":
            return json.dumps({"summary": "Measure and improve a predictor", "deliverable": "model", "needs_input": [],
                               "steps": [{"id": "fit", "title": "Fit a baseline", "acceptance": "A measured numerical predictor",
                                          "verification": "model"}]})
        assert role == "SC"
        sc_calls += 1
        if sc_calls <= 2:
            item = request("validation_measurement", identity=measured[0]["experiment"]["id"])
            item["reason"] = "incorrect-baseline-pending-claim"
        else:
            item = request("scientific_value", kind="scientific_concern")
        return json.dumps(review(context, [item]))

    async def executor(task, service, events):
        if not measured:
            await service.write_research_file("fit.py", "def fit(training,seed):\n rows=training['train']\n return {'slope':sum(r['target']/r['inputs']['x'] for r in rows)/len(rows)}\ndef predict(model,inputs):\n return {'value':model['slope']*inputs['x']}\n")
            measured.append(await service.train_research_model("Linear fixture", "fit.py", "data.json", "Measure a numerical baseline"))
        await service.submit_research_checkpoint("Baseline measured; scientific revision remains",
                                                 ["fit"], ["experiments/" + measured[0]["experiment"]["id"] + ".json"], [])
        events.append({"kind": "completed", "text": "Recorded baseline evidence", "event_audit": {"passed": True}})

    class Relay:
        calls = 0
        base_url = "http://unused"
        token = "fixture"
        def __init__(self, **kwargs):
            Path(kwargs["output"]).touch()
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(ar, "EvidenceGateway", Relay)
    monkeypatch.setattr(ar, "audit_requests", lambda *args: {"passed": True, "model_calls": 4})
    state = await ar.AutonomousResearch(store, "fixture", {}, consult=consult, executor=executor, tools=[]).run()
    assert len(measured) == 1 and len(pi_calls) == 2 and sc_calls == 3
    next_progress = pi_calls[1]["progress"]
    assert next_progress["review_evidence_error"]["code"] == "review_evidence_inconsistent"
    initial_revision = pi_calls[0]["progress"]["sc_revision"]
    assert initial_revision["issues"] == stale["issues"]
    assert initial_revision["requests"] == stale["requests"]
    assert initial_revision["study_revision_requests"] == stale["study_revision_requests"]
    assert "summary" not in initial_revision  # Typed obligations survive; prior reviewer prose does not.
    assert "sc_revision" not in next_progress
    assert "stale-missing-baseline" not in json.dumps(next_progress)
    assert "incorrect-baseline-pending-claim" not in json.dumps(next_progress)
    assert next_progress["measurement_state"]["experiments"][0]["validation_completed"]
    assert next_progress["development"]["generations"][0]["id"] == measured[0]["experiment"]["id"]
    error = store.root / "episodes/cycle_000/review_evidence_error.json"
    assert json.loads(error.read_text())["code"] == "review_evidence_inconsistent"
    assert not (store.root / "episodes/cycle_000/review.json").exists()
    assert not state["progress"]["research_completed"] and not state["progress"].get("deliveries")
    assert len(Development(store.root / "development").records()) == 1


@pytest.mark.parametrize("requirement,index", [("measured_baseline", 2), ("measured_revision", 1), ("model_submission", 1)])
def test_study_wide_missing_evidence_cannot_select_an_unrelated_experiment(tmp_path, requirement, index):
    development = Development(tmp_path / "development")
    add_measurement(development, 1)
    add_measurement(development, 2)
    context = sc.evidence_context({"objective": "Develop a useful synthetic predictor"}, [], development)
    value = review(context, [request(requirement, identity=f"{index:032x}")])
    with pytest.raises(study.StudyReviewInconsistent, match="study-wide"):
        sc.parse_study_review(json.dumps(value), context)
    if requirement != "model_submission":
        value["revision_requests"][0]["experiment_id"] = ""
        with pytest.raises(study.StudyReviewInconsistent, match="already recorded"):
            sc.parse_study_review(json.dumps(value), context)


@pytest.mark.parametrize("policy", ["unknown.v999", ""])
def test_unknown_study_policy_is_not_treated_as_a_legacy_receipt(tmp_path, policy):
    context, _, _ = packet(tmp_path)
    context["scientific_acceptance"]["study_review_policy"] = policy
    value = review(context, [])
    value.update(decision="ready_for_host_validation", issues=[])
    with pytest.raises(ValueError, match="Unsupported|Unknown"):
        sc.parse_study_review(json.dumps(value), context)
    result = sc.assess(value, context)
    assert not result["passed"] and not result["approved_models"]
    assert "Unsupported study review policy" in " ".join(result["issues"])
