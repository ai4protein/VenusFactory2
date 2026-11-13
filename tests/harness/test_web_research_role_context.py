"""Offline regression for the actual Web Scientist role path, not only expert-team."""
import copy
import json
from types import SimpleNamespace

import pytest

from agent.harness.web_research_role_context import project_role_context
from agent.harness.web_native_worker import native_prompt
from agent.harness.autonomous_research import AutonomousResearch
from agent.harness.role_context import POLICY


def task():
    return {
        "objective": "Predict the supplied target; preserve all patient-free measurement units.",
        "PI": "PI_FULL_TRANSCRIPT",
        "previous_episodes": [{"summary": "PREVIOUS_SC_ENDORSEMENT"}],
        "messages": [{"role": "assistant", "content": "CHAT_TRANSCRIPT"}],
        "memory": "CHAT_MEMORY",
        "progress": {"user_updates": ["Use the revised 20 nm acceptance threshold"],
                     "last_review": {"summary": "PRIOR_SC_VERDICT"},
                     "inputs": [{"path": "inputs.csv", "sha256": "checked"}]},
        "plan": {"summary": "CB_FULL_TRANSCRIPT", "deliverable": "analysis", "needs_input": [],
                 "steps": [{"id": "measure", "title": "Measure actual rows", "acceptance": "20 nm tolerance",
                            "verification": "execution", "approach": "CB_METHOD_ENDORSEMENT",
                            "status": "pending", "evidence": []}]},
        "model_development": {"prepared_dataset": {"id": "dataset-1"}},
    }


def test_all_roles_receive_detached_inputs_and_explicit_handoffs_only():
    original = task()
    before = copy.deepcopy(original)
    pi = project_role_context("PI", original)
    cb = project_role_context("CB", original)
    mls = project_role_context("MLS", original)
    for role, value in [("PI", pi), ("CB", cb), ("MLS", mls)]:
        assert value["role_context"] == {"policy": POLICY, "role": role, "fresh_session": True}
        assert value["objective"] == original["objective"]
        assert value["user_updates"] == original["progress"]["user_updates"]
        assert not {"PI", "messages", "memory", "previous_episodes"} & value.keys()
        assert "PRIOR_SC_VERDICT" not in json.dumps(value)
    assert cb["handoffs"]["PI"]["direction"] == original["PI"]
    assert "PI_FULL_TRANSCRIPT" not in json.dumps(pi)
    assert "PI_FULL_TRANSCRIPT" not in json.dumps(mls)
    assert mls["plan"]["steps"][0]["acceptance"] == "20 nm tolerance"
    cb["progress"]["inputs"][0]["path"] = "mutated"
    assert original == before
    assert pi["progress"]["inputs"][0]["path"] == "inputs.csv"


def test_sc_keeps_original_measurements_unknown_fields_and_contract_not_endorsements():
    context = task()
    context.update(user_updates=context["progress"]["user_updates"],
                   execution={"checkpoint": {"summary": "MLS_SELF_RATING", "completed_steps": ["measure"],
                                             "artifacts": [{"path": "result.json", "sha256": "actual"}]},
                              "receipts": [{"tool": "measurement", "result": {"unknown_future_metric": 0.0,
                                            "missing_value": None, "valid": False, "unit": "nm"}}]},
                   scientific_acceptance={"models": [], "policy": "scientific-acceptance.v1"},
                   measured_progress={"evaluated_experiment_ids": ["experiment-1"]})
    before = copy.deepcopy(context)
    result = project_role_context("SC", context)
    text = json.dumps(result)
    for marker in ["PI_FULL_TRANSCRIPT", "CB_FULL_TRANSCRIPT", "CB_METHOD_ENDORSEMENT", "MLS_SELF_RATING",
                   "CHAT_TRANSCRIPT", "CHAT_MEMORY", "PRIOR_SC_VERDICT"]:
        assert marker not in text
    assert result["execution"]["receipts"] == context["execution"]["receipts"]
    assert result["execution"]["checkpoint"]["completed_steps"] == ["measure"]
    assert result["plan"]["steps"][0]["acceptance"] == "20 nm tolerance"
    assert result["user_updates"] == context["user_updates"]
    assert result["scientific_acceptance"] == context["scientific_acceptance"]
    assert context == before


def test_native_research_starts_fresh_but_agent_retains_conversation_memory():
    request = {"text": json.dumps(project_role_context("MLS", task())),
               "messages": [{"role": "assistant", "content": "CHAT_TRANSCRIPT"}], "memory": "CHAT_MEMORY"}
    research = native_prompt(request, research=True)
    agent = native_prompt(request)
    assert "fresh MLS research role session" in research
    assert "CHAT_TRANSCRIPT" not in research and "CHAT_MEMORY" not in research
    assert "Continue the SAME conversation" not in research
    assert "CHAT_TRANSCRIPT" in agent and "CHAT_MEMORY" in agent
    assert "Continue the SAME conversation" in agent


def test_continuation_drops_previous_answer_but_keeps_receipts_and_user_changes():
    source = project_role_context("MLS", task())
    source["execution_followup"] = {"previous_response": "EXECUTOR_PREVIOUS_ANSWER", "completed_usage": {"tool_calls": 2}}
    source["execution_receipts"] = [{"result": {"measurement": 4, "unknown_field": "retained"}}]
    output = project_role_context("MLS", source)
    assert "EXECUTOR_PREVIOUS_ANSWER" not in json.dumps(output)
    assert output["execution_receipts"] == source["execution_receipts"]
    assert output["user_updates"] == source["user_updates"]
    assert "previous_response" in source["execution_followup"]


@pytest.mark.asyncio
async def test_actual_consult_entry_projects_before_override_and_appends_private_audit(tmp_path):
    seen = []
    async def consult(role, context):
        seen.append((role, context))
        return "response"
    runner = AutonomousResearch(SimpleNamespace(), "unused", {}, consult=consult)
    original = task()
    for role in ["PI", "CB", "SC"]:
        assert await runner.consult(role, original, None, None, tmp_path) == "response"
    assert len(list((tmp_path / "role_contexts").glob("*.json"))) == 3
    assert all(value["role_context"]["role"] == role for role, value in seen)
    assert "handoffs" not in seen[-1][1]
    assert original["PI"] == "PI_FULL_TRANSCRIPT"


@pytest.mark.parametrize("role", ["PI", "CB", "MLS", "SC"])
def test_large_user_goal_preserves_final_requirement_without_an_implicit_limit(role):
    original = task()
    original["objective"] = "x" * 300000 + "FINAL USER REQUIREMENT"
    projected = project_role_context(role, original)
    assert projected["objective"] == original["objective"]
    assert projected["objective"].endswith("FINAL USER REQUIREMENT")
    assert not {"messages", "memory", "previous_episodes"} & projected.keys()
    assert "PRIOR_SC_VERDICT" not in json.dumps(projected)


def test_replanning_preserves_typed_revision_obligations_not_critic_narrative():
    source = task()
    source["progress"]["sc_revision"] = {
        "summary": "SC_COMPLETE_NARRATIVE", "requests": [{"candidate_id": "c1", "verdict": "not_useful",
        "revision_request": "Measure uncertainty on the existing split"}], "experiment_ids": ["e1"]}
    for role in ["PI", "CB"]:
        result = project_role_context(role, source)
        assert result["progress"]["sc_revision"]["requests"] == source["progress"]["sc_revision"]["requests"]
        assert "SC_COMPLETE_NARRATIVE" not in json.dumps(result)
    assert "sc_revision" not in project_role_context("MLS", source)["progress"]
    assert "sc_revision" not in json.dumps(project_role_context("SC", source))


def test_real_plan_only_user_revision_survives_each_role_and_sc_evidence_boundary(tmp_path):
    from agent.harness.autonomous_research import create
    from agent.harness import Profile
    from agent.harness.research_updates import queue_revision, apply_revisions
    from agent.harness.web_research_role_context import user_plan_revisions
    from agent.harness import scientific_acceptance
    profile = Profile.from_dict({"mode": "science_expert", "harness": "expert-deepseek-harness",
        "model": {"provider": "fixture", "model": "offline", "base_url": "http://127.0.0.1:1"}})
    store = create(tmp_path / "goal", profile, "Preserve the original optical objective",
                   {"max_cycles": 2, "model_calls": 12, "tool_calls": 12, "wall_seconds": 120})
    changed = {"summary": "USER_ONLY_CONSTRAINT: keep nanometres, all negative measurements, and the original held-out split",
               "steps": [{"id": "validate", "title": "Validate measured rows", "acceptance": "Retain uncertainty"}]}
    queue_revision(store, text="", plan=changed)
    state = apply_revisions(store)
    assert not state["progress"].get("user_updates")
    context = {"objective": state["definition"]["goal"]["objective"], "progress": state["progress"],
               "plan": state["progress"]["plan"]}
    for role in ["PI", "CB", "MLS"]:
        projected = project_role_context(role, context)
        assert projected["user_plan_revisions"][0]["plan"] == changed
        assert "previous_plan" not in json.dumps(projected["user_plan_revisions"])
        assert projected["objective"] == context["objective"]
        if role == "MLS":
            assert project_role_context("MLS", projected)["user_plan_revisions"] == projected["user_plan_revisions"]
    development = SimpleNamespace(records=lambda: [])
    context["user_plan_revisions"] = user_plan_revisions(context)
    context.update(criteria=["keep exact measured endpoint"], constraints={"data": "no synthetic replacements"},
                   acceptance_criteria={"unit": "nm"}, user_requirements=["keep all measured negatives"])
    critic = scientific_acceptance.evidence_context(context, [], development)
    projected = project_role_context("SC", critic)
    assert projected["user_plan_revisions"][0]["plan"]["summary"] == changed["summary"]
    assert projected["plan"]["steps"][0]["acceptance"] == "Retain uncertainty"
    assert projected["objective"] == context["objective"]

    for key in ("criteria", "constraints", "acceptance_criteria", "user_requirements"):
        assert projected[key] == context[key]


def test_typed_study_revision_and_host_measurement_state_survive_replanning():
    context = task()
    revision = {"study_revision_requests": [{"kind": "missing_evidence", "requirement": "candidate_validation",
                                           "experiment_ids": ["e1"]}],
                "measurement_state": {"baseline_measured": True, "candidate_measured": False},
                "reviewer_commentary": {"summary": "REVIEWER_FULL_COMMENTARY"},
                "summary": "SC_FULL_SUMMARY"}
    context["progress"]["sc_revision"] = revision
    for role in ["PI", "CB"]:
        output = project_role_context(role, context)["progress"]["sc_revision"]
        assert output["study_revision_requests"] == revision["study_revision_requests"]
        assert output["measurement_state"] == revision["measurement_state"]
        assert "reviewer_commentary" not in output and "summary" not in output


def test_technical_admission_repairs_survive_without_review_narrative():
    source = task()
    review = {"id": "candidate-1", "name": "Length tool", "status": "needs_revision",
              "issues": ["Document applicability more clearly"], "summary": "TECHNICAL_REVIEW_FULL_NARRATIVE",
              "receipt": "admission/candidate-1.json", "sha256": "verified-hash",
              "proposal_file": "proposal.json", "review_stage": "independent"}
    source["progress"]["publication_reviews"] = [review]
    source["progress"]["review_submission"] = {"proposal_file": "proposal.json"}
    for role in ["PI", "CB", "MLS"]:
        projected = project_role_context(role, source)
        saved = projected["progress"]["publication_reviews"][0]
        assert saved == {key: value for key, value in review.items() if key != "summary"}
        assert projected["progress"]["review_submission"]["proposal_file"] == "proposal.json"
        assert "TECHNICAL_REVIEW_FULL_NARRATIVE" not in json.dumps(projected)
        if role == "MLS":
            assert project_role_context("MLS", projected)["progress"]["publication_reviews"] == [saved]
    assert "publication_reviews" not in json.dumps(project_role_context("SC", source))
    assert source["progress"]["publication_reviews"][0] == review
