"""Analysis SC wiring preserves model/study precedence and rechecks host bindings."""
import copy
import json

import pytest
import test_analysis_report_review as analysis_cases

from agent.harness import analysis_report_review as analysis
from agent.harness import scientific_acceptance as sc
from agent.harness.scientist_role_output import RoleOutputError, specification, validate_output

make_case = analysis_cases.make_case


def packet(context, proposals=None, development=None):
    context = copy.deepcopy(context)
    context["plan"]["deliverable"] = "analysis"
    return sc.evidence_context(context, proposals or [], development)


def test_actual_analysis_capture_selects_typed_policy_and_fresh_packet(make_case):
    _, _, context, _ = make_case()
    context["execution"]["report"] = "Executor endorsement must not be treated as SC truth."
    selected = packet(context)
    assert selected["scientific_acceptance"]["analysis_review_policy"] == analysis.POLICY
    assert sc.prompt_for_review(selected) == analysis.PROMPT
    assert "report" not in selected["execution"]
    assert selected["analysis_evidence"] == context["analysis_evidence"]
    assert selected["analysis_evidence"] is not context["analysis_evidence"]


@pytest.mark.parametrize("protocol", ["chat_completions", "responses"])
def test_analysis_role_output_schema_is_wired_for_both_protocols(make_case, protocol):
    _, _, context, review = make_case()
    selected = packet(context)
    spec = specification("SC", protocol,
                         analysis_review_policy=selected["scientific_acceptance"]["analysis_review_policy"])
    assert spec["schema"] == analysis.schema()
    assert json.loads(validate_output(json.dumps(review), spec)) == review
    parameters = spec["tool"]["function"]["parameters"] if protocol == "chat_completions" else spec["tool"]["parameters"]
    assert parameters["additionalProperties"] is False
    assert "step_assessments" in parameters["required"]
    assert analysis.parse_review(json.dumps(review), selected) == review
    legacy = {"decision": "ready_for_host_validation", "summary": "Done", "issues": []}
    with pytest.raises(RoleOutputError):
        validate_output(json.dumps(legacy), spec)


def test_assess_revalidates_valid_claims_instead_of_trusting_a_ready_string(make_case):
    _, _, context, review = make_case()
    selected = packet(context)
    result = sc.assess(review, selected)
    assert result["passed"] is True
    assert result["approved_models"] == {}
    bad = copy.deepcopy(review)
    bad["claims"][0]["result_pointer"] = "/invented"
    result = sc.assess(bad, selected)
    assert result["passed"] is False
    assert any("analysis_review_inconsistent" in item for item in result["issues"])


@pytest.mark.parametrize("fault", ["old_schema", "changed_preview", "wrong_value", "wrong_subject", "policy_version"])
def test_assess_fails_closed_on_invalid_analysis_ready(make_case, fault):
    _, _, context, review = make_case()
    selected = packet(context)
    if fault == "old_schema":
        review = {"decision": "ready_for_host_validation", "summary": "Done", "issues": []}
    elif fault == "changed_preview":
        selected["artifact_previews"][0]["content"] += "changed"
    elif fault == "wrong_value":
        review["claims"][0]["display_value"] = "99"
    elif fault == "wrong_subject":
        selected["analysis_evidence"]["computations"][0]["sequence_relations"]["records"] = []
    else:
        selected["scientific_acceptance"]["analysis_review_policy"] = "unknown-policy"
    result = sc.assess(review, selected)
    assert result["passed"] is False
    assert any("analysis_review_inconsistent" in item for item in result["issues"])


def test_model_deliverable_is_not_downgraded_to_analysis(make_case):
    _, _, context, _ = make_case()
    context["plan"]["deliverable"] = "model"
    selected = sc.evidence_context(context, [], None)
    assert "analysis_review_policy" not in selected["scientific_acceptance"]
    assert sc.prompt_for_review(selected) == sc.PROMPT


def test_submitted_model_even_with_invalid_binding_keeps_model_review(make_case, tmp_path):
    _, _, context, _ = make_case()
    bundle = tmp_path / "model"
    bundle.mkdir()
    # Deliberately invalid model: routing must retain its model binding errors,
    # never replace them with an unrelated successful analysis.
    manifest = {"id": "invalid-model-id", "name": "Candidate", "description": "Fixture",
                "review_spec": {"kind": "model", "scope": {}, "input_schema": {}, "output_schema": {}}}
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    selected = packet(context, [bundle])
    assert selected["scientific_acceptance"]["models"][0]["binding_errors"]
    assert "analysis_review_policy" not in selected["scientific_acceptance"]
    assert sc.prompt_for_review(selected) == sc.PROMPT


class ReceiptCheckedDevelopment:
    """Contract fixture for the pre-existing development interface; no training."""
    def __init__(self):
        self.measured = {
            "id": "c" * 32, "name": "Measured baseline", "variant": "baseline",
            "parent": None, "comparison": {}, "split_counts": {"train": 10, "validation": 4, "test": 4},
            "runtime_sha256": "1" * 64, "evaluation_sha256": "2" * 64, "receipt_sha256": "3" * 64,
            "score_origin": "host_validation", "fresh_process_replay": True,
            "test_used_for_feedback": False, "validation": {"metric": "rmse", "score": 0.25},
        }

    def records(self):
        return [{"id": self.measured["id"], "variant": "baseline"}]

    def record(self, identity):
        assert identity == self.measured["id"]
        return copy.deepcopy(self.measured)


def test_existing_measured_study_retains_its_original_sc_contract(make_case):
    _, _, context, _ = make_case()
    selected = packet(context, development=ReceiptCheckedDevelopment())
    acceptance = selected["scientific_acceptance"]
    assert acceptance["study_evidence_required"] is True
    assert acceptance["study_review_policy"]
    assert "analysis_review_policy" not in acceptance
    assert selected["development_measurements"]["experiments"][0]["validation"]["score"] == 0.25
    assert sc.prompt_for_review(selected).startswith(sc.STUDY_PROMPT)


@pytest.mark.parametrize("protocol", ["chat_completions", "responses"])
def test_model_and_study_output_schemas_take_precedence(protocol):
    model = specification("SC", protocol, model_acceptance=True, analysis_review_policy=analysis.POLICY)
    assert "model_assessments" in model["schema"]["properties"]
    assert "claims" not in model["schema"]["properties"]
    study = specification("SC", protocol, study_acceptance=True, analysis_review_policy=analysis.POLICY)
    assert "evidence_ids" in study["schema"]["properties"]
    assert "claims" not in study["schema"]["properties"]
    with pytest.raises(ValueError, match="Unsupported analysis"):
        specification("SC", protocol, analysis_review_policy="unknown")
