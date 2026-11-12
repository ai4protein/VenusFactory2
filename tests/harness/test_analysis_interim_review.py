"""Interim analysis cannot accept source-only checkpoints or invented evidence."""
import copy
import json

import pytest
import test_analysis_report_review as cases

from agent.harness import analysis_interim_review as interim

make_case = cases.make_case


def feedback(context):
    return {"decision": "needs_revision", "summary": "The computation exists but no final report is submitted.",
            "issues": ["A complete report is missing."],
            "evidence_ids": interim.eligibility(context)["verified_task_ids"][:1],
            "next_actions": ["Write the report using the recorded results and checkpoint it."]}


@pytest.mark.parametrize("fault", ["no_checkpoint", "empty_steps", "partial", "interrupted",
                                  "only_python", "only_json", "missing_preview", "truncated_preview",
                                  "changed_index", "copied_input"])
def test_partial_or_unverifiable_report_selects_interim(make_case, fault):
    _, _, context, _ = make_case()
    if fault == "no_checkpoint":
        context["execution"]["checkpoint"] = None
    elif fault == "empty_steps":
        context["execution"]["checkpoint"]["completed_steps"] = []
    elif fault == "partial":
        context["execution"]["checkpoint"]["partial"] = True
    elif fault == "interrupted":
        context["execution"]["interruption"] = {"reason": "native_execution_failed"}
    elif fault in {"only_python", "only_json"}:
        name = "source.py" if fault == "only_python" else "analysis/result.json"
        for rows in [context["execution"]["checkpoint"]["artifacts"], context["artifact_previews"],
                     context["analysis_evidence"]["checkpoint_artifacts"]["records"]]:
            rows[0]["path"] = name
    elif fault == "missing_preview":
        context["artifact_previews"] = []
    elif fault == "truncated_preview":
        context["artifact_previews"][0]["content"] = "Reported"
    elif fault == "changed_index":
        context["analysis_evidence"]["checkpoint_artifacts"]["records"][0]["status"] = "changed"
    else:
        item = context["execution"]["checkpoint"]["artifacts"][0]
        context["analysis_evidence"]["materials"]["records"].append(
            {"path": item["path"], "sha256": item["sha256"], "status": "verified"})
    state = interim.eligibility(context)
    assert state["stage"] == "interim" and state["reasons"]
    review = feedback(context)
    assert interim.parse_review(json.dumps(review), context) == review
    review["decision"] = "ready_for_host_validation"
    with pytest.raises(ValueError, match="needs_revision"):
        interim.validate(review, context)


def test_actual_markdown_report_is_eligible_without_new_computations(make_case):
    _, _, context, _ = make_case()
    context["analysis_evidence"]["computations"] = []
    assert interim.eligibility(context)["stage"] == "final"
    with pytest.raises(ValueError, match="final analysis policy"):
        interim.validate(feedback(context), context)


@pytest.mark.parametrize("report", ["Paragraph with a conclusion.\n", "Paragraph.\r\n"])
def test_exact_plain_text_and_crlf_reports_are_eligible(make_case, report):
    _, _, context, _ = make_case(report=report)
    # Fixture's original bytes are preserved in the preview and digest.
    for rows in [context["execution"]["checkpoint"]["artifacts"], context["artifact_previews"],
                 context["analysis_evidence"]["checkpoint_artifacts"]["records"]]:
        rows[0]["path"] = "report.txt"
    assert interim.eligibility(context)["stage"] == "final"


def test_empty_report_is_not_final(make_case):
    _, _, context, _ = make_case(report="\n")
    assert interim.eligibility(context)["stage"] == "interim"


@pytest.mark.parametrize("fault", ["unknown_id", "plan_id", "no_citation", "extra_claim", "empty_action", "duplicate_key"])
def test_interim_is_closed_and_uses_real_task_ids(make_case, fault):
    _, _, context, _ = make_case()
    context["execution"]["checkpoint"] = None
    review = feedback(context)
    if fault == "unknown_id":
        review["evidence_ids"] = ["f" * 32]
    elif fault == "plan_id":
        review["evidence_ids"] = ["compute"]
    elif fault == "no_citation":
        review["evidence_ids"] = []
    elif fault == "extra_claim":
        review["claims"] = []
    elif fault == "empty_action":
        review["next_actions"] = [" "]
    if fault == "duplicate_key":
        raw = json.dumps(review)[:-1] + ',"decision":"needs_revision"}'
    else:
        raw = json.dumps(review)
    with pytest.raises(ValueError):
        interim.parse_review(raw, context)


def test_no_computation_still_allows_honest_interim_feedback():
    context = {"plan": {"steps": []}, "execution": {"checkpoint": None}}
    review = feedback(context)
    assert review["evidence_ids"] == []
    assert interim.parse_review(json.dumps(review), context) == review


def test_eligibility_is_derived_not_copied_from_model_claim(make_case):
    _, _, context, _ = make_case()
    context["artifact_previews"][0]["content"] = "fabricated"
    context["analysis_review_state"] = {"stage": "final", "report_paths": ["report.md"]}
    before = copy.deepcopy(context)
    assert interim.eligibility(context)["stage"] == "interim"
    assert context == before
