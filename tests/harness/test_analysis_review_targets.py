"""SC IDs and bounded repair feedback refer to actual captured host evidence."""
import copy
import json
from types import SimpleNamespace

import httpx
import pytest
import test_analysis_report_review as analysis_cases
from test_scientist_role_output import policy_for, response_for

from agent.harness import analysis_report_review as analysis
from agent.harness import autonomous_research as ar
from agent.harness import scientific_acceptance as sc
from agent.harness.scientist_role_output import RoleOutputError, specification, validate_output

make_case = analysis_cases.make_case


def packet(context):
    context = copy.deepcopy(context)
    context["plan"]["deliverable"] = "analysis"
    context["plan"]["steps"].append({"id": "inspect", "verification": "artifact"})
    return sc.evidence_context(context, [], None)


def test_targets_are_recomputed_from_host_checkpoint_and_verified_tasks(make_case):
    _, _, context, review = make_case()
    context["analysis_review_targets"] = {"completed_step_ids": ["forged"]}
    context["analysis_evidence"]["computations"].append({"status": "unavailable", "task_id": "d" * 32})
    selected = packet(context)
    targets = selected["analysis_review_targets"]
    assert targets["completed_step_ids"] == ["compute"]
    assert targets["unclaimed_plan_step_ids"] == ["inspect"]
    assert targets["computation_task_ids"] == [review["claims"][0]["task_id"]]
    assert targets["completed_step_ids_omitted"] == targets["computation_task_ids_omitted"] == 0
    assert analysis.validate(review, selected) is None
    assert context["analysis_review_targets"] == {"completed_step_ids": ["forged"]}


def test_target_navigation_is_bounded_and_explicit_about_omitted_ids(make_case):
    _, _, context, _ = make_case()
    context["analysis_evidence"]["computations"] = [{"status": "verified", "task_id": format(i, "032x")}
                                                   for i in range(70)]
    targets = analysis.review_targets(context)
    assert len(targets["computation_task_ids"]) == 64
    assert targets["computation_task_ids_omitted"] == 6


@pytest.mark.parametrize("protocol", ["chat_completions", "responses"])
def test_task_and_step_schema_descriptions_distinguish_their_sources(make_case, protocol):
    _, _, context, review = make_case()
    selected = packet(context)
    spec = specification("SC", protocol, analysis_review_policy=analysis.POLICY)
    fields = spec["schema"]["properties"]
    assert "host job ID" in fields["claims"]["items"]["properties"]["task_id"]["description"]
    assert "completed_step_ids" in fields["step_assessments"]["items"]["properties"]["step_id"]["description"]
    assert json.loads(validate_output(json.dumps(review), spec)) == review
    assert sc.assess(review, selected)["passed"] is True


@pytest.mark.parametrize("bad_id", ["step1", "sk-test-fixture-secret-never-print"])
def test_pattern_error_explains_host_id_format_without_echoing_value(make_case, bad_id):
    _, _, _, review = make_case()
    review["claims"][0]["task_id"] = bad_id
    spec = specification("SC", "chat_completions", analysis_review_policy=analysis.POLICY)
    with pytest.raises(RoleOutputError) as failure:
        validate_output(json.dumps(review), spec)
    message = str(failure.value)
    assert "claims.0.task_id" in message
    assert "32-character lowercase hexadecimal host record ID" in message
    assert "not a plan step ID" in message
    assert bad_id not in message and "Expected string" not in message
    assert len(message) < 350


def test_step_mismatch_identifies_exact_known_missing_and_extra_ids(make_case):
    _, _, context, review = make_case()
    selected = packet(context)
    review["step_assessments"][0]["step_id"] = "inspect"
    with pytest.raises(ValueError) as failure:
        analysis.validate(review, selected)
    message = str(failure.value)
    assert 'Missing step_ids: ["compute"]' in message
    assert 'extra step_ids: ["inspect"]' in message
    assert len(message) < 400  # Fits the existing one-repair feedback allowance.


def test_unknown_extra_step_does_not_echo_arbitrary_payload(make_case):
    _, _, context, review = make_case()
    review["step_assessments"][0]["step_id"] = "SECRET_FIXTURE /private/config"
    with pytest.raises(ValueError) as failure:
        analysis.validate(review, packet(context))
    assert "SECRET_FIXTURE" not in str(failure.value)
    assert "<unknown step id>" in str(failure.value)


def test_need_revision_may_reject_without_claims_but_still_covers_claimed_steps(make_case):
    _, _, context, review = make_case()
    selected = packet(context)
    review.update(decision="needs_revision", issues=["The submitted evidence is insufficient."],
                  claims=[], input_assessments=[])
    review["step_assessments"][0].update(verdict="insufficient_evidence", claim_ids=[])
    analysis.validate(review, selected)
    assert sc.assess(review, selected)["passed"] is False
    review["step_assessments"] = []
    with pytest.raises(ValueError, match="Missing step_ids"):
        analysis.validate(review, selected)


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["task_id", "step_set"])
@pytest.mark.parametrize("repair_succeeds", [True, False])
async def test_actual_consult_single_repair_gets_precise_targets_without_auto_review(
        make_case, monkeypatch, tmp_path, fault, repair_succeeds):
    _, _, context, correct = make_case()
    selected = packet(context)
    bad = copy.deepcopy(correct)
    if fault == "task_id":
        bad["claims"][0]["task_id"] = "step1"
    else:
        bad["step_assessments"][0]["step_id"] = "inspect"
    original = httpx.AsyncClient
    policy = policy_for("chat_completions")
    spec = specification("SC", "chat_completions", analysis_review_policy=analysis.POLICY)
    seen = []
    relay = SimpleNamespace(base_url="http://fixture", token="fixture", calls=0, policy=policy)

    def handler(request):
        sent, metadata = policy.apply(json.loads(request.content))
        seen.append((sent, metadata))
        relay.calls += 1
        body = correct if repair_succeeds and len(seen) == 2 else bad
        return httpx.Response(200, json=response_for(spec, json.dumps(body)))

    monkeypatch.setattr(ar.httpx, "AsyncClient",
                        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    profile = SimpleNamespace(model=SimpleNamespace(protocol="chat_completions", model="fixture"),
                              budget=SimpleNamespace(request_seconds=2, model_calls=8))
    call = ar.AutonomousResearch(None, "fixture", {}).structured("SC", selected, profile, relay, tmp_path,
                                                               lambda raw: analysis.parse_review(raw, selected))
    if repair_succeeds:
        assert await call == correct
    else:
        with pytest.raises((RoleOutputError, ValueError)):
            await call
    assert len(seen) == relay.calls == 2
    contexts = [json.loads(sent["messages"][-1]["content"].split("\n\nYour task")[0]) for sent, _ in seen]
    assert all(item["analysis_review_targets"] == selected["analysis_review_targets"] for item in contexts)
    repair = contexts[1]["format_repair"]
    assert json.loads(repair["previous_response"]) == bad
    assert ("32-character lowercase hexadecimal" if fault == "task_id" else 'Missing step_ids: ["compute"]') in repair["validation_error"]
    assert [metadata["role_output"] for _, metadata in seen] == ["SC", "SC"]
    assert policy._role_output is None


def test_input_coverage_reports_exact_known_missing_and_extra_task_ids(make_case):
    from test_analysis_report_review import run_capture

    from agent.harness.research_analysis_provenance import build_evidence_index

    service, inputs, context, review = make_case()
    first = review["claims"][0]["task_id"]
    second = run_capture(service, {"sequence": "MKWVTFISLLFLFSSAYS"}, {"data": {"other": 1}})
    context["analysis_evidence"] = build_evidence_index(
        service.workspace, receipts=service.receipts, inputs=inputs,
        checkpoint=context["execution"]["checkpoint"])
    review["input_assessments"][0]["task_id"] = second
    with pytest.raises(ValueError) as failure:
        analysis.validate(review, context)
    message = str(failure.value)
    assert 'Missing task_ids: ["' + first + '"]' in message
    assert 'extra task_ids: ["' + second + '"]' in message
    assert "Multiple claims for one task share one assessment" in message
    assert len(message) <= 400


def test_unknown_input_task_ids_are_never_echoed(make_case):
    _, _, context, review = make_case()
    unknown = "deadbeef" * 4
    review["input_assessments"][0]["task_id"] = unknown
    with pytest.raises(ValueError) as failure:
        analysis.validate(review, context)
    assert unknown not in str(failure.value)
    assert "<unknown task id>" in str(failure.value)
    assert review["claims"][0]["task_id"] in str(failure.value)


def test_multiple_claims_of_one_task_share_one_input_assessment(make_case):
    import copy

    _, _, context, review = make_case()
    additional = copy.deepcopy(review["claims"][0])
    additional["id"] = "same-task-second-claim"
    review["claims"].append(additional)
    review["step_assessments"][0]["claim_ids"].append(additional["id"])
    assert analysis.validate(review, context) is None
    review["input_assessments"].append(copy.deepcopy(review["input_assessments"][0]))
    with pytest.raises(ValueError, match="Multiple claims for the same task share one input assessment"):
        analysis.validate(review, context)
