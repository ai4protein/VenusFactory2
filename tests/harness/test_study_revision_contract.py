"""SC field relationships are disclosed, enforced and repaired without approving models."""
import copy
import json
from types import SimpleNamespace

import httpx
import pytest
from jsonschema import Draft202012Validator
from test_sc_study_evidence import request
from test_scientist_role_output import policy_for, response_for
from test_study_review_state import packet, review

from agent.harness import autonomous_research as ar
from agent.harness import scientific_acceptance as sc
from agent.harness import study_review_state as study
from agent.harness.scientist_role_output import RoleOutputError, request_fields, specification, validate_output

KINDS = ("missing_evidence", "execution_failure", "scientific_concern")
REQUIREMENTS = (
    "data_partitions", "validation_measurement", "measured_baseline", "measured_revision",
    "model_submission", "execution_failure", "scientific_value",
)
VALID_PAIRS = {
    *(('missing_evidence', name) for name in REQUIREMENTS[:5]),
    ('execution_failure', 'execution_failure'),
    ('scientific_concern', 'scientific_value'),
}


def spec_for(protocol):
    return specification("SC", protocol, study_acceptance=True, study_review_policy=study.POLICY)


@pytest.mark.parametrize("protocol", ["chat_completions", "responses"])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("requirement", REQUIREMENTS)
def test_all_revision_pairs_match_disclosed_and_parsed_schema(tmp_path, protocol, kind, requirement):
    context, _, _ = packet(tmp_path)
    value = review(context, [request(requirement, kind=kind)])
    original = copy.deepcopy(value)
    spec = spec_for(protocol)
    tool = request_fields(spec)["tools"][0]
    outgoing = tool["function"]["parameters"] if protocol == "chat_completions" else tool["parameters"]
    Draft202012Validator.check_schema(outgoing)
    raw = json.dumps(value)
    if (kind, requirement) in VALID_PAIRS:
        Draft202012Validator(outgoing).validate(value)
        assert json.loads(validate_output(raw, spec)) == value
    else:
        assert not Draft202012Validator(outgoing).is_valid(value)
        with pytest.raises(RoleOutputError, match="Invalid revision pair") as error:
            validate_output(raw, spec)
        assert kind in str(error.value) and requirement in str(error.value)
        assert "requirement must be" in str(error.value)
        with pytest.raises(study.StudyReviewInconsistent, match="Invalid revision pair"):
            study.validate(value, context)
    assert value == original



@pytest.mark.parametrize("kind,requirement", [
    ("missing_evidence", "measured_revision"),
    ("execution_failure", "execution_failure"),
    ("scientific_concern", "scientific_value"),
])
def test_each_union_branch_is_a_complete_closed_function_object(kind, requirement):
    item = study.request_schema()["items"]
    matching = [branch for branch in item["anyOf"]
                if Draft202012Validator(branch).is_valid(request(requirement, kind=kind))]
    assert len(matching) == 1
    branch = matching[0]
    assert branch["type"] == "object" and branch["additionalProperties"] is False
    assert set(branch["required"]) == set(item["properties"])
    missing_reason = request(requirement, kind=kind)
    missing_reason.pop("reason")
    assert not Draft202012Validator(branch).is_valid(missing_reason)
    assert not Draft202012Validator(branch).is_valid({**request(requirement, kind=kind), "approved": True})


def test_legal_pair_cannot_invent_execution_failure_or_erase_baseline(tmp_path):
    context, measured, _ = packet(tmp_path)
    failure = review(context, [request("execution_failure", kind="execution_failure")])
    with pytest.raises(study.StudyReviewInconsistent, match="No unresolved execution failure"):
        sc.parse_study_review(json.dumps(failure), context)
    baseline = review(context, [request("measured_baseline")])
    with pytest.raises(study.StudyReviewInconsistent, match="already recorded"):
        sc.parse_study_review(json.dumps(baseline), context)
    context["execution"] = {"failures": [{"tool": "train_research_model", "error": "fit failed"}]}
    context["measurement_state"] = study.measurement_state(context)
    assert sc.parse_study_review(json.dumps(failure), context) == failure
    result = sc.assess(failure, context)
    assert not result["passed"] and result["approved_models"] == {}
    assert result["model_assessments"] == []
    assert context["measurement_state"]["experiments"][0]["experiment_id"] == measured["id"]


def test_pair_contract_does_not_change_legacy_or_final_model_schema():
    legacy = specification("SC", "chat_completions", study_acceptance=True)
    assert "study_review_policy" not in legacy
    assert "revision_requests" not in legacy["schema"]["properties"]
    final = specification("SC", "responses", model_acceptance=True, study_acceptance=True,
                          study_review_policy=study.POLICY)
    assert "study_review_policy" not in final
    assert final["schema"] == sc.model_schema()


def fake_profile(protocol):
    return SimpleNamespace(model=SimpleNamespace(protocol=protocol, model="fixture"),
                           budget=SimpleNamespace(request_seconds=2, model_calls=8))


def context_in(payload, protocol):
    text = payload["input"][0]["content"] if protocol == "responses" else payload["messages"][-1]["content"]
    return json.loads(text.split("\n\nYour task in this phase", 1)[0])


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat_completions", "responses"])
@pytest.mark.parametrize("repair_succeeds", [True, False])
async def test_historical_pair_error_gets_one_explicit_repair_without_approval(
        tmp_path, monkeypatch, protocol, repair_succeeds):
    context, measured, path = packet(tmp_path)
    evidence_before = path.read_bytes()
    # Exact malformed field relationship seen in the old Ki review, with public fixture ID.
    malformed = review(context, [request("measured_revision", kind="execution_failure", identity=measured["id"])])
    corrected = copy.deepcopy(malformed)
    corrected["revision_requests"][0].update(kind="missing_evidence", experiment_id="")
    spec = spec_for(protocol)
    policy = policy_for(protocol)
    relay = SimpleNamespace(base_url="http://fixture", token="fixture", calls=0, policy=policy)
    original_client = httpx.AsyncClient
    seen = []

    def handler(http_request):
        sent, _ = policy.apply(json.loads(http_request.content))
        seen.append(sent)
        relay.calls += 1
        value = corrected if len(seen) == 2 and repair_succeeds else malformed
        return httpx.Response(200, json=response_for(spec, json.dumps(value)))

    monkeypatch.setattr(ar.httpx, "AsyncClient", lambda **kw: original_client(transport=httpx.MockTransport(handler), **kw))
    runner = ar.AutonomousResearch(None, "fixture", {})
    call = runner.structured("SC", context, fake_profile(protocol), relay, tmp_path / "review",
                             lambda raw: sc.parse_study_review(raw, context))
    if repair_succeeds:
        value = await call
        assert value == corrected and value["decision"] == "needs_revision"
        assert not sc.assess(value, context)["passed"]
        assert sc.assess(value, context)["approved_models"] == {}
    else:
        with pytest.raises(RoleOutputError, match="Invalid revision pair"):
            await call
    assert len(seen) == relay.calls == 2
    feedback = context_in(seen[1], protocol)["format_repair"]
    assert "kind='execution_failure'" in feedback["validation_error"]
    assert "requirement='measured_revision'" in feedback["validation_error"]
    assert "experiment_id empty" in feedback["validation_error"]
    assert study.PAIRING_GUIDANCE in feedback["instruction"]
    assert "do not invent results or force acceptance" in feedback["instruction"]
    assert json.loads(feedback["previous_response"]) == malformed
    assert context_in(seen[1], protocol)["measurement_state"] == context["measurement_state"]
    assert path.read_bytes() == evidence_before
    assert policy._role_output is None


def text_response(protocol, body):
    text = json.dumps(body)
    if protocol == "responses":
        return {"model": "fixture", "status": "completed", "output": [{
            "type": "message", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": text}]}]}
    return {"model": "fixture", "choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": text}}]}


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat_completions", "responses"])
@pytest.mark.parametrize("status", [400, 404, 422])
@pytest.mark.parametrize("invalid_pair", [False, True])
async def test_provider_format_fallback_keeps_identical_study_schema_and_validation(
        tmp_path, monkeypatch, protocol, status, invalid_pair):
    context, _, _ = packet(tmp_path)
    body = review(context, [request("measured_revision", kind="execution_failure" if invalid_pair else "missing_evidence")])
    spec = spec_for(protocol)
    policy = policy_for(protocol)
    relay = SimpleNamespace(base_url="http://fixture", token="fixture", calls=0, policy=policy)
    original_client = httpx.AsyncClient
    seen = []

    def handler(http_request):
        sent, _ = policy.apply(json.loads(http_request.content))
        seen.append(sent)
        relay.calls += 1
        if len(seen) == 1:
            assert sent["tools"] == [spec["tool"]]
            return httpx.Response(status, json={"error": "Unsupported tool schema"})
        assert not sent.get("tools") and sent["tool_choice"] == "none"
        assert sent["model"] == "fixture"
        instruction = sent["instructions"] if protocol == "responses" else sent["messages"][0]["content"]
        assert json.loads(instruction.split(" Exact response schema: ", 1)[1]) == spec["schema"]
        assert study.PAIRING_GUIDANCE in instruction
        return httpx.Response(200, json=text_response(protocol, body))

    monkeypatch.setattr(ar.httpx, "AsyncClient", lambda **kw: original_client(transport=httpx.MockTransport(handler), **kw))
    call = ar.AutonomousResearch(None, "fixture", {}).consult("SC", context, fake_profile(protocol), relay, tmp_path)
    if invalid_pair:
        with pytest.raises(RoleOutputError, match="Invalid revision pair"):
            await call
    else:
        value = sc.parse_study_review(await call, context)
        assert value == body and value["decision"] == "needs_revision"
        assert not sc.assess(value, context)["passed"]
    assert len(seen) == relay.calls == 2
    assert policy._role_output is None
