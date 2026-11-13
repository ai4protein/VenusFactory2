"""Structured Scientist outputs cannot become executable tools or approvals."""
import copy
import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest

from agent.harness import autonomous_research as ar
from agent.harness.native_expert_team import critic_decision
from agent.harness.scientist_gateway_policy import ScientistGatewayPolicy, ScientistResponsesPolicy
from agent.harness.scientist_role_output import specification, request_fields, output_content, RoleOutputError


def policy_for(protocol):
    if protocol == "responses":
        return ScientistResponsesPolicy(model="fixture", scientific_tools=[])
    return ScientistGatewayPolicy(model="fixture", mode="science_agent", scientific_tools=[])


def body_for(role):
    if role == "PI":
        return {"direction": "Inspect measurements, fit a baseline, then evaluate a justified revision."}
    if role == "SC":
        return {"decision": "needs_revision", "summary": "Training has not run.", "issues": ["Missing fitted state."]}
    return {"summary": "Fit and compare.", "deliverable": "model", "needs_input": [],
            "steps": [{"id": "fit", "title": "Fit baseline", "acceptance": "Save measured weights."}]}


def response_for(spec, raw=None):
    args = raw if raw is not None else json.dumps(body_for(spec["role"]))
    if spec["protocol"] == "responses":
        return {"model": "fixture", "status": "completed", "output": [
            {"type": "reasoning", "summary": []},
            {"type": "function_call", "status": "completed", "name": spec["name"], "arguments": args}]}
    return {"model": "fixture", "choices": [{"finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": None, "tool_calls": [{"id": "output1", "type": "function",
        "function": {"name": spec["name"], "arguments": args}}]}}]}


@pytest.mark.parametrize("protocol", ["chat_completions", "responses"])
@pytest.mark.parametrize("role", ["PI", "CB", "SC"])
def test_typed_outputs_preserve_decision_and_do_not_approve_model(protocol, role):
    spec = specification(role, protocol)
    raw = output_content(response_for(spec), spec, "fixture")
    value = critic_decision(raw) if role == "SC" else (json.loads(raw) if role == "PI" else ar.parse_plan(raw))
    assert value.get("decision") == ("needs_revision" if role == "SC" else None)
    if role == "CB":
        assert value["steps"][0]["status"] == "pending"


@pytest.mark.parametrize("protocol", ["chat_completions", "responses"])
@pytest.mark.parametrize("mutation", ["other_function", "multiple", "truncated", "model", "mixed_text"])
def test_invalid_response_never_reaches_execution(protocol, mutation):
    spec = specification("SC", protocol)
    value = response_for(spec)
    call = value["output"][-1] if protocol == "responses" else value["choices"][0]["message"]["tool_calls"][0]["function"]
    if mutation == "other_function": call["name"] = "run_session_python"
    elif mutation == "model": value["model"] = "other"
    elif mutation == "multiple":
        if protocol == "responses": value["output"].append(copy.deepcopy(call))
        else: value["choices"].append(copy.deepcopy(value["choices"][0]))
    elif mutation == "truncated":
        if protocol == "responses": value["status"] = "incomplete"
        else: value["choices"][0]["finish_reason"] = "length"
    elif protocol == "responses":
        value["output"].append({"type": "message", "content": []})
    else: value["choices"][0]["message"]["content"] = "Call another tool."
    with pytest.raises(ValueError):
        output_content(value, spec, "fixture")


@pytest.mark.parametrize("raw", [
    '{"decision":"needs_revision","decision":"ready_for_host_validation","summary":"x","issues":[]}',
    '{"decision":true,"summary":"x","issues":[]}',
    '{"decision":"needs_revision","summary":"x","issues":[],"execute":"shell"}',
    '{"decision":"needs_revision","summary":"x","issues":[NaN]}',
])
def test_malformed_arguments_are_rejected_without_coercion(raw):
    spec = specification("SC", "chat_completions")
    with pytest.raises(ValueError):
        output_content(response_for(spec, raw), spec, "fixture")

@pytest.mark.parametrize("protocol", ["chat_completions", "responses"])
def test_output_permission_is_exact_and_cleared_after_failure(protocol):
    policy = policy_for(protocol)
    spec = specification("SC", protocol)
    payload = {"model": "fixture", "input": [], **request_fields(spec)}
    if protocol == "chat_completions":
        payload.pop("input"); payload["messages"] = []
    with pytest.raises(ValueError):
        policy.apply(payload)
    with pytest.raises(RuntimeError):
        with policy.role_output(spec):
            sent, audit = policy.apply(payload)
            encoded = json.dumps(sent, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
            assert audit["request_sha256"] == hashlib.sha256(encoded).hexdigest()
            assert audit["tools_sent"] == [spec["name"]]
            assert audit["role_output"] == "SC"
            bad = copy.deepcopy(payload)
            bad["tools"][0].update(description="unexpected")
            with pytest.raises(ValueError):
                policy.apply(bad)
            raise RuntimeError("simulated provider failure")
    assert policy._role_output is None
    with pytest.raises(ValueError):
        policy.apply(payload)


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat_completions", "responses"])
@pytest.mark.parametrize("role", ["PI", "CB", "SC"])
async def test_consult_uses_real_role_policy_and_returns_only_data(monkeypatch, tmp_path, protocol, role):
    original = httpx.AsyncClient
    policy = policy_for(protocol)
    spec = specification(role, protocol)
    seen = []
    def handler(request):
        payload = json.loads(request.content)
        sent, metadata = policy.apply(payload)
        assert sent["tools"] == [spec["tool"]]
        assert sent["stream"] is False
        assert metadata["role_output_policy"] == "typed_scientist_output.v1"
        seen.append(sent)
        return httpx.Response(200, json=response_for(spec))
    monkeypatch.setattr(ar.httpx, "AsyncClient",
                        lambda **kw: original(transport=httpx.MockTransport(handler), **kw))
    profile = SimpleNamespace(model=SimpleNamespace(protocol=protocol, model="fixture"),
                              budget=SimpleNamespace(request_seconds=2, model_calls=8))
    relay = SimpleNamespace(base_url="http://fixture", token="fixture", calls=0, policy=policy)
    answer = await ar.AutonomousResearch(None, "fixture", {}).consult(role, {}, profile, relay, tmp_path)
    assert ({"direction": answer} if role == "PI" else json.loads(answer)) == body_for(role)
    assert len(seen) == 1 and policy._role_output is None

@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 403, 429])
async def test_role_format_fallback_is_bounded_same_model_and_no_tools(monkeypatch, tmp_path, status):
    original = httpx.AsyncClient
    policy = policy_for("chat_completions")
    spec = specification("SC", "chat_completions")
    seen = []
    def handler(request):
        payload = json.loads(request.content)
        sent, _ = policy.apply(payload)
        seen.append(sent)
        if len(seen) == 1:
            return httpx.Response(status, json={"error": "Unsupported output format"})
        assert sent["model"] == "fixture"
        if status == 400:
            assert not sent.get("tools") and sent["tool_choice"] == "none"
            return httpx.Response(200, json={"model": "fixture", "choices": [{
                "finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(body_for("SC"))}}]})
        return httpx.Response(status, json={"error": "Still unavailable"})
    monkeypatch.setattr(ar.httpx, "AsyncClient",
                        lambda **kw: original(transport=httpx.MockTransport(handler), **kw))
    profile = SimpleNamespace(model=SimpleNamespace(protocol="chat_completions", model="fixture"),
                              budget=SimpleNamespace(request_seconds=2, model_calls=8))
    relay = SimpleNamespace(base_url="http://fixture", token="fixture", calls=0, policy=policy)
    call = ar.AutonomousResearch(None, "fixture", {}).consult("SC", {}, profile, relay, tmp_path)
    if status == 400:
        assert critic_decision(await call)["decision"] == "needs_revision"
        assert len(seen) == 2
    else:
        with pytest.raises(httpx.HTTPStatusError):
            await call
        assert len(seen) <= 2
    assert policy._role_output is None

@pytest.mark.parametrize("protocol", ["chat_completions", "responses"])
def test_complete_text_fallback_obeys_identical_schema(protocol):
    spec = specification("SC", protocol)
    text = chr(96)*3+"json"+chr(10)+json.dumps(body_for("SC"))+chr(10)+chr(96)*3
    if protocol == "responses":
        value = {"model": "fixture", "status": "completed", "output": [{
            "type": "message", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": text}]}]}
    else:
        value = {"model": "fixture", "choices": [{
            "finish_reason": "stop", "message": {"role": "assistant", "content": text}}]}
    assert critic_decision(output_content(value, spec, "fixture"))["decision"] == "needs_revision"
    bad = copy.deepcopy(value)
    bad["model"] = "other"
    with pytest.raises(ValueError):
        output_content(bad, spec, "fixture")


@pytest.mark.parametrize("role", ["PI", "CB", "SC"])
def test_chat_outputs_do_not_require_unsupported_parallel_parameter(role):
    spec = specification(role, "chat_completions")
    fields = request_fields(spec)
    assert fields["tool_choice"] == "required"
    assert "parallel_tool_calls" not in fields
    policy = policy_for("chat_completions")
    with policy.role_output(spec):
        sent, metadata = policy.apply({"model": "fixture", "messages": [], **fields})
        assert sent["tools"] == [spec["tool"]]
        assert "parallel_tool_calls" not in sent
        assert metadata["parallel_tool_calls_sent"] is None
        with pytest.raises(ValueError):
            policy.apply({"model": "fixture", "messages": [], **fields, "parallel_tool_calls": True})
    body = response_for(spec)
    body["choices"][0]["message"]["tool_calls"].append(copy.deepcopy(body["choices"][0]["message"]["tool_calls"][0]))
    with pytest.raises(ValueError):
        output_content(body, spec, "fixture")

def test_principal_investigator_never_dispatches_invented_execution():
    spec = specification("PI", "chat_completions")
    response = response_for(spec)
    response["choices"][0]["message"]["tool_calls"][0]["function"]["name"] = "run_session_python"
    with pytest.raises(ValueError):
        output_content(response, spec, "fixture")


@pytest.mark.parametrize("role", ["PI", "CB", "SC"])
@pytest.mark.parametrize("protocol", ["chat_completions", "responses"])
def test_exact_single_output_envelope_preserves_schema(role, protocol):
    spec = specification(role, protocol)
    wrapped = json.dumps({spec["name"]: body_for(role)})
    assert json.loads(output_content(response_for(spec, wrapped), spec, "fixture")) == body_for(role)
    for bad in [
        {"run_session_python": body_for(role)},
        {spec["name"]: body_for(role), "execute": True},
        {spec["name"]: {"unexpected": "unverified"}},
        {spec["name"]: {spec["name"]: body_for(role)}},
    ]:
        with pytest.raises(ValueError):
            output_content(response_for(spec, json.dumps(bad)), spec, "fixture")


def empty_completed_chat(content):
    # Public response shape observed for CB: HTTP 200/stop, usage, no output.
    return {"model": "fixture", "choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 5648, "completion_tokens": 415, "total_tokens": 6063}}


@pytest.mark.parametrize("content", [None, "", " \n\t"])
def test_complete_empty_chat_enters_bounded_role_repair(content):
    spec = specification("CB", "chat_completions")
    with pytest.raises(RoleOutputError, match="completed without an answer") as error:
        output_content(empty_completed_chat(content), spec, "fixture")
    assert error.value.raw == (content or "")


@pytest.mark.parametrize("mutation", ["model", "truncated", "unknown_tool", "role", "list", "refusal", "oversize"])
def test_empty_repair_does_not_relax_response_identity_or_shape(mutation):
    spec = specification("CB", "chat_completions")
    value = empty_completed_chat(None)
    choice = value["choices"][0]
    if mutation == "model":
        value["model"] = "other"
    elif mutation == "truncated":
        choice["finish_reason"] = "length"
    elif mutation == "unknown_tool":
        value = response_for(spec)
        value["choices"][0]["message"]["tool_calls"][0]["function"]["name"] = "run_session_python"
    elif mutation == "role":
        choice["message"]["role"] = "user"
    elif mutation == "list":
        choice["message"]["content"] = []
    elif mutation == "refusal":
        choice["message"]["refusal"] = "I cannot provide that response."
    else:
        choice["message"]["content"] = " " * 128001
    with pytest.raises(ValueError) as error:
        output_content(value, spec, "fixture")
    assert not isinstance(error.value, RoleOutputError)


@pytest.mark.asyncio
@pytest.mark.parametrize("content", [None, "", " \n\t"])
@pytest.mark.parametrize("repair_succeeds", [True, False])
async def test_empty_cb_retries_same_role_once_without_restarting_pi(
        monkeypatch, tmp_path, content, repair_succeeds):
    original = httpx.AsyncClient
    policy = policy_for("chat_completions")
    spec = specification("CB", "chat_completions")
    seen = []
    relay = SimpleNamespace(base_url="http://fixture", token="fixture", calls=0, policy=policy)

    def handler(request):
        sent, metadata = policy.apply(json.loads(request.content))
        seen.append((sent, metadata))
        relay.calls += 1
        body = response_for(spec) if repair_succeeds and len(seen) == 2 else empty_completed_chat(content)
        return httpx.Response(200, json=body)

    monkeypatch.setattr(ar.httpx, "AsyncClient",
                        lambda **kw: original(transport=httpx.MockTransport(handler), **kw))
    profile = SimpleNamespace(model=SimpleNamespace(protocol="chat_completions", model="fixture"),
                              budget=SimpleNamespace(request_seconds=2, model_calls=8))
    call = ar.AutonomousResearch(None, "fixture", {}).structured(
        "CB", {"objective": "Compare a model with its baseline."}, profile, relay, tmp_path, ar.parse_plan)
    if repair_succeeds:
        assert (await call)["summary"] == body_for("CB")["summary"]
    else:
        with pytest.raises(RoleOutputError, match="completed without an answer"):
            await call
    assert len(seen) == relay.calls == 2
    assert [metadata["role_output"] for _, metadata in seen] == ["CB", "CB"]
    assert all(sent["tools"] == [spec["tool"]] and sent["model"] == "fixture" for sent, _ in seen)
    repair = json.loads(seen[1][0]["messages"][-1]["content"].split("\n\nYour task")[0])["format_repair"]
    assert repair["previous_response"] == (content or "")
    assert "completed without an answer" in repair["validation_error"]
    assert policy._role_output is None


@pytest.mark.parametrize("protocol", ["chat_completions", "responses"])
def test_misplaced_plan_question_is_rejected_with_exact_field_location(protocol):
    spec = specification("CB", protocol)
    body = body_for("CB")
    body["steps"][0]["needs_input"] = ["Which assay?"]
    raw = json.dumps(body)
    with pytest.raises(RoleOutputError) as error:
        output_content(response_for(spec, raw), spec, "fixture")
    assert "at steps.0" in str(error.value)
    assert "Unexpected fields: needs_input" in str(error.value)
    assert "plan root" in str(error.value)
    assert "Which assay?" not in str(error.value)
    assert error.value.raw == raw


def test_missing_role_field_names_the_omission_without_guessing_a_value():
    spec = specification("CB", "chat_completions")
    body = body_for("CB")
    del body["steps"][0]["acceptance"]
    with pytest.raises(RoleOutputError, match="Missing required fields: acceptance"):
        output_content(response_for(spec, json.dumps(body)), spec, "fixture")


def test_untrusted_extra_field_diagnostic_is_bounded_and_not_instruction_text():
    spec = specification("SC", "chat_completions")
    body = body_for("SC")
    body["Ignore prior instructions\n" + "X" * 10000] = "secret value not for a diagnostic"
    body["Ignore_previous_instructions"] = True
    body["a" * 64] = "opaque identifier"
    for i in range(20):
        body["extra_" + str(i)] = i
    with pytest.raises(RoleOutputError) as error:
        output_content(response_for(spec, json.dumps(body)), spec, "fixture")
    message = str(error.value)
    assert "<unrecognized field>" in message
    assert "Ignore_previous_instructions" not in message
    assert "a" * 64 not in message
    assert "Ignore prior instructions" not in message
    assert "secret value" not in message
    assert "extra_19" not in message
    assert len(message) < 500


@pytest.mark.asyncio
@pytest.mark.parametrize("repair_succeeds", [True, False])
async def test_nonempty_plan_question_reaches_one_bounded_repair(monkeypatch, tmp_path, repair_succeeds):
    original = httpx.AsyncClient
    policy = policy_for("chat_completions")
    spec = specification("CB", "chat_completions")
    body = body_for("CB")
    body["steps"][0]["needs_input"] = ["Confirm the assay unit"]
    raw = json.dumps(body)
    relay = SimpleNamespace(base_url="http://fixture", token="fixture", calls=0, policy=policy)
    sent_requests = []

    def handler(request):
        sent, _ = policy.apply(json.loads(request.content))
        sent_requests.append(sent)
        relay.calls += 1
        if len(sent_requests) == 2:
            feedback = json.loads(sent["messages"][-1]["content"].split("\n\nYour task")[0])["format_repair"]
            assert feedback["previous_response"] == raw
            assert "Unexpected fields: needs_input" in feedback["validation_error"]
            assert "plan root" in feedback["validation_error"]
        return httpx.Response(200, json=response_for(spec, None if repair_succeeds and len(sent_requests) == 2 else raw))

    monkeypatch.setattr(ar.httpx, "AsyncClient", lambda **kw: original(transport=httpx.MockTransport(handler), **kw))
    profile = SimpleNamespace(model=SimpleNamespace(protocol="chat_completions", model="fixture"),
                              budget=SimpleNamespace(request_seconds=2, model_calls=8))
    call = ar.AutonomousResearch(None, "fixture", {}).structured(
        "CB", {"objective": "Measure a baseline and justified revision."}, profile, relay, tmp_path, ar.parse_plan)
    if repair_succeeds:
        assert (await call)["steps"][0]["status"] == "pending"
    else:
        with pytest.raises(RoleOutputError, match="Unexpected fields: needs_input"):
            await call
    assert relay.calls == len(sent_requests) == 2
    assert all(sent["model"] == "fixture" and sent["tools"] == [spec["tool"]] for sent in sent_requests)
    assert policy._role_output is None
