import asyncio
import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from agent.harness import Budget, ModelBinding, Profile, builtin_registry
from agent.harness.codex_adapter import configuration
from agent.harness.responses_bridge import (
    ResponsesChatBridge, chat_to_response, response_sse, responses_to_chat, validate_profile,
)
from agent.harness.responses_gateway import inspect_responses
from agent.rsi.engine.model_gateway import GatewayPolicy

MODEL = "glm-4-flash"
TOOL = "mcp__venusfactory__protein_probe"


def policy(calls=3):
    return GatewayPolicy(model=MODEL, mode="science_agent", scientific_tools={"protein_probe"},
                         max_calls=calls, max_tokens=4096, temperature=0.1, thinking=None)


def request(namespace=False):
    tool = {"type": "function", "name": TOOL, "description": "Measure a protein",
            "parameters": {"type": "object", "properties": {"sequence": {"type": "string"}}}}
    if namespace:
        tool = {"type": "namespace", "name": "mcp__venusfactory",
                "tools": [{**tool, "name": "protein_probe"}]}
    return {"model": MODEL, "instructions": "Use the admitted protein tool.",
            "input": [{"role": "developer", "content": [{"type": "input_text", "text": "Keep citations."}]},
                      {"role": "user", "content": "Measure ACDEFGHIKC"}],
            "stream": True, "store": False, "tools": [tool]}


def answer(tool=True):
    message = {"role": "assistant", "content": None if tool else "10 residues."}
    if tool:
        message["tool_calls"] = [{"id": "call_1", "type": "function",
                                 "function": {"name": TOOL, "arguments": '{"sequence":"ACDEFGHIKC"}'}}]
    return {"id": "upstream-1", "model": MODEL,
            "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if tool else "stop"}],
            "usage": {"prompt_tokens": 31, "completion_tokens": 10, "total_tokens": 41}}


def translate(data, native):
    return chat_to_response(json.dumps(data).encode(), model=MODEL, allowed_tools={TOOL},
                            max_tokens=4096, upstream_url="https://fixture.invalid/v1",
                            identities=native["native_tool_identities"])


@pytest.mark.parametrize("namespace", [False, True])
def test_tool_call_round_trip_keeps_identity_and_stateless_history(namespace):
    body = request(namespace)
    original = copy.deepcopy(body)
    chat, native = responses_to_chat(body, policy())
    assert chat["messages"][:2] == [
        {"role": "system", "content": "Use the admitted protein tool."},
        {"role": "system", "content": "Keep citations."},
    ]
    assert chat["tools"][0]["function"]["name"] == TOOL
    response, usage = translate(answer(), native)
    item = response["output"][0]
    assert item["name"] == ("protein_probe" if namespace else TOOL)
    assert item.get("namespace") == ("mcp__venusfactory" if namespace else None)
    body["input"] += [item, {"type": "function_call_output", "call_id": "call_1", "output": '{"count":10}'}]
    next_chat, _ = responses_to_chat(body, policy())
    assert next_chat["messages"][-1] == {"role": "tool", "tool_call_id": "call_1", "content": '{"count":10}'}
    assert next_chat["messages"][-2]["tool_calls"][0]["function"]["name"] == TOOL
    assert usage["total_tokens"] == 41
    assert original["tools"] == body["tools"]


@pytest.mark.parametrize("tool,namespace", [(False, False), (True, False), (True, True)])
def test_native_stream_satisfies_responses_terminal_and_usage_contract(tool, namespace):
    _, native = responses_to_chat(request(namespace), policy())
    response, usage = translate(answer(tool), native)
    raw = response_sse(response)
    evidence = inspect_responses(raw, True, model=MODEL, allowed_tools={TOOL}, max_output_tokens=4096)
    assert evidence["usage"]["total_tokens"] == usage["total_tokens"]
    events = [json.loads(line[6:]) for line in raw.decode().splitlines() if line.startswith("data: ")]
    assert [event["sequence_number"] for event in events] == list(range(len(events)))
    assert events[-1]["type"] == "response.completed"
    assert events[-1]["response"] == response


@pytest.mark.parametrize("change", [
    {"model": "paid"}, {"stream": "yes"}, {"store": True}, {"previous_response_id": "old"},
    {"conversation": "foreign"}, {"background": True}, {"text": None},
    {"text": {"format": {"type": "json_schema"}}},
    {"input": [{"type": "compaction", "encrypted_content": "unverifiable"}]},
    {"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "file:///private"}]}]},
    {"input": [{"type": "function_call_output", "call_id": "missing", "output": "{}"}]},
    {"input": [], "instructions": None},
])
def test_unrepresentable_semantics_are_rejected(change):
    body = request()
    body.update(change)
    with pytest.raises(ValueError):
        responses_to_chat(body, policy())


def test_historical_calls_cannot_be_unpaired_duplicated_or_cross_namespace():
    body = request(True)
    _, native = responses_to_chat(body, policy())
    call = translate(answer(), native)[0]["output"][0]
    body["input"].append(call)
    with pytest.raises(ValueError, match="stateless"):
        responses_to_chat(body, policy())
    result = {"type": "function_call_output", "call_id": "call_1", "output": "{}"}
    body["input"] += [result, result]
    with pytest.raises(ValueError, match="duplicated"):
        responses_to_chat(body, policy())
    body["input"].pop()
    body["input"][-2]["namespace"] = "untrusted"
    with pytest.raises(ValueError, match="historical"):
        responses_to_chat(body, policy())


@pytest.mark.parametrize("mutation", ["model", "tool", "truncated", "usage", "arguments"])
def test_provider_response_cannot_bypass_the_existing_evidence_gate(mutation):
    _, native = responses_to_chat(request(), policy())
    data = answer()
    if mutation == "model": data["model"] = "paid"
    elif mutation == "tool": data["choices"][0]["message"]["tool_calls"][0]["function"]["name"] = "shell"
    elif mutation == "truncated": data["choices"][0]["finish_reason"] = "length"
    elif mutation == "usage": data["usage"]["total_tokens"] = 999
    elif mutation == "arguments": data["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "[]"
    with pytest.raises(ValueError):
        translate(data, native)


@pytest.mark.parametrize("mode,harness", [("science_agent", "codex"), ("science_expert", "expert-codex")])
def test_native_preflight_and_configuration_keep_upstream_chat_identity(tmp_path, mode, harness):
    binary = tmp_path / "codex"
    binary.write_text("fixture")
    profile = Profile(mode, harness, ModelBinding("dmxapi", MODEL, "https://fixture.invalid/v1"),
                      Budget(thinking=None, temperature=0.1))
    validate_profile(profile)
    assert builtin_registry().preflight(profile, {"codex_bin": str(binary)})["ready"]
    ctx = SimpleNamespace(profile=profile, task=SimpleNamespace(tools={"protein_probe"}, mcp_entry="fixture_tools.py"),
                          relay=SimpleNamespace(base_url="http://127.0.0.1:12345/v1"))
    text = configuration(ctx)
    assert 'wire_api = "responses"' in text
    assert MODEL in text and profile.model.protocol == "chat_completions"
    with pytest.raises(ValueError):
        validate_profile(replace(profile, budget=replace(profile.budget, reasoning_effort="high")))


def test_gateway_role_and_native_calls_share_one_budget_and_report_upstream_usage(tmp_path):
    async def run():
        seen = []
        async def upstream(req):
            assert req.url.path == "/v1/chat/completions"
            seen.append(json.loads(req.content))
            return httpx.Response(200, json=answer(False))
        log = tmp_path / "requests.jsonl"
        async with ResponsesChatBridge(policy=policy(2), upstream_url="https://fixture.invalid/v1",
                                       api_key="fixture-secret", output=log,
                                       transport=httpx.MockTransport(upstream)) as gateway:
            async with httpx.AsyncClient(trust_env=False) as client:
                headers = {"Authorization": "Bearer " + gateway.token}
                assert (await client.post(gateway.base_url + "/responses", json=request())).status_code == 401
                result = await client.post(gateway.base_url + "/responses", json=request(True), headers=headers)
                assert result.status_code == 200 and b"response.completed" in result.content
                role = await client.post(gateway.base_url + "/chat/completions",
                                         json={"model": MODEL, "messages": [{"role": "user", "content": "Plan"}]},
                                         headers=headers)
                assert role.status_code == 200
                exhausted = await client.post(gateway.base_url + "/responses", json=request(), headers=headers)
                assert exhausted.status_code == 400 and gateway.calls == 2
        assert len(seen) == 2 and all(x["model"] == MODEL and x["max_tokens"] == 4096 for x in seen)
        rows = [json.loads(x) for x in log.read_text().splitlines()]
        assert [x["call"] for x in rows if x["event"] == "request"] == [1, 2]
        assert all(x["usage"]["total_tokens"] == 41 for x in rows if x["event"] == "response")
        assert "fixture-secret" not in log.read_text() and "Measure ACDEFGHIKC" not in log.read_text()
    asyncio.run(run())
