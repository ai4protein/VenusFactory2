"""Recovery of empty length stops must not replay real output or hide costs."""

import asyncio
import copy
import json

import httpx
import pytest

from agent.harness import Budget, ModelBinding, Profile
from agent.harness.contracts import digest
from agent.harness.empty_response_recovery import POLICY, eligible, response_shape
from agent.harness.events import audit_requests
from agent.harness.gateway import EvidenceGateway
from agent.rsi.engine.model_gateway import GatewayPolicy


def reply(content=None, reason="length", **extra):
    return {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content, "reasoning_content": "private-reasoning", **extra},
                "finish_reason": reason,
            }
        ],
        "usage": {"completion_tokens": 8, "completion_tokens_details": {"reasoning_tokens": 8}, "total_tokens": 12},
    }


def raw_response(value, stream):
    if not stream:
        return json.dumps(value).encode()
    chunks = [
        {
            "choices": [
                {"index": c.get("index", 0), "delta": c["message"], "finish_reason": c["finish_reason"]}
                for c in value["choices"]
            ]
        },
        {"choices": [], "usage": value.get("usage")},
    ]
    return b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks) + b"data: [DONE]\n\n"


async def fixture(
    tmp_path,
    *,
    first=None,
    stream=False,
    limit=4,
    policy=POLICY,
    always_empty=False,
    timeout=2,
    mode="science_agent",
    second=None,
):
    profile = Profile(
        mode,
        "langgraph",
        ModelBinding("fixture", "exact-model", "https://provider.example/v1"),
        Budget(model_calls=limit, output_tokens=8, request_seconds=timeout),
        request_recovery=policy,
    )
    lock = {"profile": profile.public()}
    lock["sha256"] = digest(lock)
    (tmp_path / "profile.lock.json").write_text(json.dumps(lock))
    bodies = []

    def upstream(request):
        bodies.append(request.content)
        value = (
            (first if first is not None else reply())
            if len(bodies) == 1 or always_empty
            else (second if second is not None else reply("Completed report", "stop"))
        )
        return httpx.Response(200, content=raw_response(value, stream))

    log = tmp_path / "model_requests.jsonl"
    async with EvidenceGateway(
        policy=GatewayPolicy(
            model="exact-model",
            mode=mode,
            scientific_tools={"submit_model_code"},
            max_calls=limit,
            max_tokens=8,
            thinking="enabled",
        ),
        upstream_url="https://provider.example/v1",
        api_key="private-key",
        output=log,
        request_recovery=policy,
        timeout_s=timeout,
        transport=httpx.MockTransport(upstream),
    ) as relay:
        async with httpx.AsyncClient(trust_env=False) as client:
            response = await client.post(
                relay.base_url + "/chat/completions",
                headers={"Authorization": "Bearer " + relay.token},
                json={
                    "model": "exact-model",
                    "stream": stream,
                    "messages": [
                        {"role": "user", "content": "private-prompt"},
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "previous",
                                    "type": "function",
                                    "function": {"name": "mcp__venusfactory__submit_model_code", "arguments": "{}"},
                                }
                            ],
                        },
                        {"role": "tool", "tool_call_id": "previous", "content": "candidate-already-submitted"},
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "mcp__venusfactory__submit_model_code",
                                "parameters": {"type": "object"},
                            },
                        }
                    ],
                },
            )
    return response, bodies, log


@pytest.mark.parametrize("stream", [False, True])
def test_parser_recognizes_only_empty_length_and_never_retains_text(stream):
    for content in [None, "", " \n"]:
        shape = response_shape(raw_response(reply(content), stream), stream)
        assert eligible(shape)
        assert "private-reasoning" not in json.dumps(shape)
    for value in [
        reply("partial", "length"),
        reply(None, "stop"),
        reply(None, "content_filter"),
        reply(None, "length", refusal="private-refusal"),
        reply(None, "length", tool_calls=[{"function": {"arguments": "partial-private-source"}}]),
        reply(None, "length", function_call={"name": "partial"}),
    ]:
        assert not eligible(response_shape(raw_response(value, stream), stream))
    assert not eligible(response_shape(b"malformed", stream))
    multi = reply()
    multi["choices"].append({**copy.deepcopy(multi["choices"][0]), "index": 1})
    assert not eligible(response_shape(raw_response(multi, stream), stream))


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("mode", ["science_agent", "science_expert"])
async def test_same_request_both_modes_streaming_and_nonstreaming_with_physical_accounting(tmp_path, stream, mode):
    response, bodies, log = await fixture(tmp_path, stream=stream, mode=mode)
    assert response.status_code == 200 and len(bodies) == 2 and bodies[0] == bodies[1]
    body = json.loads(bodies[0])
    assert body["model"] == "exact-model" and body["max_tokens"] == 8
    assert body["thinking"] == {"type": "enabled", "clear_thinking": True}
    assert body["messages"][-1]["content"] == "candidate-already-submitted"
    result = audit_requests(log)
    assert result["passed"] and not result["uninterrupted_passed"]
    assert result["model_calls"] == 2 and result["logical_calls"] == 1 and result["failed_attempts"] == 1
    assert len(result["usage"]) == 2 and result["recovered_calls"] == [
        {"logical_call": 1, "failed_attempt": 1, "successful_attempt": 2}
    ]
    assert not any(
        v in log.read_text()
        for v in ["private-prompt", "private-key", "private-reasoning", "candidate-already-submitted"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "first",
    [
        reply("Answer", "stop"),
        reply("partial", "length"),
        reply(None, "stop"),
        reply(None, "content_filter"),
        reply(None, "length", refusal="private-refusal"),
        reply(
            None, "length", tool_calls=[{"id": "tool", "function": {"name": "submit_model_code", "arguments": "{}"}}]
        ),
    ],
)
async def test_normal_answers_refusals_partial_text_and_tool_calls_are_not_replayed(tmp_path, first):
    response, bodies, log = await fixture(tmp_path, first=first)
    assert response.status_code == 200 and len(bodies) == 1
    assert audit_requests(log)["passed"]
    assert not any(r["event"] == "retry_scheduled" for r in map(json.loads, log.read_text().splitlines()))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options,status,count", [({"always_empty": True}, 200, 2), ({"limit": 1}, 429, 1), ({"timeout": 0.05}, 502, 1)]
)
async def test_repeated_empty_result_and_budget_deadline_are_not_hidden(tmp_path, options, status, count):
    response, bodies, log = await fixture(tmp_path, **options)
    assert response.status_code == status and len(bodies) == count
    assert not audit_requests(log)["passed"]
    assert audit_requests(log)["model_calls"] == count


@pytest.mark.asyncio
async def test_legacy_profile_does_not_silently_gain_retries(tmp_path):
    _, bodies, log = await fixture(tmp_path, policy="none")
    assert len(bodies) == 1 and "response_shape" not in log.read_text()
    assert "request_recovery" not in json.loads((tmp_path / "profile.lock.json").read_text())["profile"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", ["payload", "shape", "binding", "grant", "policy", "missing_lock", "ancestry", "schedule"]
)
async def test_missing_or_forged_policy_and_recovery_evidence_rejected(tmp_path, change):
    _, _, log = await fixture(tmp_path)
    rows = list(map(json.loads, log.read_text().splitlines()))
    path = tmp_path / "profile.lock.json"
    if change == "missing_lock":
        path.unlink()
    elif change in {"grant", "policy"}:
        lock = json.loads(path.read_text())
        if change == "grant":
            lock["profile"]["budget"]["model_calls"] = 1
        else:
            lock["profile"].pop("request_recovery")
        lock["sha256"] = digest({k: v for k, v in lock.items() if k != "sha256"})
        path.write_text(json.dumps(lock))
    else:
        if change == "payload":
            next(x for x in rows if x["event"] == "request" and x["call"] == 2)["request_sha256"] = "changed"
        elif change == "shape":
            next(x for x in rows if x["event"] == "response_shape")["content_empty"] = False
        elif change == "binding":
            next(x for x in rows if x["event"] == "response_shape")["response_sha256"] = "0" * 64
        elif change == "ancestry":
            next(x for x in rows if x["event"] == "request" and x["call"] == 2)["retry_of"] = 99
        elif change == "schedule":
            rows = [x for x in rows if x["event"] != "retry_scheduled"]
        log.write_text("".join(json.dumps(x) + "\n" for x in rows))
    assert not audit_requests(log)["passed"]


def test_policy_is_portable_across_providers_but_not_unknown_protocols():
    p = Profile(
        "science_agent",
        "deepseek-harness",
        ModelBinding("other", "m", "https://example.org/v1"),
        request_recovery=POLICY,
    )
    assert Profile.from_dict(p.public()).public() == p.public()
    with pytest.raises(ValueError, match="chat_completions"):
        Profile(
            "science_agent",
            "codex",
            ModelBinding("other", "m", "https://example.org/v1", protocol="responses"),
            request_recovery=POLICY,
        )


@pytest.mark.asyncio
async def test_retry_that_changes_to_empty_stop_is_not_a_successful_recovery(tmp_path):
    response, bodies, log = await fixture(tmp_path, second=reply(None, "stop"))
    assert response.status_code == 200 and len(bodies) == 2
    result = audit_requests(log)
    assert not result["passed"] and not result["recovered_calls"]


@pytest.mark.parametrize(
    "part", [{"content": "partial output"}, {"tool_calls": [{"index": 0, "function": {"arguments": "partial"}}]}]
)
def test_earlier_streamed_content_or_tool_fragment_cannot_be_discarded(part):
    values = [
        {"choices": [{"index": 0, "delta": part, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]},
    ]
    raw = b"".join(b"data: " + json.dumps(v).encode() + b"\n\n" for v in values) + b"data: [DONE]\n\n"
    assert not eligible(response_shape(raw, True))


@pytest.mark.asyncio
async def test_concurrent_logical_requests_share_the_physical_budget(tmp_path):
    profile = Profile(
        "science_agent",
        "langgraph",
        ModelBinding("fixture", "exact-model", "https://provider.example/v1"),
        Budget(model_calls=3),
        request_recovery=POLICY,
    )
    lock = {"profile": profile.public()}
    lock["sha256"] = digest(lock)
    (tmp_path / "profile.lock.json").write_text(json.dumps(lock))
    calls = []

    def upstream(request):
        calls.append(request.content)
        return httpx.Response(200, json=reply() if len(calls) <= 2 else reply("done", "stop"))

    log = tmp_path / "model_requests.jsonl"
    async with EvidenceGateway(
        policy=GatewayPolicy(model="exact-model", mode="science_agent", scientific_tools=set(), max_calls=3),
        upstream_url="https://provider.example/v1",
        api_key="fixture",
        output=log,
        request_recovery=POLICY,
        transport=httpx.MockTransport(upstream),
    ) as relay:
        async with httpx.AsyncClient(trust_env=False) as client:

            async def call():
                return await client.post(
                    relay.base_url + "/chat/completions",
                    headers={"Authorization": "Bearer " + relay.token},
                    json={"model": "exact-model", "messages": []},
                )

            responses = await asyncio.gather(call(), call())
    assert sorted(response.status_code for response in responses) == [200, 429]
    assert len(calls) == audit_requests(log)["model_calls"] == 3
    assert not audit_requests(log)["passed"]
