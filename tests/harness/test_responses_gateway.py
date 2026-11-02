import asyncio
import copy
import json

import httpx
import pytest

from agent.harness.events import audit_requests
from agent.harness.responses_gateway import ResponsesGateway, ResponsesPolicy, inspect_responses

NAME = "mcp__venusfactory__inspect_task"


def response(tool=False):
    item = (
        {
            "id": "item1",
            "type": "function_call",
            "call_id": "call1",
            "name": NAME,
            "arguments": '{"x":1}',
            "status": "completed",
        }
        if tool
        else {
            "id": "item1",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "fixture answer", "annotations": []}],
        }
    )
    return {
        "id": "resp1",
        "model": "fixture-model",
        "status": "completed",
        "output": [item],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }


def events(tool=False):
    final = response(tool)
    item = final["output"][0]
    initial = {**item, "status": "in_progress"}
    initial["arguments" if tool else "content"] = "" if tool else []
    rows = [
        {"type": "response.created", "response": {"id": "resp1", "status": "in_progress"}},
        {"type": "response.output_item.added", "output_index": 0, "item": initial},
    ]
    if tool:
        rows.extend(
            [
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": "item1",
                    "output_index": 0,
                    "delta": '{"x":',
                },
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": "item1",
                    "output_index": 0,
                    "delta": "1}",
                },
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": "item1",
                    "output_index": 0,
                    "arguments": '{"x":1}',
                },
            ]
        )
    else:
        rows.extend(
            [
                {
                    "type": "response.output_text.delta",
                    "item_id": "item1",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "fixture ",
                },
                {
                    "type": "response.output_text.delta",
                    "item_id": "item1",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "answer",
                },
                {
                    "type": "response.output_text.done",
                    "item_id": "item1",
                    "output_index": 0,
                    "content_index": 0,
                    "text": "fixture answer",
                },
            ]
        )
    rows.extend(
        [
            {"type": "response.output_item.done", "output_index": 0, "item": item},
            {"type": "response.completed", "response": final},
        ]
    )
    return [{**row, "sequence_number": i} for i, row in enumerate(rows)]


def stream(rows):
    return b"".join(("event: " + row["type"] + "\ndata: " + json.dumps(row) + "\n\n").encode() for row in rows)


def inspect(raw, streaming):
    return inspect_responses(raw, streaming, model="fixture-model", allowed_tools={NAME}, max_output_tokens=100)


def policy(**kwargs):
    return ResponsesPolicy(model="fixture-model", scientific_tools={"inspect_task"}, **kwargs)


def test_explicit_model_parameters_tools_and_opaque_reasoning_are_preserved():
    payload = {
        "model": "fixture-model",
        "input": [{"type": "reasoning", "encrypted_content": "opaque-fixture"}],
        "tools": [
            {"type": "function", "name": NAME, "parameters": {"type": "object"}},
            {"type": "function", "name": "exec_command"},
            {"type": "web_search"},
        ],
        "reasoning": {"effort": "high", "summary": "auto"},
        "temperature": 0.9,
        "top_p": 0.2,
        "seed": 5,
        "max_output_tokens": 9999,
    }
    before = copy.deepcopy(payload)
    sent, metadata = policy(max_output_tokens=128, reasoning_effort="low").apply(payload)
    assert sent["model"] == before["model"] and sent["input"] == before["input"]
    assert payload == before
    assert sent["tools"] == [before["tools"][0]]
    assert sent["reasoning"] == {"effort": "low", "summary": "auto"}
    assert sent["store"] is False and sent["max_output_tokens"] == 128
    assert not {"temperature", "top_p", "seed"} & set(sent)
    assert metadata["tools_sent"] == [NAME]
    assert "opaque-fixture" not in json.dumps(metadata)


@pytest.mark.parametrize(
    "extra",
    [
        {"model": "different-model"},
        {"messages": []},
        {"max_tokens": 3},
        {"thinking": {}},
        {"previous_response_id": "hidden"},
        {"conversation": "hidden"},
        {"store": True},
        {"background": True},
        {"stream": "true"},
        {"store": 0},
        {"tools": ["invalid"]},
        {"tools": [{"type": "function", "name": NAME}] * 2},
        {"tool_choice": {"type": "function", "name": "exec_command"}},
        {"tool_choice": "required"},
        {"reasoning": {"effort": "low", "unknown": "field"}},
    ],
)
def test_unsupported_or_ambiguous_request_rejected(extra):
    with pytest.raises((ValueError, TypeError)):
        policy().apply({"model": "fixture-model", "input": "fixture", **extra})


@pytest.mark.parametrize("tool", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
def test_complete_message_and_function_evidence(tool, streaming):
    raw = stream(events(tool)) if streaming else json.dumps(response(tool)).encode()
    assert inspect(raw, streaming)["usage"]["output_tokens"] == 5


@pytest.mark.parametrize(
    "attack",
    [
        "missing_completed",
        "missing_created",
        "duplicate_event",
        "reorder",
        "post_terminal",
        "mixed_id",
        "incomplete",
        "provider_error",
        "item_mismatch",
        "delta_mismatch",
        "no_item_done",
        "no_text_done",
        "wrong_item_id",
        "usage",
        "empty",
        "wrong_model",
        "truncated",
    ],
)
def test_sse_failure_never_counts_as_completion(attack):
    rows = events()
    if attack == "missing_completed":
        rows.pop()
    elif attack == "missing_created":
        rows.pop(0)
    elif attack == "duplicate_event":
        rows.insert(3, copy.deepcopy(rows[2]))
    elif attack == "reorder":
        rows[2], rows[3] = rows[3], rows[2]
    elif attack == "post_terminal":
        rows.append(copy.deepcopy(rows[0]))
    elif attack == "mixed_id":
        rows[-1]["response"]["id"] = "another"
    elif attack == "incomplete":
        rows[-1]["response"]["status"] = "incomplete"
    elif attack == "provider_error":
        rows[3] = {"type": "error", "sequence_number": 3}
    elif attack == "item_mismatch":
        rows[-2]["item"] = {**rows[-2]["item"], "role": "user"}
    elif attack == "delta_mismatch":
        rows[2]["delta"] = "forged"
    elif attack == "no_item_done":
        rows.pop(-2)
    elif attack == "no_text_done":
        rows.pop(4)
    elif attack == "wrong_item_id":
        rows[2]["item_id"] = "other"
    elif attack == "usage":
        rows[-1]["response"]["usage"]["output_tokens"] = 10000
    elif attack == "empty":
        rows[-1]["response"]["output"] = []
    elif attack == "wrong_model":
        rows[-1]["response"]["model"] = "another-model"
    raw = stream(rows)
    if attack == "truncated":
        raw = raw[:-1]
    with pytest.raises((ValueError, TypeError)):
        inspect(raw, True)


def test_undeclared_tool_and_nonobject_arguments_are_rejected():
    for change in ({"name": "exec_command"}, {"arguments": "[]"}, {"call_id": ""}):
        value = response(True)
        value["output"][0].update(change)
        with pytest.raises(ValueError):
            inspect(json.dumps(value).encode(), False)


@pytest.mark.asyncio
async def test_real_http_relay_keeps_wire_bytes_and_counts_physical_requests(tmp_path):
    raw = stream(events(True))
    seen = []

    async def upstream(request):
        seen.append(json.loads(request.content))
        assert request.url.path == "/v1/responses"
        assert request.headers["Authorization"] == "Bearer upstream-test-secret"
        return httpx.Response(200, content=raw, headers={"content-type": "text/event-stream"})

    log = tmp_path / "requests.jsonl"
    async with ResponsesGateway(
        policy=policy(max_calls=1),
        upstream_url="https://fixture.invalid/v1",
        api_key="upstream-test-secret",
        output=log,
        transport=httpx.MockTransport(upstream),
    ) as relay:
        async with httpx.AsyncClient(trust_env=False) as client:
            args = {
                "headers": {"Authorization": "Bearer " + relay.token},
                "json": {
                    "model": "fixture-model",
                    "input": "private-fixture-prompt",
                    "stream": True,
                    "tools": [{"type": "function", "name": NAME}],
                },
            }
            result = await client.post(relay.base_url + "/responses", **args)
            assert result.status_code == 200 and result.content == raw
            assert audit_requests(log)["passed"]
            assert (await client.post(relay.base_url + "/responses", **args)).status_code == 429
            assert (await client.post(relay.base_url + "/chat/completions", **args)).status_code == 404
            assert (await client.post(relay.base_url + "/responses", json=args["json"])).status_code == 401
    assert len(seen) == 1
    evidence = log.read_text()
    assert "upstream-test-secret" not in evidence and "private-fixture-prompt" not in evidence
    assert not audit_requests(log)["passed"] and audit_requests(log)["model_calls"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 403])
async def test_failure_does_not_relay_partial_content_or_retry(tmp_path, status):
    calls = []

    async def upstream(request):
        calls.append(request)
        return httpx.Response(
            status, content=stream(events()[:-1]) if status == 200 else b'{"error":"sensitive-provider-body"}'
        )

    log = tmp_path / "requests.jsonl"
    async with ResponsesGateway(
        policy=policy(),
        upstream_url="https://fixture.invalid/v1",
        api_key="test",
        output=log,
        transport=httpx.MockTransport(upstream),
    ) as relay:
        async with httpx.AsyncClient(trust_env=False) as client:
            result = await client.post(
                relay.base_url + "/responses",
                json={"model": "fixture-model", "input": "fixture", "stream": True},
                headers={"Authorization": "Bearer " + relay.token},
            )
    assert result.status_code == (502 if status == 200 else 403)
    assert "fixture answer" not in result.text and "sensitive-provider-body" not in result.text + log.read_text()
    assert len(calls) == 1 and not audit_requests(log)["passed"]


@pytest.mark.asyncio
async def test_absolute_deadline_stops_heartbeat_stream(tmp_path):
    closed = asyncio.Event()

    class Endless(httpx.AsyncByteStream):
        async def __aiter__(self):
            while True:
                yield b": heartbeat\n\n"
                await asyncio.sleep(0.01)

        async def aclose(self):
            closed.set()

    async def upstream(_):
        return httpx.Response(200, stream=Endless())

    log = tmp_path / "requests.jsonl"
    async with ResponsesGateway(
        policy=policy(),
        upstream_url="https://fixture.invalid/v1",
        api_key="test",
        output=log,
        timeout_s=0.08,
        transport=httpx.MockTransport(upstream),
    ) as relay:
        async with httpx.AsyncClient(trust_env=False) as client:
            result = await client.post(
                relay.base_url + "/responses",
                json={"model": "fixture-model", "input": "fixture", "stream": True},
                headers={"Authorization": "Bearer " + relay.token},
            )
    assert result.status_code == 502 and closed.is_set()
    assert not audit_requests(log)["passed"]


@pytest.mark.asyncio
async def test_shutdown_waits_for_cancelled_request_evidence(tmp_path):
    started, cleaned = asyncio.Event(), asyncio.Event()

    async def upstream(_):
        started.set()
        try:
            await asyncio.Future()
        finally:
            await asyncio.sleep(0.03)
            cleaned.set()

    log = tmp_path / "requests.jsonl"
    relay = ResponsesGateway(
        policy=policy(),
        upstream_url="https://fixture.invalid/v1",
        api_key="test",
        output=log,
        timeout_s=10,
        transport=httpx.MockTransport(upstream),
    )
    await relay.__aenter__()
    async with httpx.AsyncClient(trust_env=False) as client:
        pending = asyncio.create_task(
            client.post(
                relay.base_url + "/responses",
                json={"model": "fixture-model", "input": "fixture"},
                headers={"Authorization": "Bearer " + relay.token},
            )
        )
        await asyncio.wait_for(started.wait(), 2)
        await relay.__aexit__(None, None, None)
        assert cleaned.is_set()
        before = log.read_bytes()
        await asyncio.sleep(0.06)
        assert log.read_bytes() == before and not audit_requests(log)["passed"]
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
