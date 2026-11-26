import asyncio
import copy
import json

import httpx
import pytest

from agent.harness.events import audit_requests
from agent.harness.messages_bridge import MessagesBridge, chat_to_message, message_sse, messages_to_chat
from agent.rsi.engine.model_gateway import GatewayPolicy

MODEL = "unisound-u2-flash"
TOOL = "mcp__venusfactory__protein_probe"


def policy(calls=2):
    return GatewayPolicy(
        model=MODEL,
        mode="science_agent",
        scientific_tools={"protein_probe"},
        max_calls=calls,
        max_tokens=8192,
        thinking="disabled",
    )


def request():
    return {
        "model": MODEL,
        "max_tokens": 8192,
        "stream": True,
        "system": [{"type": "text", "text": "Scientist fixture", "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": "Measure ACDEFGHIKC"}],
        "tools": [
            {
                "name": TOOL,
                "description": "Measure protein counts",
                "input_schema": {
                    "type": "object",
                    "properties": {"sequence": {"type": "string"}},
                    "required": ["sequence"],
                },
            }
        ],
    }


def answer(tool=True):
    msg = {"role": "assistant", "content": None if tool else "10 residues, 2 cysteines"}
    if tool:
        msg["tool_calls"] = [
            {"id": "c1", "type": "function", "function": {"name": TOOL, "arguments": '{"sequence":"ACDEFGHIKC"}'}}
        ]
    return {
        "id": "fixture1",
        "model": MODEL,
        "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if tool else "stop"}],
        "usage": {"prompt_tokens": 31, "completion_tokens": 10, "total_tokens": 41},
    }


def translate(data):
    return chat_to_message(json.dumps(data).encode(), model=MODEL, allowed_tools={TOOL}, max_tokens=8192)


def test_paired_round_trip_preserves_arguments_and_error_result():
    native, usage = translate(answer())
    body = request()
    body["messages"] += [
        {"role": "assistant", "content": native["content"]},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "c1",
                    "is_error": True,
                    "content": [{"type": "text", "text": "fixture failure"}],
                },
                {"type": "text", "text": "Try another admitted argument"},
            ],
        },
    ]
    chat, metadata = messages_to_chat(body, policy())
    assert chat["stream"] is False
    assert chat["messages"][-2] == {
        "role": "tool",
        "tool_call_id": "c1",
        "content": '{"is_error": true, "content": "fixture failure"}',
    }
    assert chat["messages"][-1]["role"] == "user"
    assert json.loads(chat["messages"][-3]["tool_calls"][0]["function"]["arguments"]) == {"sequence": "ACDEFGHIKC"}
    assert usage["total_tokens"] == 41 and metadata["cache_hints_forwarded"] is False
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.parametrize(
    "change",
    [
        {"model": "paid-model"},
        {"max_tokens": 32000},
        {"max_tokens": True},
        {"stream": "true"},
        {"thinking": {"type": "adaptive"}},
        {"output_config": {"effort": "high"}},
        {"context_management": {"edits": [{"type": "compact"}]}},
        {"betas": ["unverified"]},
        {"tools": [{"name": "Bash", "input_schema": {}}]},
        {"tools": [{"name": TOOL, "input_schema": {}, "type": "computer"}]},
        {"messages": [{"role": "user", "content": [{"type": "image", "source": {}}]}]},
        {"messages": [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "absent", "content": "x"}]}]},
        {"tool_choice": {"type": "tool", "name": "Bash"}},
    ],
)
def test_rejects_unmatched_native_semantics(change):
    body = request()
    body.update(change)
    with pytest.raises(ValueError):
        messages_to_chat(body, policy())


def test_rejects_unfinished_duplicate_and_out_of_order_history():
    msg, _ = translate(answer())
    body = request()
    body["messages"].append({"role": "assistant", "content": msg["content"]})
    with pytest.raises(ValueError):
        messages_to_chat(body, policy())
    body["messages"].append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "too early"},
                {"type": "tool_result", "tool_use_id": "c1", "content": "{}"},
            ],
        }
    )
    with pytest.raises(ValueError):
        messages_to_chat(body, policy())
    body["messages"][-1]["content"] = [{"type": "tool_result", "tool_use_id": "c1", "content": "{}"}] * 2
    with pytest.raises(ValueError):
        messages_to_chat(body, policy())


@pytest.mark.parametrize("field,value", [("model", "different"), ("id", ""), ("choices", []), ("usage", None)])
def test_rejects_incomplete_response_identity(field, value):
    data = answer()
    data[field] = value
    with pytest.raises(ValueError):
        translate(data)


@pytest.mark.parametrize("reason", [None, "length", "content_filter", "stop"])
def test_never_delivers_partial_or_mismatched_tool_response(reason):
    data = answer()
    data["choices"][0]["finish_reason"] = reason
    with pytest.raises(ValueError):
        translate(data)


@pytest.mark.parametrize(
    "mutation",
    ["unknown_tool", "duplicate", "bad_json", "array_args", "nan_args", "overspend", "bad_total", "bool_usage"],
)
def test_rejects_malformed_tool_and_usage(mutation):
    data = answer()
    calls = data["choices"][0]["message"]["tool_calls"]
    if mutation == "unknown_tool":
        calls[0]["function"]["name"] = "Bash"
    elif mutation == "duplicate":
        calls.append(copy.deepcopy(calls[0]))
    elif mutation == "bad_json":
        calls[0]["function"]["arguments"] = '{"x":'
    elif mutation == "array_args":
        calls[0]["function"]["arguments"] = "[]"
    elif mutation == "nan_args":
        calls[0]["function"]["arguments"] = '{"x":NaN}'
    elif mutation == "overspend":
        data["usage"].update(completion_tokens=8193, total_tokens=8224)
    elif mutation == "bad_total":
        data["usage"]["total_tokens"] = 1
    elif mutation == "bool_usage":
        data["usage"]["prompt_tokens"] = True
    with pytest.raises(ValueError):
        translate(data)


def test_sse_matches_complete_answer_and_usage():
    msg, _ = translate(answer())
    raw = message_sse(msg)
    rows = [json.loads(line[6:]) for line in raw.decode().splitlines() if line.startswith("data: ")]
    assert rows[0]["type"] == "message_start" and rows[-1]["type"] == "message_stop"
    assert rows[0]["message"]["usage"] == {"input_tokens": 31, "output_tokens": 0}
    assert rows[-2]["usage"] == {"output_tokens": 10}
    assert json.loads(rows[2]["delta"]["partial_json"]) == {"sequence": "ACDEFGHIKC"}


def test_gateway_enforces_real_request_count_and_no_key_leak(tmp_path):
    async def run():
        seen = []

        async def upstream(req):
            seen.append(json.loads(req.content))
            assert req.headers["Authorization"] == "Bearer fixture-upstream-secret"
            return httpx.Response(200, json=answer())

        output = tmp_path / "requests.jsonl"
        async with MessagesBridge(
            policy=policy(1),
            upstream_url="https://fixture.invalid/v1",
            api_key="fixture-upstream-secret",
            output=output,
            transport=httpx.MockTransport(upstream),
        ) as gateway:
            async with httpx.AsyncClient(trust_env=False) as client:
                url = gateway.base_url + "/messages"
                bad = await client.post(url, json=request(), headers={"x-api-key": "wrong"})
                assert bad.status_code == 401
                headers = {"x-api-key": gateway.token}
                invalid = request()
                invalid["model"] = "paid-model"
                assert (await client.post(url, json=invalid, headers=headers)).status_code == 400
                result = await client.post(url, json=request(), headers=headers)
                assert result.status_code == 200 and b"tool_use" in result.content
                assert (await client.post(url, json=request(), headers=headers)).status_code == 400
                assert gateway.calls == 1 and len(seen) == 1
        text = output.read_text()
        assert "fixture-upstream-secret" not in text and "Measure ACDEFGHIKC" not in text
        assert (
            seen[0]["stream"] is False and seen[0]["max_tokens"] == 8192 and seen[0]["thinking"]["type"] == "disabled"
        )
        audit = audit_requests(output)
        assert audit["model_calls"] == 1 and audit["passed"] is False

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["http", "truncated", "undeclared"])
def test_gateway_failure_no_partial_tool_delivery_or_retry(tmp_path, failure):
    async def run():
        attempts = []

        async def upstream(req):
            attempts.append(req)
            if failure == "http":
                return httpx.Response(503, json={"error": {"message": "fixture failure"}})
            data = answer()
            if failure == "truncated":
                data["choices"][0]["finish_reason"] = "length"
            else:
                data["choices"][0]["message"]["tool_calls"][0]["function"]["name"] = "Bash"
            return httpx.Response(200, json=data)

        output = tmp_path / "requests.jsonl"
        async with MessagesBridge(
            policy=policy(),
            upstream_url="https://fixture.invalid/v1",
            api_key="fixture",
            output=output,
            transport=httpx.MockTransport(upstream),
        ) as gateway:
            async with httpx.AsyncClient(trust_env=False) as client:
                response = await client.post(
                    gateway.base_url + "/messages", json=request(), headers={"x-api-key": gateway.token}
                )
                assert response.status_code in {502, 503} and b"content_block_start" not in response.content
        assert len(attempts) == 1
        assert audit_requests(output)["passed"] is False
        responses = [
            json.loads(line) for line in output.read_text().splitlines() if json.loads(line)["event"] == "response"
        ]
        assert len(responses) == 1 and responses[0]["completed"] is False

    asyncio.run(run())


def test_shutdown_retains_cancelled_physical_request(tmp_path):
    async def run():
        entered = asyncio.Event()

        async def upstream(req):
            entered.set()
            await asyncio.Event().wait()

        output = tmp_path / "requests.jsonl"
        gateway = MessagesBridge(
            policy=policy(),
            upstream_url="https://fixture.invalid/v1",
            api_key="fixture",
            output=output,
            transport=httpx.MockTransport(upstream),
        )
        await gateway.__aenter__()
        async with httpx.AsyncClient(trust_env=False) as client:
            task = asyncio.create_task(
                client.post(gateway.base_url + "/messages", json=request(), headers={"x-api-key": gateway.token})
            )
            await asyncio.wait_for(entered.wait(), 3)
            await gateway.__aexit__(None, None, None)
            await asyncio.gather(task, return_exceptions=True)
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        response = [r for r in rows if r["event"] == "response"]
        assert len(response) == 1 and response[0]["error_type"] == "CancelledError" and not response[0]["completed"]
        assert audit_requests(output)["model_calls"] == 1

    asyncio.run(run())


@pytest.mark.parametrize(
    "origin",
    [
        None,
        "https://fixture.invalid/v1",
        "https://www.dmxapi.cn.invalid/v1",
        "http://www.dmxapi.cn/v1",
        "https://www.dmxapi.cn/v2",
    ],
)
def test_dmx_response_alias_rejected_without_exact_origin(origin):
    data = answer()
    data["model"] = "u2-flash"
    with pytest.raises(ValueError, match="identity mismatch"):
        chat_to_message(
            json.dumps(data).encode(), model=MODEL, allowed_tools={TOOL}, max_tokens=8192, upstream_url=origin
        )


@pytest.mark.parametrize("requested,returned", [("glm-4-flash", "u2-flash"), (MODEL, "paid-model"), (MODEL, "u2-pro")])
def test_dmx_alias_does_not_admit_other_models(requested, returned):
    data = answer()
    data["model"] = returned
    with pytest.raises(ValueError, match="identity mismatch"):
        chat_to_message(
            json.dumps(data).encode(),
            model=requested,
            allowed_tools={TOOL},
            max_tokens=8192,
            upstream_url="https://www.dmxapi.cn/v1",
        )


def test_exact_dmx_alias_preserves_request_and_records_response_identity(tmp_path):
    async def run():
        seen = []

        async def upstream(req):
            seen.append(json.loads(req.content))
            data = answer()
            data["model"] = "u2-flash"
            return httpx.Response(200, json=data)

        output = tmp_path / "requests.jsonl"
        async with MessagesBridge(
            policy=policy(1),
            upstream_url="https://www.dmxapi.cn/v1",
            api_key="fixture",
            output=output,
            transport=httpx.MockTransport(upstream),
        ) as gateway:
            async with httpx.AsyncClient(trust_env=False) as client:
                native = request()
                native["stream"] = False
                response = await client.post(
                    gateway.base_url + "/messages", json=native, headers={"x-api-key": gateway.token}
                )
                assert response.status_code == 200
                assert response.json()["model"] == MODEL
                assert response.json()["content"][0]["type"] == "tool_use"
        assert len(seen) == 1 and seen[0]["model"] == MODEL
        assert audit_requests(output)["passed"]
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        receipt = [row for row in rows if row["event"] == "response"][0]
        assert receipt["requested_model"] == MODEL and receipt["response_model"] == "u2-flash"
        assert receipt["response_model_alias_used"] is True

    asyncio.run(run())
