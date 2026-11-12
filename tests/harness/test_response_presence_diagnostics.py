"""Record presence without promoting private reasoning or changing transport policy."""
import json

import httpx
import pytest

from agent.harness.failure_summary import public_failure, research_failure
from agent.harness.gateway import EvidenceGateway
from agent.harness.provider_response_diagnostics import response_evidence
from agent.rsi.engine.model_gateway import GatewayPolicy


def response(message, streaming, finish="stop"):
    data = {"id": "synthetic-private-id", "model": "test", "choices": [
        {"index": 0, "delta" if streaming else "message": {"role": "assistant", "content": None, **message},
         "finish_reason": finish}], "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}
    encoded = json.dumps(data).encode()
    return b"data: " + encoded + b"\n\ndata: [DONE]\n\n" if streaming else encoded


def save_response(tmp_path, shape, **changes):
    path = tmp_path / "requests.jsonl"
    row = {"event": "response", "status": 200, "completed": True, "response_completion": shape, **changes}
    path.write_text(json.dumps(row) + "\n")
    return path


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("field", ["reasoning_content", "reasoning", "refusal"])
def test_private_text_records_only_its_presence(tmp_path, streaming, field):
    text = "private-provider-text-that-must-not-enter-diagnostics"
    shape = response_evidence(response({field: text}, streaming), streaming)
    assert shape["reasoning_present"] is (field != "refusal")
    assert shape["refusal_present"] is (field == "refusal")
    assert shape["content_present"] is False and shape["tools_present"] is False
    assert text not in json.dumps(shape) and "synthetic-private-id" not in json.dumps(shape)
    path = save_response(tmp_path, shape)
    expected = "model_response_refused" if field == "refusal" else "model_response_empty"
    assert public_failure(path)["code"] == expected
    assert research_failure(path, RuntimeError("runtime_error"))["code"] == expected
    assert text not in path.read_text()


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning", "refusal"])
@pytest.mark.parametrize("value", [None, "", " \n ", {}, ["not-a-supported-text-field"]])
def test_empty_or_nontext_extension_is_not_positive_presence(field, value):
    shape = response_evidence(response({field: value}, False), False)
    assert shape["reasoning_present"] is False and shape["refusal_present"] is False


def test_presence_accumulates_across_sse_frames_without_retaining_private_text():
    first = response({"reasoning_content": "private reasoning"}, True).split(b"data: [DONE]")[0]
    last = response({"refusal": "private refusal"}, True)
    shape = response_evidence(first + last, True)
    assert shape["reasoning_present"] and shape["refusal_present"]
    assert not shape["content_present"] and not shape["tools_present"]
    assert "private reasoning" not in json.dumps(shape) and "private refusal" not in json.dumps(shape)


@pytest.mark.parametrize("finish", ["stop", "length", "content_filter"])
def test_explicit_refusal_with_public_content_is_not_mislabeled_empty(tmp_path, finish):
    shape = response_evidence(response({"content": "Public notice", "refusal": "Private refusal"}, False, finish), False)
    assert public_failure(save_response(tmp_path, shape))["code"] == "model_response_refused"


@pytest.mark.parametrize("change", [{"malformed": True}, {"provider_error": True}, {"done": False},
                                     {"finish_reasons": ["unknown"]}])
def test_ambiguous_completion_does_not_gain_refusal_diagnosis(tmp_path, change):
    shape = response_evidence(response({"refusal": "Private refusal"}, True), True)
    shape.update(change)
    assert public_failure(save_response(tmp_path, shape))["code"] == "runtime_error"


def test_legacy_presence_metadata_and_specific_errors_keep_existing_behavior(tmp_path):
    shape = response_evidence(response({}, False), False)
    shape.pop("refusal_present")
    shape.pop("reasoning_present")
    path = save_response(tmp_path, shape)
    assert public_failure(path)["code"] == "model_response_empty"
    specific = ValueError("Source module lacks fit/predict")
    assert research_failure(path, specific) == {"error_type": "ValueError", "message": str(specific)}
    refused = response_evidence(response({"refusal": "Private refusal"}, False), False)
    assert public_failure(save_response(tmp_path, refused, completed=False, status=403))["code"] == "model_access_403"


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("field", ["reasoning_content", "refusal"])
async def test_gateway_delivers_same_response_and_keeps_physical_budget(tmp_path, streaming, field):
    text = "private-response-text"
    raw = response({field: text}, streaming)
    dispatched = []

    def provider(request):
        dispatched.append(request)
        return httpx.Response(200, content=raw)

    log = tmp_path / "requests.jsonl"
    policy = GatewayPolicy(model="test", mode="science_agent", scientific_tools=set(), max_calls=1)
    async with EvidenceGateway(policy=policy, upstream_url="https://provider.example/v1", api_key="fixture-key",
                               output=log, transport=httpx.MockTransport(provider)) as relay:
        async with httpx.AsyncClient() as client:
            result = await client.post(relay.base_url + "/chat/completions",
                headers={"Authorization": "Bearer " + relay.token},
                json={"model": "test", "messages": [{"role": "user", "content": "synthetic"}], "stream": streaming})
        assert relay.calls == 1
    assert result.status_code == 200 and len(dispatched) == 1
    # This diagnostic change neither removes provider fields nor turns reasoning into content.
    actual = result.content.split(b"data: ", 1)[1].split(b"\n", 1)[0] if streaming else result.content
    delivered = json.loads(actual)["choices"][0]["delta" if streaming else "message"]
    assert delivered[field] == text and delivered["content"] is None
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [row["event"] for row in rows].count("request") == 1
    terminal = next(row for row in rows if row["event"] == "response")
    assert terminal["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    assert terminal["response_completion"]["response_bytes"] == len(raw)
    assert text not in log.read_text() and "fixture-key" not in log.read_text()
