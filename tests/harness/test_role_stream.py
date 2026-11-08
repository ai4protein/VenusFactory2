import json

import httpx
import pytest

from agent.harness.role_stream import read_role_stream
from agent.harness.native_expert_team import role_content


def stream(rows, done=True):
    return httpx.Response(200, content="".join("data: " + json.dumps(row) + "\n\n" for row in rows)
                          + ("data: [DONE]\n\n" if done else ""),
                          headers={"content-type": "text/event-stream"})


@pytest.mark.asyncio
async def test_scientific_text_uses_complete_stream_and_preserves_split_unicode_json():
    response = stream([
        {"model": "free", "choices": [{"index": 0, "delta": {"role": "assistant", "content": '{"summary": "'}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "蛋白研究"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": '"}'}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"completion_tokens": 12}},
    ])
    value = await read_role_stream(response)
    assert json.loads(role_content(value, "chat_completions", "free")) == {"summary": "蛋白研究"}


@pytest.mark.asyncio
@pytest.mark.parametrize("rows,done", [
    ([{"choices": [{"delta": {"content": "unfinished"}, "finish_reason": None}]}], True),
    ([{"choices": [{"delta": {"content": "missing terminal"}, "finish_reason": "stop"}]}], False),
    ([{"choices": [{"delta": {"tool_calls": [{"id": "unallowed"}]}, "finish_reason": "tool_calls"}]}], True),
    ([{"error": {"message": "upstream unavailable"}}], True),
    ([{"choices": [{"index": 1, "delta": {"content": "extra"}, "finish_reason": "stop"}]}], True),
])
async def test_incomplete_or_executable_role_streams_are_rejected(rows, done):
    with pytest.raises(ValueError):
        await read_role_stream(stream(rows, done))


@pytest.mark.asyncio
async def test_truncation_is_not_promoted_to_success():
    value = await read_role_stream(stream([{"choices": [
        {"delta": {"content": "truncated"}, "finish_reason": "length"}]}]))
    with pytest.raises(ValueError):
        role_content(value, "chat_completions", "free")
