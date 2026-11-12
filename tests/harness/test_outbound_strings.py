"""Nested transport encodings must neither invent paths nor hide sensitive text."""

import json
import re

import pytest

from agent.harness.outbound_strings import decoded_strings

HOST_PATH = re.compile(r"/(?:home|inspire|root|mnt)/|(?<![\w])[A-Za-z]:[\\/]")


def nested(source):
    args = json.dumps({"path": "inspect.py", "content": source})
    result = json.dumps({"text": source}) + "\nExecution budget: 8 calls remain."
    return {"messages": [{"tool_calls": [{"function": {"arguments": args}}]}, {"content": result}]}


def test_code_newlines_are_not_windows_paths_even_in_nested_tool_json():
    source = 'with open("summary.json", "w") as f:\n    f.write("done")\n'
    payload = nested(source)
    strings = list(decoded_strings(payload))
    assert strings.count(source) == 2
    assert not any(HOST_PATH.search(text) for text in strings)
    assert any("Execution budget" in text for text in strings)
    assert json.loads(payload["messages"][0]["tool_calls"][0]["function"]["arguments"])["content"] == source


@pytest.mark.parametrize(
    "text",
    [
        "/home/user/private/file",
        "/mnt/secret",
        r"C:\Users\private",
        r"D:\private",
        "C:/private",
        "not-a-real-key-0123456789",
    ],
)
@pytest.mark.parametrize("mode", ["raw", "json_string", "nested", "suffix", "key"])
def test_real_paths_and_sensitive_values_remain_visible(text, mode):
    values = {
        "raw": text,
        "json_string": json.dumps(text),
        "nested": nested(text),
        "suffix": json.dumps({"ok": True}) + "\n" + text,
        "key": {text: 1},
    }
    found = list(decoded_strings(values[mode]))
    assert any(text in value for value in found)
    if "not-a-real-key" not in text:
        assert any(HOST_PATH.search(value) for value in found)


def test_multiple_json_prefixes_and_invalid_json_preserve_the_remaining_content():
    text = '{"value":"safe"} {"next":"/home/user/private"}'
    assert "/home/user/private" in list(decoded_strings(text))
    malformed = '{"source": "/home/user/private"'
    assert malformed in list(decoded_strings(malformed))


def test_deep_or_cyclic_content_fails_closed():
    cycle = []
    cycle.append(cycle)
    with pytest.raises(ValueError, match="nesting"):
        list(decoded_strings(cycle))
