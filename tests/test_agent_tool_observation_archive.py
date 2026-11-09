"""Long tool records remain retrievable across harnesses and resumed turns."""

import copy
import json

import pytest

from agent.session_store import persistable_state
from backend.web_v2.agent_conversation import completed_messages, continuation_messages, record_tool_observation
from utils.scoped_files import read_bytes


def record(pending, arguments, result, workspace, harness="codex", failed=False):
    record_tool_observation(
        pending,
        {"kind": "tool_call_start", "tool_call_id": "c1", "tool_name": "predict_with_evolved_model", "args": arguments},
        harness,
        workspace=workspace,
    )
    record_tool_observation(
        pending,
        {"kind": "tool_result", "tool_call_id": "c1", "is_error": failed, "output": result},
        harness,
        workspace=workspace,
    )
    row = pending["observations"][-1]
    text = row.get("data", row)["content"]
    assert len(text.encode("utf-8")) <= 12000
    return json.loads(text.split("\n", 1)[1]), text


@pytest.mark.parametrize("harness", ["kimi-code", "codex", "deepseek-harness", "claude-code", "langgraph"])
@pytest.mark.parametrize("serialized", [False, True])
def test_large_prediction_input_keeps_result_and_full_record_after_reload(tmp_path, harness, serialized):
    args = {"inputs": [{"id": f"sample-{i}", "sequence": "ACDEFGHIKLMNPQRSTVWY" * 20} for i in range(256)]}
    result = {
        "success": True,
        "predictions": [{"id": f"sample-{i}", "prediction": 0.9942129558371191 + i} for i in range(256)],
        "scope": {
            "protein": "fixture",
            "unit": "ratio_to_reference",
            "covariates": [{"name": "temperature", "minimum": 21.8, "maximum": 65.5, "unit": "C"}],
        },
        "result_files": {"json": "results/predictions/fixed/result.json"},
    }
    output = json.dumps(result) if serialized else result
    before = copy.deepcopy((args, output))
    pending = {"text": "Predict this batch", "observations": []}
    payload, _ = record(pending, args, output, tmp_path, harness)
    assert "0.9942129558371191" in payload["result_excerpt"]
    assert "ratio_to_reference" in payload["result_excerpt"] and "65.5" in payload["result_excerpt"]
    assert "results/predictions/fixed/result.json" in payload["result_excerpt"]
    assert payload["status"] == "completed" and payload["excerpted"] is True
    relative = payload["full_observation"].removeprefix("/workspace/")
    original = json.loads((tmp_path / relative).read_text())
    assert original == {
        "tool": "predict_with_evolved_model",
        "arguments": args,
        "status": "completed",
        "result": output,
    }
    assert (args, output) == before
    assert (tmp_path / relative).stat().st_mode & 0o077 == 0
    state = {"agent_pending_turn": pending, "agent_messages": []}
    restored = json.loads(json.dumps(persistable_state(state)))
    assert payload["full_observation"] in json.dumps(continuation_messages(restored, harness=harness))
    saved = completed_messages([], pending["text"], pending["observations"], "Done.", harness)
    assert continuation_messages({"agent_messages": saved}, harness=harness) == saved
    assert payload["full_observation"] in json.dumps(saved)
    other = tmp_path / "another-session"
    other.mkdir()
    with pytest.raises(FileNotFoundError):
        read_bytes(other, relative)
    assert json.loads(read_bytes(tmp_path, relative)) == original


@pytest.mark.parametrize(
    "value", [("雪" + chr(92) + chr(10) + chr(34)) * 12000, "plain" * 20000], ids=["unicode", "plain"]
)
def test_multibyte_and_escaped_failure_records_keep_valid_bounded_json(tmp_path, value):
    payload, _ = record({}, {"input": value}, value, tmp_path, failed=True)
    assert payload["status"] == "failed"
    saved = json.loads((tmp_path / payload["full_observation"].removeprefix("/workspace/")).read_text())
    assert saved["result"] == value and saved["arguments"] == {"input": value}


def test_archive_failure_keeps_completed_observation_without_false_file_pointer(tmp_path):
    workspace = tmp_path / "session"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace / "context").symlink_to(outside, target_is_directory=True)
    payload, text = record({}, {"input": "a" * 20000}, {"prediction": 0.123}, workspace)
    assert payload["status"] == "completed" and "0.123" in text
    assert payload["archive_status"] == "unavailable" and "full_observation" not in payload
    assert not list(outside.iterdir())
    assert "see conversation archive" not in text


def test_repeated_snapshot_uses_one_owned_archive(tmp_path):
    first, _ = record({}, {"input": "x" * 20000}, {"value": 7}, tmp_path)
    second, _ = record({}, {"input": "x" * 20000}, {"value": 7}, tmp_path)
    assert first == second
    assert len(list((tmp_path / "context/tool-observations").glob("*.json"))) == 1


def test_short_observation_preserves_its_existing_structure(tmp_path):
    payload, _ = record({}, {"sequence": "ACD"}, {"value": 42}, tmp_path)
    assert payload == {
        "tool": "predict_with_evolved_model",
        "arguments": {"sequence": "ACD"},
        "status": "completed",
        "result": {"value": 42},
    }
    assert not (tmp_path / "context").exists()
