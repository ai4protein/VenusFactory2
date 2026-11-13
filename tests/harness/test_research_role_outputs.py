"""Complete private role text and explicit excerpts preserve bounded SC repair."""
import copy
import hashlib
import json
import stat
from types import SimpleNamespace

import pytest
import test_analysis_report_review as cases

from agent.harness import analysis_report_review as analysis
from agent.harness import research_role_outputs as outputs
from agent.harness.autonomous_research import AutonomousResearch

make_case = cases.make_case


def test_private_response_is_complete_hash_bound_and_not_in_workspace(tmp_path):
    control, workspace = tmp_path / "control", tmp_path / "workspace"
    control.mkdir()
    workspace.mkdir()
    text = "scientific evidence\n" * 1000
    record = outputs.record_response(control, "SC", 0, text)
    path = control / record["path"]
    assert path.read_bytes() == text.encode()
    assert record["truncated"] is False
    assert record["response_bytes"] == record["saved_bytes"] == len(text.encode())
    assert record["saved_sha256"] == record["response_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.with_suffix(".json").stat().st_mode) == 0o600
    assert not list(workspace.iterdir())
    second = outputs.record_response(control, "SC", 0, "a later separate invocation")
    assert second["path"] != record["path"]
    assert path.read_bytes() == text.encode()


def test_saved_limit_never_claims_an_incomplete_utf8_prefix_is_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(outputs, "MAX_SAVED_CHARACTERS", 10)
    monkeypatch.setattr(outputs, "MAX_SAVED_BYTES", 11)
    raw = "研究" * 20
    record = outputs.record_response(tmp_path, "PI", 0, raw)
    saved = (tmp_path / record["path"]).read_bytes()
    assert len(saved) <= 11
    assert saved.decode().encode() == saved
    assert record["truncated"] is True
    assert record["saved_bytes"] == len(saved)
    assert record["response_bytes"] == len(raw.encode())
    assert record["saved_sha256"] == hashlib.sha256(saved).hexdigest()
    assert record["response_sha256"] == hashlib.sha256(raw.encode()).hexdigest()


@pytest.mark.parametrize("fault", ["directory_symlink", "root_symlink", "role_path", "provider_envelope"])
def test_private_recording_refuses_path_escape_and_nontext_provider_data(tmp_path, fault):
    workspace = tmp_path / "workspace"
    control = tmp_path / "control"
    workspace.mkdir()
    control.mkdir()
    role, text = "SC", "role text"
    if fault == "directory_symlink":
        (control / "role_outputs").symlink_to(workspace, target_is_directory=True)
    elif fault == "root_symlink":
        link = tmp_path / "linked-control"
        link.symlink_to(workspace, target_is_directory=True)
        control = link
    elif fault == "role_path":
        role = "../workspace"
    else:
        text = {"headers": {"Authorization": "synthetic-provider-credential"}}
    with pytest.raises((ValueError, OSError)):
        outputs.record_response(control, role, 0, text)
    assert not list(workspace.iterdir())
    if fault in {"role_path", "provider_envelope"}:
        assert not list(control.iterdir())


@pytest.mark.asyncio
@pytest.mark.parametrize("oversized_repair", [False, True])
async def test_long_invalid_sc_keeps_original_text_and_requires_real_validation(make_case, tmp_path, oversized_repair):
    _, _, context, good = make_case()
    invalid = copy.deepcopy(good)
    invalid["input_assessments"] = []
    invalid["claims"] = [{**good["claims"][0], "id": "claim-" + str(i)} for i in range(64)]
    invalid["step_assessments"][0]["claim_ids"] = [row["id"] for row in invalid["claims"]]
    if oversized_repair:
        for row in invalid["claims"]:
            row["report_quote"] = "unsupported report text " * 70
    raw = json.dumps(invalid)
    assert len(raw) > 12000
    assert (len(raw) > outputs.MAX_REPAIR_CHARACTERS) is oversized_repair
    seen = []

    async def consult(role, packet):
        assert role == "SC"
        seen.append(packet)
        return raw if len(seen) == 1 else json.dumps(good)

    folder = tmp_path / "private-episode"
    folder.mkdir()
    profile = SimpleNamespace(budget=SimpleNamespace(model_calls=20))
    runner = AutonomousResearch(None, "fixture", {}, consult=consult, tools=[])
    result = await runner.structured("SC", context, profile, SimpleNamespace(calls=0), folder,
                                     lambda value: analysis.parse_review(value, context))
    assert result == good
    assert len(seen) == 2
    repair = seen[1]["format_repair"]
    expected_task = good["claims"][0]["task_id"]
    assert expected_task in repair["validation_error"]
    assert "Missing task_ids" in repair["validation_error"]
    assert len(repair["validation_error"]) <= 400
    assert repair["previous_response_complete"] is (not oversized_repair)
    if oversized_repair:
        assert "previous_response" not in repair
        assert len(repair["previous_response_excerpt"]) == outputs.MAX_EXCERPT_CHARACTERS
        assert "not a complete JSON document" in repair["instruction"]
    else:
        assert repair["previous_response"] == raw
        assert "previous_response_excerpt" not in repair
    records = [json.loads(path.read_text()) for path in (folder / "role_outputs").glob("*.json")]
    assert len(records) == 2
    first = next(row for row in records if row["attempt"] == 0)
    assert first["truncated"] is False
    assert (folder / first["path"]).read_text() == raw
    assert first["response_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    assert len(json.loads((folder / "activity.json").read_text())[0]["content"]) == 12000


@pytest.mark.asyncio
async def test_full_recording_does_not_add_requests_or_repair_the_verdict(make_case, tmp_path):
    _, _, context, review = make_case()
    review["input_assessments"] = []
    raw = json.dumps(review)
    seen = []

    async def consult(_role, packet):
        seen.append(packet)
        return raw

    runner = AutonomousResearch(None, "fixture", {}, consult=consult, tools=[])
    with pytest.raises(ValueError, match="Missing task_ids"):
        await runner.structured("SC", context, SimpleNamespace(budget=SimpleNamespace(model_calls=20)),
                                SimpleNamespace(calls=0), tmp_path,
                                lambda value: analysis.parse_review(value, context))
    assert len(seen) == 2
    assert len(list((tmp_path / "role_outputs").glob("*.txt"))) == 2


def test_private_response_creates_missing_control_tree_without_following_ancestors(tmp_path):
    control = tmp_path / "missing" / "episode"
    record = outputs.record_response(control, "SC", 0, "Complete review")
    assert (control / record["path"]).read_text() == "Complete review"
    assert stat.S_IMODE(control.stat().st_mode) == 0o700
    assert stat.S_IMODE(control.parent.stat().st_mode) == 0o700
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "linked-parent"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises((ValueError, OSError)):
        outputs.record_response(link / "missing-episode", "SC", 0, "Must not escape")
    assert not list(outside.iterdir())
    invalid = tmp_path / "invalid-missing"
    with pytest.raises(ValueError):
        outputs.record_response(invalid, "../SC", 0, "Invalid identity")
    assert not invalid.exists()
