"""Synthetic screening fixtures do not establish autonomous model acquisition."""

import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_context_transfer import fixture as context_fixture  # noqa: F401
from test_context_transfer import task

from agent.harness import Budget, ModelBinding, Profile
from agent.harness.contracts import file_hash
from agent.rsi import context_interfaces, model_tasks
from agent.rsi import library_program_screen as screening
from agent.rsi.code_sandbox import CandidateCodeError
from agent.rsi.storage import atomic_json

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
from run_science_goal import GeneratedModelGoal, prepare


@pytest.fixture
def fixture(tmp_path, request):
    selection, target, donor, donor_rows, rows = request.getfixturevalue("context_fixture")
    p = tmp_path / "task.json"
    atomic_json(p, target)
    request = {
        "schema": screening.SCHEMA,
        "selection": selection,
        "tasks": [{"path": str(p), "sha256": file_hash(p)}],
        "keep": 1,
    }
    return request, target, rows, donor, donor_rows


def test_real_group_refit_preserves_sources_and_emits_no_weights_or_labels(fixture):
    request, target, rows, donor, donor_rows = fixture
    report = screening.screen(request)
    result = report["outcomes"][0]
    api = context_interfaces.module_for(target["plugin_id"])
    fit, feedback = api.development_split(rows["training"])
    expected = screening._evaluate(target, donor["packet"]["model"]["recipe"]["source_code"], fit, feedback)
    assert result["candidates"][0]["training_feedback"] == expected
    assert result["selection"] == request["selection"]
    assert not {r["group"] for r in fit} & {r["group"] for r in feedback}
    assert report["candidate_source_tests"] == report["maximum_candidate_source_tests"] == 1
    assert report["external_model_calls"] == report["new_acquired_models"] == report["new_published_models"] == 0
    assert not report["validation_scored"] and not report["future_scored"]
    text = json.dumps(report)
    assert "private_fitted_weight" not in text and "917.123" not in text
    assert '"model_state"' not in text and '"source_code"' not in text and '"targets"' not in text
    assert donor_rows["training"] != rows["training"]


def test_hidden_labels_do_not_change_screening_metrics(fixture):
    request, target, rows, _, _ = fixture
    before = screening.screen(request)["outcomes"][0]
    for phase in ("validation", "future"):
        changed = deepcopy(rows[phase])
        for row in changed:
            row["target"] += 1000
        p = Path(target["data"][phase]["path"])
        atomic_json(p, changed)
        target["data"][phase]["sha256"] = file_hash(p)
    p = Path(request["tasks"][0]["path"])
    atomic_json(p, target)
    request["tasks"][0]["sha256"] = file_hash(p)
    after = screening.screen(request)["outcomes"][0]
    assert before["candidates"][0]["training_feedback"] == after["candidates"][0]["training_feedback"]
    assert before["candidates"][0]["source_sha256"] == after["candidates"][0]["source_sha256"]
    assert before["selection"] == after["selection"]
    assert before["training_sha256"] == after["training_sha256"]
    assert before["admission"] != after["admission"]


def test_corrupt_target_or_donor_evidence_aborts_instead_of_low_ranking(fixture):
    request, target, _, donor, _ = fixture
    p = Path(target["data"]["training"]["path"])
    original = p.read_bytes()
    p.write_text("[]")
    with pytest.raises(ValueError, match="changed"):
        screening.screen(request)
    p.write_bytes(original)
    (Path(donor["origin_goal"]) / "groups.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        screening.screen(request)


def test_same_study_is_retained_as_ineligible_without_any_fit(tmp_path, fixture, monkeypatch):
    request, _, _, _, _ = fixture
    target, _ = task(tmp_path / "same", "10.1234/donor", "Other")
    p = tmp_path / "same.json"
    atomic_json(p, target)
    request["tasks"] = [{"path": str(p), "sha256": file_hash(p)}]
    monkeypatch.setattr(screening, "_evaluate", lambda *args: pytest.fail("Ineligible donor executed"))
    report = screening.screen(request)
    result = report["outcomes"][0]
    assert result["candidates"][0]["status"] == "ineligible"
    assert result["selection"]["entry_ids"] == [] and report["candidate_source_tests"] == 0


def test_candidate_failure_kept_without_error_payload_or_selection(fixture, monkeypatch):
    request, *_ = fixture

    def broken(*args):
        raise CandidateCodeError("arbitrary code output and possible input echo")

    monkeypatch.setattr(screening, "_evaluate", broken)
    report = screening.screen(request)
    result = report["outcomes"][0]
    assert result["candidates"][0]["status"] == "execution_failed"
    assert result["selection"]["entry_ids"] == [] and report["candidate_source_tests"] == 1
    assert "possible input echo" not in json.dumps(report)


def test_rank_ties_are_deterministic_and_negative_scores_do_not_claim_acceptance(fixture, monkeypatch):
    request, *_ = fixture
    request["selection"]["entry_ids"] = ["b" * 64, "a" * 64]
    monkeypatch.setattr(screening, "_evaluate", lambda *args: {"mse_skill_vs_training_mean": -0.5})
    report = screening.screen(request)
    assert report["outcomes"][0]["selection"]["entry_ids"] == ["a" * 64]
    assert report["candidate_source_tests"] == 2 and report["new_acquired_models"] == 0


def test_handoff_round_trips_through_real_scientist_prepare(tmp_path, fixture):
    request, target, *_ = fixture
    out = tmp_path / "screen"
    screening.export(request, out)
    handoff = json.loads((out / "handoff.json").read_text())["tasks"][0]
    adopted, _ = model_tasks.read_task(model_tasks.bound_file(handoff["task"]))
    assert adopted["scope"] == target["scope"] and adopted["minimum_skill"] == target["minimum_skill"]
    assert adopted["references"][:-1] == target["references"]
    assert adopted["references"][-1]["sha256"] == file_hash(out / "screen.json")
    profile = Profile(
        "science_expert",
        "expert-kimi-code",
        ModelBinding("test_fixture", "model", "https://example.org/v1"),
        Budget(wall_seconds=10, request_seconds=5, model_calls=3, tool_calls=2),
    )
    atomic_json(tmp_path / "profile.json", profile.public())
    goal = prepare(
        SimpleNamespace(
            output=tmp_path / "goal",
            profile=tmp_path / "profile.json",
            task_definition=Path(handoff["task"]["path"]),
            experience_selection=Path(handoff["selection"]["path"]),
            objective="Synthetic program-screen handoff",
            max_cycles=2,
            model_calls=6,
            tool_calls=4,
            wall_seconds=60,
        )
    )
    driver = GeneratedModelGoal(goal, None)
    assert len(driver.library_experience["entries"]) == 1
    assert driver.scope == target["scope"] and goal.read()["usage"]["model_calls"] == 0
    with pytest.raises(ValueError, match="already exists"):
        screening.export(request, out)
    (out / "screen.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        model_tasks.read_task(Path(handoff["task"]["path"]))


@pytest.mark.parametrize(
    "change", ["schema", "extra", "none", "empty", "too_many", "bool_keep", "zero_keep", "too_many_keep", "duplicate"]
)
def test_request_bounds_fail_closed(fixture, change):
    request, *_ = fixture
    if change == "schema":
        request["schema"] = "unknown"
    elif change == "extra":
        request["future_labels"] = []
    elif change == "none":
        request["selection"]["condition"] = "none"
    elif change == "empty":
        request["tasks"] = []
    elif change == "too_many":
        request["tasks"] *= 33
    elif change == "bool_keep":
        request["keep"] = True
    elif change == "zero_keep":
        request["keep"] = 0
    elif change == "too_many_keep":
        request["keep"] = 6
    else:
        request["tasks"] *= 2
    with pytest.raises(ValueError):
        screening.screen(request)


@pytest.mark.parametrize("case_alias", [False, True])
def test_distinct_task_id_cannot_duplicate_same_scope(tmp_path, fixture, case_alias):
    request, target, *_ = fixture
    target["id"] = "renamed"
    if case_alias:
        target["scope"]["context"] = {k: v.upper() for k, v in target["scope"]["context"].items()}
    p = tmp_path / "renamed.json"
    atomic_json(p, target)
    request["tasks"].append({"path": str(p), "sha256": file_hash(p)})
    with pytest.raises(ValueError, match="scientific scope"):
        screening.screen(request)


def test_no_output_if_target_changes_during_screen(tmp_path, fixture, monkeypatch):
    request, target, *_ = fixture
    real = screening._evaluate

    def changed(*args):
        result = real(*args)
        Path(target["data"]["training"]["path"]).write_text("[]")
        return result

    monkeypatch.setattr(screening, "_evaluate", changed)
    with pytest.raises(ValueError, match="changed"):
        screening.export(request, tmp_path / "screen")
    assert not (tmp_path / "screen").exists()


def test_v2_numeric_adaptation_and_multiple_independent_recipient_fits(tmp_path, fixture):
    from test_context_source_adaptation import adapted, receiver

    request, first, _, _, _ = fixture
    second, parts = task(tmp_path / "second", protein="Second")
    second, _ = receiver(second, parts)
    p = tmp_path / "second.json"
    atomic_json(p, second)
    request["tasks"].append({"path": str(p), "sha256": file_hash(p)})
    request["selection"] = adapted(request["selection"])
    report = screening.screen(request)
    left, right = report["outcomes"]
    assert left["plugin_id"] == "generated_context.v1" and right["plugin_id"] == "generated_context.v2"
    assert left["training_sha256"] != right["training_sha256"]
    assert all(v["candidates"][0]["status"] == "screened" for v in report["outcomes"])
    assert report["candidate_source_tests"] == 2 and report["new_published_models"] == 0


def test_nonfinite_score_cannot_rank(fixture, monkeypatch):
    request, target, *_ = fixture
    api = context_interfaces.module_for(target["plugin_id"])
    monkeypatch.setattr(api, "evaluate", lambda *args: {api.ACCEPTANCE_METRIC: float("nan")})
    with pytest.raises(ValueError, match="Nonfinite"):
        screening.screen(request)
