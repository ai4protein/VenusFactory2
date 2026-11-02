"""Training-only selection and real numerical/MCP integration without an LLM."""

import copy
import hashlib
import json

import pytest

from agent.rsi import shared_goal_history as history
from agent.rsi import shared_history_selection as selection
from agent.rsi.protein_model import digest


def attempt(index, skills, *, code=None, status="tested_not_published"):
    code = code if code is not None else f"def fit(rows):\n    return {{'index': {index}}}\n"
    return {
        "episode": index,
        "source_code": code,
        "source_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "rationale": "fixture",
        "results": [
            {
                "task_id": task,
                "feedback": {"status": status, "training_only_feedback": {"mse_skill_vs_training_mean": skill}},
            }
            for task, skill in skills.items()
        ],
    }


def hashes(packet):
    return [row["source_sha256"] for row in packet["attempts"]]


def test_selects_complementary_training_programs_not_recent_duplicates():
    old = attempt(0, {"a": 0.9, "b": -0.4})
    complement = attempt(1, {"a": -0.5, "b": 0.8})
    recent = attempt(2, {"a": 0.6, "b": -0.3})
    duplicate = {**copy.deepcopy(recent), "episode": 3}
    observations = [old, complement, recent, duplicate]
    before = copy.deepcopy(observations)
    packet = selection.select(observations, ["a", "b"])
    assert hashes(packet) == [old["source_sha256"], complement["source_sha256"]]
    assert packet["available_attempts"] == 4 and packet["distinct_sources"] == 3
    assert packet["verified_attempts_sha256"] == digest(observations)
    assert observations == before
    assert selection.select(observations, ["a", "b"]) == packet


def test_empty_option_for_nonpositive_failed_and_missing_training_feedback():
    observations = [attempt(0, {"a": -1}), attempt(1, {"a": 0}), attempt(2, {"a": 0.9}, status="development_failed")]
    unknown = attempt(3, {"a": None})
    unknown["results"][0]["feedback"]["validation_score"] = 999
    unknown["results"][0]["feedback"]["final_score"] = 999
    observations.append(unknown)
    packet = selection.select(observations, ["a"])
    assert packet["no_source_selected"] and not packet["attempts"]
    assert {d["reason"] for d in packet["selection_decisions"]} == {"no_positive_training_evidence"}
    assert "999" not in json.dumps(packet)


def test_repeated_source_uses_conservative_scores_and_preserves_feedback_binding():
    first = attempt(0, {"a": 0.95})
    repeated = attempt(1, {"a": 0.1}, code=first["source_code"])
    better = attempt(2, {"a": 0.7})
    packet = selection.select([first, repeated, better], ["a"])
    assert hashes(packet) == [better["source_sha256"]]
    record = next(d for d in packet["selection_decisions"] if d["source_sha256"] == first["source_sha256"])
    assert record["occurrences"] == 2
    assert packet["attempts"][0]["results"][0]["feedback_sha256"] == digest(better["results"][0]["feedback"])


def test_failed_repeat_cannot_inherit_an_earlier_success_score():
    first = attempt(0, {"a": 0.9})
    failed = attempt(1, {"a": 0.9}, code=first["source_code"], status="development_failed")
    assert selection.select([first, failed], ["a"])["no_source_selected"]


def test_target_filter_does_not_reward_unfinished_task_without_evidence():
    old = attempt(0, {"a": 0.8})
    packet = selection.select([old], ["a", "b"])
    assert packet["marginal_training_gains"][old["source_sha256"]] == 0.4
    assert set(packet["attempts"][0]["task_training_skills"]) == {"a"}
    with pytest.raises(ValueError, match="target identity"):
        selection.select([old], ["b"])


@pytest.mark.parametrize("value", [True, "0.9", None, 1.001])
def test_invalid_skill_types_are_not_ranked(value):
    assert selection.select([attempt(0, {"a": value})], ["a"])["no_source_selected"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_evidence_is_rejected(value):
    with pytest.raises(ValueError):
        selection.select([attempt(0, {"a": value})], ["a"])


def test_classification_skill_supported_without_mixing_metric_types():
    one = attempt(0, {"a": 0.7})
    metrics = one["results"][0]["feedback"]["training_only_feedback"]
    metrics["brier_skill_vs_training_prior"] = metrics.pop("mse_skill_vs_training_mean")
    assert hashes(selection.select([one], ["a"])) == [one["source_sha256"]]
    metrics["mse_skill_vs_training_mean"] = 0.9
    assert selection.select([one], ["a"])["no_source_selected"]


def test_budget_omits_whole_program_then_tries_a_smaller_candidate():
    huge = attempt(0, {"a": 0.9}, code="#" + "蛋白" * 10000)
    small = attempt(1, {"a": 0.6})
    packet = selection.select([huge, small], ["a"])
    assert hashes(packet) == [small["source_sha256"]]
    assert packet["attempts"][0]["source_code"] == small["source_code"]
    assert any(d["reason"] == "context_budget" for d in packet["selection_decisions"])
    assert selection.history_bytes(packet) <= selection.MAX_HISTORY_BYTES


def test_long_archive_keeps_old_useful_source_and_binds_all_omissions():
    old = attempt(0, {"a": 0.9})
    observations = [old, *[attempt(i, {"a": -0.1}) for i in range(1, 100)]]
    packet = selection.select(observations, ["a"])
    assert hashes(packet) == [old["source_sha256"]]
    assert len(packet["selection_decisions"]) == selection.MAX_DECISIONS
    assert packet["omitted_decision_records"] == 36
    assert packet["verified_attempts_sha256"] == digest(observations)
    assert selection.history_bytes(packet) <= selection.MAX_HISTORY_BYTES


@pytest.mark.parametrize("mutation", ["source", "chronology", "duplicate_target"])
def test_corrupt_training_evidence_fails_closed(mutation):
    observations = [attempt(0, {"a": 0.8}), attempt(1, {"a": 0.5})]
    if mutation == "source":
        observations[0]["source_code"] += "# changed"
    elif mutation == "chronology":
        observations.reverse()
    else:
        observations[0]["results"] *= 2
    with pytest.raises(ValueError):
        selection.select(observations, ["a"])


def test_policy_collects_cumulative_training_records_only(monkeypatch, tmp_path):
    calls = []

    def observe(root, state, selected, *, all_completed=False):
        calls.append(all_completed)
        return [attempt(0, {"a": 0.8})]

    monkeypatch.setattr(history, "_observations", observe)
    packet = history.project(tmp_path, {"cycles": []}, [{"id": "a"}], selection.POLICY)
    assert calls == [True]
    prompt = history.make_prompt("goal", "develop", packet)
    assert prompt.endswith(json.dumps(packet, ensure_ascii=False))
    assert "new program" in prompt
    with pytest.raises(ValueError, match="cannot enter reuse"):
        history.make_prompt("goal", "reuse", packet)
    assert history.POLICY == "bounded_training_program_history.v2"


@pytest.mark.asyncio
async def test_selective_goal_uses_real_training_feedback_across_episodes(tmp_path, monkeypatch):
    from test_shared_science_goal import prepared, read, shared, synthetic_executor

    from agent.science_goal import GoalStore, run_goal

    store, driver = prepared(tmp_path, reject_second=True, max_cycles=5, history_policy=selection.POLICY)
    calls = synthetic_executor(monkeypatch)
    state = await run_goal(store, driver.cycle, driver.evaluate)
    assert state["status"] == "budget_exhausted"
    assert [c["phase"] for c in calls] == ["develop", "reuse", "develop", "develop", "develop"]
    packet = read(store.root / "episodes/cycle_004/shared_training_history.json")
    assert packet["available_attempts"] == 3 and packet["distinct_sources"] == 1
    assert len(packet["attempts"]) == 1 and packet["attempts"][0]["occurrences"] == 3
    assert packet["target_ids"] == ["synthetic_1"]
    assert all("1234567" not in c["prompt"] for c in calls)
    assert "Prior paired" not in calls[1]["prompt"]
    protocol = read(store.root / "protocol.json")
    assert "src/agent/rsi/shared_history_selection.py" in protocol["implementation"]
    resumed = shared.SharedScientistGoal(GoalStore(store.root), None)
    assert resumed.training_history(state, [driver.by_id["synthetic_1"]]) == driver.training_history(
        state, [driver.by_id["synthetic_1"]]
    )
    # All completed episodes remain bound, including ones beyond the old two-episode window.
    path = store.root / "episodes/cycle_000/execution/events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    for event in events:
        if event.get("tool_name", "").endswith("evaluate_model_code"):
            event["args"]["source_code"] += "# changed old source"
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    with pytest.raises(ValueError, match="source differs"):
        driver.training_history(state, [driver.by_id["synthetic_1"]])
