import json

import pytest
from test_training_attempt_memory import episode

from agent.harness.contracts import digest
from agent.harness.host import write_json
from agent.rsi import training_attempt_memory as memory


def empty_episode(root, index):
    protocol, cycle, folder = episode(root, index)
    (folder / "execution/events.jsonl").write_text("")
    receipt = memory.capture(folder, protocol)
    assert not receipt["attempts"]
    write_json(folder / "attempt_evidence.json", receipt)
    cycle["attempt_evidence_sha256"] = digest(receipt)
    return protocol, cycle, folder


def cumulative(protocol):
    return {**protocol, memory.FIELD: memory.CUMULATIVE_POLICY}


def test_empty_recent_episodes_do_not_erase_older_tested_program(tmp_path):
    protocol, first, folder = episode(tmp_path, sources=["def fit(): return 1"], feedback={"rows": 20, "rmse": 1.1})
    _, second, _ = empty_episode(tmp_path, 1)
    _, third, _ = empty_episode(tmp_path, 2)
    state = {"cycles": [first, second, third]}
    assert memory.project(tmp_path, state, protocol, "develop")["entries"] == []
    packet = memory.project(tmp_path, state, cumulative(protocol), "develop")
    assert len(packet["entries"]) == 1 and packet["entries"][0]["cycle"] == 0
    assert packet["entries"][0]["source_code"] == "def fit(): return 1"
    assert packet["summary"]["development_episodes"] == 3
    assert packet["summary"]["empty_episodes"] == 2
    assert "CUMULATIVE TRAINING ATTEMPT ARCHIVE" in memory.context(packet)
    assert not memory.context(memory.project(tmp_path, state, cumulative(protocol), "reuse"))
    # Old evidence is still checked even after two empty episodes.
    data = json.loads((folder / "workspace/code_attempt_1.json").read_text())
    data["source_code"] += "\nx=2"
    write_json(folder / "workspace/code_attempt_1.json", data)
    with pytest.raises(ValueError, match="differs"):
        memory.project(tmp_path, state, cumulative(protocol), "develop")


def test_all_empty_episodes_still_produce_a_bound_context(tmp_path):
    protocol, cycle, _ = empty_episode(tmp_path, 0)
    packet = memory.project(tmp_path, {"cycles": [cycle]}, cumulative(protocol), "develop")
    assert packet["entries"] == []
    assert packet["summary"]["empty_episodes"] == 1
    assert memory.context(packet)


def test_comments_do_not_inflate_program_inventory(tmp_path):
    protocol, first, _ = episode(tmp_path, sources=["def fit():\n    return 1\n"])
    _, second, _ = episode(tmp_path, 1, sources=["# renamed explanation\ndef fit():\n    return 1\n"])
    packet = memory.project(tmp_path, {"cycles": [first, second]}, cumulative(protocol), "develop")
    assert packet["summary"]["verified_attempts"] == 2 and packet["summary"]["distinct_programs"] == 1
    item = packet["entries"][0]
    assert item["first_cycle"] == 0 and item["cycle"] == 1 and item["occurrences"] == 2
    assert "_observations" not in item


def test_source_budget_keeps_recent_and_best_training(tmp_path):
    cycles = []
    for i, score in enumerate([0.9, 0.1, 0.2, 0.3]):
        protocol, cycle, _ = episode(
            tmp_path, i, sources=[f"def fit(): return {i}"], feedback={"mse_skill_vs_training_mean": score}
        )
        cycles.append(cycle)
    packet = memory.project(tmp_path, {"cycles": cycles}, cumulative(protocol), "develop")
    assert {e["cycle"] for e in packet["entries"] if "source_code" in e} == {0, 3}
    assert len(packet["entries"]) == 4


def test_archive_cap_keeps_early_history_and_protected_mid_history(tmp_path):
    cycles = []
    for i in range(70):
        protocol, cycle, _ = episode(
            tmp_path,
            i,
            sources=[f"def fit(): return {i}"],
            feedback={"mse_skill_vs_training_mean": 1.0 if i == 34 else 0.1},
        )
        cycles.append(cycle)
    _, recent, _ = episode(tmp_path, 70, sources=["def fit(): return 35"], feedback={"mse_skill_vs_training_mean": 0.2})
    cycles.append(recent)
    packet = memory.project(tmp_path, {"cycles": cycles}, cumulative(protocol), "develop")
    assert len(packet["entries"]) == 64 and packet["summary"]["omitted_programs"] == 6
    assert {e["cycle"] for e in packet["entries"] if "source_code" in e} == {34, 70}
    assert packet["summary"]["verified_attempts"] == 71
    assert any(e["first_cycle"] == 0 for e in packet["entries"])


def test_invalid_and_large_source_retains_diagnostics_without_partial_code(tmp_path):
    protocol, first, _ = episode(tmp_path, sources=["def invalid("])
    _, second, _ = episode(tmp_path, 1, sources=["#" + "x" * 12001])
    packet = memory.project(tmp_path, {"cycles": [first, second]}, cumulative(protocol), "develop")
    assert packet["entries"][0]["identity"].startswith("source:")
    assert all("training_only_feedback" in e for e in packet["entries"])
    assert all("source_code" not in e for e in packet["entries"])


def test_projection_never_opens_assessment_or_future_files(tmp_path, monkeypatch):
    protocol, cycle, folder = episode(
        tmp_path,
        feedback={"rows": 20, "rmse": 1.0, "assessment_metric": "SECRET", "future_labels": ["SECRET"], "weights": [99]},
    )
    write_json(folder / "acceptance.json", {"secret": "ASSESSMENT"})
    write_json(tmp_path / "future_metrics.json", {"secret": "FUTURE"})
    original = memory.read

    def safe_read(root, name):
        assert name not in {"acceptance.json", "future_metrics.json"} and "assessment" not in name
        return original(root, name)

    monkeypatch.setattr(memory, "read", safe_read)
    packet = memory.project(tmp_path, {"cycles": [cycle]}, cumulative(protocol), "develop")
    assert all(e["training_only_feedback"] == {"rows": 20, "rmse": 1.0} for e in packet["entries"])
    assert "SECRET" not in json.dumps(packet) and "weights" not in json.dumps(packet)


@pytest.mark.parametrize("indices", [[1, 0], [0, 0], [-1], [True]])
def test_nonchronological_history_rejected(tmp_path, indices):
    with pytest.raises(ValueError, match="chronological"):
        memory.project(
            tmp_path,
            {"cycles": [{"phase": "develop", "index": i} for i in indices]},
            {memory.FIELD: memory.CUMULATIVE_POLICY},
            "develop",
        )


@pytest.mark.parametrize("policy", memory.POLICIES)
def test_policy_binding_cannot_be_changed(policy):
    state = {"definition": {memory.FIELD: memory.CUMULATIVE_POLICY}}
    protocol = {memory.FIELD: policy}
    if policy == memory.CUMULATIVE_POLICY:
        assert memory.verify_goal(state, protocol) == policy
    else:
        with pytest.raises(ValueError, match="policy changed"):
            memory.verify_goal(state, protocol)


def test_new_policy_requires_binding_and_old_missing_binding_remains_valid():
    with pytest.raises(ValueError, match="policy changed"):
        memory.verify_goal({"definition": {}}, {memory.FIELD: memory.CUMULATIVE_POLICY})
    assert memory.verify_goal({"definition": {}}, {memory.FIELD: memory.POLICY}) == memory.POLICY
    assert memory.verify_goal({"definition": {}}, {}) == "none"


@pytest.mark.parametrize("history", ["metrics_only", "independent_restart"])
def test_source_withheld_studies_cannot_enable_archive(history):
    with pytest.raises(ValueError, match="Source-withheld"):
        memory.policy_for({memory.FIELD: memory.CUMULATIVE_POLICY, "history_policy": history})
