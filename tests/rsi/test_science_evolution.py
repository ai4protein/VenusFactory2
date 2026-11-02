"""Regression coverage for genuine promotion, ancestry and continuation gates."""

import asyncio
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_generated_models import CODE, SCOPE, TRAIN, VALID

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
from run_science_evolution import EvolvingModelGoal
from run_science_goal import GeneratedModelGoal

from agent.rsi.generated_model import fit_candidate, predict
from agent.rsi.model_lifecycle import ModelLifecycle
from agent.rsi.storage import atomic_json
from backend.web_v2.science_goals import EvolvingModelDriver, ScienceGoalService, goal_activity


@pytest.fixture
def lineage(tmp_path):
    plugin = object.__new__(EvolvingModelGoal)
    plugin.store = SimpleNamespace(root=tmp_path)
    plugin.protocol = {"evolution": {"generations": 2}}
    plugin.scope = SCOPE
    plugin.data = {"training": TRAIN, "validation": VALID, "future": VALID}
    state = {
        "definition": {"goal": {"criteria": ["accepted_model", "fresh_session_reuse", "verified_lineage"]}},
        "progress": {"generations": []},
    }
    for index, shift in enumerate([1.0, 0.0]):
        source = CODE.replace(
            "return (features(sequences) @ np.array(model['weights'])).tolist()",
            f"return (features(sequences) @ np.array(model['weights']) + {shift}).tolist()",
        )
        folder = tmp_path / f"g{index}"
        policy = plugin.acceptance_policy(state)
        lc = ModelLifecycle(folder, policy)
        model = fit_candidate(TRAIN, SCOPE, {"source_code": source, "seed": 1})
        atomic_json(folder / "candidate.json", model)
        mid = lc.submit(folder / "candidate.json", session_id=f"develop-{index}", mode="science_expert")
        assert lc.assess(mid, training=TRAIN, validation=VALID)["passed"]
        parents = [state["progress"]["generations"][-1]["experience_id"]] if index else []
        eid = plugin.experience().publish(lc, mid, summary="training-only source improvement", parents=parents)
        receipt = lc.record_reuse(
            mid,
            session_id=f"reuse-{index}",
            mode="science_expert",
            inputs=VALID,
            predictions=predict(model, VALID, SCOPE),
            scope=SCOPE,
        )
        atomic_json(folder / "receipt.json", receipt)
        state["progress"]["generations"].append(
            {
                "controller": folder.name,
                "model_id": mid,
                "experience_id": eid,
                "generation": index + 1,
                "reuse_receipt": str((folder / "receipt.json").relative_to(tmp_path)),
            }
        )
    return plugin, state


def test_every_generation_requires_acceptance_and_fresh_reuse(lineage):
    plugin, state = lineage
    assert all(v["passed"] for v in plugin.evaluate(state)["criteria"].values())
    state["progress"]["generations"][0].pop("reuse_receipt")
    checks = plugin.evaluate(state)["criteria"]
    assert checks["accepted_model"]["passed"] and not checks["fresh_session_reuse"]["passed"]


def test_single_complete_generation_does_not_finish_goal(lineage):
    plugin, state = lineage
    state["progress"]["generations"].pop()
    assert not any(v["passed"] for v in plugin.evaluate(state)["criteria"].values())
    packet = plugin.development_experience(state)
    assert packet["entries"][0]["source_code"]
    assert not {"metrics", "model_state", "assessment", "training_labels"} & packet["entries"][0].keys()
    policy = plugin.acceptance_policy(state)
    assert 0 < policy["minimum_skill"] < 1


def test_duplicate_generation_cannot_fake_evolution(lineage):
    plugin, state = lineage
    state["progress"]["generations"][1] = copy.deepcopy(state["progress"]["generations"][0])
    with pytest.raises(ValueError, match="promotion"):
        plugin.evaluate(state)


def test_changed_ancestry_rejected(lineage):
    plugin, state = lineage
    state["progress"]["generations"][1]["experience_id"] = state["progress"]["generations"][0]["experience_id"]
    with pytest.raises(ValueError, match="ancestry"):
        plugin.evaluate(state)


def test_continuation_after_reuse_starts_new_development(lineage, monkeypatch):
    plugin, state = lineage
    state["progress"]["generations"].pop()
    record = state["progress"]["generations"][0]
    state["progress"].update(accepted_model=record, reuse_receipt=record["reuse_receipt"])
    observed = {}

    async def cycle(self, index, profile, current, context):
        observed.update(current=current, context=context)
        return {"status": "failed", "progress": current["progress"]}

    monkeypatch.setattr(GeneratedModelGoal, "cycle", cycle)
    folder = plugin.store.root / "episodes/cycle_002"
    folder.mkdir(parents=True)
    outcome = asyncio.run(plugin.cycle(2, None, state, "Explicit objective"))
    assert outcome["generation"] == 2
    assert "accepted_model" not in observed["current"]["progress"]
    assert "read_model_experience" in observed["context"]
    assert state["progress"]["accepted_model"] == record


def test_driver_rejects_underbudget_and_unknown_generation(tmp_path):
    service = ScienceGoalService()
    spec = {
        "objective": "Test evolution",
        "parameters": {"case": "mcba_itopride", "generations": "2"},
        "budget": {"max_cycles": 8, "model_calls": 192, "tool_calls": 96, "wall_seconds": 7200},
    }
    command = EvolvingModelDriver().prepare_command(service, tmp_path, spec)
    assert command[1].endswith("run_science_evolution.py") and command[-2:] == ["--generations", "2"]
    spec["budget"]["max_cycles"] = 3
    with pytest.raises(ValueError, match="episodes"):
        EvolvingModelDriver().prepare_command(service, tmp_path, spec)
    spec["parameters"]["generations"] = "99"
    with pytest.raises(ValueError, match="generations"):
        EvolvingModelDriver().prepare_command(service, tmp_path, spec)


def test_activity_only_exposes_bounded_public_role_reports(tmp_path):
    folder = tmp_path / "episodes/cycle_000/execution"
    folder.mkdir(parents=True)
    atomic_json(
        folder / "graph_history.json",
        [
            {"role_id": "principal_investigator", "content": "public report"},
            {"role_id": "private_reasoning", "content": "not public"},
        ],
    )
    (folder / "model_calls.jsonl").write_text("private request data")
    result = goal_activity(tmp_path, {"cycles": [], "inflight": {"index": 0}})
    assert result == [{"id": "0-0", "episode": 1, "role": "PI", "content": "public report"}]
