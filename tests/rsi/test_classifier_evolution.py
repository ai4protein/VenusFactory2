"""Classification lineage, promotion rejection and future-scoring barriers."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
import run_science_goal as base
from scripts.rsi.evaluation.audit_science_evolution import audit
from run_classifier_evolution import EvolvingClassifierGoal, score_terminal_campaign
from test_generated_classifier import CODE, ROWS, SCOPE
from test_nucb_science_goal import setup_goal

from agent.harness.contracts import digest
from agent.harness.host import write_json
from agent.rsi.engine.model_plugins import get_model_plugin
from agent.rsi.datasets.source_lineage import program_hash
from agent.science_goal import run_goal
from backend.web_v2.science_goals import EvolvingModelDriver, ScienceGoalService


def classifier_goal(tmp_path):
    store = setup_goal(tmp_path)

    def configure(state):
        state["definition"]["goal"].update(
            max_cycles=6, criteria=["accepted_model", "fresh_session_reuse", "verified_lineage"]
        )
        state["definition_sha256"] = digest(state["definition"])

    store.mutate(configure)
    path = store.root / "protocol.json"
    protocol = json.loads(path.read_text())
    protocol.update(case="nucb", development_feedback="source_and_metrics", evolution={"schema": 2, "generations": 2})
    write_json(path, protocol)
    return store


def test_comment_docstring_and_seed_changes_are_not_new_programs():
    assert program_hash(CODE) == program_hash('"""New description."""\n# refactor\n' + CODE)
    assert program_hash(CODE) != program_hash(CODE.replace("/16.0", "/64.0"))


@pytest.mark.asyncio
@pytest.mark.parametrize("history_policy", ["source_and_metrics", "metrics_only", "independent_restart"])
async def test_classifier_successor_rejection_recovery_and_independent_audit(tmp_path, monkeypatch, history_policy):
    store = classifier_goal(tmp_path)
    path = store.root / "protocol.json"
    protocol = json.loads(path.read_text())
    protocol.update(schema=3, history_policy=history_policy)
    write_json(path, protocol)
    goal = EvolvingClassifierGoal(store, "fixture")
    plugin = get_model_plugin(goal.model_plugin_id)
    phases, development = [], []
    feedback = []

    async def execute(profile, task, runtimes, credential):
        phases.append(task.phase)
        if task.phase == "develop":
            feedback.append(task.prompt)
            # First weak model passes. Duplicate successor cannot improve.
            # Third source genuinely changes probability computation and improves.
            source = CODE.replace("/16.0", "/64.0") if len(development) < 2 else CODE
            model = plugin.fit(ROWS[:720], SCOPE, {"source_code": source, "seed": 19})
            write_json(task.workspace / "candidate.json", model)
            write_json(task.workspace / "source_provenance.json", {"rationale": "Fixture code revision"})
            development.append(source)
        else:
            state = store.read()
            record = state["progress"]["generations"][-1]
            lc, mid = goal.lifecycle({"progress": {"accepted_model": record}})
            inputs = json.loads((task.workspace / "inputs.json").read_text())
            assert all(set(row) == {"id", "sequence"} for row in inputs)
            write_json(task.workspace / "predictions.json", plugin.predict(lc._model(mid), inputs, SCOPE))
        return {
            "status": "completed",
            "session_id": f"session-{len(phases)}",
            "host_wall_seconds": 0.01,
            "audit": {"requests": {"model_calls": 1}, "events": {"tool_calls": 1}},
        }

    monkeypatch.setattr(base, "execute", execute)
    monkeypatch.setattr(base, "audit_events", lambda *a, **k: {"passed": True})
    state = await run_goal(store, goal.cycle, goal.evaluate)
    assert state["status"] == "completed"
    assert phases == ["develop", "reuse", "develop", "develop", "reuse"]
    assert state["usage"]["model_calls"] == 5  # Rejected successor still charged.
    assert state["cycles"][2]["status"] == "candidate_rejected"
    assert len(state["progress"]["generations"]) == 2
    if history_policy == "independent_restart":
        assert "DEVELOPMENT acceptance feedback" not in feedback[-1]
    else:
        assert "DEVELOPMENT acceptance feedback" in feedback[-1]
    exposed = json.loads((store.root / "episodes/cycle_003/development_feedback.json").read_text())
    if history_policy == "independent_restart":
        assert exposed == []
    elif history_policy == "metrics_only":
        assert exposed and all("source_code" not in row for row in exposed)
    else:
        assert exposed and all("source_code" in row for row in exposed)
    projection = goal.development_experience(state)
    assert bool(projection["entries"]) == (history_policy == "source_and_metrics")
    entries = [goal.experience()._entry(r["experience_id"]) for r in state["progress"]["generations"]]
    assert bool(entries[1]["parents"]) == (history_policy == "source_and_metrics")
    records = state["progress"]["generations"]
    lc0, mid0 = goal.lifecycle({"progress": {"accepted_model": records[0]}})
    lc1, mid1 = goal.lifecycle({"progress": {"accepted_model": records[1]}})
    assert (
        lc1.policy["minimum_skill"]
        == json.loads((lc0.root / f"assessment_{mid0}.json").read_text())["metrics"][plugin.acceptance_metric]
    )
    assert goal.promotion_error(lc0._model(mid0), [records[0]]) is not None
    assert goal.promotion_error(lc1._model(mid1), [records[0]]) is None
    assert not (store.root / "future_metrics.json").exists()
    independent = audit(store.root, score_future=False)
    assert independent["status"] == "passed" and not independent["future_scored"]
    assert all(x["reuse_replayed"] for x in independent["generations"])
    scored = score_terminal_campaign([store.root], tmp_path / "scoring")
    assert scored[0]["generations"][0]["source_sha256"] == records[0]["source_sha256"]
    assert len(scored[0]["generations"]) == 2
    assert scored[0]["selected_future_skill"] > scored[0]["frozen_first_source_skill"]
    with pytest.raises(FileExistsError):
        score_terminal_campaign([store.root], tmp_path / "scoring")


def test_future_barrier_rejects_active_goal_before_any_scoring(tmp_path):
    store = classifier_goal(tmp_path)
    output = tmp_path / "forbidden"
    with pytest.raises(ValueError, match="Every declared allocation"):
        score_terminal_campaign([store.root], output)
    assert not output.exists()


def test_failed_goal_remains_in_future_scoring_denominator(tmp_path):
    store = classifier_goal(tmp_path)
    store.mutate(lambda state: state.update(status="budget_exhausted"))
    result = score_terminal_campaign([store.root], tmp_path / "score")
    assert len(result) == 1 and result[0]["generations"] == []
    assert result[0]["selected_future_skill"] == result[0]["frozen_first_source_skill"] == 0.0


@pytest.mark.parametrize("case", ["nucb", "gfp"])
def test_web_routes_classifier_prepare_and_resume_by_frozen_case(tmp_path, case):
    service = ScienceGoalService()
    spec = {
        "objective": "Develop classifier versions",
        "parameters": {"case": case, "generations": "2"},
        "budget": {"max_cycles": 8, "model_calls": 96, "tool_calls": 48, "wall_seconds": 3600},
    }
    driver = EvolvingModelDriver()
    command = driver.prepare_command(service, tmp_path, spec)
    assert command[1].endswith("run_classifier_evolution.py")
    assert command[command.index("--case") + 1] == case
    write_json(tmp_path / "protocol.json", {"case": case})
    service.frozen_profile = lambda folder: type(
        "Profile", (), {"model": type("Model", (), {"credential_env": "NO_CLASSIFIER_TEST_KEY"})()}
    )()
    service.frozen_credential_path = lambda folder: None
    command = driver.run_command(service, tmp_path, "resume")
    assert command[2].endswith("run_classifier_evolution.py") and command[3] == "resume"


@pytest.mark.parametrize("case", ["nucb", "gfp"])
def test_prepare_initializes_data_and_freezes_classifier_protocol(tmp_path, monkeypatch, case):
    from types import SimpleNamespace

    import run_classifier_evolution as entry

    reference = setup_goal(tmp_path / "reference")
    profile_path = tmp_path / "profile.json"
    write_json(profile_path, reference.read()["definition"]["profile"])
    data = (SCOPE, ROWS[:720], ROWS[720:840], ROWS[840:900], {"source": "fixture"}, {"split": "fixture"})
    monkeypatch.setattr(entry, "prepare_nucb_case", lambda *args: data)
    monkeypatch.setattr(entry, "prepare_htfunclib_model_case", lambda *args: data)
    args = SimpleNamespace(
        profile=profile_path,
        objective="Evolve and reuse a classifier",
        case=case,
        generations=2,
        max_cycles=6,
        model_calls=64,
        tool_calls=32,
        wall_seconds=1800,
        assets=tmp_path,
        rebuilt=tmp_path,
        output=tmp_path / "prepared",
        condition="science_goal",
    )
    store = entry.prepare(args)
    protocol = json.loads((store.root / "protocol.json").read_text())
    assert protocol["model_plugin_id"] == entry.PLUGINS[case]
    assert protocol["evolution"]["metric"] == "brier_skill_vs_training_prior"
    for name, rows in zip(["training", "validation", "future"], data[1:4], strict=True):
        assert json.loads((store.root / "data" / (name + ".json")).read_text()) == rows
        assert protocol["data_hashes"][name] == digest(rows)
    assert store.read()["cycles"] == [] and not (store.root / "future_metrics.json").exists()
