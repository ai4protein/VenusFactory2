"""Synthetic scheduling/native seams; real numerical fits, no external API."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import test_shared_model_campaign as fixtures

from agent.rsi.datasets import model_tasks
from agent.rsi.engine import shared_campaign_evaluation as evaluation
from agent.rsi.engine import shared_campaign_study as study
from agent.rsi.engine.campaign_evidence import read
from agent.rsi.engine.scientist_model_library import ScientistModelLibrary
from agent.rsi.engine.storage import atomic_json


@pytest.fixture
def terminal(tmp_path):
    specifications = [fixtures.batch(tmp_path, 0), fixtures.batch(tmp_path, 1)]
    store = fixtures.create(tmp_path / "run", specifications)

    def finish(state):
        for record in state["batches"].values():
            record["status"] = "failed"
            record["attempts"] = [
                {
                    "status": "failed",
                    "usage_known": False,
                    "charged": {"model_calls": 1, "tool_calls": 0, "wall_seconds": 1.0},
                }
            ]
        state.update(status="publication_invalid", usage={"model_calls": 2, "tool_calls": 0, "wall_seconds": 2.0})

    store.mutate(finish)
    return SimpleNamespace(store=store, library=tmp_path / "run/library", study=tmp_path / "study", tmp=tmp_path)


def admit(terminal):
    return study.admit(terminal.store.root, terminal.library, terminal.study)


def test_admission_never_deserializes_labels_or_runs_a_model(terminal, monkeypatch):
    monkeypatch.setattr(model_tasks, "load_task", lambda *a, **k: pytest.fail("Decoded data before barrier"))
    monkeypatch.setattr(ScientistModelLibrary, "resolve_many", lambda *a, **k: pytest.fail("Executed before barrier"))
    definition = admit(terminal)
    assert len(definition["snapshot"]["tasks"]) == 4
    assert not definition["final_labels_decoded"]
    assert study.load(terminal.study) == definition


@pytest.mark.parametrize("status", ["running", "queued", "paused", "waiting_for_model"])
def test_active_allocations_block_admission(terminal, status):
    terminal.store.mutate(lambda s: s["batches"]["batch_1"].update(status=status))
    with pytest.raises(ValueError, match="terminal"):
        admit(terminal)
    assert not terminal.study.exists()


@pytest.mark.parametrize("target", ["asset", "source", "state", "environment", "study"])
def test_changed_bindings_reject_before_scoring(terminal, monkeypatch, target):
    definition = admit(terminal)
    if target == "asset":
        task = definition["snapshot"]["specifications"]["batch_0"]["tasks"][0]
        Path(task["data"]["future"]["path"]).write_text("[]")
    elif target == "source":
        monkeypatch.setattr(study, "source_identity", lambda: {})
    elif target == "environment":
        monkeypatch.setattr(study, "versions", lambda: {})
    elif target == "study":
        saved = read(terminal.study / "study.json")
        saved["definition"]["snapshot"]["tasks"].popitem()
        atomic_json(terminal.study / "study.json", saved)
    else:
        terminal.store.mutate(lambda s: s.update(status="changed"))
    monkeypatch.setattr(evaluation, "assess_batch", lambda *a, **k: pytest.fail("Scored altered evidence"))
    with pytest.raises(ValueError):
        evaluation.evaluate(terminal.study, terminal.tmp / "scores")
    assert not (terminal.tmp / "scores").exists()


def test_all_failed_scopes_receive_controls_and_cost_is_once_per_batch(terminal):
    admit(terminal)
    report = evaluation.evaluate(terminal.study, terminal.tmp / "scores")
    assert report["status"] == "completed"
    assert report["allocated"] == report["assessed"] == report["controls_completed"] == 4
    assert report["verified_callable"] == report["future_scored"] == 0
    assert sum(batch["cost"]["charged_budget"]["model_calls"] for batch in report["batches"]) == 2
    assert all(
        "cost" not in row and row["final_metrics"] is None and row["evaluation_status"] == "no_published_model"
        for batch in report["batches"]
        for row in batch["rows"]
    )
    assert all(
        row["controls"]["independent_refit"]["baseline_refit_performed"]
        for batch in report["batches"]
        for row in batch["rows"]
    )
    assert evaluation.audit(terminal.study, terminal.tmp / "scores", terminal.tmp / "replay")["status"] == "passed"


def test_resume_reuses_verified_checkpoints_and_rejects_modified_prediction(terminal, monkeypatch):
    admit(terminal)

    class SimulatedStop(Exception):
        pass

    def stop(_):
        raise SimulatedStop()

    with pytest.raises(SimulatedStop):
        evaluation.evaluate(terminal.study, terminal.tmp / "scores", progress=stop)
    delegate = evaluation.assess_batch
    calls = []

    def record(*args, **kwargs):
        calls.append(args[3])
        return delegate(*args, **kwargs)

    monkeypatch.setattr(evaluation, "assess_batch", record)
    report = evaluation.evaluate(terminal.study, terminal.tmp / "scores", resume=True)
    assert report["assessed"] == 4 and calls == ["batch_1"]
    artifact = evaluation.artifact_bindings(report["batches"][0])[0]
    (terminal.tmp / "scores" / artifact["path"]).write_text("[]")
    with pytest.raises(ValueError, match="predictions changed"):
        evaluation.evaluate(terminal.study, terminal.tmp / "scores", resume=True)


def test_control_failure_is_not_hidden_by_missing_model(terminal, monkeypatch):
    admit(terminal)

    def fail(*a, **k):
        raise ValueError("synthetic independent refit failure")

    monkeypatch.setattr(evaluation, "fixed_controls", fail)
    result = evaluation.evaluate(terminal.study, terminal.tmp / "scores")
    assert result["status"] == "evidence_failed" and result["assessed"] == 4 and not result["controls_completed"]


@pytest.mark.parametrize("area", ["study", "campaign", "library"])
def test_outputs_cannot_overwrite_their_own_evidence(terminal, area):
    admit(terminal)
    base = {"study": terminal.study, "campaign": terminal.store.root, "library": terminal.library}[area]
    with pytest.raises(ValueError, match="outside"):
        evaluation.evaluate(terminal.study, base / "scores")


@pytest.mark.asyncio
async def test_partial_native_fixture_retains_denominator_and_real_predictions(tmp_path, monkeypatch):
    store = fixtures.create(tmp_path / "run", [fixtures.batch(tmp_path, 0, reject=True)], target=2, full_source=True)
    run, _ = fixtures.executor(store, monkeypatch, native_seam=True)
    state = await fixtures.campaign.run_campaign(store, run)
    assert state["batches"]["batch_0"]["status"] == "budget_exhausted"
    library = tmp_path / "run/library"
    definition = study.admit(store.root, library, tmp_path / "study")
    assert len(definition["snapshot"]["publications"]["batch_0"]["models"]) == 1
    report = evaluation.evaluate(tmp_path / "study", tmp_path / "scores")
    assert report["status"] == "completed" and report["allocated"] == 2 and report["future_scored"] == 1
    batch = report["batches"][0]
    assert batch["original_goal_status"] == batch["original_batch_status"] == "budget_exhausted"
    assert not batch["completed_goal_with_all_scopes_verified"]
    assert all(row["controls"]["status"] == "completed" for row in batch["rows"])
    assert fixtures.read(store.root / "campaign.json") == state
    evaluation.audit(tmp_path / "study", tmp_path / "scores", tmp_path / "replay")
    # Removing only the registry index cannot silently improve or reduce the cohort.
    model = next(iter(definition["snapshot"]["publications"]["batch_0"]["models"].values()))
    (library / "index" / (model["key_id"] + ".json")).unlink()
    with pytest.raises(ValueError, match="inventory is incomplete"):
        study.admit(store.root, library, tmp_path / "incomplete")


def test_mixed_cohort_keeps_sequence_scopes_and_unsupported_alignment(tmp_path):
    from test_generated_sequence import FUTURE, SCOPE, TRAIN, VALID

    from agent.harness.contracts import file_hash

    batches = [fixtures.batch(tmp_path, 0), fixtures.batch(tmp_path, 1)]
    batches[1]["training_baseline_feedback"] = "none"
    for index, task in enumerate(batches[1]["tasks"]):
        task.update(plugin_id="generated_sequence.v1", scope={**SCOPE, "protein": f"sequence_fixture_{index}"})
        reference = tmp_path / f"sequence_reference_{index}.txt"
        reference.write_text("Synthetic sequence-control fixture")
        task["references"] = [{"path": str(reference), "sha256": file_hash(reference)}]
        for phase, rows in (("training", TRAIN), ("validation", VALID), ("future", FUTURE)):
            path = tmp_path / f"sequence_{index}_{phase}.json"
            atomic_json(path, rows)
            task["data"][phase] = {"path": str(path), "sha256": file_hash(path)}
    store = fixtures.create(tmp_path / "run", batches)

    def finish(state):
        for record in state["batches"].values():
            record.update(
                status="failed",
                attempts=[
                    {
                        "status": "failed",
                        "usage_known": False,
                        "charged": {"model_calls": 1, "tool_calls": 0, "wall_seconds": 1.0},
                    }
                ],
            )
        state.update(status="publication_invalid", usage={"model_calls": 2, "tool_calls": 0, "wall_seconds": 2.0})

    store.mutate(finish)
    definition = study.admit(store.root, tmp_path / "run/library", tmp_path / "study")
    assert len(definition["snapshot"]["tasks"]) == 4
    report = evaluation.evaluate(tmp_path / "study", tmp_path / "scores")
    assert report["status"] == "completed" and report["allocated"] == report["assessed"] == 4
    sequence_rows = report["batches"][1]["rows"]
    assert all(row["controls"]["status"] == "partial" for row in sequence_rows)
    assert all(row["controls"]["independent_refit_methods"] == [] for row in sequence_rows)
    for row in sequence_rows:
        controls = row["controls"]["outcomes"]
        assert [c["status"] for c in controls] == ["completed"] * 4 + ["unsupported"] * 2
        assert controls[0]["metrics"]["mse_skill_vs_training_mean"] == pytest.approx(0)
        assert all(c["metrics"] is None for c in controls[-2:])
        assert "seconds" not in str(row["controls"])
    assert evaluation.audit(tmp_path / "study", tmp_path / "scores", tmp_path / "replay")["status"] == "passed"
