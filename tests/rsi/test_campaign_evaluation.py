"""Synthetic publication boundaries plus real isolated numerical prediction.

Fixtures are not native scientific publications or autonomous model evidence.
"""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_scientist_model_campaign as campaign_fixtures
from test_generated_models import CODE, SCOPE, TRAIN

from agent.harness import Budget, ModelBinding, Profile
from agent.harness.contracts import digest, file_hash
from agent.rsi import campaign_evaluation as evaluation
from agent.rsi import training_attempt_memory as memory
from agent.rsi.generated_model import fit_candidate
from agent.rsi.scientist_model_campaign import ALL_ALLOCATIONS, ModelCampaignStore
from agent.rsi.storage import atomic_json

task = campaign_fixtures.task


@pytest.fixture
def campaign(tmp_path, task, monkeypatch, request):
    repo = tmp_path / "repository"
    repo.mkdir()
    source = repo / "worker.py"
    source.write_text("# synthetic executor\n")
    profile = Profile(
        "science_expert",
        "expert-kimi-code",
        ModelBinding("test_fixture", "model", "https://example.org/v1"),
        Budget(wall_seconds=10, request_seconds=5, model_calls=3, tool_calls=2),
    ).public()
    allocations = [
        {
            "id": condition,
            "task": copy.deepcopy(task),
            "profile": copy.deepcopy(profile),
            "generations": getattr(request, "param", 1),
            "goal_budget": {"max_cycles": 4, "model_calls": 12, "tool_calls": 8, "wall_seconds": 60},
            "experience": {"condition": condition, "selection": "explicit", "entry_ids": []},
        }
        for condition in ("none", "source")
    ]
    store = ModelCampaignStore.create(
        tmp_path / "campaign",
        counting="program_entry.v1",  # Preserve the historical protocol in these fixtures.
        execution=ALL_ALLOCATIONS,
        objective="Synthetic evaluation",
        target_models=1001,
        library_root=tmp_path / "library",
        repository=repo,
        implementation={"worker.py": file_hash(source)},
        allocations=allocations,
        resources={"model_calls": 24, "tool_calls": 16, "wall_seconds": 120},
    )
    monkeypatch.setattr(evaluation, "source_identity", lambda: {"worker.py": file_hash(source)})
    pairs = [{"id": "source_minus_none", "left": "source", "right": "none", "factor": "library_experience"}]
    study = tmp_path / "study"
    return SimpleNamespace(store=store, study=study, tmp=tmp_path, pairs=pairs, source=source)


def admit(campaign):
    return evaluation.admit(campaign.store.root, campaign.study, pairs=campaign.pairs)


@pytest.mark.parametrize("change", [None, "missing", "same", "unknown", "budget", "feedback", "donor"])
def test_memory_pair_only_allows_declared_memory_difference(campaign, change):
    specs = campaign.store.specifications()
    for row in specs.values():
        row.pop("experience")
    specs["source"][memory.FIELD] = memory.CUMULATIVE_POLICY
    specs["none"][memory.FIELD] = memory.POLICY
    pairs = [{"id": "cumulative_minus_recent", "left": "source", "right": "none", "factor": memory.FIELD}]
    if change == "missing":
        del specs["none"][memory.FIELD]
    elif change == "same":
        specs["none"][memory.FIELD] = memory.CUMULATIVE_POLICY
    elif change == "unknown":
        specs["none"][memory.FIELD] = "unknown"
    elif change == "budget":
        specs["source"]["goal_budget"]["model_calls"] += 1
    elif change == "feedback":
        specs["source"]["training_baseline_feedback"] = "none"
    elif change == "donor":
        specs["source"]["experience"] = {"condition": "source", "selection": "explicit", "entry_ids": []}
    before = copy.deepcopy(specs)
    if change:
        with pytest.raises(ValueError):
            evaluation.paired_design(specs, pairs)
    else:
        assert evaluation.paired_design(specs, pairs) == pairs
    assert before == specs


def finish(campaign, *, successful=()):
    def update(state):
        state["status"] = "needs_tasks"
        for identity, row in state["allocations"].items():
            row["status"] = "completed" if identity in successful else "failed"
            row["attempts"] = [
                {
                    "status": row["status"],
                    "usage_known": identity in successful,
                    "charged": {
                        "model_calls": 1 if identity in successful else 12,
                        "tool_calls": 1 if identity in successful else 8,
                        "wall_seconds": 0.5 if identity in successful else 60,
                    },
                }
            ]

    campaign.store.mutate(update)
    for identity in successful:
        folder = campaign.store.goal(identity) / "episodes/cycle_000/execution"
        folder.mkdir(parents=True)
        (folder / "model_requests.jsonl").write_text(
            "\n".join(
                json.dumps(r)
                for r in [
                    {"event": "request", "call": 1},
                    {
                        "event": "response",
                        "call": 1,
                        "status": 200,
                        "usage": {"prompt_tokens": 12, "completion_tokens": 3},
                    },
                ]
            )
            + "\n"
        )
        (folder / "events.jsonl").write_text(json.dumps({"kind": "tool_call_start"}) + "\n")
        atomic_json(folder / "run.json", {"host_wall_seconds": 0.5})


@pytest.fixture
def model():
    return fit_candidate(TRAIN, SCOPE, {"source_code": CODE, "seed": 1})


def install_publications(monkeypatch, model, *, count=1):
    def published(store, identity, allocation, record):
        if record["status"] != "completed":
            return []
        return [
            {
                "generation": i + 1,
                "model_id": digest([identity, i]),
                "entry_id": digest(identity),
                "canonical_origin_is_this_goal": True,
                "model": copy.deepcopy(model),
            }
            for i in range(count)
        ]

    monkeypatch.setattr(evaluation, "published_models", published)


@pytest.mark.parametrize("state", ["queued", "running", "paused", "waiting_for_model"])
def test_future_scoring_requires_every_allocation_terminal(campaign, monkeypatch, state):
    admit(campaign)
    campaign.store.mutate(lambda s: s["allocations"]["none"].update(status=state))
    monkeypatch.setattr(evaluation, "published_models", lambda *a: pytest.fail("Scored before barrier"))
    with pytest.raises(ValueError, match="terminal"):
        evaluation.evaluate(campaign.study, campaign.tmp / "scores")
    assert not (campaign.tmp / "scores").exists()


def test_all_failed_allocations_remain_in_denominator(campaign):
    admit(campaign)
    finish(campaign)
    report = evaluation.evaluate(campaign.study, campaign.tmp / "scores")
    assert report["status"] == "completed" and report["allocated"] == 2
    assert report["publication_and_reuse_successes"] == report["future_scored"] == 0
    assert all(r["final_metrics"] is None and r["original_status"] == "failed" for r in report["rows"])
    contrast = report["paired_contrasts"][0]
    assert not contrast["both_scored"] and contrast["physical_request_difference"] is None
    assert all(v is None for v in contrast["metric_differences"].values())
    assert all(
        r["cost"]["physical_requests_observed"] == 0 and r["cost"]["charged_budget"]["model_calls"] == 12
        for r in report["rows"]
    )


def test_real_isolated_prediction_and_replay_preserve_failure(campaign, monkeypatch, model):
    admit(campaign)
    finish(campaign, successful=["source"])
    install_publications(monkeypatch, model)
    actual = evaluation.get_model_plugin

    def checked(plugin_id):
        plugin = actual(plugin_id)

        def predict(model, rows, scope):
            assert all("target" not in row for row in rows)
            return plugin.predict(model, rows, scope)

        return SimpleNamespace(**{**plugin.__dict__, "predict": predict})

    monkeypatch.setattr(evaluation, "get_model_plugin", checked)
    output = campaign.tmp / "scores"
    report = evaluation.evaluate(campaign.study, output)
    assert report["allocated"] == 2 and report["publication_and_reuse_successes"] == report["future_scored"] == 1
    row = next(r for r in report["rows"] if r["allocation_id"] == "source")
    assert row["final_metrics"]["mse_skill_vs_training_mean"] == pytest.approx(1)
    assert row["cost"]["physical_requests_observed"] == 1 and row["cost"]["input_tokens_observed"] == 12
    assert evaluation.audit(campaign.study, output, campaign.tmp / "audit")["status"] == "passed"


@pytest.mark.parametrize("attack", ["report", "barrier", "predictions"])
def test_saved_results_cannot_be_rehashed_into_success(campaign, monkeypatch, model, attack):
    admit(campaign)
    finish(campaign, successful=["source"])
    install_publications(monkeypatch, model)
    output = campaign.tmp / "scores"
    evaluation.evaluate(campaign.study, output)
    if attack == "report":
        path = output / "report.json"
        value = evaluation.read(path)
        value["future_scored"] = 2
        atomic_json(path, value)
    elif attack == "barrier":
        path = output / "development_barrier.json"
        value = evaluation.read(path)
        value["future_scoring_started"] = True
        atomic_json(path, value)
    else:
        path = output / "predictions/source/1.json"
        value = evaluation.read(path)
        value[0]["prediction"] += 1
        atomic_json(path, value)
    with pytest.raises(ValueError):
        evaluation.audit(campaign.study, output, campaign.tmp / "audit")
    assert evaluation.read(campaign.tmp / "audit/audit.json")["status"] == "failed"


def test_admission_cannot_be_retrofitted_after_execution(campaign):
    finish(campaign)
    with pytest.raises(ValueError, match="precede"):
        admit(campaign)


@pytest.mark.parametrize("field", ["budget", "donors", "model", "generations"])
def test_unmatched_contrasts_are_rejected(campaign, field):
    specs = copy.deepcopy(campaign.store.specifications())
    row = specs["source"]
    if field == "budget":
        row["goal_budget"]["model_calls"] += 1
    elif field == "donors":
        row["experience"]["entry_ids"] = ["a" * 64]
    elif field == "model":
        row["profile"]["model"]["model"] = "other"
    else:
        row["generations"] = 2
    with pytest.raises(ValueError, match="differ"):
        evaluation.paired_design(specs, campaign.pairs)


def test_evaluator_source_change_invalidates_admission(campaign):
    admit(campaign)
    campaign.source.write_text("# changed\n")
    with pytest.raises(ValueError, match="evaluation source"):
        evaluation.load_study(campaign.study)


@pytest.mark.parametrize("campaign", [2], indirect=True)
def test_final_generation_is_not_selected_by_future_performance(campaign, monkeypatch, model):
    admit(campaign)
    finish(campaign, successful=["source"])
    worse = copy.deepcopy(model)
    worse["model_state"]["weights"][0] += 50

    def publications(store, identity, allocation, record):
        if identity == "none":
            return []
        return [
            {"generation": i + 1, "model_id": digest(i), "entry_id": digest(i), "model": copy.deepcopy(m)}
            for i, m in enumerate([model, worse])
        ]

    monkeypatch.setattr(evaluation, "published_models", publications)
    report = evaluation.evaluate(campaign.study, campaign.tmp / "scores")
    row = report["rows"][1]
    assert row["generations"][0]["metrics"]["mse"] < 1e-12
    assert row["final_metrics"]["mse"] > 2400 and row["future_lineage_skill_change"] < 0


def test_cost_counts_retries_and_keeps_missing_usage_unknown(campaign):
    finish(campaign, successful=["source"])
    folder = campaign.store.goal("source") / "episodes/cycle_000/execution"
    with (folder / "model_requests.jsonl").open("a") as f:
        f.write(json.dumps({"event": "request", "call": 2, "logical_call": 1}) + "\n")
    record = campaign.store.read()["allocations"]["source"]
    record["attempts"][0]["charged"]["model_calls"] = 2
    result = evaluation.observed_usage(campaign.store.goal("source"), record)
    assert result["physical_requests_observed"] == 2 and result["call_accounting_complete"]
    assert not result["token_accounting_complete"] and result["input_tokens_observed"] == 12


def test_duplicate_publication_uses_this_runs_fitted_state(campaign, monkeypatch, model):
    root = campaign.store.goal("source")
    root.mkdir(parents=True)
    atomic_json(root / "goal.json", {"fixture": True})
    mid, entry = digest("own_model"), digest("canonical_entry")
    key = {"program": "fixture", "training": digest(TRAIN)}
    key_id = digest(key)
    packet = {"model": model}
    own = {"key": key, "packet": packet, "origin": {"model_id": mid}}
    origin = {
        **own["origin"],
        "goal": str(root.relative_to(campaign.tmp)),
        "key_id": key_id,
        "packet_id": digest(packet),
    }
    outcome = {"status": "published", "model_id": mid, "entry_id": entry, "key_id": key_id, "origin_id": digest(origin)}
    report = {
        "kind": "model_goal_publication_attempt",
        "origin_goal": origin["goal"],
        "goal_sha256": file_hash(root / "goal.json"),
        "outcomes": [outcome],
    }
    objects = {
        ("publication_attempts", digest(report)): report,
        ("origins/" + key_id, digest(origin)): origin,
        ("packets", digest(packet)): packet,
    }
    canonical_model = copy.deepcopy(model)
    canonical_model["model_state"]["weights"][0] += 999

    class Library:
        owner = campaign.tmp

        def __init__(self, root):
            pass

        def goal(self, path):
            return path

        def _object(self, area, identity):
            return objects[(area, identity)]

        def resolve(self, identity, *, replay_training):
            assert replay_training
            return {
                "key": key,
                "packet": {"model": canonical_model},
                "origin": {},
                "origin_goal": str(campaign.tmp / "earlier_goal"),
            }

    monkeypatch.setattr(evaluation, "ScientistModelLibrary", Library)
    monkeypatch.setattr(evaluation, "bind_goal", lambda *a: {"progress": {"accepted_model": {"model_id": mid}}})
    monkeypatch.setattr(evaluation, "validate_model_goal", lambda *a, **k: own)
    record = {
        "status": "completed",
        "entries": [entry],
        "attempts": [{"publication": {**report, "report_id": digest(report)}}],
    }
    got = evaluation.published_models(campaign.store, "source", campaign.store.specifications()["source"], record)
    assert got[0]["model"] == model and got[0]["model"] != canonical_model
    assert got[0]["canonical_origin_is_this_goal"] is False


def test_changed_task_asset_rejected_before_study_creation(campaign):
    task = campaign.store.specifications()["source"]["task"]
    path = Path(task["data"]["training"]["path"])
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="asset changed"):
        admit(campaign)
    assert not campaign.study.exists()


def test_invalid_publication_is_an_evidence_failure_not_a_missing_model(campaign, monkeypatch):
    admit(campaign)
    finish(campaign, successful=["source"])

    def invalid(*args):
        raise ValueError("Synthetic changed origin")

    monkeypatch.setattr(evaluation, "published_models", invalid)
    report = evaluation.evaluate(campaign.study, campaign.tmp / "scores")
    assert report["status"] == "evidence_failed"
    assert report["allocated"] == 2 and report["paired_contrasts"] == []
    assert all(row["evaluation_status"] == "evidence_failed" for row in report["rows"])


def test_new_study_rejects_coverage_target_early_stopping(campaign):
    def change(state):
        state["definition"].pop("execution_policy")
        state["definition_sha256"] = digest(state["definition"])

    campaign.store.mutate(change)
    with pytest.raises(ValueError, match="all_allocations"):
        admit(campaign)
    assert not campaign.study.exists()


def test_nominal_terminal_without_execution_attempt_cannot_unlock_future(campaign, monkeypatch):
    admit(campaign)
    finish(campaign)
    campaign.store.mutate(lambda s: s["allocations"]["none"].update(attempts=[]))
    monkeypatch.setattr(evaluation, "published_models", lambda *a: pytest.fail("Scored unexecuted arm"))
    with pytest.raises(ValueError, match="execution attempt"):
        evaluation.evaluate(campaign.study, campaign.tmp / "scores")
    assert not (campaign.tmp / "scores").exists()


def test_insufficient_budget_rejected_before_study_admission(campaign):
    campaign.store.mutate(lambda s: s["usage"].update(model_calls=1))
    with pytest.raises(ValueError, match="reserve all"):
        admit(campaign)
    assert not campaign.study.exists()


def test_admission_binds_resources_for_every_predeclared_allocation(campaign):
    value = admit(campaign)
    assert value["execution_policy"] == ALL_ALLOCATIONS
    assert value["execution_resources"]["required_remaining_reservations"] == {
        "model_calls": 24,
        "tool_calls": 16,
        "wall_seconds": 120,
    }
