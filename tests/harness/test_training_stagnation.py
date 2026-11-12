"""Repeated failure handling uses host fit receipts, not critic claims."""

import csv
import hashlib
import io
import json
from pathlib import Path

import pytest
from test_autonomous_research import profile
from test_research_training import LINEAR, dataset, service_at

from agent.harness import autonomous_research as ar
from agent.harness.research_attempt_resolution import training_progress, training_snapshot
from agent.harness.research_data_store import store_for_goal

BROKEN = """def fit(training, seed):
    return training['inputs']
def predict(model, inputs):
    return {'value': 1}
"""


async def fail(service, name="broken.py", code=BROKEN, seed=0):
    await service.write_research_file(name, code)
    with pytest.raises(ValueError, match="Training attempt"):
        await service.train_research_model(
            "Failed fixture", name, "evaluation.json", "Synthetic repair test", seed=seed
        )


@pytest.mark.asyncio
async def test_same_failed_algorithm_across_cycles_is_not_repaired_by_renaming(tmp_path):
    service = await service_at(tmp_path)
    report = None
    for index in range(3):
        before = training_snapshot(service.development)
        await fail(service, f"revision_{index}.py", BROKEN + f"\n# revision {index}\n")
        report = training_progress(service.development, before, report)
        assert report["same_failure_cycles"] == index + 1
        assert report["intervention_required"] is (index == 2)
    assert report["fit_attempts"] == 1 and report["measured_generations"] == 0
    assert "KeyError" in report["error"]
    assert len(report["evidence"]) == 1
    assert "splits" not in json.dumps(report) and str(tmp_path) not in json.dumps(report)
    before = training_snapshot(service.development)
    await service.write_research_file("repaired.py", LINEAR)
    await service.train_research_model("Repaired fixture", "repaired.py", "evaluation.json", "Real synthetic fit")
    fixed = training_progress(service.development, before, report)
    assert fixed["measured_generations"] == 1
    assert fixed["same_failure_cycles"] == 0 and not fixed["intervention_required"]


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["algorithm", "seed", "data", "no_attempt", "tampered_receipt"])
async def test_changed_experiment_or_unproven_receipt_breaks_the_streak(tmp_path, change):
    service = await service_at(tmp_path)
    before = training_snapshot(service.development)
    await fail(service)
    prior = training_progress(service.development, before)
    before = training_snapshot(service.development)
    if change == "data":
        observations = dataset()
        observations["splits"]["train"][0]["target"] += 0.01
        await service.write_research_file("evaluation.json", json.dumps(observations))
    if change != "no_attempt":
        code = (
            BROKEN.replace("training['inputs']", "training['different_missing_field']")
            if change == "algorithm"
            else BROKEN
        )
        await fail(service, code=code, seed=1 if change == "seed" else 0)
    if change == "tampered_receipt":
        identity = (training_snapshot(service.development)["attempts"] - before["attempts"]).pop()
        path = service.development.root / "training" / identity / "receipt.json"
        value = json.loads(path.read_text())
        value["error"] = "Forged matching failure"
        path.write_text(json.dumps(value))
    report = training_progress(service.development, before, prior)
    assert report["same_failure_cycles"] <= 1 and not report["intervention_required"]


@pytest.mark.asyncio
async def test_real_scientist_stops_unchanged_failures_before_exhaustion(tmp_path, monkeypatch):
    class Relay:
        calls = 0
        base_url = "http://unused"
        token = "offline-fixture"

        def __init__(self, **kwargs):
            Path(kwargs["output"]).touch()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(ar, "EvidenceGateway", Relay)
    monkeypatch.setattr(ar, "audit_requests", lambda path: {"passed": True, "model_calls": 4})
    monkeypatch.setenv("VENUS_JOB_QUEUE_ROOT", str(tmp_path / "queue"))
    monkeypatch.setenv("VENUS_USAGE_DB", str(tmp_path / "usage.sqlite3"))
    store = ar.create(
        tmp_path / "goal",
        profile(),
        "Develop a model using synthetic observations",
        {"max_cycles": 5, "model_calls": 60, "tool_calls": 60, "wall_seconds": 600},
    )
    data = dataset()
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["id", "group", "split", "x", "target"])
    writer.writeheader()
    for split, rows in data["splits"].items():
        for row in rows:
            writer.writerow(
                {
                    "id": row["id"],
                    "group": row["group"],
                    "split": split,
                    "x": row["inputs"]["x"],
                    "target": row["target"],
                }
            )
    raw = buffer.getvalue().encode()
    store_for_goal(store.root).ingest("inputs/data.csv", raw, hashlib.sha256(raw).hexdigest())
    seen = []
    attempts = []
    evaluation = None

    async def consult(role, context):
        if role == "PI":
            seen.append(context["progress"].get("training_progress", {}).get("same_failure_cycles", 0))
            return "Repair the actual fitting error using the supplied training contract."
        if role == "CB":
            return json.dumps(
                {
                    "summary": "Synthetic fitting fixture",
                    "deliverable": "model",
                    "needs_input": [],
                    "steps": [
                        {
                            "id": "fit",
                            "title": "Fit a predictor",
                            "acceptance": "Measured predictions",
                            "approach": "Fit the synthetic relation",
                            "verification": "model",
                        }
                    ],
                }
            )
        assert role == "SC"
        return json.dumps(
            {
                "decision": "needs_revision",
                "summary": "There is no measured model",
                "issues": ["Repair the implementation and fit it"],
            }
        )

    async def execute(request, service, events):
        nonlocal evaluation
        if evaluation is None:
            prepared = await service.prepare_research_dataset(
                data_file="inputs/data.csv",
                target_column="target",
                input_columns=["x"],
                group_column="group",
                id_column="id",
                split_column="split",
                provenance="Synthetic fixture",
                grouping="Distinct observations",
            )
            evaluation = prepared["evaluation_file"]
        name = f"revision_{len(attempts)}.py"
        await service.write_research_file(name, BROKEN)
        identity = f"fit_{len(attempts)}"
        events.append(
            {
                "kind": "tool_call_start",
                "tool_name": "train_research_model",
                "tool_call_id": identity,
                "args": {"source_file": name, "parent_experiment_id": "", "variant": "baseline"},
            }
        )
        try:
            await service.train_research_model("Failed fixture", name, evaluation, "Synthetic repair test")
        except ValueError as error:
            attempts.append(str(error))
            events.append({"kind": "tool_result", "tool_call_id": identity, "is_error": True, "output": str(error)})
        else:
            raise AssertionError("Fixture must fail actual fitting")
        await service.submit_research_checkpoint("Actual fitting failed", [], [name], [])
        events.append({"kind": "completed", "text": "Repair required", "event_audit": {"passed": True}})

    result = await ar.AutonomousResearch(
        store, "offline-fixture", {}, consult=consult, executor=execute, tools=[]
    ).run()
    assert result["status"] == "interrupted" and len(result["cycles"]) == 3
    assert seen == [0, 1, 2] and len(attempts) == 3
    assert result["usage"]["model_calls"] == 12 < 60
    assert result["events"][-1]["event"] == "unchanged_training_failure"
    assert result["progress"]["execution_blocker"]["source"] == "research_progress"
    assert not result["progress"].get("deliveries") and not result["progress"].get("research_completed")
    assert (store.root / "episodes/cycle_002/training_stagnation.json").is_file()
    assert len(list((store.root / "development/training").glob("*/receipt.json"))) == 3
    saved = (store.root / "cycle_002.json").read_bytes()
    store.resume()
    assert store.read()["status"] == "active" and store.read()["usage"] == result["usage"]
    assert (store.root / "cycle_002.json").read_bytes() == saved
