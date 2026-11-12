"""Kill/resume the real Scientist driver after isolated fitting.

Only role/executor responses are controlled fixtures. Training, evaluation,
receipts, SC binding, technical validation, packaging and reuse execute for real.
Synthetic observations are never counted as biological research models.
"""

import asyncio
import csv
import hashlib
import io
import json
import multiprocessing
import os
import sys
import time
import traceback
import zipfile
from pathlib import Path

import pytest
from test_autonomous_research import profile
from test_private_research_capabilities import configure_access, verify_private_reuse
from test_research_development import review_spec
from test_scientific_acceptance import BASELINE, LEARNER, accepted

from agent.harness import autonomous_research as ar
from agent.harness import scientific_acceptance as sc
from agent.harness.research_data_store import store_for_goal
from agent.harness.research_development import Development
from agent.harness.research_workspace import invoke_bundle, run_capability
from agent.science_goal import GoalStore
from backend.web_v2.science_goals import ROOT, ScienceGoalService


class LocalRelay:
    calls = 0
    base_url = "http://unused"
    token = "offline-fixture"

    def __init__(self, **kwargs):
        Path(kwargs["output"]).touch()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


def save(path, value):
    path.write_text(json.dumps(value, indent=2))


def worker(folder, action):
    """Every process reconstructs state from disk, with no shared in-memory learner."""
    log = folder / (action + ".log")
    with log.open("w") as output:
        os.dup2(output.fileno(), 1)
        os.dup2(output.fileno(), 2)
        try:
            asyncio.run(run_worker(folder, action))
        except BaseException:
            traceback.print_exc()
            sys.stderr.flush()
            raise


async def run_worker(folder, action):
    if action == "reuse":
        state = GoalStore(folder).read()
        delivery = state["progress"]["deliveries"][0]
        value = await run_capability(delivery["candidate_id"], {"x": "3"}, library=folder / "private_capabilities")
        save(folder / "reuse-result.json", value)
        return
    # Deliberately no external model calls. Physical request accounting is
    # separately exercised by the HTTP gateway and cancellation tests.
    ar.EvidenceGateway = LocalRelay
    ar.audit_requests = lambda path: {"passed": True, "model_calls": 4}
    store = GoalStore(folder)

    async def consult(role, context):
        if role == "PI":
            if action == "resume":
                development = context["model_development"]
                baseline = json.loads((folder / "first-measured.json").read_text())["experiment"]
                assert development["prepared_dataset"]["status"] == "verified"
                assert development["prior_generations"]["count"] == 1
                assert development["next_action"]["stage"] == "evaluate_revision"
                assert development["next_action"]["eligible_parent_ids"] == [baseline["id"]]
                save(folder / "recovered-context.json", development)
            return "Use the measured baseline and fit the synthetic quadratic relation."
        if role == "CB":
            return json.dumps(
                {
                    "summary": "Synthetic process-recovery test",
                    "deliverable": "model",
                    "needs_input": [],
                    "steps": [
                        {
                            "id": "fit",
                            "title": "Fit and measure a predictive model",
                            "acceptance": "Measured input-dependent predictions on fixed synthetic partitions",
                            "approach": "Fit a quadratic response and compare with the measured constant control",
                            "verification": "model",
                        }
                    ],
                }
            )
        if role == "SC":
            assert action == "resume"
            assert "PI" not in context and "progress" not in context
            return json.dumps(accepted(context))
        assert role == "REVIEW" and action == "resume"
        return json.dumps(
            {
                "decision": "approve",
                "summary": "Synthetic recovery fixture; no biological claim",
                "checks": {
                    key: {"passed": True, "reason": "Host verified execution and synthetic scope"}
                    for key in context["required_checks"]
                },
                "issues": [],
                "tests": [
                    {"inputs": {"x": "3"}, "invariants": {"finite_prediction": True, "min": 8.9, "max": 9.1}},
                    {"inputs": {"x": "not-a-number"}, "reject": True},
                ],
            }
        )

    async def execute(request, service, events):
        if action == "first":
            prepared = await service.prepare_research_dataset(
                data_file="inputs/observations.csv",
                target_column="target",
                input_columns=["x"],
                group_column="group",
                id_column="id",
                split_column="split",
                provenance="Synthetic software fixture",
                grouping="Disjoint observations",
            )
            await service.write_research_file("baseline.py", BASELINE)
            baseline = await service.train_research_model(
                "Baseline", "baseline.py", prepared["evaluation_file"], "Fit a training-only constant control"
            )
            save(folder / "first-measured.json", baseline)
            # The parent kills only this test process after the actual measured
            # generation is saved, before the episode/SC checkpoint can finish.
            await asyncio.Event().wait()
        baseline = json.loads((folder / "first-measured.json").read_text())
        await service.write_research_file("learner.py", LEARNER)
        child = await service.train_research_model(
            "Quadratic candidate",
            "learner.py",
            "",
            "Compare an input-dependent fit with the retained baseline",
            parent_experiment_id=baseline["experiment"]["id"],
        )
        training, experiment = child["training"], child["experiment"]
        await service.write_research_file(
            "README.md", "Synthetic worker-recovery test; no biological or novelty claim."
        )
        spec = review_spec(experiment["id"])
        spec.update(
            evaluation_file=training["evaluation_file"],
            reproduction_script=training["reproduction_script"],
            fitted_files=training["fitted_files"],
            evidence_files=training["evidence_files"] + ["README.md"],
            dependencies=list(training["dependency_versions"]),
        )
        spec["input_schema"]["properties"]["x"] = {"type": "string"}
        example = {"x": "2"}
        expected = await invoke_bundle(
            service.workspace,
            {"source": training["source"], "files": experiment["files"]},
            example,
            service.control / "example",
            runner=service.runner,
        )
        spec["cases"] = [{"inputs": example, "expected": expected}, {"inputs": {}, "reject": True}]
        await service.propose_research_capability(
            "Recovery fixture",
            "Synthetic quadratic response only",
            training["source"],
            list(experiment["files"]),
            [example],
            spec,
        )
        await service.submit_research_checkpoint(
            "Measured resumed candidate",
            ["fit"],
            ["README.md", "learner.py", "experiments/" + experiment["id"] + ".json"],
            [],
        )
        events.extend(
            [
                {"kind": "tool_call_start"},
                {"kind": "completed", "text": "Measured candidate", "event_audit": {"passed": True}},
            ]
        )

    result = await ar.AutonomousResearch(
        store, "offline-fixture", {}, consult=consult, executor=execute, tools=[]
    ).run()
    save(folder / (action + "-result.json"), {"status": result["status"], "usage": result["usage"]})


def wait_for_file(path, process, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        if not process.is_alive():
            raise AssertionError(
                "Test worker ended: "
                + str(process.exitcode)
                + "\n"
                + "\n".join(p.read_text()[-3000:] for p in path.parent.glob("*.log"))
            )
        time.sleep(0.05)
    raise AssertionError("Test worker did not reach its saved experiment")


@pytest.mark.skipif(sys.platform != "linux", reason="Isolated fitting and process ownership run on Linux")
def test_real_scientist_recovers_after_measured_fit_without_retraining_baseline(tmp_path, monkeypatch):
    access = configure_access(tmp_path, monkeypatch)
    source_session = "scientist-fixture"
    store = ar.create(
        access.root / hashlib.sha256(source_session.encode()).hexdigest() / ("c" * 32),
        profile(),
        "Develop a quadratic predictor for synthetic recovery testing",
        {"max_cycles": 4, "model_calls": 48, "tool_calls": 48, "wall_seconds": 480},
    )
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["id", "group", "split", "x", "target"])
    writer.writeheader()
    for split, start, count in [("train", 1, 8), ("validation", 9, 3), ("test", 12, 3)]:
        for i in range(start, start + count):
            writer.writerow(
                {"id": f"{split}-{i}", "group": f"{split}-{i}", "split": split, "x": i / 10, "target": (i / 10) ** 2}
            )
    raw = buffer.getvalue().encode()
    store_for_goal(store.root).ingest("inputs/observations.csv", raw, hashlib.sha256(raw).hexdigest())
    processes = []
    context = multiprocessing.get_context("fork")

    def start(action):
        process = context.Process(target=worker, args=(store.root, action))
        process.start()
        processes.append(process)
        return process

    try:
        first = start("first")
        wait_for_file(store.root / "first-measured.json", first)
        reserved = store.read()["usage"].copy()
        baseline = json.loads((store.root / "first-measured.json").read_text())["experiment"]
        snapshot = {
            str(p.relative_to(store.root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (store.root / "development/experiments" / baseline["id"]).rglob("*")
            if p.is_file()
        }
        first.kill()
        first.join(5)
        assert not first.is_alive()
        service = ScienceGoalService(ROOT, state_root=tmp_path / "api-state")
        interrupted = service.reconcile(store.root)
        assert interrupted["status"] == "interrupted" and interrupted["inflight"] is None
        assert interrupted["usage"] == reserved and len(interrupted["cycles"]) == 1
        assert not interrupted["progress"].get("deliveries")
        assert not list((store.root / "private_capabilities").glob("index/*"))
        store.resume()
        resumed = start("resume")
        wait_for_file(store.root / "resume-result.json", resumed, timeout=90)
        resumed.join(5)
        assert resumed.exitcode == 0
        final = store.read()
        assert final["status"] == "completed" and len(final["cycles"]) == 2
        assert final["usage"]["model_calls"] >= reserved["model_calls"]
        records = Development(store.root / "development").records()
        assert len(records) == 2 and [r["variant"] for r in records] == ["baseline", "candidate"]
        assert records[1]["parent"] == baseline["id"]
        assert len(list((store.root / "development/training").glob("*/receipt.json"))) == 2
        for name, digest in snapshot.items():
            assert hashlib.sha256((store.root / name).read_bytes()).hexdigest() == digest
        delivery = final["progress"]["deliveries"][0]
        assert sc.verify_delivery(store.root, delivery)
        with zipfile.ZipFile(store.root / "workspace" / delivery["package"]) as archive:
            assert "evaluation/scientific-acceptance.json" in archive.namelist()
        fresh = start("reuse")
        wait_for_file(store.root / "reuse-result.json", fresh)
        fresh.join(5)
        assert fresh.exitcode == 0
        assert json.loads((store.root / "reuse-result.json").read_text())["value"] == pytest.approx(9)
        asyncio.run(verify_private_reuse(access, store, source_session, monkeypatch))
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
            process.join(5)
