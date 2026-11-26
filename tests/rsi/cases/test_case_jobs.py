import asyncio
import json
import sys

import pytest

from agent.rsi import case_jobs


def setup_broker(tmp_path, monkeypatch, valid=True):
    root = tmp_path / "repo"
    (root / "research_runs/rsi").mkdir(parents=True)
    script = root / "fixture.py"
    body = (
        {
            "status": "completed",
            "comparisons": [
                {
                    "indexing": "author_notebook",
                    "checkpoint": "PF01083_19.pt",
                    "independent_author_function_max_abs_error": 0.0,
                }
            ],
            "measurement_rows": 80,
            "claim_scope": "fixture",
        }
        if valid
        else {"status": "running"}
    )
    script.write_text(
        "import pathlib,sys,json\np=pathlib.Path(sys.argv[1]);p.mkdir()\n(p/'report.json').write_text("
        + repr(json.dumps(body))
        + ")\n"
    )

    def recipe(plugin_id, parameters, **kwargs):
        return [sys.executable, "-u", str(script), str(kwargs["output"])]

    monkeypatch.setattr(case_jobs, "invocation", recipe)
    return case_jobs.CaseJobBroker(
        root=root,
        workspace=root / "workspace",
        output=root / "research_runs/rsi/jobs",
        allowed_plugins={"turbopetase.v1"},
        timeout_s=10,
    )


@pytest.mark.asyncio
async def test_jobs_are_idempotent_bounded_and_require_domain_evidence(tmp_path, monkeypatch):
    broker = setup_broker(tmp_path, monkeypatch)
    async with broker:
        args = {"models": ["PF01083"], "device": "cpu"}
        first = broker.submit("turbopetase.v1", args, "one")
        assert broker.submit("turbopetase.v1", args, "one")["job_id"] == first["job_id"]
        with pytest.raises(ValueError, match="different request"):
            broker.submit("turbopetase.v1", {"models": ["PTM311"]}, "one")
        with pytest.raises(ValueError, match="budget"):
            broker.submit("turbopetase.v1", args, "two")
        with pytest.raises(ValueError, match="outside campaign"):
            broker.submit("rfantibody.v2", {}, "three")
        await broker.wait_jobs()
        result = broker.status(first["job_id"])
        assert result["status"] == "completed"
        assert result["evidence"]["author_inference_replayed"]
        assert "argv" not in result and "output" not in result
    assert json.loads((broker.output / "jobs.json").read_text())["jobs"][first["job_id"]]["report_sha256"]


@pytest.mark.asyncio
async def test_zero_exit_with_incomplete_report_is_failed(tmp_path, monkeypatch):
    broker = setup_broker(tmp_path, monkeypatch, valid=False)
    async with broker:
        job = broker.submit("turbopetase.v1", {"models": ["PF01083"]}, "one")
        await broker.wait_jobs()
        assert broker.status(job["job_id"])["status"] == "failed"


@pytest.mark.asyncio
async def test_rpc_catalog_and_invalid_request_are_audited(tmp_path, monkeypatch):
    broker = setup_broker(tmp_path, monkeypatch)
    async with broker:
        rid = "a" * 32
        (broker.rpc / (rid + ".request.json")).write_text(json.dumps({"action": "submit", "command": "arbitrary"}))
        destination = broker.rpc / (rid + ".response.json")
        async with asyncio.timeout(3):
            while not destination.exists():
                await asyncio.sleep(0.01)
        assert json.loads(destination.read_text())["success"] is False
        assert not broker.jobs


@pytest.mark.asyncio
async def test_cpu_case_is_not_queued_behind_gpu_lock(tmp_path, monkeypatch):
    import fcntl

    broker = setup_broker(tmp_path, monkeypatch)
    with (broker.root / "research_runs/rsi/.case_gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        async with broker:
            job = broker.submit("turbopetase.v1", {"models": ["PF01083"], "device": "cpu"}, "cpu")
            await asyncio.wait_for(broker.wait_jobs(), 3)
            assert broker.status(job["job_id"])["status"] == "completed"
            assert broker.status(job["job_id"])["resource"] == "cpu"


@pytest.mark.asyncio
async def test_allocation_binding_rejects_agent_parameter_drift_before_launch(tmp_path, monkeypatch):
    broker = setup_broker(tmp_path, monkeypatch)
    broker.expected_parameters = {"turbopetase.v1": {"models": ["PF01083"], "device": "cpu"}}
    async with broker:
        with pytest.raises(ValueError, match="predeclared"):
            broker.submit("turbopetase.v1", {"models": ["PTM311"], "device": "cpu"}, "changed")
        assert not broker.jobs and not broker.tasks


@pytest.mark.asyncio
async def test_antibody_large_allocation_requires_explicit_campaign_limit(tmp_path, monkeypatch):
    broker = setup_broker(tmp_path, monkeypatch)
    broker.allowed_plugins = frozenset({"rfantibody.targets.v1"})
    parameters = {"target": "ha", "backbones": 64, "sequences": 4}
    async with broker:
        with pytest.raises(ValueError, match="design budget"):
            broker.submit("rfantibody.targets.v1", parameters, "too-large")
        assert not broker.jobs
        broker.maximum_backbones = 64
        broker.expected_parameters = {"rfantibody.targets.v1": parameters}
        with pytest.raises(ValueError, match="predeclared"):
            broker.submit("rfantibody.targets.v1", {**parameters, "target": "rsv_site3"}, "wrong-target")
        job = broker.submit("rfantibody.targets.v1", parameters, "declared")
        assert job["parameters"] == parameters and broker.catalog()["maximum_backbones"] == 64
        assert broker.submit("rfantibody.targets.v1", parameters, "declared")["job_id"] == job["job_id"]
        await broker.wait_jobs()
        # Fixture emits PET evidence: a zero-exit wrong-domain report must fail.
        assert broker.status(job["job_id"])["status"] == "failed"


@pytest.mark.asyncio
async def test_version_qualified_idempotency_key_is_preserved_without_truncation(tmp_path, monkeypatch):
    broker = setup_broker(tmp_path, monkeypatch)
    broker.max_jobs = 2
    key = "catapro_training_v1_reproduce_" + "a" * 64
    parameters = {"models": ["PF01083"], "device": "cpu"}
    async with broker:
        first = broker.submit("turbopetase.v1", parameters, key)
        assert broker.submit("turbopetase.v1", parameters, key)["job_id"] == first["job_id"]
        second = broker.submit("turbopetase.v1", parameters, key[:-1] + "b")
        assert first["job_id"] != second["job_id"]
        with pytest.raises(ValueError, match="different request"):
            broker.submit("turbopetase.v1", {"models": ["PTM311"], "device": "cpu"}, key)
        await broker.wait_jobs()
        assert all(broker.status(job["job_id"])["status"] == "completed" for job in [first, second])


@pytest.mark.parametrize("key", ["", "a" * 257, "contains space", "contains/slash", "non_ascii_甲"])
def test_invalid_job_key_explains_its_contract_before_allocating(tmp_path, monkeypatch, key):
    broker = setup_broker(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="1..256 ASCII"):
        broker.submit("turbopetase.v1", {}, key)
    assert not broker.jobs and not broker.tasks
