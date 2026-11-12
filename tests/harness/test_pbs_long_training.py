"""Bounded long GPU fitting policy; actual GPU execution has a separate PBS test."""

import json
from contextlib import asynccontextmanager

import pytest

from agent.harness import pbs_gpu, session_sandbox
from agent.harness.research_training import validate_training_timeout
from agent.harness.research_workspace import ResearchWorkspace
from agent.harness.sandbox_resources import Limits


@pytest.mark.parametrize("device,value", [("cpu", 121), ("cuda", 1801), ("cuda", 0), ("cuda", True), ("cuda", 120.0)])
@pytest.mark.asyncio
async def test_bad_fit_allowance_fails_before_data_or_execution(tmp_path, device, value):
    async def forbidden(*args, **kwargs):
        raise AssertionError("Invalid allowance reached execution")

    service = ResearchWorkspace(tmp_path / "work", tmp_path / "control", {"steps": []}, runner=forbidden)
    with pytest.raises(ValueError, match="timeout_seconds"):
        await service.train_research_model(
            "fixture", "missing.py", "missing.json", "fixture", device=device, timeout_seconds=value
        )
    assert not list((service.development.root / "training").glob("*/receipt.json"))


def test_training_and_generic_cpu_limits_remain_distinct():
    assert validate_training_timeout(120, "cpu") == 120
    assert validate_training_timeout(1800, "cuda") == 1800
    session_sandbox.validate_execution("python", "fit.py", 300)
    with pytest.raises(ValueError, match="1..300"):
        session_sandbox.validate_execution("python", "fit.py", 301)
    with pytest.raises(ValueError, match="Python only"):
        session_sandbox.validate_execution("shell", "echo test", 1800, gpu=True)


def test_long_gpu_limit_reaches_isolated_worker_command(tmp_path):
    gpu = pbs_gpu.Allocation("123.fs", "GPU-11111111-2222-3333-4444-555555555555", "/dev/nvidia3", 16 * 1024**3)
    argv = session_sandbox.command(
        tmp_path, "python", "fit.py", 1800, scratch=tmp_path / "scratch", memory=8 * 1024**3, gpu=gpu
    )
    assert argv[-3:] == ["python", "1800", "fit.py"]
    assert "--clearenv" in argv and "--unshare-all" in argv
    assert "--cap-drop" in argv and "--dev-bind" in argv


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,backend", [("local", "bwrap"), ("online", "bwrap"), ("local", "e2b")])
async def test_long_gpu_fit_never_spawns_without_verified_local_allocation(tmp_path, monkeypatch, mode, backend):
    monkeypatch.setenv("WEBUI_V2_MODE", mode)
    monkeypatch.delenv("VENUS_SCIENTIST_ONLINE", raising=False)
    monkeypatch.setenv("VENUS_RESEARCH_GPU_BACKEND", "pbs")
    monkeypatch.delenv("PBS_JOBID", raising=False)
    monkeypatch.delenv("VENUS_QUEUE_JOB_ID", raising=False)
    monkeypatch.setattr(session_sandbox, "backend", lambda: backend)

    @asynccontextmanager
    async def admitted(*args, **kwargs):
        yield None, "fixture", Limits.load()

    async def forbidden(*args, **kwargs):
        raise AssertionError("Unverified GPU execution spawned")

    monkeypatch.setattr(session_sandbox, "admission", admitted)
    monkeypatch.setattr(session_sandbox, "_execute_bwrap", forbidden)
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(session_sandbox.SandboxUnavailable):
        await session_sandbox.execute(work, tmp_path / "control", "python", "fit.py", 1800, device="cuda")


@pytest.mark.asyncio
async def test_progressive_training_contract_exposes_allowance_and_hash_requirement(tmp_path, monkeypatch):
    from agent.harness.session_boundary import mcp_input_schema
    from agent.harness.tool_disclosure import ToolDisclosure

    service = ResearchWorkspace(tmp_path / "work", tmp_path / "control", {"steps": []})
    tools = {tool.name: tool for tool in service.tools()}
    training = mcp_input_schema(tools["train_research_model"])
    allowance = training["properties"]["timeout_seconds"]
    assert allowance["default"] == 120 and allowance["maximum"] == 1800
    assert "CPU" in allowance["description"] and "Review" in allowance["description"]
    surface = ToolDisclosure(
        tools, schema=mcp_input_schema, direct={"edit_session_file"}, allowed=lambda tool: True
    ).surface()
    edit = next(row for row in surface if row["name"] == "edit_session_file")
    assert "Read" in edit["description"] and "expected_sha256" in edit["description"]
    assert "expected_sha256" in edit["inputSchema"]["required"]
    assert "full-file" in edit["inputSchema"]["properties"]["expected_sha256"]["description"]
    monkeypatch.setattr(session_sandbox, "backend", lambda: "bwrap")
    monkeypatch.setattr(pbs_gpu, "capture", lambda limits: object())
    compute = (await service.get_research_strategy_guide())["compute"]
    assert compute["max_training_seconds"] == {"cpu": 120, "cuda": 1800}
    assert "GPU-" not in json.dumps(compute)


@pytest.mark.parametrize(
    "binding_timeout,record_timeout,execution_timeout,valid",
    [
        (None, None, None, True),
        (1800, 1800, 1800, True),
        (1800, None, None, False),
        (1800, 1800, 120, False),
        (None, 1800, 1800, False),
        (True, True, True, False),
    ],
)
def test_timeout_is_bound_to_host_experiment_and_execution(
    tmp_path, binding_timeout, record_timeout, execution_timeout, valid
):
    from types import SimpleNamespace

    from agent.harness.contracts import digest
    from agent.harness.research_training import selected_training_compute

    attempt = "a" * 32
    binding = {"attempt_id": attempt, "device": "cuda"}
    if binding_timeout is not None:
        binding["timeout_seconds"] = binding_timeout
    record = {
        "status": "completed",
        "device": "cuda",
        "experiment_id": "measured",
        "source": "runtime.py",
        "runtime_files": ["runtime.py", "model.json"],
        "reproduction_script": "train.py",
        "fitted_files": ["model.json"],
        "execution": {"exit_code": 0, "gpu": {"backend": "pbs", "device": "cuda", "inference_device": "cpu"}},
    }
    if record_timeout is not None:
        record["timeout_seconds"] = record_timeout
    if execution_timeout is not None:
        record["execution"]["timeout_seconds"] = execution_timeout
    record["receipt_sha256"] = digest(record)
    folder = tmp_path / "training" / attempt
    folder.mkdir(parents=True)
    (folder / "receipt.json").write_text(json.dumps(record))
    development = SimpleNamespace(root=tmp_path, record=lambda identity: {"training": binding})
    manifest = {
        "source": "runtime.py",
        "files": {"runtime.py": "digest", "model.json": "digest"},
        "review_spec": {"reproduction_script": "train.py", "fitted_files": ["model.json"]},
    }
    if valid:
        assert selected_training_compute(development, "measured", manifest)["training_timeout_seconds"] == (
            record_timeout or 120
        )
    else:
        with pytest.raises(ValueError, match="timeout"):
            selected_training_compute(development, "measured", manifest)
