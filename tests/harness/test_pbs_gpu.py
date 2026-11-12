"""GPU policy and fit/review routing; fake-device tests are software fixtures."""

import copy
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.harness import pbs_gpu
from agent.harness.capability_review import model_evaluation
from agent.harness.contracts import file_hash
from agent.harness.research_training import selected_training_compute
from agent.harness.research_workspace import ResearchWorkspace, run_python
from agent.harness.sandbox_resources import Limits
from agent.harness.session_sandbox import SandboxUnavailable, command

UUID = "GPU-11111111-2222-3333-4444-555555555555"
JOB = "123.fs"
GIB = 1024**3
LIMITS = Limits(8 * GIB, 2, 16 * GIB, 4 * GIB, GIB, 30)


@pytest.fixture
def allocation(monkeypatch):
    for key in ("WEBUI_V2_MODE", "VENUS_SCIENTIST_ONLINE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("VENUS_RESEARCH_GPU_BACKEND", "pbs")
    monkeypatch.setenv("PBS_JOBID", JOB)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", UUID)
    relative = "/pbs_jobs.service/jobid/" + JOB
    files = {
        "/proc/self/cgroup": f"1:memory:{relative}\n2:devices:{relative}\n",
        "/proc/driver/nvidia/gpus/one/information": f"GPU UUID: {UUID}\nDevice Minor: 3\n",
    }
    for controller in ("memory", "devices"):
        files[f"/sys/fs/cgroup/{controller}{relative}/cgroup.procs"] = str(os.getpid())
    files[f"/sys/fs/cgroup/memory{relative}/memory.limit_in_bytes"] = str(16 * GIB)
    device_file = f"/sys/fs/cgroup/devices{relative}/devices.list"
    files[device_file] = "c 195:3 rwm\nc 195:255 rwm\nc 511:0 rwm\n"

    class FakePath:
        def __init__(self, name):
            self.name = str(name)

        def __truediv__(self, child):
            return FakePath(self.name + "/" + child)

        def read_text(self):
            return files[self.name]

        def glob(self, pattern):
            return [FakePath("/proc/driver/nvidia/gpus/one/information")]

        def stat(self):
            major, minor = {"/dev/nvidia3": (195, 3), "/dev/nvidiactl": (195, 255), "/dev/nvidia-uvm": (511, 0)}[
                self.name
            ]
            return SimpleNamespace(st_mode=stat.S_IFCHR, st_rdev=os.makedev(major, minor))

    monkeypatch.setattr(pbs_gpu, "Path", FakePath)
    opened = []
    monkeypatch.setattr(pbs_gpu.os, "open", lambda path, flags: opened.append(path) or 71)
    monkeypatch.setattr(pbs_gpu.os, "close", lambda descriptor: None)
    return files, opened, device_file


def test_uuid_uses_physical_minor_not_visible_ordinal(allocation):
    _, opened, _ = allocation
    result = pbs_gpu.capture(LIMITS)
    assert result.device == "/dev/nvidia3"
    assert result.memory_limit == 16 * GIB
    assert opened == list(result.devices)
    assert result.receipt()["memory_scope"] == "whole PBS allocation"


@pytest.mark.parametrize(
    "damage",
    [
        "disabled",
        "online",
        "many_gpus",
        "bad_job",
        "no_cgroup",
        "unbounded_memory",
        "small_memory",
        "budget",
        "wrong_member",
        "wildcard_device",
        "wrong_device",
        "unknown_uuid",
    ],
)
def test_unverified_allocation_fails_closed(allocation, monkeypatch, damage):
    files, opened, device_file = allocation
    if damage == "disabled":
        monkeypatch.delenv("VENUS_RESEARCH_GPU_BACKEND")
    elif damage == "online":
        monkeypatch.setenv("WEBUI_V2_MODE", "online")
    elif damage == "many_gpus":
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", UUID + "," + UUID)
    elif damage == "bad_job":
        monkeypatch.setenv("PBS_JOBID", "../../123.fs")
    elif damage == "no_cgroup":
        files["/proc/self/cgroup"] = "0::/user.slice\n"
    elif damage in {"unbounded_memory", "small_memory", "budget"}:
        key = next(k for k in files if k.endswith("memory.limit_in_bytes"))
        files[key] = str({"unbounded_memory": 2**63 - 4096, "small_memory": 8 * GIB, "budget": 32 * GIB}[damage])
    elif damage == "wrong_member":
        for key in list(files):
            if key.endswith("cgroup.procs"):
                files[key] = "-1"
    elif damage == "wildcard_device":
        files[device_file] += "c 195:* rwm\n"
    elif damage == "wrong_device":
        files[device_file] = files[device_file].replace("195:3", "195:2")
    else:
        files["/proc/driver/nvidia/gpus/one/information"] = "GPU UUID: other\nDevice Minor: 3\n"
    with pytest.raises(SandboxUnavailable):
        pbs_gpu.capture(LIMITS)
    assert not opened


def test_cpu_mount_policy_unchanged_gpu_only_mounts_allocated_devices(tmp_path):
    cpu = command(tmp_path, "python", "model.py", 60, scratch=tmp_path / "scratch", memory=8 * GIB)
    gpu = command(
        tmp_path,
        "python",
        "model.py",
        60,
        scratch=tmp_path / "scratch",
        memory=8 * GIB,
        gpu=pbs_gpu.Allocation(JOB, UUID, "/dev/nvidia3", 16 * GIB),
    )

    def pairs(value):
        return list(zip(value, value[1:]))

    assert ("--remount-ro", "/proc") in pairs(cpu)
    assert "VENUS_SANDBOX_MEMORY_POLICY" not in cpu and "--dev-bind" not in cpu
    assert ("--remount-ro", "/proc") not in pairs(gpu)
    assert [gpu[i + 1] for i, x in enumerate(gpu) if x == "--dev-bind"] == [
        "/dev/nvidia3",
        "/dev/nvidiactl",
        "/dev/nvidia-uvm",
    ]
    assert "--unshare-all" in gpu and "--cap-drop" in gpu and "--clearenv" in gpu
    assert ("--remount-ro", "/dev") in pairs(gpu)
    assert "/sandbox_seccomp.py" in gpu
    if Path("/proc/sys").exists():
        assert ("--ro-bind", "/proc/sys") in pairs(gpu)


@pytest.mark.asyncio
async def test_gpu_fit_cpu_validation_and_gpu_refit_are_distinct(tmp_path, monkeypatch):
    calls = []
    compute = pbs_gpu.Allocation(JOB, UUID, "/dev/nvidia3", 16 * GIB).receipt()

    async def gpu_runner(workspace, control, script, timeout=120, **kwargs):
        calls.append(script)
        result = await run_python(workspace, control, script, timeout, **kwargs)
        result["gpu"] = compute
        return result

    monkeypatch.setattr(pbs_gpu, "run_python", gpu_runner)
    service = ResearchWorkspace(tmp_path / "work", tmp_path / "control", {"steps": []})

    def rows(start):
        return [{"id": str(i), "group": str(i), "inputs": {"x": i}, "target": 2 * i} for i in range(start, start + 8)]

    data = {
        "task": "regression",
        "provenance": "Synthetic software fixture",
        "grouping": "Independent fixture rows",
        "splits": {"train": rows(1), "validation": rows(9), "test": rows(17)},
    }
    await service.write_research_file("data.json", json.dumps(data))
    baseline = "def fit(training,seed):\n return {'mean':sum(r['target'] for r in training['train'])/len(training['train'])}\ndef predict(model,inputs):\n return {'value':model['mean']}\n"
    await service.write_research_file("model.py", baseline)
    parent = await service.train_research_model("mean", "model.py", "data.json", "Reference")
    revision = "def fit(training,seed):\n rows=training['train']\n return {'slope':sum(r['target']/r['inputs']['x'] for r in rows)/len(rows)}\ndef predict(model,inputs):\n return {'value':model['slope']*inputs['x']}\n"
    await service.write_research_file("model.py", revision)
    child = await service.train_research_model(
        "linear", "model.py", "", "Fit a slope", parent_experiment_id=parent["experiment"]["id"], device="cuda"
    )
    assert calls == ["train.py"]  # held-out replay must use the original CPU runner
    training = child["training"]
    experiment = child["experiment"]
    spec = {
        "input_schema": {"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]},
        "output_schema": {"type": "object", "properties": {"value": {"type": "number"}}, "required": ["value"]},
        "evaluation_file": training["evaluation_file"],
        "prediction_key": "value",
        "reproduction_script": training["reproduction_script"],
        "fitted_files": training["fitted_files"],
    }
    manifest = {
        "source": training["source"],
        "files": experiment["files"],
        "review_spec": spec,
        "evidence_files": {name: file_hash(service.workspace / name) for name in training["evidence_files"]},
    }
    manifest["development"] = service.development.selected(experiment["id"], manifest, data)
    assert manifest["development"]["training_compute"]["training_receipt_sha256"] == training["receipt_sha256"]
    assessment = await model_evaluation(service.workspace, manifest, tmp_path / "review", run_python)
    assert calls == ["train.py", spec["reproduction_script"]]
    assert assessment["score"] == 0 and assessment["training_reproduced"]
    assert assessment["reproduction_compute"] == compute
    changed = copy.deepcopy(manifest)
    changed["review_spec"]["reproduction_script"] = "other.py"
    with pytest.raises(ValueError, match="entrypoint"):
        selected_training_compute(service.development, experiment["id"], changed)
    path = service.development.root / "training" / training["id"] / "receipt.json"
    saved = json.loads(path.read_text())
    saved["execution"]["gpu"]["gpu_uuid"] = "forged"
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="receipt changed"):
        selected_training_compute(service.development, experiment["id"], manifest)


def test_device_option_is_in_progressive_training_schema(tmp_path):
    service = ResearchWorkspace(tmp_path / "work", tmp_path / "control", {"steps": []})
    tool = next(row for row in service.tools() if row.name == "train_research_model")
    schema = tool.args_schema.model_json_schema()["properties"]["device"]
    assert schema["enum"] == ["cpu", "cuda"] and schema["default"] == "cpu"
    assert "PBS" in schema["description"] and "CPU-portable" in schema["description"]


@pytest.mark.asyncio
async def test_gpu_resident_watchdog_reaps_process_and_releases_reservation(tmp_path, monkeypatch):
    # CPU executable with a fake allocation exercises the real GPU watchdog and
    # cleanup without allocating memory or requiring a GPU in the test runner.
    from agent.harness import sandbox_resources as resources
    from agent.harness import session_sandbox as sandbox
    from backend.web_v2.job_queue import process_token

    monkeypatch.setenv("VENUS_SANDBOX_CONTROL_ROOT", str(tmp_path / "admission"))
    monkeypatch.setenv("VENUS_SESSION_SANDBOX", "bwrap")
    monkeypatch.setenv("VENUS_SANDBOX_LOCAL_MEMORY_MB", "512")
    monkeypatch.setenv("VENUS_SANDBOX_TOTAL_MEMORY_MB", "1024")
    monkeypatch.setenv("WEBUI_V2_MODE", "local")
    monkeypatch.delenv("VENUS_SCIENTIST_ONLINE", raising=False)
    allocation = pbs_gpu.Allocation(JOB, UUID, "/dev/nvidia3", 16 * GIB)
    monkeypatch.setattr(pbs_gpu, "capture", lambda limits: allocation)
    original = sandbox.command

    def cpu_command(*args, gpu=None, **kwargs):
        return original(*args, **kwargs)

    monkeypatch.setattr(sandbox, "command", cpu_command)
    observed = []

    def over_limit(pid):
        observed.append((pid, process_token(pid)))
        return 1024 * GIB

    monkeypatch.setattr(pbs_gpu, "process_rss", over_limit)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "run.py").write_text("import time\ntime.sleep(20)")
    with pytest.raises(MemoryError, match="resident-memory limit"):
        await sandbox.execute(workspace, tmp_path / "control", "python", "run.py", 10, device="cuda")
    assert observed and all(process_token(pid) != token for pid, token in observed)
    assert not list((tmp_path / "control").glob("sandbox-*"))
    receipt = json.loads(next((tmp_path / "control").glob("*.json")).read_text())
    assert receipt["status"] == "failed" and receipt["temporary_files_removed"]
    assert receipt["memory_policy"] == "resident-watchdog-and-pbs-cgroup"
    with resources.Reservations().locked() as rows:
        assert not rows


@pytest.mark.parametrize("rule", ["c 195:* m", "c *:* m"])
def test_pbs_node_creation_permission_is_not_gpu_read_write_access(allocation, rule):
    files, _, device_file = allocation
    files[device_file] += rule + "\n"
    assert pbs_gpu.capture(LIMITS).device == "/dev/nvidia3"


@pytest.mark.parametrize("permission", ["r", "w"])
def test_broad_read_or_write_is_still_rejected(allocation, permission):
    files, opened, device_file = allocation
    files[device_file] += "c 195:* " + permission + "\n"
    with pytest.raises(SandboxUnavailable, match="explicit CUDA-device permissions"):
        pbs_gpu.capture(LIMITS)
    assert not opened


@pytest.mark.asyncio
@pytest.mark.parametrize("available", [False, True])
async def test_strategy_guide_discloses_only_verified_training_devices(tmp_path, monkeypatch, available):
    def capture(limits):
        if not available:
            raise SandboxUnavailable("unavailable")
        return pbs_gpu.Allocation(JOB, UUID, "/dev/nvidia3", 16 * GIB)

    monkeypatch.setattr(pbs_gpu, "capture", capture)
    service = ResearchWorkspace(tmp_path / "work", tmp_path / "control", {"steps": []})
    compute = (await service.get_research_strategy_guide())["compute"]
    assert compute["training_devices"] == (["cpu", "cuda"] if available else ["cpu"])
    assert compute["prediction_device"] == "cpu"
    assert UUID not in json.dumps(compute) and JOB not in json.dumps(compute)
