import json
from pathlib import Path

import pytest

from agent.rsi import execution
from agent.rsi.model_policy import ModelBindingError, read_model_configuration
from agent.rsi.pilot import build_tasks
from agent.rsi.versions import VersionStore


def prepare(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "executor_prompt.md").write_text("candidate-specific sentinel policy")
    (source / "pilot_tools.py").write_text("# public tool fixture")
    store = VersionStore(tmp_path / "store")
    version = store.snapshot(source)
    capdir = store.root / "versions" / version["version_id"]
    config_path = tmp_path / "runtime.toml"
    config_path.write_text(
        'default_model = "dmx/glm-4-flash"\n[models."dmx/glm-4-flash"]\nprovider = "dmx"\nmodel = "glm-4-flash"\n[providers.dmx]\ntype = "openai"\nbase_url = "https://www.dmxapi.cn/v1"\napi_key = "private-sentinel"\n'
    )
    _, identity = read_model_configuration(config_path)
    config = {
        "kimi_config": str(config_path),
        "kimi_bin": "/unused/kimi",
        "python_root": "/unused/python",
        "task_timeout_s": 10,
        "max_tool_calls": 12,
        "required_model": {"model": "glm-4-flash"},
    }
    return capdir, version, config, {"model": identity}


class FakeSandbox:
    stopped = True
    client = object()
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls.append(kwargs)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


def evidence(run_dir, task, *, successful_result=True):
    (run_dir / "workspace/output.json").write_text(json.dumps(task["expected"]))
    events = [
        {
            "kind": "tool_call_start",
            "tool_call_id": "one",
            "tool_name": "mcp__venusfactory__convert_temperature",
            "args": {"value": 310.15, "source_unit": "K", "target_unit": "C"},
        }
    ]
    if successful_result:
        events.append({"kind": "tool_result", "tool_call_id": "one", "output": json.dumps(task["expected"])})
    (run_dir / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))


@pytest.mark.asyncio
@pytest.mark.parametrize("successful_result", [True, False])
async def test_actual_candidate_prompt_and_strict_evidence(tmp_path, monkeypatch, successful_result):
    capdir, version, config, baseline = prepare(tmp_path)
    task = next(t for t in build_tasks() if t["kind"] == "temperature")
    observed = {}

    async def run(*args, **kwargs):
        observed.update(kwargs)
        evidence(kwargs["output_dir"], task, successful_result=successful_result)
        return {"status": "completed"}

    monkeypatch.setattr(execution, "SandboxKimi", FakeSandbox)
    monkeypatch.setattr(execution, "run_task", run)
    run_dir = tmp_path / "run"
    _, score = await execution.execute_task(
        task, config=config, baseline=baseline, capability_dir=capdir, version_id=version["version_id"], run_dir=run_dir
    )
    assert observed["system_prompt"] == "candidate-specific sentinel policy"
    assert observed["model"] == "dmx/glm-4-flash"
    assert score["passed"] is successful_result
    assert score["legacy_passed"]  # Legacy check accepts a start without a result.
    binding_text = (run_dir / "policy_binding.json").read_text()
    assert "private-sentinel" not in binding_text
    assert json.loads(binding_text)["prompt_sha256"] == version["files"]["executor_prompt.md"]
    assert json.loads((run_dir / "evaluation.json").read_text())["audit_version"] == 2


@pytest.mark.parametrize(
    "old,new",
    [("glm-4-flash", "another-model"), ("www.dmxapi.cn", "changed.example"), ('type = "openai"', 'type = "anthropic"')],
)
def test_model_or_endpoint_drift_rejected(tmp_path, old, new):
    capdir, version, config, baseline = prepare(tmp_path)
    path = Path(config["kimi_config"])
    path.write_text(path.read_text().replace(old, new))
    with pytest.raises(ModelBindingError):
        execution.load_binding(capdir, version["version_id"], config, baseline)


def test_changed_candidate_cannot_be_loaded(tmp_path):
    capdir, version, config, baseline = prepare(tmp_path)
    prompt = capdir / "executor_prompt.md"
    prompt.chmod(0o600)
    prompt.write_text("tampered")
    with pytest.raises(ValueError, match="changed"):
        execution.load_binding(capdir, version["version_id"], config, baseline)


def test_endpoint_credentials_never_enter_public_binding(tmp_path):
    _, _, config, _ = prepare(tmp_path)
    path = Path(config["kimi_config"])
    path.write_text(
        path.read_text().replace("https://www.dmxapi.cn/v1", "https://user:private-sentinel@example.org/v1")
    )
    with pytest.raises(ModelBindingError) as caught:
        read_model_configuration(path)
    assert "private-sentinel" not in str(caught.value)


@pytest.mark.asyncio
async def test_rewritten_input_rejected_even_with_correct_artifact(tmp_path, monkeypatch):
    capdir, version, config, baseline = prepare(tmp_path)
    task = next(t for t in build_tasks() if t["kind"] == "temperature")
    task["files"] = {"input.txt": "original"}

    async def run(*args, **kwargs):
        evidence(kwargs["output_dir"], task)
        (kwargs["output_dir"] / "workspace/input.txt").write_text("rewritten")
        return {"status": "completed"}

    monkeypatch.setattr(execution, "SandboxKimi", FakeSandbox)
    monkeypatch.setattr(execution, "run_task", run)
    _, result = await execution.execute_task(
        task,
        config=config,
        baseline=baseline,
        capability_dir=capdir,
        version_id=version["version_id"],
        run_dir=tmp_path / "run",
    )
    assert not result["passed"] and result["reason"] == "task_inputs_modified"


@pytest.mark.asyncio
async def test_model_rechecked_at_sandbox_start(tmp_path, monkeypatch):
    from agent.rsi.sandbox import SandboxKimi

    capdir, _, config, baseline = prepare(tmp_path)
    path = Path(config["kimi_config"])
    path.write_text(path.read_text().replace("www.dmxapi.cn", "changed.example"))
    spawned = []

    async def forbidden_spawn(*args, **kwargs):
        spawned.append(True)
        raise AssertionError("must not start")

    monkeypatch.setattr("asyncio.create_subprocess_exec", forbidden_spawn)
    sandbox = SandboxKimi(
        workspace=tmp_path / "work",
        control_dir=tmp_path / "control",
        capability_dir=capdir,
        kimi_bin=Path("/unused/kimi"),
        config_path=path,
        python_root=Path("/unused/python"),
        expected_model=baseline["model"],
    )
    with pytest.raises(ModelBindingError):
        await sandbox.start()
    assert not spawned
    assert not (tmp_path / "control/private_home/.kimi-code/config.toml").exists()
