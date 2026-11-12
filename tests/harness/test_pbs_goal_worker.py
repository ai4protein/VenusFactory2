"""Offline PBS worker/source tests; no qsub, GPU or model provider calls."""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.harness import Profile
from agent.harness import pbs_goal_snapshot as snapshot
from agent.harness import pbs_goal_worker as worker
from agent.science_goal import GoalSpec, GoalStore

ROOT = Path(__file__).resolve().parents[2]


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.email", "fixture@example.invalid")
    git(root, "config", "user.name", "Synthetic fixture")
    git(root, "config", "commit.gpgsign", "false")
    (root / "src").mkdir()
    (root / "src/code.py").write_text("VALUE = 1\n")
    (root / "scripts").mkdir()
    (root / "scripts/worker.py").write_text("print('fixture')\n")
    (root / ".env.example").write_text("PROVIDER_API_KEY=\n")
    git(root, "add", ".")
    git(root, "commit", "-qm", "Synthetic source fixture")
    return root


def test_freeze_includes_full_tracked_source_not_deployment_files(repo, tmp_path):
    (repo / ".env").write_text("NOT_A_REAL_KEY=untracked\n")
    (repo / "untracked.py").write_text("raise RuntimeError('not source')")
    value = snapshot.freeze_source(repo, tmp_path / "frozen")
    assert set(value["files"]) == {"src/code.py", "scripts/worker.py", ".env.example"}
    assert value["commit"] == git(repo, "rev-parse", "HEAD").strip()
    assert not (Path(value["root"]) / ".env").exists()
    assert not (Path(value["root"]) / "src/code.py").stat().st_mode & 0o222
    snapshot.verify_source(json.loads(json.dumps(value)))


@pytest.mark.parametrize("kind", ["dirty", "staged", "symlink", "tracked_env"])
def test_freeze_rejects_uncommitted_or_nonexact_source(repo, tmp_path, kind):
    if kind in {"dirty", "staged"}:
        (repo / "src/code.py").write_text("VALUE = 2\n")
        if kind == "staged":
            git(repo, "add", "src/code.py")
    else:
        if kind == "symlink":
            (repo / "link.py").symlink_to("src/code.py")
        elif kind == "tracked_env":
            (repo / ".env").write_text("NOT_A_REAL_KEY=placeholder\n")
        git(repo, "add", ".")
        git(repo, "commit", "-qm", "Invalid snapshot fixture")
    with pytest.raises(ValueError):
        snapshot.freeze_source(repo, tmp_path / "frozen")
    assert not (tmp_path / "frozen").exists()
    assert not list(tmp_path.glob(".pbs-source-*"))


@pytest.mark.parametrize("change", ["modified", "missing", "extra", "symlink", "manifest", "script"])
def test_verify_detects_source_and_entrypoint_changes(repo, tmp_path, change):
    value = snapshot.freeze_source(repo, tmp_path / "frozen")
    root = Path(value["root"])
    target = root / "src/code.py"
    if change in {"modified", "script"}:
        target = root / ("scripts/worker.py" if change == "script" else "src/code.py")
        target.chmod(0o600)
        target.write_text("CHANGED = True\n")
    elif change == "missing":
        target.unlink()
    elif change == "extra":
        (root / "injected.py").write_text("pass\n")
    elif change == "symlink":
        (root / "link.py").symlink_to(target)
    else:
        value["files"]["src/code.py"] = "0" * 64
    with pytest.raises(ValueError):
        snapshot.verify_source(value)


def test_snapshot_size_limit_stops_before_publication(repo, tmp_path, monkeypatch):
    monkeypatch.setattr(snapshot, "MAX_BYTES", 4)
    with pytest.raises(ValueError, match="512 MiB"):
        snapshot.freeze_source(repo, tmp_path / "frozen")
    assert not (tmp_path / "frozen").exists()
    assert not list(tmp_path.glob(".pbs-source-*"))


@pytest.mark.parametrize("name", ["../outside", "/absolute", "a/../b", "a\\b", ".git/config", ".env.local"])
def test_snapshot_paths_reject_escapes_and_deployment_state(name):
    with pytest.raises(ValueError):
        snapshot._relative(name)


@pytest.fixture
def context(tmp_path, monkeypatch):
    import agent.harness
    from agent.harness import autonomous_research, catalog, installation, pbs_goal, pbs_gpu
    profile = Profile.from_dict({"mode": "science_expert", "harness": "expert-deepseek-harness",
        "model": {"provider": "dmx", "model": "glm-4-flash", "base_url": "https://www.dmxapi.cn/v1",
                  "credential_env": "TEST_PBS_KEY"},
        "budget": {"model_calls": 12, "tool_calls": 12, "wall_seconds": 120, "output_tokens": 4096, "thinking": "disabled"}})
    store = GoalStore.create(tmp_path / "goal", GoalSpec("Synthetic arithmetic research", ("verified",)), profile)
    (store.root / "workspace").mkdir()
    env_file = tmp_path / "credential.env"
    env_file.write_text("TEST_PBS_KEY=fixture-selected-value\nUNRELATED_KEY=fixture-unrelated-value\n")
    env_file.chmod(0o600)
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    path = attempt / "submission.json"
    record = {"goal": str(store.root), "record_path": str(path), "source": {"root": str(ROOT)},
        "execution": {"python": sys.executable, "runtime_root": str(ROOT), "env_file": str(env_file),
                      "credential_toml": None, "environment": {}, "extra_python_paths": []},
        "resources": {"queue": "fixture", "ncpus": 2, "ngpus": 1, "memory_gb": 16, "wall_seconds": 120},
        "job_id": "123.fixture", "nonce": "a" * 32, "cancel_requested": False}
    from agent.harness.pbs_goal_policy import freeze_settings
    catalog_file = tmp_path / "catalog.json"
    catalog_file.write_text('{"models": []}')
    record["settings"] = {"runtime_root": str(tmp_path), "repository": str(ROOT), "environment": {}, "execution_catalog": str(catalog_file)}
    record["tool_settings"] = freeze_settings(record["settings"], attempt)
    from contextlib import asynccontextmanager

    from agent.harness import pbs_goal_queue, pbs_goal_tools
    @asynccontextmanager
    async def private_context(_):
        yield
    monkeypatch.setattr(pbs_goal_queue, "queue_run", private_context)
    monkeypatch.setattr(pbs_goal_tools, "tool_service", private_context)
    calls, finished = [], []
    monkeypatch.setattr(os, "environ", dict(os.environ, PBS_JOBID="123.fixture", CUDA_VISIBLE_DEVICES="GPU-fixture",
                                           UNRELATED_HOST_SECRET="not-forwarded", VENUS_PBS_GOAL_ATTEMPT="a" * 32))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.setattr(pbs_goal, "worker_claim", lambda _: (calls.append("claim"), copy.deepcopy(record))[1])
    monkeypatch.setattr(pbs_goal, "worker_finish", lambda path, status, error_type=None: finished.append((status, error_type)))
    def ready(path, receipt):
        assert receipt["backend"] == "pbs" and receipt["job_id"] == "123.fixture"
        assert receipt["memory_hard_limit_bytes"] == 16 * 1024**3
        calls.append("ready")
    monkeypatch.setattr(pbs_goal, "worker_ready", ready, raising=False)
    monkeypatch.setattr(snapshot, "verify_source", lambda _: calls.append("source"))
    def capture(limits):
        calls.append("allocation")
        assert "UNRELATED_HOST_SECRET" not in os.environ
        return pbs_gpu.Allocation("123.fixture", "GPU-fixture", "/dev/nvidia9", 16 * 1024**3)
    monkeypatch.setattr(pbs_gpu, "capture", capture)
    monkeypatch.setattr(catalog, "free_model", lambda _: (calls.append("free"), True)[1])
    monkeypatch.setattr(installation, "default_runtimes", lambda _: {"python_bin": sys.executable})
    monkeypatch.setattr(agent.harness, "builtin_registry", lambda: SimpleNamespace(preflight=lambda *_: {"ready": True}))
    class Runtime:
        def __init__(self, actual_store, credential, runtimes):
            calls.append("runtime")
            assert actual_store.root == store.root and credential == "fixture-selected-value"
            assert "TEST_PBS_KEY" not in os.environ and "UNRELATED_KEY" not in os.environ
        async def run(self):
            calls.append("run")
            return {"status": "waiting_for_input", "cycles": []}
    monkeypatch.setattr(autonomous_research, "AutonomousResearch", Runtime)
    return SimpleNamespace(record=record, path=path, store=store, profile=profile, calls=calls, finished=finished)


def test_worker_verifies_allocation_then_free_route_and_preserves_goal_status(context):
    result = worker.run_record(context.path)
    assert result["status"] == "waiting_for_input"
    assert context.calls == ["claim", "source", "allocation", "ready", "free", "runtime", "run"]
    assert context.finished == [("waiting_for_input", None)]
    assert os.environ["VENUS_SCIENTIST_PBS_RECORD"] == str(context.path)
    for name in ("VENUS_JOB_QUEUE_ROOT", "VENUS_SANDBOX_CONTROL_ROOT", "VENUS_RUNTIME_ROOT"):
        assert Path(os.environ[name]).is_relative_to(context.path.parent)
    assert os.environ["PYTHON_DOTENV_DISABLED"] == "1"
    assert context.store.read()["usage"]["model_calls"] == 0


def test_worker_missing_allocation_never_reads_credentials(context, monkeypatch):
    from agent.harness import pbs_gpu
    from agent.harness.session_sandbox import SandboxUnavailable
    def capture(_):
        raise SandboxUnavailable("Synthetic unallocated host")
    monkeypatch.setattr(pbs_gpu, "capture", capture)
    monkeypatch.setattr(worker, "_selected_credential", lambda *_: pytest.fail("credentials read before allocation"))
    with pytest.raises(SandboxUnavailable):
        worker.run_record(context.path)
    assert context.finished == [("interrupted", "SandboxUnavailable")]
    assert "runtime" not in context.calls


def test_worker_paid_route_never_reads_credentials(context, monkeypatch):
    from agent.harness import catalog
    monkeypatch.setattr(catalog, "free_model", lambda _: False)
    monkeypatch.setattr(worker, "_selected_credential", lambda *_: pytest.fail("paid route reached credentials"))
    with pytest.raises(ValueError, match="free model"):
        worker.run_record(context.path)
    assert context.finished == [("interrupted", "ValueError")]


def test_worker_claim_denial_cannot_finish_someone_elses_record(context, monkeypatch):
    from agent.harness import pbs_goal
    def deny(_):
        raise ValueError("cancelled or wrong owner")
    monkeypatch.setattr(pbs_goal, "worker_claim", deny)
    with pytest.raises(ValueError):
        worker.run_record(context.path)
    assert context.finished == [] and "run" not in context.calls


def test_worker_does_not_resume_paused_goal(context):
    context.store.pause()
    with pytest.raises(ValueError, match="active checkpointed"):
        worker.run_record(context.path)
    assert context.store.read()["status"] == "paused"
    assert "runtime" not in context.calls
    assert context.finished == [("paused", "ValueError")]


def test_worker_rejects_credential_environment_override(context):
    context.record["execution"]["environment"]["TEST_PBS_KEY"] = "must-not-be-exported"
    with pytest.raises(ValueError, match="environment"):
        worker.run_record(context.path)
    assert "allocation" not in context.calls


def test_credential_sources_are_private_selected_and_endpoint_bound(tmp_path, context):
    source = Path(context.record["execution"]["env_file"])
    assert worker._selected_credential(context.profile.model, context.record["execution"]) == "fixture-selected-value"
    source.chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        worker._selected_credential(context.profile.model, context.record["execution"])
    source.chmod(0o600)
    link = tmp_path / "linked.env"
    link.symlink_to(source)
    with pytest.raises(ValueError, match="regular"):
        worker._protected_text(link)
    toml = tmp_path / "provider.toml"
    toml.write_text('[providers.dmx]\nbase_url="https://wrong.invalid/v1"\napi_key="fixture-only"\n')
    toml.chmod(0o600)
    with pytest.raises(ValueError, match="endpoint"):
        worker._selected_credential(context.profile.model, {"credential_toml": str(toml)})


def test_worker_runtime_error_records_type_not_provider_or_exception_text(context, monkeypatch):
    from agent.harness import autonomous_research
    class Failed:
        def __init__(self, *_):
            pass
        async def run(self):
            raise RuntimeError("private detail never recorded")
    monkeypatch.setattr(autonomous_research, "AutonomousResearch", Failed)
    with pytest.raises(RuntimeError):
        worker.run_record(context.path)
    assert context.finished == [("interrupted", "RuntimeError")]


@pytest.mark.asyncio
async def test_assignment_guard_holds_runner_lease_before_reserving_budget(context, monkeypatch):
    import fcntl

    from agent.harness import pbs_goal
    from agent.science_goal import run_goal
    checked = []
    def guard(goal):
        with (Path(goal) / ".runner.lock").open("a") as duplicate:
            with pytest.raises(BlockingIOError):
                fcntl.flock(duplicate, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checked.append(True)
        raise RuntimeError("PBS already owns this goal")
    monkeypatch.setattr(pbs_goal, "assert_execution_allowed", guard)
    async def forbidden(*_):
        pytest.fail("Conflicting runner executed")
    with pytest.raises(RuntimeError, match="already owns"):
        await run_goal(context.store, forbidden, lambda _: pytest.fail("must not evaluate"))
    state = context.store.read()
    assert checked == [True] and state["cycles"] == [] and state["inflight"] is None
    assert state["usage"]["model_calls"] == 0


def test_worker_rejects_preloaded_platform_module_from_other_checkout(context, monkeypatch):
    from types import ModuleType
    foreign = ModuleType("mcp_tools.synthetic_foreign")
    foreign.__file__ = "/outside/synthetic_foreign.py"
    monkeypatch.setitem(sys.modules, foreign.__name__, foreign)
    monkeypatch.setattr(worker, "_selected_credential", lambda *_: pytest.fail("mixed source reached credentials"))
    with pytest.raises(ValueError, match="outside the frozen"):
        worker.run_record(context.path)
    assert "runtime" not in context.calls


def test_fixed_bootstrap_ignores_foreign_pythonpath_and_sanitizes_exception(tmp_path):
    import shutil
    frozen = tmp_path / "frozen"
    scripts = frozen / "scripts/rsi"
    scripts.mkdir(parents=True)
    shutil.copyfile(ROOT / "scripts/rsi/pbs_goal_worker.py", scripts / "pbs_goal_worker.py")
    package = frozen / "src/agent/harness"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "pbs_goal_worker.py").write_text(
        'def run_record(path):\n    raise RuntimeError("fixture-sensitive-detail")\n')
    foreign = tmp_path / "foreign"
    (foreign / "agent").mkdir(parents=True)
    (foreign / "agent/__init__.py").write_text('raise AssertionError("wrong checkout imported")\n')
    record = tmp_path / "submission.json"
    record.write_text(json.dumps({"source": {"root": str(frozen)}, "execution": {"extra_python_paths": []}}))
    result = subprocess.run([sys.executable, str(scripts / "pbs_goal_worker.py"), "--record", str(record)],
        env=dict(os.environ, PYTHONPATH=str(foreign)), capture_output=True, text=True, timeout=20)
    assert result.returncode == 1
    assert json.loads(result.stdout) == {"status": "interrupted", "error_type": "RuntimeError"}
    assert "sensitive" not in result.stdout + result.stderr
    assert "wrong checkout" not in result.stdout + result.stderr
    assert not list(frozen.rglob("*.pyc"))


def test_distinct_venv_executable_is_not_accepted_by_symlink_equivalence(context):
    other = context.path.parent / "other-venv/bin/python"
    other.parent.mkdir(parents=True)
    other.symlink_to(sys.executable)
    assert other.resolve() == Path(sys.executable).resolve()
    context.record["execution"]["python"] = str(other)
    with pytest.raises(ValueError, match="interpreter"):
        worker.run_record(context.path)
    assert "allocation" not in context.calls


def test_credential_fifo_rejects_without_waiting_for_a_writer(tmp_path):
    fifo = tmp_path / "credential.pipe"
    os.mkfifo(fifo, 0o600)
    code = ("import sys; from agent.harness.pbs_goal_worker import _protected_text; "
            "\ntry: _protected_text(sys.argv[1])\nexcept ValueError: sys.exit(0)\nelse: sys.exit(2)")
    result = subprocess.run([sys.executable, "-c", code, str(fifo)],
        env=dict(os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep + str(ROOT)),
        capture_output=True, text=True, timeout=5)
    assert result.returncode == 0


def test_archive_transport_deadline_kills_and_reaps_a_stalled_git(tmp_path, monkeypatch):
    import tempfile
    import time
    original = subprocess.Popen
    children = []
    def stalled(_command, **kwargs):
        child = original([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(snapshot.subprocess, "Popen", stalled)
    monkeypatch.setattr(snapshot, "ARCHIVE_SECONDS", .05)
    started = time.monotonic()
    with tempfile.TemporaryFile() as output, pytest.raises(TimeoutError, match="deadline"):
        snapshot._archive(tmp_path, "fixture", output)
    assert time.monotonic() - started < 5
    assert children and children[0].poll() is not None


def test_archive_transport_stops_oversized_output_before_writing_it(tmp_path, monkeypatch):
    import tempfile
    original = subprocess.Popen
    children = []
    def oversized(_command, **kwargs):
        child = original([sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'x'*65536)"], **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(snapshot.subprocess, "Popen", oversized)
    monkeypatch.setattr(snapshot, "MAX_BYTES", 1024)
    with tempfile.TemporaryFile() as output:
        with pytest.raises(ValueError, match="512 MiB"):
            snapshot._archive(tmp_path, "fixture", output)
        assert output.tell() <= 1024
    assert children[0].poll() is not None


def test_allocation_receipt_must_persist_before_credentials(context, monkeypatch):
    from agent.harness import pbs_goal
    def fail_ready(*_):
        raise OSError("Synthetic receipt persistence failure")
    monkeypatch.setattr(pbs_goal, "worker_ready", fail_ready)
    monkeypatch.setattr(worker, "_selected_credential", lambda *_: pytest.fail("receipt not saved"))
    with pytest.raises(OSError):
        worker.run_record(context.path)
    assert "free" not in context.calls and "runtime" not in context.calls
    assert context.finished == [("interrupted", "OSError")]


@pytest.mark.parametrize("invalid", [None, "cancelled", "different_job", "different_receipt", "already_running"])
def test_worker_ready_real_private_record_contract(tmp_path, monkeypatch, invalid):
    import pwd

    from agent.harness import pbs_goal, pbs_gpu
    folder = tmp_path / "goal/attempt"
    folder.mkdir(parents=True)
    path = folder / "submission.json"
    record = {"schema": pbs_goal.SCHEMA, "uid": os.getuid(), "owner": pwd.getpwuid(os.getuid()).pw_name,
              "nonce": "b" * 32, "record_path": str(path), "job_id": "123.fixture",
              "cancel_requested": invalid == "cancelled",
              "worker": {"state": "running" if invalid == "already_running" else "claimed"}}
    path.write_text(json.dumps(record))
    path.chmod(0o600)
    before = path.read_bytes()
    monkeypatch.setenv("PBS_JOBID", "456.fixture" if invalid == "different_job" else "123.fixture")
    receipt = pbs_gpu.Allocation("456.fixture" if invalid == "different_receipt" else "123.fixture",
                                 "GPU-fixture", "/dev/nvidia9", 16 * 1024**3).receipt()
    if invalid:
        with pytest.raises(ValueError, match="claimed worker"):
            pbs_goal.worker_ready(path, receipt)
        assert path.read_bytes() == before
    else:
        pbs_goal.worker_ready(path, receipt)
        saved = json.loads(path.read_text())
        assert saved["worker"]["state"] == "running"
        assert saved["worker"]["allocation"] == receipt
        assert saved["worker"]["ready_at"] > 0
        assert path.stat().st_mode & 0o077 == 0


@pytest.mark.parametrize("initialized", [False, True])
def test_real_gitlink_is_declared_but_host_checkout_is_never_packaged(repo, tmp_path, initialized):
    external = tmp_path / "external"
    external.mkdir()
    git(external, "init", "-q")
    git(external, "config", "user.email", "fixture@example.invalid")
    git(external, "config", "user.name", "Synthetic fixture")
    git(external, "config", "commit.gpgsign", "false")
    (external / "optional.py").write_text("EXTERNAL = True\n")
    git(external, "add", ".")
    git(external, "commit", "-qm", "External fixture")
    commit = git(external, "rev-parse", "HEAD").strip()
    location = "vendor/scientific-skills"
    if initialized:
        (repo / "vendor").mkdir()
        git(repo, "clone", "-q", "--no-hardlinks", str(external), location)
        # Host submodule files are not a source of snapshot bytes, even if dirty.
        (repo / location / "optional.py").write_text("HOST_MODIFIED = True\n")
        (repo / location / ".env").write_text("FIXTURE=not-a-real-credential\n")
    git(repo, "update-index", "--add", "--cacheinfo", "160000," + commit + "," + location)
    git(repo, "commit", "-qm", "Declare optional gitlink")
    value = snapshot.freeze_source(repo, tmp_path / "frozen-with-gitlink")
    assert value["gitlinks"] == {location: commit}
    assert set(value["files"]) == {"src/code.py", "scripts/worker.py", ".env.example"}
    assert not (Path(value["root"]) / location).exists()
    snapshot.verify_source(value)
    # Even a rehashed self-consistent manifest may not claim packed gitlink data.
    forbidden = Path(value["root"]) / location / "optional.py"
    forbidden.parent.mkdir(parents=True)
    forbidden.write_text("INJECTED = True\n")
    with pytest.raises(ValueError):
        snapshot.verify_source(value)
    value["files"][location + "/optional.py"] = snapshot._hash(forbidden)
    value["sha256"] = snapshot._digest({k: value[k] for k in ("commit", "files", "gitlinks")})
    with pytest.raises(ValueError, match="External submodule"):
        snapshot.verify_source(value)


def test_external_reference_commit_is_bound_to_manifest(repo, tmp_path):
    value = snapshot.freeze_source(repo, tmp_path / "frozen")
    value["gitlinks"] = {"vendor/optional": "a" * 40}
    with pytest.raises(ValueError, match="manifest changed"):
        snapshot.verify_source(value)


@pytest.mark.parametrize("kind", ["export_ignore_file", "export_ignore_directory", "export_subst", "nested_attributes"])
def test_release_archive_attributes_do_not_drop_or_replace_tracked_bytes(repo, tmp_path, kind):
    code = "COMMIT = '$Format:%H$'\n"
    (repo / "src/code.py").write_text(code)
    if kind == "export_ignore_file":
        attributes = "src/code.py export-ignore\n"
    elif kind == "export_ignore_directory":
        attributes = "src export-ignore\n"
    elif kind == "nested_attributes":
        attributes = "scripts export-ignore\n"
        (repo / "src/.gitattributes").write_text("*.py export-subst export-ignore\n")
    else:
        attributes = "src/code.py export-subst\n"
    (repo / ".gitattributes").write_text(attributes)
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "Release archive attributes fixture")
    info = repo / ".git/info/attributes"
    info.parent.mkdir(exist_ok=True)
    info.write_text(".env.example export-ignore\n")
    before = {"attributes": (repo / ".gitattributes").read_bytes(), "info": info.read_bytes(),
              "config": (repo / ".git/config").read_bytes(), "status": git(repo, "status", "--porcelain")}
    value = snapshot.freeze_source(repo, tmp_path / "frozen-release")
    expected = set(git(repo, "ls-tree", "-r", "--name-only", "HEAD").splitlines())
    assert set(value["files"]) == expected
    for name in expected:
        actual = (Path(value["root"]) / name).read_bytes()
        original = subprocess.run(["git", "-C", str(repo), "show", "HEAD:" + name],
                                  check=True, capture_output=True).stdout
        assert actual == original
    assert (Path(value["root"]) / "src/code.py").read_text() == code
    assert before == {"attributes": (repo / ".gitattributes").read_bytes(), "info": info.read_bytes(),
                      "config": (repo / ".git/config").read_bytes(), "status": git(repo, "status", "--porcelain")}
    snapshot.verify_source(value)
