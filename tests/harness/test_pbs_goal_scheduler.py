"""PBS CLI fault protocol tests with real goals, file locks and process restarts.

The fake batch server persists PBS-shaped JSON; it never starts a worker, fits a
model, reads credentials, or claims to prove real GPU execution.
"""

from __future__ import annotations

import copy
import fcntl
import getpass
import html
import json
import multiprocessing
import shlex
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from agent.harness import Profile, pbs_goal
from agent.harness.autonomous_research import create
from agent.harness.pbs_goal import Controller


class FakePBS:
    """A tiny persistent scheduler, shared by independently constructed clients."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "server.json"
        if not self.path.exists():
            self.path.write_text(
                json.dumps({"jobs": {}, "calls": [], "counter": 1200, "lose_qsub": False, "query_error": False})
            )

    @contextmanager
    def transaction(self):
        with (self.root / "server.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            state = json.loads(self.path.read_text())
            yield state
            self.path.write_text(json.dumps(state))

    def read(self):
        with self.transaction() as state:
            return copy.deepcopy(state)

    def configure(self, **values):
        with self.transaction() as state:
            state.update(values)

    def change(self, identity, **values):
        with self.transaction() as state:
            state["jobs"][identity].update(values)

    def count(self, command):
        return sum(Path(call[0]).name == command for call in self.read()["calls"])

    def __call__(self, argv, *, timeout, env, **kwargs):
        assert isinstance(argv, (list, tuple)) and all(isinstance(item, str) for item in argv)
        assert timeout > 0
        assert "PBS_FAULT_TEST_SECRET" not in env, "Scheduler must not inherit provider secrets"
        command = Path(argv[0]).name
        lose = False
        with self.transaction() as state:
            state["calls"].append(list(argv))
            if command == "qsub":
                assert "-h" in argv and "-V" not in argv
                assert argv[argv.index("-r") + 1] == "n"
                assert argv[argv.index("-m") + 1] == "n"
                name = argv[argv.index("-N") + 1]
                exported = argv[argv.index("-v") + 1]
                variables = dict(part.split("=", 1) for part in exported.split(","))
                assert set(variables) == {"VENUS_PBS_GOAL_ATTEMPT"}
                nonce = variables["VENUS_PBS_GOAL_ATTEMPT"]
                assert len(nonce) == 32 and name == "vf_" + nonce
                direct = argv.index("--")
                executable, args = argv[direct + 1], argv[direct + 2 :]
                resources = {}
                for i, item in enumerate(argv[:-1]):
                    if item == "-l":
                        resources.update(part.split("=", 1) for part in argv[i + 1].split(","))
                select = resources.get("select", "")
                for part in select.split(":")[1:]:
                    key, value = part.split("=", 1)
                    resources[key] = int(value) if key in {"ncpus", "ngpus"} else value
                resources["nodect"] = 1
                state["counter"] += 1
                identity = str(state["counter"]) + ".fs"
                state["jobs"][identity] = {
                    "Job_Name": name,
                    "Job_Owner": getpass.getuser() + "@submit-node",
                    "job_state": "H",
                    "Hold_Types": "u",
                    "Rerunable": "False",
                    "Resource_List": resources,
                    "Variable_List": variables,
                    "queue": argv[argv.index("-q") + 1],
                    "server": "fs",
                    "executable": "<jsdl-hpcpa:Executable>" + html.escape(executable) + "</jsdl-hpcpa:Executable>",
                    "argument_list": "".join(
                        "<jsdl-hpcpa:Argument>" + html.escape(arg) + "</jsdl-hpcpa:Argument>" for arg in args
                    ),
                    "Submit_arguments": shlex.join(argv[1:]),
                }
                response = subprocess.CompletedProcess(argv, 0, identity + "\n", "")
                lose = state.pop("lose_qsub", False)
                state["query_error"] = state.pop("query_error_after_submit", state.get("query_error", False))
            elif command == "qstat":
                if state.get("query_error"):
                    return subprocess.CompletedProcess(
                        argv, 1, "", state.get("stderr_secret", "Cannot connect to server")
                    )
                ids = [arg for arg in argv[1:] if arg in state["jobs"]]
                # The only positional non-option arguments besides JSON format are job IDs.
                requested = [arg for arg in argv[1:] if arg.endswith(".fs")]
                jobs = {identity: state["jobs"][identity] for identity in ids} if requested else state["jobs"]
                response = subprocess.CompletedProcess(
                    argv, 0, json.dumps({"pbs_version": "23.06.06", "pbs_server": "fs", "Jobs": jobs}), ""
                )
            elif command == "qrls":
                assert argv[1:3] == ["-h", "u"]
                identity = argv[-1]
                job = state["jobs"][identity]
                job["Hold_Types"] = job["Hold_Types"].replace("u", "") or "n"
                if job["Hold_Types"] == "n":
                    job["job_state"] = "Q"
                response = subprocess.CompletedProcess(argv, 0, "", "")
                lose = state.pop("lose_qrls", False)
            elif command == "qdel":
                assert len(argv) == 2, "Normal cancellation must not force or erase history"
                identity = argv[1]
                state["jobs"][identity].update(job_state="F", Exit_status=271)
                response = subprocess.CompletedProcess(argv, 0, "", "")
            else:
                raise AssertionError("Unexpected scheduler command: " + command)
        if lose:
            raise subprocess.TimeoutExpired(argv, timeout, output="", stderr="lost acknowledgement")
        return response


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setenv("WEBUI_V2_MODE", "local")
    monkeypatch.setenv("VENUS_SCIENTIST_ONLINE", "0")
    monkeypatch.setenv("PBS_FAULT_TEST_SECRET", "must-not-reach-qsub")
    repository = tmp_path / "repository with spaces"
    entry = repository / "scripts/rsi/pbs_goal_worker.py"
    entry.parent.mkdir(parents=True)
    entry.write_text("# Trusted worker entry fixture; the fake scheduler never executes it.\n")
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    catalog = repository / "configs/harness/free_execution_catalog.json"
    catalog.parent.mkdir(parents=True)
    catalog.write_text('{"models": []}')
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "-qm",
            "Fixture worker",
        ],
        cwd=repository,
        check=True,
    )
    roots = tmp_path / "goals"
    roots.mkdir()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    commands = tmp_path / "pbs-bin"
    commands.mkdir()
    for command in ["qsub", "qstat", "qdel", "qrls"]:
        binary = commands / command
        binary.write_text("#!/bin/sh\nexit 99\n")
        binary.chmod(0o700)
    settings = {
        "schema": "venusfactory.pbs-config.v1",
        "repository": str(repository),
        "control_root": str(tmp_path / "control"),
        "python": sys.executable,
        "runtime_root": str(runtime),
        "pbs_bin": str(commands),
        "goal_roots": [str(roots)],
        "profiles": {"gpu": {"queue": "ai4090d", "ncpus": 2, "ngpus": 1, "memory_gb": 16, "wall_seconds": 7200}},
    }
    profile = Profile.from_dict(
        {
            "mode": "science_expert",
            "harness": "expert-deepseek-harness",
            "model": {"provider": "dmx", "model": "glm-4-flash", "base_url": "https://www.dmxapi.cn/v1"},
            "budget": {
                "model_calls": 12,
                "tool_calls": 12,
                "wall_seconds": 120,
                "output_tokens": 4096,
                "thinking": "disabled",
            },
        }
    )
    store = create(
        roots / "protein-study",
        profile,
        "Develop a protein model; $(touch SHOULD_NOT_EXIST)",
        {"max_cycles": 3, "model_calls": 36, "tool_calls": 36, "wall_seconds": 7200},
    )
    return settings, store, FakePBS(tmp_path / "fake-pbs")


def recoverable_call(fn):
    """Transport ambiguity may be returned or raised; neither permits a resubmit."""
    try:
        return fn()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        return None


def test_submit_is_held_before_release_and_preserves_goal_budget(prepared):
    settings, store, server = prepared
    before = store.read()
    result = Controller(settings, runner=server).submit(store.root)
    assert result["job_id"] and result["attempt_id"]
    commands = [Path(call[0]).name for call in server.read()["calls"]]
    assert commands.count("qsub") == 1 and commands.count("qrls") == 1
    assert commands.index("qsub") < commands.index("qstat") < commands.index("qrls")
    after = store.read()
    assert after["usage"] == before["usage"] and after["cycles"] == [] and after["inflight"] is None
    assert result["state"] in {"queued", "held"} and result["goal_status"] == "active"
    submitted = next(call for call in server.read()["calls"] if Path(call[0]).name == "qsub")
    assert not any("SHOULD_NOT_EXIST" in arg or "Develop a protein" in arg for arg in submitted)


def _submit_process(settings, goal, fake_root, start, results):
    try:
        start.wait(10)
        result = Controller(settings, runner=FakePBS(fake_root)).submit(Path(goal))
        results.put(("ok", result))
    except BaseException as exc:
        results.put(("error", type(exc).__name__ + ": " + str(exc)))


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="Linux shared-file-lock boundary")
def test_two_processes_submit_one_pbs_job(prepared):
    settings, store, server = prepared
    ctx = multiprocessing.get_context("fork")
    start, results = ctx.Event(), ctx.Queue()
    children = [
        ctx.Process(target=_submit_process, args=(settings, str(store.root), str(server.root), start, results))
        for _ in range(2)
    ]
    for child in children:
        child.start()
    start.set()
    outcomes = [results.get(timeout=20) for _ in children]
    for child in children:
        child.join(20)
        assert child.exitcode == 0
    assert all(kind == "ok" for kind, _ in outcomes), outcomes
    assert len({value["job_id"] for _, value in outcomes}) == 1
    assert server.count("qsub") == 1


def test_lost_qsub_reply_recovers_same_held_job_after_controller_restart(prepared):
    settings, store, server = prepared
    server.configure(lose_qsub=True)
    recoverable_call(lambda: Controller(settings, runner=server).submit(store.root))
    original = next(iter(server.read()["jobs"]))
    restored = Controller(settings, runner=server).status(store.root)
    assert restored["job_id"] == original
    repeated = Controller(settings, runner=server).submit(store.root)
    assert repeated["job_id"] == original and server.count("qsub") == 1
    assert store.read()["usage"]["model_calls"] == 0


def test_unresolved_submission_does_not_blindly_submit_again(prepared):
    settings, store, server = prepared
    server.configure(lose_qsub=True, query_error_after_submit=True)
    recoverable_call(lambda: Controller(settings, runner=server).submit(store.root))
    for _ in range(2):
        recoverable_call(lambda: Controller(settings, runner=server).status(store.root))
        recoverable_call(lambda: Controller(settings, runner=server).submit(store.root))
    assert server.count("qsub") == 1 and server.count("qrls") == 0
    assert store.read()["usage"]["model_calls"] == 0


@pytest.mark.parametrize("fault", ["owner", "nonce", "two_matching_jobs"])
def test_mismatched_or_ambiguous_recovery_never_releases_or_cancels_an_unbound_job(prepared, fault):
    settings, store, server = prepared
    server.configure(lose_qsub=True, query_error_after_submit=True)
    recoverable_call(lambda: Controller(settings, runner=server).submit(store.root))
    with server.transaction() as state:
        identity = next(iter(state["jobs"]))
        if fault == "owner":
            state["jobs"][identity]["Job_Owner"] = "different-owner@submit-node"
        elif fault == "nonce":
            state["jobs"][identity]["Variable_List"]["VENUS_PBS_GOAL_ATTEMPT"] = "f" * 32
        else:
            state["jobs"]["9999.fs"] = copy.deepcopy(state["jobs"][identity])
        state["query_error"] = False
    recoverable_call(lambda: Controller(settings, runner=server).status(store.root))
    recoverable_call(lambda: Controller(settings, runner=server).submit(store.root))
    recoverable_call(lambda: Controller(settings, runner=server).cancel(store.root))
    assert server.count("qsub") == 1 and server.count("qrls") == 0 and server.count("qdel") == 0


def test_cancel_accepted_but_unacknowledged_submission_uses_exact_job_and_no_budget(prepared):
    settings, store, server = prepared
    before = store.read()["usage"]
    server.configure(lose_qsub=True, query_error_after_submit=True)
    recoverable_call(lambda: Controller(settings, runner=server).submit(store.root))
    job_id = next(iter(server.read()["jobs"]))
    server.configure(query_error=False)
    result = Controller(settings, runner=server).cancel(store.root)
    assert result["cancel_requested"] is True
    assert server.count("qsub") == 1 and server.count("qdel") == 1 and server.count("qrls") == 0
    deletion = next(call for call in server.read()["calls"] if Path(call[0]).name == "qdel")
    assert deletion[-1] == job_id
    assert store.read()["usage"] == before and store.read()["cycles"] == []


def test_pbs_exit_zero_does_not_complete_a_waiting_research_goal(prepared):
    settings, store, server = prepared
    result = Controller(settings, runner=server).submit(store.root)
    store.mutate(lambda state: state.update(status="waiting_for_input"))
    server.change(result["job_id"], job_state="F", Exit_status=0)
    observed = Controller(settings, runner=server).status(store.root)
    assert observed["state"] == "finished" and observed["exit_code"] == 0
    assert observed["goal_status"] == "waiting_for_input"
    assert store.read()["status"] == "waiting_for_input" and server.count("qsub") == 1


@pytest.mark.parametrize("policy", ["online", "scientist_online"])
def test_online_policy_blocks_submitting_and_releasing_pending_jobs(prepared, monkeypatch, policy):
    settings, store, server = prepared
    server.configure(lose_qsub=True, query_error_after_submit=True)
    recoverable_call(lambda: Controller(settings, runner=server).submit(store.root))
    server.configure(query_error=False)
    monkeypatch.setenv("WEBUI_V2_MODE", "online" if policy == "online" else "local")
    monkeypatch.setenv("VENUS_SCIENTIST_ONLINE", "1" if policy == "scientist_online" else "0")
    recoverable_call(lambda: Controller(settings, runner=server).submit(store.root))
    recoverable_call(lambda: Controller(settings, runner=server).status(store.root))
    assert server.count("qsub") == 1 and server.count("qrls") == 0
    assert store.read()["usage"]["model_calls"] == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("ngpus", 2),
        ("ngpus", 0),
        ("memory_gb", 0),
        ("memory_gb", 128),
        ("ncpus", 0),
        ("wall_seconds", 0),
        ("queue", "ai4090d;touch bad"),
    ],
)
def test_invalid_resources_never_reach_pbs(prepared, field, value):
    settings, store, server = prepared
    settings["profiles"]["gpu"][field] = value
    with pytest.raises((ValueError, OSError)):
        Controller(settings, runner=server).submit(store.root)
    assert server.count("qsub") == 0


def test_goal_must_be_inside_configured_roots_and_clean_source_required(prepared, tmp_path):
    settings, store, server = prepared
    settings["goal_roots"] = [str(tmp_path / "different-root")]
    with pytest.raises((ValueError, OSError)):
        Controller(settings, runner=server).submit(store.root)
    assert server.count("qsub") == 0
    settings["goal_roots"] = [str(store.root.parent)]
    entry = Path(settings["repository"]) / "scripts/rsi/pbs_goal_worker.py"
    entry.write_text("changed after frozen git revision\n")
    with pytest.raises((ValueError, OSError)):
        Controller(settings, runner=server).submit(store.root)
    assert server.count("qsub") == 0


def test_lost_release_reply_recovers_without_resubmitting_or_releasing_twice(prepared):
    settings, store, server = prepared
    server.configure(lose_qrls=True)
    recoverable_call(lambda: Controller(settings, runner=server).submit(store.root))
    identity = next(iter(server.read()["jobs"]))
    assert server.read()["jobs"][identity]["job_state"] == "Q"
    observed = Controller(settings, runner=server).status(store.root)
    assert observed["job_id"] == identity and observed["state"] == "queued"
    assert Controller(settings, runner=server).submit(store.root)["job_id"] == identity
    assert server.count("qsub") == 1 and server.count("qrls") == 1


def test_config_file_resolves_relative_paths_without_inheriting_secrets(prepared, tmp_path, monkeypatch):
    settings, store, server = prepared
    config = copy.deepcopy(settings)
    for key in ["repository", "control_root", "runtime_root", "pbs_bin"]:
        config[key] = str(Path(settings[key]).relative_to(tmp_path))
    config["goal_roots"] = [str(Path(root).relative_to(tmp_path)) for root in settings["goal_roots"]]
    path = tmp_path / "pbs.json"
    path.write_text(json.dumps(config))
    controller = Controller.from_file(path)
    controller.runner = server  # Use the same public dependency as Controller(..., runner=...).
    result = controller.submit(store.root)
    assert result["job_id"] and server.count("qsub") == 1


def test_confirmed_release_does_not_clear_a_later_manual_hold(prepared):
    settings, store, server = prepared
    created = Controller(settings, runner=server).submit(store.root)
    confirmed = Controller(settings, runner=server).status(store.root)
    assert confirmed["state"] == "queued"
    server.change(created["job_id"], job_state="H", Hold_Types="u")
    observed = Controller(settings, runner=server).status(store.root)
    assert observed["state"] == "held"
    assert Controller(settings, runner=server).submit(store.root)["state"] == "held"
    assert server.count("qrls") == 1 and server.count("qsub") == 1


def test_preparation_failure_remains_explicit_without_querying_pbs(prepared):
    settings, store, server = prepared
    entry = Path(settings["repository"]) / "scripts/rsi/pbs_goal_worker.py"
    entry.write_text("dirty source must not execute\n")
    with pytest.raises((ValueError, OSError)):
        Controller(settings, runner=server).submit(store.root)
    for _ in range(2):
        observed = Controller(settings, runner=server).status(store.root)
        assert observed["state"] == "preparation_failed" and observed["job_id"] is None
        assert observed["goal_status"] == "active"
    assert server.read()["calls"] == []


@pytest.mark.parametrize("failure", ["history_missing", "server_unavailable"])
def test_stale_job_or_query_error_is_not_terminal_and_cannot_force_a_new_attempt(prepared, failure):
    settings, store, server = prepared
    created = Controller(settings, runner=server).submit(store.root)
    with server.transaction() as state:
        if failure == "history_missing":
            state["jobs"].clear()
        else:
            state["query_error"] = True
    observed = Controller(settings, runner=server).status(store.root)
    assert observed["state"] == "unknown" and observed["job_id"] == created["job_id"]
    assert observed["goal_status"] == "active" and store.read()["usage"]["model_calls"] == 0
    with pytest.raises((ValueError, RuntimeError)):
        Controller(settings, runner=server).submit(store.root, new_attempt=True)
    assert server.count("qsub") == 1 and server.count("qrls") == 1


def test_reading_old_terminal_attempt_does_not_interrupt_user_resumed_goal(prepared):
    settings, store, server = prepared
    created = Controller(settings, runner=server).submit(store.root)
    server.change(created["job_id"], job_state="F", Exit_status=0)
    terminal = Controller(settings, runner=server).status(store.root)
    assert terminal["state"] == "finished" and store.read()["status"] == "interrupted"
    store.resume()
    resumed = store.read()
    observed = Controller(settings, runner=server).status(store.root)
    assert observed["state"] == "finished" and observed["goal_status"] == "active"
    assert store.read() == resumed, "Inspecting old PBS history must not change a newly resumed goal"
    assert server.count("qsub") == 1


@pytest.mark.parametrize("fault", ["gpu_count", "memory", "cpus", "walltime", "rerunnable", "nodes", "chunks"])
def test_altered_pbs_resources_block_execution_but_allow_exact_owner_cancellation(prepared, fault, monkeypatch):
    settings, store, server = prepared
    server.configure(lose_qsub=True, query_error_after_submit=True)
    recoverable_call(lambda: Controller(settings, runner=server).submit(store.root))
    with server.transaction() as state:
        identity = next(iter(state["jobs"]))
        job = state["jobs"][identity]
        if fault == "gpu_count":
            job["Resource_List"]["ngpus"] = 2
            job["Resource_List"]["select"] = "1:ncpus=2:ngpus=2:mem=16gb"
        elif fault == "memory":
            job["Resource_List"]["mem"] = "32gb"
        elif fault == "cpus":
            job["Resource_List"]["ncpus"] = 4
        elif fault == "walltime":
            job["Resource_List"]["walltime"] = "03:00:00"
        elif fault == "nodes":
            job["Resource_List"]["nodect"] = 2
            job["Resource_List"]["select"] = "2:ncpus=1:ngpus=1:mem=8gb"
        elif fault == "chunks":
            job["Resource_List"]["select"] = "1:ncpus=1:ngpus=1:mem=8gb+1:ncpus=1:mem=8gb"
        else:
            job["Rerunable"] = "True"
        state["query_error"] = False
    recoverable_call(lambda: Controller(settings, runner=server).status(store.root))
    recoverable_call(lambda: Controller(settings, runner=server).submit(store.root))
    assert server.count("qsub") == 1 and server.count("qrls") == 0
    assert server.read()["jobs"][identity]["job_state"] == "H"

    # An external release cannot bypass worker-side resource admission either.
    server.change(identity, job_state="R", Hold_Types="n")
    monkeypatch.setenv("PBS_JOBID", identity)
    monkeypatch.setattr(pbs_goal, "bounded_run", server)
    records = list(Path(settings["control_root"]).glob("*/*/submission.json"))
    assert len(records) == 1
    with pytest.raises(ValueError, match="PBS"):
        pbs_goal.worker_claim(records[0])
    assert not json.loads(records[0].read_text()).get("worker")

    # The exact owner/nonce receipt still authorizes releasing occupied resources.
    cancelled = Controller(settings, runner=server).cancel(store.root)
    assert cancelled["cancel_requested"] and cancelled["job_id"] == identity
    deletions = [call for call in server.read()["calls"] if Path(call[0]).name == "qdel"]
    assert deletions == [[str(Path(settings["pbs_bin"]) / "qdel"), identity]]
    assert server.count("qsub") == 1 and server.count("qrls") == 0
    assert Controller(settings, runner=server).status(store.root)["state"] == "cancelled"
    assert store.read()["usage"]["model_calls"] == 0


def test_failed_scheduler_diagnostics_do_not_echo_sensitive_stderr(prepared, capsys):
    settings, store, server = prepared
    secret = "PRIVATE_PROVIDER_TOKEN_SENTINEL"
    server.configure(lose_qsub=True, query_error_after_submit=True, stderr_secret=secret)
    for operation in [
        lambda: Controller(settings, runner=server).submit(store.root),
        lambda: Controller(settings, runner=server).status(store.root),
    ]:
        try:
            response = operation()
        except Exception as exc:
            response = {"exception": str(exc)}
        assert secret not in json.dumps(response)
    captured = capsys.readouterr()
    assert secret not in captured.out and secret not in captured.err
    assert server.count("qsub") == 1 and server.count("qrls") == 0
