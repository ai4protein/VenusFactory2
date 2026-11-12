"""Explicit local input snapshots and real offline Scientist CLI creation."""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent.harness import Profile
from agent.harness.autonomous_research import create
from agent.harness.research_cli_inputs import CHUNK_BYTES, prepare_inputs, stage_inputs
from agent.harness.research_data_store import store_for_goal
from agent.harness.research_input_index import input_descriptions
from agent.science_goal import GoalStore

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/rsi/run_autonomous_research.py"
PROFILE = {
    "mode": "science_expert", "harness": "expert-deepseek-harness",
    "model": {"provider": "dmx", "model": "glm-4-flash", "base_url": "https://www.dmxapi.cn/v1"},
    "budget": {"model_calls": 12, "tool_calls": 12, "wall_seconds": 120,
               "output_tokens": 4096, "thinking": "disabled"},
}


@pytest.fixture(autouse=True)
def local_mode(monkeypatch):
    monkeypatch.setenv("WEBUI_V2_MODE", "local")
    monkeypatch.setenv("VENUS_SCIENTIST_ONLINE", "0")


@pytest.fixture
def cli():
    spec = importlib.util.spec_from_file_location("scientist_input_cli_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def new_goal(path, objective="Analyze the supplied protein sequence"):
    return create(path, Profile.from_dict(PROFILE), objective,
                  {"max_cycles": 2, "model_calls": 24, "tool_calls": 24, "wall_seconds": 300})


def create_argv(tmp_path, *inputs, action="create"):
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(PROFILE))
    args = [action, "--output", str(tmp_path / "goal"), "--profile", str(profile),
            "--objective", "Analyze the supplied protein sequence"]
    for path in inputs:
        args.extend(["--input", str(path)])
    return args


def test_cli_copies_only_declared_files_and_records_relative_web_compatible_metadata(tmp_path, cli):
    source = tmp_path / "source"
    source.mkdir()
    fasta = source / "protein.fasta"
    notes = source / "notes.md"
    fasta.write_text(">control\nACDEFG\n")
    notes.write_text("Measure protein solubility.\n")
    (source / "not-selected.csv").write_text("do not register")
    cli.main(create_argv(tmp_path, fasta, notes))
    store = GoalStore(tmp_path / "goal")
    inputs = store.read()["progress"]["inputs"]
    assert [row["name"] for row in inputs] == ["protein.fasta", "notes.md"]
    assert all(set(row) == {"id", "name", "size", "sha256", "path", "source_type"} for row in inputs)
    assert all(row["source_type"] == "cli" for row in inputs)
    assert str(source) not in json.dumps(inputs)
    assert inputs[0]["sha256"] == hashlib.sha256(fasta.read_bytes()).hexdigest()
    assert inputs[0]["size"] == fasta.stat().st_size
    assert inputs[0]["path"] == "inputs/protein.fasta"
    copied = store.root / "workspace" / inputs[0]["path"]
    assert copied.read_bytes() == fasta.read_bytes()
    assert copied.stat().st_ino != fasta.stat().st_ino
    fasta.write_text(">changed\nAAAA\n")
    assert copied.read_text() == ">control\nACDEFG\n"
    assert not (store.root / "workspace/inputs/not-selected.csv").exists()
    (store.root / "workspace/generated.csv").write_text("generated result")
    assert len(store.read()["progress"]["inputs"]) == 2


@pytest.mark.parametrize("fault", ["missing", "directory", "symlink", "hardlink", "fifo", "empty", "hidden", "unsafe_name"])
def test_all_sources_are_validated_before_goal_creation(tmp_path, cli, fault):
    valid = tmp_path / "valid.fasta"
    valid.write_text(">good\nACDE\n")
    bad = tmp_path / "invalid.txt"
    if fault == "directory":
        bad.mkdir()
    elif fault == "symlink":
        bad.symlink_to(valid)
    elif fault == "hardlink":
        os.link(valid, bad)
    elif fault == "fifo":
        os.mkfifo(bad)
    elif fault == "empty":
        bad.touch()
    elif fault in {"hidden", "unsafe_name"}:
        bad = tmp_path / (".private" if fault == "hidden" else "bad\nname.txt")
        bad.write_text("fixture")
    with pytest.raises((OSError, ValueError)):
        cli.main(create_argv(tmp_path, valid, bad))
    assert not (tmp_path / "goal").exists()
    assert valid.read_text() == ">good\nACDE\n"


@pytest.mark.parametrize("other_name", ["input.csv", "INPUT.CSV"])
def test_duplicate_target_names_are_rejected_without_creating_goal(tmp_path, cli, other_name):
    one = tmp_path / "one"
    two = tmp_path / "two"
    one.mkdir()
    two.mkdir()
    first, second = one / "input.csv", two / other_name
    first.write_text("first")
    second.write_text("second")
    with pytest.raises(ValueError, match="unique"):
        cli.main(create_argv(tmp_path, first, second))
    assert not (tmp_path / "goal").exists()


@pytest.mark.parametrize("action", ["run", "resume", "submit-pbs", "status-pbs", "cancel-pbs"])
def test_input_flag_is_create_only_and_never_silently_attaches(tmp_path, cli, action):
    source = tmp_path / "input.txt"
    source.write_text("fixture")
    with pytest.raises(SystemExit) as error:
        cli.main(create_argv(tmp_path, source, action=action))
    assert error.value.code == 2
    assert not (tmp_path / "goal").exists()


@pytest.mark.parametrize("variable", ["WEBUI_V2_MODE", "VENUS_SCIENTIST_ONLINE"])
def test_online_cannot_import_arbitrary_host_paths(tmp_path, cli, monkeypatch, variable):
    source = tmp_path / "input.txt"
    source.write_text("fixture")
    monkeypatch.setenv(variable, "online" if variable == "WEBUI_V2_MODE" else "1")
    with pytest.raises(ValueError, match="online mode"):
        cli.main(create_argv(tmp_path, source))
    assert not (tmp_path / "goal").exists()


def test_existing_goal_is_never_overwritten(tmp_path, cli):
    source = tmp_path / "input.txt"
    source.write_text("new input")
    goal = tmp_path / "goal"
    goal.mkdir()
    sentinel = goal / "keep.txt"
    sentinel.write_text("existing work")
    with pytest.raises(FileExistsError):
        cli.main(create_argv(tmp_path, source))
    assert sentinel.read_text() == "existing work"
    assert list(goal.iterdir()) == [sentinel]


def test_destination_collision_retains_other_file_and_failure_evidence(tmp_path):
    source = tmp_path / "input.txt"
    source.write_text("source")
    with prepare_inputs([source]) as prepared:
        store = new_goal(tmp_path / "goal")
        destination = store.root / "workspace/inputs/input.txt"
        destination.parent.mkdir()
        destination.write_text("keep existing bytes")
        with pytest.raises(RuntimeError, match="diagnostics"):
            stage_inputs(store, prepared)
    assert destination.read_text() == "keep existing bytes"
    state = store.read()
    assert state["status"] == "interrupted"
    assert state["progress"]["input_preparation"]["status"] == "failed"
    receipt = json.loads((store.root / "cli-input-preparation.json").read_text())
    assert receipt["error_type"] == "FileExistsError"
    assert receipt["inputs"] == []


def test_source_change_after_validation_does_not_register_unverified_bytes(tmp_path):
    source = tmp_path / "input.txt"
    source.write_text("before")
    with prepare_inputs([source]) as prepared:
        store = new_goal(tmp_path / "goal")
        source.write_text("changed")
        with pytest.raises(RuntimeError, match="retained"):
            stage_inputs(store, prepared)
    assert store.read()["progress"]["inputs"] == []
    assert not (store.root / "workspace/inputs/input.txt").exists()
    assert source.read_text() == "changed"


def test_copy_is_streamed_and_change_during_copy_preserves_failed_attempt(tmp_path):
    source = tmp_path / "large.txt"
    source.write_bytes(b"x" * (CHUNK_BYTES * 2 + 3))
    with prepare_inputs([source]) as prepared:
        item = prepared[0]
        original_stream = item.stream

        class ChangingStream:
            def fileno(self):
                return original_stream.fileno()

            def read(self, size):
                assert size == CHUNK_BYTES
                block = original_stream.read(size)
                if not getattr(self, "changed", False):
                    self.changed = True
                    with source.open("ab") as target:
                        target.write(b"changed")
                return block

        item.stream = ChangingStream()
        store = new_goal(tmp_path / "goal")
        with pytest.raises(RuntimeError, match="retained"):
            stage_inputs(store, prepared)
    assert store.read()["status"] == "interrupted"
    assert store.read()["progress"]["inputs"] == []
    assert (store.root / "workspace/inputs/large.txt").exists()


def test_model_table_is_protected_before_any_run_and_original_hash_is_preserved(tmp_path, cli):
    source = tmp_path / "measurements.csv"
    raw = b"id,sequence,target\np1,ACDEFGHIKLMNPQRSTVWY,591.234\np2,AAAAAAAAAAAAAAAAAAAA,712.345\n"
    source.write_bytes(raw)
    argv = create_argv(tmp_path, source)
    argv[argv.index("--objective") + 1] = "Develop and evaluate a regression model from supplied measurements"
    cli.main(argv)
    store = GoalStore(tmp_path / "goal")
    row = store.read()["progress"]["inputs"][0]
    assert row["sha256"] == hashlib.sha256(raw).hexdigest()
    assert row["size"] == len(raw)
    view = (store.root / "workspace" / row["path"]).read_text()
    assert "protected_research_table" in view
    assert "591.234" not in view and "712.345" not in view
    data = store_for_goal(store.root)
    assert data.original_bytes(row["path"]) == raw
    description = input_descriptions(store.root / "workspace", [row], source_reader=data.input_bytes)
    assert "591.234" not in json.dumps(description)
    assert description["files"][0]["columns"]
    assert source.read_bytes() == raw


def test_later_model_plan_uses_existing_episode_protection(tmp_path):
    source = tmp_path / "measurements.csv"
    raw = b"id,sequence,target\np1,ACDEFGHIKLMNPQRSTVWY,591.234\n"
    source.write_bytes(raw)
    with prepare_inputs([source]) as prepared:
        store = new_goal(tmp_path / "goal", "Study the supplied measurements")
        records = stage_inputs(store, prepared)
    store.mutate(lambda state: state["progress"].update(plan={"deliverable": "model"}))
    data = store_for_goal(store.root)
    assert data.protect_inputs(records, prior_agent_execution=False) == ["inputs/measurements.csv"]
    assert "591.234" not in (store.root / "workspace/inputs/measurements.csv").read_text()
    assert data.original_bytes("inputs/measurements.csv") == raw


def test_no_input_create_preserves_existing_web_preparation_behavior(tmp_path, cli, monkeypatch):
    monkeypatch.setenv("WEBUI_V2_MODE", "online")
    cli.main(create_argv(tmp_path))
    state = GoalStore(tmp_path / "goal").read()
    assert state["progress"] == {}
    assert not (tmp_path / "goal/cli-input-preparation.json").exists()


def test_input_count_and_second_registration_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="32"):
        with prepare_inputs(["unused-" + str(n) for n in range(33)]):
            pytest.fail("Invalid batch should not be opened")
    source = tmp_path / "input.txt"
    source.write_text("fixture")
    with prepare_inputs([source]) as prepared:
        store = new_goal(tmp_path / "goal")
        stage_inputs(store, prepared)
    with prepare_inputs([source]) as prepared:
        with pytest.raises(ValueError, match="only during goal creation"):
            stage_inputs(store, prepared)


def test_documented_cli_create_works_without_pythonpath_or_provider_keys(tmp_path):
    source = tmp_path / "protein.fasta"
    source.write_text(">control\nACDEFG\n")
    argv = create_argv(tmp_path, source)
    env = {key: value for key, value in os.environ.items() if not any(
        marker in key.upper() for marker in ("KEY", "TOKEN", "SECRET", "PASSWORD"))}
    env.update(PYTHONPATH="", PYTHON_DOTENV_DISABLED="1", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, str(SCRIPT), *argv], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    state = GoalStore(tmp_path / "goal").read()
    assert state["usage"]["model_calls"] == 0
    assert state["cycles"] == []
    assert state["progress"]["inputs"][0]["path"] == "inputs/protein.fasta"

def test_failed_second_copy_keeps_first_snapshot_and_its_receipt(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("first source")
    second.write_text("second source")
    with prepare_inputs([first, second]) as prepared:
        store = new_goal(tmp_path / "goal")
        second.write_text("changed after validation")
        with pytest.raises(RuntimeError, match="retained"):
            stage_inputs(store, prepared)
    state = store.read()
    assert state["status"] == "interrupted"
    assert [row["name"] for row in state["progress"]["inputs"]] == ["first.txt"]
    assert (store.root / "workspace/inputs/first.txt").read_text() == "first source"
    receipt = json.loads((store.root / "cli-input-preparation.json").read_text())
    assert receipt["failed_input"] == "second.txt"
    assert receipt["inputs"] == state["progress"]["inputs"]


def test_destination_directory_symlink_cannot_write_outside_workspace(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("source data")
    outside = tmp_path / "outside"
    outside.mkdir()
    with prepare_inputs([source]) as prepared:
        store = new_goal(tmp_path / "goal")
        (store.root / "workspace/inputs").symlink_to(outside, target_is_directory=True)
        with pytest.raises(RuntimeError, match="retained"):
            stage_inputs(store, prepared)
    assert list(outside.iterdir()) == []
    assert store.read()["status"] == "interrupted"
