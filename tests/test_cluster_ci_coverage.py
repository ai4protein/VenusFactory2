"""Keep cluster CI exercising the offline suite and its real CPU sandbox.

No scheduler, provider, or research model is invoked here. Report fixtures test
the workflow's fail-closed execution gate, not a copy of its implementation.
"""
from pathlib import Path
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def cluster_job():
    workflow = yaml.safe_load((ROOT / ".github/workflows/checks.yml").read_text())
    return workflow["jobs"]["cluster-runtime"]


def step(job, identity):
    return next(value for value in job["steps"] if value.get("id") == identity)


def selected_modules(job):
    command = shlex.split(step(job, "cluster-tests")["run"])
    assert command[:4] == ["python", "-B", "-m", "pytest"]
    assert not any(token == flag or token.startswith(flag + "=")
                   for token in command[4:] for flag in ("-k", "-m", "--ignore", "--ignore-glob", "--deselect"))
    assert not any(token in {"||", ";", "&&", "|"} for token in command)
    assert not job.get("continue-on-error")
    assert not step(job, "cluster-tests").get("continue-on-error")
    assert not step(job, "cluster-results").get("continue-on-error")
    assert not step(job, "cluster-results").get("if")
    selected = set()
    for token in command:
        if token.startswith("tests/"):
            assert "::" not in token, "Cluster CI must run whole modules"
            matches = set(ROOT.glob(token))
            assert matches, f"Unmatched cluster test selector: {token}"
            selected.update(matches)
    return selected


def test_cluster_ci_covers_all_scheduler_family_tests_and_durable_queues():
    # Naming families intentionally discover future regression modules as well.
    required = {path for pattern in ("test_cluster_*.py", "test_pbs_*.py")
                for folder in (ROOT / "tests", ROOT / "tests/harness")
                for path in folder.glob(pattern)}
    required.update(ROOT / "tests" / name for name in (
        "test_compute_pbs.py", "test_compute_slurm.py",
        "test_job_queue.py", "test_durable_tasks.py",
    ))
    assert required <= selected_modules(cluster_job())


def test_cluster_ci_disables_dotenv_and_model_downloads():
    env = cluster_job()["env"]
    for name in ("PYTHON_DOTENV_DISABLED", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        assert str(env[name]) == "1"
    assert env["VENUS_MODEL_CONFIG"] == "model-config.yaml.example"


def result_validator():
    run = step(cluster_job(), "cluster-results")["run"]
    prefix = "python - <<'PY'\n"
    assert run.startswith(prefix) and run.rstrip().endswith("\nPY")
    return run[len(prefix):].rstrip()[:-3]


@pytest.mark.parametrize("outcome", ["passed", "skipped", "failure", "error", "missing", "empty"])
def test_cluster_ci_report_gate_rejects_skipped_or_unexecuted_sandbox(tmp_path, outcome):
    suite = ET.Element("testsuite")
    if outcome != "empty":
        name = ("some_other_test" if outcome == "missing"
                else "test_cpu_sandbox_still_executes_with_verified_cpu_parent")
        case = ET.SubElement(suite, "testcase", name=name, classname="offline.fixture")
        if outcome in {"skipped", "failure", "error"}:
            ET.SubElement(case, outcome, message="synthetic fixture")
    ET.ElementTree(suite).write(tmp_path / "cluster-results.xml")
    result = subprocess.run([sys.executable, "-I", "-c", result_validator()],
                            cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert (result.returncode == 0) == (outcome == "passed"), result.stdout + result.stderr


def test_cluster_ci_reports_all_skips_not_just_the_sandbox(tmp_path):
    suite = ET.Element("testsuite")
    ET.SubElement(suite, "testcase", name="test_cpu_sandbox_still_executes_with_verified_cpu_parent")
    other = ET.SubElement(suite, "testcase", name="scheduler_boundary")
    ET.SubElement(other, "skipped", message="synthetic unavailable dependency")
    ET.ElementTree(suite).write(tmp_path / "cluster-results.xml")
    result = subprocess.run([sys.executable, "-I", "-c", result_validator()],
                            cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0, "Skipped scheduler coverage must not pass CI"


def test_cluster_ci_preserves_absolute_import_paths_for_child_processes():
    assert cluster_job()["env"]["PYTHONPATH"] == "${{ github.workspace }}:${{ github.workspace }}/src"
