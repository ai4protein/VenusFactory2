"""Keep cluster CI exercising the offline suite and its real CPU sandbox.

No scheduler, provider, or research model is invoked here. Report fixtures test
the workflow's fail-closed execution gate, not a copy of its implementation.
"""
from pathlib import Path
import copy
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


# These currently scheduled modules execute generated Python in the real sandbox.
# Keep complete modules in one job so later tests cannot silently inherit a runner
# that only happens to have bwrap installed (as our development host does).
REAL_SANDBOX_MODULES = {
    "tests/harness/" + name for name in (
        "test_autonomous_research.py",
        "test_multilabel_references.py",
        "test_multilabel_training.py",
        "test_research_advanced_data.py",
        "test_research_data_store.py",
        "test_research_dataset.py",
        "test_research_input_profile.py",
        "test_research_worker_recovery.py",
        "test_scientific_acceptance.py",
        "test_study_review_state.py",
        "test_training_stagnation.py",
    )
}


def assert_sandbox_test_ownership(workflow):
    owners = {name: [] for name in REAL_SANDBOX_MODULES}
    for job_name, job in workflow["jobs"].items():
        for index, test_step in enumerate(job["steps"]):
            run = test_step.get("run", "")
            if "pytest" not in run:
                continue
            command = shlex.split(run)
            if "pytest" not in command:
                continue
            for name in owners:
                path = ROOT / name
                selectors = [token for token in command if token.startswith("tests/")]
                if not any(path in ROOT.glob(token.split("::")[0]) for token in selectors):
                    continue
                owners[name].append(job_name)
                assert any(path in ROOT.glob(token) for token in selectors if "::" not in token), (
                    f"{name}: CI must run the entire sandbox module"
                )
                assert not any(token == flag or token.startswith(flag + "=")
                               for token in command[command.index("pytest") + 1:]
                               for flag in ("-k", "-m", "--ignore", "--ignore-glob", "--deselect"))
                assert not job.get("continue-on-error") and not job.get("if")
                assert not test_step.get("continue-on-error") and not test_step.get("if")
                preparation = job["steps"][:index]
                install = [i for i, item in enumerate(preparation)
                           if "apt-get install" in item.get("run", "") and "bubblewrap" in item["run"]
                           and not item.get("if") and not item.get("continue-on-error")]
                probe = [i for i, item in enumerate(preparation)
                         if "bwrap --unshare-all --die-with-parent --ro-bind / / -- /usr/bin/true" in item.get("run", "")
                         and not item.get("if") and not item.get("continue-on-error")]
                assert install and probe and min(install) <= max(probe), (
                    f"{name}: {job_name} must install and execute bubblewrap before tests"
                )
    assert all(len(jobs) == 1 for jobs in owners.values()), owners


def test_real_sandbox_modules_have_one_prepared_ci_job():
    workflow = yaml.safe_load((ROOT / ".github/workflows/checks.yml").read_text())
    assert_sandbox_test_ownership(workflow)


@pytest.mark.parametrize("regression", ["lightweight_job", "duplicate", "missing_probe", "filtered"])
def test_sandbox_ownership_guard_rejects_ci_regressions(regression):
    workflow = copy.deepcopy(yaml.safe_load((ROOT / ".github/workflows/checks.yml").read_text()))
    registry = workflow["jobs"]["model-registry"]
    tests = next(item for item in registry["steps"] if "python -m pytest" in item.get("run", ""))
    name = "tests/harness/test_study_review_state.py"
    if regression == "lightweight_job":
        tests["run"] = tests["run"].replace(" " + name, "")
        protocol = workflow["jobs"]["harness-protocols"]
        next(item for item in protocol["steps"] if "python -m pytest" in item.get("run", ""))["run"] += " " + name
    elif regression == "duplicate":
        registry["steps"].append(copy.deepcopy(tests))
    elif regression == "missing_probe":
        for item in registry["steps"]:
            item["run"] = item.get("run", "").replace(
                "bwrap --unshare-all --die-with-parent --ro-bind / / -- /usr/bin/true", "true")
    else:
        tests["run"] += " -k not_real_training"
    with pytest.raises(AssertionError):
        assert_sandbox_test_ownership(workflow)
