"""Runtime discovery is read-only, checkout-scoped and shared by Web and CLI."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent.harness import installation

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def environment(tmp_path, monkeypatch):
    for variable in installation.ENV_KEYS.values():
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    monkeypatch.setattr(installation.sys, "executable", str(tmp_path / "python/bin/python"))
    monkeypatch.setattr(installation.shutil, "which", lambda name: None)
    return tmp_path


def test_empty_checkout_uses_legacy_paths_without_creating_files(environment):
    root = environment / "checkout"
    paths = installation.default_runtimes(root)
    site = root / "research_runs/rsi/envs/dsh-0.1.5rc1/lib/python3.12/site-packages"
    assert paths == {
        "python_root": str(environment / "python"),
        "python_bin": str(environment / "python/bin/python"),
        "kimi_bin": str(environment / "home/.kimi-code/bin/kimi"),
        "claude_bin": str(root / "research_runs/rsi/envs/claude-code-2.1.312/package/claude"),
        "codex_bin": str(root / "research_runs/rsi/envs/codex-0.163.0/vendor/x86_64-unknown-linux-musl/bin/codex"),
        "dsh_site": str(site),
        "dsh_bin": str(site / "deepseek_harness_runtime/runtime/deepseek-harness-sdk-runtime-linux-x64"),
    }
    assert not root.exists()


def test_existing_legacy_binary_precedes_path(environment, monkeypatch):
    root = environment / "checkout"
    legacy = environment / "home/.kimi-code/bin/kimi"
    legacy.parent.mkdir(parents=True)
    legacy.touch()
    monkeypatch.setattr(installation.shutil, "which", lambda name: str(environment / "bin" / name))
    paths = installation.default_runtimes(root)
    assert paths["kimi_bin"] == str(legacy)
    assert paths["codex_bin"] == str(environment / "bin/codex")
    assert paths["claude_bin"] == str(environment / "bin/claude")


def test_receipt_and_environment_precedence_is_checkout_scoped(environment, monkeypatch):
    root = environment / "one"
    binary = root / "runtime/local/harnesses/codex/bin/codex"
    binary.parent.mkdir(parents=True)
    binary.touch()
    receipt = binary.parent.parent / "installation.json"
    receipt.write_text(json.dumps({"paths": {"codex_bin": str(binary.relative_to(root))}}))
    monkeypatch.setattr(installation.shutil, "which", lambda name: str(environment / "bin" / name))
    assert installation.default_runtimes(root)["codex_bin"] == str(binary)
    assert installation.default_runtimes(environment / "two")["codex_bin"] == str(environment / "bin/codex")
    monkeypatch.setenv("VENUS_CODEX_BIN", str(environment / "explicit/codex"))
    assert installation.default_runtimes(root)["codex_bin"] == str(environment / "explicit/codex")


def test_web_and_cli_share_root_without_executing_probe(environment, monkeypatch):
    from backend.web_v2 import agent_execution
    from backend.web_v2.science_goals import ScienceGoalService
    from scripts.rsi import run_harness_probe

    root = environment / "checkout"
    poison = root / "scripts/rsi/run_harness_probe.py"
    poison.parent.mkdir(parents=True)
    poison.write_text("raise AssertionError('Web runtime discovery executed a probe script')")
    monkeypatch.setattr(agent_execution, "ROOT", root)
    monkeypatch.setattr(run_harness_probe, "ROOT", root)
    service = ScienceGoalService(root=root, drivers={"fixture": object()})
    expected = installation.default_runtimes(root)
    assert agent_execution.runtime_paths() == service.runtime_paths() == run_harness_probe.default_runtimes() == expected
    assert not (root / "research_runs").exists()


def test_agent_discovery_does_not_load_scientist_or_probe():
    program = """
import sys
from backend.web_v2 import agent_execution
paths = agent_execution.runtime_paths()
assert paths['python_bin']
assert 'backend.web_v2.science_goals' not in sys.modules
assert 'run_harness_probe' not in sys.modules
assert 'scripts.rsi.run_harness_probe' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", program], cwd=ROOT, capture_output=True, text=True,
        env={"PATH": os.defpath, "PYTHONPATH": str(ROOT / "src"), "PYTHON_DOTENV_DISABLED": "1"},
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
