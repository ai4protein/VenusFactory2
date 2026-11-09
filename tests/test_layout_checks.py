"""Protect source boundaries and references shipped in the application registry."""

import ast
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("layout_check", ROOT / "scripts/check_layout.py")
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


@pytest.mark.parametrize("source", [
    "events", "session_boundary", "web_native_worker", "dsh_worker", "installation",
    "adapters", "kimi_adapter", "kimi_web_client", "kimi_web_events", "kimi_web_sandbox",
])
@pytest.mark.parametrize("statement", [
    "from .web_worker import emit",
    "from agent.harness.web_worker import _mcp_input_schema",
    "import agent.harness.web_native_worker as native",
    "from . import web_native_worker",
    "from ..harness import web_worker",
    "from src.agent.harness.web_worker import emit",
    "def run():\n    from .web_worker import emit",
])
def test_shared_contracts_and_adapters_reject_entry_point_imports(source, statement):
    path = ROOT / f"src/agent/harness/{source}.py"
    assert checks.harness_worker_import_errors(path, ast.parse(statement), ROOT)


@pytest.mark.parametrize("source,statement", [
    ("web_worker", "from agent.harness.web_native_worker import run"),
    ("autonomous_research", "from .web_native_worker import run"),
    ("session_boundary", "from .events import emit"),
    ("events", "from .request_recovery import audit_recovery"),
    ("web_native_worker", "from .session_boundary import prepare_tool_arguments"),
    ("dsh_worker", "import library.worker"),
    ("kimi_adapter", "from .kimi_web_sandbox import KimiWebSandbox"),
    ("kimi_web_client", "from .kimi_web_events import KimiTranscript"),
    ("kimi_web_events", "from agent.kimi_client import KimiEvent"),
    ("kimi_web_sandbox", "from .prepared_mcp import PreparedMcp"),
    ("installation", "from agent.rsi.engine.model_policy import read_model_configuration"),
    ("adapters", "from .kimi_adapter import KimiAdapter"),
])
def test_dispatch_and_shared_contract_imports_remain_valid(source, statement):
    path = ROOT / f"src/agent/harness/{source}.py"
    assert not checks.harness_worker_import_errors(path, ast.parse(statement), ROOT)


def write_dataset_registry(root, reference):
    (root / "src").mkdir()
    (root / "src/constant.json").write_text(json.dumps({"dataset_configs": {"Example": reference}}))


@pytest.mark.parametrize("reference", ["data/missing.json", "data/config", "../outside.json", ""])
def test_dataset_registry_rejects_unusable_config_paths(tmp_path, reference):
    write_dataset_registry(tmp_path, reference)
    errors = checks.dataset_config_errors(tmp_path)
    assert len(errors) == 1
    assert "dataset_configs.Example" in errors[0]


@pytest.mark.parametrize("content", ["not json", "[]", "null"])
def test_dataset_registry_rejects_invalid_config_documents(tmp_path, content):
    write_dataset_registry(tmp_path, "data/preset.json")
    (tmp_path / "data").mkdir()
    (tmp_path / "data/preset.json").write_text(content)
    assert checks.dataset_config_errors(tmp_path)


def test_dataset_registry_accepts_nested_shipped_config(tmp_path):
    write_dataset_registry(tmp_path, "data/example/preset.json")
    (tmp_path / "data/example").mkdir(parents=True)
    (tmp_path / "data/example/preset.json").write_text('{"dataset": "public/example"}')
    assert checks.dataset_config_errors(tmp_path) == []


def test_shipped_dataset_registry_has_no_broken_references():
    assert checks.dataset_config_errors(ROOT) == []
