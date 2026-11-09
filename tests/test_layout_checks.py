"""Prevent shared contracts from depending on executable Harness adapters."""

import ast
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("layout_check", ROOT / "scripts/check_layout.py")
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


@pytest.mark.parametrize("source", ["events", "session_boundary", "web_native_worker", "dsh_worker"])
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
])
def test_dispatch_and_shared_contract_imports_remain_valid(source, statement):
    path = ROOT / f"src/agent/harness/{source}.py"
    assert not checks.harness_worker_import_errors(path, ast.parse(statement), ROOT)
