"""Tests for sandbox executor."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from agent.sandbox import (
    Capability,
    ExecResult,
    PathGrant,
    SandboxConfig,
    SandboxExecutor,
    DEFAULT_SANDBOX_CONFIG,
    RESTRICTED_SANDBOX_CONFIG,
)
from exceptions import ToolValidationError, ToolTimeoutError


class TestSandboxConfig:
    def test_defaults(self):
        cfg = SandboxConfig()
        assert cfg.timeout_seconds == 120
        assert Capability.READ_FILES in cfg.capabilities

    def test_invalid_timeout(self):
        with pytest.raises(ValueError, match="timeout_seconds"):
            SandboxConfig(timeout_seconds=0)

    def test_restricted_config(self):
        assert RESTRICTED_SANDBOX_CONFIG.timeout_seconds == 60
        assert Capability.NETWORK not in RESTRICTED_SANDBOX_CONFIG.capabilities


class TestPathGrant:
    def test_resolved(self):
        g = PathGrant(path="/tmp")
        assert g.resolved.is_absolute()


class TestExecResult:
    def test_success_output(self):
        r = ExecResult(success=True, stdout="hello", stderr="")
        assert r.output == "hello"

    def test_failure_output(self):
        r = ExecResult(success=False, stdout="", stderr="error msg")
        assert r.output == "error msg"


class TestSandboxExecutor:
    def test_validate_safe_code(self):
        executor = SandboxExecutor()
        ok, msg = executor.validate_code("import numpy as np\nprint(np.pi)")
        assert ok
        assert msg == ""

    def test_validate_forbidden_os_system(self):
        executor = SandboxExecutor()
        ok, msg = executor.validate_code("os.system('ls')")
        assert not ok
        assert "os.system" in msg

    def test_validate_forbidden_subprocess(self):
        executor = SandboxExecutor()
        ok, msg = executor.validate_code("subprocess.run(['ls'])")
        assert not ok
        assert "subprocess" in msg

    def test_validate_forbidden_eval(self):
        executor = SandboxExecutor()
        ok, msg = executor.validate_code("result = eval('1+1')")
        assert not ok
        assert "eval" in msg

    def test_validate_forbidden_exec(self):
        executor = SandboxExecutor()
        ok, msg = executor.validate_code("exec('print(1)')")
        assert not ok
        assert "exec" in msg

    def test_validate_subprocess_allowed_with_capability(self):
        cfg = SandboxConfig(capabilities=frozenset({Capability.SUBPROCESS}))
        executor = SandboxExecutor(cfg)
        ok, msg = executor.validate_code("subprocess.run(['ls'])")
        assert ok

    def test_execute_simple_code(self):
        executor = SandboxExecutor()
        result = executor.execute("print('hello world')")
        assert result.success
        assert "hello world" in result.stdout
        assert result.return_code == 0
        assert result.duration_seconds > 0

    def test_execute_math(self):
        executor = SandboxExecutor()
        result = executor.execute("print(2 + 3)")
        assert result.success
        assert "5" in result.stdout

    def test_execute_failure(self):
        executor = SandboxExecutor()
        result = executor.execute("raise ValueError('test error')")
        assert not result.success
        assert result.return_code != 0
        assert "ValueError" in result.stderr

    def test_execute_validation_fails(self):
        executor = SandboxExecutor()
        with pytest.raises(ToolValidationError, match="os.system"):
            executor.execute("os.system('ls')")

    def test_execute_timeout(self):
        cfg = SandboxConfig(timeout_seconds=1)
        executor = SandboxExecutor(cfg)
        with pytest.raises(ToolTimeoutError):
            executor.execute("import time; time.sleep(10)")

    def test_execute_output_truncation(self):
        cfg = SandboxConfig(max_output_bytes=100)
        executor = SandboxExecutor(cfg)
        result = executor.execute("print('A' * 1000)")
        assert result.truncated

    def test_restricted_env_no_network(self):
        executor = SandboxExecutor(RESTRICTED_SANDBOX_CONFIG)
        env = executor._build_env()
        assert env.get("no_proxy") == "*"
