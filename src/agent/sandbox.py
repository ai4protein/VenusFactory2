"""
Sandbox executor for VenusFactory code execution tools.

Provides capability-based security model with:
  - Subprocess isolation with restricted environment
  - Resource limits (timeout, output size)
  - Structured ExecResult
  - Path-level access grants
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from exceptions import ErrorContext, ToolExecutionError, ToolTimeoutError

# ---------------------------------------------------------------------------
# Capabilities and grants
# ---------------------------------------------------------------------------

class Capability(Enum):
    """Fine-grained permissions for sandboxed execution."""

    READ_FILES = "read_files"
    WRITE_FILES = "write_files"
    NETWORK = "network"
    SUBPROCESS = "subprocess"
    IMPORT_ALL = "import_all"


@dataclass(frozen=True)
class PathGrant:
    """Grants read/write access to a specific directory."""

    path: str
    read: bool = True
    write: bool = False

    @property
    def resolved(self) -> Path:
        return Path(self.path).expanduser().resolve()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SandboxConfig:
    """Configuration for sandboxed code execution."""

    timeout_seconds: int = 120
    max_output_bytes: int = 10 * 1024 * 1024  # 10 MB
    capabilities: frozenset[Capability] = frozenset({Capability.READ_FILES, Capability.IMPORT_ALL})
    path_grants: tuple[PathGrant, ...] = ()
    python_executable: str = ""
    working_directory: str = ""
    extra_env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timeout_seconds < 1:
            raise ValueError(f"timeout_seconds must be >= 1, got {self.timeout_seconds}")
        if self.max_output_bytes < 0:
            raise ValueError(f"max_output_bytes must be >= 0, got {self.max_output_bytes}")


DEFAULT_SANDBOX_CONFIG = SandboxConfig()

RESTRICTED_SANDBOX_CONFIG = SandboxConfig(
    timeout_seconds=60,
    max_output_bytes=5 * 1024 * 1024,
    capabilities=frozenset({Capability.READ_FILES}),
)


# ---------------------------------------------------------------------------
# Exec result
# ---------------------------------------------------------------------------

@dataclass
class ExecResult:
    """Structured result of sandboxed code execution."""

    success: bool
    stdout: str = ""
    stderr: str = ""
    return_code: int = -1
    timed_out: bool = False
    truncated: bool = False
    duration_seconds: float = 0.0
    script_path: str = ""

    @property
    def output(self) -> str:
        return self.stdout if self.success else self.stderr or self.stdout


# ---------------------------------------------------------------------------
# Sandbox executor
# ---------------------------------------------------------------------------

class SandboxExecutor:
    """Execute Python code in a restricted subprocess."""

    def __init__(self, config: SandboxConfig | None = None):
        self._config = config or DEFAULT_SANDBOX_CONFIG

    @property
    def config(self) -> SandboxConfig:
        return self._config

    def validate_code(self, code: str) -> tuple[bool, str]:
        """Static analysis: check for forbidden patterns before execution."""
        import re

        forbidden = [
            (r"\bos\.system\b", "os.system"),
            (r"\bsubprocess\.(?:run|Popen|call|check_output)\b", "subprocess"),
            (r"\b__import__\b", "__import__"),
            (r"\beval\s*\(", "eval()"),
            (r"\bexec\s*\(", "exec()"),
            (r"\bshutil\.rmtree\b", "shutil.rmtree"),
            (r"\bos\.remove\b", "os.remove"),
            (r"\bos\.unlink\b", "os.unlink"),
            (r"\bos\.rmdir\b", "os.rmdir"),
            (r"\bsocket\.\w+", "socket"),
            (r"\bos\.environ\[", "os.environ write"),
        ]

        if Capability.SUBPROCESS in self._config.capabilities:
            forbidden = [(p, n) for p, n in forbidden if "subprocess" not in n]

        for pattern, name in forbidden:
            match = re.search(pattern, code, re.IGNORECASE)
            if match:
                return False, f"Forbidden pattern: {name} at position {match.start()}"

        return True, ""

    def validate_paths(self, code: str) -> tuple[bool, str]:
        """Check that file paths in code are within granted directories."""
        if not self._config.path_grants:
            return True, ""

        import re
        path_literals = re.findall(r'["\']([/~][^"\']+)["\']', code)

        for path_str in path_literals:
            resolved = Path(path_str).expanduser().resolve()
            allowed = False
            for grant in self._config.path_grants:
                grant_path = grant.resolved
                try:
                    resolved.relative_to(grant_path)
                    allowed = True
                    break
                except ValueError:
                    continue
            if not allowed and resolved.exists():
                return False, f"Path '{path_str}' is outside granted directories"

        return True, ""

    def _build_env(self) -> dict[str, str]:
        """Build restricted environment for subprocess."""
        env: dict[str, str] = {}

        safe_keys = {
            "PATH", "HOME", "USER", "LANG", "LC_ALL", "PYTHONPATH",
            "CONDA_DEFAULT_ENV", "CONDA_PREFIX", "VIRTUAL_ENV",
            "CUDA_VISIBLE_DEVICES", "HF_ENDPOINT", "TRANSFORMERS_CACHE",
        }
        for key in safe_keys:
            val = os.environ.get(key)
            if val is not None:
                env[key] = val

        env["PYTHONDONTWRITEBYTECODE"] = "1"

        if Capability.NETWORK not in self._config.capabilities:
            env["no_proxy"] = "*"

        env.update(self._config.extra_env)
        return env

    def execute(self, code: str, *, context: ErrorContext | None = None) -> ExecResult:
        """Execute Python code in a sandboxed subprocess.

        Raises:
            ToolValidationError: If code validation fails.
            ToolTimeoutError: If execution exceeds timeout.
            ToolExecutionError: If execution fails for other reasons.
        """
        import time

        is_valid, msg = self.validate_code(code)
        if not is_valid:
            from exceptions import ToolValidationError
            raise ToolValidationError(msg, tool_name="sandbox", context=context)

        path_ok, path_msg = self.validate_paths(code)
        if not path_ok:
            from exceptions import ToolValidationError
            raise ToolValidationError(path_msg, tool_name="sandbox", context=context)

        python_exe = self._config.python_executable or sys.executable
        work_dir = self._config.working_directory or None

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir=work_dir
        ) as f:
            f.write(code)
            script_path = f.name

        env = self._build_env()
        start_time = time.monotonic()

        try:
            proc = subprocess.run(
                [python_exe, script_path],
                capture_output=True,
                text=True,
                timeout=self._config.timeout_seconds,
                env=env,
                cwd=work_dir,
            )
            duration = time.monotonic() - start_time

            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            truncated = False

            if len(stdout.encode()) > self._config.max_output_bytes:
                stdout = stdout[: self._config.max_output_bytes] + "\n... [output truncated]"
                truncated = True

            return ExecResult(
                success=proc.returncode == 0,
                stdout=stdout,
                stderr=stderr,
                return_code=proc.returncode,
                duration_seconds=duration,
                script_path=script_path,
                truncated=truncated,
            )

        except subprocess.TimeoutExpired as e:
            duration = time.monotonic() - start_time
            raise ToolTimeoutError(
                f"Code execution timed out after {self._config.timeout_seconds}s",
                tool_name="sandbox",
                timeout_seconds=self._config.timeout_seconds,
                context=context,
            ) from e

        except Exception as e:
            duration = time.monotonic() - start_time
            raise ToolExecutionError(
                f"Code execution failed: {e}",
                tool_name="sandbox",
                context=context,
            ) from e

        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass
