"""Isolated execution for agent-authored model code; independent of LLM runtime.

Only Python/packages, the code, one JSON request and an output directory are
mounted. Host repository, credentials and evaluator labels are never mounted.
OS namespaces remove networking. AST screening is an additional early check,
not the security boundary. Resource limits are set in the trusted worker.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

try:
    from .protein_references import require_reference_parity
except ImportError:  # Read-only standalone capability bundle.
    from protein_references import require_reference_parity

ALLOWED_IMPORTS = {"numpy", "sklearn", "math", "itertools", "collections", "statistics", "json"}
FORBIDDEN_CALLS = {
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "input",
    "breakpoint",
}


_EXECUTION_RECORDS = ContextVar("numeric_execution_records", default=None)


@contextmanager
def capture_code_runs():
    """Capture the exact sandbox calls made by one trusted assessment."""
    if _EXECUTION_RECORDS.get() is not None:
        raise RuntimeError("Nested numeric execution capture is unsupported")
    records = []
    token = _EXECUTION_RECORDS.set(records)
    try:
        yield records
    finally:
        _EXECUTION_RECORDS.reset(token)


def _json_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class CodeExecutionError(RuntimeError):
    """Infrastructure or unclassified isolated-execution failure; never repair silently."""


class CandidateCodeError(CodeExecutionError):
    """Candidate code started but failed within the isolated worker."""


class ModelOutputError(ValueError):
    """Candidate predictions violate the task's output ABI."""


def guard_source(source: str) -> dict:
    if not isinstance(source, str) or not 1 <= len(source.encode()) <= 60000:
        raise ValueError("model code must contain 1..60000 bytes")
    tree = ast.parse(source)
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    if not {"fit", "predict"} <= functions:
        raise ValueError("required ABI: fit(sequences, targets, seed), predict(model, sequences)")
    definitions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name, count in (("fit", 3), ("predict", 2)):
        node = definitions[name]
        if len(node.args.args) != count or node.args.vararg or node.args.kwarg:
            raise ValueError("ABI requires fit(sequences, targets, seed) and predict(model, sequences)")
        substantive = [
            item
            for item in node.body
            if not isinstance(item, ast.Pass)
            and not (isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant))
        ]
        if not substantive:
            raise ValueError("model functions must implement computation; placeholder bodies are invalid")
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            if node.level:
                raise ValueError("relative imports are unavailable")
            imports.add((node.module or "").split(".")[0])
        if isinstance(node, ast.Name) and (node.id in FORBIDDEN_CALLS or node.id.startswith("__")):
            raise ValueError("code uses a forbidden dynamic or I/O operation")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("private Python introspection is unavailable")
    if imports - ALLOWED_IMPORTS:
        raise ValueError("unsupported imports: " + ", ".join(sorted(imports - ALLOWED_IMPORTS)))
    return {
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "imports": sorted(imports),
        "functions": sorted(functions),
        "reference_audit": require_reference_parity(source),
    }


def implementation_hash() -> str:
    root = Path(__file__).resolve().parent
    payload = b"".join(
        (root / name).read_bytes()
        for name in [
            "generated_model.py",
            "code_sandbox.py",
            "code_worker.py",
            "protein_references.py",
            "numeric_runtime.py",
        ]
    )
    return hashlib.sha256(payload).hexdigest()


def run_code(source: str, request: dict, *, timeout_s: float = 120) -> dict:
    guard_source(source)
    if request.get("operation") not in {"fit", "predict"}:
        raise ValueError("unknown model operation")
    if request.get("input_kind", "sequences") not in {"sequences", "features"}:
        raise ValueError("unknown model input kind")
    records = _EXECUTION_RECORDS.get()
    original_request = request
    request = {**request, "record_runtime": records is not None}
    python_root = Path(sys.prefix).resolve()
    worker = Path(__file__).with_name("code_worker.py").resolve()
    with tempfile.TemporaryDirectory(prefix="vf-model-code-") as raw:
        root = Path(raw)
        (root / "model.py").write_text(source)
        (root / "request.json").write_text(json.dumps(request, allow_nan=False))
        output = root / "output"
        output.mkdir()
        command = ["bwrap", "--unshare-all", "--new-session", "--die-with-parent", "--cap-drop", "ALL"]
        for directory in ["/usr", "/bin", "/lib", "/lib64"]:
            if Path(directory).exists():
                command += ["--ro-bind", directory, directory]
        command += [
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--dir",
            "/home",
            "--dir",
            "/opt",
            "--dir",
            "/runner",
            "--dir",
            "/candidate",
            "--dir",
            "/input",
            "--ro-bind",
            str(python_root),
            "/opt/venus-runtime",
            "--ro-bind",
            str(worker),
            "/runner/worker.py",
            "--ro-bind",
            str(worker.with_name("numeric_runtime.py")),
            "/runner/numeric_runtime.py",
            "--ro-bind",
            str(root / "model.py"),
            "/candidate/model.py",
            "--ro-bind",
            str(root / "request.json"),
            "/input/request.json",
            "--bind",
            str(output),
            "/output",
            "--chdir",
            "/output",
            "--clearenv",
        ]
        for key, value in {
            "HOME": "/home",
            "PATH": "/opt/venus-runtime/bin:/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
        }.items():
            command += ["--setenv", key, value]
        command += ["--", "/opt/venus-runtime/bin/python", "-I", "/runner/worker.py"]

        def candidate_started():
            marker = output / "worker_started"
            return (
                marker.is_file() and not marker.is_symlink() and marker.read_text() == "candidate_execution_started\n"
            )

        # Generated stdout is not part of the JSON ABI. A regular stderr file is
        # bounded by the worker's existing RLIMIT_FSIZE, unlike an unbounded PIPE.
        with (root / "stderr.log").open("wb") as error_log:
            proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=error_log, start_new_session=True)
            try:
                proc.communicate(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                import signal

                os.killpg(proc.pid, signal.SIGKILL)
                proc.communicate()
                error = CandidateCodeError if candidate_started() else CodeExecutionError
                raise error("generated model exceeded wall-clock budget") from None
        diagnostics = (root / "stderr.log").read_bytes()[-3000:].decode(errors="replace")
        result_file = output / "result.json"
        error = CandidateCodeError if candidate_started() else CodeExecutionError
        if records is not None and (output / "runtime_failed").exists():
            raise CodeExecutionError("Numeric runtime evidence collection failed")
        if proc.returncode or not result_file.is_file():
            raise error(f"generated model failed ({proc.returncode}): {diagnostics}")
        if result_file.is_symlink() or result_file.stat().st_size > 16 * 1024**2:
            raise error("invalid output artifact")
        try:
            result = json.loads(result_file.read_text())
            json.dumps(result, allow_nan=False)
        except (ValueError, TypeError) as exc:
            raise error("invalid generated JSON artifact") from exc
        if isinstance(result, dict) and set(result) == {"candidate_error"} and candidate_started():
            detail = result["candidate_error"]
            if not isinstance(detail, dict) or set(detail) != {"type", "message", "traceback"}:
                raise CodeExecutionError("invalid worker error record")
            raise CandidateCodeError(
                "generated model failed: " + str(detail["type"])[:128] + ": " + str(detail["message"])[:2000]
            )
        expected = "model" if request["operation"] == "fit" else "predictions"
        if not isinstance(result, dict) or set(result) != {expected}:
            raise error("invalid generated JSON ABI")
        if records is not None:
            runtime_file = output / "runtime.json"
            if not runtime_file.is_file() or runtime_file.is_symlink() or runtime_file.stat().st_size > 1024 * 1024:
                raise CodeExecutionError("Missing or invalid numeric runtime evidence")
            runtime = json.loads(runtime_file.read_text())
            if runtime.get("schema") != 1 or not runtime.get("python") or not runtime.get("packages", {}).get("numpy"):
                raise CodeExecutionError("Invalid numeric runtime description")
            records.append(
                {
                    "operation": original_request["operation"],
                    "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    "request_sha256": _json_digest(original_request),
                    "result_sha256": _json_digest(result),
                    "worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
                    "runtime_probe_sha256": hashlib.sha256(
                        worker.with_name("numeric_runtime.py").read_bytes()
                    ).hexdigest(),
                    "runtime": runtime,
                }
            )
        return result
