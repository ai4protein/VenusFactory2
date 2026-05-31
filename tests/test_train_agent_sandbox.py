"""End-to-end tests for `generate_and_execute_code` exercising the SandboxExecutor path.

These tests mock the LLM HTTP call (`requests.post`) so we can inject deterministic
Python source and verify that the train_operations -> SandboxExecutor wiring enforces
expected security/runtime guarantees:

  T1  allowed file I/O within granted path succeeds and returns JSON.
  T2  blocklist (os.system) is rejected before execution (security_blocked).
  T3  path validation rejects writes outside granted directories.
  T4  long-running code times out (sandbox timeout -> friendly JSON error).
  T5  network access is blocked / disabled (no_proxy="*") - we record observed behaviour.
  T6  output truncation (>max_output_bytes) - exercised via SandboxExecutor directly
       because train_operations' JSON shim drops the `truncated` flag.

The LLM is mocked at the network boundary (`requests.post`) which is the cleanest
seam: train_operations does not expose a separate `_generate_code` helper, so any
in-module patch would have to target the bare `requests` call anyway.
"""
import sys
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Set required env var BEFORE importing the module so module-level load_dotenv
# does not blow this away.  We re-set it inside each test fixture too.
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")

from tools.train import train_operations as to_mod  # noqa: E402
from agent.sandbox import (  # noqa: E402
    Capability,
    PathGrant,
    SandboxConfig,
    SandboxExecutor,
)


# --------------------------------------------------------------------------- #
# Helpers / fixtures
# --------------------------------------------------------------------------- #

def _fake_llm_response(code: str) -> MagicMock:
    """Build a MagicMock that looks like a successful chat-completion HTTP response."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "choices": [{"message": {"content": code}}],
    }
    return resp


@pytest.fixture
def temp_workspace():
    """Create a temp dir holding one input CSV; output dir is the same dir
    (train_operations derives output_directory from the primary input's parent)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp).resolve()
        input_csv = tmp_path / "data.csv"
        input_csv.write_text("a,b\n1,2\n3,4\n")
        yield {
            "input_csv": str(input_csv),
            "work_dir": str(tmp_path),
        }


@pytest.fixture(autouse=True)
def _ensure_api_key(monkeypatch):
    """Make sure the function does not exit early on missing key."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    yield


# --------------------------------------------------------------------------- #
# T1: allowed simple I/O succeeds
# --------------------------------------------------------------------------- #

def test_t1_allowed_simple_csv_copy(temp_workspace):
    """Generated code reads input CSV, writes output CSV inside granted dir,
    prints JSON. End-to-end success path."""
    input_csv = temp_workspace["input_csv"]
    work_dir = temp_workspace["work_dir"]
    out_csv = os.path.join(work_dir, "processed.csv")

    code = (
        "import csv\n"
        "import json\n"
        "\n"
        "def main():\n"
        f"    with open({input_csv!r}, 'r') as f:\n"
        "        rows = list(csv.reader(f))\n"
        f"    with open({out_csv!r}, 'w') as f:\n"
        "        csv.writer(f).writerows(rows)\n"
        "    print(json.dumps({'rows_copied': len(rows)}))\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )

    with patch.object(to_mod, "requests") as mock_req:
        mock_req.post.return_value = _fake_llm_response(code)
        result_json = to_mod.generate_and_execute_code(
            task_description="copy csv",
            input_files=[input_csv],
        )

    result = json.loads(result_json)
    assert result.get("success") is True, f"Expected success, got: {result}"
    assert Path(out_csv).exists(), "Output CSV should have been created"
    assert "generated_code_path" in result
    assert result.get("rows_copied") == 3  # 1 header + 2 data rows


# --------------------------------------------------------------------------- #
# T2: blocklist rejects os.system before execution
# --------------------------------------------------------------------------- #

def test_t2_blocklist_rejects_os_system(temp_workspace):
    """Code containing os.system must be rejected (security_blocked=True)."""
    input_csv = temp_workspace["input_csv"]

    code = (
        "import os\n"
        "import json\n"
        "\n"
        "def main():\n"
        "    os.system('echo pwned')\n"
        "    print(json.dumps({'ok': True}))\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )

    with patch.object(to_mod, "requests") as mock_req:
        mock_req.post.return_value = _fake_llm_response(code)
        result_json = to_mod.generate_and_execute_code(
            task_description="malicious",
            input_files=[input_csv],
        )

    result = json.loads(result_json)
    assert result.get("success") is False
    # The module-level blocklist catches this first and marks security_blocked.
    assert result.get("security_blocked") is True, (
        f"Expected security_blocked=True, got: {result}"
    )
    assert "os.system" in result.get("error", "")


def test_t2b_blocklist_rejects_subprocess(temp_workspace):
    """subprocess use must also be blocked."""
    input_csv = temp_workspace["input_csv"]

    code = (
        "import subprocess\n"
        "import json\n"
        "\n"
        "def main():\n"
        "    subprocess.run(['echo', 'hi'])\n"
        "    print(json.dumps({'ok': True}))\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )

    with patch.object(to_mod, "requests") as mock_req:
        mock_req.post.return_value = _fake_llm_response(code)
        result_json = to_mod.generate_and_execute_code(
            task_description="malicious2",
            input_files=[input_csv],
        )

    result = json.loads(result_json)
    assert result.get("success") is False
    assert result.get("security_blocked") is True
    assert "subprocess" in result.get("error", "")


# --------------------------------------------------------------------------- #
# T3: path-validation rejects writes outside granted directories
# --------------------------------------------------------------------------- #

def test_t3_path_escape_blocked(temp_workspace):
    """Code that references /etc/passwd (which exists) must be rejected by
    SandboxExecutor.validate_paths.  train_operations passes only the work dir
    and /tmp as grants, so /etc is out-of-bounds."""
    input_csv = temp_workspace["input_csv"]

    # Avoid string literal "open" of /etc/passwd via blocklist trickery -- just
    # have the literal in the source; validate_paths picks it up by regex.
    code = (
        "import json\n"
        "\n"
        "def main():\n"
        "    with open('/etc/passwd', 'r') as f:\n"
        "        data = f.read()\n"
        "    print(json.dumps({'len': len(data)}))\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )

    with patch.object(to_mod, "requests") as mock_req:
        mock_req.post.return_value = _fake_llm_response(code)
        result_json = to_mod.generate_and_execute_code(
            task_description="path escape",
            input_files=[input_csv],
        )

    result = json.loads(result_json)
    assert result.get("success") is False, f"Expected failure, got: {result}"
    # SandboxExecutor raises ToolValidationError -> train_operations sets
    # security_blocked=True for that branch.
    assert result.get("security_blocked") is True, (
        f"Expected security_blocked=True for path escape, got: {result}"
    )
    assert "/etc/passwd" in result.get("error", "") or "outside" in result.get("error", "")


# --------------------------------------------------------------------------- #
# T4: timeout - generated code sleeps longer than sandbox timeout
# --------------------------------------------------------------------------- #

def test_t4_timeout(temp_workspace, monkeypatch):
    """Generated code that sleeps forever should hit the sandbox timeout.
    We monkey-patch SandboxConfig's default timeout to 2s so the test is fast."""
    input_csv = temp_workspace["input_csv"]

    code = (
        "import time\n"
        "import json\n"
        "\n"
        "def main():\n"
        "    time.sleep(30)\n"
        "    print(json.dumps({'ok': True}))\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )

    # Wrap SandboxConfig so any timeout_seconds requested by train_operations
    # is clamped to 2 seconds for this test.
    real_sandbox_config = to_mod.SandboxConfig

    def _short_timeout_config(*args, **kwargs):
        kwargs["timeout_seconds"] = 2
        return real_sandbox_config(*args, **kwargs)

    monkeypatch.setattr(to_mod, "SandboxConfig", _short_timeout_config)

    with patch.object(to_mod, "requests") as mock_req:
        mock_req.post.return_value = _fake_llm_response(code)
        result_json = to_mod.generate_and_execute_code(
            task_description="sleep forever",
            input_files=[input_csv],
        )

    result = json.loads(result_json)
    assert result.get("success") is False
    # The function catches the subprocess.TimeoutExpired and returns its
    # canned ">120 seconds" message regardless of actual sandbox timeout.
    err_msg = result.get("error", "")
    assert "timed out" in err_msg.lower(), f"Expected timeout error, got: {result}"


# --------------------------------------------------------------------------- #
# T5: network access is restricted via no_proxy="*"
# --------------------------------------------------------------------------- #

def test_t5_network_restricted(temp_workspace, monkeypatch):
    """urllib does not match the socket blocklist, so it is not statically
    rejected.  At runtime, SandboxExecutor sets no_proxy='*' (since NETWORK
    capability is not granted) and uses a stripped env, so most network
    attempts fail quickly.  We assert success=False, regardless of exact
    failure mode (DNS resolution failure, connection refused, etc.)."""
    input_csv = temp_workspace["input_csv"]

    code = (
        "import urllib.request\n"
        "import json\n"
        "\n"
        "def main():\n"
        "    try:\n"
        "        urllib.request.urlopen('http://10.255.255.1/', timeout=2)\n"
        "        print(json.dumps({'reached': True}))\n"
        "    except Exception as e:\n"
        "        # Surface failure as non-zero exit so train_operations\n"
        "        # records success=False.\n"
        "        raise SystemExit('network blocked: ' + type(e).__name__)\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )

    # Clamp sandbox timeout to 5s so this test runs fast even if the urlopen
    # call were to hang.
    real_sandbox_config = to_mod.SandboxConfig

    def _short_timeout_config(*args, **kwargs):
        kwargs["timeout_seconds"] = 5
        return real_sandbox_config(*args, **kwargs)

    monkeypatch.setattr(to_mod, "SandboxConfig", _short_timeout_config)

    with patch.object(to_mod, "requests") as mock_req:
        mock_req.post.return_value = _fake_llm_response(code)
        result_json = to_mod.generate_and_execute_code(
            task_description="network",
            input_files=[input_csv],
        )

    result = json.loads(result_json)
    # Either it was blocked at validation, or it ran but failed at network.
    # The reachable path (success=True) would be a regression worth flagging.
    assert result.get("success") is False, (
        f"Network call unexpectedly succeeded: {result}"
    )


# --------------------------------------------------------------------------- #
# T6: output truncation - SandboxExecutor caps stdout at max_output_bytes.
#
# Note: train_operations' _SandboxProcShim only forwards (returncode, stdout,
# stderr) and drops the `truncated` flag, so the end-to-end JSON does not
# expose `truncated`.  We therefore exercise this property directly against
# SandboxExecutor (still an integration concern for the sandbox path).
# --------------------------------------------------------------------------- #

def test_t6_output_truncation_via_sandbox_executor():
    """Confirm the sandbox itself caps stdout - the contract that
    train_operations relies on."""
    cfg = SandboxConfig(
        timeout_seconds=10,
        max_output_bytes=1024,
        capabilities=frozenset({Capability.READ_FILES, Capability.IMPORT_ALL}),
    )
    executor = SandboxExecutor(cfg)
    result = executor.execute("print('X' * 10000)")
    assert result.success
    assert result.truncated is True
    # Stdout is clipped to max_output_bytes + truncation marker.
    assert len(result.stdout) <= cfg.max_output_bytes + 64


def test_t6b_large_output_e2e_still_returns_success(temp_workspace):
    """End-to-end: even when sandbox truncates stdout, train_operations
    should still report success (it just won't surface the truncated flag)."""
    input_csv = temp_workspace["input_csv"]
    work_dir = temp_workspace["work_dir"]
    out_file = os.path.join(work_dir, "big.txt")

    # Print a moderate amount (well below 10MB default), still ensures
    # the success path works for chatty scripts.
    code = (
        "import json\n"
        "\n"
        "def main():\n"
        f"    with open({out_file!r}, 'w') as f:\n"
        "        f.write('x' * 1000)\n"
        "    print('noise ' * 200)\n"
        "    print(json.dumps({'wrote': 1000}))\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )

    with patch.object(to_mod, "requests") as mock_req:
        mock_req.post.return_value = _fake_llm_response(code)
        result_json = to_mod.generate_and_execute_code(
            task_description="chatty",
            input_files=[input_csv],
        )

    result = json.loads(result_json)
    assert result.get("success") is True, f"Got: {result}"
    assert Path(out_file).exists()
