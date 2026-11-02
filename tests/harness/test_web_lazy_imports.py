"""Path utilities must not load UI dependencies in scientific processes."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("prefix", ["mcp_tools.common"])
def test_path_helpers_and_package_exports_do_not_load_gradio(prefix):
    root = Path(__file__).resolve().parents[2]
    script = """
import importlib,sys,types
package=importlib.import_module(PREFIX)
common=importlib.import_module(PREFIX+'.common_utils')
assert 'gradio' not in sys.modules
assert package.get_save_path is common.get_save_path
assert package.sanitize_filename('a/b')=='a'
assert 'gradio' not in sys.modules
""".replace("PREFIX", repr(prefix))
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src") + os.pathsep + str(root)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
