"""Tool discovery and file jobs must not load model engines or MCP servers."""
import json
import os
from pathlib import Path
import subprocess
import sys


def test_native_catalog_is_transport_and_model_independent(tmp_path):
    root = Path(__file__).resolve().parents[2]
    fasta = tmp_path / "example.fasta"
    fasta.write_text(">example\nMKWVTFISLLFLFSSAYS\n")
    script = """
import importlib.abc, json, sys
class BlockHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "transformers", "langgraph", "langchain", "fastmcp"}:
            raise AssertionError("Unrelated dependency loaded: " + fullname)
sys.meta_path.insert(0, BlockHeavyImports())
from mcp_tools.scheduler.catalog import native_tools
tools = native_tools()
assert len(tools) > 100
assert "proteinmpnn_sequence_design_from_structure" in tools
assert "proteinmpnn_sequence_scoring_from_structure" in tools
result = tools["read_fasta"].invoke({"file_path": sys.argv[1]})
assert "MKWVTFISLLFLFSSAYS" in str(result), result
print(json.dumps({"tools": len(tools), "read_fasta": True}))
"""
    env = dict(os.environ, PYTHONPATH=str(root / "src") + os.pathsep + str(root),
               WEBUI_V2_MODE="local", VENUS_RUNTIME_ROOT=str(tmp_path / "runtime"))
    result = subprocess.run([sys.executable, "-c", script, str(fasta)],
                            cwd=root, env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1])["read_fasta"] is True
