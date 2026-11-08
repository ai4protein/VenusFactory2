"""Run ESM3 preprocessing without importing its incompatible stack into Factory."""
from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile

from mcp_tools.models import execution
from mcp_tools.models.profiles import load_runtime_settings


def encode_structures(pdb_files, device="cpu"):
    paths = [str(Path(path).absolute()) for path in pdb_files]
    if not paths or any(not Path(path).is_file() for path in paths):
        raise ValueError("ESM3 needs at least one existing PDB file")
    settings = load_runtime_settings()
    cfg = settings.get("esm3", {})
    if not cfg.get("python"):
        raise ValueError("Install the ESM3 runtime with python initialize.py install --tools esm3")
    if cfg.get("enabled") is not True:
        raise ValueError("ESM3 runtime is disabled")
    weights = Path(cfg.get("weights") or "")
    if not weights.is_file():
        raise ValueError("Configure the ESM3 structure-encoder checkpoint in the runtime profile")
    with tempfile.TemporaryDirectory(prefix="venus-esm3-") as folder:
        manifest = Path(folder) / "inputs.json"
        output = Path(folder) / "tokens.json"
        manifest.write_text(json.dumps(paths), encoding="utf-8")
        spec = execution.command("esm3", [
            "-m", "mcp_tools.models.factory.training.data.get_esm3_structure_seq",
            "--runtime-worker", "--pdb_manifest", manifest, "--weights", weights,
            "--device", device, "--out_file", output,
        ], settings=settings)
        execution.execute(spec)
        if not output.is_file() or output.stat().st_size > 64 * 2**20:
            raise ValueError("ESM3 returned missing or oversized results")
        rows = json.loads(output.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != len(paths):
        raise ValueError("ESM3 result count does not match the input structures")
    for row, path in zip(rows, paths):
        if not isinstance(row, dict) or row.get("name") != Path(path).name.split(".")[0]:
            raise ValueError("ESM3 result does not match the requested structure")
        tokens = row.get("esm3_structure_seq")
        if not isinstance(tokens, list) or not tokens or any(type(x) is not int or not 0 <= x < 4096 for x in tokens):
            raise ValueError("ESM3 returned invalid structure tokens")
        confidence = row.get("plddt")
        if type(confidence) not in (float, int) or not math.isfinite(confidence):
            raise ValueError("ESM3 returned invalid confidence")
    return rows
