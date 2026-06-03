"""Headless PyMOL runner.

Executes user-supplied PyMOL scripts (`.py` or `.pml`) via the `pymol -cq` CLI,
which renders with OSMesa (no display, no GPU required). Captures stdout/stderr
and returns the path of any PNG/PSE output the script produced.

If `pymol` is not on PATH, returns a structured error directing the user to the
install instructions.

Adapted from google-deepmind/science-skills (Apache-2.0):
- skills/pymol/SKILL.md (the script execution pattern)
"""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

PYMOL_INSTALL_HINT = (
    "PyMOL is required. Install one of:\n"
    "  • conda: `conda install -n venus -c conda-forge pymol-open-source`\n"
    "  • pip:   `pip install pymol-open-source-whl` (3.10 ≤ python < 3.13)\n"
    "Then verify with `pymol -V`."
)


def is_pymol_available() -> Tuple[bool, str]:
    """Return (available, hint). hint is the install message if unavailable."""
    if shutil.which("pymol"):
        return True, "pymol"
    return False, PYMOL_INSTALL_HINT


def _resolve_outputs(out_dir: str, declared: List[str]) -> List[str]:
    found = []
    for p in declared:
        candidate = p if os.path.isabs(p) else os.path.join(out_dir, p)
        if os.path.exists(candidate):
            found.append(candidate)
    if not found:
        for ext in (".png", ".pse", ".pdb", ".cif"):
            for p in Path(out_dir).glob(f"*{ext}"):
                found.append(str(p))
    return sorted(set(found))


def run_pymol_script(
    script_source: str,
    out_dir: str,
    expected_outputs: List[str] = None,
    timeout_secs: int = 5 * 60,
    extra_env: Dict[str, str] = None,
) -> Dict[str, object]:
    """Write `script_source` to a temp .py, run `pymol -cq`, return summary.

    Returns dict: {
        "success": bool,
        "stdout": str, "stderr": str, "returncode": int,
        "script_path": str, "outputs": list[str], "error": str | None,
    }
    """
    out_dir = str(out_dir or "").strip().rstrip(os.sep)
    if not out_dir:
        return {"success": False, "stdout": "", "stderr": "", "returncode": -1,
                "script_path": "", "outputs": [], "error": "empty out_dir"}
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    ok, hint = is_pymol_available()
    if not ok:
        return {"success": False, "stdout": "", "stderr": "", "returncode": -1,
                "script_path": "", "outputs": [], "error": hint}

    script_dir = tempfile.mkdtemp(prefix="pymol_script_", dir=out_dir)
    script_path = os.path.join(script_dir, "render.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_source)

    env = os.environ.copy()
    env.setdefault("PYOPENGL_PLATFORM", "osmesa")
    if extra_env:
        env.update(extra_env)

    try:
        proc = subprocess.run(
            ["pymol", "-cq", script_path],
            capture_output=True,
            text=True,
            timeout=timeout_secs,
            env=env,
            cwd=out_dir,
        )
    except subprocess.TimeoutExpired as e:
        return {"success": False, "stdout": e.stdout or "", "stderr": e.stderr or "",
                "returncode": -1, "script_path": script_path, "outputs": [],
                "error": f"pymol timed out after {timeout_secs}s"}
    except FileNotFoundError:
        return {"success": False, "stdout": "", "stderr": "", "returncode": -1,
                "script_path": script_path, "outputs": [], "error": PYMOL_INSTALL_HINT}

    outputs = _resolve_outputs(out_dir, expected_outputs or [])
    return {
        "success": proc.returncode == 0,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "returncode": proc.returncode,
        "script_path": script_path,
        "outputs": outputs,
        "error": None if proc.returncode == 0 else (proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else f"pymol exited with code {proc.returncode}"),
    }


__all__ = ["is_pymol_available", "run_pymol_script", "PYMOL_INSTALL_HINT"]
