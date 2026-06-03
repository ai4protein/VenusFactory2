"""High-level PyMOL rendering operations.

Two operations:
- render_protein_structure: cartoon-render a single structure with several
  coloring strategies (plddt | bfactor | chain | ss).
- superpose_two_structures: align two structures with `cealign`, render the
  alignment, report RMSD.

Both return the standard VenusFactory rich JSON envelope. The PNG goes into
`out_dir`; the response carries the file path.

Adapted from google-deepmind/science-skills (Apache-2.0):
- skills/pymol/SKILL.md and references/RECIPES.md
"""
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from src.tools.path_sanitizer import to_client_file_path

try:
    from .pymol_runner import run_pymol_script, is_pymol_available, PYMOL_INSTALL_HINT
except ImportError:
    from src.tools.visualize.pymol.pymol_runner import (
        run_pymol_script, is_pymol_available, PYMOL_INSTALL_HINT,
    )

_SOURCE = "PyMOL (open-source, OSMesa)"
_PREVIEW_LEN = 1500

_VALID_COLOR_BY = {"plddt", "bfactor", "chain", "ss"}


def _error_response(error_type: str, message: str, suggestion: Optional[str] = None) -> str:
    out: Dict[str, Any] = {
        "status": "error",
        "error": {"type": error_type, "message": message},
        "file_info": None,
    }
    if suggestion:
        out["error"]["suggestion"] = suggestion
    return json.dumps(out, ensure_ascii=False)


def _success_response(
    image_path: str,
    biological_metadata: Dict[str, Any],
    elapsed_ms: int,
    stdout_preview: str = "",
) -> str:
    path = Path(image_path)
    size = path.stat().st_size if path.exists() else 0
    out: Dict[str, Any] = {
        "status": "success",
        "file_info": {
            "file_path": to_client_file_path(path if path.exists() else image_path),
            "file_name": path.name,
            "file_size": size,
            "format": path.suffix.lstrip(".").lower() or "png",
        },
        "content_preview": stdout_preview[:_PREVIEW_LEN],
        "biological_metadata": biological_metadata,
        "execution_context": {"elapsed_ms": elapsed_ms, "source": _SOURCE},
    }
    return json.dumps(out, ensure_ascii=False)


_RENDER_SCRIPT_TEMPLATE = '''import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import pymol
pymol.pymol_argv = ["pymol", "-cq"]
pymol.finish_launching()
from pymol import cmd

import sys as _sys
_pdb_path = {pdb_path!r}
cmd.load(_pdb_path, "target")
n = cmd.count_atoms("all")
if n == 0:
    print(f"[ERROR] structure {{_pdb_path!r}} loaded zero atoms", flush=True)
    _sys.stdout.flush()
    cmd.quit()

cmd.bg_color("white")
cmd.hide("everything")
cmd.show("cartoon")
{color_block}
cmd.orient()
cmd.set("ray_opaque_background", 1)
cmd.png({png_path!r}, width={width}, height={height}, dpi={dpi})
cmd.save({pse_path!r})
print(f"atoms={{n}}", flush=True)
_sys.stdout.flush()
cmd.quit()
'''

_COLOR_BLOCKS = {
    "plddt": (
        'cmd.spectrum("b", "red_white_blue", "target", minimum=50, maximum=100)\n'
    ),
    "bfactor": (
        'cmd.spectrum("b", "blue_white_red", "target")\n'
    ),
    "chain": (
        'cmd.util.color_chains("target")\n'
    ),
    "ss": (
        'cmd.color("green", "target and ss H")\n'
        'cmd.color("yellow", "target and ss S")\n'
        'cmd.color("gray80", "target and (ss L or ss \\"\\")")\n'
    ),
}


def render_protein_structure(
    pdb_path: str,
    out_dir: str,
    color_by: str = "plddt",
    width: int = 1200,
    height: int = 900,
    dpi: int = 150,
    timeout_secs: int = 5 * 60,
) -> str:
    """Render a single structure to PNG (+ a .pse session file)."""
    t0 = time.perf_counter()
    pdb_path = str(pdb_path or "").strip()
    if not pdb_path:
        return _error_response("ValidationError", "empty pdb_path")
    if not os.path.exists(pdb_path):
        return _error_response("NotFound", f"structure file not found: {pdb_path}")
    color_by = (color_by or "").lower()
    if color_by not in _VALID_COLOR_BY:
        return _error_response(
            "ValidationError",
            f"color_by must be one of {sorted(_VALID_COLOR_BY)}; got {color_by!r}",
        )
    ok, hint = is_pymol_available()
    if not ok:
        return _error_response("DependencyMissing", hint)

    out_dir = str(out_dir or "").strip().rstrip(os.sep)
    if not out_dir:
        return _error_response("ValidationError", "empty out_dir")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    stem = Path(pdb_path).stem
    png_path = os.path.join(out_dir, f"{stem}_{color_by}.png")
    pse_path = os.path.join(out_dir, f"{stem}_{color_by}.pse")
    script = _RENDER_SCRIPT_TEMPLATE.format(
        pdb_path=pdb_path,
        color_block=_COLOR_BLOCKS[color_by],
        png_path=png_path,
        pse_path=pse_path,
        width=width,
        height=height,
        dpi=dpi,
    )

    result = run_pymol_script(script, out_dir, expected_outputs=[png_path, pse_path], timeout_secs=timeout_secs)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    if not result["success"]:
        return _error_response("RenderError", result["error"] or "pymol failed", suggestion=result["stderr"][-500:] if result["stderr"] else None)
    if not os.path.exists(png_path):
        return _error_response("RenderError", "pymol exited 0 but no PNG was produced", suggestion=result["stdout"][-500:])

    meta = {
        "input_structure": os.path.basename(pdb_path),
        "color_by": color_by,
        "png_path": to_client_file_path(png_path),
        "pse_path": to_client_file_path(pse_path) if os.path.exists(pse_path) else None,
        "image_dimensions": {"width": width, "height": height, "dpi": dpi},
    }
    return _success_response(png_path, meta, elapsed_ms, stdout_preview=result["stdout"])


_SUPERPOSE_SCRIPT_TEMPLATE = '''import os
import sys as _sys
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import pymol
pymol.pymol_argv = ["pymol", "-cq"]
pymol.finish_launching()
from pymol import cmd

cmd.load({pdb_a!r}, "mobile")
cmd.load({pdb_b!r}, "ref")
if cmd.count_atoms("mobile") == 0 or cmd.count_atoms("ref") == 0:
    print("[ERROR] one or both structures failed to load", flush=True)
    _sys.stdout.flush()
    cmd.quit()

try:
    res = cmd.cealign("ref", "mobile")
    rmsd = float(res.get("RMSD", res.get("rmsd", -1.0)))
    n_aligned = int(res.get("alignment_length", res.get("alignment", -1) or -1))
except Exception as exc:
    res = cmd.align("mobile", "ref")
    rmsd = float(res[0])
    n_aligned = int(res[1])

cmd.bg_color("white")
cmd.hide("everything")
cmd.show("cartoon")
cmd.color("cyan", "ref")
cmd.color("magenta", "mobile")
cmd.orient()
cmd.set("ray_opaque_background", 1)
cmd.png({png_path!r}, width={width}, height={height}, dpi={dpi})
cmd.save({pse_path!r})
print(f"rmsd={{rmsd:.3f}} aligned={{n_aligned}}", flush=True)
_sys.stdout.flush()
cmd.quit()
'''


def superpose_two_structures(
    pdb_a: str,
    pdb_b: str,
    out_dir: str,
    width: int = 1200,
    height: int = 900,
    dpi: int = 150,
    timeout_secs: int = 5 * 60,
) -> str:
    """Align `pdb_a` (mobile) onto `pdb_b` (reference), render, report RMSD."""
    t0 = time.perf_counter()
    if not pdb_a or not os.path.exists(pdb_a):
        return _error_response("NotFound", f"pdb_a not found: {pdb_a}")
    if not pdb_b or not os.path.exists(pdb_b):
        return _error_response("NotFound", f"pdb_b not found: {pdb_b}")
    ok, hint = is_pymol_available()
    if not ok:
        return _error_response("DependencyMissing", hint)

    out_dir = str(out_dir or "").strip().rstrip(os.sep)
    if not out_dir:
        return _error_response("ValidationError", "empty out_dir")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    stem_a = Path(pdb_a).stem
    stem_b = Path(pdb_b).stem
    png_path = os.path.join(out_dir, f"superpose_{stem_a}_to_{stem_b}.png")
    pse_path = os.path.join(out_dir, f"superpose_{stem_a}_to_{stem_b}.pse")
    script = _SUPERPOSE_SCRIPT_TEMPLATE.format(
        pdb_a=pdb_a, pdb_b=pdb_b, png_path=png_path, pse_path=pse_path,
        width=width, height=height, dpi=dpi,
    )
    result = run_pymol_script(script, out_dir, expected_outputs=[png_path, pse_path], timeout_secs=timeout_secs)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    if not result["success"]:
        return _error_response("RenderError", result["error"] or "pymol failed", suggestion=result["stderr"][-500:] if result["stderr"] else None)
    if not os.path.exists(png_path):
        return _error_response("RenderError", "pymol exited 0 but no PNG was produced", suggestion=result["stdout"][-500:])

    rmsd = None
    n_aligned = None
    for line in result["stdout"].splitlines():
        if line.startswith("rmsd="):
            try:
                parts = dict(p.split("=") for p in line.split())
                rmsd = float(parts.get("rmsd", "nan"))
                n_aligned = int(parts.get("aligned", "-1"))
            except (ValueError, KeyError):
                pass
            break
    meta = {
        "mobile": os.path.basename(pdb_a),
        "reference": os.path.basename(pdb_b),
        "rmsd_angstroms": rmsd,
        "aligned_residues": n_aligned,
        "png_path": to_client_file_path(png_path),
        "pse_path": to_client_file_path(pse_path) if os.path.exists(pse_path) else None,
    }
    return _success_response(png_path, meta, elapsed_ms, stdout_preview=result["stdout"])


__all__ = ["render_protein_structure", "superpose_two_structures"]
