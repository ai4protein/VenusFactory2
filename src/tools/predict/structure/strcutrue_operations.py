"""
Unified structure operations: single entry that references all operations under
src/tools/predict/structure/ (e.g. esmfold). All implementations live in this directory;
this module only wraps them and returns consistent rich JSON.

- predict_structure_esmfold: from .esmfold (local prediction).
MCP entry: tools_mcp mcp_predict_structure_esmfold (local only).

Success: status, file_info, content_preview, biological_metadata, execution_context.
Error: status "error", error { type, message, suggestion }, file_info null.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional
from src.tools.path_sanitizer import to_client_file_path

from .esmfold import predict_structure_sync, _get_default_backend


_PREVIEW_LEN = 500
_SOURCE = "Predict_Structure"


def _error_response(error_type: str, message: str, suggestion: Optional[str] = None) -> str:
    out: Dict[str, Any] = {
        "status": "error",
        "error": {"type": error_type, "message": message},
        "file_info": None,
    }
    if suggestion:
        out["error"]["suggestion"] = suggestion
    return json.dumps(out, ensure_ascii=False)


def _download_success_response(
    file_path: str,
    content_preview: Optional[str] = None,
    biological_metadata: Optional[Dict[str, Any]] = None,
    download_time_ms: int = 0,
    source: str = _SOURCE,
) -> str:
    path = Path(file_path)
    file_size = path.stat().st_size if path.exists() else 0
    fmt = path.suffix.lstrip(".").lower() or "pdb"
    out: Dict[str, Any] = {
        "status": "success",
        "file_info": {
            "file_path": to_client_file_path(path if path.exists() else file_path),
            "file_name": path.name,
            "file_size": file_size,
            "format": fmt,
        },
        "content_preview": (content_preview or "")[:_PREVIEW_LEN],
        "biological_metadata": biological_metadata or {},
        "execution_context": {"download_time_ms": download_time_ms, "source": source},
    }
    return json.dumps(out, ensure_ascii=False)


def predict_structure_esmfold(
    sequence: str,
    output_dir: Optional[str] = None,
    output_file: Optional[str] = None,
    verbose: bool = True,
) -> str:
    """Predict protein structure with ESMFold. Backend chosen by ESMFOLD_BACKEND env var (local/pjlab). Returns rich JSON with file_info (PDB path)."""
    t0 = time.perf_counter()
    if not sequence or not sequence.strip():
        return _error_response("ValidationError", "Sequence is required.", suggestion="Provide a non-empty protein sequence.")
    try:
        pdb_path, result_info = predict_structure_sync(
            sequence.strip(),
            output_dir=output_dir or "./protein_structures",
            verbose=verbose,
            backend=_get_default_backend(),
            output_file=output_file,
        )
        if not pdb_path:
            return _error_response(
                "PredictionError",
                "ESMFold returned no PDB path.",
                suggestion="Check ESMFOLD_BACKEND, GPU/env, and sequence.",
            )
        content_preview = json.dumps(result_info, ensure_ascii=False)[:_PREVIEW_LEN] if result_info else ""
        biological_metadata = dict(result_info) if isinstance(result_info, dict) else {"result_info": str(result_info)}
        return _download_success_response(
            pdb_path,
            content_preview=content_preview,
            biological_metadata=biological_metadata,
            download_time_ms=int((time.perf_counter() - t0) * 1000),
            source=_SOURCE,
        )
    except Exception as e:
        return _error_response(
            "PredictionError",
            str(e),
            suggestion="Check sequence, output_dir, and ESMFold backend (ESMFOLD_BACKEND).",
        )


__all__ = ["predict_structure_esmfold"]
