"""
AlphaFold operations: single exit for query and download; both return rich JSON.

Success: status, file_info (download) or content (query), content_preview, biological_metadata, execution_context.
Error: status "error", error { type, message, suggestion }, file_info null.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

try:
    from .alphafold_structure import (
        query_alphafold_structure as _query_structure,
        download_alphafold_structure as _download_structure_impl,
    )
except ImportError:
    _dir = Path(__file__).resolve().parent
    if _dir.name == "alphafold" and str(_dir.parents[3]) not in sys.path:
        sys.path.insert(0, str(_dir.parents[3]))
    from src.tools.database.alphafold.alphafold_structure import (
        query_alphafold_structure as _query_structure,
        download_alphafold_structure as _download_structure_impl,
    )

try:
    from .alphafold_metadata import (
        query_alphafold_metadata as _query_metadata,
        download_alphafold_metadata as _download_metadata,
    )
except ImportError:
    from src.tools.database.alphafold.alphafold_metadata import (
        query_alphafold_metadata as _query_metadata,
        download_alphafold_metadata as _download_metadata,
    )


_PREVIEW_LEN = 500
_SOURCE_ALPHAFOLD = "AlphaFold DB"


def _error_response(error_type: str, message: str, suggestion: Optional[str] = None) -> str:
    """Build JSON for error: status error, error { type, message, suggestion }, file_info null."""
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
    source: str = _SOURCE_ALPHAFOLD,
) -> str:
    """Build JSON for download success: status, file_info, content_preview, biological_metadata, execution_context."""
    path = Path(file_path)
    file_size = path.stat().st_size if path.exists() else 0
    fmt = path.suffix.lstrip(".").lower() or "pdb"
    out: Dict[str, Any] = {
        "status": "success",
        "file_info": {
            "file_path": str(path.resolve()) if path.exists() else file_path,
            "file_name": path.name,
            "file_size": file_size,
            "format": fmt,
        },
        "content_preview": (content_preview or "")[: _PREVIEW_LEN],
        "biological_metadata": biological_metadata or {},
        "execution_context": {"download_time_ms": download_time_ms, "source": source},
    }
    return json.dumps(out, ensure_ascii=False)


def _query_success_response(
    content: str,
    content_preview: Optional[str] = None,
    biological_metadata: Optional[Dict[str, Any]] = None,
    query_time_ms: int = 0,
    source: str = _SOURCE_ALPHAFOLD,
) -> str:
    """Build JSON for query success: status, content, content_preview, biological_metadata, execution_context."""
    preview = (content_preview or content or "")[: _PREVIEW_LEN]
    out: Dict[str, Any] = {
        "status": "success",
        "content": content,
        "content_preview": preview,
        "biological_metadata": biological_metadata or {},
        "execution_context": {"query_time_ms": query_time_ms, "source": source},
    }
    return json.dumps(out, ensure_ascii=False)


# ---------- query_*: return JSON with success + content ----------


def query_alphafold_structure_by_uniprot_id(
    uniprot_id: str,
    format: Literal["pdb", "cif"] = "pdb",
    version: str = "v6",
    fragment: int = 1,
) -> str:
    """Query AlphaFold structure by UniProt ID. Returns rich JSON: status, content, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    raw = _query_structure(
        uniprot_id, format=format, version=version, fragment=fragment
    )
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    if raw.strip().startswith(f"{uniprot_id} failed,") or raw.strip().startswith("failed"):
        return _error_response(
            "NotFound",
            raw.strip(),
            suggestion="Check UniProt ID or try a different AlphaFold version (v1, v2, v4, v6).",
        )
    meta = {"uniprot_id": uniprot_id, "format": format, "version": version, "fragment": fragment}
    return _query_success_response(raw, content_preview=raw, biological_metadata=meta, query_time_ms=elapsed_ms)


def query_alphafold_metadata_by_uniprot_id(uniprot_id: str) -> str:
    """Query AlphaFold prediction metadata by UniProt ID. Returns rich JSON: status, content, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    raw = _query_metadata(uniprot_id)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and data.get("success") is False:
            return _error_response(
                "NotFound",
                data.get("error", "unknown"),
                suggestion="Check UniProt ID or try another accession.",
            )
    except json.JSONDecodeError:
        pass
    meta = {"uniprot_id": uniprot_id, "format": "json"}
    return _query_success_response(raw, content_preview=raw, biological_metadata=meta, query_time_ms=elapsed_ms)


# ---------- download_*: save to file, return JSON with success + file_path ----------


def download_alphafold_structure_by_uniprot_id(
    uniprot_id: str,
    out_dir: str,
    format: Literal["pdb", "cif"] = "pdb",
    version: str = "v6",
    fragment: int = 1,
) -> str:
    """Download AlphaFold structure to a file or directory. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    out_dir = str(out_dir).strip()
    if not out_dir:
        return _error_response("ValidationError", "invalid out_path", suggestion="Provide a non-empty output directory.")
    if os.path.isdir(out_dir) or out_dir.endswith(os.sep):
        out_dir = out_dir.rstrip(os.sep)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    success, path = _download_structure_impl(
        uniprot_id, out_dir, format=format, version=version, fragment=fragment
    )
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    if not success or not path:
        return _error_response(
            "NotFound",
            f"Failed to download structure for {uniprot_id}.",
            suggestion="Check UniProt ID or try a different AlphaFold version (v1, v2, v4, v6).",
        )
    full_path = os.path.join(out_dir, f"{uniprot_id}.{format}")
    content_preview = ""
    try:
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            content_preview = f.read(_PREVIEW_LEN)
    except Exception:
        pass
    meta = {"uniprot_id": uniprot_id, "format": format, "version": version, "fragment": fragment}
    return _download_success_response(
        full_path, content_preview=content_preview, biological_metadata=meta, download_time_ms=elapsed_ms
    )


def download_alphafold_metadata_by_uniprot_id(uniprot_id: str, out_dir: str) -> str:
    """Download AlphaFold metadata to a file or directory. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    out_dir = str(out_dir).strip()
    if not out_dir:
        return _error_response("ValidationError", "invalid out_dir", suggestion="Provide a non-empty output directory.")
    out_dir = out_dir.rstrip(os.sep)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    msg = _download_metadata(uniprot_id, out_dir)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    if "failed" in msg.lower():
        return _error_response(
            "NotFound", msg, suggestion="Check UniProt ID or try another accession."
        )
    full_path = os.path.join(out_dir, f"{uniprot_id}.json")
    content_preview = ""
    try:
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            content_preview = f.read(_PREVIEW_LEN)
    except Exception:
        pass
    meta = {"uniprot_id": uniprot_id, "format": "json"}
    return _download_success_response(
        full_path, content_preview=content_preview, biological_metadata=meta, download_time_ms=elapsed_ms
    )


def _download_alphafold_structure_impl(
    uniprot_id: str,
    out_dir: str,
    format: Literal["pdb", "cif"] = "pdb",
    version: str = "v6",
    fragment: int = 1,
) -> Tuple[bool, Optional[str]]:
    """Internal: download structure and return (success, path). Used by tools_mcp for OSS upload."""
    return _download_structure_impl(
        uniprot_id, out_dir, format=format, version=version, fragment=fragment
    )


__all__ = [
    "download_alphafold_structure_by_uniprot_id",
    "download_alphafold_metadata_by_uniprot_id",
]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="AlphaFold operations: query_* (return JSON with content) and download_* (return JSON with file_path)."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run query_* and download_* samples; output under example/database/alphafold",
    )
    parser.add_argument("--uniprot_id", type=str, default="A0A1B0GTW7", help="UniProt ID for test")
    parser.add_argument("--format", type=str, default="pdb", choices=["pdb", "cif"], help="Structure format: 'pdb' (default) or 'cif'.")
    parser.add_argument("--version", type=str, default="v6", choices=["v1", "v2", "v4", "v6"], help="AlphaFold DB version: v1, v2, v4, or v6. Default v6.")
    parser.add_argument("--out_dir", type=str, default="example/database/alphafold", help="Output directory for AlphaFold structure and metadata. Default 'example/database/alphafold'.")
    args = parser.parse_args()

    if not args.test:
        print("Use --test to run operations tests.")
        exit(0)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    uid = args.uniprot_id or "A0A1B0GTW7"
    fmt = "cif" if (args.format or "").lower() in ("cif", "mmcif") else "pdb"

    print("=== query_* (return JSON: success + content) ===")
    print("  query_alphafold_structure_by_uniprot_id(...)")
    res = query_alphafold_structure_by_uniprot_id(uid, format=fmt, version=args.version)
    obj = json.loads(res)
    print(f"  structure: {res}")
    if obj.get("content") and len(obj["content"]) > 500:
        print(f"  (first 500 chars of content): {obj['content'][:500]}...")
    elif obj.get("content"):
        print(f"  content: {obj['content']}")
    if obj.get("error"):
        print(f"  error: {obj['error']}")
    query_structure_path = os.path.join(out_dir, f"query_structure_{uid}.txt")
    with open(query_structure_path, "w", encoding="utf-8") as f:
        f.write(res)
    print(f"  full JSON saved to {query_structure_path}")

    print("  query_alphafold_metadata_by_uniprot_id(...)")
    meta_res = query_alphafold_metadata_by_uniprot_id(uid)
    meta_obj = json.loads(meta_res)
    print(f"  metadata: {meta_res}")
    if meta_obj.get("content") and len(meta_obj["content"]) > 500:
        print(f"  (first 500 chars of content): {meta_obj['content'][:500]}...")
    elif meta_obj.get("content"):
        print(f"  content: {meta_obj['content']}")
    if meta_obj.get("error"):
        print(f"  error: {meta_obj['error']}")
    query_metadata_path = os.path.join(out_dir, f"query_metadata_{uid}.txt")
    with open(query_metadata_path, "w", encoding="utf-8") as f:
        f.write(meta_res)
    print(f"  full JSON saved to {query_metadata_path}")

    print("=== download_* (return JSON: success + file_path) ===")
    dl_struct = json.loads(download_alphafold_structure_by_uniprot_id(uid, out_dir, format=fmt, version=args.version))
    print(f"  structure: {dl_struct}")
    dl_meta = json.loads(download_alphafold_metadata_by_uniprot_id(uid, out_dir))
    print(f"  metadata: {dl_meta}")

    print(f"Done. Output under {out_dir}")
