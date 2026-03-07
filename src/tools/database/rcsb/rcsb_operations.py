"""
RCSB PDB operations: single exit for query and download; both return rich JSON.

Success: status, file_info (download) or content (query), content_preview, biological_metadata, execution_context.
Error: status "error", error { type, message, suggestion }, file_info null.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from .rcsb_metadata import query_rcsb_entry as _raw_query_entry
except ImportError:
    _dir = Path(__file__).resolve().parent
    if _dir.name == "rcsb" and str(_dir.parents[3]) not in sys.path:
        sys.path.insert(0, str(_dir.parents[3]))
    from src.tools.database.rcsb.rcsb_metadata import query_rcsb_entry as _raw_query_entry

try:
    from .rcsb_structure import query_rcsb_structure as _raw_query_structure
    from .rcsb_structure import download_rcsb_structure as _raw_download_structure
except ImportError:
    from src.tools.database.rcsb.rcsb_structure import query_rcsb_structure as _raw_query_structure
    from src.tools.database.rcsb.rcsb_structure import download_rcsb_structure as _raw_download_structure


_PREVIEW_LEN = 500
_SOURCE_RCSB = "RCSB PDB"


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
    source: str = _SOURCE_RCSB,
) -> str:
    """Build JSON for download success: status, file_info, content_preview, biological_metadata, execution_context."""
    path = Path(file_path)
    file_size = path.stat().st_size if path.exists() else 0
    fmt = path.suffix.lstrip(".").lower() or "json"
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
    source: str = _SOURCE_RCSB,
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


def _read_preview(path: str, max_chars: int = _PREVIEW_LEN) -> str:
    """Read first max_chars from file for content_preview."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(max_chars)
    except Exception:
        return ""


# ---------- query_rcsb_*: return rich JSON (status, content, content_preview, biological_metadata, execution_context) ----------


def query_rcsb_entry_metadata_by_pdb_id(pdb_id: str) -> str:
    """Query RCSB entry metadata by PDB ID. Returns rich JSON: status, content, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        raw = _raw_query_entry(pdb_id)
        data = json.loads(raw)
        if "error" in data:
            return _error_response("QueryError", data["error"], suggestion="Check PDB ID format (e.g. 4HHB).")
        content = json.dumps(data, ensure_ascii=False)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        # Extract biological metadata from the entry
        meta: Dict[str, Any] = {"pdb_id": pdb_id.upper()}
        struct = data.get("struct", {})
        if struct.get("title"):
            meta["title"] = struct["title"]
        cell = data.get("cell", {})
        if cell:
            meta["cell"] = {
                "a": cell.get("length_a"),
                "b": cell.get("length_b"),
                "c": cell.get("length_c"),
            }
        exptl = data.get("exptl", [])
        if exptl:
            meta["experimental_method"] = exptl[0].get("method") if isinstance(exptl, list) else None
        rcsb_info = data.get("rcsb_entry_info", {})
        if rcsb_info.get("resolution_combined"):
            meta["resolution"] = rcsb_info["resolution_combined"]
        if rcsb_info.get("polymer_entity_count"):
            meta["polymer_entity_count"] = rcsb_info["polymer_entity_count"]
        return _query_success_response(content, content_preview=content, biological_metadata=meta, query_time_ms=elapsed_ms)
    except json.JSONDecodeError:
        # raw text is not JSON → unlikely for metadata, but handle gracefully
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"pdb_id": pdb_id.upper()}
        return _query_success_response(raw, content_preview=raw, biological_metadata=meta, query_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("QueryError", str(e), suggestion="Check PDB ID format (e.g. 4HHB).")


def query_rcsb_structure_by_pdb_id(
    pdb_id: str,
    file_type: str = "pdb",
) -> str:
    """Query RCSB structure data by PDB ID. Returns rich JSON: status, content, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        raw = _raw_query_structure(pdb_id, file_type=file_type)
        # Check if the raw response is an error JSON
        if raw.strip().startswith("{"):
            try:
                err = json.loads(raw)
                if "error" in err:
                    return _error_response("QueryError", err["error"], suggestion="Check PDB ID format (e.g. 4HHB).")
            except json.JSONDecodeError:
                pass
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta: Dict[str, Any] = {"pdb_id": pdb_id.upper(), "file_type": file_type, "content_length": len(raw)}
        return _query_success_response(raw, content_preview=raw, biological_metadata=meta, query_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("QueryError", str(e), suggestion="Check PDB ID and file_type (pdb, cif, xml, etc.).")


# ---------- download_rcsb_*: save to file, return JSON with success + file_path ----------


def download_rcsb_entry_metadata_by_pdb_id(
    pdb_id: str,
    out_path: str,
) -> str:
    """Download RCSB entry metadata by PDB ID to JSON file. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        raw = _raw_query_entry(pdb_id)
        data = json.loads(raw)
        if "error" in data:
            return _error_response("DownloadError", data["error"], suggestion="Check PDB ID format (e.g. 4HHB).")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta: Dict[str, Any] = {"pdb_id": pdb_id.upper()}
        struct = data.get("struct", {})
        if struct.get("title"):
            meta["title"] = struct["title"]
        exptl = data.get("exptl", [])
        if exptl and isinstance(exptl, list):
            meta["experimental_method"] = exptl[0].get("method")
        return _download_success_response(out_path, content_preview=_read_preview(out_path), biological_metadata=meta, download_time_ms=elapsed_ms)
    except json.JSONDecodeError as e:
        return _error_response("DownloadError", f"Invalid JSON response: {e}", suggestion="Check PDB ID format (e.g. 4HHB).")
    except Exception as e:
        return _error_response("DownloadError", str(e), suggestion="Check PDB ID format (e.g. 4HHB).")


def download_rcsb_structure_by_pdb_id(
    pdb_id: str,
    out_dir: str,
    file_type: str = "pdb",
    unzip: bool = True,
) -> str:
    """Download RCSB structure file by PDB ID. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        result_path = _raw_download_structure(pdb_id, out_dir, file_type=file_type, unzip=unzip)
        if result_path is None:
            return _error_response("DownloadError", f"Failed to download structure for {pdb_id}", suggestion="Check PDB ID and file_type (pdb, cif, xml, etc.).")
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta: Dict[str, Any] = {
            "pdb_id": pdb_id.upper(),
            "file_type": file_type,
            "unzipped": unzip,
        }
        preview = _read_preview(result_path)
        return _download_success_response(result_path, content_preview=preview, biological_metadata=meta, download_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("DownloadError", str(e), suggestion="Check PDB ID and file_type (pdb, cif, xml, etc.).")


__all__ = [
    "query_rcsb_entry_metadata_by_pdb_id",
    "query_rcsb_structure_by_pdb_id",
    "download_rcsb_entry_metadata_by_pdb_id",
    "download_rcsb_structure_by_pdb_id",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="RCSB PDB operations: query_* (return JSON with content) and download_* (return JSON with file_path)."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run query_* and download_* samples; output under example/database/rcsb",
    )
    parser.add_argument("--pdb_id", type=str, default="4HHB", help="PDB ID for test. Default 4HHB.")
    parser.add_argument("--file_type", type=str, default="pdb", help="Structure file type (pdb, cif, xml). Default pdb.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="example/database/rcsb",
        help="Output directory. Default example/database/rcsb.",
    )
    args = parser.parse_args()

    if not args.test:
        print("Use --test to run operations tests.")
        exit(0)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    pdb_id = args.pdb_id

    def _print_result(name: str, res: str) -> None:
        obj = json.loads(res)
        print(f"  {name}: status={obj.get('status')}")
        if obj.get("status") == "success":
            if obj.get("file_info"):
                print(f"    file_info: {obj['file_info']}")
            if obj.get("content") and len(obj["content"]) > 200:
                print(f"    content_preview: {obj.get('content_preview', '')[:200]}...")
            elif obj.get("content"):
                print(f"    content: {obj['content'][:200]}...")
            if obj.get("biological_metadata"):
                print(f"    biological_metadata: {obj['biological_metadata']}")
            if obj.get("execution_context"):
                print(f"    execution_context: {obj['execution_context']}")
        if obj.get("status") == "error" and obj.get("error"):
            print(f"    error: {obj['error']}")

    print("=== query_rcsb_* (return rich JSON: status, content, content_preview, biological_metadata, execution_context) ===")
    _print_result("query_rcsb_entry_metadata_by_pdb_id", query_rcsb_entry_metadata_by_pdb_id(pdb_id))
    _print_result("query_rcsb_structure_by_pdb_id", query_rcsb_structure_by_pdb_id(pdb_id, file_type=args.file_type))

    print("=== download_rcsb_* (return rich JSON: status, file_info, content_preview, biological_metadata, execution_context) ===")
    for name, res in [
        ("download_rcsb_entry_metadata_by_pdb_id", download_rcsb_entry_metadata_by_pdb_id(pdb_id, os.path.join(out_dir, f"rcsb_entry_{pdb_id}.json"))),
        ("download_rcsb_structure_by_pdb_id", download_rcsb_structure_by_pdb_id(pdb_id, out_dir, file_type=args.file_type)),
    ]:
        _print_result(name, res)

    print(f"Done. Output under {out_dir}")
