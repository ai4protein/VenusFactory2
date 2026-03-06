"""
UniProt operations: single exit for query and download; both return rich JSON.

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
    from .uniprot_search import (
        uniprot_search,
        uniprot_retrieve,
        uniprot_mapping,
    )
except ImportError:
    _dir = Path(__file__).resolve().parent
    if _dir.name == "uniprot" and str(_dir.parents[4]) not in sys.path:
        sys.path.insert(0, str(_dir.parents[4]))
    from src.tools.search.database.uniprot.uniprot_search import (  # noqa: E501
        uniprot_search,
        uniprot_retrieve,
        uniprot_mapping,
    )

try:
    from .uniprot_sequence import query_uniprot_seq as _raw_query_uniprot_seq
except ImportError:
    from src.tools.search.database.uniprot.uniprot_sequence import (  # noqa: E501
        query_uniprot_seq as _raw_query_uniprot_seq,
    )

try:
    from .uniprot_metadata import query_uniprot_meta as _raw_query_uniprot_meta
except ImportError:
    from src.tools.search.database.uniprot.uniprot_metadata import (  # noqa: E501
        query_uniprot_meta as _raw_query_uniprot_meta,
    )


_PREVIEW_LEN = 500
_SOURCE_UNIPROT = "UniProt"


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
    source: str = _SOURCE_UNIPROT,
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
    source: str = _SOURCE_UNIPROT,
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


def _is_json_error(text: str) -> bool:
    try:
        data = json.loads(text)
        return isinstance(data, dict) and data.get("success") is False
    except (json.JSONDecodeError, TypeError):
        return False


# ---------- query_*: return rich JSON ----------

def query_uniprot_search_by_query(
    query: str,
    frmt: str = "tsv",
    columns: Optional[str] = None,
    limit: Optional[int] = 100,
    database: str = "uniprotkb",
    **filters
) -> str:
    """Search UniProt by query string. Returns rich JSON: status, content, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        res = uniprot_search(query, frmt=frmt, columns=columns, limit=limit, database=database, **filters)
        if _is_json_error(res):
            err_data = json.loads(res)
            return _error_response("QueryError", err_data.get("error", "Unknown error"), suggestion="Check query and parameters.")
        
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"query": query, "format": frmt, "limit": limit, "database": database}
        meta.update(filters)
        return _query_success_response(res, content_preview=res, biological_metadata=meta, query_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("QueryError", str(e), suggestion="Ensure bioservices is installed and query is valid.")


def query_uniprot_retrieve_by_id(
    uniprot_id: str,
    frmt: str = "fasta",
) -> str:
    """Retrieve one UniProt entry by ID. Returns rich JSON: status, content, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        res = uniprot_retrieve(uniprot_id, frmt=frmt)
        if _is_json_error(res):
            err_data = json.loads(res)
            return _error_response("QueryError", err_data.get("error", "Unknown error"), suggestion="Check UniProt ID.")
            
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"uniprot_id": uniprot_id, "format": frmt}
        return _query_success_response(res, content_preview=res, biological_metadata=meta, query_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("QueryError", str(e), suggestion="Check UniProt ID and format.")


def query_uniprot_mapping(
    fr: str,
    to: str,
    query: str,
) -> str:
    """Map identifiers between databases. Returns rich JSON: status, content, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        res = uniprot_mapping(fr=fr, to=to, query=query)
        if _is_json_error(res):
            err_data = json.loads(res)
            return _error_response("QueryError", err_data.get("error", "Unknown error"), suggestion="Check fr/to databases and ID.")
            
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"from_db": fr, "to_db": to, "query": query}
        return _query_success_response(res, content_preview=res, biological_metadata=meta, query_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("QueryError", str(e), suggestion="Check mapping database names.")


def query_uniprot_seq_by_id(uniprot_id: str) -> str:
    """Query UniProt sequence (FASTA) by ID. Returns rich JSON: status, content, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        res = _raw_query_uniprot_seq(uniprot_id)
        if _is_json_error(res):
            err_data = json.loads(res)
            return _error_response("QueryError", err_data.get("error", "Unknown error"), suggestion="Check UniProt ID.")
            
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"uniprot_id": uniprot_id, "type": "sequence"}
        return _query_success_response(res, content_preview=res, biological_metadata=meta, query_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("QueryError", str(e), suggestion="Check UniProt ID.")


def query_uniprot_meta_by_id(uniprot_id: str) -> str:
    """Query UniProt metadata (JSON) by ID. Returns rich JSON: status, content, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        res = _raw_query_uniprot_meta(uniprot_id)
        if _is_json_error(res):
            err_data = json.loads(res)
            return _error_response("QueryError", err_data.get("error", "Unknown error"), suggestion="Check UniProt ID.")
            
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"uniprot_id": uniprot_id, "type": "metadata"}
        return _query_success_response(res, content_preview=res, biological_metadata=meta, query_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("QueryError", str(e), suggestion="Check UniProt ID.")


# ---------- download_*: save to file, return JSON with success + file_path ----------


def download_uniprot_search_by_query(
    query: str,
    out_path: str,
    frmt: str = "tsv",
    columns: Optional[str] = None,
    limit: Optional[int] = 100,
    database: str = "uniprotkb",
    **filters
) -> str:
    """Download UniProt search results. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        res = uniprot_search(query, frmt=frmt, columns=columns, limit=limit, database=database, **filters)
        if _is_json_error(res):
            err_data = json.loads(res)
            return _error_response("DownloadError", err_data.get("error", "Unknown error"), suggestion="Check query and parameters.")
            
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(res)
            
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"query": query, "format": frmt, "limit": limit, "database": database}
        meta.update(filters)
        return _download_success_response(out_path, content_preview=_read_preview(out_path), biological_metadata=meta, download_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("DownloadError", str(e), suggestion="Check query and parameters.")


def download_uniprot_retrieve_by_id(
    uniprot_id: str,
    out_path: str,
    frmt: str = "fasta",
) -> str:
    """Download one UniProt entry. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        res = uniprot_retrieve(uniprot_id, frmt=frmt)
        if _is_json_error(res):
            err_data = json.loads(res)
            return _error_response("DownloadError", err_data.get("error", "Unknown error"), suggestion="Check UniProt ID.")

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(res)
            
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"uniprot_id": uniprot_id, "format": frmt}
        return _download_success_response(out_path, content_preview=_read_preview(out_path), biological_metadata=meta, download_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("DownloadError", str(e), suggestion="Check UniProt ID and format.")


def download_uniprot_mapping(
    fr: str,
    to: str,
    query: str,
    out_path: str,
) -> str:
    """Download mapped identifiers. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        res = uniprot_mapping(fr=fr, to=to, query=query)
        if _is_json_error(res):
            err_data = json.loads(res)
            return _error_response("DownloadError", err_data.get("error", "Unknown error"), suggestion="Check fr/to databases and ID.")

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(res)
            
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"from_db": fr, "to_db": to, "query": query}
        return _download_success_response(out_path, content_preview=_read_preview(out_path), biological_metadata=meta, download_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("DownloadError", str(e), suggestion="Check mapping database names.")


def download_uniprot_seq_by_id(uniprot_id: str, out_path: str) -> str:
    """Download UniProt sequence (FASTA) by ID. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        res = _raw_query_uniprot_seq(uniprot_id)
        if _is_json_error(res):
            err_data = json.loads(res)
            return _error_response("DownloadError", err_data.get("error", "Unknown error"), suggestion="Check UniProt ID.")
            
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(res)
            
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"uniprot_id": uniprot_id, "type": "sequence"}
        return _download_success_response(out_path, content_preview=_read_preview(out_path), biological_metadata=meta, download_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("DownloadError", str(e), suggestion="Check UniProt ID.")


def download_uniprot_meta_by_id(uniprot_id: str, out_path: str) -> str:
    """Download UniProt metadata (JSON) by ID. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        res = _raw_query_uniprot_meta(uniprot_id)
        if _is_json_error(res):
            err_data = json.loads(res)
            return _error_response("DownloadError", err_data.get("error", "Unknown error"), suggestion="Check UniProt ID.")
            
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(res)
            
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"uniprot_id": uniprot_id, "type": "metadata"}
        return _download_success_response(out_path, content_preview=_read_preview(out_path), biological_metadata=meta, download_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("DownloadError", str(e), suggestion="Check UniProt ID.")


__all__ = [
    "query_uniprot_search_by_query",
    "query_uniprot_retrieve_by_id",
    "query_uniprot_mapping",
    "query_uniprot_seq_by_id",
    "query_uniprot_meta_by_id",
    "download_uniprot_search_by_query",
    "download_uniprot_retrieve_by_id",
    "download_uniprot_mapping",
    "download_uniprot_seq_by_id",
    "download_uniprot_meta_by_id",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="UniProt operations: query_* (return JSON with content) and download_* (return JSON with file_path)."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run query_* and download_* samples; output under example/database/uniprot",
    )
    parser.add_argument("--query", type=str, default="urate", help="Search query string.")
    parser.add_argument("--id", type=str, default="P43403", help="UniProt ID for tests. Default P43403.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="example/database/uniprot",
        help="Output directory. Default example/database/uniprot_ops.",
    )
    args = parser.parse_args()

    if not args.test:
        print("Use --test to run operations tests.")
        exit(0)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    q = args.query
    uid = args.id

    def _print_query(name: str, res: str) -> None:
        obj = json.loads(res)
        print(f"  {name}: status={obj.get('status')} ...")
        if obj.get("status") == "success":
            if obj.get("content") and len(obj["content"]) > 500:
                print(f"  (content_preview): {obj.get('content_preview', '')[:200]}...")
            elif obj.get("content"):
                print(f"  content: {obj['content'][:200]}...")
            if obj.get("execution_context"):
                print(f"  execution_context: {obj['execution_context']}")
        if obj.get("status") == "error" and obj.get("error"):
            print(f"  error: {obj['error']}")

    print("=== query_* (return rich JSON) ===")
    res_search = query_uniprot_search_by_query(q, limit=2)
    _print_query("query_uniprot_search_by_query(...)", res_search)

    res_retrieve = query_uniprot_retrieve_by_id(uid, frmt="fasta")
    _print_query("query_uniprot_retrieve_by_id(...)", res_retrieve)

    res_map = query_uniprot_mapping("UniProtKB_AC-ID", "KEGG", uid)
    _print_query("query_uniprot_mapping(...)", res_map)

    res_seq = query_uniprot_seq_by_id(uid)
    _print_query("query_uniprot_seq_by_id(...)", res_seq)

    res_meta = query_uniprot_meta_by_id(uid)
    _print_query("query_uniprot_meta_by_id(...)", res_meta)

    print("=== download_* (return rich JSON) ===")
    for name, res in [
        ("download_uniprot_search_by_query", download_uniprot_search_by_query(q, os.path.join(out_dir, f"{q}_uniprot_search.tsv"), limit=2)),
        ("download_uniprot_retrieve_by_id", download_uniprot_retrieve_by_id(uid, os.path.join(out_dir, f"{uid}_uniprot_retrieve.fasta"))),
        ("download_uniprot_mapping", download_uniprot_mapping("UniProtKB_AC-ID", "KEGG", uid, os.path.join(out_dir, f"{uid}_uniprot_mapping.json"))),
        ("download_uniprot_seq_by_id", download_uniprot_seq_by_id(uid, os.path.join(out_dir, f"{uid}_uniprot_seq.fasta"))),
        ("download_uniprot_meta_by_id", download_uniprot_meta_by_id(uid, os.path.join(out_dir, f"{uid}_uniprot_meta.json"))),
    ]:
        dl_obj = json.loads(res)
        status = dl_obj.get("status")
        if status == "success":
            fi = dl_obj.get("file_info", {})
            print(f"  {name}: status={status}, file_path={fi.get('file_path')}")
        else:
            print(f"  {name}: {dl_obj}")

    print(f"Done. Output under {out_dir}")
