"""
KEGG REST API operations: single exit for query and download; both return rich JSON.

Success: status, file_info (download) or content (query), content_preview, biological_metadata, execution_context.
Error: status "error", error { type, message, suggestion }, file_info null.

Operations: info, list, find, get, conv, link, ddi.
Academic use only. Base client: kegg_rest.
See kegg.md and src/agent/skills/kegg/ for references.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from src.tools.path_sanitizer import to_client_file_path

try:
    from .kegg_rest import kegg_request, _join_ids, BASE_URL
except ImportError:
    _dir = Path(__file__).resolve().parent
    if _dir.name == "kegg" and str(_dir.parents[3]) not in sys.path:
        sys.path.insert(0, str(_dir.parents[3]))
    from src.tools.database.kegg.kegg_rest import kegg_request, _join_ids, BASE_URL


_PREVIEW_LEN = 500
_SOURCE_KEGG = "KEGG"


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
    source: str = _SOURCE_KEGG,
) -> str:
    """Build JSON for download success: status, file_info, content_preview, biological_metadata, execution_context."""
    path = Path(file_path)
    file_size = path.stat().st_size if path.exists() else 0
    fmt = path.suffix.lstrip(".").lower() or "txt"
    out: Dict[str, Any] = {
        "status": "success",
        "file_info": {
            "file_path": to_client_file_path(path if path.exists() else file_path),
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
    source: str = _SOURCE_KEGG,
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


# ---------- query_*: return rich JSON (status, content, content_preview, biological_metadata, execution_context) ----------


def query_kegg_info_by_database(database: str) -> str:
    """Query KEGG database metadata/statistics by database name. Returns rich JSON: status, content, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        text = kegg_request("info", database)
        content = json.dumps({"database": database, "info": text}, ensure_ascii=False)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"database": database}
        return _query_success_response(content, content_preview=content, biological_metadata=meta, query_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("QueryError", str(e), suggestion="Check KEGG database name (e.g. pathway, compound, gene).")


def query_kegg_list_by_database(database: str, org_or_ids: Optional[Union[str, list]] = None) -> str:
    """Query KEGG list entries by database. Returns rich JSON: status, content, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        if org_or_ids is None:
            text = kegg_request("list", database)
        else:
            sid = _join_ids(org_or_ids) if isinstance(org_or_ids, list) else str(org_or_ids).strip()
            text = kegg_request("list", database, sid)
        content = json.dumps({"database": database, "org_or_ids": org_or_ids, "list": text}, ensure_ascii=False)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"database": database, "org_or_ids": org_or_ids}
        return _query_success_response(content, content_preview=content, biological_metadata=meta, query_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("QueryError", str(e), suggestion="Check KEGG database and organism/ID.")


def query_kegg_find_by_database(database: str, query: str, option: Optional[str] = None) -> str:
    """Search KEGG database by query string. Returns rich JSON: status, content, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        if option:
            text = kegg_request("find", database, query, option)
        else:
            text = kegg_request("find", database, query)
        content = json.dumps({"database": database, "query": query, "option": option, "result": text}, ensure_ascii=False)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"database": database, "query": query, "option": option}
        return _query_success_response(content, content_preview=content, biological_metadata=meta, query_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("QueryError", str(e), suggestion="Check KEGG database and query string.")


def query_kegg_entry_by_id(entry_id: Union[str, List[str]], format: Optional[str] = None) -> str:
    """Get KEGG entry/entries by ID. Returns rich JSON: status, content, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        eid = _join_ids(entry_id)
        if format:
            text = kegg_request("get", eid, format)
        else:
            text = kegg_request("get", eid)
        content = json.dumps({"entry_id": eid, "format": format, "entry": text}, ensure_ascii=False)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"entry_id": eid, "format": format}
        return _query_success_response(content, content_preview=content, biological_metadata=meta, query_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("QueryError", str(e), suggestion="Check KEGG entry ID (e.g. hsa:7535, C00001).")


def query_kegg_conv_by_id(target_db: str, source_id: Union[str, List[str]]) -> str:
    """Convert KEGG IDs to/from external DBs by ID. Returns rich JSON: status, content, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        sid = _join_ids(source_id)
        text = kegg_request("conv", target_db, sid)
        content = json.dumps({"target_db": target_db, "source_id": sid, "conversion": text}, ensure_ascii=False)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"target_db": target_db, "source_id": sid}
        return _query_success_response(content, content_preview=content, biological_metadata=meta, query_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("QueryError", str(e), suggestion="Check target database and source IDs.")


def query_kegg_link_by_id(target_db: str, source_id: Union[str, List[str]]) -> str:
    """Query KEGG cross-references by ID. Returns rich JSON: status, content, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        sid = _join_ids(source_id)
        text = kegg_request("link", target_db, sid)
        content = json.dumps({"target_db": target_db, "source_id": sid, "links": text}, ensure_ascii=False)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"target_db": target_db, "source_id": sid}
        return _query_success_response(content, content_preview=content, biological_metadata=meta, query_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("QueryError", str(e), suggestion="Check target database and source IDs.")


def query_kegg_ddi_by_id(drug_id: Union[str, List[str]]) -> str:
    """Query KEGG drug-drug interactions by drug ID. Returns rich JSON: status, content, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        did = _join_ids(drug_id)
        text = kegg_request("ddi", did)
        content = json.dumps({"drug_id": did, "interactions": text}, ensure_ascii=False)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"drug_id": did}
        return _query_success_response(content, content_preview=content, biological_metadata=meta, query_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("QueryError", str(e), suggestion="Check KEGG drug ID (e.g. D00001).")


# ---------- download_*: save to file, return JSON with success + file_path ----------


def download_kegg_info_by_database(database: str, out_path: str) -> str:
    """Download KEGG info by database name to file. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        text = kegg_request("info", database)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"database": database}
        return _download_success_response(out_path, content_preview=_read_preview(out_path), biological_metadata=meta, download_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("DownloadError", str(e), suggestion="Check KEGG database name.")


def download_kegg_list_by_database(database: str, out_path: str, org_or_ids: Optional[Union[str, list]] = None) -> str:
    """Download KEGG list result by database to file. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        if org_or_ids is None:
            text = kegg_request("list", database)
        else:
            sid = _join_ids(org_or_ids) if isinstance(org_or_ids, list) else str(org_or_ids).strip()
            text = kegg_request("list", database, sid)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"database": database, "org_or_ids": org_or_ids}
        return _download_success_response(out_path, content_preview=_read_preview(out_path), biological_metadata=meta, download_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("DownloadError", str(e), suggestion="Check KEGG database and organism/ID.")


def download_kegg_find_by_database(database: str, query: str, out_path: str, option: Optional[str] = None) -> str:
    """Download KEGG find result by database to file. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        if option:
            text = kegg_request("find", database, query, option)
        else:
            text = kegg_request("find", database, query)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"database": database, "query": query, "option": option}
        return _download_success_response(out_path, content_preview=_read_preview(out_path), biological_metadata=meta, download_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("DownloadError", str(e), suggestion="Check KEGG database and query string.")


def download_kegg_entry_by_id(entry_id: Union[str, List[str]], out_path: str, format: Optional[str] = None) -> str:
    """Download KEGG entry by ID to file. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        eid = _join_ids(entry_id) if isinstance(entry_id, list) else str(entry_id).strip()
        if format:
            text = kegg_request("get", eid, format)
        else:
            text = kegg_request("get", eid)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"entry_id": eid, "format": format}
        return _download_success_response(out_path, content_preview=_read_preview(out_path), biological_metadata=meta, download_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("DownloadError", str(e), suggestion="Check KEGG entry ID.")


def download_kegg_conv_by_id(target_db: str, source_id: Union[str, List[str]], out_path: str) -> str:
    """Download KEGG conv result by ID to file. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        sid = _join_ids(source_id)
        text = kegg_request("conv", target_db, sid)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"target_db": target_db, "source_id": sid}
        return _download_success_response(out_path, content_preview=_read_preview(out_path), biological_metadata=meta, download_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("DownloadError", str(e), suggestion="Check target database and source IDs.")


def download_kegg_link_by_id(target_db: str, source_id: Union[str, List[str]], out_path: str) -> str:
    """Download KEGG link result by ID to file. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        sid = _join_ids(source_id)
        text = kegg_request("link", target_db, sid)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"target_db": target_db, "source_id": sid}
        return _download_success_response(out_path, content_preview=_read_preview(out_path), biological_metadata=meta, download_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("DownloadError", str(e), suggestion="Check target database and source IDs.")


def download_kegg_ddi_by_id(drug_id: Union[str, List[str]], out_path: str) -> str:
    """Download KEGG ddi result by drug ID to file. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        did = _join_ids(drug_id)
        text = kegg_request("ddi", did)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"drug_id": did}
        return _download_success_response(out_path, content_preview=_read_preview(out_path), biological_metadata=meta, download_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("DownloadError", str(e), suggestion="Check KEGG drug ID.")


# Backward-compat aliases (tools_agent and others use kegg_info, kegg_find, kegg_get)
kegg_info = query_kegg_info_by_database
kegg_list = query_kegg_list_by_database
kegg_find = query_kegg_find_by_database


def kegg_get(entry_id: Union[str, List[str]], format: Optional[str] = None, option: Optional[str] = None) -> str:
    """Get KEGG entry/entries. option is alias for format (backward compat). Returns rich JSON."""
    return query_kegg_entry_by_id(entry_id, format=format or option)


kegg_conv = query_kegg_conv_by_id
kegg_link = query_kegg_link_by_id
kegg_ddi = query_kegg_ddi_by_id


__all__ = [
    "query_kegg_info_by_database",
    "query_kegg_list_by_database",
    "query_kegg_find_by_database",
    "query_kegg_entry_by_id",
    "query_kegg_conv_by_id",
    "query_kegg_link_by_id",
    "query_kegg_ddi_by_id",
    "download_kegg_info_by_database",
    "download_kegg_list_by_database",
    "download_kegg_find_by_database",
    "download_kegg_entry_by_id",
    "download_kegg_conv_by_id",
    "download_kegg_link_by_id",
    "download_kegg_ddi_by_id",
    "kegg_info",
    "kegg_list",
    "kegg_find",
    "kegg_get",
    "kegg_conv",
    "kegg_link",
    "kegg_ddi",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="KEGG operations: query_* (return JSON with content) and download_* (return JSON with file_path). Academic use only."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run query_* and download_* samples; output under example/database/kegg",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="example/database/kegg",
        help="Output directory. Default example/database/kegg.",
    )
    args = parser.parse_args()

    if not args.test:
        print("Use --test to run operations tests.")
        exit(0)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

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

    print("=== query_* (return rich JSON: status, content, content_preview, biological_metadata, execution_context) ===")

    # 1. info
    res_info = query_kegg_info_by_database("pathway")
    _print_query("query_kegg_info_by_database('pathway')", res_info)
    with open(os.path.join(out_dir, "query_info_pathway_sample.txt"), "w", encoding="utf-8") as f:
        f.write(res_info)
    print(f"  full JSON saved to {os.path.join(out_dir, 'query_info_pathway_sample.txt')}")

    # 2. list
    res_list = query_kegg_list_by_database("pathway", "hsa")
    _print_query("query_kegg_list_by_database('pathway', 'hsa')", res_list)
    with open(os.path.join(out_dir, "query_list_pathway_hsa_sample.txt"), "w", encoding="utf-8") as f:
        f.write(res_list)

    # 3. find
    res_find = query_kegg_find_by_database("genes", "p53")
    _print_query("query_kegg_find_by_database('genes', 'p53')", res_find)

    # 4. get (by ID: hsa:7535)
    res_get = query_kegg_entry_by_id("hsa:7535")
    _print_query("query_kegg_entry_by_id('hsa:7535')", res_get)

    # 5. conv (by ID: hsa:7535)
    res_conv = query_kegg_conv_by_id("ncbi-geneid", "hsa:7535")
    _print_query("query_kegg_conv_by_id('ncbi-geneid', 'hsa:7535')", res_conv)

    # 6. link (by ID: hsa:7535)
    res_link = query_kegg_link_by_id("pathway", "hsa:7535")
    _print_query("query_kegg_link_by_id('pathway', 'hsa:7535')", res_link)

    # 7. ddi (by ID: D00001)
    res_ddi = query_kegg_ddi_by_id("D00001")
    _print_query("query_kegg_ddi_by_id('D00001')", res_ddi)

    print("=== download_* (return rich JSON: status, file_info, content_preview, biological_metadata, execution_context) ===")
    for name, res in [
        ("download_kegg_info_by_database", download_kegg_info_by_database("pathway", os.path.join(out_dir, "kegg_info_pathway.txt"))),
        ("download_kegg_list_by_database", download_kegg_list_by_database("pathway", os.path.join(out_dir, "kegg_list_pathway_hsa.txt"), "hsa")),
        ("download_kegg_find_by_database", download_kegg_find_by_database("genes", "p53", os.path.join(out_dir, "kegg_find_genes_p53.txt"))),
        ("download_kegg_entry_by_id", download_kegg_entry_by_id("hsa:7535", os.path.join(out_dir, "kegg_entry_hsa7535.txt"))),
        ("download_kegg_conv_by_id", download_kegg_conv_by_id("ncbi-geneid", "hsa:7535", os.path.join(out_dir, "kegg_conv_hsa7535.txt"))),
        ("download_kegg_link_by_id", download_kegg_link_by_id("pathway", "hsa:7535", os.path.join(out_dir, "kegg_link_hsa7535.txt"))),
        ("download_kegg_ddi_by_id", download_kegg_ddi_by_id("D00001", os.path.join(out_dir, "kegg_ddi_D00001.txt"))),
    ]:
        dl_obj = json.loads(res)
        print(f"  {name}: {dl_obj}")

    print(f"Done. Output under {out_dir}")
