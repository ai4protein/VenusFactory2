"""
InterPro operations: single exit for query and download; both return rich JSON.

Success: status, file_info (download) or content (query), content_preview, biological_metadata, execution_context.
Error: status "error", error { type, message, suggestion }, file_info null.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .interpro_metadata import (
        query_interpro_metadata as _raw_query_metadata,
        download_interpro_metadata as _raw_download_metadata,
    )
    from .interpro_proteins import (
        query_interpro_by_uniprot as _raw_query_by_uniprot,
        download_interpro_by_uniprot as _raw_download_by_uniprot,
        query_interpro_proteins as _raw_query_proteins,
        download_interpro_proteins as _raw_download_proteins,
        query_interpro_uniprot_list as _raw_query_uniprot_list,
        download_interpro_uniprot_list as _raw_download_uniprot_list,
    )
except ImportError:
    _dir = Path(__file__).resolve().parent
    if _dir.name == "interpro" and str(_dir.parents[3]) not in sys.path:
        sys.path.insert(0, str(_dir.parents[3]))
    from src.tools.database.interpro.interpro_metadata import (
        query_interpro_metadata as _raw_query_metadata,
        download_interpro_metadata as _raw_download_metadata,
    )
    from src.tools.database.interpro.interpro_proteins import (
        query_interpro_by_uniprot as _raw_query_by_uniprot,
        download_interpro_by_uniprot as _raw_download_by_uniprot,
        query_interpro_proteins as _raw_query_proteins,
        download_interpro_proteins as _raw_download_proteins,
        query_interpro_uniprot_list as _raw_query_uniprot_list,
        download_interpro_uniprot_list as _raw_download_uniprot_list,
    )


_PREVIEW_LEN = 500
_SOURCE_INTERPRO = "InterPro"


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
    source: str = _SOURCE_INTERPRO,
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
    source: str = _SOURCE_INTERPRO,
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


def query_interpro_metadata_by_id(interpro_id: str) -> str:
    """Query InterPro entry/family metadata by InterPro ID. Returns rich JSON: status, content, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        text = _raw_query_metadata(interpro_id)
        data = json.loads(text)
        if isinstance(data, dict) and data.get("success") is False:
            return _error_response("QueryError", data.get("error", "Not found"), suggestion="Check InterPro ID format (e.g. IPR001557).")
        content = json.dumps(data, ensure_ascii=False)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta: Dict[str, Any] = {"interpro_id": interpro_id}
        if isinstance(data, dict):
            meta["name"] = data.get("metadata", {}).get("name", {}).get("name", "") if isinstance(data.get("metadata"), dict) else ""
            meta["type"] = data.get("metadata", {}).get("type", "") if isinstance(data.get("metadata"), dict) else ""
        return _query_success_response(content, content_preview=content, biological_metadata=meta, query_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("QueryError", str(e), suggestion="Check InterPro ID format (e.g. IPR001557).")


def query_interpro_annotations_by_uniprot_id(uniprot_id: str) -> str:
    """Query InterPro entries and GO annotations for a UniProt ID. Returns rich JSON: status, content, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        text = _raw_query_by_uniprot(uniprot_id)
        data = json.loads(text)
        if isinstance(data, dict) and data.get("success") is False:
            return _error_response("QueryError", data.get("error_message", "Unknown error"), suggestion="Check UniProt ID format (e.g. P40925).")
        content = json.dumps(data, ensure_ascii=False)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta: Dict[str, Any] = {"uniprot_id": uniprot_id}
        if isinstance(data, dict):
            meta["num_entries"] = data.get("num_entries", 0)
            meta["protein_name"] = data.get("basic_info", {}).get("protein_name", "")
        return _query_success_response(content, content_preview=content, biological_metadata=meta, query_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("QueryError", str(e), suggestion="Check UniProt ID format (e.g. P40925).")


def query_interpro_proteins_by_id(
    interpro_id: str,
    page_size: int = 200,
    max_results: Optional[int] = None,
) -> str:
    """Query protein list for an InterPro ID. Returns rich JSON: status, content, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        text = _raw_query_proteins(interpro_id, page_size=page_size, max_results=max_results)
        data = json.loads(text)
        if isinstance(data, dict) and data.get("success") is False:
            return _error_response("QueryError", data.get("error", "Unknown error"), suggestion="Check InterPro ID format (e.g. IPR001557).")
        content = json.dumps(data, ensure_ascii=False)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        num_proteins = data.get("metadata", {}).get("num_proteins", 0) if isinstance(data, dict) else 0
        meta = {"interpro_id": interpro_id, "num_proteins": num_proteins, "page_size": page_size}
        if max_results is not None:
            meta["max_results"] = max_results
        return _query_success_response(content, content_preview=content, biological_metadata=meta, query_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("QueryError", str(e), suggestion="Check InterPro ID format (e.g. IPR001557).")


def query_interpro_uniprot_list_by_id(
    interpro_id: str,
    filter_name: str = None,
    page_size: int = 200,
    max_results: Optional[int] = None,
) -> str:
    """Query UniProt ID list for an InterPro entry. Returns rich JSON: status, content, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        text = _raw_query_uniprot_list(interpro_id, filter_name=filter_name, page_size=page_size, max_results=max_results)
        data = json.loads(text)
        if isinstance(data, dict) and data.get("success") is False:
            return _error_response("QueryError", data.get("error", "Unknown error"), suggestion="Check InterPro ID format (e.g. IPR001557).")
        content = json.dumps(data, ensure_ascii=False)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        count = data.get("count", 0) if isinstance(data, dict) else 0
        meta = {"interpro_id": interpro_id, "accession_count": count, "page_size": page_size}
        if max_results is not None:
            meta["max_results"] = max_results
        return _query_success_response(content, content_preview=content, biological_metadata=meta, query_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("QueryError", str(e), suggestion="Check InterPro ID format (e.g. IPR001557).")


# ---------- download_*: save to file, return JSON with success + file_path ----------


def download_interpro_metadata_by_id(interpro_id: str, out_dir: str) -> str:
    """Download InterPro entry metadata by InterPro ID to JSON file. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{interpro_id}.json")
        text = _raw_query_metadata(interpro_id)
        data = json.loads(text)
        if isinstance(data, dict) and data.get("success") is False:
            return _error_response("DownloadError", data.get("error", "Not found"), suggestion="Check InterPro ID format (e.g. IPR001557).")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta: Dict[str, Any] = {"interpro_id": interpro_id}
        return _download_success_response(out_path, content_preview=_read_preview(out_path), biological_metadata=meta, download_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("DownloadError", str(e), suggestion="Check InterPro ID format (e.g. IPR001557).")


def download_interpro_annotations_by_uniprot_id(uniprot_id: str, out_dir: str) -> str:
    """Download InterPro annotation by UniProt ID to JSON file. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{uniprot_id}_interpro.json")
        text = _raw_query_by_uniprot(uniprot_id)
        data = json.loads(text)
        if isinstance(data, dict) and data.get("success") is False:
            return _error_response("DownloadError", data.get("error_message", "Unknown error"), suggestion="Check UniProt ID format (e.g. P40925).")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta: Dict[str, Any] = {"uniprot_id": uniprot_id}
        if isinstance(data, dict):
            meta["num_entries"] = data.get("num_entries", 0)
        return _download_success_response(out_path, content_preview=_read_preview(out_path), biological_metadata=meta, download_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("DownloadError", str(e), suggestion="Check UniProt ID format (e.g. P40925).")


def download_interpro_proteins_by_id(
    interpro_id: str,
    out_dir: str,
    max_results: Optional[int] = None,
) -> str:
    """Download family proteins for an InterPro ID. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        interpro_dir = os.path.join(out_dir, interpro_id)
        os.makedirs(interpro_dir, exist_ok=True)
        detail_path = os.path.join(interpro_dir, "detail.json")
        text = _raw_query_proteins(interpro_id, page_size=20, max_results=max_results)
        data = json.loads(text)
        if isinstance(data, dict) and data.get("success") is False:
            return _error_response("DownloadError", data.get("error", "Unknown error"), suggestion="Check InterPro ID format (e.g. IPR001557).")
        results = data.get("results", []) if isinstance(data, dict) else []
        if not results:
            return _error_response("DownloadError", f"No data found for {interpro_id}", suggestion="Try a different InterPro ID.")
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(results, f)
        meta_data = {"metadata": {"accession": interpro_id}, "num_proteins": len(results)}
        with open(os.path.join(interpro_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta_data, f)
        uids = [d["metadata"]["accession"] for d in results if "metadata" in d]
        with open(os.path.join(interpro_dir, "uids.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(uids))
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"interpro_id": interpro_id, "num_proteins": len(results)}
        if max_results is not None:
            meta["max_results"] = max_results
        return _download_success_response(detail_path, content_preview=_read_preview(detail_path), biological_metadata=meta, download_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("DownloadError", str(e), suggestion="Check InterPro ID format (e.g. IPR001557).")


def download_interpro_uniprot_list_by_id(
    interpro_id: str,
    out_dir: str,
    protein_name: str = "",
    chunk_size: int = 5000,
    filter_name: str = None,
    page_size: int = 200,
    max_results: Optional[int] = None,
) -> str:
    """Download UniProt ID list for an InterPro entry to chunked txt files. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        os.makedirs(out_dir, exist_ok=True)
        text = _raw_query_uniprot_list(interpro_id, filter_name=filter_name, page_size=page_size, max_results=max_results)
        data = json.loads(text)
        if isinstance(data, dict) and data.get("success") is False:
            return _error_response("DownloadError", data.get("error", "Unknown error"), suggestion="Check InterPro ID format (e.g. IPR001557).")
        names = data.get("accessions", []) if isinstance(data, dict) else []
        length = len(names)
        prefix = protein_name or interpro_id
        max_i = (length // chunk_size) + 1 if chunk_size else 1
        first_file = None
        for i in range(max_i):
            chunk = names[i * chunk_size: (i + 1) * chunk_size] if chunk_size else names
            out_file = os.path.join(out_dir, f"af_raw_{prefix}_{i}.txt")
            with open(out_file, "w", encoding="utf-8") as f:
                f.write("\n".join(chunk))
            if first_file is None:
                first_file = out_file
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"interpro_id": interpro_id, "accession_count": length}
        if max_results is not None:
            meta["max_results"] = max_results
        return _download_success_response(first_file or out_dir, content_preview=_read_preview(first_file or ""), biological_metadata=meta, download_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("DownloadError", str(e), suggestion="Check InterPro ID format (e.g. IPR001557).")


__all__ = [
    "query_interpro_metadata_by_id",
    "query_interpro_annotations_by_uniprot_id",
    "query_interpro_proteins_by_id",
    "query_interpro_uniprot_list_by_id",
    "download_interpro_metadata_by_id",
    "download_interpro_annotations_by_uniprot_id",
    "download_interpro_proteins_by_id",
    "download_interpro_uniprot_list_by_id",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="InterPro operations: query_* (return JSON with content) and download_* (return JSON with file_path)."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run query_* and download_* samples; output under example/database/interpro",
    )
    parser.add_argument("--interpro_id", type=str, default="IPR001557", help="InterPro ID for test. Default IPR001557.")
    parser.add_argument("--uniprot_id", type=str, default="P40925", help="UniProt ID for test. Default P40925.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="example/database/interpro",
        help="Output directory. Default example/database/interpro.",
    )
    args = parser.parse_args()

    if not args.test:
        print("Use --test to run operations tests.")
        sys.exit(0)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    meta_dir = os.path.join(out_dir, "metadata")
    proteins_dir = os.path.join(out_dir, "proteins")
    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(proteins_dir, exist_ok=True)

    interpro_id = args.interpro_id
    uniprot_id = args.uniprot_id

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
    res_meta = query_interpro_metadata_by_id(interpro_id)
    _print_query(f"query_interpro_metadata_by_id({interpro_id})", res_meta)
    with open(os.path.join(meta_dir, f"query_{interpro_id}_metadata_sample.txt"), "w", encoding="utf-8") as f:
        f.write(res_meta)
    print(f"  full JSON saved to {os.path.join(meta_dir, f'query_{interpro_id}_metadata_sample.txt')}")

    res_annot = query_interpro_annotations_by_uniprot_id(uniprot_id)
    _print_query(f"query_interpro_annotations_by_uniprot_id({uniprot_id})", res_annot)
    with open(os.path.join(proteins_dir, f"query_{uniprot_id}_annotations_sample.txt"), "w", encoding="utf-8") as f:
        f.write(res_annot)

    res_proteins = query_interpro_proteins_by_id(interpro_id, page_size=10, max_results=15)
    _print_query(f"query_interpro_proteins_by_id({interpro_id}, max_results=15)", res_proteins)

    res_unilist = query_interpro_uniprot_list_by_id(interpro_id, page_size=20, max_results=30)
    _print_query(f"query_interpro_uniprot_list_by_id({interpro_id}, max_results=30)", res_unilist)

    print("=== download_* (return rich JSON: status, file_info, content_preview, biological_metadata, execution_context) ===")
    for name, res in [
        (
            f"download_interpro_metadata_by_id({interpro_id})",
            download_interpro_metadata_by_id(interpro_id, meta_dir),
        ),
        (
            f"download_interpro_annotations_by_uniprot_id({uniprot_id})",
            download_interpro_annotations_by_uniprot_id(uniprot_id, proteins_dir),
        ),
        (
            f"download_interpro_proteins_by_id({interpro_id}, max_results=15)",
            download_interpro_proteins_by_id(interpro_id, proteins_dir, max_results=15),
        ),
        (
            f"download_interpro_uniprot_list_by_id({interpro_id}, max_results=30)",
            download_interpro_uniprot_list_by_id(interpro_id, proteins_dir, chunk_size=50, page_size=20, max_results=30),
        ),
    ]:
        dl_obj = json.loads(res)
        print(f"  {name}: {dl_obj}")

    print(f"Done. Output under {out_dir}")
