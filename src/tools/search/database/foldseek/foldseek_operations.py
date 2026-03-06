"""
FoldSeek operations: single exit for submit, query, and download; all return rich JSON.

Success: status, file_info (download) or content (query), content_preview, biological_metadata, execution_context.
Error: status "error", error { type, message, suggestion }, file_info null.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from .foldseek_submit import (
        submit_foldseek_job as _submit_job,
        query_foldseek_status as _query_status,
        wait_foldseek_complete as _wait_complete,
    )
    from .download_foldseek_m8 import (
        download_foldseek_m8 as _download_m8,
        prepare_foldseek_sequences as _prepare_sequences,
    )
except ImportError:
    _dir = Path(__file__).resolve().parent
    if _dir.name == "foldseek" and str(_dir.parents[4]) not in sys.path:
        sys.path.insert(0, str(_dir.parents[4]))
    from src.tools.search.database.foldseek.foldseek_submit import (
        submit_foldseek_job as _submit_job,
        query_foldseek_status as _query_status,
        wait_foldseek_complete as _wait_complete,
    )
    from src.tools.search.database.foldseek.download_foldseek_m8 import (
        download_foldseek_m8 as _download_m8,
        prepare_foldseek_sequences as _prepare_sequences,
    )


_PREVIEW_LEN = 500
_SOURCE_FOLDSEEK = "FoldSeek"


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
    source: str = _SOURCE_FOLDSEEK,
) -> str:
    """Build JSON for download success: status, file_info, content_preview, biological_metadata, execution_context."""
    path = Path(file_path)
    file_size = path.stat().st_size if path.exists() else 0
    fmt = path.suffix.lstrip(".").lower() or "txt"
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
    source: str = _SOURCE_FOLDSEEK,
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


# ---------- query_*: return rich JSON (status, content, content_preview, biological_metadata, execution_context) ----------


def query_foldseek_submit_job(
    pdb_file_path: str,
    databases: Optional[List[str]] = None,
    mode: str = "3diaa",
) -> str:
    """Submit PDB to FoldSeek and return job_id. Returns rich JSON: status, content, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        job_id = _submit_job(pdb_file_path, databases=databases, mode=mode)
        payload = {"job_id": job_id, "pdb_file": pdb_file_path, "mode": mode}
        content = json.dumps(payload, ensure_ascii=False)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"pdb_file": pdb_file_path, "mode": mode, "job_id": job_id}
        return _query_success_response(content, content_preview=content, biological_metadata=meta, query_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("SubmitError", str(e), suggestion="Check PDB file path and FoldSeek API availability.")


def query_foldseek_job_status(job_id: str) -> str:
    """Query FoldSeek job status. Returns rich JSON: status, content, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        job_status = _query_status(job_id)
        payload = {"job_id": job_id, "job_status": job_status}
        content = json.dumps(payload, ensure_ascii=False)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"job_id": job_id, "job_status": job_status}
        return _query_success_response(content, content_preview=content, biological_metadata=meta, query_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("QueryError", str(e), suggestion="Check job_id is valid.")


def query_foldseek_wait_complete(job_id: str, poll_interval: float = 2.0) -> str:
    """Poll until FoldSeek job is COMPLETE or ERROR. Returns rich JSON: status, content, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        _wait_complete(job_id, poll_interval=poll_interval)
        payload = {"job_id": job_id, "job_status": "COMPLETE"}
        content = json.dumps(payload, ensure_ascii=False)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"job_id": job_id, "job_status": "COMPLETE", "poll_interval": poll_interval}
        return _query_success_response(content, content_preview=content, biological_metadata=meta, query_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("WaitError", str(e), suggestion="Check job_id or increase poll_interval.")


# ---------- download_*: save to file, return JSON with success + file_path ----------


def _read_preview(path: str, max_chars: int = _PREVIEW_LEN) -> str:
    """Read first max_chars from file for content_preview."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(max_chars)
    except Exception:
        return ""


def download_foldseek_alignments_m8(
    job_id: str,
    out_dir: str,
    databases: Optional[List[str]] = None,
) -> str:
    """Download alignment result files (.m8) for a FoldSeek job_id. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        m8_files = _download_m8(job_id, out_dir, databases=databases)
        # Use first m8 file as primary file_info, list all in biological_metadata
        primary = m8_files[0] if m8_files else ""
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"job_id": job_id, "m8_file_count": len(m8_files), "m8_files": m8_files}
        preview = _read_preview(primary) if primary else ""
        return _download_success_response(primary, content_preview=preview, biological_metadata=meta, download_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("DownloadError", str(e), suggestion="Check job_id and ensure job is COMPLETE.")


def download_foldseek_sequences_to_fasta(
    alignments_files: List[str],
    output_fasta: str,
    protect_start: int,
    protect_end: int,
) -> str:
    """Extract sequences from FoldSeek alignments covering the protected region into a FASTA file. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        total = _prepare_sequences(
            alignments_files,
            Path(output_fasta),
            protect_start,
            protect_end,
        )
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {
            "protect_start": protect_start,
            "protect_end": protect_end,
            "total_sequences": total,
            "input_files": len(alignments_files),
        }
        return _download_success_response(output_fasta, content_preview=_read_preview(output_fasta), biological_metadata=meta, download_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("DownloadError", str(e), suggestion="Check alignment file paths and protect region.")


# ---------- high-level pipeline ----------


def download_foldseek_results_by_pdb_file(
    pdb_file_path: str,
    protect_start: int,
    protect_end: int,
    out_dir: Optional[str] = None,
) -> str:
    """
    Submit PDB to FoldSeek, wait for completion, download alignments, extract sequences to FASTA.
    Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context.
    """
    t0 = time.perf_counter()
    try:
        out_dir = out_dir or "download/FoldSeek"
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        job_id = _submit_job(pdb_file_path)
        _wait_complete(job_id)
        download_files = _download_m8(job_id, out_path)
        stem = os.path.basename(pdb_file_path).replace(".pdb", "")
        foldseek_fasta = out_path / f"{stem}_{datetime.now().strftime('%Y%m%d%H%M%S')}.fasta"
        total_sequences = _prepare_sequences(
            download_files, foldseek_fasta, protect_start, protect_end
        )
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {
            "pdb_file": pdb_file_path,
            "job_id": job_id,
            "protect_start": protect_start,
            "protect_end": protect_end,
            "total_sequences": total_sequences,
            "m8_file_count": len(download_files),
        }
        return _download_success_response(
            str(foldseek_fasta),
            content_preview=_read_preview(str(foldseek_fasta)),
            biological_metadata=meta,
            download_time_ms=elapsed_ms,
        )
    except Exception as e:
        return _error_response("PipelineError", str(e), suggestion="Check PDB file path and FoldSeek API availability.")


__all__ = [
    "query_foldseek_submit_job",
    "query_foldseek_job_status",
    "query_foldseek_wait_complete",
    "download_foldseek_alignments_m8",
    "download_foldseek_sequences_to_fasta",
    "download_foldseek_results_by_pdb_file",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="FoldSeek operations: query_foldseek_* (return JSON with content) and download_foldseek_* (return JSON with file_path)."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run download_foldseek_results_by_pdb_file (full pipeline); output under example/database/foldseek",
    )
    parser.add_argument("--pdb_file", type=str, default="example/database/alphafold/A0A1B0GTW7.pdb")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument(
        "--protect_start",
        type=int,
        default=1,
        help="Protected region start (1-based query residue). Only alignments covering [protect_start, protect_end] are kept.",
    )
    parser.add_argument(
        "--protect_end",
        type=int,
        default=10,
        help="Protected region end (1-based query residue). Default 10 is arbitrary; set to your region of interest.",
    )
    args = parser.parse_args()

    if not args.test:
        print("Use --test to run operations tests.")
        sys.exit(0)

    out_dir = os.path.join("example", "database", "foldseek")
    os.makedirs(out_dir, exist_ok=True)
    search_dir = os.path.join(out_dir, "search")
    os.makedirs(search_dir, exist_ok=True)
    out_dir = args.out_dir or search_dir
    protect_start = args.protect_start
    protect_end = args.protect_end

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

    print("=== download_* (return rich JSON: status, file_info, content_preview, biological_metadata, execution_context) ===")
    res = download_foldseek_results_by_pdb_file(
        args.pdb_file, protect_start, protect_end, out_dir=out_dir
    )
    dl_obj = json.loads(res)
    print(f"  download_foldseek_results_by_pdb_file: {dl_obj}")
    print(f"Done. Output under {out_dir}")
