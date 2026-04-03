"""
NCBI operations: single exit for query and download; both return rich JSON.

Success: status, file_info (download) or content (query), content_preview, biological_metadata, execution_context.
Error: status "error", error { type, message, suggestion }, file_info null.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.tools.path_sanitizer import to_client_file_path

try:
    from .ncbi_sequence import query_ncbi_seq
except ImportError:
    _dir = Path(__file__).resolve().parent
    if _dir.name == "ncbi" and str(_dir.parents[3]) not in sys.path:
        sys.path.insert(0, str(_dir.parents[3]))
    from src.tools.database.ncbi.ncbi_sequence import query_ncbi_seq

from src.tools.database.ncbi.ncbi_metadata import query_ncbi_meta
from src.tools.database.ncbi.ncbi_blast import query_ncbi_blast as _blast_query
from src.tools.database.ncbi.ncbi_clinvar import (
    query_clinvar_variants,
    clinvar_esearch,
    clinvar_esummary,
    clinvar_efetch,
)
from src.tools.database.ncbi.fetch_gene_data import fetch_gene_by_id, fetch_gene_by_symbol
from src.tools.database.ncbi.batch_gene_lookup import (
    batch_esearch,
    batch_esummary,
    batch_lookup_by_ids,
    batch_lookup_by_symbols,
)
from src.tools.database.ncbi.query_gene import (
    esearch as gene_esearch,
    esummary as gene_esummary,
    efetch as gene_efetch,
)


_PREVIEW_LEN = 500
_SOURCE_NCBI = "NCBI"


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
    source: str = _SOURCE_NCBI,
) -> str:
    """Build JSON for download success: status, file_info, content_preview, biological_metadata, execution_context."""
    path = Path(file_path)
    file_size = path.stat().st_size if path.exists() else 0
    fmt = path.suffix.lstrip(".").lower() or "json"
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
    source: str = _SOURCE_NCBI,
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


def _is_error_json(text: str) -> bool:
    if not text.strip().startswith("{"):
        return False
    try:
        data = json.loads(text)
        return str(data.get("success", True)).lower() == "false" or "error" in data
    except Exception:
        return False


# ---------- string/general queries ----------

def query_ncbi_sequence(ncbi_id: str, db: str = "protein") -> str:
    """Query NCBI sequence by accession. Returns FASTA text wrapped in rich JSON."""
    t0 = time.perf_counter()
    try:
        res = query_ncbi_seq(ncbi_id, db=db)
        if _is_error_json(res):
            return _error_response("QueryError", res, suggestion="Check NCBI ID and database.")
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"ncbi_id": ncbi_id, "db": db}
        return _query_success_response(res, content_preview=res, biological_metadata=meta, query_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("QueryError", str(e), suggestion="Ensure network is accessible and ID is correct.")


def download_ncbi_sequence(ncbi_id: str, out_path: str, db: str = "protein") -> str:
    """Download NCBI sequence as FASTA. Returns rich JSON with file_info."""
    t0 = time.perf_counter()
    try:
        res = query_ncbi_seq(ncbi_id, db=db)
        if _is_error_json(res):
            return _error_response("DownloadError", res, suggestion="Check NCBI ID and database.")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(res)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"ncbi_id": ncbi_id, "db": db}
        return _download_success_response(out_path, content_preview=_read_preview(out_path), biological_metadata=meta, download_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("DownloadError", str(e))


def query_ncbi_metadata(ncbi_id: str, db: str = "protein", rettype: str = "gb") -> str:
    """Query NCBI metadata by accession. Returns GenBank/XML text wrapped in rich JSON."""
    t0 = time.perf_counter()
    try:
        res = query_ncbi_meta(ncbi_id, db=db, rettype=rettype)
        if _is_error_json(res):
            return _error_response("QueryError", res)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"ncbi_id": ncbi_id, "db": db, "rettype": rettype}
        return _query_success_response(res, content_preview=res, biological_metadata=meta, query_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("QueryError", str(e))


def download_ncbi_metadata(ncbi_id: str, out_path: str, db: str = "protein", rettype: str = "gb") -> str:
    """Download NCBI metadata by accession. Returns rich JSON with file_info."""
    t0 = time.perf_counter()
    try:
        res = query_ncbi_meta(ncbi_id, db=db, rettype=rettype)
        if _is_error_json(res):
            return _error_response("DownloadError", res)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(res)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"ncbi_id": ncbi_id, "db": db, "rettype": rettype}
        return _download_success_response(out_path, content_preview=_read_preview(out_path), biological_metadata=meta, download_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("DownloadError", str(e))


def query_ncbi_blast(sequence: str, program: str = "blastp", database: str = "swissprot", hitlist_size: int = 50, alignments: int = 25, format_type: str = "XML", entrez_query: Optional[str] = None) -> str:
    """Submit BLAST search to NCBI via Biopython. Returns XML string wrapped in rich JSON."""
    t0 = time.perf_counter()
    try:
        res = _blast_query(sequence, program=program, database=database, hitlist_size=hitlist_size, alignments=alignments, format_type=format_type, entrez_query=entrez_query)
        if _is_error_json(res):
            return _error_response("QueryError", res)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"program": program, "database": database, "hitlist_size": hitlist_size}
        return _query_success_response(res, content_preview=res, biological_metadata=meta, query_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("QueryError", str(e))


def download_ncbi_blast(sequence: str, out_path: str, program: str = "blastp", database: str = "swissprot", hitlist_size: int = 50, alignments: int = 25, format_type: str = "XML", entrez_query: Optional[str] = None) -> str:
    """Submit BLAST search and save XML result. Returns rich JSON with file_info."""
    t0 = time.perf_counter()
    try:
        res = _blast_query(sequence, program=program, database=database, hitlist_size=hitlist_size, alignments=alignments, format_type=format_type, entrez_query=entrez_query)
        if _is_error_json(res):
            return _error_response("DownloadError", res)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(res)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"program": program, "database": database, "hitlist_size": hitlist_size}
        return _download_success_response(out_path, content_preview=_read_preview(out_path), biological_metadata=meta, download_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("DownloadError", str(e))

# ---------- clinvar atomic ----------

def query_ncbi_clinvar_esearch(term: str, retmax: int = 20, retstart: int = 0, retmode: str = "json") -> str:
    """ClinVar esearch: search for variation IDs matching term. Returns JSON text wrapped in rich JSON."""
    t0 = time.perf_counter()
    try:
        res = clinvar_esearch(term, retmax=retmax, retstart=retstart, retmode=retmode)
        if _is_error_json(res):
            return _error_response("QueryError", res)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"term": term, "retmax": retmax}
        return _query_success_response(res, content_preview=res, biological_metadata=meta, query_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("QueryError", str(e))

def download_ncbi_clinvar_esearch(term: str, out_path: str, retmax: int = 20, retstart: int = 0, retmode: str = "json") -> str:
    """ClinVar esearch: search for variation IDs matching term and save to file. Returns rich JSON with file_info."""
    t0 = time.perf_counter()
    try:
        res = clinvar_esearch(term, retmax=retmax, retstart=retstart, retmode=retmode)
        if _is_error_json(res):
            return _error_response("DownloadError", res)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(res)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"term": term, "retmax": retmax}
        return _download_success_response(out_path, content_preview=_read_preview(out_path), biological_metadata=meta, download_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("DownloadError", str(e))


def query_ncbi_clinvar_esummary(id_list: List[str], retmode: str = "json") -> str:
    """ClinVar esummary: get summaries for variation IDs. Returns JSON text wrapped in rich JSON."""
    t0 = time.perf_counter()
    try:
        res = clinvar_esummary(id_list, retmode=retmode)
        if _is_error_json(res):
            return _error_response("QueryError", res)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"id_count": len(id_list)}
        return _query_success_response(res, content_preview=res, biological_metadata=meta, query_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("QueryError", str(e))

def download_ncbi_clinvar_esummary(id_list: List[str], out_path: str, retmode: str = "json") -> str:
    """ClinVar esummary: get summaries for variation IDs and save to file. Returns JSON text wrapped in rich JSON."""
    t0 = time.perf_counter()
    try:
        res = clinvar_esummary(id_list, retmode=retmode)
        if _is_error_json(res):
            return _error_response("DownloadError", res)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(res)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"id_count": len(id_list)}
        return _download_success_response(out_path, content_preview=_read_preview(out_path), biological_metadata=meta, download_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("DownloadError", str(e))


def query_ncbi_clinvar_efetch(id_list: List[str], rettype: str = "vcv", retmode: str = "xml") -> str:
    """ClinVar efetch: fetch full records for variation IDs. Returns XML/text wrapped in rich JSON."""
    t0 = time.perf_counter()
    try:
        res = clinvar_efetch(id_list, rettype=rettype, retmode=retmode)
        if _is_error_json(res):
            return _error_response("QueryError", res)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"id_count": len(id_list), "rettype": rettype}
        return _query_success_response(res, content_preview=res, biological_metadata=meta, query_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("QueryError", str(e))

def download_ncbi_clinvar_efetch(id_list: List[str], out_path: str, rettype: str = "vcv", retmode: str = "xml") -> str:
    """ClinVar efetch: fetch full records for variation IDs and save to file. Returns rich JSON with file_info."""
    t0 = time.perf_counter()
    try:
        res = clinvar_efetch(id_list, rettype=rettype, retmode=retmode)
        if _is_error_json(res):
            return _error_response("DownloadError", res)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(res)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"id_count": len(id_list), "rettype": rettype}
        return _download_success_response(out_path, content_preview=_read_preview(out_path), biological_metadata=meta, download_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("DownloadError", str(e))

# ---------- clinvar compound ----------

def query_ncbi_clinvar_variants(term: str, retmax: int = 20) -> str:
    """Search ClinVar variants using esearch. Returns JSON text wrapped in rich JSON."""
    return query_ncbi_clinvar_esearch(term, retmax=retmax)


def download_ncbi_clinvar_variants(term: str, out_path: str, retmax: int = 20) -> str:
    """Search ClinVar variants and save JSON result. Returns rich JSON with file_info."""
    return download_ncbi_clinvar_esearch(term, out_path, retmax=retmax)

# ---------- gene dataset API ----------

def query_ncbi_gene_by_id(gene_id: str) -> str:
    """Fetch gene data by Gene ID. Returns rich JSON."""
    t0 = time.perf_counter()
    try:
        data = fetch_gene_by_id(gene_id)
        if not data:
            return _error_response("QueryError", "Gene not found or error fetching.")
        content = json.dumps(data, ensure_ascii=False)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"gene_id": gene_id}
        return _query_success_response(content, content_preview=content, biological_metadata=meta, query_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("QueryError", str(e))


def download_ncbi_gene_by_id(gene_id: str, out_path: str) -> str:
    """Fetch gene data by Gene ID and save. Returns rich JSON with file_info."""
    t0 = time.perf_counter()
    try:
        data = fetch_gene_by_id(gene_id)
        if not data:
            return _error_response("DownloadError", "Gene not found or error fetching.")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"gene_id": gene_id}
        return _download_success_response(out_path, content_preview=_read_preview(out_path), biological_metadata=meta, download_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("DownloadError", str(e))


def query_ncbi_gene_by_symbol(symbol: str, taxon: str) -> str:
    """Fetch gene data by gene symbol and taxon. Returns rich JSON."""
    t0 = time.perf_counter()
    try:
        data = fetch_gene_by_symbol(symbol, taxon)
        if not data:
            return _error_response("QueryError", "Gene not found or error fetching.")
        content = json.dumps(data, ensure_ascii=False)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"symbol": symbol, "taxon": taxon}
        return _query_success_response(content, content_preview=content, biological_metadata=meta, query_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("QueryError", str(e))


def download_ncbi_gene_by_symbol(symbol: str, taxon: str, out_path: str) -> str:
    """Fetch gene data by gene symbol and taxon and save. Returns rich JSON with file_info."""
    t0 = time.perf_counter()
    try:
        data = fetch_gene_by_symbol(symbol, taxon)
        if not data:
            return _error_response("DownloadError", "Gene not found or error fetching.")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"symbol": symbol, "taxon": taxon}
        return _download_success_response(out_path, content_preview=_read_preview(out_path), biological_metadata=meta, download_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("DownloadError", str(e))


# ---------- gene utilities (esearch, esummary, efetch) ----------

def query_ncbi_gene_esearch(query: str, retmax: int = 20) -> str:
    """Search NCBI Gene database and return list of Gene IDs. Returns JSON list of IDs wrapped in rich JSON."""
    t0 = time.perf_counter()
    try:
        ids = gene_esearch(query, retmax=retmax)
        content = json.dumps(ids, ensure_ascii=False)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"query": query, "retmax": retmax, "id_count": len(ids)}
        return _query_success_response(content, content_preview=content, biological_metadata=meta, query_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("QueryError", str(e))


def download_ncbi_gene_esearch(query: str, out_path: str, retmax: int = 20) -> str:
    """Search NCBI Gene database and save list of Gene IDs to JSON. Returns rich JSON with file_info."""
    t0 = time.perf_counter()
    try:
        ids = gene_esearch(query, retmax=retmax)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(ids, f, indent=2, ensure_ascii=False)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"query": query, "retmax": retmax, "id_count": len(ids)}
        return _download_success_response(out_path, content_preview=_read_preview(out_path), biological_metadata=meta, download_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("DownloadError", str(e))

def query_ncbi_gene_esummary(gene_ids: List[str]) -> str:
    """Get document summaries for Gene IDs. Returns raw JSON object wrapped in rich JSON."""
    t0 = time.perf_counter()
    try:
        d = gene_esummary(gene_ids)
        content = json.dumps(d, ensure_ascii=False)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"id_count": len(gene_ids)}
        return _query_success_response(content, content_preview=content, biological_metadata=meta, query_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("QueryError", str(e))


def download_ncbi_gene_esummary(gene_ids: List[str], out_path: str) -> str:
    """Get document summaries for Gene IDs and save to JSON. Returns rich JSON with file_info."""
    t0 = time.perf_counter()
    try:
        d = gene_esummary(gene_ids)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"id_count": len(gene_ids)}
        return _download_success_response(out_path, content_preview=_read_preview(out_path), biological_metadata=meta, download_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("DownloadError", str(e))


def query_ncbi_gene_efetch(gene_ids: List[str], retmode: str = 'xml') -> str:
    """Fetch full gene records by IDs. Returns XML/text wrapped in rich JSON."""
    t0 = time.perf_counter()
    try:
        res = gene_efetch(gene_ids, retmode=retmode)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"id_count": len(gene_ids), "retmode": retmode}
        return _query_success_response(res, content_preview=res, biological_metadata=meta, query_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("QueryError", str(e))


def download_ncbi_gene_efetch(gene_ids: List[str], out_path: str, retmode: str = 'xml') -> str:
    """Fetch full gene records by IDs and save to file. Returns rich JSON with file_info."""
    t0 = time.perf_counter()
    try:
        res = gene_efetch(gene_ids, retmode=retmode)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(res)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"id_count": len(gene_ids), "retmode": retmode}
        return _download_success_response(out_path, content_preview=_read_preview(out_path), biological_metadata=meta, download_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("DownloadError", str(e))

# ---------- gene batch operations ----------

def query_ncbi_batch_esearch(queries: List[str], organism: Optional[str] = None) -> str:
    """Search for multiple gene symbols and return their IDs. Returns dict mapping wrapped in rich JSON."""
    t0 = time.perf_counter()
    try:
        d = batch_esearch(queries, organism=organism)
        content = json.dumps(d, ensure_ascii=False)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"query_count": len(queries), "organism": organism}
        return _query_success_response(content, content_preview=content, biological_metadata=meta, query_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("QueryError", str(e))


def download_ncbi_batch_esearch(queries: List[str], out_path: str, organism: Optional[str] = None) -> str:
    """Search for multiple gene symbols and return their IDs, saved as JSON. Returns rich JSON with file_info."""
    t0 = time.perf_counter()
    try:
        d = batch_esearch(queries, organism=organism)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"query_count": len(queries), "organism": organism}
        return _download_success_response(out_path, content_preview=_read_preview(out_path), biological_metadata=meta, download_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("DownloadError", str(e))


def query_ncbi_batch_lookup_by_ids(gene_ids: List[str]) -> str:
    """Lookup multiple genes by IDs and return structured list wrapped in rich JSON."""
    t0 = time.perf_counter()
    try:
        d = batch_lookup_by_ids(gene_ids)
        content = json.dumps(d, ensure_ascii=False)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"id_count": len(gene_ids)}
        return _query_success_response(content, content_preview=content, biological_metadata=meta, query_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("QueryError", str(e))

def download_ncbi_batch_lookup_by_ids(gene_ids: List[str], out_path: str) -> str:
    """Lookup multiple genes by IDs, save to JSON. Returns rich JSON with file_info."""
    t0 = time.perf_counter()
    try:
        d = batch_lookup_by_ids(gene_ids)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"id_count": len(gene_ids)}
        return _download_success_response(out_path, content_preview=_read_preview(out_path), biological_metadata=meta, download_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("DownloadError", str(e))

def query_ncbi_batch_lookup_by_symbols(gene_symbols: List[str], organism: str) -> str:
    """Lookup multiple genes by symbols. Returns structured list wrapped in rich JSON."""
    t0 = time.perf_counter()
    try:
        d = batch_lookup_by_symbols(gene_symbols, organism)
        content = json.dumps(d, ensure_ascii=False)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"symbol_count": len(gene_symbols), "organism": organism}
        return _query_success_response(content, content_preview=content, biological_metadata=meta, query_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("QueryError", str(e))


def download_ncbi_batch_lookup_by_symbols(gene_symbols: List[str], organism: str, out_path: str) -> str:
    """Lookup multiple genes by symbols, save to JSON. Returns rich JSON with file_info."""
    t0 = time.perf_counter()
    try:
        d = batch_lookup_by_symbols(gene_symbols, organism)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"symbol_count": len(gene_symbols), "organism": organism}
        return _download_success_response(out_path, content_preview=_read_preview(out_path), biological_metadata=meta, download_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("DownloadError", str(e))



__all__ = [
    "query_ncbi_sequence",
    "download_ncbi_sequence",
    "query_ncbi_metadata",
    "download_ncbi_metadata",
    "query_ncbi_blast",
    "download_ncbi_blast",
    
    "query_ncbi_clinvar_esearch",
    "download_ncbi_clinvar_esearch",
    "query_ncbi_clinvar_esummary",
    "download_ncbi_clinvar_esummary",
    "query_ncbi_clinvar_efetch",
    "download_ncbi_clinvar_efetch",
    "query_ncbi_clinvar_variants",
    "download_ncbi_clinvar_variants",

    "query_ncbi_gene_by_id",
    "download_ncbi_gene_by_id",
    "query_ncbi_gene_by_symbol",
    "download_ncbi_gene_by_symbol",
    "query_ncbi_gene_esearch",
    "download_ncbi_gene_esearch",
    "query_ncbi_gene_esummary",
    "download_ncbi_gene_esummary",
    "query_ncbi_gene_efetch",
    "download_ncbi_gene_efetch",
    
    "query_ncbi_batch_esearch",
    "download_ncbi_batch_esearch",
    "query_ncbi_batch_lookup_by_ids",
    "download_ncbi_batch_lookup_by_ids",
    "query_ncbi_batch_lookup_by_symbols",
    "download_ncbi_batch_lookup_by_symbols",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="NCBI operations: query_* (return JSON with content) and download_* (return JSON with file_path)."
    )
    parser.add_argument("--test", action="store_true", help="Run operations tests")
    parser.add_argument("--out_dir", type=str, default="example/database/ncbi", help="Output directory")
    args = parser.parse_args()

    if not args.test:
        print("Use --test to run operations tests.")
        exit(0)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    def _print_res(name: str, res: str) -> None:
        obj = json.loads(res)
        print(f"  {name}: status={obj.get('status')} ...")
        if obj.get("status") == "success":
            if "content_preview" in obj:
                print(f"  preview: {obj['content_preview'][:100].replace(chr(10), ' ')}...")
            if obj.get("execution_context"):
                print(f"  execution_context: {obj['execution_context']}")
            if obj.get("file_info"):
                print(f"  file_info: {obj['file_info']}")
        else:
            print(f"  error: {obj.get('error')}")

    print("=== sequence ===")
    seq_id = "NP_000483.1"
    _print_res("query_ncbi_sequence", query_ncbi_sequence(seq_id))
    _print_res("download_ncbi_sequence", download_ncbi_sequence(seq_id, os.path.join(out_dir, f"{seq_id}.fasta")))

    print("=== metadata ===")
    _print_res("query_ncbi_metadata", query_ncbi_metadata(seq_id))
    _print_res("download_ncbi_metadata", download_ncbi_metadata(seq_id, os.path.join(out_dir, f"{seq_id}.gb")))

    print("=== blast ===")
    short_seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQD"
    _print_res("query_ncbi_blast", query_ncbi_blast(short_seq, hitlist_size=3, alignments=3))
    _print_res("download_ncbi_blast", download_ncbi_blast(short_seq, os.path.join(out_dir, "ncbi_blast.xml"), hitlist_size=3, alignments=3))

    print("=== clinvar ===")
    clinvar_term = "BRCA1[gene] AND pathogenic[CLNSIG]"
    _print_res("query_ncbi_clinvar_esearch", query_ncbi_clinvar_esearch(clinvar_term, retmax=5))
    _print_res("download_ncbi_clinvar_esearch", download_ncbi_clinvar_esearch(clinvar_term, os.path.join(out_dir, "ncbi_clinvar_esearch.json"), retmax=5))
    
    esearch_res = json.loads(query_ncbi_clinvar_esearch(clinvar_term, retmax=5))
    if esearch_res.get("status") == "success":
        esearch_content = json.loads(esearch_res["content"])
        if "esearchresult" in esearch_content and "idlist" in esearch_content["esearchresult"]:
            id_list = esearch_content["esearchresult"]["idlist"]
            _print_res("query_ncbi_clinvar_esummary", query_ncbi_clinvar_esummary(id_list))
            _print_res("download_ncbi_clinvar_esummary", download_ncbi_clinvar_esummary(id_list, os.path.join(out_dir, "ncbi_clinvar_esummary.json")))
            
            _print_res("query_ncbi_clinvar_efetch", query_ncbi_clinvar_efetch(id_list))
            _print_res("download_ncbi_clinvar_efetch", download_ncbi_clinvar_efetch(id_list, os.path.join(out_dir, "ncbi_clinvar_efetch.xml")))

    print("=== clinvar queries ===")
    _print_res("query_ncbi_clinvar_variants", query_ncbi_clinvar_variants(clinvar_term, retmax=5))
    _print_res("download_ncbi_clinvar_variants", download_ncbi_clinvar_variants(clinvar_term, os.path.join(out_dir, "ncbi_clinvar_variants.json"), retmax=5))

    print("=== gene ===")
    gene_id = "672"
    _print_res("query_ncbi_gene_by_id", query_ncbi_gene_by_id(gene_id))
    _print_res("download_ncbi_gene_by_id", download_ncbi_gene_by_id(gene_id, os.path.join(out_dir, "ncbi_gene_by_id.json")))
    _print_res("query_ncbi_gene_by_symbol", query_ncbi_gene_by_symbol("BRCA1", "human"))
    _print_res("download_ncbi_gene_by_symbol", download_ncbi_gene_by_symbol("BRCA1", "human", os.path.join(out_dir, "ncbi_gene_by_symbol.json")))

    print("=== gene eutils ===")
    query_gene = "BRCA1[gene] AND human[organism]"
    _print_res("query_ncbi_gene_esearch", query_ncbi_gene_esearch(query_gene, retmax=5))
    _print_res("download_ncbi_gene_esearch", download_ncbi_gene_esearch(query_gene, os.path.join(out_dir, "ncbi_gene_esearch.json"), retmax=5))
    g_search_res = json.loads(query_ncbi_gene_esearch(query_gene, retmax=5))
    if g_search_res.get("status") == "success":
        g_ids = json.loads(g_search_res["content"])
        _print_res("query_ncbi_gene_esummary", query_ncbi_gene_esummary(g_ids))
        _print_res("download_ncbi_gene_esummary", download_ncbi_gene_esummary(g_ids, os.path.join(out_dir, "ncbi_gene_esummary.json")))
        _print_res("query_ncbi_gene_efetch", query_ncbi_gene_efetch(g_ids))
        _print_res("download_ncbi_gene_efetch", download_ncbi_gene_efetch(g_ids, os.path.join(out_dir, "ncbi_gene_efetch.xml")))

    print("=== batch gene lookups ===")
    q_batch = ["BRCA1", "TP53"]
    _print_res("query_ncbi_batch_esearch", query_ncbi_batch_esearch(q_batch, "human"))
    _print_res("download_ncbi_batch_esearch", download_ncbi_batch_esearch(q_batch, os.path.join(out_dir, "ncbi_batch_esearch.json"), "human"))
    
    b_ids = ["672", "7157"]
    _print_res("query_ncbi_batch_lookup_by_ids", query_ncbi_batch_lookup_by_ids(b_ids))
    _print_res("download_ncbi_batch_lookup_by_ids", download_ncbi_batch_lookup_by_ids(b_ids, os.path.join(out_dir, "ncbi_batch_lookup_by_ids.json")))
    
    _print_res("query_ncbi_batch_lookup_by_symbols", query_ncbi_batch_lookup_by_symbols(q_batch, "human"))
    _print_res("download_ncbi_batch_lookup_by_symbols", download_ncbi_batch_lookup_by_symbols(q_batch, "human", os.path.join(out_dir, "ncbi_batch_lookup_by_symbols.json")))

    print(f"Done. Output under {out_dir}")
