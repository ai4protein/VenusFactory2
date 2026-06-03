"""NCBI E-utilities extras: CDS-to-protein translation and gene+organism→protein search.

Two operations that complement the existing `download_ncbi_sequence` /
`download_ncbi_gene_by_*` tools:

1. translate_ncbi_cds_to_protein: given a nuccore accession, retrieve the
   pre-translated CDS protein via efetch(rettype=fasta_cds_aa), pick the
   sequence closest to a target length, and write FASTA + metadata.

2. search_ncbi_protein_by_gene_and_organism: query NCBI Protein DB with
   `<gene>[Gene Name] AND <organism>[Organism]`, optionally narrow by length,
   write a multi-FASTA + JSON of metadata.

Both honor NCBI_API_KEY when present (env), raising QPS limit from 3 to 10.

Adapted from google-deepmind/science-skills (Apache-2.0):
- skills/ncbi_sequence_fetch/scripts/ncbi_fetch.py
"""
import json
import os
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.tools.path_sanitizer import to_client_file_path


EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

_PREVIEW_LEN = 1500
_SOURCE = "NCBI E-utilities"


def _session() -> requests.Session:
    s = requests.Session()
    r = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=frozenset({"GET"}))
    s.mount("https://", HTTPAdapter(max_retries=r))
    s.mount("http://", HTTPAdapter(max_retries=r))
    return s


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
    file_path: str, content_preview: str, biological_metadata: Dict[str, Any], elapsed_ms: int,
) -> str:
    path = Path(file_path)
    size = path.stat().st_size if path.exists() else 0
    out: Dict[str, Any] = {
        "status": "success",
        "file_info": {
            "file_path": to_client_file_path(path if path.exists() else file_path),
            "file_name": path.name,
            "file_size": size,
            "format": "fasta",
        },
        "content_preview": content_preview[:_PREVIEW_LEN],
        "biological_metadata": biological_metadata,
        "execution_context": {"elapsed_ms": elapsed_ms, "source": _SOURCE},
    }
    return json.dumps(out, ensure_ascii=False)


def _with_api_key(params: Dict[str, Any]) -> Dict[str, Any]:
    key = os.environ.get("NCBI_API_KEY")
    if key:
        params = {**params, "api_key": key}
    return params


def _parse_fasta(text: str) -> List[Tuple[str, str]]:
    """Returns list of (header, sequence) tuples."""
    entries: List[Tuple[str, str]] = []
    header: Optional[str] = None
    chunks: List[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if header is not None:
                entries.append((header, "".join(chunks)))
            header = line.strip()
            chunks = []
        elif line.strip():
            chunks.append(line.strip())
    if header is not None:
        entries.append((header, "".join(chunks)))
    return entries


def _pick_by_length(entries: List[Tuple[str, str]], target_length: int) -> Optional[Tuple[str, str]]:
    if not entries:
        return None
    if target_length and target_length > 0:
        return min(entries, key=lambda e: abs(len(e[1]) - target_length))
    return max(entries, key=lambda e: len(e[1]))


def _efetch(
    sess: requests.Session,
    db: str,
    accession: str,
    rettype: str = "fasta",
    retmode: str = "text",
    timeout: int = 60,
) -> Optional[str]:
    params = _with_api_key({"db": db, "id": accession, "rettype": rettype, "retmode": retmode})
    resp = sess.get(f"{EUTILS_BASE}/efetch.fcgi", params=params, timeout=timeout)
    time.sleep(0.4)  # be polite to NCBI
    if resp.status_code != 200 or not resp.text:
        return None
    return resp.text


def _esearch(
    sess: requests.Session,
    db: str,
    term: str,
    retmax: int = 20,
    timeout: int = 60,
) -> Tuple[List[str], int]:
    params = _with_api_key({"db": db, "term": term, "retmax": retmax, "retmode": "json"})
    resp = sess.get(f"{EUTILS_BASE}/esearch.fcgi", params=params, timeout=timeout)
    time.sleep(0.4)
    if resp.status_code != 200:
        raise RuntimeError(f"NCBI esearch failed [{resp.status_code}]: {resp.text[:300]}")
    body = resp.json()
    result = body.get("esearchresult") or {}
    ids = result.get("idlist") or []
    count = int(result.get("count", 0) or 0)
    return list(ids), count


def translate_ncbi_cds_to_protein(
    accession: str,
    out_dir: str,
    target_length: int = 0,
    timeout: int = 60,
) -> str:
    """Fetch CDS-translated protein FASTA for an nuccore accession.

    Uses `efetch(db=nuccore, rettype=fasta_cds_aa)`. If multiple CDS
    translations exist, picks the one closest to `target_length` (0 means
    longest). Writes `<out_dir>/<accession>_protein.fasta` and returns the
    standard rich JSON envelope.
    """
    t0 = time.perf_counter()
    accession = (accession or "").strip()
    if not accession:
        return _error_response("ValidationError", "empty accession")
    out_dir = str(out_dir or "").strip().rstrip(os.sep)
    if not out_dir:
        return _error_response("ValidationError", "empty out_dir")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    sess = _session()
    try:
        text = _efetch(sess, "nuccore", accession, rettype="fasta_cds_aa", retmode="text", timeout=timeout)
    except requests.RequestException as e:
        return _error_response("NetworkError", str(e))

    if not text or text.lstrip().startswith("<"):  # NCBI returns XML on error
        return _error_response(
            "NotFound",
            f"no fasta_cds_aa for accession {accession}",
            suggestion="Verify the accession is a nuccore record (mRNA / genomic) and contains CDS features.",
        )

    entries = _parse_fasta(text)
    best = _pick_by_length(entries, target_length)
    if not best:
        return _error_response("NotFound", f"no parseable CDS translations for {accession}")

    header, seq = best
    seq = seq.replace("*", "")
    out_path = os.path.join(out_dir, f"{accession}_protein.fasta")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"{header}\n{seq}\n")
    except OSError as e:
        return _error_response("IOError", f"failed to write FASTA: {e}")

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    meta = {
        "accession": accession,
        "method": "fasta_cds_aa",
        "translations_found": len(entries),
        "selected_length": len(seq),
        "target_length": target_length or None,
        "header": header,
    }
    preview = f"{header}\n{seq[:200]}{'...' if len(seq) > 200 else ''}\n(length={len(seq)})"
    return _success_response(out_path, preview, meta, elapsed_ms)


def search_ncbi_protein_by_gene_and_organism(
    gene: str,
    organism: str,
    out_dir: str,
    target_length: int = 0,
    retmax: int = 10,
    timeout: int = 60,
) -> str:
    """Search NCBI Protein DB by gene name + organism and fetch all hits as FASTA.

    Writes `<out_dir>/<gene>_<organism>_proteins.fasta` (concatenated multi-FASTA)
    and a sibling `.json` summary with per-hit metadata.
    """
    t0 = time.perf_counter()
    gene = (gene or "").strip()
    organism = (organism or "").strip()
    if not gene:
        return _error_response("ValidationError", "empty gene")
    if not organism:
        return _error_response("ValidationError", "empty organism")
    out_dir = str(out_dir or "").strip().rstrip(os.sep)
    if not out_dir:
        return _error_response("ValidationError", "empty out_dir")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    term = f"{gene}[Gene Name] AND {organism}[Organism]"
    if target_length and target_length > 0:
        lo = max(target_length - 25, 1)
        hi = target_length + 25
        term += f" AND {lo}:{hi}[Sequence Length]"

    sess = _session()
    try:
        ids, count = _esearch(sess, "protein", term, retmax=retmax, timeout=timeout)
    except (requests.RequestException, RuntimeError) as e:
        return _error_response("SearchError", str(e), suggestion="Check the gene symbol and organism (e.g. 'Homo sapiens').")

    if not ids:
        return _error_response("NotFound", f"no protein hits for query: {term}")

    hits: List[Dict[str, Any]] = []
    fasta_chunks: List[str] = []
    for uid in ids[:retmax]:
        try:
            fasta = _efetch(sess, "protein", uid, rettype="fasta", retmode="text", timeout=timeout)
        except requests.RequestException:
            continue
        if not fasta:
            continue
        for header, seq in _parse_fasta(fasta):
            hits.append({"uid": uid, "header": header, "length": len(seq)})
            fasta_chunks.append(f"{header}\n{seq}\n")

    safe_gene = gene.replace(" ", "_")
    safe_org = organism.replace(" ", "_").replace("/", "_")
    stem = f"{safe_gene}_{safe_org}_proteins"
    fasta_path = os.path.join(out_dir, f"{stem}.fasta")
    summary_path = os.path.join(out_dir, f"{stem}.json")

    try:
        with open(fasta_path, "w", encoding="utf-8") as f:
            f.writelines(fasta_chunks)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "gene": gene, "organism": organism, "query": term,
                "total_count": count, "returned_count": len(hits), "hits": hits,
            }, f, ensure_ascii=False, indent=2)
    except OSError as e:
        return _error_response("IOError", f"failed to write outputs: {e}")

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    meta = {
        "gene": gene,
        "organism": organism,
        "query": term,
        "total_count": count,
        "returned_count": len(hits),
        "summary_path": to_client_file_path(summary_path),
    }
    preview_lines = [f"NCBI Protein search: {term}", f"total_count={count}, returned={len(hits)}"]
    for h in hits[:5]:
        preview_lines.append(f"  {h['uid']}: {h['header'][:80]} (len={h['length']})")
    return _success_response(fasta_path, "\n".join(preview_lines), meta, elapsed_ms)


__all__ = [
    "translate_ncbi_cds_to_protein",
    "search_ncbi_protein_by_gene_and_organism",
]
