"""EBI NCBI BLAST homologue search.

Submit a protein sequence → poll → fetch JSON results → flatten hits.

Adapted from google-deepmind/science-skills (Apache-2.0):
- skills/protein_sequence_similarity_search/scripts/uniprot_blast.py
"""
import os
import time
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BLAST_BASE_URL = "https://www.ebi.ac.uk/Tools/services/rest/ncbiblast"
MAX_ALIGNMENT_HITS = 300
POLLING_TIMEOUT = 15 * 60

ALLOWED_DATABASES = {
    "uniprotkb", "uniprotkb_swissprot", "uniprotkb_swissprotsv",
    "uniprotkb_reference_proteomes", "uniprotkb_trembl",
    "uniprotkb_refprotswissprot", "uniprotkb_archaea", "uniprotkb_arthropoda",
    "uniprotkb_bacteria", "uniprotkb_complete_microbial_proteomes",
    "uniprotkb_eukaryota", "uniprotkb_fungi", "uniprotkb_human",
    "uniprotkb_mammals", "uniprotkb_nematoda", "uniprotkb_rodents",
    "uniprotkb_vertebrates", "uniprotkb_viridiplantae", "uniprotkb_viruses",
    "uniprotkb_enzyme", "uniprotkb_covid19",
    "uniref100", "uniref90", "uniref50", "pdb",
}

_TERMINAL_OK = {"FINISHED"}
_TERMINAL_ERR = {"ERROR", "FAILURE", "NOT_FOUND"}


def _session() -> requests.Session:
    s = requests.Session()
    r = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=frozenset({"GET", "POST"}))
    s.mount("https://", HTTPAdapter(max_retries=r))
    s.mount("http://", HTTPAdapter(max_retries=r))
    return s


def _normalize_sequence(query: str) -> str:
    q = (query or "").strip()
    if not q:
        raise ValueError("empty sequence")
    if os.path.isfile(q):
        with open(q, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if text.startswith(">"):
            text = "".join(text.split("\n")[1:])
        return text.strip()
    if q.startswith(">"):
        return "".join(q.split("\n")[1:]).strip()
    return q


def search_ebi_blast(
    sequence_or_fasta_path: str,
    database: str = "uniprotkb_swissprot",
    email: Optional[str] = None,
    poll_interval: float = 30.0,
    timeout_secs: int = POLLING_TIMEOUT,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run a BLAST search. `database` may be a single name or comma-separated list.

    Returns (hits, metadata). Hits are flattened dicts with target_id, q_cov,
    e_value, identity_pct, aln_len, description.
    """
    sequence = _normalize_sequence(sequence_or_fasta_path)
    q_len = len(sequence)

    dbs = [d.strip().lower() for d in database.split(",") if d.strip()]
    bad = [d for d in dbs if d not in ALLOWED_DATABASES]
    if bad:
        raise ValueError(f"unsupported BLAST database(s): {bad}; allowed: {sorted(ALLOWED_DATABASES)}")

    sess = _session()
    use_email = (email or "").strip() or os.environ.get("USER_EMAIL", "").strip()
    if not use_email:
        raise ValueError(
            "EBI BLAST requires a valid email address. "
            "Pass the `email` parameter, or set the USER_EMAIL environment variable."
        )
    params: Dict[str, Any] = {
        "program": "blastp",
        "stype": "protein",
        "sequence": sequence,
        "database": ",".join(dbs),
        "email": use_email,
    }
    sresp = sess.post(
        f"{BLAST_BASE_URL}/run",
        data=urllib.parse.urlencode(params),
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "text/plain"},
        timeout=60,
    )
    if sresp.status_code != 200:
        raise RuntimeError(f"EBI BLAST submit failed [{sresp.status_code}]: {sresp.text[:500]}")
    job_id = sresp.text.strip()
    if not job_id:
        raise RuntimeError("EBI BLAST returned empty job id")

    start = time.monotonic()
    while time.monotonic() - start < timeout_secs:
        st = sess.get(f"{BLAST_BASE_URL}/status/{job_id}", headers={"Accept": "text/plain"}, timeout=30)
        if st.status_code != 200:
            raise RuntimeError(f"EBI BLAST status check failed [{st.status_code}]: {st.text[:200]}")
        status = st.text.strip()
        if status in _TERMINAL_OK:
            break
        if status in _TERMINAL_ERR:
            raise RuntimeError(f"EBI BLAST job {job_id} failed with status: {status}")
        time.sleep(poll_interval)
    else:
        raise TimeoutError(f"EBI BLAST job {job_id} did not finish within {timeout_secs}s")

    rresp = sess.get(f"{BLAST_BASE_URL}/result/{job_id}/json", timeout=60)
    if rresp.status_code != 200:
        raise RuntimeError(f"EBI BLAST result fetch failed [{rresp.status_code}]: {rresp.text[:200]}")
    res = rresp.json()

    flat: List[Dict[str, Any]] = []
    for hit in res.get("hits", [])[:MAX_ALIGNMENT_HITS]:
        target_acc = hit.get("hit_acc", "N/A")
        target_desc = hit.get("hit_desc", hit.get("hit_def", ""))
        hsps = hit.get("hit_hsps") or [{}]
        best = hsps[0] if hsps else {}
        try:
            qf = int(best.get("hsp_query_from", 0))
            qt = int(best.get("hsp_query_to", 0))
        except (TypeError, ValueError):
            qf = qt = 0
        cov = min((qt - qf + 1) / q_len * 100.0, 100.0) if (q_len > 0 and qt > qf) else 0.0
        flat.append({
            "target_id": str(target_acc),
            "description": str(target_desc),
            "q_cov": cov,
            "e_value": best.get("hsp_expect", None),
            "identity_pct": best.get("hsp_identity", None),
            "aln_len": best.get("hsp_align_len", None),
        })

    meta = {
        "engine": "EBI BLAST",
        "job_id": job_id,
        "query_length": q_len,
        "databases": dbs,
        "hit_count": len(flat),
        "email": use_email,
    }
    return flat, meta


__all__ = [
    "BLAST_BASE_URL",
    "MAX_ALIGNMENT_HITS",
    "ALLOWED_DATABASES",
    "search_ebi_blast",
]
