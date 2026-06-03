"""MMseqs2 homologue search via ColabFold web API.

Submit a protein sequence → poll → download tar.gz → extract A3M → parse hits.

Adapted from google-deepmind/science-skills (Apache-2.0):
- skills/protein_sequence_similarity_search/scripts/mmseqs2_search.py
"""
import os
import shutil
import tarfile
import tempfile
import time
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


COLABFOLD_HOST = "https://api.colabfold.com"
MAX_ALIGNMENT_HITS = 300
POLLING_TIMEOUT = 15 * 60  # seconds

A3M_FIELDS = [
    "target",
    "bit_score",
    "identity",
    "e_value",
    "q_start",
    "q_end",
    "q_len",
    "t_start",
    "t_end",
    "t_len",
]


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


def _parse_a3m_headers(file_path: str, q_len: int) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith(">"):
                continue
            parts = line.strip().split()
            if len(parts) < 10:
                continue
            try:
                hit = dict(zip(A3M_FIELDS, parts, strict=True))
                for col in ("q_start", "q_end", "q_len", "t_start", "t_end", "t_len"):
                    hit[col] = int(hit[col])
                for col in ("bit_score", "identity", "e_value"):
                    hit[col] = float(hit[col])
                if q_len > 0 and hit["q_end"] > hit["q_start"]:
                    cov = min((hit["q_end"] - hit["q_start"] + 1) / q_len * 100.0, 100.0)
                else:
                    cov = 0.0
                aln_len = (hit["t_end"] - hit["t_start"] + 1) if hit["t_end"] > hit["t_start"] else 0
                hit["target_id"] = hit["target"][1:]
                hit["q_cov"] = cov
                hit["aln_len"] = aln_len
                hits.append(hit)
            except (ValueError, IndexError):
                continue
    hits.sort(key=lambda x: x["e_value"])
    return hits


def search_mmseqs2_colabfold(
    sequence_or_fasta_path: str,
    include_mgnify: bool = False,
    poll_interval: float = 10.0,
    timeout_secs: int = POLLING_TIMEOUT,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run a full submit-poll-download-parse pipeline.

    Returns (hits, metadata) where hits is a list of dicts (sorted by e_value,
    capped at MAX_ALIGNMENT_HITS) and metadata describes the run.
    """
    sequence = _normalize_sequence(sequence_or_fasta_path)
    q_len = len(sequence)

    sess = _session()
    payload = urllib.parse.urlencode({"q": f">Query_1\n{sequence}\n", "mode": "all"})

    resp = sess.post(
        f"{COLABFOLD_HOST}/ticket/msa",
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"ColabFold MSA submit failed [{resp.status_code}]: {resp.text[:500]}")
    ticket = resp.json()
    ticket_id = ticket.get("id")
    if not ticket_id:
        status = ticket.get("status", "UNKNOWN")
        raise RuntimeError(f"ColabFold submission rejected: status={status} payload={ticket}")

    start = time.monotonic()
    while time.monotonic() - start < timeout_secs:
        sresp = sess.get(f"{COLABFOLD_HOST}/ticket/{ticket_id}", timeout=30)
        if sresp.status_code != 200:
            raise RuntimeError(f"ColabFold status check failed [{sresp.status_code}]: {sresp.text[:200]}")
        state = sresp.json().get("status", "UNKNOWN")
        if state == "COMPLETE":
            break
        if state in ("ERROR", "MAINTENANCE"):
            raise RuntimeError(f"ColabFold MMseqs2 job {ticket_id} failed with status: {state}")
        time.sleep(poll_interval)
    else:
        raise TimeoutError(f"ColabFold MMseqs2 job {ticket_id} did not complete within {timeout_secs}s")

    tmp_dir = tempfile.mkdtemp(prefix="mmseqs2_colabfold_")
    try:
        dresp = sess.get(f"{COLABFOLD_HOST}/result/download/{ticket_id}", timeout=120)
        if dresp.status_code != 200:
            raise RuntimeError(f"ColabFold result download failed [{dresp.status_code}]: {dresp.text[:200]}")
        tar_path = os.path.join(tmp_dir, f"{ticket_id}.tar.gz")
        with open(tar_path, "wb") as f:
            f.write(dresp.content)
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    continue
                tar.extract(member, path=tmp_dir)
        os.remove(tar_path)

        all_hits: List[Dict[str, Any]] = []
        uniref_path = os.path.join(tmp_dir, "uniref.a3m")
        if os.path.exists(uniref_path):
            all_hits.extend(_parse_a3m_headers(uniref_path, q_len))
        if include_mgnify:
            mgn_path = os.path.join(tmp_dir, "bfd.mgnify30.metaeuk30.smag30.a3m")
            if os.path.exists(mgn_path):
                all_hits.extend(_parse_a3m_headers(mgn_path, q_len))
        all_hits.sort(key=lambda x: x["e_value"])
        all_hits = all_hits[:MAX_ALIGNMENT_HITS]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    meta = {
        "engine": "MMseqs2 (ColabFold)",
        "ticket_id": ticket_id,
        "query_length": q_len,
        "hit_count": len(all_hits),
        "include_mgnify": include_mgnify,
    }
    return all_hits, meta


__all__ = ["COLABFOLD_HOST", "MAX_ALIGNMENT_HITS", "search_mmseqs2_colabfold"]
