"""Clustal Omega (EBI): submit MSA job, poll status, fetch alignment.

API:
- POST https://www.ebi.ac.uk/Tools/services/rest/clustalo/run
  body: email=&title=&sequence=<FASTA text>
  → text/plain job id
- GET  …/status/<job_id> → RUNNING | PENDING | FINISHED | ERROR | FAILURE | NOT_FOUND
- GET  …/result/<job_id>/fa → FASTA alignment text
"""
import time
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


CLUSTALO_BASE_URL = "https://www.ebi.ac.uk/Tools/services/rest/clustalo"

_TERMINAL_OK = {"FINISHED"}
_TERMINAL_ERR = {"ERROR", "FAILURE", "NOT_FOUND"}


def _session() -> requests.Session:
    s = requests.Session()
    r = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=frozenset({"GET", "POST"}))
    s.mount("https://", HTTPAdapter(max_retries=r))
    s.mount("http://", HTTPAdapter(max_retries=r))
    return s


def submit_clustalo_job(
    sequences_fasta: str,
    email: str,
    title: str = "VenusFactory MSA",
    timeout: int = 60,
) -> str:
    """Submit a FASTA alignment job. Returns job_id. Raises on submit failure."""
    if not sequences_fasta or not sequences_fasta.strip():
        raise ValueError("sequences_fasta is empty")
    if not email or "@" not in email:
        raise ValueError(f"invalid email: {email!r}")
    data = {"email": email, "title": title, "sequence": sequences_fasta}
    resp = _session().post(
        f"{CLUSTALO_BASE_URL}/run",
        data=data,
        headers={"Accept": "text/plain"},
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Clustal Omega submit failed [{resp.status_code}]: {resp.text[:500]}")
    return resp.text.strip()


def query_clustalo_status(job_id: str, timeout: int = 30) -> str:
    """Returns the raw status string from EBI."""
    resp = _session().get(
        f"{CLUSTALO_BASE_URL}/status/{job_id}",
        headers={"Accept": "text/plain"},
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Clustal Omega status check failed [{resp.status_code}]: {resp.text[:500]}")
    return resp.text.strip()


def wait_clustalo_complete(
    job_id: str,
    poll_interval: float = 10.0,
    timeout_secs: int = 15 * 60,
) -> None:
    """Block until status is FINISHED. Raises on error or timeout."""
    start = time.monotonic()
    while time.monotonic() - start < timeout_secs:
        status = query_clustalo_status(job_id)
        if status in _TERMINAL_OK:
            return
        if status in _TERMINAL_ERR:
            raise RuntimeError(f"Clustal Omega job {job_id} failed with status: {status}")
        time.sleep(poll_interval)
    raise TimeoutError(f"Clustal Omega job {job_id} did not finish within {timeout_secs}s")


def fetch_clustalo_alignment_fasta(job_id: str, timeout: int = 60) -> str:
    """Download FASTA-formatted alignment text for a finished job."""
    resp = _session().get(
        f"{CLUSTALO_BASE_URL}/result/{job_id}/fa",
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Clustal Omega result fetch failed [{resp.status_code}]: {resp.text[:500]}")
    return resp.text


__all__ = [
    "CLUSTALO_BASE_URL",
    "submit_clustalo_job",
    "query_clustalo_status",
    "wait_clustalo_complete",
    "fetch_clustalo_alignment_fasta",
]
