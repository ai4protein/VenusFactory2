"""FoldSeek: submit job and query job status (no download)."""
import time
import requests
from typing import List, Optional

FOLDSEEK_API_URL = "https://search.foldseek.com/api"
DEFAULT_DATABASES = ["afdb-proteome", "afdb-swissprot", "afdb50", "cath50", "gmgcl_id", "mgnify_esm30", "pdb100"]


def submit_foldseek_job(
    pdb_file_path: str,
    databases: Optional[List[str]] = None,
    mode: str = "3diaa",
) -> str:
    """
    Submit PDB content to FoldSeek and return job_id.
    Raises on submit failure.
    """
    databases = databases or DEFAULT_DATABASES
    try:
        with open(pdb_file_path) as f:
            pdb_content = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"PDB file not found: {pdb_file_path}")

    data = {"q": pdb_content, "database[]": databases, "mode": mode}
    response = requests.post(f"{FOLDSEEK_API_URL}/ticket", data=data)
    if response.status_code != 200:
        raise RuntimeError(f"FoldSeek submit failed: {response.text}")
    return response.json()["id"]


def query_foldseek_status(job_id: str) -> str:
    """
    Query FoldSeek job status. Returns status string: PENDING, RUNNING, COMPLETE, or ERROR.
    """
    response = requests.get(f"{FOLDSEEK_API_URL}/ticket/{job_id}")
    if response.status_code != 200:
        raise RuntimeError(f"FoldSeek status check failed: {response.text}")
    return response.json()["status"]


def wait_foldseek_complete(job_id: str, poll_interval: float = 2.0) -> None:
    """Poll until job status is COMPLETE or ERROR. Raises on ERROR."""
    while True:
        status = query_foldseek_status(job_id)
        if status == "COMPLETE":
            return
        if status == "ERROR":
            raise RuntimeError(f"FoldSeek job {job_id} failed (ERROR).")
        time.sleep(poll_interval)
