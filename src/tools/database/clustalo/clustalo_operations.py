"""Clustal Omega operations: single entry point with rich JSON envelope.

Reads a FASTA file with 2-4000 sequences (≤ 4 MB), submits to EBI Clustal Omega,
polls until finished, downloads the FASTA-formatted MSA into out_dir.

Adapted from google-deepmind/science-skills (Apache-2.0):
- skills/protein_sequence_msa/scripts/msa_align.py
"""
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from src.tools.path_sanitizer import to_client_file_path

try:
    from .clustalo_submit import (
        submit_clustalo_job,
        wait_clustalo_complete,
        fetch_clustalo_alignment_fasta,
    )
except ImportError:
    from src.tools.database.clustalo.clustalo_submit import (
        submit_clustalo_job,
        wait_clustalo_complete,
        fetch_clustalo_alignment_fasta,
    )

_PREVIEW_LEN = 500
_SOURCE = "EBI Clustal Omega"
_MAX_FILE_BYTES = 4 * 1024 * 1024  # 4 MB
_MIN_SEQUENCES = 2
_MAX_SEQUENCES = 4000


def _error_response(error_type: str, message: str, suggestion: Optional[str] = None) -> str:
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
    content_preview: str,
    biological_metadata: Dict[str, Any],
    elapsed_ms: int,
) -> str:
    path = Path(file_path)
    file_size = path.stat().st_size if path.exists() else 0
    out: Dict[str, Any] = {
        "status": "success",
        "file_info": {
            "file_path": to_client_file_path(path if path.exists() else file_path),
            "file_name": path.name,
            "file_size": file_size,
            "format": "fasta",
        },
        "content_preview": content_preview[:_PREVIEW_LEN],
        "biological_metadata": biological_metadata,
        "execution_context": {"download_time_ms": elapsed_ms, "source": _SOURCE},
    }
    return json.dumps(out, ensure_ascii=False)


def download_clustalo_msa_by_fasta(
    fasta_path: str,
    out_dir: str,
    email: Optional[str] = None,
    poll_interval: float = 10.0,
    timeout_secs: int = 15 * 60,
) -> str:
    """Run EBI Clustal Omega MSA on a FASTA file and save the alignment.

    Returns the standard VenusFactory JSON envelope. The output file is
    `<out_dir>/<input_stem>_msa.fasta`.
    """
    t0 = time.perf_counter()

    fasta_path = str(fasta_path or "").strip()
    if not fasta_path:
        return _error_response("ValidationError", "empty fasta_path", suggestion="Provide the path to a FASTA file with ≥2 sequences.")
    if not os.path.exists(fasta_path):
        return _error_response("NotFound", f"input FASTA not found: {fasta_path}")

    file_size = os.path.getsize(fasta_path)
    if file_size > _MAX_FILE_BYTES:
        return _error_response(
            "ValidationError",
            f"FASTA file is {file_size/1024/1024:.2f} MB, exceeds EBI 4 MB limit.",
            suggestion="Reduce the number of sequences or split into multiple jobs.",
        )

    try:
        with open(fasta_path, "r", encoding="utf-8") as f:
            sequences = f.read().strip()
    except OSError as e:
        return _error_response("IOError", f"failed to read FASTA: {e}")

    if not sequences:
        return _error_response("ValidationError", "FASTA file is empty")

    n_seqs = sequences.count(">")
    if n_seqs < _MIN_SEQUENCES:
        return _error_response("ValidationError", f"need ≥{_MIN_SEQUENCES} sequences, found {n_seqs}")
    if n_seqs > _MAX_SEQUENCES:
        return _error_response("ValidationError", f"EBI limit is {_MAX_SEQUENCES} sequences, found {n_seqs}")

    out_dir = str(out_dir or "").strip().rstrip(os.sep)
    if not out_dir:
        return _error_response("ValidationError", "empty out_dir")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    use_email = (email or "").strip() or os.environ.get("USER_EMAIL", "").strip()
    if not use_email:
        return _error_response(
            "ValidationError",
            "EBI Clustal Omega requires a valid email address.",
            suggestion="Pass the `email` parameter, or set the USER_EMAIL environment variable to a real address EBI can contact if the job misbehaves.",
        )

    try:
        job_id = submit_clustalo_job(sequences_fasta=sequences, email=use_email)
    except (RuntimeError, ValueError) as e:
        return _error_response("SubmitError", str(e), suggestion="Check FASTA formatting and network access to ebi.ac.uk.")

    try:
        wait_clustalo_complete(job_id, poll_interval=poll_interval, timeout_secs=timeout_secs)
    except TimeoutError as e:
        return _error_response("Timeout", str(e), suggestion="Increase timeout_secs or rerun later.")
    except RuntimeError as e:
        return _error_response("JobError", str(e))

    try:
        alignment_text = fetch_clustalo_alignment_fasta(job_id)
    except RuntimeError as e:
        return _error_response("FetchError", str(e))

    input_stem = Path(fasta_path).stem
    out_path = os.path.join(out_dir, f"{input_stem}_msa.fasta")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(alignment_text)
    except OSError as e:
        return _error_response("IOError", f"failed to write alignment: {e}")

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    n_aligned = alignment_text.count(">")
    meta = {
        "input_sequences": n_seqs,
        "aligned_sequences": n_aligned,
        "job_id": job_id,
        "email": use_email,
    }
    return _download_success_response(out_path, alignment_text, meta, elapsed_ms)


__all__ = ["download_clustalo_msa_by_fasta"]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EBI Clustal Omega MSA")
    parser.add_argument("fasta_path")
    parser.add_argument("out_dir")
    parser.add_argument("--email", default=None)
    args = parser.parse_args()
    print(download_clustalo_msa_by_fasta(args.fasta_path, args.out_dir, email=args.email))
