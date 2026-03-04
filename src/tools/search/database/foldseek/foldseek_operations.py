"""
FoldSeek operations: single exit for submit, query, and download.

All public functions are named with database source (foldseek):
- submit_foldseek_*: submit job (return job_id).
- query_foldseek_*: get status or wait for completion (return str or None).
- download_foldseek_*: write to file and return paths or count.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union

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


# ---------- submit_foldseek_* ----------


def submit_foldseek_job(
    pdb_file_path: str,
    databases: Optional[List[str]] = None,
    mode: str = "3diaa",
) -> str:
    """Submit PDB to FoldSeek and return job_id. Raises on failure."""
    return _submit_job(pdb_file_path, databases=databases, mode=mode)


# ---------- query_foldseek_*: status, wait ----------


def query_foldseek_job_status(job_id: str) -> str:
    """Query FoldSeek job status. Returns PENDING, RUNNING, COMPLETE, or ERROR."""
    return _query_status(job_id)


def query_foldseek_wait_complete(job_id: str, poll_interval: float = 2.0) -> None:
    """Poll until job is COMPLETE or ERROR. Raises on ERROR."""
    _wait_complete(job_id, poll_interval=poll_interval)


# ---------- download_foldseek_*: save to file, return paths or count ----------


def download_foldseek_alignments_m8(
    job_id: str,
    out_dir: Union[Path, str],
    databases: Optional[List[str]] = None,
) -> List[str]:
    """Download alignment result files (.m8) for a FoldSeek job_id. Returns list of .m8 paths."""
    return _download_m8(job_id, out_dir, databases=databases)


def download_foldseek_sequences_to_fasta(
    alignments_files: List[str],
    output_fasta: Union[Path, str],
    protect_start: int,
    protect_end: int,
) -> int:
    """Extract sequences from FoldSeek alignments covering the protected region into a FASTA file. Returns count."""
    return _prepare_sequences(
        alignments_files,
        Path(output_fasta),
        protect_start,
        protect_end,
    )


# ---------- high-level pipeline ----------


def download_foldseek_results_by_pdb_file(
    pdb_file_path: str,
    protect_start: int,
    protect_end: int,
    out_dir: Optional[Union[Path, str]] = None,
) -> Tuple[Path, int]:
    """
    Submit PDB to FoldSeek, wait for completion, download alignments, extract sequences to FASTA.
    Returns (path_to_output_fasta, total_sequences).
    """
    out_dir = Path(out_dir) if out_dir else Path("download/FoldSeek")
    job_id = submit_foldseek_job(pdb_file_path)
    query_foldseek_wait_complete(job_id)
    download_files = download_foldseek_alignments_m8(job_id, out_dir)
    stem = os.path.basename(pdb_file_path).replace(".pdb", "")
    foldseek_fasta = out_dir / f"{stem}_{datetime.now().strftime('%Y%m%d%H%M%S')}.fasta"
    total_sequences = download_foldseek_sequences_to_fasta(
        download_files, foldseek_fasta, protect_start, protect_end
    )
    return foldseek_fasta, total_sequences


__all__ = [
    "download_foldseek_results_by_pdb_file",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="FoldSeek operations: submit_foldseek_*, query_foldseek_*, download_foldseek_*."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run download_foldseek_results_by_pdb_file (full pipeline); output under example/database/foldseek",
    )
    parser.add_argument("--pdb_file", type=str, default="example/database/alphafold/structure/A0A1B0GTW7.pdb")
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

    out_base = os.path.join("example", "database", "foldseek")
    os.makedirs(out_base, exist_ok=True)
    search_dir = os.path.join(out_base, "search")
    os.makedirs(search_dir, exist_ok=True)
    out_dir = args.out_dir or search_dir
    protect_start = args.protect_start
    protect_end = args.protect_end

    print("=== download_foldseek_results_by_pdb_file (submit_* + query_* + download_* pipeline) ===")
    fasta_file, total = download_foldseek_results_by_pdb_file(
        args.pdb_file, protect_start, protect_end, out_dir=out_dir
    )
    print(f"  fasta: {fasta_file}, total_sequences: {total}")
    print(f"Done. Output under {out_base}")
