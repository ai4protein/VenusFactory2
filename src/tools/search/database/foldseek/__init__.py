# FoldSeek: query (submit job), download alignments (.m8), parse and prepare FASTA

import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Union, Tuple

from .foldseek_submit import submit_foldseek_job, query_foldseek_status, wait_foldseek_complete, FOLDSEEK_API_URL
from .download_foldseek_m8 import (
    download_foldseek_m8,
    FoldSeekAlignment,
    FoldSeekAlignmentParser,
    prepare_foldseek_sequences,
)


def get_foldseek_sequences(
    pdb_file_path: str,
    protect_start: int,
    protect_end: int,
    output_dir: Optional[Union[Path, str]] = None,
) -> Tuple[Path, int]:
    """
    Submit PDB to FoldSeek, download alignments, extract sequences covering the protected region.
    Returns (path_to_output_fasta, total_sequences).
    """
    output_dir = Path(output_dir) if output_dir else Path("download/FoldSeek")
    job_id = submit_foldseek_job(pdb_file_path)
    wait_foldseek_complete(job_id)
    download_files = download_foldseek_m8(job_id, output_dir)
    stem = os.path.basename(pdb_file_path).replace(".pdb", "")
    foldseek_fasta = output_dir / f"{stem}_{datetime.now().strftime('%Y%m%d%H%M%S')}.fasta"
    total_sequences = prepare_foldseek_sequences(download_files, foldseek_fasta, protect_start, protect_end)
    return foldseek_fasta, total_sequences
