# database: per-database modules (rcsb, interpro, alphafold, uniprot, ncbi, foldseek) + utils
# Each DB has query_* (query only) and download_* (download to local).

from .foldseek import (
    get_foldseek_sequences,
    download_foldseek_m8,
    FoldSeekAlignment,
    FoldSeekAlignmentParser,
    prepare_foldseek_sequences,
)

__all__ = [
    "get_foldseek_sequences",
    "download_foldseek_m8",
    "FoldSeekAlignment",
    "FoldSeekAlignmentParser",
    "prepare_foldseek_sequences",
]
