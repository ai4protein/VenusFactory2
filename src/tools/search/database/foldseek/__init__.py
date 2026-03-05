# FoldSeek: single public API via foldseek_operations (query_foldseek_* / download_foldseek_* return JSON).

from .foldseek_operations import (
    download_foldseek_results_by_pdb_file,
)

__all__ = [
    "download_foldseek_results_by_pdb_file",
]
