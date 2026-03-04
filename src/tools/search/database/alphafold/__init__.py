# AlphaFold DB: single public API via alphafold_operations (query_* return text, download_* save to file).

from .alphafold_operations import (
    query_alphafold_structure,
    query_alphafold_metadata,
    download_alphafold_structure,
    download_alphafold_metadata,
)

__all__ = [
    "query_alphafold_structure",
    "query_alphafold_metadata",
    "download_alphafold_structure",
    "download_alphafold_metadata",
]
