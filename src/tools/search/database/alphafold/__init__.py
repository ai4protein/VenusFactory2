# AlphaFold DB: single public API via alphafold_operations (query_* return text, download_* save to file).

from .alphafold_operations import (
    download_alphafold_structure_by_uniprot_id,
    download_alphafold_metadata_by_uniprot_id,
)

__all__ = [
    "download_alphafold_structure_by_uniprot_id",
    "download_alphafold_metadata_by_uniprot_id",
]
