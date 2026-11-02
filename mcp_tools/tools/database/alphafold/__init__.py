# AlphaFold DB: single public API via alphafold_operations (query_* return text, download_* save to file).
# alphafold_analyze provides local pLDDT/PAE analysis over previously downloaded JSON.

from .alphafold_operations import (
    download_alphafold_structure_by_uniprot_id,
    download_alphafold_metadata_by_uniprot_id,
)
from .alphafold_analyze import (
    analyze_alphafold_plddt_by_metadata_file,
    analyze_alphafold_pae_by_pae_file,
)

__all__ = [
    "download_alphafold_structure_by_uniprot_id",
    "download_alphafold_metadata_by_uniprot_id",
    "analyze_alphafold_plddt_by_metadata_file",
    "analyze_alphafold_pae_by_pae_file",
]
