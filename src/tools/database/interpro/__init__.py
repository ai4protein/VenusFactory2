# InterPro: single exit via interpro_operations (query_interpro_*, download_interpro_*).

from .interpro_operations import (
    download_interpro_metadata_by_id,
    download_interpro_annotations_by_uniprot_id,
    download_interpro_proteins_by_id,
    download_interpro_uniprot_list_by_id,
)

__all__ = [
    "download_interpro_metadata_by_id",
    "download_interpro_annotations_by_uniprot_id",
    "download_interpro_proteins_by_id",
    "download_interpro_uniprot_list_by_id",
]
