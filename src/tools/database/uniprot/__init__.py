# UniProt: seq (query/download), meta (query/download), search, mapping

from .uniprot_operations import (
    download_uniprot_search_by_query,
    download_uniprot_retrieve_by_id,
    download_uniprot_mapping,
    download_uniprot_seq_by_id,
    download_uniprot_meta_by_id,
)

# For backwards compatibility with other modules if they directly import these
from .uniprot_sequence import query_uniprot_seq, download_uniprot_seq, download_uniprot_sequence
from .uniprot_metadata import query_uniprot_meta, download_uniprot_meta
from .uniprot_search import uniprot_search, uniprot_retrieve, uniprot_mapping, uniprot_search_and_retrieve

__all__ = [
    "download_uniprot_search_by_query",
    "download_uniprot_retrieve_by_id",
    "download_uniprot_mapping",
    "download_uniprot_seq_by_id",
    "download_uniprot_meta_by_id",
]
