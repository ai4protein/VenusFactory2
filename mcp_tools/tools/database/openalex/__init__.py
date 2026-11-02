# OpenAlex: scholarly works / authors / institutions / sources / topics REST API.

from .openalex_operations import (
    download_openalex_entries_by_query,
    download_openalex_entry_by_id,
)

__all__ = [
    "download_openalex_entries_by_query",
    "download_openalex_entry_by_id",
]
