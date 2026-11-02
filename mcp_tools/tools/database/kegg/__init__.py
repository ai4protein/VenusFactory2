# KEGG REST API: query/download (academic use only). See kegg.md.
# All query_*/download_* return rich JSON (status, content/file_info, content_preview, biological_metadata, execution_context).

from .kegg_rest import kegg_request, BASE_URL
from .kegg_operations import (
    download_kegg_info_by_database,
    download_kegg_list_by_database,
    download_kegg_find_by_database,
    download_kegg_entry_by_id,
    download_kegg_conv_by_id,
    download_kegg_link_by_id,
    download_kegg_ddi_by_id,
)

__all__ = [
    "download_kegg_info_by_database",
    "download_kegg_list_by_database",
    "download_kegg_find_by_database",
    "download_kegg_entry_by_id",
    "download_kegg_conv_by_id",
    "download_kegg_link_by_id",
    "download_kegg_ddi_by_id",
]
