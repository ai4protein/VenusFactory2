"""
KEGG REST API entry point: re-exports from kegg_operations (query + download).
Backward compat: kegg_info, kegg_find, kegg_get(entries, option=...) etc.
Academic use only.
"""
from src.tools.search.database.kegg.kegg_rest import kegg_request, BASE_URL
from src.tools.search.database.kegg.kegg_operations import (
    kegg_info,
    kegg_list,
    kegg_find,
    kegg_get,
    kegg_conv,
    kegg_link,
    kegg_ddi,
    query_kegg_info,
    download_kegg_info,
    query_kegg_list,
    download_kegg_list,
    query_kegg_find,
    download_kegg_find,
    query_kegg_get,
    download_kegg_get,
    query_kegg_conv,
    download_kegg_conv,
    query_kegg_link,
    download_kegg_link,
    query_kegg_ddi,
    download_kegg_ddi,
)

KEGG_BASE_URL = BASE_URL

if __name__ == "__main__":
    print("Run tests: python src/tools/search/database/kegg/kegg_operations.py --test", flush=True)

__all__ = [
    "kegg_request",
    "BASE_URL",
    "KEGG_BASE_URL",
    "kegg_info",
    "kegg_list",
    "kegg_find",
    "kegg_get",
    "kegg_conv",
    "kegg_link",
    "kegg_ddi",
    "query_kegg_info",
    "download_kegg_info",
    "query_kegg_list",
    "download_kegg_list",
    "query_kegg_find",
    "download_kegg_find",
    "query_kegg_get",
    "download_kegg_get",
    "query_kegg_conv",
    "download_kegg_conv",
    "query_kegg_link",
    "download_kegg_link",
    "query_kegg_ddi",
    "download_kegg_ddi",
]
