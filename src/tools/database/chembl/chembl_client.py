"""
ChEMBL Web Resource client singleton. Academic / EBI terms apply.
Returns the shared new_client from chembl_webresource_client.
Skill: src/agent/skills/chembl_database/
"""
import sys
import types
from typing import Any

try:
    import importlib.metadata
    _HAS_IMPORTLIB_METADATA = True
except ImportError:
    _HAS_IMPORTLIB_METADATA = False


def _use_importlib_metadata_for_version() -> None:
    """Use importlib.metadata instead of deprecated pkg_resources for ChEMBL client version."""
    if "pkg_resources" in sys.modules:
        return
    stub = types.ModuleType("pkg_resources")

    def get_distribution(name: str) -> object:
        return type("Distribution", (), {"version": importlib.metadata.version(name)})()

    stub.get_distribution = get_distribution
    sys.modules["pkg_resources"] = stub


try:
    if _HAS_IMPORTLIB_METADATA:
        _use_importlib_metadata_for_version()
    from chembl_webresource_client.settings import Settings
    # Avoid hanging: library default NEW_CLIENT_TIMEOUT is None (no timeout)
    Settings.Instance().NEW_CLIENT_TIMEOUT = 30
    from chembl_webresource_client.new_client import new_client
    _client: Any = new_client
    CLIENT_AVAILABLE = True
except ImportError:
    _client = None
    CLIENT_AVAILABLE = False


def get_client():
    """Return the ChEMBL new_client. Raises ImportError if chembl_webresource_client not installed."""
    if not CLIENT_AVAILABLE:
        raise ImportError(
            "chembl_webresource_client is required. Install with: uv pip install chembl_webresource_client"
        )
    return _client


# Cap on filter() result size: list(queryset) without slice fetches ALL pages (can hang on huge sets).
# QuerySet supports [:n]; use this so we only request up to N results from the API.
DEFAULT_CHEMBL_MAX_FILTER_RESULTS = 500
# Upper bound for max_results parameter (avoid unbounded requests).
MAX_CHEMBL_FILTER_RESULTS_CAP = 5000

__all__ = [
    "get_client",
    "CLIENT_AVAILABLE",
    "DEFAULT_CHEMBL_MAX_FILTER_RESULTS",
    "MAX_CHEMBL_FILTER_RESULTS_CAP",
]
