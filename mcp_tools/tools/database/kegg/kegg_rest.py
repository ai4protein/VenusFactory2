"""
KEGG REST API base client. Academic use only.
Base URL: https://rest.kegg.jp/

Skill and reference (for Agent): src/agent/skills/kegg/
  - SKILL.md
  - references/kegg_reference.md
"""
import urllib.error
import urllib.parse
import urllib.request
from typing import List, Optional, Union

BASE_URL = "https://rest.kegg.jp/"


def _join_ids(entry_id: Union[str, List[str]]) -> str:
    """Format one or multiple entry IDs for URL (max 10; + separated)."""
    if isinstance(entry_id, list):
        return "+".join(str(x).strip() for x in entry_id[:10])
    return str(entry_id).strip()


def kegg_request(operation: str, *path_parts: str, query_params: Optional[dict] = None) -> str:
    """
    Perform a GET request to KEGG REST API. Returns response text.
    operation: info, list, find, get, conv, link, ddi
    path_parts: encoded path segments (e.g. 'pathway', 'hsa').
    query_params: optional dict of query parameters (not all operations use them).
    """
    path = "/".join(urllib.parse.quote(str(p), safe="+:") for p in path_parts if p is not None and str(p))
    url = urllib.parse.urljoin(BASE_URL, f"{operation}/{path}")
    if query_params:
        url += "?" + urllib.parse.urlencode(query_params)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return f"URL error: {e.reason}"
    except Exception as e:
        return f"Error: {e}"
