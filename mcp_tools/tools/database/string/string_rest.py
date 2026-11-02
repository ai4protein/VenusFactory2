"""
STRING REST API base client. See string_reference.md and SKILL in src/agent/skills/string_database/.
"""
import urllib.error
import urllib.parse
import urllib.request
from typing import List, Union

STRING_BASE_URL = BASE_URL = "https://string-db.org/api"


def _identifiers_post(identifiers: Union[str, List[str]]) -> str:
    """Format identifiers for POST body (newline-separated)."""
    if isinstance(identifiers, list):
        return "\n".join(str(x).strip() for x in identifiers)
    return str(identifiers).strip()


def _identifiers_get(identifiers: Union[str, List[str]]) -> str:
    """Format identifiers for GET query (%%0d separated)."""
    if isinstance(identifiers, list):
        return "%0d".join(str(x).strip() for x in identifiers)
    return str(identifiers).strip()


def string_request_get(url: str, params: dict) -> str:
    """GET request; returns response text. On HTTP error returns 'Error: {code} - {reason}'."""
    full_url = url + "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(full_url, timeout=60) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return f"Error: {e.code} - {e.reason}"
    except urllib.error.URLError as e:
        return f"Error: {e.reason}"


def string_request_post(url: str, data: dict) -> str:
    """POST request with form-encoded data; returns response text."""
    body = urllib.parse.urlencode(data).encode("utf-8")
    try:
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return f"Error: {e.code} - {e.reason}"
    except urllib.error.URLError as e:
        return f"Error: {e.reason}"


def string_request_image_get(url: str, params: dict):
    """GET request for binary (PNG); returns bytes. On error returns error message as bytes."""
    full_url = url + "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(full_url, timeout=60) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        return f"Error: {e.code} - {e.reason}".encode("utf-8")
    except urllib.error.URLError as e:
        return f"Error: {e.reason}".encode("utf-8")
