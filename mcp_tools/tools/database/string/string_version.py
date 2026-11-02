"""STRING database version. Skill: src/agent/skills/string_database/."""
import urllib.error
import urllib.request

from .string_rest import STRING_BASE_URL


def string_version() -> str:
    """Get current STRING database version (TSV text)."""
    url = f"{STRING_BASE_URL}/tsv/version"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return f"Error: {e.code} - {e.reason}"
    except urllib.error.URLError as e:
        return f"Error: {e.reason}"
