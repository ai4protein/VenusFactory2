"""arXiv paper download (PDF / HTML / source tarball).

VF already has `query_arxiv_tool` for search; this complements it with full-text
download. Single entry: `download_arxiv_paper_by_id(arxiv_id, out_dir, format)`.

Adapted from google-deepmind/science-skills (Apache-2.0):
- skills/literature_search_arxiv/scripts/{download_paper.py, download_paper_source.py}
"""
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.tools.path_sanitizer import to_client_file_path


_PREVIEW_LEN = 500
_SOURCE = "arXiv"

# arxiv asks for 3-second polite spacing; we don't bind multiple requests here, but
# embed sensible per-request handling.
_VALID_FORMATS = {"pdf", "html", "source"}
_EXT = {"pdf": "pdf", "html": "html", "source": "tar.gz"}


def _session() -> requests.Session:
    s = requests.Session()
    r = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=frozenset({"GET"}))
    s.mount("https://", HTTPAdapter(max_retries=r))
    s.mount("http://", HTTPAdapter(max_retries=r))
    return s


def _error_response(error_type: str, message: str, suggestion: Optional[str] = None) -> str:
    out: Dict[str, Any] = {
        "status": "error",
        "error": {"type": error_type, "message": message},
        "file_info": None,
    }
    if suggestion:
        out["error"]["suggestion"] = suggestion
    return json.dumps(out, ensure_ascii=False)


def _success_response(file_path: str, content_preview: str, biological_metadata: Dict[str, Any], elapsed_ms: int, fmt: str) -> str:
    path = Path(file_path)
    size = path.stat().st_size if path.exists() else 0
    out: Dict[str, Any] = {
        "status": "success",
        "file_info": {
            "file_path": to_client_file_path(path if path.exists() else file_path),
            "file_name": path.name,
            "file_size": size,
            "format": fmt,
        },
        "content_preview": content_preview[:_PREVIEW_LEN],
        "biological_metadata": biological_metadata,
        "execution_context": {"elapsed_ms": elapsed_ms, "source": _SOURCE},
    }
    return json.dumps(out, ensure_ascii=False)


def _build_url(arxiv_id: str, fmt: str) -> str:
    if fmt == "pdf":
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    if fmt == "html":
        return f"https://arxiv.org/html/{arxiv_id}"
    # source tarball is hosted on export.arxiv.org
    return f"https://export.arxiv.org/e-print/{arxiv_id}"


def download_arxiv_paper_by_id(
    arxiv_id: str,
    out_dir: str,
    format: str = "pdf",
    timeout: int = 60,
) -> str:
    """Download an arXiv paper as PDF, HTML, or source tarball.

    `arxiv_id` may be the bare numeric ID (e.g. `2106.04559`), a version-tagged
    ID (`2106.04559v2`), or an old-style category id (`hep-th/9510017`).
    """
    t0 = time.perf_counter()
    arxiv_id = (arxiv_id or "").strip().lstrip("arxiv:").lstrip("arXiv:")
    if not arxiv_id:
        return _error_response("ValidationError", "empty arxiv_id")
    fmt = (format or "").strip().lower()
    if fmt not in _VALID_FORMATS:
        return _error_response("ValidationError", f"format must be one of {sorted(_VALID_FORMATS)}; got {fmt!r}")
    out_dir = str(out_dir or "").strip().rstrip(os.sep)
    if not out_dir:
        return _error_response("ValidationError", "empty out_dir")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    url = _build_url(arxiv_id, fmt)
    sess = _session()
    try:
        resp = sess.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "VenusFactory2 (https://venusfactory.cn)"},
            allow_redirects=True,
        )
    except requests.RequestException as e:
        return _error_response("NetworkError", str(e))

    if resp.status_code == 404:
        return _error_response(
            "NotFound",
            f"arXiv {fmt} for id {arxiv_id} not found (404)",
            suggestion="Verify the arXiv id; HTML format only exists for newer papers.",
        )
    if resp.status_code != 200:
        return _error_response("DownloadError", f"arXiv {resp.status_code}: {resp.text[:300]}")

    # Sanity: PDF should start with %PDF
    if fmt == "pdf" and not resp.content.startswith(b"%PDF"):
        return _error_response("DownloadError", "response is not a PDF (arXiv may have served an HTML error page)", suggestion="Try a different arxiv_id; check whether the abstract page exists.")

    safe_id = arxiv_id.replace("/", "_")
    out_path = os.path.join(out_dir, f"arxiv_{safe_id}.{_EXT[fmt]}")
    try:
        with open(out_path, "wb") as f:
            f.write(resp.content)
    except OSError as e:
        return _error_response("IOError", f"failed to write file: {e}")

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    meta = {
        "arxiv_id": arxiv_id,
        "format": fmt,
        "size_bytes": len(resp.content),
        "source_url": url,
    }
    preview = f"arXiv {fmt.upper()} {arxiv_id} — {len(resp.content)} bytes saved to {os.path.basename(out_path)}"
    return _success_response(out_path, preview, meta, elapsed_ms, fmt=fmt)


__all__ = ["download_arxiv_paper_by_id"]
