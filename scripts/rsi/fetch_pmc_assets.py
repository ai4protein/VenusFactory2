"""Fetch a pinned PMC article version via the current official public S3 API.

No login, scraping challenge or legacy OA endpoint is used. Save metadata,
verify NCBI's per-object MD5 and record SHA256 for reproducible case acquisition.
Article figures are excluded; data attachments and textual sources are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import requests

BASE = "https://pmc-oa-opendata.s3.amazonaws.com/"


def resolve_asset(url, article_version):
    parsed = urlsplit(url)
    path = unquote(parsed.path).lstrip("/")
    if parsed.scheme != "s3" or parsed.netloc != "pmc-oa-opendata":
        raise ValueError("only official PMC bucket objects are accepted")
    parts = path.split("/")
    if len(parts) != 2 or parts[0] != article_version or parts[1] in {"", ".", ".."} or "\\" in path:
        raise ValueError("asset is outside the requested article version")
    query = parse_qs(parsed.query)
    md5 = query.get("md5", [""])[0]
    if set(query) != {"md5"} or not re.fullmatch(r"[0-9a-f]{32}", md5):
        raise ValueError("asset must carry the official object checksum")
    return BASE + path, parts[1], md5


def verified_bytes(content, expected_md5):
    if hashlib.md5(content).hexdigest() != expected_md5:
        raise ValueError("download differs from PMC metadata checksum")
    return {"md5": expected_md5, "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}


def main(args):
    if not re.fullmatch(r"PMC[1-9][0-9]*", args.pmcid) or args.version < 1:
        raise ValueError("explicit PMCID and positive article version required")
    article_version = args.pmcid + "." + str(args.version)
    metadata_url = BASE + article_version + "/" + article_version + ".json"
    response = requests.get(metadata_url, timeout=45, allow_redirects=False)
    if response.status_code != 200:
        raise ValueError("article-version metadata unavailable: HTTP " + str(response.status_code))
    metadata = response.json()
    if metadata.get("pmcid") != args.pmcid or metadata.get("version") != args.version:
        raise ValueError("article metadata identity mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "metadata.json").write_bytes(response.content)
    manifest = {
        "schema": 1,
        "metadata_url": metadata_url,
        "metadata_sha256": hashlib.sha256(response.content).hexdigest(),
        "attribution": "NIH NLM NCBI PubMed Central Article Datasets; publisher/author supplied files",
        "doi": metadata.get("doi"),
        "article_version": article_version,
        "license_code": metadata.get("license_code"),
        "is_manuscript": metadata.get("is_manuscript"),
        "is_retracted": metadata.get("is_retracted"),
        "files": [],
        "status": "downloading",
    }
    manifest_path = args.output / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    urls = [metadata.get("xml_url"), metadata.get("text_url"), *metadata.get("media_urls", [])]
    seen = set()
    for url in urls:
        if not url:
            continue
        https, name, checksum = resolve_asset(url, article_version)
        if Path(name).suffix.lower() in {".jpg", ".jpeg", ".png", ".gif", ".tif", ".tiff", ".svg"}:
            continue
        if name in seen:
            raise ValueError("duplicate asset filename")
        seen.add(name)
        with requests.get(https, timeout=(15, 90), stream=True, allow_redirects=False) as response:
            if response.status_code != 200:
                raise ValueError("asset unavailable: HTTP " + str(response.status_code))
            payload = bytearray()
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                payload.extend(chunk)
                if len(payload) > 64 * 1024**2:
                    raise ValueError("asset exceeds 64 MiB acquisition limit")
        record = {"url": https, "local_file": name, **verified_bytes(payload, checksum)}
        (args.output / name).write_bytes(payload)
        manifest["files"].append(record)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({"asset": name, "bytes": len(payload), "verified": True}), flush=True)
    manifest["status"] = "completed"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pmcid", required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
