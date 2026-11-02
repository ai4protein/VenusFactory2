"""Cache public NucB author assets with byte, ETag/MD5 and SHA-256 checks."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import threading
import time
import urllib.request
from pathlib import Path

BUCKET = "https://storage.googleapis.com/nuclease_design"


def write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def fingerprints(path):
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            sha.update(block)
            md5.update(block)
    return sha.hexdigest(), md5.hexdigest()


def main(args):
    rows = json.loads((args.assets / "gcs_listing.json").read_text())
    rows = [r for r in rows if not r["Key"].endswith("_$folder$")]
    args.output.mkdir(parents=True, exist_ok=False)
    data = args.assets / "gcs"
    data.mkdir(exist_ok=True)
    report = {
        "status": "running",
        "source": BUCKET,
        "planned_files": len(rows),
        "planned_bytes": sum(int(r["Size"]) for r in rows),
        "files": [],
    }
    write(
        args.output / "protocol.json",
        {
            "listing": rows,
            "concurrency": 2,
            "entry_sha256": fingerprints(Path(__file__))[0],
            "source_manifest_sha256": fingerprints(args.assets / "download_manifest.json")[0],
            "policy": "Freeze public objects from listing; no automatic retry or reference overwrite.",
        },
    )
    write(args.output / "report.json", report)
    lock = threading.Lock()

    def fetch(row):
        name = row["Key"]
        target = (data / name).resolve()
        target.relative_to(data.resolve())
        expected_size, expected_md5 = int(row["Size"]), row["ETag"].strip('"')
        assert len(expected_md5) == 32 and all(c in "0123456789abcdef" for c in expected_md5)
        record = {
            "path": name,
            "url": BUCKET + "/" + name,
            "expected_bytes": expected_size,
            "expected_etag": row["ETag"],
            "last_modified": row["LastModified"],
            "status": "running",
        }
        with lock:
            report["files"].append(record)
            write(args.output / "report.json", report)
        started = time.monotonic()
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            existed = target.exists()
            part = target.with_suffix(target.suffix + ".part")
            if not existed:
                req = urllib.request.Request(record["url"], headers={"If-Match": row["ETag"]})
                with urllib.request.urlopen(req, timeout=90) as response, part.open("xb") as out:
                    record["response_etag"] = response.headers.get("ETag")
                    record["gcs_generation"] = response.headers.get("x-goog-generation")
                    for block in iter(lambda: response.read(4 * 1024 * 1024), b""):
                        out.write(block)
                checked = part
            else:
                checked = target
            sha, md5 = fingerprints(checked)
            if checked.stat().st_size != expected_size or md5 != expected_md5:
                raise ValueError("public object differs from frozen GCS size/MD5")
            if not existed:
                part.replace(target)
            record.update(status="verified", sha256=sha, md5=md5, bytes=expected_size, reused=existed)
        except Exception as exc:
            record.update(status="failed", error_type=type(exc).__name__)
        record["wall_seconds"] = time.monotonic() - started
        with lock:
            write(args.output / "report.json", report)
        print(json.dumps({"path": name, "status": record["status"]}), flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(fetch, rows))
    report["status"] = "verified" if all(x["status"] == "verified" for x in report["files"]) else "incomplete"
    write(args.output / "report.json", report)
    return report["status"] == "verified"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    raise SystemExit(0 if main(p.parse_args()) else 1)
