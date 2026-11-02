"""Fetch only author GB1/TrpB fitness tables from a 5.8 GB Zenodo ZIP.

Enforces exact HTTP Content-Range and ZIP CRC. No embeddings are transferred.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

URL = "https://zenodo.org/api/records/12196802/files/data.zip/content"
SIZE = 5867321110


class RangeFile(io.RawIOBase):
    def __init__(self):
        self.pos = 0
        self.transferred = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("invalid seek")
        self.pos = offset if whence == 0 else (self.pos + offset if whence == 1 else SIZE + offset)
        if not 0 <= self.pos <= SIZE:
            raise ValueError("seek outside archive")
        return self.pos

    def read(self, n=-1):
        n = min(SIZE - self.pos, n if n >= 0 else SIZE - self.pos)
        if n == 0:
            return b""
        if n < 0 or n > 32 * 1024**2:
            raise ValueError("unexpected large range")
        end = self.pos + n - 1
        req = urllib.request.Request(
            URL + "?range_start=" + str(self.pos), headers={"Range": f"bytes={self.pos}-{end}"}
        )
        with urllib.request.urlopen(req, timeout=60) as res:
            if res.status != 206 or res.headers.get("Content-Range") != f"bytes {self.pos}-{end}/{SIZE}":
                raise ValueError("invalid range response; full download refused")
            data = res.read(n + 1)
            if len(data) != n:
                raise ValueError("range length differs")
        self.pos += n
        self.transferred += n
        return data


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    remote = RangeFile()
    records = []
    with zipfile.ZipFile(remote) as archive:
        for name in ["data/GB1/fitness.csv", "data/TrpB/fitness.csv"]:
            info = archive.getinfo(name)
            if info.file_size > 10 * 1024**2:
                raise ValueError("unexpected table size")
            data = archive.read(name)
            filename = name.split("/")[1] + "_fitness.csv"
            path = output / filename
            if path.exists() and path.read_bytes() != data:
                raise ValueError("cached source changed")
            path.write_bytes(data)
            records.append(
                {
                    "member": name,
                    "local_file": filename,
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "zip_crc32": info.CRC,
                }
            )
    report = {
        "record": "https://zenodo.org/records/12196802",
        "archive_url": URL,
        "archive_bytes": SIZE,
        "transferred_bytes": remote.transferred,
        "files": records,
        "labels_warning": "Preserve fitness values in the released simulation tables, including zeros. Never add absent variants with production-domain placeholder zeros.",
    }
    (output / "source_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args().output)
