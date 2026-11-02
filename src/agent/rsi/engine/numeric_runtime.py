"""Read-only numeric runtime description, usable inside an isolated model worker.

This is a diagnostic fingerprint, not a portable image or a relaxed replay gate.
Only selected numeric packages, mapped libraries and non-secret runtime settings
are recorded. No user environment dump, credentials or task data are collected.
"""

import hashlib
import os
import platform
import sys
from pathlib import Path


def file_identity(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise RuntimeError("Numeric runtime file changed while fingerprinting")
    return {"path": str(path), "bytes": after.st_size, "sha256": h.hexdigest()}


def capture_runtime():
    packages = {}
    for name in ("numpy", "scipy", "sklearn"):
        module = sys.modules.get(name)
        if module is not None:
            packages[name] = {
                "version": str(module.__version__),
                "module": file_identity(module.__file__),
            }
    libraries = set()
    maps = Path("/proc/self/maps")
    if maps.is_file():
        for line in maps.read_text().splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) != 6 or not fields[5].startswith("/"):
                continue
            name = fields[5]
            if any(
                token in name.lower()
                for token in (
                    "/numpy",
                    "/scipy",
                    "/sklearn",
                    "libopenblas",
                    "libblas",
                    "liblapack",
                    "libpython",
                    "libgomp",
                    "libmkl",
                )
            ):
                libraries.add(name)
    cpu = Path("/proc/cpuinfo")
    cpu_lines = cpu.read_text().splitlines() if cpu.is_file() else []
    models = sorted({line.split(":", 1)[1].strip() for line in cpu_lines if line.startswith("model name")})
    flags = sorted({flag for line in cpu_lines if line.startswith("flags") for flag in line.split(":", 1)[1].split()})
    return {
        "schema": 1,
        "python": {"version": sys.version, "executable": file_identity(sys.executable)},
        "platform": {"system": platform.system(), "machine": platform.machine(), "kernel": platform.release()},
        "cpu": {"models": models, "flags": flags},
        "packages": packages,
        "numeric_libraries": [file_identity(name) for name in sorted(libraries)],
        "thread_settings": {
            name: os.environ.get(name)
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONHASHSEED")
        },
        "scope": "Loaded numeric libraries and selected runtime settings; not a full environment image",
    }
