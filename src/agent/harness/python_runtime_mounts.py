"""Read-only Python runtime mounts shared by native harnesses and code sandboxes."""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re
import subprocess
import sys


def runtime_mounts(*, prefix, base_prefix, executable, version):
    prefix, base = Path(prefix).resolve(), Path(base_prefix).resolve()
    if not re.fullmatch(r"3\.\d+", version):
        raise ValueError("Unsupported Python runtime version")
    executable = str(Path(executable).resolve())
    args = ["--dir", "/opt/venus-runtime/bin"]
    for name in ("lib", "lib64"):
        library = base / name
        if library.is_dir():
            args += ["--ro-bind", str(library.resolve()), "/opt/venus-runtime/" + name]
    if prefix != base:
        for name in ("lib", "lib64"):
            packages = prefix / name / ("python" + version) / "site-packages"
            if not packages.is_dir():
                continue
            stdlib = base / name / ("python" + version)
            destination = "/opt/venus-runtime/" + name + "/python" + version
            # System Python may have no site-packages mount point. Compose the
            # standard-library directory privately; do not modify the host or
            # try to create a directory below an already read-only bind mount.
            compose = stdlib.is_dir() and not (stdlib / "site-packages").exists()
            if compose:
                args += ["--tmpfs", destination]
                for entry in sorted(stdlib.iterdir()):
                    args += ["--ro-bind", str(entry.resolve()), destination + "/" + entry.name]
            args += ["--ro-bind", str(packages.resolve()), destination + "/site-packages"]
            if compose:
                args += ["--remount-ro", destination]
    for name in ("python", "python3", "python" + version):
        args += ["--ro-bind", executable, "/opt/venus-runtime/bin/" + name]
    return args


@lru_cache(maxsize=16)
def _external_runtime(root: str):
    # root comes only from administrator-owned harness configuration.
    executable = Path(root) / "bin/python"
    if not executable.is_file():
        raise ValueError("Configured harness Python is unavailable")
    code = ("import json,sys; print(json.dumps(dict(prefix=sys.prefix,"
            "base_prefix=sys.base_prefix,executable=sys.executable,"
            "version=str(sys.version_info.major)+'.'+str(sys.version_info.minor))))")
    response = subprocess.run(
        [str(executable), "-I", "-c", code], check=True, capture_output=True,
        text=True, timeout=15, env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
    )
    metadata = json.loads(response.stdout)
    if Path(metadata["prefix"]).resolve() != Path(root).resolve():
        raise ValueError("Configured harness Python does not own its runtime directory")
    return metadata


def python_runtime_mounts(python_root):
    root = Path(python_root).resolve()
    if root == Path(sys.prefix).resolve():
        metadata = dict(prefix=sys.prefix, base_prefix=sys.base_prefix,
                        executable=sys.executable,
                        version=f"{sys.version_info.major}.{sys.version_info.minor}")
    else:
        metadata = _external_runtime(str(root))
    return runtime_mounts(**metadata)
