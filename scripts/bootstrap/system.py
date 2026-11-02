from __future__ import annotations

import os
import json
import platform
import re
import shlex
import shutil
import sys
from pathlib import Path

from .assets import download, fetch_json
from .common import atomic_json, capture, digest, inside, read_json


def executable(name):
    found = shutil.which(name)
    if found:
        return found
    candidates = [Path.home() / ".local/bin" / name]
    if name == "conda":
        candidates += [Path.home() / p / "bin/conda" for p in ("miniforge3", "miniconda3", "anaconda3")]
    if name in ("node", "npm"):
        candidates += sorted(Path.home().glob(f".nvm/versions/node/*/bin/{name}"), reverse=True)
    return next((str(p) for p in candidates if p.is_file() and os.access(p, os.X_OK)), None)


def probe(root):
    nvidia = executable("nvidia-smi")
    cuda, gpu = None, False
    if nvidia:
        try:
            text = capture([nvidia], timeout=10)
            match = re.search(r"CUDA Version:\s*(\d+\.\d+)", text)
            cuda = float(match.group(1)) if match else None
            gpu = bool(capture([nvidia, "-L"], timeout=10))
        except Exception:
            pass
    return {"os": platform.system(), "arch": platform.machine(),
            "python": platform.python_version(), "gpu": gpu, "driver_cuda": cuda,
            "torch_type": "cu128" if gpu and cuda is not None and cuda >= 12.8 else "cpu",
            "free_gib": round(shutil.disk_usage(root).free / 2**30, 1),
            "executables": {x: executable(x) for x in ("git", "uv", "conda", "node", "npm", "bwrap", "gcc")}}


class Environment:
    def __init__(self, root, run, manager="auto", torch_type="cpu"):
        self.root, self.run = root, run
        self.base = inside(root, "runtime/local/bootstrap")
        self.manager = choose_manager(manager)
        self.torch_type = torch_type
        self.command = None

    def ensure_manager(self):
        if self.command:
            return self.command
        existing = executable(self.manager)
        if existing:
            self.command = [existing]
        elif self.manager == "uv":
            folder = self.base / "uv-manager"
            binary = folder / "bin/uv"
            if not binary.is_file():
                # A stock Linux Python may have neither pip nor ensurepip.
                # The official wheel contains a standalone executable; bootstrap
                # it directly without modifying the system Python installation.
                import zipfile
                metadata = fetch_json("https://pypi.org/pypi/uv/0.9.5/json")
                arch = platform.machine()
                wheels = [item for item in metadata["urls"] if item["filename"].endswith(".whl")
                          and f"manylinux_2_17_{arch}" in item["filename"]]
                if len(wheels) != 1:
                    raise ValueError("No supported official uv wheel for this architecture")
                item = wheels[0]
                archive = download(item["url"], folder / item["filename"], expected=item["digests"]["sha256"])
                with zipfile.ZipFile(archive) as wheel:
                    candidates = [entry for entry in wheel.infolist() if Path(entry.filename).name == "uv" and not entry.is_dir()]
                    if len(candidates) != 1:
                        raise ValueError("Official uv wheel has an unexpected executable layout")
                    binary.parent.mkdir(parents=True, exist_ok=True)
                    temporary = binary.with_suffix(".tmp")
                    with wheel.open(candidates[0]) as source, temporary.open("wb") as target:
                        shutil.copyfileobj(source, target)
                    temporary.chmod(0o755)
                    os.replace(temporary, binary)
                atomic_json(folder / "installation.json", {"version": "0.9.5", "url": item["url"],
                                                           "sha256": item["digests"]["sha256"], "binary_sha256": digest(binary)})
            self.command = [str(binary)]
        else:
            folder = self.base / "miniforge"
            binary = folder / "bin/conda"
            if not binary.exists():
                arch = {"x86_64": "x86_64", "aarch64": "aarch64"}.get(platform.machine())
                if not arch:
                    raise ValueError("Miniforge bootstrap supports Linux x86_64/aarch64")
                release = fetch_json("https://api.github.com/repos/conda-forge/miniforge/releases/latest")
                filename = f"Miniforge3-Linux-{arch}.sh"
                assets = {a["name"]: a["browser_download_url"] for a in release["assets"]}
                checks = download(assets[filename + ".sha256"], self.base / (release["tag_name"] + ".sha256"), max_bytes=4096)
                expected = checks.read_text().split()[0]
                if not re.fullmatch(r"[0-9a-f]{64}", expected):
                    raise ValueError("Invalid Miniforge checksum")
                installer = download(assets[filename], self.base / filename, expected=expected)
                self.run(["bash", installer, "-b", "-p", folder])
            self.command = [str(binary)]
        return self.command

    def create(self, folder: Path, version):
        inside(self.root, folder)
        python = folder / "bin/python"
        marker = folder / ".venus-bootstrap.json"
        if python.exists():
            if not marker.exists():
                raise ValueError(f"Environment already exists and is not initializer-owned: {folder}. Use --use-existing-environment for platform dependencies")
            return python
        if folder.exists() and any(folder.iterdir()):
            raise ValueError(f"Incomplete environment at {folder}; inspect/move it aside before retrying")
        self.ensure_manager()
        if self.manager == "uv":
            self.run([*self.command, "venv", "--seed", "--python", version, folder])
        else:
            self.run([*self.command, "create", "--yes", "--prefix", folder,
                      "--override-channels", "-c", "conda-forge", f"python={version}", "pip"])
        atomic_json(marker, {"manager": self.manager, "python": version})
        return python

    def pip(self, python, *packages, torch=False, links=None, constraints=None):
        self.ensure_manager()
        if self.manager == "uv":
            command = [*self.command, "pip", "install", "--python", python]
        else:
            command = [python, "-m", "pip", "install"]
        command += list(packages)
        command += ["--index-url", f"https://download.pytorch.org/whl/{self.torch_type}" if torch else "https://pypi.org/simple"]
        if links:
            command += ["--find-links", links, "--only-binary", ":all:"]
        if constraints:
            command += ["--constraint", constraints]
        self.run(command)

    def platform(self, folder, reuse=False):
        python = folder / "bin/python"
        if reuse:
            if not python.is_file():
                raise ValueError(f"Existing environment has no Python: {python}")
            return python
        python = self.create(folder, "3.12")
        signature = {"pyproject": digest(self.root / "pyproject.toml"), "torch_type": self.torch_type,
                     "constraints": digest(self.root / "requirements/bootstrap-constraints.txt"), "recipe": 3}
        receipt = folder / ".venus-dependencies.json"
        if read_json(receipt) != signature:
            self.pip(python, "torch==2.8.0", "torchvision==0.23.0", torch=True)
            links = f"https://data.pyg.org/whl/torch-2.8.0+{self.torch_type}.html"
            self.pip(python, "pyg_lib", "torch_scatter", "torch_sparse", "torch_cluster", "torch_spline_conv", links=links)
            constraint = self.root / "requirements/bootstrap-constraints.txt"
            # Read metadata with the newly created Python 3.12, even when the
            # bootstrap itself runs on Python 3.10. pip cannot consume TOML via -r.
            dependencies = json.loads(capture([
                python, "-c",
                "import json,sys,tomllib; from pathlib import Path; "
                "print(json.dumps(tomllib.loads(Path(sys.argv[1]).read_text())['project']['dependencies']))",
                self.root / "pyproject.toml",
            ]))
            if not isinstance(dependencies, list) or not all(isinstance(item, str) and item for item in dependencies):
                raise ValueError("Invalid project dependency metadata")
            self.pip(python, "torch_geometric", *dependencies, constraints=constraint)
            self.run([python, "-m", "pip", "check"])
            self.run([python, "-c", "import fastapi,uvicorn,fastmcp,torch,transformers,torch_geometric,peft"])
            atomic_json(receipt, signature)
        return python


def choose_manager(manager='auto'):
    if manager != 'auto':
        return manager
    return 'uv' if executable('uv') else ('conda' if executable('conda') else 'uv')


def node_runtime(root):
    base = inside(root, "runtime/local/bootstrap")
    node = executable("node")
    if node:
        major = int(capture([node, "--version"]).lstrip("v").split(".")[0])
        if major != 24:
            node = None
    if not node:
        candidates = sorted(Path.home().glob(".nvm/versions/node/v24.*/bin/node"), reverse=True)
        node = next((str(p) for p in candidates if p.is_file()), None)
    if not node:
        arch = {"x86_64": "x64", "aarch64": "arm64"}.get(platform.machine())
        if not arch:
            raise ValueError("Install Node.js 24 LTS for this architecture")
        version = "v24.15.0"
        filename = f"node-{version}-linux-{arch}.tar.xz"
        checks = download(f"https://nodejs.org/dist/{version}/SHASUMS256.txt", base / f"node-{version}-sha256.txt")
        entries = dict(reversed(line.split()) for line in checks.read_text().splitlines() if len(line.split()) == 2)
        archive = download(f"https://nodejs.org/dist/{version}/{filename}", base / filename, expected=entries[filename])
        target = base / filename.removesuffix(".tar.xz")
        if not target.is_dir():
            # Node distributions include safe relative symlinks; extract with tar's
            # Python data filter into a fresh staging directory, never over source.
            import tarfile, tempfile
            with tempfile.TemporaryDirectory(dir=base) as temp:
                with tarfile.open(archive) as handle:
                    if not hasattr(tarfile, "data_filter"):
                        raise ValueError("Node bootstrap requires Python >=3.12; use the new platform Python")
                    handle.extractall(temp, filter="data")
                os.replace(Path(temp) / target.name, target)
        node = str(target / "bin/node")
    npm = Path(node).parent / "npm"
    if not npm.exists():
        raise ValueError("Node.js installation has no npm")
    return Path(node), npm


def frontend(root, run, python=None):
    base = inside(root, "runtime/local/bootstrap")
    node, npm = node_runtime(root)
    platform_python = Path(python) if python and Path(python).is_file() else Path(sys.executable)
    # npm's documentation prebuild calls `python`; bootstrap may itself have
    # been invoked as python3 without any activated environment.
    shim = inside(root, base / "bin")
    shim.mkdir(parents=True, exist_ok=True)
    launcher = shim / "python"
    header = "#!/bin/sh\n# VenusFactory2 bootstrap\n"
    if launcher.exists() and not launcher.read_text().startswith(header):
        raise ValueError("Unexpected bootstrap Python launcher; inspect it before continuing")
    launcher.write_text(header + "exec " + shlex.quote(str(platform_python)) + ' "$@"\n', encoding="utf-8")
    launcher.chmod(0o755)
    env = {"PATH": os.pathsep.join([str(Path(node).parent), str(shim), os.environ.get("PATH", "")])}
    run([npm, "ci", "--no-audit", "--no-fund"], cwd=root / "frontend", extra_env=env)
    run([npm, "run", "build"], cwd=root / "frontend", extra_env=env)
