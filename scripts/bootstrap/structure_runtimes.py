"""Isolated, pinned installations of all-atom structure predictors.

The installer prepares native CLIs. It never publishes a tool or starts a Web worker.
AlphaFold's sequence databases are a separate, explicit installation choice.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

from . import progress
from .assets import clone, download, extract
from .common import atomic_json, capture, digest, inside, read_json
from .system import Environment

COMMON = (
    "components.cif", "components.cif.rdkit_mol.pkl", "clusters-by-entity-40.txt",
    "obsolete_release_date.csv", "obsolete_to_successor.json", "release_date_cache.json",
)


def source_snapshot(run, spec, target):
    """Use a pinned official archive; Git transport is not required for these tools."""
    source = target / "source"
    repository, revision = spec["repository"], spec["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Structure runtime sources must use a full commit hash")
    match = re.fullmatch(r"https://github.com/([\w.-]+/[\w.-]+)\.git", repository)
    if not match:
        raise ValueError("Expected an official GitHub source repository")
    identity = {"repository": repository, "revision": revision}
    if source.exists():
        if (source / ".git").is_dir():
            return clone(run, repository, revision, source)
        if read_json(source / ".venus-source.json", {}).get("source") != identity:
            raise ValueError("Existing source is not this managed revision; preserve it separately")
        return revision
    archive = download(
        f"https://codeload.github.com/{match[1]}/tar.gz/{revision}",
        target / "downloads" / f"source-{revision}.tar.gz", max_bytes=2**30,
    )
    with tempfile.TemporaryDirectory(prefix=".source-", dir=target) as temporary:
        unpack = Path(temporary)
        extract(archive, unpack, max_bytes=2**30)
        children = list(unpack.iterdir())
        if len(children) != 1 or not children[0].is_dir():
            raise ValueError("Unexpected source archive structure")
        atomic_json(children[0] / ".venus-source.json",
                    {"source": identity, "archive_sha256": digest(archive)})
        os.replace(children[0], source)
    return revision


def data_archive(target, url, expected_paths, *, expected=None):
    """Atomic unpacking and a completion marker allow safe interrupted retries."""
    data = inside(target, "data")
    marker = target / "downloads" / (url.rsplit("/", 1)[1] + ".extracted.json")
    required = [inside(data, p) for p in expected_paths]
    if (read_json(marker, {}).get("url") == url
            and (not expected or read_json(marker, {}).get("sha256") == expected)
            and all(p.is_file() and p.stat().st_size for p in required)):
        return required
    archive = download(url, target / "downloads" / url.rsplit("/", 1)[1], expected=expected, max_bytes=4 * 2**30)
    with tempfile.TemporaryDirectory(prefix=".data-", dir=target) as temporary:
        unpack = Path(temporary)
        extract(archive, unpack, max_bytes=16 * 2**30)
        if not all((unpack / p).is_file() and (unpack / p).stat().st_size for p in expected_paths):
            raise ValueError("Reference archive is incomplete")
        data.mkdir(exist_ok=True)
        for item in unpack.iterdir():
            destination = inside(data, item.name)
            if item.is_dir():
                shutil.copytree(item, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(item, destination)
    atomic_json(marker, {"url": url, "sha256": digest(archive)})
    return required


def runtime_environment(name, target, python):
    cache = inside(target, "cache")
    cache.mkdir(exist_ok=True)
    env = {
        "PATH": os.pathsep.join([str(python.parent), str(target / "bioenv/bin"), os.environ.get("PATH", "")]),
        "XDG_CACHE_HOME": str(cache), "TORCH_HOME": str(cache / "torch"),
        "TORCH_EXTENSIONS_DIR": str(cache / "extensions"), "TRITON_CACHE_DIR": str(cache / "triton"),
        "HF_HOME": str(cache / "huggingface"), "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "MOL_ROOT_DIR": str(target / "data"), "PROTENIX_ROOT_DIR": str(target / "data"),
    }
    if (target / "bioenv/bin/nvcc").is_file():
        env["CUDA_HOME"] = str(target / "bioenv")
        wheel_nvidia = list((python.parent.parent / "lib").glob("python*/site-packages/nvidia"))
        include = list((target / "bioenv/targets").glob("*/include"))
        libraries = list((target / "bioenv/targets").glob("*/lib"))
        for folder in wheel_nvidia:
            include += list(folder.glob("*/include"))
            libraries += list(folder.glob("*/lib"))
        # Conda CUDA uses targets/<arch>, while Torch wheels provide library headers.
        env["CPATH"] = os.pathsep.join(map(str, include))
        env["LIBRARY_PATH"] = os.pathsep.join(map(str, libraries))
        env["LD_LIBRARY_PATH"] = env["LIBRARY_PATH"]
    if name == "alphafold2":
        env["PYTHONPATH"] = str(target / "source")
        # The upstream CUDA 12 JAX wheel needs its bundled CUDA libraries.
        nvidia = next(iter((python.parent.parent / "lib").glob("python*/site-packages/nvidia")), None)
        if nvidia:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(str(p) for p in nvidia.glob("*/lib"))
            nvcc = nvidia / "cuda_nvcc"
            if nvcc.is_dir():
                env["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=" + str(nvcc)
    return env


def bio_dependencies(r, name, target):
    # Native alignment/search binaries do not belong in the platform Python.
    manager = Environment(r.root, r.run, manager="conda", torch_type="cpu")
    manager.create(target / "bioenv", "3.11")
    conda = manager.ensure_manager()[0]
    packages = ["hmmer=3.3.2", "kalign2=2.04", "hhsuite=3.3.0", "aria2", "rsync"] if name == "alphafold2" else ["hmmer=3.4", "kalign3=3.3.5"]
    if name in {"protenix", "mol1"} and getattr(r, "gpu", False) and r.env.torch_type != "cpu":
        cuda = "12.6" if name == "protenix" else "12.8"
        packages += [f"cuda-nvcc={cuda}", f"cuda-cudart-dev={cuda}", f"cuda-cccl={cuda}"]
    marker = target / "bioenv" / ".venus-bio-packages.json"
    if read_json(marker, {}).get("packages") != packages:
        r.run([conda, "install", "--yes", "--prefix", target / "bioenv", "--override-channels",
               "-c", "conda-forge", "-c", "bioconda", "-c", "nvidia", *packages], timeout=3600)
        atomic_json(marker, {"packages": packages})


def dependencies(r, name, target, source, spec):
    python = r.env.create(target / "env", spec["python"])
    gpu = r.gpu and r.env.torch_type != "cpu"
    signature = {"revision": spec["revision"], "gpu": gpu, "recipe": 2}
    marker = target / "dependencies.json"
    if read_json(marker) != signature:
        if name == "protenix":
            # Keep the source's complete dependency contract, including training support.
            r.env.pip(python, str(source), "cuequivariance==0.8.0", "click>=8.1,<9")
        elif name == "mol1":
            # CUDA 12 is compatible with a wider range of drivers; each extra matches Torch.
            r.env.pip(python, "torch==2.10.0", torch=True)
            r.env.pip(python, str(source) + ("[cuda12]" if gpu else "[cpu]"))
        else:
            r.env.pip(python, "-r", str(source / "requirements.txt"))
            if gpu:
                r.env.pip(python, "jax[cuda12_pip]==0.4.26", "jaxlib==0.4.26+cuda12.cudnn89",
                          "nvidia-cudnn-cu12==8.9.7.29", "nvidia-cuda-nvcc-cu12==12.2.140",
                          links="https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")
        atomic_json(marker, signature)
    bio_dependencies(r, name, target)
    return python


def alphafold_databases(r, target, choice):
    if choice == "none":
        return
    # Called only after an explicit CLI choice; full profile alone never downloads these.
    required_gib = {"reduced": 650, "full": 3000}[choice]
    if shutil.disk_usage(target).free < required_gib * 2**30:
        raise ValueError(f"AlphaFold2 {choice} databases require at least {required_gib} GiB free")
    env = {"PATH": str(target / "bioenv/bin") + os.pathsep + os.environ.get("PATH", "")}
    args = ["bash", target / "source/scripts/download_all_data.sh", target / "data"]
    if choice == "reduced":
        args.append("reduced_dbs")
    r.run(args, extra_env=env, timeout=7 * 86400)
    atomic_json(target / "database-installation.json", {"preset": choice, "installed_at": time.time()})


def install(r, name):
    spec = r.manifest["tools"][name]
    target = inside(r.root, r.base / name)
    target.mkdir(parents=True, exist_ok=True)
    record = target / "installation.json"
    previous = read_json(record, {})
    existing = read_json(r.config, {}).get(name)
    template = read_json(r.root / "configs/model_runtimes.example.json", {}).get(name)
    if existing and not previous and existing != template:
        return {"status": "existing", "message": "Administrator-owned runtime preserved"}
    if name == "protenix" and r.env.torch_type == "cpu":
        return {"status": "blocked", "message": "Protenix needs CUDA compilation; select a supported GPU host and CUDA profile"}
    if shutil.disk_usage(target).free < spec.get("minimum_free_gib", 30) * 2**30:
        raise ValueError("Insufficient free space for this isolated runtime")
    progress.ui.line(f"  {name} [1/5] Pinned source")
    revision = source_snapshot(r.run, spec, target)
    source = target / "source"
    progress.ui.line(f"  {name} [2/5] Private Python and alignment tools")
    python = dependencies(r, name, target, source, spec)
    progress.ui.line(f"  {name} [3/5] Model weights and reference data")
    if name in {"protenix", "mol1"}:
        base = spec["asset_base"]
        required = data_archive(target, base + "/common.tar.gz", ["common/" + p for p in COMMON],
                                expected=spec.get("asset_sha256", {}).get("common.tar.gz"))
        if name == "mol1":
            required += data_archive(target, base + "/search_database.tar.gz", [spec["sequence_database"]],
                                     expected=spec.get("asset_sha256", {}).get("search_database.tar.gz"))
        weight = download(base + "/checkpoint/" + spec["checkpoint"],
                          target / "data/checkpoint" / spec["checkpoint"], expected=spec.get("checkpoint_sha256"), max_bytes=8 * 2**30)
        required.append(weight)
    else:
        # Same reference table bundled by gget; the upstream GitLab raw URL can return 403.
        stereo = download(
            "https://raw.githubusercontent.com/scverse/gget/"
            "2838b068e4fccb0cc6a056580fe2038ef7ec3b5d/gget/bins/alphafold/stereo_chemical_props.txt",
            source / "alphafold/common/stereo_chemical_props.txt",
            expected="24510899eeb49167cffedec8fa45363a4d08279c0c637a403b452f7d0ac09451",
            max_bytes=2**20)
        archive = download(spec["weights_url"], target / "downloads/alphafold_params_2022-12-06.tar",
                           expected=spec["weights_md5"], algorithm="md5", max_bytes=6 * 2**30)
        params = target / "data/params"
        files = [params / f"params_model_{i}{suffix}.npz"
                 for suffix in ("", "_ptm", "_multimer_v3") for i in range(1, 6)]
        if not all(p.is_file() and p.stat().st_size for p in files):
            extract(archive, params, max_bytes=6 * 2**30)
        required = [stereo, *files]
        alphafold_databases(r, target, getattr(r, "alphafold_databases", "none"))
    progress.ui.line(f"  {name} [4/5] Import and CLI checks")
    env = runtime_environment(name, target, python)
    if name == "mol1":
        r.run([python.parent / "mol", "doctor"], extra_env=env)
        executable = python.parent / "mol"
    elif name == "protenix":
        r.run([python.parent / "protenix", "pred", "--help"], extra_env=env)
        executable = python.parent / "protenix"
    else:
        r.run([python, "-c", "import run_alphafold; print('AlphaFold2 imports OK')"], extra_env=env)
        executable = source / "run_alphafold.py"
    if not all(p.is_file() and p.stat().st_size for p in required):
        raise ValueError("A required runtime asset is absent")
    progress.ui.line(f"  {name} [5/5] Installation receipt")
    packages = capture([python, "-c", "import importlib.metadata as m,json; print(json.dumps(sorted((p.metadata['Name'],p.version) for p in m.distributions())))"])
    (target / "packages.installed.json").write_text(packages)
    native = []
    for meta in sorted((target / "bioenv/conda-meta").glob("*.json")):
        entry = read_json(meta, {})
        native.append({key: entry.get(key) for key in ("name", "version", "build", "sha256")})
    atomic_json(target / "packages.native.json", native)
    profile = {"enabled": False, "python": r.relative(python), "root": r.relative(source),
               "data_dir": r.relative(target / "data"), "cache": r.relative(target / "cache"),
               "bio_bin": r.relative(target / "bioenv/bin"), "executable": r.relative(executable),
               "installation": r.relative(record), "timeout_seconds": 3600}
    if name in {"protenix", "mol1"}:
        profile.update(weights=r.relative(weight), model_id=spec["model_id"])
    else:
        profile["weights"] = r.relative(params)
    receipt = {"runtime": name, "verified": False, "verification": "dependencies_and_files",
               "repository": spec["repository"], "revision": revision, "installed_at": time.time(),
               "environment_sha256": digest(target / "packages.installed.json"),
               "native_environment_sha256": digest(target / "packages.native.json"), "profile": profile,
               "required_files": [r.relative(p) for p in required],
               "assets": {r.relative(p): {"bytes": p.stat().st_size, "sha256": digest(p)} for p in required}}
    if (previous.get("verified") and previous.get("revision") == revision
            and previous.get("environment_sha256") == receipt["environment_sha256"]
            and previous.get("native_environment_sha256") == receipt["native_environment_sha256"]
            and previous.get("assets") == receipt["assets"]):
        for key in ("verified", "verification", "evidence", "evidence_sha256", "scope", "verified_at"):
            if key in previous:
                receipt[key] = previous[key]
        profile["enabled"] = previous.get("profile", {}).get("enabled", False)
    if previous:
        atomic_json(target / "installation-history" / f"{time.time_ns()}.json", previous)
    atomic_json(record, receipt)
    r.merge(name, profile)
    return {"status": "ready" if receipt["verified"] else "needs_validation", "verification": receipt["verification"],
            "message": ("Verified native installation reused" if receipt["verified"] else
                        "Native CLI installed; run a bounded prediction before publishing") +
                       ("; sequence databases are separate" if name == "alphafold2" else "")}
