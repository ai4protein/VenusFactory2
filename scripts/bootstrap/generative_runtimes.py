"""Pinned optional model environments, isolated from the Web service."""
from __future__ import annotations
import time
import os
import json
from pathlib import Path
from .assets import clone, download
from .common import atomic_json, capture, digest, read_json
from .system import executable

HF_BUNDLE = """
import sys,json
from pathlib import Path
from huggingface_hub import snapshot_download
repo,revision,target,kind=sys.argv[1:]
snapshot_download(repo,revision=revision,local_dir=target,token=False,
                  allow_patterns=['*.safetensors','*.json','*.txt'] + (['pytorch_model*.bin'] if kind == 'esmfold' else []),
                  ignore_patterns=['images/*'])
Path(target,'source.json').write_text(json.dumps({'repository':repo,'revision':revision}))
"""

def checkpoint_files(folder, *, allow_legacy_bin=False):
    """Require all indexed shards, metadata and tokenizer files."""
    weights = list(folder.glob("*.safetensors"))
    if allow_legacy_bin:
        weights += list(folder.glob("pytorch_model*.bin"))
    required = set(weights + [folder / "config.json"])
    required.update(p for p in folder.glob("*.json") if not p.name.endswith(".source.json"))
    required.update(folder.glob("*.txt"))
    if not weights:
        raise ValueError("Model checkpoint download is incomplete")
    for index in folder.glob("*.index.json"):
        data = read_json(index, {})
        for name in data.get("weight_map", {}).values():
            if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
                raise ValueError("Invalid checkpoint shard path")
            required.add(folder / name)
    if not all(p.is_file() and p.stat().st_size for p in required):
        raise ValueError("Model checkpoint shard or metadata is missing")
    return sorted(required)


def install(r, name):
    spec = r.manifest["tools"][name]
    target = r.base / name
    target.mkdir(parents=True, exist_ok=True)
    record = target / "installation.json"
    previous = read_json(record, {})
    existing = read_json(r.config, {}).get(name)
    template = read_json(r.root / "configs/model_runtimes.example.json", {}).get(name)
    if existing and not previous and existing != template:
        return {"status": "existing", "message": "Administrator-owned runtime preserved"}
    source = target / "source"
    revision = clone(r.run, spec["repository"], spec["revision"], source) if spec.get("repository") else None
    uv = executable("uv")
    if name in {"rfantibody", "rfd4_proteina"} and not uv:
        from .system import Environment
        uv = Environment(r.root, r.run, manager="uv", torch_type=r.env.torch_type).ensure_manager()[0]
    if name == "rfantibody":
        r.run([uv, "sync", "--frozen", "--no-dev", "--project", source])
        python = source / ".venv/bin/python"
    else:
        python = r.env.create(target / "env", spec["python"])
        if name == "rfd4_proteina":
            r.run([uv, "pip", "install", "--python", python, "--no-sources",
                   "--torch-backend", "cu126", str(source) + "[all-cuda12]"])
        elif name == "esmfold2":
            r.env.pip(python, str(source))
        else:
            r.env.pip(python, "torch==2.8.0", torch=True)
            r.env.pip(python, *spec["packages"])
    profile = {"enabled": False, "python": r.relative(python),
               "installation": r.relative(record), "timeout_seconds": 1800}
    if spec.get("repository"):
        profile["root"] = r.relative(source)
    if name == "rfantibody":
        required = []
        for filename in ("RFdiffusion_Ab.pt", "ProteinMPNN_v48_noise_0.2.pt", "RF2_ab.pt"):
            path = source / "weights" / filename
            download("https://files.ipd.uw.edu/pub/RFantibody/" + filename, path)
            required.append(path)
        profile["weights"] = r.relative(source / "weights")
    else:
        weights = target / "weights"
        r.run([r.python, "-c", HF_BUNDLE, spec["weight_repository"], spec["weight_revision"], weights, name])
        required = checkpoint_files(weights, allow_legacy_bin=name == "esmfold")
        profile.update(weights=r.relative(weights), model_id=spec["weight_repository"])
    if name == "esmfold2":
        ccd = target / "ccd/ccd.pkl"
        endpoint = os.getenv("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
        download(endpoint + "/" + spec["ccd_repository"] + "/resolve/" + spec["ccd_revision"] + "/ccd.pkl",
                 ccd, expected=spec["ccd_sha256"])
        required.append(ccd)
        profile["ccd"] = r.relative(ccd)
    if not all(p.is_file() and p.stat().st_size for p in required):
        raise ValueError("Required runtime assets are missing or empty")
    imports = {"rfantibody": "torch,rfantibody", "rfd4_proteina": "torch,rfproteina",
               "esmfold2": "torch,esm.models.esmfold2", "esmfold": "torch,transformers,biotite"}
    r.run([python, "-c", "import " + imports[name]])
    packages = capture([python, "-c", "import importlib.metadata as m,json; print(json.dumps(sorted((p.metadata['Name'],p.version) for p in m.distributions())))"])
    (target / "packages.installed.json").write_text(packages)
    receipt = {"runtime": name, "environment_sha256": digest(target / "packages.installed.json"), "repository": spec.get("repository"), "revision": revision,
               "weight_repository": spec.get("weight_repository"), "weight_revision": spec.get("weight_revision"),
               "verified": False, "verification": "dependencies_and_files", "profile": profile,
               "required_files": [r.relative(p) for p in required],
               "assets": {r.relative(p): {"bytes": p.stat().st_size, "sha256": digest(p)} for p in required},
               "installed_at": time.time()}
    if previous.get("verified") and previous.get("revision") == revision and previous.get("assets") == receipt["assets"] and previous.get("environment_sha256") == receipt["environment_sha256"]:
        receipt.update(verified=True, verification=previous["verification"])
        for key in ("evidence", "evidence_sha256", "scope", "verified_at"):
            if key in previous:
                receipt[key] = previous[key]
        profile["enabled"] = previous["profile"].get("enabled", False)
    if previous:
        # Preserve evidence from every prior install, including failed validation.
        atomic_json(target / "installation-history" / f"{time.time_ns()}.json", previous)
    atomic_json(record, receipt)
    r.merge(name, profile)
    return {"status": "ready" if receipt["verified"] else "needs_validation",
            "verification": receipt["verification"]}
