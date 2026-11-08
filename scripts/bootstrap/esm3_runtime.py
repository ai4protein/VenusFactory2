"""Optional ESM3 encoder environment; model assets require their own license."""
from __future__ import annotations

import hashlib
from pathlib import Path

from .assets import download
from .common import atomic_json, digest, inside, read_json


def install(r, name):
    target = inside(r.root, r.base / name)
    record = target / "installation.json"
    previous = read_json(record, {})
    existing = read_json(r.config, {}).get(name, {})
    template = read_json(r.root / "configs/model_runtimes.example.json", {}).get(name, {})
    if existing and not previous and existing != template:
        return {"status": "existing", "message": "Administrator-owned ESM3 runtime preserved"}
    requirements = inside(r.root, r.manifest["tools"][name]["requirements"])
    signature = hashlib.sha256((
        digest(requirements) + digest(Path(__file__)) +
        digest(Path(__file__).with_name("system.py")) + r.env.torch_type
    ).encode()).hexdigest()
    target.mkdir(parents=True, exist_ok=True)
    python = r.env.create(target / "env", r.manifest["tools"][name]["python"])
    if read_json(target / "dependencies.json", {}).get("recipe_sha256") != signature:
        r.env.pip(python, "torch==2.11.0", torch=True)
        r.env.pip(python, "-r", requirements)
        r.run([python, "-m", "pip", "check"])
        r.run([python, "-c", "from esm.models.vqvae import StructureTokenEncoder; from esm.utils.structure.protein_chain import ProteinChain"])
        atomic_json(target / "dependencies.json", {"recipe_sha256": signature})
    weights = r.root / existing["weights"] if existing.get("weights") else target / "weights/esm3_structure_encoder_v0.pth"
    profile = {
        "enabled": False, "python": r.relative(python), "weights": r.relative(weights),
        "installation": r.relative(record), "timeout_seconds": 1800,
    }
    verified = False
    required = [python]
    if weights.is_file():
        pdb = download("https://files.rcsb.org/download/1CRN.pdb", target / "smoke/1CRN.pdb", max_bytes=2**20)
        result = target / "smoke/tokens.json"
        from mcp_tools.models.execution import environment
        r.run([python, "-m", "mcp_tools.models.factory.training.data.get_esm3_structure_seq",
               "--runtime-worker", "--pdb_file", pdb, "--weights", weights,
               "--device", "cpu", "--out_file", result],
              cwd=r.root, timeout=300, extra_env={
                  **environment({}), "PYTHONPATH": str(r.root),
                  "CUDA_VISIBLE_DEVICES": "",
              })
        rows = read_json(result, [])
        tokens = rows[0].get("esm3_structure_seq") if len(rows) == 1 else None
        if not isinstance(tokens, list) or len(tokens) != 46 or any(type(x) is not int or not 0 <= x < 4096 for x in tokens):
            raise ValueError("ESM3 did not encode the reference structure")
        required.append(weights)
        verified = True
        profile["enabled"] = True
    receipt = {
        "runtime": name, "verified": verified, "verification": "inference" if verified else "dependencies_only",
        "recipe_sha256": signature, "profile": profile,
        "required_files": [r.relative(p) for p in required],
        "assets": {r.relative(p): {"bytes": p.stat().st_size, "sha256": digest(p)} for p in required},
    }
    atomic_json(record, receipt)
    if existing == template:
        config = read_json(r.config, {})
        config[name] = {**profile, "enabled": existing.get("enabled", False)}
        atomic_json(r.config, config)
    else:
        r.merge(name, profile)
    if not verified:
        return {"status": "needs_validation", "verification": "dependencies_only",
                "message": "ESM3 environment installed. Supply an authorized structure-encoder checkpoint at " +
                r.relative(weights) + ", rerun this installer, then enable the runtime profile."}
    return {"status": "ready", "verification": "inference"}
