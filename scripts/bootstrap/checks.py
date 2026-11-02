from __future__ import annotations

import json
import urllib.request
from fnmatch import fnmatch
from pathlib import Path

from .common import capture, digest, inside, read_json


def factory_files(root, preset):
    if preset == "none":
        return []
    manifest = read_json(root / "ckpt/manifest.json", {})
    patterns = manifest.get("presets", {}).get(preset)
    if not patterns:
        raise ValueError(f"Unknown Factory preset: {preset}")
    entries = [f for f in manifest["files"] if any(fnmatch(f["path"], p) for p in patterns)]
    if not entries:
        raise ValueError("Factory preset matched no manifest entries")
    for entry in entries:
        inside(root / "ckpt", entry["path"])
    return entries


def factory_health(root, preset, hashes=False):
    files = factory_files(root, preset)
    missing, corrupt = [], []
    for entry in files:
        path = root / "ckpt" / entry["path"]
        if not path.is_file():
            missing.append(entry["path"])
        elif path.stat().st_size != entry["size"] or (hashes and digest(path) != entry["sha256"]):
            corrupt.append(entry["path"])
    return {"status": "failed" if missing or corrupt else "ready", "files": len(files),
            "bytes": sum(f["size"] for f in files), "missing": missing, "corrupt": corrupt,
            "verification": "sha256" if hashes else "size"}


def tool_health(root, name, python, hashes=False):
    config = read_json(root / "configs/model_runtimes.json", {}).get(name, {})
    if not config and name == "proteinmpnn" and (root / "ckpt/ProteinMPNN").is_dir():
        config = {"weights": "ckpt/ProteinMPNN", "python": str(python)}
    if not config:
        return {"status": "missing", "enabled": False}
    missing = []
    for key in ("python", "executable", "root", "weights", "packing_weights", "model_dir", "confidence_model_dir", "structure_weights", "installation"):
        if config.get(key) and not (root / config[key]).exists():
            missing.append(key)
    checks = []
    if name == "proteinmpnn":
        base = root / config.get("weights", "ckpt/ProteinMPNN")
        checks = [base / f"{v}_model_weights/v_48_020.pt" for v in ("vanilla", "soluble", "ca")]
    elif name == "s4pred" and config.get("root"):
        checks = [root / config["root"] / f"weights/weights_{i}.pt" for i in range(1, 6)]
    elif name == "prosst":
        base = root / config.get("structure_weights", "runtime/local/model-runtimes/prosst/structure-tokenizer/weights")
        checks = [base / "AE.pt", base / "2048.joblib"]
    missing += [str(p.relative_to(root)) if p.is_relative_to(root) else p.name for p in checks if not p.is_file() or not p.stat().st_size]
    receipt = read_json(root / config["installation"], {}) if config.get("installation") and (root / config["installation"]).is_file() else {}
    if hashes:
        for path, info in receipt.get("assets", {}).items():
            file = root / path
            if not file.is_file() or digest(file) != info["sha256"]:
                missing.append("asset_hash:" + path)
    executable = root / config["python"] if config.get("python") else python
    imports = {"ligandmpnn": "torch,prody,ml_collections,tree", "antibmpnn": "torch,peptides,pandas",
               "s4pred": "torch,Bio", "proteinmpnn": "torch,numpy", "prosst": "torch,transformers,torch_geometric",
               "esm1v": "torch,transformers", "esmif1": "torch,torch_geometric", "protssn": "torch,transformers,torch_geometric",
               "admet_ai": "admet_ai", "bindcraft": "jax,colabdesign,pyrosetta", "diffdock": "torch,esm,torch_geometric,rdkit"}
    if not missing and name in imports:
        try:
            capture([executable, "-c", "import " + imports[name]], timeout=90)
        except Exception:
            missing.append("dependency_import")
    if name == "interproscan" and not config.get("contact_email"):
        missing.append("contact_email")
    return {"status": "failed" if missing else "installed", "enabled": config.get("enabled", True),
            "verification": receipt.get("verification") or ("previous_inference" if receipt.get("verified") else "dependencies_and_files"),
            "issues": missing}


def doctor(root, python, selected, preset, system, *, hashes=False, urls=(), stages=(), dependencies='platform'):
    results = {"system": system, "tools": {}}
    if "environment" in stages:
        try:
            imports = 'import sys,ssl,sqlite3,venv' if dependencies == 'base' else 'import fastapi,uvicorn,fastmcp,torch,transformers,torch_geometric,peft; import backend; import mcp_tools'
            capture([python, '-c', imports], cwd=root, timeout=90)
            results["environment"] = {"status": "ready", 'dependencies': dependencies}
        except Exception:
            results["environment"] = {"status": "failed", "message": "Platform Python or required imports are missing"}
    if "frontend" in stages:
        results["frontend"] = {"status": "ready" if (root / "frontend/dist/index.html").is_file() else "missing"}
    if "factory" in stages:
        results["factory"] = factory_health(root, preset, hashes)
    if "skills" in stages:
        vendor = root / "src/agent/skills/vendor/scientific-agent-skills"
        count = sum(1 for _ in vendor.rglob("SKILL.md")) if vendor.is_dir() else 0
        results["skills"] = {"status": "ready" if count else "missing", "vendor_skills": count}
    if "tools" in stages:
        for name in selected:
            try:
                results["tools"][name] = tool_health(root, name, python, hashes)
            except Exception as error:
                results["tools"][name] = {"status": "failed", "message": type(error).__name__}
    results["services"] = []
    for url in urls:
        try:
            if not url.startswith(("http://", "https://")):
                raise ValueError("Health URL must use HTTP(S)")
            with urllib.request.urlopen(url, timeout=10) as response:
                body = json.load(response)
                healthy = response.status == 200 and isinstance(body, dict) and body.get("status") not in ("failed", "error", "unhealthy")
            results["services"].append({"url": url, "status": "ready" if healthy else "failed"})
        except Exception:
            results["services"].append({"url": url, "status": "failed"})
    results["sandbox"] = {"status": "available" if system["executables"]["bwrap"] else "missing",
                          "message": "Availability only; host namespaces/resource limits require the sandbox operational tests"}
    return results


def failed(report):
    if isinstance(report, dict):
        return report.get("status") in {"failed", "missing", "blocked", "manual", "needs_validation"} or any(failed(v) for k, v in report.items() if k != "sandbox")
    return isinstance(report, list) and any(failed(v) for v in report)
