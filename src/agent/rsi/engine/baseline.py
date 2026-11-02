"""Capture reproducibility metadata without exporting provider credentials."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from .model_policy import read_model_configuration
from .storage import atomic_json, sha256_file


def public_model_config(config_path: Path) -> dict:
    return read_model_configuration(config_path)[1]


def capture_baseline(repo: Path, kimi_bin: Path, config_path: Path, output: Path) -> dict:
    repo = repo.resolve()
    tracked = subprocess.check_output(
        ["git", "ls-files", "src", "scripts/rsi", "configs/rsi"], cwd=repo, text=True
    ).splitlines()
    paths = set(tracked)
    for folder in ("src/agent/rsi", "scripts/rsi", "configs/rsi"):
        paths.update(
            str(p.relative_to(repo))
            for p in (repo / folder).rglob("*")
            if p.is_file() and p.suffix in {".py", ".json", ".toml", ".yaml", ".md"} and "__pycache__" not in p.parts
        )
    hashes = {f: sha256_file(repo / f) for f in sorted(paths) if (repo / f).is_file()}
    packages = {}
    for name in ("httpx", "websockets", "fastmcp", "numpy", "scikit-learn", "biopython", "pytest"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "git_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip(),
        "kimi_version": subprocess.check_output([str(kimi_bin), "--version"], text=True).strip(),
        "kimi_binary_sha256": sha256_file(kimi_bin),
        "model": public_model_config(config_path),
        "python": platform.python_version(),
        "packages": packages,
        "source_hashes": hashes,
        "scope": "engineering pilot, not a preregistered efficacy study",
        "internal_training_llm": "not invoked in this pilot; must pin before native training-tool experiments",
    }
    atomic_json(output, result)
    return result
