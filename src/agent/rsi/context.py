"""Process-scoped resource roots; configure before importing tool modules."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class ResourceContext:
    workspace: Path
    skills: Path
    registry: Path


@lru_cache(maxsize=1)
def get_resources() -> ResourceContext | None:
    raw = os.environ.get("VENUS_RSI_WORKSPACE")
    if not raw:
        return None
    workspace = Path(raw)
    if not workspace.is_absolute() or not workspace.is_dir():
        raise ValueError("VENUS_RSI_WORKSPACE must be an existing absolute directory")
    workspace = workspace.resolve()

    def child(name: str, fallback: str) -> Path:
        path = Path(os.environ.get(name, str(workspace / fallback)))
        if not path.is_absolute():
            raise ValueError(f"{name} must be absolute")
        resolved = path.resolve()
        resolved.relative_to(workspace)
        return resolved

    return ResourceContext(
        workspace, child("VENUS_RSI_SKILLS_ROOT", "skills"), child("VENUS_RSI_REGISTRY_ROOT", "registry")
    )
