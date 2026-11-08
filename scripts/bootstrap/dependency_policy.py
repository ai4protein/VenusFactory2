"""Shared platform pins and compiled extension wheel policy."""
from __future__ import annotations

import json
from pathlib import Path
import re


def platform_policy(root: Path, profile: str) -> tuple[dict, list[Path], list[str]]:
    """Validate constraints and the compiled wheel ABI before installing anything."""
    config = json.loads((root / "configs/install.json").read_text(encoding="utf-8"))
    constraints = [root / "requirements" / name for name in
                   ("bootstrap-constraints.txt", "service-security.txt")]
    pins: dict[str, str] = {}
    for path in constraints:
        if not path.is_file() or not path.read_text(encoding="utf-8").strip():
            raise ValueError(f"Missing dependency policy: {path.name}")
        for line in path.read_text(encoding="utf-8").splitlines():
            match = re.fullmatch(r"(torch|torchvision)==([0-9]+(?:\.[0-9]+)+)", line.strip())
            if match:
                name, version = match.groups()
                if name in pins and pins[name] != version:
                    raise ValueError(f"Conflicting dependency policy for {name}")
                pins[name] = version
    if set(pins) != {"torch", "torchvision"}:
        raise ValueError("The platform policy must pin Torch and torchvision")
    selected = dict(config["torch_configs"][profile])
    expected = f"https://data.pyg.org/whl/torch-{pins['torch']}+{profile}.html"
    if selected["torch_aug_url"] != expected:
        raise ValueError("The configured PyG wheel index does not match the pinned Torch ABI")
    selected["index_url"] = config["pip_mirror"]["index_url"]
    return selected, constraints, [f"{name}=={pins[name]}" for name in ("torch", "torchvision")]
