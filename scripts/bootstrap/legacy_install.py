"""Compatibility installation entry point with the platform dependency policy."""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .dependency_policy import platform_policy as policy

ROOT = Path(__file__).resolve().parents[2]
UV_VERSION = "0.11.6"
PYG_PACKAGES = ("torch_geometric", "pyg_lib", "torch_scatter", "torch_sparse",
                "torch_cluster", "torch_spline_conv")


def run(command: list[str], label: str) -> None:
    print(label, flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def resolve_profile(requested: str) -> tuple[str, str]:
    spec = importlib.util.spec_from_file_location("venus_torch_detect", ROOT / ".install/torch_detect.py")
    if spec is None or spec.loader is None:
        raise ValueError("Torch profile detection is unavailable")
    detector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(detector)
    return detector.resolve_install_type(requested)


def ensure_uv(index_url: str) -> list[str]:
    executable = shutil.which("uv")
    if executable:
        return [executable]
    run([sys.executable, "-m", "pip", "install", f"uv=={UV_VERSION}",
         "--index-url", index_url], "Preparing the dependency installer")
    # pip may install the command outside PATH; use the interpreter that installed it.
    return [sys.executable, "-m", "uv"]


def main(argv: list[str] | None = None, *, environment: str = "venv") -> int:
    parser = argparse.ArgumentParser(description="Install validated platform dependencies")
    parser.add_argument("--type", choices=("auto", "cpu", "cu128"), default="auto")
    parser.add_argument("--skip-frpc", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if environment not in ("venv", "system"):
            raise ValueError("Environment must be venv or system")
        if environment == "system" and sys.version_info < (3, 12):
            raise ValueError("The current environment requires Python 3.12 or newer; use --env venv")
        profile, reason = resolve_profile(args.type)
        config, constraint_files, torch_packages = policy(ROOT, profile)
        constraints = [value for path in constraint_files for value in ('--constraint', str(path))]
        print(f"Torch profile: {profile} ({reason})", flush=True)
        uv = ensure_uv(config["index_url"])
        if environment == "venv":
            folder = ROOT / ".venv"
            python = folder / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            if not python.is_file():
                if folder.exists() and any(folder.iterdir()):
                    raise ValueError("Existing .venv has no usable Python; inspect it before retrying")
                run([*uv, "venv", "--python", "3.12", str(folder)], "[1/5] Creating the platform environment")
            target = str(python)
        else:
            target = sys.executable
            print("[1/5] Using the explicitly selected current environment", flush=True)
        base = [*uv, "pip", "install", "--python", target]
        if environment == "system":
            base.append("--system")
        run([*base, *torch_packages, "--index-url", config["torch_url"], *constraints],
            "[2/5] Installing the compatible Torch stack")
        run([*base, *PYG_PACKAGES, "--index-url", config["index_url"],
             "--find-links", config["torch_aug_url"], "--only-binary", ":all:", *constraints],
            "[3/5] Installing matching PyG wheels")
        run([*base, "--requirement", str(ROOT / "pyproject.toml"),
             "--index-url", config["index_url"], *constraints],
            "[4/5] Installing platform dependencies")
        run([*uv, "pip", "check", "--python", target], "[5/5] Checking dependencies")
        run([target, "-c",
             "import fastapi,uvicorn,fastmcp,torch,torchvision,torch_geometric,"
             "pyg_lib,torch_scatter,torch_sparse,torch_cluster,torch_spline_conv,transformers,peft"],
            "Checking service and model imports")
    except subprocess.CalledProcessError as exc:
        print(f"Installation stopped: command exited with status {exc.returncode}", file=sys.stderr)
        return exc.returncode if exc.returncode > 0 else 1
    except (OSError, ValueError, KeyError) as exc:
        print(f"Installation stopped: {exc}", file=sys.stderr)
        return 1
    print("Platform dependencies installed and checked.", flush=True)
    return 0
