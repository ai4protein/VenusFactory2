"""Detect a recommended torch install profile for VenusFactory2.

Returns one of the keys in ``configs/install.json`` → ``torch_configs``
(currently ``cpu`` or ``cu128``).
"""
from __future__ import annotations

import platform
import re
import shutil
import subprocess
from pathlib import Path


def _nvidia_cuda_version() -> float | None:
    """Best-effort CUDA version advertised by the driver (nvidia-smi header)."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        proc = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    match = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)", text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def detect_install_type() -> tuple[str, str]:
    """Return ``(type, reason)`` for the local machine."""
    system = platform.system().lower()
    if system == "darwin":
        return "cpu", "macOS detected → CPU wheels (PyTorch MPS via CPU package)"

    cuda_ver = _nvidia_cuda_version()
    has_gpu = False
    gpu_label = ""

    if shutil.which("nvidia-smi"):
        try:
            proc = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                has_gpu = True
                gpu_label = proc.stdout.strip().splitlines()[0]
        except (OSError, subprocess.SubprocessError):
            pass

    if not has_gpu and Path("/dev/nvidia0").exists():
        has_gpu = True
        gpu_label = "/dev/nvidia0"

    if not has_gpu:
        return "cpu", "No NVIDIA GPU detected → CPU wheels"

    # Match the supported initializer: unknown/older drivers use CPU by default.
    # An explicit cu128 override remains available to an informed operator.
    if cuda_ver is None or cuda_ver < 12.8:
        detail = "unknown" if cuda_ver is None else str(cuda_ver)
        return (
            "cpu",
            f"NVIDIA GPU found but driver CUDA is {detail}; cu128 requires "
            f"CUDA 12.8 or newer for automatic selection. GPU={gpu_label}",
        )

    detail = gpu_label
    if cuda_ver is not None:
        detail = f"{gpu_label}; driver CUDA {cuda_ver}"
    return "cu128", f"NVIDIA GPU → cu128 ({detail})"


def resolve_install_type(requested: str) -> tuple[str, str]:
    """Resolve ``auto|cpu|cu128`` into a concrete type + reason."""
    key = (requested or "auto").strip().lower()
    if key == "auto":
        return detect_install_type()
    if key in ("cpu", "cu128"):
        return key, f"user override → {key}"
    raise ValueError(f"Unsupported torch install type: {requested!r} (use auto|cpu|cu128)")
