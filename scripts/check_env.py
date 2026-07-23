#!/usr/bin/env python
"""Quick readiness check for a VenusFactory environment.

Run from project root:
    python scripts/check_env.py

Exit code is non-zero if any required check fails.
"""

from __future__ import annotations

import importlib
import os
import platform
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RESET = "\033[0m"


def _fmt_status(ok: bool, optional: bool = False) -> str:
    if ok:
        return f"{GREEN}OK{RESET}"
    return f"{YELLOW}MISS (optional){RESET}" if optional else f"{RED}FAIL{RESET}"


def check_import(modname: str, attrs: tuple[str, ...] = (), optional: bool = False):
    try:
        mod = importlib.import_module(modname)
    except Exception as exc:
        return False, optional, f"{type(exc).__name__}: {exc}"
    info_bits = []
    for a in attrs:
        try:
            info_bits.append(f"{a}={getattr(mod, a)}")
        except Exception:
            pass
    version = getattr(mod, "__version__", None)
    if version:
        info_bits.insert(0, f"v{version}")
    return True, optional, ", ".join(info_bits) if info_bits else "imported"


GROUPS: dict[str, list[tuple[str, tuple[str, ...], bool]]] = {
    "Core ML": [
        ("torch", ("__version__",), False),
        ("torchvision", ("__version__",), False),
        ("numpy", ("__version__",), False),
        ("scipy", ("__version__",), False),
        ("pandas", ("__version__",), False),
        ("sklearn", ("__version__",), False),
    ],
    "PyG suite": [
        ("torch_geometric", ("__version__",), False),
        ("torch_scatter", (), False),
        ("torch_sparse", (), False),
        ("torch_cluster", (), False),
        ("torch_spline_conv", (), False),
        ("pyg_lib", (), False),
    ],
    "Transformers / PEFT": [
        ("transformers", ("__version__",), False),
        ("peft", ("__version__",), False),
        ("accelerate", ("__version__",), False),
        ("datasets", ("__version__",), False),
        ("tokenizers", ("__version__",), False),
        ("safetensors", ("__version__",), False),
        ("sentencepiece", (), False),
        ("torchmetrics", ("__version__",), False),
        ("bitsandbytes", ("__version__",), True),  # optional in conda env
    ],
    "Protein stack": [
        ("esm", ("__version__",), False),
        ("Bio", ("__version__",), False),
        ("biotite", ("__version__",), False),
        ("vplm", (), False),
        ("py3Dmol", (), False),
    ],
    # ProSST / VenusREM structure tokenization (SSTPredictor)
    "Mutation / ProSST": [
        ("pathos", ("__version__",), False),
        ("joblib", ("__version__",), False),
        ("tqdm", ("__version__",), False),
    ],
    "Web / API": [
        ("gradio", ("__version__",), False),
        ("fastapi", ("__version__",), False),
        ("uvicorn", ("__version__",), False),
        ("starlette", ("__version__",), False),
        ("httpx", ("__version__",), False),
    ],
    "Agent stack": [
        ("langchain", ("__version__",), False),
        ("langgraph", (), False),
        ("langchain_core", (), False),
        ("langchain_openai", (), False),
        ("openai", ("__version__",), False),
        ("anthropic", ("__version__",), False),
        ("mcp", (), False),
        ("fastmcp", (), False),
    ],
    "Reporting": [
        ("xhtml2pdf", (), False),
        ("reportlab", ("Version",), False),
        ("svglib", (), False),
        ("cairo", ("version",), True),  # pycairo (PyPI) imports as `cairo`
    ],
    "Project src": [
        ("config", (), False),
        ("agent.models", (), False),
        ("logger", (), False),
        ("exceptions", (), False),
    ],
}


def cuda_report():
    try:
        import torch
    except Exception:
        return None
    info = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if info["cuda_available"]:
        info["device_0"] = torch.cuda.get_device_name(0)
        try:
            t = torch.randn(1024, 1024, device="cuda")
            torch.matmul(t, t)
            info["matmul_smoke"] = "OK"
        except Exception as exc:
            info["matmul_smoke"] = f"FAIL: {exc}"
    return info


def main() -> int:
    print(f"{CYAN}{'=' * 70}{RESET}")
    print(f"{CYAN}VenusFactory environment readiness check{RESET}")
    print(f"{CYAN}{'=' * 70}{RESET}")
    print(f"Python   : {sys.version.split()[0]}  ({sys.executable})")
    print(f"Platform : {platform.platform()}")
    print(f"CWD      : {os.getcwd()}")
    print(f"Project  : {ROOT}")
    print()

    fail_count = 0
    optional_miss = 0
    total = 0

    for group, items in GROUPS.items():
        print(f"{CYAN}[{group}]{RESET}")
        for modname, attrs, optional in items:
            total += 1
            ok, opt, info = check_import(modname, attrs, optional)
            status = _fmt_status(ok, opt)
            print(f"  {status:<28} {modname:<28} {info}")
            if not ok:
                if opt:
                    optional_miss += 1
                else:
                    fail_count += 1
        print()

    cu = cuda_report()
    if cu is not None:
        print(f"{CYAN}[CUDA / GPU]{RESET}")
        for k, v in cu.items():
            print(f"  {k:<20} {v}")
        print()

    print(f"{CYAN}{'-' * 70}{RESET}")
    summary = f"Checks: {total}  Required FAIL: {fail_count}  Optional missing: {optional_miss}"
    color = GREEN if fail_count == 0 else RED
    print(f"{color}{summary}{RESET}")
    if cu and not cu.get("cuda_available"):
        print(f"{YELLOW}Note: CUDA not available — GPU features will be CPU-only.{RESET}")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
