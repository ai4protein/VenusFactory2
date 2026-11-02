import subprocess
import sys
import os
import shutil
import argparse
import platform
import urllib.request
import zipfile
import tarfile

import json

from pathlib import Path
with open(Path(__file__).resolve().parents[1] / 'configs' / 'install.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

configs = config['torch_configs']
pip_mirror = config['pip_mirror']




def get_arch_str():
    arch = platform.machine().lower()

    # Map architecture names
    if 'x86_64' in arch or 'amd64' in arch:
        arch_str = 'amd64'
    elif 'arm64' in arch or 'aarch64' in arch:
        arch_str = 'arm64'
    elif 'x86' in arch or 'i386' in arch or 'i686' in arch:
        arch_str = 'x86'
    else:
        arch_str = arch

    return arch_str



def ensure_uv_installed():
    """
    Check if uv exists.
    If not, attempt to automatically install using pip from current Python environment.
    """
    uv_path = shutil.which("uv")

    if uv_path:
        print(f"✅ Detected uv is installed ({uv_path})")
        return

    print("⚠️  uv not found in current environment, automatically installing via pip...")

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "uv", "-i", pip_mirror["index_url"]])
        print("✅ uv installation successful!")
    except subprocess.CalledProcessError:
        print("❌ Automatic uv installation failed.")
        print("Please try manually running: pip install uv")
        sys.exit(1)

def run(cmd, step_name):
    print(f"\n🚀 [Step: {step_name}] Executing...")
    full_cmd = f"uv pip install --system {cmd}"
    try:
        subprocess.check_call(full_cmd, shell=True)
        print(f"✅ [Step: {step_name}] Success!")
    except subprocess.CalledProcessError:
        print(f"❌ [Step: {step_name}] Failed! Please check the error above.")
        sys.exit(1)

def _resolve_type(requested: str) -> str:
    detect_dir = os.path.dirname(os.path.abspath(__file__))
    if detect_dir not in sys.path:
        sys.path.insert(0, detect_dir)
    from torch_detect import resolve_install_type

    resolved, reason = resolve_install_type(requested)
    print(f"🔍 Torch profile: {resolved} ({reason})")
    if resolved not in configs:
        print(f"❌ Unknown torch profile '{resolved}'. Available: {', '.join(configs)}")
        sys.exit(1)
    return resolved


def main():
    parser = argparse.ArgumentParser(description="Automatically configure PyTorch and PyG environment (supports uv acceleration)")
    parser.add_argument(
        "--type",
        choices=["auto", "cpu", "cu128"],
        default="auto",
        help="Install profile: auto-detect GPU/CPU (default), or force cpu / cu128",
    )
    parser.add_argument(
        "--skip-frpc",
        action="store_true",
        help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    install_type = _resolve_type(args.type)


    # Get current selected configuration
    current_config = configs[install_type]
    print(f"\n⚙️  Current installation mode: \033[1;36m{install_type.upper()}\033[0m")
    print(f"   Torch source: {current_config['torch_url']}")
    print(f"   PyG   source: {current_config['torch_aug_url']}")

    # Ensure uv exists ---
    ensure_uv_installed()

    # ==========================================
    # Step 1: Install PyTorch
    # ==========================================
    run(
        f'"torch==2.8.0" torchvision --index-url {current_config["torch_url"]} --prerelease=allow',
        f"Install PyTorch 2.8.0 ({install_type})"
    )

    # ==========================================
    # Step 2: Install PyG dependencies
    # ==========================================
    pyg_packages = "torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv"

    run(
        f"{pyg_packages} -f {current_config['torch_aug_url']}",
        "Install PyG suite (upper-level packages)"
    )

    # ==========================================
    # Step 3: Install pyproject.toml
    # ==========================================
    if os.path.exists("pyproject.toml"):
        print("\n📦 Synchronizing other regular dependencies...")
        index_url = pip_mirror["index_url"]
        run(
            f"--index-url {index_url} -r pyproject.toml",
            "Install regular dependencies"
        )

    print("\n🎉🎉🎉 All dependencies installed! Environment is ready.")

if __name__ == "__main__":
    main()