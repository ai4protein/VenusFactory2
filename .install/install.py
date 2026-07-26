import subprocess
import sys
import os
import shutil
import argparse
import platform
import urllib.request
import zipfile
import tarfile

VENV_PATH = ".venv"

import json
with open('./install_config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

configs = config['torch_configs']
FRPC_VERSIONS = config['frpc_versions']
pip_mirror = config['pip_mirror']

def download_frpc(system, arch_str, version="v0.3"):
    """
    Download the frpc executable for the corresponding system.

    Prefer the shared helper (CDN → Hugging Face fallback). Keep a small
    inline CDN attempt so this script still works if ``src/`` is unavailable.
    """
    # Preferred path: src/frpc_hub.py (CDN + HF assets/frpc fallback).
    try:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        src_dir = os.path.join(repo_root, "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from frpc_hub import download_frpc as hub_download_frpc

        path = hub_download_frpc(system=system, arch=arch_str, version=version, force=True)
        print(f"✅ Download complete via frpc_hub: {path}")
        return os.path.basename(path)
    except Exception as hub_exc:
        print(f"⚠️  frpc_hub download failed ({hub_exc}); trying CDN only...")

    url_key = f"{system}_{arch_str}"
    
    if url_key not in FRPC_VERSIONS[version]:
        print(f"❌ Unsupported system architecture: {system}_{arch_str}")
        return None
    
    url = FRPC_VERSIONS[version][url_key]
    
    if system == "windows":
        download_filename = f"frpc_{system}_{arch_str}_{version}.exe"
        base_filename = f"frpc_{system}_{arch_str}_{version}"
    else:
        download_filename = f"frpc_{system}_{arch_str}_{version}"
        base_filename = download_filename
    
    print(f"📥 Downloading frpc: {download_filename}")
    print(f"   URL: {url}")
    
    try:
        urllib.request.urlretrieve(url, download_filename)
        print(f"✅ Download complete: {download_filename}")
        
        if system != "windows":
            os.chmod(download_filename, 0o755)
            print(f"✅ Executable permissions set")
        if system == "windows":
            if os.path.exists(base_filename):
                os.remove(base_filename)
            # Rename file
            os.rename(download_filename, base_filename)
            print(f"✅ Renamed to: {base_filename}")
            return base_filename
        
        return download_filename
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("   Tip: python scripts/download_frpc.py --to-gradio-cache")
        return None


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

def setup_frpc():
    """
    Copy frpc to the correct cache directory based on operating system
    If not available locally, attempt to download
    """
    print("\n🔧 Setting up Gradio frpc...")
    
    system = platform.system().lower()
    
    arch_str = get_arch_str()
    print(f"🔍 Architecture: {arch_str}")

    frpc_file = None
    version = "v0.3"
    
    for file in os.listdir('.'):
        if file.startswith('frpc_') and system in file and arch_str in file:
            frpc_file = file
            if frpc_file.endswith(".exe"):
                base_filename = frpc_file[:-4]
                shutil.copy(frpc_file, base_filename)
                frpc_file = base_filename
            print(f"📦 Found local frpc file: {frpc_file}")
            break
    
    # If not found locally, attempt to download
    if not frpc_file:
        print("📭 No local frpc file found, attempting download...")
        frpc_file = download_frpc(system, arch_str, version)
    
    if not frpc_file:
        print("⚠️  frpc setup failed, Gradio will download automatically when needed")
        return
    
    # Determine target directory
    cache_dir = get_frpc_cache_dir()
    if not cache_dir:
        print("❌ Unable to determine cache directory")
        return
    
    os.makedirs(cache_dir, exist_ok=True)
    
    dest_file = os.path.join(cache_dir, frpc_file)
    
    try:
        if os.path.exists(dest_file):
            os.remove(dest_file)
        
        shutil.copy2(frpc_file, dest_file)
        
        if system != 'windows':
            os.chmod(dest_file, 0o755)
            print(f"✅ frpc copied to: {dest_file} (executable permissions set)")
        else:
            print(f"✅ frpc copied to: {dest_file}")
            
        if os.path.exists(dest_file):
            file_size = os.path.getsize(dest_file)
            print(f"📊 File size: {file_size} bytes")
            
    except Exception as e:
        print(f"❌ Failed to copy frpc: {e}")
        print("Gradio will download frpc automatically when needed")

def get_frpc_cache_dir():
    """
    Get the frpc cache directory for the corresponding system
    """
    system = platform.system().lower()
    
    if system == 'windows':
        path = os.path.join(os.environ.get('USERPROFILE', ''), '.cache', 'huggingface', 'gradio', 'frpc')
        if not os.path.exists(path):
            os.makedirs(path)
        return path
    else:
        home = os.path.expanduser('~')
        return os.path.join(home, '.cache', 'huggingface', 'gradio', 'frpc')
    
    return None

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
    full_cmd = f"uv pip install -p {VENV_PATH} {cmd}"
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
        help="Skip frpc setup"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force redownload frpc even if it exists locally"
    )
    args = parser.parse_args()
    install_type = _resolve_type(args.type)
    
    if not args.skip_frpc:
        if args.force_download:
            print("🔄 Forcing frpc redownload...")
            system = platform.system().lower()
            arch_str = get_arch_str()
            for file in os.listdir('.'):
                if file.startswith(f"frpc_{system}_{arch_str}"):
                    os.remove(file)
                    print(f"🗑️  Deleted old file: {file}")
        
        setup_frpc()
    
    # Get current selected configuration
    current_config = configs[install_type]
    print(f"\n⚙️  Current installation mode: \033[1;36m{install_type.upper()}\033[0m")
    print(f"   Torch source: {current_config['torch_url']}")
    print(f"   PyG   source: {current_config['torch_aug_url']}")

    # Ensure uv exists ---
    ensure_uv_installed()
    
    # Ensure virtual environment exists
    if not os.path.exists(VENV_PATH):
        print("📦 Creating virtual environment...")
        subprocess.check_call(f"uv venv --python 3.12 {VENV_PATH}", shell=True)

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