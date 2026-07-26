"""Gradio share ``frpc`` binary helper.

The binary is **not** stored in git. Download sources (in order):

1. Official Gradio CDN URLs from ``install_config.json``
2. Hugging Face fallback: ``AI4Protein/VenusFactory2-ckpts`` → ``assets/frpc/``

Environment:
  VENUS_FRPC_VERSION     default ``v0.3``
  VENUS_FRPC_REPO_ID     HF fallback repo (default VenusFactory2-ckpts)
  VENUS_FRPC_HF_PREFIX   path prefix inside the HF repo (default ``assets/frpc``)
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import stat
import urllib.request
from pathlib import Path
from typing import Any

try:
    from logger import get_logger
except ImportError:  # pragma: no cover
    import logging

    def get_logger(name: str):  # type: ignore[misc]
        return logging.getLogger(name)

try:
    from hub_progress import announce, format_bytes, urllib_reporthook
except ImportError:  # pragma: no cover
    from src.hub_progress import announce, format_bytes, urllib_reporthook


logger = get_logger("frpc_hub")

DEFAULT_VERSION = "v0.3"
DEFAULT_HF_REPO = "AI4Protein/VenusFactory2-ckpts"
DEFAULT_HF_PREFIX = "assets/frpc"

_REPO_ROOT = Path(__file__).resolve().parent.parent


def repo_root() -> Path:
    return _REPO_ROOT


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def arch_str() -> str:
    arch = platform.machine().lower()
    if "x86_64" in arch or "amd64" in arch:
        return "amd64"
    if "arm64" in arch or "aarch64" in arch:
        return "arm64"
    if "x86" in arch or "i386" in arch or "i686" in arch:
        return "x86"
    return arch


def system_name() -> str:
    return platform.system().lower()


def frpc_filename(system: str | None = None, arch: str | None = None, version: str | None = None) -> str:
    system = system or system_name()
    arch = arch or arch_str()
    version = version or (_env("VENUS_FRPC_VERSION", DEFAULT_VERSION) or DEFAULT_VERSION)
    if system == "windows":
        return f"frpc_{system}_{arch}_{version}.exe"
    return f"frpc_{system}_{arch}_{version}"


def load_install_config() -> dict[str, Any]:
    path = repo_root() / "install_config.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read install_config.json: %s", exc)
        return {}


def cdn_url(system: str | None = None, arch: str | None = None, version: str | None = None) -> str | None:
    system = system or system_name()
    arch = arch or arch_str()
    version = version or (_env("VENUS_FRPC_VERSION", DEFAULT_VERSION) or DEFAULT_VERSION)
    cfg = load_install_config().get("frpc_versions", {})
    mapping = cfg.get(version) or {}
    return mapping.get(f"{system}_{arch}")


def hf_repo_id() -> str:
    return _env("VENUS_FRPC_REPO_ID", DEFAULT_HF_REPO) or DEFAULT_HF_REPO


def hf_prefix() -> str:
    return (_env("VENUS_FRPC_HF_PREFIX", DEFAULT_HF_PREFIX) or DEFAULT_HF_PREFIX).strip("/")


def gradio_cache_dir() -> Path:
    if system_name() == "windows":
        base = Path(os.environ.get("USERPROFILE", str(Path.home())))
    else:
        base = Path.home()
    return base / ".cache" / "huggingface" / "gradio" / "frpc"


def local_project_path(filename: str | None = None) -> Path:
    return repo_root() / (filename or frpc_filename())


def _is_ready(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _make_executable(path: Path) -> None:
    if system_name() == "windows":
        return
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _download_cdn(url: str, dest: Path) -> Path:
    announce(f"[frpc] Downloading from CDN: {url}", log=logger)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    try:
        urllib.request.urlretrieve(url, tmp, reporthook=urllib_reporthook("[frpc] CDN"))
        tmp.replace(dest)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)  # type: ignore[arg-type]
    if not _is_ready(dest):
        raise FileNotFoundError(f"CDN download produced empty file: {dest}")
    _make_executable(dest)
    announce(f"[frpc] CDN download complete ({format_bytes(dest.stat().st_size)})", log=logger)
    return dest


def _download_hf(filename: str, dest: Path) -> Path:
    from huggingface_hub import hf_hub_download

    remote = f"{hf_prefix()}/{filename}"
    announce(f"[frpc] Downloading from Hugging Face {hf_repo_id()}:{remote}", log=logger)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Use hub cache, then copy to the requested destination (keeps repo root clean).
    cached = Path(
        hf_hub_download(
            repo_id=hf_repo_id(),
            filename=remote,
            repo_type="model",
        )
    )
    if not _is_ready(cached):
        raise FileNotFoundError(f"HF download missing for {remote}")
    if cached.resolve() != dest.resolve():
        shutil.copy2(cached, dest)
    _make_executable(dest)
    if not _is_ready(dest):
        raise FileNotFoundError(f"HF download produced empty file: {dest}")
    announce(f"[frpc] HF download complete ({format_bytes(dest.stat().st_size)})", log=logger)
    return dest


def download_frpc(
    *,
    dest_dir: str | Path | None = None,
    force: bool = False,
    system: str | None = None,
    arch: str | None = None,
    version: str | None = None,
) -> Path:
    """Download frpc into ``dest_dir`` (default: repo root). Returns local path."""
    filename = frpc_filename(system=system, arch=arch, version=version)
    out_dir = Path(dest_dir) if dest_dir is not None else repo_root()
    dest = out_dir / filename
    if not force and _is_ready(dest):
        announce(f"[frpc] Using local binary: {dest}", log=logger)
        return dest

    announce(f"[frpc] Preparing {filename} → {dest}", log=logger)
    errors: list[str] = []
    url = cdn_url(system=system, arch=arch, version=version)
    if url:
        try:
            return _download_cdn(url, dest)
        except Exception as exc:
            errors.append(f"CDN: {exc}")
            announce(f"[frpc] CDN failed, trying Hugging Face fallback... ({exc})", log=logger)

    try:
        return _download_hf(filename, dest)
    except Exception as exc:
        errors.append(f"HF: {exc}")
        raise FileNotFoundError(
            "Failed to download frpc from CDN and Hugging Face. "
            + " | ".join(errors)
            + f". Manual: python scripts/download_frpc.py"
        ) from exc


def install_to_gradio_cache(
    *,
    force: bool = False,
    system: str | None = None,
    arch: str | None = None,
    version: str | None = None,
) -> Path:
    """Ensure frpc exists in Gradio's expected cache directory."""
    filename = frpc_filename(system=system, arch=arch, version=version)
    cache = gradio_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / filename
    if not force and _is_ready(dest):
        return dest

    # Prefer project-local binary if present; else download into repo root then copy.
    local = local_project_path(filename)
    if force or not _is_ready(local):
        local = download_frpc(dest_dir=repo_root(), force=force, system=system, arch=arch, version=version)
    shutil.copy2(local, dest)
    _make_executable(dest)
    return dest


def ensure_frpc(*, force: bool = False) -> Path:
    """Ensure project-local frpc + Gradio cache copy are available."""
    local = download_frpc(force=force)
    cache_copy = install_to_gradio_cache(force=force)
    logger.info("frpc ready: local=%s cache=%s", local, cache_copy)
    return local
