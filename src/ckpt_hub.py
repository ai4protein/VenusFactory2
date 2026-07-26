"""Hugging Face checkpoint hub for VenusFactory2.

Remote (default): https://huggingface.co/AI4Protein/VenusFactory2-ckpts
Local root (default): <repo>/ckpt

Environment:
  VENUS_CKPT_REPO_ID          HF model repo id (default: AI4Protein/VenusFactory2-ckpts)
  VENUS_CKPT_DIR              Local cache directory (default: <repo>/ckpt)
  VENUS_CKPT_REVISION         HF revision / tag / commit (default: main)
  VENUS_CKPT_AUTO_DOWNLOAD    1/true to auto-fetch missing weights (default: 1)
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Iterable, Sequence

try:
    from logger import get_logger
except ImportError:  # pragma: no cover - direct script execution edge case
    import logging

    def get_logger(name: str):  # type: ignore[misc]
        return logging.getLogger(name)


logger = get_logger("ckpt_hub")

DEFAULT_REPO_ID = "AI4Protein/VenusFactory2-ckpts"
DEFAULT_REVISION = "main"

# Task folders that ship finetuned adapters (mirrors current ckpt/ layout).
FINETUNE_TASKS: tuple[str, ...] = (
    "DeepET_Topt",
    "DeepLocBinary",
    "DeepLocMulti",
    "DeepSol",
    "DeepSoluE",
    "DLKcat",
    "EpHod",
    "MetalIonBinding",
    "ProtSolM",
    "SortingSignal",
    "Thermostability",
    "VenusVaccine_BacteriaBinary",
    "VenusVaccine_TumorBinary",
    "VenusVaccine_VirusBinary",
    "VenusX_Res_Act_MP90",
    "VenusX_Res_BindI_MP90",
    "VenusX_Res_Evo_MP90",
    "VenusX_Res_Motif_MP90",
)

# Named download presets → HF allow_patterns
PRESETS: dict[str, tuple[str, ...]] = {
    "demo": ("demo/**",),
    "predict-core": (
        "demo/**",
        *tuple(f"{task}/ankh-large/**" for task in FINETUNE_TASKS),
    ),
    "predict-all": (
        "demo/**",
        *tuple(f"{task}/**" for task in FINETUNE_TASKS),
    ),
    "proteinmpnn": ("ProteinMPNN/**",),
    "all": ("**",),
}

_REPO_ROOT = Path(__file__).resolve().parent.parent
_path_locks: dict[str, threading.Lock] = {}
_path_locks_guard = threading.Lock()


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def repo_root() -> Path:
    return _REPO_ROOT


def ckpt_repo_id() -> str:
    return _env("VENUS_CKPT_REPO_ID", DEFAULT_REPO_ID) or DEFAULT_REPO_ID


def ckpt_revision() -> str:
    return _env("VENUS_CKPT_REVISION", DEFAULT_REVISION) or DEFAULT_REVISION


def auto_download_enabled() -> bool:
    if _env_bool("HF_HUB_OFFLINE", False) or _env_bool("TRANSFORMERS_OFFLINE", False):
        return False
    return _env_bool("VENUS_CKPT_AUTO_DOWNLOAD", True)


def ckpt_root() -> Path:
    """Local checkpoint root. Relative paths resolve against the repo root."""
    raw = _env("VENUS_CKPT_DIR", "ckpt") or "ckpt"
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = repo_root() / path
    return path.resolve()


def manifest_path() -> Path:
    return ckpt_root() / "manifest.json"


def bundled_manifest_path() -> Path:
    """Manifest shipped in git (may exist even when weights are absent)."""
    return repo_root() / "ckpt" / "manifest.json"


def load_manifest() -> dict:
    for candidate in (manifest_path(), bundled_manifest_path()):
        if candidate.is_file():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Failed to read manifest %s: %s", candidate, exc)
    return {
        "repo_id": ckpt_repo_id(),
        "version": 1,
        "presets": {name: list(patterns) for name, patterns in PRESETS.items()},
        "files": [],
    }


def list_presets() -> list[str]:
    manifest = load_manifest()
    preset_names = list((manifest.get("presets") or {}).keys())
    if preset_names:
        return sorted(set(preset_names) | set(PRESETS))
    return sorted(PRESETS)


def preset_patterns(preset: str) -> list[str]:
    key = preset.strip().lower()
    manifest = load_manifest()
    from_manifest = (manifest.get("presets") or {}).get(key)
    if from_manifest:
        return list(from_manifest)
    if key in PRESETS:
        return list(PRESETS[key])
    raise ValueError(
        f"Unknown ckpt preset {preset!r}. Choose one of: {', '.join(list_presets())}"
    )


def normalize_rel_path(path: str | Path) -> str:
    """Normalize to a repo-relative path inside the HF/local ckpt tree (no 'ckpt/' prefix)."""
    raw = Path(str(path).strip())
    root = ckpt_root()

    if raw.is_absolute():
        try:
            rel = raw.resolve().relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"Path {path!r} is outside ckpt root {root}. "
                "Auto-download only applies to files under VENUS_CKPT_DIR."
            ) from exc
        return rel.as_posix()

    parts = raw.as_posix().lstrip("./")
    if parts == "ckpt" or parts.startswith("ckpt/"):
        parts = parts[5:].lstrip("/")
    return parts


def local_path(rel_path: str | Path) -> Path:
    rel = normalize_rel_path(rel_path)
    return (ckpt_root() / rel).resolve()


def _lock_for(key: str) -> threading.Lock:
    with _path_locks_guard:
        lock = _path_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _path_locks[key] = lock
        return lock


def _is_nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _dir_ready(path: Path, *, require_json: bool = False) -> bool:
    if not path.is_dir():
        return False
    try:
        pts = list(path.glob("*.pt"))
        if not pts:
            return False
        if require_json and not list(path.glob("*.json")):
            return False
        return any(p.stat().st_size > 0 for p in pts)
    except OSError:
        return False


def _missing_hint(rel: str) -> str:
    return (
        f"Missing checkpoint '{rel}' under {ckpt_root()}. "
        f"Download with: python scripts/download_ckpts.py --include '{rel}' "
        f"(repo={ckpt_repo_id()}, revision={ckpt_revision()}), "
        "or set VENUS_CKPT_AUTO_DOWNLOAD=1."
    )


def _import_hub():
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub is required for checkpoint download. "
            "Install with: pip install huggingface-hub"
        ) from exc
    return hf_hub_download, snapshot_download


def download_file(rel_path: str | Path, *, force: bool = False) -> Path:
    """Download a single file into the local ckpt root. Returns the local path."""
    rel = normalize_rel_path(rel_path)
    dest = local_path(rel)
    if not force and _is_nonempty_file(dest):
        return dest

    hf_hub_download, _ = _import_hub()
    ckpt_root().mkdir(parents=True, exist_ok=True)
    logger.info(
        "Downloading ckpt file %s from %s@%s",
        rel,
        ckpt_repo_id(),
        ckpt_revision(),
    )
    hf_hub_download(
        repo_id=ckpt_repo_id(),
        filename=rel,
        repo_type="model",
        revision=ckpt_revision(),
        local_dir=str(ckpt_root()),
        force_download=force,
    )
    if not _is_nonempty_file(dest):
        raise FileNotFoundError(
            f"Downloaded '{rel}' but local file is missing/empty at {dest}. "
            f"Check that it exists in {ckpt_repo_id()}."
        )
    return dest


def download_patterns(
    patterns: Sequence[str],
    *,
    force: bool = False,
) -> Path:
    """Download files matching HF allow_patterns into the local ckpt root."""
    cleaned = [p.strip() for p in patterns if p and p.strip()]
    if not cleaned:
        raise ValueError("patterns must be a non-empty sequence of glob strings")

    _, snapshot_download = _import_hub()
    ckpt_root().mkdir(parents=True, exist_ok=True)
    logger.info(
        "Downloading ckpt patterns %s from %s@%s",
        cleaned,
        ckpt_repo_id(),
        ckpt_revision(),
    )
    snapshot_download(
        repo_id=ckpt_repo_id(),
        repo_type="model",
        revision=ckpt_revision(),
        local_dir=str(ckpt_root()),
        allow_patterns=list(cleaned),
        force_download=force,
    )
    return ckpt_root()


def download_preset(preset: str, *, force: bool = False) -> Path:
    return download_patterns(preset_patterns(preset), force=force)


def ensure_ckpt_file(path: str | Path, *, download: bool | None = None) -> Path:
    """Ensure a checkpoint *file* exists locally; download from HF when missing."""
    rel = normalize_rel_path(path)
    dest = local_path(rel)
    should_download = auto_download_enabled() if download is None else download

    with _lock_for(f"file:{rel}"):
        if _is_nonempty_file(dest):
            return dest
        if not should_download:
            raise FileNotFoundError(_missing_hint(rel))
        return download_file(rel)


def ensure_ckpt_dir(
    path: str | Path,
    *,
    download: bool | None = None,
    require_json: bool = False,
) -> Path:
    """Ensure a checkpoint *directory* exists with weights; download prefix when missing."""
    rel = normalize_rel_path(path)
    dest = local_path(rel)
    should_download = auto_download_enabled() if download is None else download
    pattern = f"{rel.rstrip('/')}/**"

    with _lock_for(f"dir:{rel}"):
        if _dir_ready(dest, require_json=require_json):
            return dest
        if not should_download:
            raise FileNotFoundError(_missing_hint(rel))
        download_patterns([pattern])
        if not _dir_ready(dest, require_json=require_json):
            raise FileNotFoundError(
                f"Downloaded prefix '{rel}' but directory is still incomplete at {dest}. "
                f"Verify files exist under {ckpt_repo_id()}/{rel}/."
            )
        return dest


def ensure_ckpt_path(path: str | Path, *, download: bool | None = None) -> Path:
    """Ensure a ckpt file or directory exists.

    - If the path already exists as a file/dir, return it (after download fill-in when incomplete).
    - If the path has a suffix like ``.pt``/``.json``, treat it as a file.
    - Otherwise treat it as an adapter/weights directory.
    """
    raw = Path(str(path))
    # Absolute/local hit first (custom models outside the hub tree).
    if raw.exists():
        if raw.is_file():
            return raw.resolve()
        if raw.is_dir() and (_dir_ready(raw, require_json=False) or any(raw.iterdir())):
            # Existing custom/user dirs: don't force hub layout checks.
            if _dir_ready(raw) or not str(raw.resolve()).startswith(str(ckpt_root())):
                return raw.resolve()

    rel = normalize_rel_path(path)
    suffix = Path(rel).suffix.lower()
    if suffix in {".pt", ".json", ".pth", ".bin", ".safetensors", ".ckpt"}:
        return ensure_ckpt_file(rel, download=download)

    # Finetuned adapters ship .pt + .json; ProteinMPNN dirs are .pt-only.
    require_json = not rel.startswith("ProteinMPNN")
    try:
        return ensure_ckpt_dir(rel, download=download, require_json=require_json)
    except ValueError:
        # Path outside ckpt root and missing — re-raise as FileNotFoundError.
        raise FileNotFoundError(f"Checkpoint path not found: {path}") from None


def ensure_proteinmpnn_weights(
    *,
    model_name: str = "v_48_020",
    variant: str = "vanilla",
    download: bool | None = None,
) -> Path:
    """Ensure a ProteinMPNN weight file exists; return its local path."""
    variant = variant.strip().lower()
    folder = {
        "vanilla": "ProteinMPNN/vanilla_model_weights",
        "soluble": "ProteinMPNN/soluble_model_weights",
        "ca": "ProteinMPNN/ca_model_weights",
    }.get(variant)
    if folder is None:
        raise ValueError("variant must be one of: vanilla, soluble, ca")
    rel = f"{folder}/{model_name}.pt"
    try:
        return ensure_ckpt_file(rel, download=download)
    except FileNotFoundError:
        # Some layouts also keep copies directly under ProteinMPNN/.
        fallback = f"ProteinMPNN/{model_name}.pt"
        return ensure_ckpt_file(fallback, download=download)


def iter_local_weight_files(root: Path | None = None) -> Iterable[Path]:
    base = root or ckpt_root()
    if not base.is_dir():
        return []
    files: list[Path] = []
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        if path.name in {"manifest.json", "README.md", ".gitkeep"}:
            continue
        if path.suffix.lower() in {".pt", ".json", ".pth", ".bin", ".safetensors", ".joblib"}:
            files.append(path)
    return files
