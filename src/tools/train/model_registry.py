"""Persistent registry for locally trained protein models under ``ckpt/``.

Local training writes checkpoints into ``ckpt/user_trained/<model_id>/`` and
indexes them in ``ckpt/user_trained/registry.json`` so later sessions (Agent MCP
or Custom Model Predict) can reuse the same config + weights.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from web.utils.common_utils import get_project_root, to_project_relative_path

_REGISTRY_ROOT_NAME = "user_trained"
_INDEX_NAME = "registry.json"
_MANIFEST_NAME = "manifest.json"
_CONFIG_COPY_NAME = "training_config.json"


def _project_root() -> Path:
    return Path(get_project_root()).resolve()


def registry_root() -> Path:
    root = _project_root() / "ckpt" / _REGISTRY_ROOT_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def index_path() -> Path:
    return registry_root() / _INDEX_NAME


def _slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", (text or "").strip())
    s = s.strip("._-") or "model"
    return s[:80]


def _load_index() -> dict[str, Any]:
    path = index_path()
    if not path.is_file():
        return {"version": 1, "models": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "models": []}
    if not isinstance(data, dict):
        return {"version": 1, "models": []}
    models = data.get("models")
    if not isinstance(models, list):
        data["models"] = []
    data.setdefault("version", 1)
    return data


def _save_index(data: dict[str, Any]) -> None:
    path = index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_model_file(output_dir: Path, output_model_name: str) -> Optional[Path]:
    candidates = [
        output_dir / output_model_name,
        output_dir / "best_model.pt",
        output_dir / "model.pt",
    ]
    for c in candidates:
        if c.is_file():
            return c
    # Fallback: first .pt/.pth in the directory
    if output_dir.is_dir():
        for p in sorted(output_dir.iterdir()):
            if p.suffix.lower() in {".pt", ".pth"} and p.is_file():
                return p
    return None


def _allowed_source_roots() -> list[Path]:
    root = _project_root()
    return [
        (root / "ckpt").resolve(),
        (root / "temp_outputs").resolve(),
        root.resolve(),
    ]


def _is_under_allowed_roots(path: Path) -> bool:
    resolved = path.resolve()
    if resolved.is_symlink():
        return False
    for root in _allowed_source_roots():
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def register_trained_model(
    config_path: str,
    *,
    model_id: Optional[str] = None,
    output_dir: Optional[str] = None,
    model_path: Optional[str] = None,
    metrics: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Copy/link a trained checkpoint into ``ckpt/user_trained`` and index it."""
    cfg_path = Path(config_path).expanduser().resolve()
    if not cfg_path.is_file():
        return {"success": False, "error": f"Configuration file not found: {config_path}"}
    if not _is_under_allowed_roots(cfg_path):
        return {"success": False, "error": "config_path must be under the project (ckpt/ or temp_outputs/)"}

    try:
        config = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"success": False, "error": f"Failed to read config: {exc}"}

    out_dir = Path(
        output_dir
        or config.get("output_dir")
        or str(cfg_path.parent)
    ).expanduser()
    if not out_dir.is_absolute():
        out_dir = (_project_root() / out_dir).resolve()
    else:
        out_dir = out_dir.resolve()
    if not _is_under_allowed_roots(out_dir):
        return {"success": False, "error": "output_dir must be under the project (ckpt/ or temp_outputs/)"}

    output_model_name = str(config.get("output_model_name") or "best_model.pt")
    src_model = Path(model_path).expanduser().resolve() if model_path else None
    if src_model is None or not src_model.is_file():
        src_model = _resolve_model_file(out_dir, output_model_name)
    if src_model is None or not src_model.is_file():
        return {
            "success": False,
            "error": f"Trained model file not found under {out_dir}",
        }
    if not _is_under_allowed_roots(src_model):
        return {"success": False, "error": "model_path must be under the project (ckpt/ or temp_outputs/)"}

    base_name = model_id or config.get("dataset_custom") or config.get("output_name") or out_dir.name
    slug = _slugify(str(base_name))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    final_id = _slugify(model_id) if model_id else f"{slug}_{stamp}"

    dest_dir = (registry_root() / final_id).resolve()
    try:
        dest_dir.relative_to(registry_root().resolve())
    except ValueError:
        return {"success": False, "error": "Refusing model_id that escapes ckpt/user_trained"}
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_model = dest_dir / Path(src_model.name).name
    if src_model.resolve() != dest_model.resolve():
        shutil.copy2(src_model, dest_model)

    dest_config = dest_dir / _CONFIG_COPY_NAME
    # Point config at the registered model location for future predict calls.
    registered_config = dict(config)
    registered_config["output_dir"] = to_project_relative_path(dest_dir)
    registered_config["output_model_name"] = dest_model.name
    registered_config["model_path"] = to_project_relative_path(dest_model)
    registered_config["registered_model_id"] = final_id
    dest_config.write_text(json.dumps(registered_config, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "model_id": final_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": to_project_relative_path(dest_config),
        "model_path": to_project_relative_path(dest_model),
        "output_dir": to_project_relative_path(dest_dir),
        "plm_model": registered_config.get("plm_model"),
        "training_method": registered_config.get("training_method"),
        "problem_type": registered_config.get("problem_type"),
        "num_labels": registered_config.get("num_labels"),
        "source_config_path": to_project_relative_path(cfg_path),
        "source_output_dir": to_project_relative_path(out_dir),
        "metrics": metrics or {},
    }
    (dest_dir / _MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    index = _load_index()
    models = [m for m in index.get("models", []) if isinstance(m, dict) and m.get("model_id") != final_id]
    models.insert(0, {
        "model_id": final_id,
        "config_path": manifest["config_path"],
        "model_path": manifest["model_path"],
        "output_dir": manifest["output_dir"],
        "created_at": manifest["created_at"],
        "plm_model": manifest.get("plm_model"),
        "training_method": manifest.get("training_method"),
        "problem_type": manifest.get("problem_type"),
    })
    index["models"] = models
    _save_index(index)

    return {
        "success": True,
        "message": f"Registered trained model '{final_id}' under ckpt/{_REGISTRY_ROOT_NAME}/",
        "model_id": final_id,
        "manifest": manifest,
        "config_path": manifest["config_path"],
        "model_path": manifest["model_path"],
        "registry_index": to_project_relative_path(index_path()),
    }


def list_trained_models() -> dict[str, Any]:
    index = _load_index()
    models = [m for m in index.get("models", []) if isinstance(m, dict)]
    return {
        "success": True,
        "count": len(models),
        "models": models,
        "registry_root": to_project_relative_path(registry_root()),
    }


def resolve_registered_config_path(config_path_or_model_id: str) -> str:
    """Resolve a registry model_id or filesystem path to a training config path."""
    raw = (config_path_or_model_id or "").strip()
    if not raw:
        return raw
    path = Path(raw).expanduser()
    if path.is_file():
        return str(path.resolve())
    # model_id lookup
    index = _load_index()
    for m in index.get("models", []):
        if not isinstance(m, dict):
            continue
        if m.get("model_id") == raw:
            cfg = m.get("config_path") or ""
            abs_cfg = Path(cfg)
            if not abs_cfg.is_absolute():
                abs_cfg = _project_root() / cfg
            if abs_cfg.is_file():
                return str(abs_cfg.resolve())
    # directory with training_config.json / manifest
    if path.is_dir():
        for name in (_CONFIG_COPY_NAME, "config.json", _MANIFEST_NAME):
            cand = path / name
            if cand.is_file():
                if name == _MANIFEST_NAME:
                    try:
                        man = json.loads(cand.read_text(encoding="utf-8"))
                        cfg = man.get("config_path")
                        if cfg:
                            return resolve_registered_config_path(cfg)
                    except Exception:
                        pass
                return str(cand.resolve())
    return raw


def register_trained_model_json(**kwargs: Any) -> str:
    return json.dumps(register_trained_model(**kwargs), ensure_ascii=False, indent=2)


def list_trained_models_json() -> str:
    return json.dumps(list_trained_models(), ensure_ascii=False, indent=2)
