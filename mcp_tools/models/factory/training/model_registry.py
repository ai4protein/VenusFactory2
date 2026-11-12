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
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from mcp_tools.common.common_utils import get_project_root, to_project_relative_path
from agent.rsi.engine.context import get_resources
from mcp_tools.models.identity import trained_metadata
from mcp_tools.models.factory.target_transform import (
    SCHEMA as TARGET_TRANSFORM_SCHEMA,
    checkpoint_transform_path,
    load_checkpoint_transform,
)
from agent.rsi.engine.storage import atomic_json, file_lock

_REGISTRY_ROOT_NAME = "user_trained"
_INDEX_NAME = "registry.json"
_MANIFEST_NAME = "manifest.json"
_CONFIG_COPY_NAME = "training_config.json"


def _project_root() -> Path:
    context = get_resources()
    return context.workspace if context is not None else Path(get_project_root()).resolve()


def registry_root() -> Path:
    context = get_resources()
    root = context.registry if context is not None else _project_root() / "ckpt" / _REGISTRY_ROOT_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def index_path() -> Path:
    return registry_root() / _INDEX_NAME


def _stored_path(path: Path | str) -> str:
    if get_resources() is not None:
        return str(Path(path).resolve().relative_to(_project_root()))
    return to_project_relative_path(path)


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
        if get_resources() is not None:
            raise ValueError("Research registry index is corrupt; refusing to discard history")
        return {"version": 1, "models": []}
    if not isinstance(data, dict):
        if get_resources() is not None:
            raise ValueError("Research registry index must be an object; refusing to discard history")
        return {"version": 1, "models": []}
    models = data.get("models")
    if get_resources() is not None and (
        not isinstance(models, list) or any(not isinstance(m, dict) or not m.get("model_id") for m in models)
    ):
        raise ValueError("Research registry models are malformed; refusing to discard history")
    if not isinstance(models, list):
        data["models"] = []
    data.setdefault("version", 1)
    return data


def _save_index(data: dict[str, Any]) -> None:
    path = index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, data)


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
    context = get_resources()
    if context is not None:
        return [context.workspace]
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


def _register_trained_model_unlocked(
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

    # Check original training state before creating any published artifact.
    # A missing legacy transform stays missing; never refit it during registration.
    try:
        target_transform = load_checkpoint_transform(src_model, config)
    except (ValueError, OSError) as exc:
        return {"success": False, "error": f"Invalid checkpoint target transform: {exc}"}

    base_name = model_id or config.get("dataset_custom") or config.get("output_name") or out_dir.name
    slug = _slugify(str(base_name))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    final_id = _slugify(model_id) if model_id else f"{slug}_{stamp}"

    research = get_resources() is not None
    dest_dir = (registry_root() / ("candidates" if research else "") / final_id).resolve()
    try:
        dest_dir.relative_to(registry_root().resolve())
    except ValueError:
        return {"success": False, "error": "Refusing model_id that escapes ckpt/user_trained"}
    if dest_dir.exists():
        return {"success": False, "error": "Model IDs are immutable; choose a new ID for a new version"}
    # Validate the index before publication, then build the whole artifact away
    # from its final path. Failed copies cannot damage an existing registration.
    index = _load_index()
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    dest_model = dest_dir / src_model.name
    dest_config = dest_dir / _CONFIG_COPY_NAME
    registered_config = dict(config)
    if target_transform is not None:
        registered_config["target_transform_schema"] = TARGET_TRANSFORM_SCHEMA
        registered_config["target_transform"] = target_transform
        registered_config.setdefault("problem_type", "regression")
        registered_config.setdefault("normalize", None if target_transform["method"] == "none" else target_transform["method"])
        registered_config.setdefault("target_unit", target_transform["target_unit"])
        registered_config.setdefault("label_column_name", target_transform["label_column"])
    registered_config["output_dir"] = _stored_path(dest_dir)
    registered_config["output_model_name"] = dest_model.name
    registered_config["model_path"] = _stored_path(dest_model)
    registered_config["registered_model_id"] = final_id

    try:
        with tempfile.TemporaryDirectory(prefix=".register-", dir=dest_dir.parent) as temporary:
            staging = Path(temporary)
            staged_model = staging / src_model.name
            shutil.copy2(src_model, staged_model)
            if target_transform is not None:
                shutil.copy2(checkpoint_transform_path(src_model), checkpoint_transform_path(staged_model))
                # Keep the original receipt bytes; never re-sign a changed copy.
                if load_checkpoint_transform(staged_model, registered_config) != target_transform:
                    raise ValueError("Target transform changed during registration")
            atomic_json(staging / _CONFIG_COPY_NAME, registered_config)
            atomic_json(staged_model.with_suffix(".json"), registered_config)
            manifest = {
                "model_id": final_id,
                "status": "candidate" if research else "registered",
                "independent_validation": False,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "config_path": _stored_path(dest_config),
                "model_path": _stored_path(dest_model),
                "output_dir": _stored_path(dest_dir),
                "plm_model": registered_config.get("plm_model"),
                "training_method": registered_config.get("training_method"),
                "problem_type": registered_config.get("problem_type"),
                "num_labels": registered_config.get("num_labels"),
                "source_config_path": _stored_path(cfg_path),
                "source_output_dir": _stored_path(out_dir),
                "metrics": metrics or {},
                **trained_metadata(registered_config, weights=staged_model),
            }
            atomic_json(staging / _MANIFEST_NAME, manifest)
            os.rename(staging, dest_dir)
    except (ValueError, OSError) as exc:
        return {"success": False, "error": f"Could not stage checkpoint artifact: {exc}"}

    models = [m for m in index.get("models", []) if isinstance(m, dict) and m.get("model_id") != final_id]
    models.insert(0, {
        "model_id": final_id,
        "status": manifest["status"],
        "config_path": manifest["config_path"],
        "model_path": manifest["model_path"],
        "output_dir": manifest["output_dir"],
        "created_at": manifest["created_at"],
        "plm_model": manifest.get("plm_model"),
        "training_method": manifest.get("training_method"),
        "problem_type": manifest.get("problem_type"),
        **{k:manifest[k] for k in ("schema_version","model_type","runtime_id","backbone","training_dataset","training_method","provenance","artifact","version")},
    })
    index["models"] = models
    try:
        _save_index(index)
    except (OSError, ValueError) as exc:
        # A failure after an atomic index replacement may already be committed.
        # Inspect the durable index before deleting this newly created artifact.
        try:
            persisted = json.loads(index_path().read_text()) if index_path().exists() else {"models": []}
            if not isinstance(persisted, dict) or not isinstance(persisted.get("models"), list):
                raise ValueError("Registration index is malformed")
            committed = any(
                item.get("model_id") == final_id and item.get("model_path") == manifest["model_path"]
                for item in persisted.get("models", []) if isinstance(item, dict)
            )
        except (OSError, ValueError):
            return {"success": False, "error": f"Registration index state is unknown: {exc}"}
        if not committed:
            shutil.rmtree(dest_dir)
            return {"success": False, "error": f"Could not publish registration index: {exc}"}

    return {
        "success": True,
        "registered": not research,
        "status": manifest["status"],
        "message": "Research checkpoint staged; independent validation is required before publication." if research else f"Registered trained model '{final_id}' under ckpt/{_REGISTRY_ROOT_NAME}/",
        "model_id": final_id,
        "manifest": manifest,
        "config_path": manifest["config_path"],
        "model_path": manifest["model_path"],
        "registry_index": _stored_path(index_path()),
    }


def register_trained_model(config_path: str, *, model_id: Optional[str] = None,
                           output_dir: Optional[str] = None, model_path: Optional[str] = None,
                           metrics: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    kwargs = dict(model_id=model_id, output_dir=output_dir, model_path=model_path, metrics=metrics)
    with file_lock(registry_root() / ".registry.lock"):
        return _register_trained_model_unlocked(config_path, **kwargs)


def list_trained_models() -> dict[str, Any]:
    index = _load_index()
    # Native checkpoints have no trusted validation adapter yet. In research
    # contexts this agent-writable index cannot publish even a forged status.
    models = [] if get_resources() is not None else [m for m in index.get("models", []) if isinstance(m, dict)]
    return {
        "success": True,
        "count": len(models),
        "models": [{**trained_metadata(m), **m} for m in models],
        "registry_root": _stored_path(registry_root()),
    }


def resolve_registered_config_path(config_path_or_model_id: str) -> str:
    """Resolve a registry model_id or filesystem path to a training config path."""
    raw = (config_path_or_model_id or "").strip()
    if not raw:
        return raw
    from mcp_tools.tools.capability_policy import assert_model_enabled
    for model in _load_index().get('models', []):
        if not isinstance(model,dict): continue
        candidate = Path(model.get('config_path') or '__missing__').expanduser()
        if not candidate.is_absolute(): candidate = _project_root() / candidate
        requested = Path(raw).expanduser()
        if not requested.is_absolute(): requested = _project_root() / requested
        if model.get('model_id') == raw or candidate.resolve() == requested.resolve():
            assert_model_enabled('checkpoint:' + model['model_id'])
    path = Path(raw).expanduser()
    if get_resources() is not None:
        if not path.is_absolute():
            path = _project_root() / path
        if not _is_under_allowed_roots(path):
            raise ValueError("Research config path outside workspace")
    if path.is_file():
        return str(path.resolve())
    # model_id lookup
    index = _load_index()
    for m in index.get("models", []):
        if not isinstance(m, dict):
            continue
        if m.get("model_id") == raw:
            if get_resources() is not None:
                raise ValueError("Research checkpoint is a candidate; independent validation is required")
            cfg = m.get("config_path") or ""
            abs_cfg = Path(cfg)
            if not abs_cfg.is_absolute():
                abs_cfg = _project_root() / cfg
            if get_resources() is not None and not _is_under_allowed_roots(abs_cfg):
                raise ValueError("Research registry points outside workspace")
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
