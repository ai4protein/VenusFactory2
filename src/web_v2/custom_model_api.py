import json
import math
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import load_dataset
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from web.utils.command import preview_command, preview_eval_command, preview_predict_command
from web.utils.common_utils import (
    build_web_v2_download_url,
    build_run_id_utc,
    create_run_manifest,
    ensure_within_roots,
    get_web_v2_area_dir,
    get_web_v2_root_dir,
    make_web_v2_upload_name,
    redact_path_text,
    resolve_web_v2_client_path,
    to_project_relative_path,
    to_web_v2_public_path,
)


router = APIRouter(prefix="/api/custom-model", tags=["custom-model-v2"])

_CONSTANT_PATH = Path(__file__).resolve().parent.parent / "constant.json"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CUSTOM_MODEL_UPLOAD_DIR = get_web_v2_area_dir("uploads", tool="custom_model")
_CUSTOM_MODEL_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
_WEB_V2_ROOT = get_web_v2_root_dir().resolve()
_WEB_V2_RESULTS_ROOT = get_web_v2_area_dir("results")
_ALLOWED_FILE_ROOTS = [
    _PROJECT_ROOT.resolve(),
    (_PROJECT_ROOT / "ckpt").resolve(),
    _WEB_V2_ROOT,
]
_DEBUG_LOG_PATH = os.getenv("CUSTOM_MODEL_DEBUG_LOG_PATH", "").strip()


def _debug_log(hypothesis_id: str, location: str, message: str, data: Dict[str, Any], run_id: str = "pre-fix") -> None:
    if not _DEBUG_LOG_PATH:
        return
    payload = {
        "sessionId": "029b38",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        log_path = Path(_DEBUG_LOG_PATH).expanduser().resolve()
        if not ensure_within_roots(log_path, _ALLOWED_FILE_ROOTS):
            return
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass


def _to_project_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(_PROJECT_ROOT.resolve()).as_posix()
    except Exception:
        return path.name


def _resolve_project_path(path_value: str) -> Path:
    p = Path(path_value).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (_PROJECT_ROOT / p).resolve()


def _assert_allowed_path(path: Path, *, detail: str = "Access denied.") -> Path:
    resolved = path.resolve()
    if not ensure_within_roots(resolved, _ALLOWED_FILE_ROOTS):
        raise HTTPException(status_code=403, detail=detail)
    return resolved


def _resolve_client_file_path(path_str: str, *, allow_ckpt: bool = True) -> Path:
    raw = str(path_str or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="File path is required.")
    try:
        resolved = resolve_web_v2_client_path(raw, allowed_areas=("uploads", "results", "work", "sessions", "manifests"))
        return _assert_allowed_path(resolved)
    except ValueError:
        pass
    local_path = _resolve_project_path(raw)
    if local_path.exists():
        if not allow_ckpt and ensure_within_roots(local_path, [(_PROJECT_ROOT / "ckpt").resolve()]):
            raise HTTPException(status_code=403, detail="Access denied.")
        return _assert_allowed_path(local_path)
    raise HTTPException(status_code=404, detail="File not found.")


def _safe_read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Invalid JSON file: {to_project_relative_path(path)}") from exc


def _scan_folders_under(root: Path, max_depth: int = 5) -> List[str]:
    if not root or not root.is_dir():
        return []
    _assert_allowed_path(root, detail="Root path is not allowed.")
    result: List[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_dir():
            continue
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        if not rel.parts or len(rel.parts) > max_depth:
            continue
        result.append(_to_project_rel(path))
    return result


def _scan_models_in_folder(folder_path: str) -> List[Dict[str, str]]:
    resolved_folder = _resolve_project_path(folder_path) if folder_path else None
    if not resolved_folder or not resolved_folder.is_dir():
        return []
    _assert_allowed_path(resolved_folder)
    folder = resolved_folder
    models: List[Dict[str, str]] = []
    for pt_file in sorted(folder.rglob("*.pt")):
        stem = pt_file.stem
        if any(s in stem for s in ("_lora", "_qlora", "_dora", "_adalora", "_ia3")):
            continue
        models.append(
            {
                "label": str(pt_file.relative_to(folder)),
                "path": _to_project_rel(pt_file),
            }
        )
    return models


def _read_tabular_df(file_path: str, sample_n: int = 3) -> pd.DataFrame:
    path = _resolve_client_file_path(file_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    ext = path.suffix.lower()
    try:
        if ext == ".csv":
            return pd.read_csv(path, nrows=sample_n)
        if ext == ".tsv":
            return pd.read_csv(path, sep="\t", nrows=sample_n)
        if ext in {".xlsx", ".xls"}:
            return pd.read_excel(path, nrows=sample_n)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read file {file_path}: {exc}") from exc
    raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Only csv/tsv/xlsx/xls are supported.")


def _read_tabular_columns(file_path: str) -> List[str]:
    path = _resolve_client_file_path(file_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    ext = path.suffix.lower()
    try:
        if ext == ".csv":
            return list(pd.read_csv(path, nrows=0).columns)
        if ext == ".tsv":
            return list(pd.read_csv(path, sep="\t", nrows=0).columns)
        if ext in {".xlsx", ".xls"}:
            return list(pd.read_excel(path, nrows=0).columns)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse headers from {file_path}: {exc}") from exc
    raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Only csv/tsv/xlsx/xls are supported.")


def _load_model_config(model_path: str) -> Dict[str, Any]:
    if not model_path:
        return {}
    model = _resolve_client_file_path(model_path, allow_ckpt=True)
    cfg_path = model.parent / f"{model.stem}.json"
    if not cfg_path.exists():
        return {}
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _sse(event: str, data: Dict[str, Any]) -> str:
    def _sanitize(value: Any) -> Any:
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, list):
            return [_sanitize(v) for v in value]
        if isinstance(value, dict):
            return {str(k): _sanitize(v) for k, v in value.items()}
        return value

    safe_data = _sanitize(data)
    return f"event: {event}\ndata: {json.dumps(safe_data, ensure_ascii=True, allow_nan=False)}\n\n"


@dataclass
class TrainingLiveState:
    current_epoch: int = -1
    epochs: List[int] = field(default_factory=list)
    train_loss: List[Optional[float]] = field(default_factory=list)
    val_loss: List[Optional[float]] = field(default_factory=list)
    val_metrics: Dict[str, List[Optional[float]]] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)
    output_dir: str = ""
    csv_path: str = ""


def _extract_log_content(line: str) -> str:
    match = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} - [A-Za-z]+ - INFO - (.*)", line)
    return match.group(1).strip() if match else line.strip()


def _ensure_epoch_slot(state: TrainingLiveState, epoch: int) -> int:
    if epoch not in state.epochs:
        state.epochs.append(epoch)
        state.train_loss.append(None)
        state.val_loss.append(None)
        for metric in state.val_metrics.values():
            metric.append(None)
    idx = state.epochs.index(epoch)
    for metric in state.val_metrics.values():
        while len(metric) < len(state.epochs):
            metric.append(None)
    return idx


def _write_test_metrics_csv(output_dir: str, metrics: Dict[str, float]) -> str:
    try:
        run_id = build_run_id_utc()
        out_dir = get_web_v2_area_dir("results", tool="custom_model", run_id=run_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "test_results_metrics.csv"
        lines = ["Metric,Value"]
        for key, value in sorted(metrics.items()):
            lines.append(f"{key},{value:.6f}")
        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(csv_path)
    except Exception:
        return ""


def _parse_training_line(line: str, state: TrainingLiveState) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    content = _extract_log_content(line)
    num_pattern = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|nan|inf|-inf)"

    train_match = re.search(rf"Epoch (\d+) Train Loss: {num_pattern}", content, re.IGNORECASE)
    if train_match:
        epoch = int(train_match.group(1))
        loss = float(train_match.group(2))
        idx = _ensure_epoch_slot(state, epoch)
        state.train_loss[idx] = loss
        events.append(
            {
                "event": "training_metrics",
                "data": {
                    "epochs": state.epochs,
                    "train_loss": state.train_loss,
                    "val_loss": state.val_loss,
                    "val_metrics": state.val_metrics,
                },
            }
        )
        return events

    val_loss_match = re.search(rf"Epoch (\d+) Val Loss: {num_pattern}", content, re.IGNORECASE)
    if val_loss_match:
        epoch = int(val_loss_match.group(1))
        loss = float(val_loss_match.group(2))
        idx = _ensure_epoch_slot(state, epoch)
        state.val_loss[idx] = loss
        events.append(
            {
                "event": "training_metrics",
                "data": {
                    "epochs": state.epochs,
                    "train_loss": state.train_loss,
                    "val_loss": state.val_loss,
                    "val_metrics": state.val_metrics,
                },
            }
        )
        return events

    val_metric_match = re.search(rf"Epoch (\d+) Val ([A-Za-z0-9_\-]+): {num_pattern}", content, re.IGNORECASE)
    if val_metric_match:
        epoch = int(val_metric_match.group(1))
        metric_name = val_metric_match.group(2).strip().lower()
        metric_value = float(val_metric_match.group(3))
        idx = _ensure_epoch_slot(state, epoch)
        if metric_name not in state.val_metrics:
            state.val_metrics[metric_name] = [None] * len(state.epochs)
        while len(state.val_metrics[metric_name]) < len(state.epochs):
            state.val_metrics[metric_name].append(None)
        state.val_metrics[metric_name][idx] = metric_value
        events.append(
            {
                "event": "training_metrics",
                "data": {
                    "epochs": state.epochs,
                    "train_loss": state.train_loss,
                    "val_loss": state.val_loss,
                    "val_metrics": state.val_metrics,
                },
            }
        )
        return events

    test_loss_match = re.search(rf"Test Loss:\s*{num_pattern}", content, re.IGNORECASE)
    if test_loss_match:
        state.test_metrics["loss"] = float(test_loss_match.group(1))
        state.csv_path = _write_test_metrics_csv(state.output_dir, state.test_metrics)
        events.append(
            {
                "event": "test_results",
                "data": {
                    "metrics": state.test_metrics,
                    "csv_download_url": build_web_v2_download_url(to_web_v2_public_path(state.csv_path)) if state.csv_path else "",
                },
            }
        )
        return events

    test_metric_match = re.search(rf"Test ([A-Za-z0-9_\-]+):\s*{num_pattern}", content, re.IGNORECASE)
    if test_metric_match:
        metric_name = test_metric_match.group(1).strip().lower()
        if metric_name != "loss":
            state.test_metrics[metric_name] = float(test_metric_match.group(2))
            state.csv_path = _write_test_metrics_csv(state.output_dir, state.test_metrics)
            events.append(
                {
                    "event": "test_results",
                    "data": {
                        "metrics": state.test_metrics,
                        "csv_download_url": build_web_v2_download_url(to_web_v2_public_path(state.csv_path)) if state.csv_path else "",
                    },
                }
            )
    return events


def _normalized_output_model_name(raw_name: str) -> str:
    name = str(raw_name or "").strip()
    if not name:
        name = "best_model.pt"
    if not name.lower().endswith(".pt"):
        name = f"{name}.pt"
    return name


_STRUCTURE_MODELS = ("prosst", "protssn", "saprot")
_SES_STRUCTURE_COLUMNS = {"foldseek_seq", "ss8_seq"}


def _is_ses_adapter(method: str) -> bool:
    m = str(method or "").strip().lower()
    return m in {"ses-adapter", "ses_adapter"}


def _is_structure_plm(plm_display: str, plm_model: str) -> bool:
    hint = f"{plm_display} {plm_model}".lower()
    return any(key in hint for key in _STRUCTURE_MODELS)


def _is_residue_level_problem(problem_type: str) -> bool:
    return str(problem_type or "").strip().lower().startswith("residue_")


def _normalize_structure_seq_list(raw: Any) -> List[str]:
    if isinstance(raw, list):
        values = [str(x).strip() for x in raw]
    else:
        values = [x.strip() for x in str(raw or "").split(",")]
    return [x for x in values if x in _SES_STRUCTURE_COLUMNS]


def _infer_dataset_columns(dataset_ref: str, split_hint: str = "train") -> List[str]:
    ref = str(dataset_ref or "").strip()
    if not ref:
        return []
    ref_path = Path(ref).expanduser()
    if ref_path.suffix.lower() in {".csv", ".tsv", ".xlsx", ".xls"}:
        return [str(x) for x in _read_tabular_columns(ref)]
    dataset = load_dataset(ref)
    if isinstance(dataset, dict):
        if split_hint in dataset:
            split_data = dataset[split_hint]
        else:
            first_split = next(iter(dataset.keys()), None)
            if not first_split:
                return []
            split_data = dataset[first_split]
    else:
        split_data = dataset
    columns = getattr(split_data, "column_names", None) or []
    return [str(x) for x in columns]


def _assert_ses_adapter_source(
    *,
    structure_seq_list: List[str],
    pdb_dir: str,
    sources: List[Dict[str, str]],
    context: str,
) -> None:
    requested = [x for x in structure_seq_list if x in _SES_STRUCTURE_COLUMNS]
    if not requested:
        raise HTTPException(
            status_code=400,
            detail=f"{context}: ses-adapter requires structure_seq with foldseek_seq and/or ss8_seq.",
        )
    if pdb_dir:
        return
    missing_by_source: List[str] = []
    for src in sources:
        src_name = src.get("name", "dataset")
        src_type = src.get("type", "")
        src_value = str(src.get("value", "")).strip()
        if not src_value:
            continue
        try:
            if src_type == "file":
                columns = set(str(x) for x in _read_tabular_columns(src_value))
            else:
                columns = set(_infer_dataset_columns(src_value, split_hint=src.get("split", "train")))
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{context}: unable to verify structure columns for {src_name}. "
                    f"Provide PDB Folder or a dataset with required columns. ({exc})"
                ),
            ) from exc
        miss = [c for c in requested if c not in columns]
        if miss:
            missing_by_source.append(f"{src_name} missing {', '.join(miss)}")
    if missing_by_source:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{context}: ses-adapter requires selected structure columns in dataset "
                f"or a PDB Folder. " + "; ".join(missing_by_source)
            ),
        )


def _validate_structure_requirements(
    *,
    context: str,
    plm_display: str,
    plm_model: str,
    method: str,
    problem_type: str,
    structure_seq_list: List[str],
    pdb_dir: str,
    sources: List[Dict[str, str]],
) -> None:
    is_structure_model = _is_structure_plm(plm_display, plm_model)
    is_ses = _is_ses_adapter(method)
    if is_structure_model and not pdb_dir:
        raise HTTPException(status_code=400, detail=f"{context}: PDB Folder is required for ProSST/ProtSSN/SaProt.")
    if context == "Training" and is_ses and _is_residue_level_problem(problem_type):
        raise HTTPException(status_code=400, detail=f"{context}: ses-adapter is not supported for residue-level training tasks.")
    if is_ses:
        _assert_ses_adapter_source(
            structure_seq_list=structure_seq_list,
            pdb_dir=pdb_dir,
            sources=sources,
            context=context,
        )


def _normalize_training_args(payload: Dict[str, Any], constant: Dict[str, Any]) -> Dict[str, Any]:
    plm_models = constant.get("plm_models", {})
    dataset_configs = constant.get("dataset_configs", {})

    plm_display = payload.get("plm_model", "")
    if plm_display not in plm_models:
        raise HTTPException(status_code=400, detail=f"Unknown plm_model: {plm_display}")

    dataset_selection = payload.get("dataset_selection", "Custom")
    output_dir = str(payload.get("output_dir", "demo"))
    if output_dir and not output_dir.startswith("ckpt") and not os.path.isabs(output_dir):
        output_dir = f"ckpt/{output_dir}"

    args: Dict[str, Any] = {
        "plm_model": plm_models[plm_display],
        "training_method": payload.get("training_method", "full"),
        "pooling_method": payload.get("pooling_method", "mean"),
        "learning_rate": payload.get("learning_rate", 5e-4),
        "num_epochs": payload.get("num_epochs", 20),
        "max_seq_len": payload.get("max_seq_len", 1024),
        "gradient_accumulation_steps": payload.get("gradient_accumulation_steps", 1),
        "warmup_steps": payload.get("warmup_steps", 0),
        "scheduler": payload.get("scheduler", "linear"),
        "output_model_name": _normalized_output_model_name(payload.get("output_model_name", "best_model.pt")),
        "output_dir": output_dir,
        "patience": payload.get("patience", 10),
        "num_workers": payload.get("num_workers", 4),
        "max_grad_norm": payload.get("max_grad_norm", -1),
        "monitor": payload.get("monitored_metrics", "accuracy"),
        "monitor_strategy": payload.get("monitored_strategy", "max"),
    }

    if dataset_selection == "Pre-defined":
        dataset_key = payload.get("dataset_config")
        if dataset_key not in dataset_configs:
            raise HTTPException(status_code=400, detail=f"Unknown dataset_config: {dataset_key}")
        cfg_path = dataset_configs[dataset_key]
        cfg = _safe_read_json(Path(cfg_path))
        args["dataset_config"] = cfg_path
        args["sequence_column_name"] = cfg.get("sequence_column_name", "aa_seq")
        args["label_column_name"] = cfg.get("label_column_name", "label")
    else:
        metrics = payload.get("metrics", [])
        if isinstance(metrics, list):
            metrics = ",".join(metrics)
        dataset_custom = str(payload.get("dataset_custom", "")).strip()
        train_file = str(payload.get("train_file", "")).strip()
        valid_file = str(payload.get("valid_file", "")).strip()
        test_file = str(payload.get("test_file", "")).strip()
        if train_file:
            train_file = str(_resolve_client_file_path(train_file))
        if valid_file:
            valid_file = str(_resolve_client_file_path(valid_file))
        if test_file:
            test_file = str(_resolve_client_file_path(test_file))
        effective_dataset = dataset_custom or train_file or valid_file or test_file
        args.update(
            {
                "dataset": effective_dataset,
                "problem_type": payload.get("problem_type", "single_label_classification"),
                "num_labels": payload.get("num_labels", 2),
                "metrics": metrics or "accuracy,mcc,f1,precision,recall,auroc",
                "sequence_column_name": payload.get("sequence_column_name", "aa_seq"),
                "label_column_name": payload.get("label_column_name", "label"),
            }
        )
        if train_file:
            args["train_file"] = train_file
        if valid_file:
            args["valid_file"] = valid_file
        if test_file:
            args["test_file"] = test_file

    batch_mode = payload.get("batch_mode", "Batch Size Mode")
    if batch_mode == "Batch Size Mode":
        args["batch_size"] = payload.get("batch_size", 8)
    else:
        args["batch_token"] = payload.get("batch_token", 4000)

    if payload.get("wandb_enabled"):
        args["wandb"] = True
        if payload.get("wandb_project"):
            args["wandb_project"] = payload["wandb_project"]
        if payload.get("wandb_entity"):
            args["wandb_entity"] = payload["wandb_entity"]

    method = args.get("training_method")
    if method in ["plm-lora", "plm-qlora", "plm-adalora", "plm-dora", "plm-ia3"]:
        lora_modules = payload.get("lora_target_modules", "")
        if isinstance(lora_modules, str):
            lora_modules = [m.strip() for m in lora_modules.split(",") if m.strip()]
        args["lora_r"] = payload.get("lora_r", 8)
        args["lora_alpha"] = payload.get("lora_alpha", 32)
        args["lora_dropout"] = payload.get("lora_dropout", 0.1)
        args["lora_target_modules"] = lora_modules

    structure_seq = payload.get("structure_seq", [])
    structure_seq_list = _normalize_structure_seq_list(structure_seq)
    if method == "ses-adapter" and structure_seq_list:
        args["structure_seq"] = ",".join(structure_seq_list)
    pdb_dir = str(payload.get("pdb_dir", "")).strip()
    if pdb_dir:
        args["pdb_dir"] = pdb_dir

    sources: List[Dict[str, str]] = []
    if dataset_selection == "Custom":
        if args.get("train_file"):
            sources.append({"name": "train_file", "type": "file", "value": str(args.get("train_file", ""))})
        if args.get("valid_file"):
            sources.append({"name": "valid_file", "type": "file", "value": str(args.get("valid_file", ""))})
        if args.get("test_file"):
            sources.append({"name": "test_file", "type": "file", "value": str(args.get("test_file", ""))})
        dataset_custom = str(args.get("dataset", "")).strip()
        if dataset_custom and not sources:
            sources.append({"name": "dataset_custom", "type": "dataset", "value": dataset_custom, "split": "train"})
    _validate_structure_requirements(
        context="Training",
        plm_display=plm_display,
        plm_model=plm_models[plm_display],
        method=str(args.get("training_method", "")),
        problem_type=str(args.get("problem_type", "")),
        structure_seq_list=structure_seq_list,
        pdb_dir=pdb_dir,
        sources=sources,
    )

    # region agent log
    _debug_log(
        "H1",
        "src/web_v2/custom_model_api.py:_normalize_training_args",
        "normalized training args snapshot",
        {
            "dataset_selection": dataset_selection,
            "payload_dataset_custom": str(payload.get("dataset_custom", "")).strip(),
            "final_dataset": args.get("dataset"),
            "has_train_file": bool(args.get("train_file")),
            "has_valid_file": bool(args.get("valid_file")),
            "has_test_file": bool(args.get("test_file")),
            "training_method": args.get("training_method"),
        },
    )
    # endregion
    return args


def _normalize_evaluation_args(payload: Dict[str, Any], constant: Dict[str, Any]) -> Dict[str, Any]:
    plm_models = constant.get("plm_models", {})
    dataset_configs = constant.get("dataset_configs", {})
    plm_display = payload.get("plm_model", "")
    if plm_display not in plm_models:
        raise HTTPException(status_code=400, detail=f"Unknown plm_model: {plm_display}")

    args: Dict[str, Any] = {
        "plm_model": plm_models[plm_display],
        "model_path": payload.get("model_path", ""),
        "eval_method": payload.get("eval_method", "full"),
        "pooling_method": payload.get("pooling_method", "mean"),
        "sequence_column_name": payload.get("sequence_column_name", "aa_seq"),
        "label_column_name": payload.get("label_column_name", "label"),
    }

    if payload.get("dataset_selection", "Custom") == "Custom":
        dataset_custom = str(payload.get("dataset_custom", "")).strip()
        test_file = str(payload.get("test_file", "")).strip()
        if test_file:
            test_file = str(_resolve_client_file_path(test_file))
        effective_test_file = test_file or dataset_custom
        if not effective_test_file:
            raise HTTPException(status_code=400, detail="Custom evaluation requires test_file or dataset_custom")
        metrics = payload.get("metrics", [])
        if isinstance(metrics, list):
            metrics = ",".join(metrics)
        args.update(
            {
                "dataset": dataset_custom or effective_test_file,
                "test_file": effective_test_file,
                "problem_type": payload.get("problem_type", "single_label_classification"),
                "num_labels": payload.get("num_labels", 2),
                "metrics": metrics or "accuracy,mcc,f1,precision,recall,auroc",
            }
        )
    else:
        dataset_key = payload.get("dataset_config")
        if dataset_key not in dataset_configs:
            raise HTTPException(status_code=400, detail=f"Unknown dataset_config: {dataset_key}")
        cfg = _safe_read_json(Path(dataset_configs[dataset_key]))
        dataset_name = str(cfg.get("dataset", "")).strip()
        test_file = str(cfg.get("test_file", "")).strip() or dataset_name
        if not test_file:
            raise HTTPException(status_code=400, detail=f"Dataset config {dataset_key} missing dataset/test_file")

        cfg_metrics = cfg.get("metrics", "")
        if isinstance(cfg_metrics, list):
            cfg_metrics = ",".join([str(x).strip() for x in cfg_metrics if str(x).strip()])

        args.update(
            {
                "dataset": dataset_name or test_file,
                "test_file": test_file,
                "problem_type": cfg.get("problem_type", "single_label_classification"),
                "num_labels": cfg.get("num_labels", 2),
                "metrics": cfg_metrics or "accuracy,mcc,f1,precision,recall,auroc",
                "sequence_column_name": cfg.get("sequence_column_name", args.get("sequence_column_name", "aa_seq")),
                "label_column_name": cfg.get("label_column_name", args.get("label_column_name", "label")),
                "split": cfg.get("split", payload.get("split", "")),
            }
        )

    if payload.get("batch_mode", "Batch Size Mode") == "Batch Size Mode":
        args["batch_size"] = payload.get("batch_size", 1)
    else:
        args["batch_token"] = payload.get("batch_token", 2000)

    structure_seq = payload.get("structure_seq", [])
    structure_seq_list = _normalize_structure_seq_list(structure_seq)
    if args["eval_method"] == "ses-adapter" and structure_seq_list:
        args["structure_seq"] = ",".join(structure_seq_list)
        if "foldseek_seq" in structure_seq_list:
            args["use_foldseek"] = True
        if "ss8_seq" in structure_seq_list:
            args["use_ss8"] = True

    pdb_dir = str(payload.get("pdb_dir", "")).strip()
    if pdb_dir:
        args["pdb_dir"] = pdb_dir
    sources: List[Dict[str, str]] = []
    if payload.get("dataset_selection", "Custom") == "Custom":
        if args.get("test_file"):
            sources.append({"name": "test_file", "type": "file", "value": str(args.get("test_file", ""))})
        dataset_custom = str(payload.get("dataset_custom", "")).strip()
        if dataset_custom and not sources:
            sources.append({"name": "dataset_custom", "type": "dataset", "value": dataset_custom, "split": "test"})
    _validate_structure_requirements(
        context="Evaluation",
        plm_display=plm_display,
        plm_model=plm_models[plm_display],
        method=str(args.get("eval_method", "")),
        problem_type=str(args.get("problem_type", "")),
        structure_seq_list=structure_seq_list,
        pdb_dir=pdb_dir,
        sources=sources,
    )
    return args


def _normalize_predict_args(payload: Dict[str, Any], constant: Dict[str, Any]) -> Dict[str, Any]:
    plm_models = constant.get("plm_models", {})
    plm_display = payload.get("plm_model", "")
    if plm_display not in plm_models:
        raise HTTPException(status_code=400, detail=f"Unknown plm_model: {plm_display}")

    args: Dict[str, Any] = {
        "plm_model": plm_models[plm_display],
        "model_path": payload.get("model_path", ""),
        "eval_method": payload.get("eval_method", "full"),
        "pooling_method": payload.get("pooling_method", "mean"),
        "problem_type": payload.get("problem_type", "single_label_classification"),
        "num_labels": payload.get("num_labels", 2),
    }

    if payload.get("prediction_mode", "single") == "single":
        args["aa_seq"] = payload.get("aa_seq", "")
    else:
        input_file = str(payload.get("input_file", "")).strip()
        args["input_file"] = str(_resolve_client_file_path(input_file)) if input_file else ""
        if payload.get("batch_size") is not None:
            args["batch_size"] = payload.get("batch_size")

    structure_seq = payload.get("structure_seq", [])
    structure_seq_list = _normalize_structure_seq_list(structure_seq)
    if args["eval_method"] == "ses-adapter" and structure_seq_list:
        args["structure_seq"] = ",".join(structure_seq_list)
        if "foldseek_seq" in structure_seq_list:
            args["use_foldseek"] = True
        if "ss8_seq" in structure_seq_list:
            args["use_ss8"] = True

    if payload.get("foldseek_seq"):
        args["foldseek_seq"] = payload["foldseek_seq"]
    if payload.get("ss8_seq"):
        args["ss8_seq"] = payload["ss8_seq"]
    pdb_dir = str(payload.get("pdb_dir", "")).strip()
    if pdb_dir:
        args["pdb_dir"] = pdb_dir

    sources: List[Dict[str, str]] = []
    prediction_mode = str(payload.get("prediction_mode", "single"))
    if prediction_mode == "batch":
        input_file = str(args.get("input_file", "")).strip()
        if input_file:
            sources.append({"name": "input_file", "type": "file", "value": input_file})
    _validate_structure_requirements(
        context="Predict",
        plm_display=plm_display,
        plm_model=plm_models[plm_display],
        method=str(args.get("eval_method", "")),
        problem_type=str(args.get("problem_type", "")),
        structure_seq_list=structure_seq_list,
        pdb_dir=pdb_dir,
        sources=sources,
    )
    if _is_ses_adapter(str(args.get("eval_method", ""))) and prediction_mode == "single":
        if "foldseek_seq" in structure_seq_list and not str(args.get("foldseek_seq", "")).strip() and not pdb_dir:
            raise HTTPException(status_code=400, detail="Predict: single mode ses-adapter with foldseek_seq requires Foldseek Sequence or PDB Folder.")
        if "ss8_seq" in structure_seq_list and not str(args.get("ss8_seq", "")).strip() and not pdb_dir:
            raise HTTPException(status_code=400, detail="Predict: single mode ses-adapter with ss8_seq requires SS8 Sequence or PDB Folder.")

    return args


def _build_exec_cmd(script: str, args: Dict[str, Any]) -> List[str]:
    # Use unbuffered Python output so SSE can stream epoch logs immediately.
    cmd = [sys.executable, "-u", script]
    for key, value in args.items():
        if value is None or value == "":
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        elif isinstance(value, list):
            if value:
                cmd.append(f"--{key}")
                cmd.extend([str(v) for v in value])
        else:
            cmd.extend([f"--{key}", str(value)])
    # region agent log
    _debug_log(
        "H2",
        "src/web_v2/custom_model_api.py:_build_exec_cmd",
        "built execution command",
        {
            "script": script,
            "has_dataset_key": "dataset" in args,
            "dataset_value": args.get("dataset"),
            "cmd_has_dataset_flag": "--dataset" in cmd,
            "cmd_len": len(cmd),
        },
    )
    # endregion
    return cmd


@dataclass
class ProcessHandle:
    proc: Optional[subprocess.Popen] = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    abort_requested: bool = False
    running: bool = False
    run_id: str = ""


_PROC_TABLE: Dict[str, ProcessHandle] = {
    "training": ProcessHandle(),
    "evaluation": ProcessHandle(),
    "predict": ProcessHandle(),
}


def _abort_process(task: str) -> bool:
    handle = _PROC_TABLE[task]
    with handle.lock:
        handle.abort_requested = True
        proc = handle.proc
    if not proc or proc.poll() is not None:
        return False
    try:
        if hasattr(os, "killpg") and hasattr(os, "getpgid"):
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
        return True
    except Exception:
        return False


def _task_stream(task: str, script: str, args: Dict[str, Any]):
    handle = _PROC_TABLE[task]
    with handle.lock:
        if handle.running:
            yield _sse("error", {"message": f"{task} is already running"})
            return
        handle.running = True
        handle.abort_requested = False
        handle.run_id = uuid.uuid4().hex
        run_id = handle.run_id

    cmd = _build_exec_cmd(script, args)
    yield _sse("start", {"run_id": run_id, "command": redact_path_text(" ".join(cmd)), "task": task})
    train_state = TrainingLiveState(output_dir=str(args.get("output_dir", ""))) if task == "training" else None

    output_queue: queue.Queue[str] = queue.Queue()

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )
        with handle.lock:
            handle.proc = proc

        def _pump():
            while True:
                line = proc.stdout.readline() if proc.stdout else ""
                if line == "" and proc.poll() is not None:
                    break
                if line:
                    output_queue.put(line.rstrip("\n"))
            if proc.stdout:
                proc.stdout.close()

        thread = threading.Thread(target=_pump, daemon=True)
        thread.start()

        line_count = 0
        while proc.poll() is None:
            with handle.lock:
                aborted = handle.abort_requested
            if aborted:
                yield _sse("log", {"line": "Abort requested, terminating process..."})
                break
            try:
                while True:
                    line = output_queue.get_nowait()
                    line_count += 1
                    line_progress = min(0.95, 0.05 + line_count * 0.002)
                    if train_state is None:
                        yield _sse("progress", {"progress": line_progress, "message": f"{task} running"})
                    yield _sse("log", {"line": redact_path_text(line)})
                    if train_state is not None:
                        parsed_events = _parse_training_line(line, train_state)
                        for evt in parsed_events:
                            yield _sse(evt["event"], evt["data"])
                            if evt["event"] == "training_metrics":
                                epochs = evt["data"].get("epochs", []) if isinstance(evt.get("data"), dict) else []
                                total_epochs = int(args.get("num_epochs", 0) or 0)
                                if epochs and total_epochs > 0:
                                    current_epoch = int(epochs[-1])
                                    epoch_progress = min(0.98, (current_epoch + 1) / float(total_epochs))
                                    yield _sse(
                                        "progress",
                                        {
                                            "progress": epoch_progress,
                                            "message": f"Epoch {current_epoch + 1}/{total_epochs}",
                                        },
                                    )
            except queue.Empty:
                pass
            time.sleep(0.12)

        thread.join(timeout=2.0)
        while True:
            try:
                line = output_queue.get_nowait()
            except queue.Empty:
                break
            line_count += 1
            yield _sse("log", {"line": redact_path_text(line)})
            if train_state is not None:
                parsed_events = _parse_training_line(line, train_state)
                for evt in parsed_events:
                    yield _sse(evt["event"], evt["data"])
                    if evt["event"] == "training_metrics":
                        epochs = evt["data"].get("epochs", []) if isinstance(evt.get("data"), dict) else []
                        total_epochs = int(args.get("num_epochs", 0) or 0)
                        if epochs and total_epochs > 0:
                            current_epoch = int(epochs[-1])
                            epoch_progress = min(0.98, (current_epoch + 1) / float(total_epochs))
                            yield _sse(
                                "progress",
                                {
                                    "progress": epoch_progress,
                                    "message": f"Epoch {current_epoch + 1}/{total_epochs}",
                                },
                            )
            else:
                line_progress = min(0.95, 0.05 + line_count * 0.002)
                yield _sse("progress", {"progress": line_progress, "message": f"{task} running"})

        return_code = proc.poll()
        with handle.lock:
            aborted = handle.abort_requested

        if aborted:
            yield _sse("done", {"success": False, "aborted": True, "message": f"{task} aborted", "run_id": run_id, "final_progress": 0.0})
        elif return_code == 0:
            done_payload: Dict[str, Any] = {
                "success": True,
                "aborted": False,
                "message": f"{task} completed",
                "run_id": run_id,
                "final_progress": 1.0,
            }
            if train_state is not None:
                final_metrics = {
                    "epochs": train_state.epochs,
                    "train_loss": train_state.train_loss,
                    "val_loss": train_state.val_loss,
                    "val_metrics": train_state.val_metrics,
                }
                if train_state.epochs:
                    yield _sse("training_metrics", final_metrics)
                done_payload["training_metrics"] = final_metrics
                if train_state.csv_path:
                    done_payload["csv_download_url"] = build_web_v2_download_url(to_web_v2_public_path(train_state.csv_path))
                    done_payload["test_metrics"] = train_state.test_metrics
            yield _sse("done", done_payload)
        else:
            yield _sse(
                "done",
                {
                    "success": False,
                    "aborted": False,
                    "message": f"{task} failed",
                    "return_code": return_code,
                    "run_id": run_id,
                    "final_progress": 0.0,
                },
            )
    except Exception as exc:
        yield _sse("error", {"message": f"Failed to start {task}: {exc}"})
    finally:
        with handle.lock:
            proc = handle.proc
            handle.proc = None
            handle.running = False
            handle.abort_requested = False
        if proc and proc.poll() is None:
            try:
                if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                else:
                    proc.terminate()
            except Exception:
                pass


class GenericArgsRequest(BaseModel):
    args: Dict[str, Any] = Field(default_factory=dict)


class DatasetPreviewRequest(BaseModel):
    dataset_selection: str = "Custom"
    dataset_config: str = ""
    dataset_custom: str = ""
    train_file: str = ""
    valid_file: str = ""
    test_file: str = ""


class PredictBatchTextRequest(BaseModel):
    text: str = ""
    filename: str = "batch_input.fasta"


@router.get("/meta")
async def get_meta():
    data = _safe_read_json(_CONSTANT_PATH)
    return {
        "plm_models": data.get("plm_models", {}),
        "dataset_configs": data.get("dataset_configs", {}),
        "training_methods": [
            "full",
            "freeze",
            "ses-adapter",
            "plm-lora",
            "plm-qlora",
            "plm-adalora",
            "plm-dora",
            "plm-ia3",
        ],
        "pooling_methods": ["mean", "attention1d", "light_attention"],
        "problem_types": [
            "single_label_classification",
            "multi_label_classification",
            "regression",
            "residue_single_label_classification",
            "residue_multi_label_classification",
            "residue_regression",
        ],
        "metrics_options": [
            "accuracy",
            "mcc",
            "f1",
            "precision",
            "recall",
            "auroc",
            "aupr",
            "f1_max",
            "f1_positive",
            "f1_negative",
            "spearman_corr",
            "mse",
        ],
        "structure_seq_options": ["foldseek_seq", "ss8_seq"],
    }


@router.get("/models/folders")
async def model_folders(root: str = Query(default="ckpt"), max_depth: int = Query(default=5, ge=1, le=10)):
    root_path = _resolve_project_path(root)
    _assert_allowed_path(root_path, detail="Root path is not allowed.")
    base = [_to_project_rel(root_path)] if root_path.is_dir() else []
    return {"folders": base + _scan_folders_under(root_path, max_depth=max_depth)}


@router.get("/models")
async def models_in_folder(folder: str):
    return {"models": _scan_models_in_folder(folder)}


@router.get("/models/config")
async def model_config(model_path: str):
    return {"config": _load_model_config(model_path)}


@router.post("/upload")
async def upload_custom_model_file(file: UploadFile = File(...)):
    filename = os.path.basename(file.filename or f"custom-model-{uuid.uuid4().hex}.csv")
    ext = Path(filename).suffix.lower()
    if ext not in {".csv", ".tsv", ".xlsx", ".xls"}:
        raise HTTPException(status_code=400, detail="Only .csv/.tsv/.xlsx/.xls files are supported.")
    run_id = build_run_id_utc()
    upload_dir = get_web_v2_area_dir("uploads", tool="custom_model", run_id=run_id)
    save_path = upload_dir / make_web_v2_upload_name(1, filename)
    content = await file.read()
    with open(save_path, "wb") as out:
        out.write(content)
    headers = _read_tabular_columns(str(save_path))
    create_run_manifest(
        run_id=run_id,
        tool="custom_model",
        status="uploaded",
        inputs=[{"path": str(save_path), "name": filename, "size": len(content)}],
    )
    return {"file_path": to_web_v2_public_path(save_path), "name": filename, "suffix": ext, "columns": headers, "run_id": run_id}


@router.post("/predict/upload")
async def upload_predict_input_file(file: UploadFile = File(...)):
    filename = os.path.basename(file.filename or f"predict-input-{uuid.uuid4().hex}.fasta")
    ext = Path(filename).suffix.lower()
    if ext not in {".csv", ".tsv", ".xlsx", ".xls", ".fasta", ".fa", ".txt"}:
        raise HTTPException(status_code=400, detail="Only .csv/.tsv/.xlsx/.xls/.fasta/.fa/.txt files are supported.")
    run_id = build_run_id_utc()
    upload_dir = get_web_v2_area_dir("uploads", tool="custom_model", run_id=run_id)
    save_path = upload_dir / make_web_v2_upload_name(1, filename)
    content = await file.read()
    with open(save_path, "wb") as out:
        out.write(content)
    headers: List[str] = []
    if ext in {".csv", ".tsv", ".xlsx", ".xls"}:
        headers = _read_tabular_columns(str(save_path))
    create_run_manifest(
        run_id=run_id,
        tool="custom_model",
        status="uploaded",
        inputs=[{"path": str(save_path), "name": filename, "size": len(content)}],
    )
    return {"file_path": to_web_v2_public_path(save_path), "name": filename, "suffix": ext, "columns": headers, "run_id": run_id}


@router.post("/predict/text-upload")
async def upload_predict_text(payload: PredictBatchTextRequest):
    text = str(payload.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="FASTA text cannot be empty.")
    filename = os.path.basename(str(payload.filename or "batch_input.fasta").strip()) or "batch_input.fasta"
    suffix = Path(filename).suffix.lower()
    if suffix not in {".fasta", ".fa", ".txt"}:
        filename = f"{Path(filename).stem}.fasta"
    run_id = build_run_id_utc()
    upload_dir = get_web_v2_area_dir("uploads", tool="custom_model", run_id=run_id)
    save_path = upload_dir / make_web_v2_upload_name(1, filename)
    save_path.write_text(text, encoding="utf-8")
    create_run_manifest(
        run_id=run_id,
        tool="custom_model",
        status="uploaded",
        inputs=[{"path": str(save_path), "name": filename, "size": len(text.encode('utf-8'))}],
    )
    return {"file_path": to_web_v2_public_path(save_path), "name": filename, "suffix": Path(filename).suffix.lower(), "columns": [], "run_id": run_id}


@router.get("/dataset/config-values")
async def dataset_config_values(dataset_config: str):
    constant = _safe_read_json(_CONSTANT_PATH)
    dataset_configs = constant.get("dataset_configs", {})
    if dataset_config not in dataset_configs:
        raise HTTPException(status_code=400, detail=f"Unknown dataset config: {dataset_config}")
    cfg = _safe_read_json(Path(dataset_configs[dataset_config]))
    metrics = cfg.get("metrics", [])
    if isinstance(metrics, str):
        metrics = [x.strip() for x in metrics.split(",") if x.strip()]
    if not isinstance(metrics, list):
        metrics = []
    structure_seq = cfg.get("structure_seq", [])
    if isinstance(structure_seq, str):
        structure_seq = [x.strip() for x in structure_seq.split(",") if x.strip()]
    if not isinstance(structure_seq, list):
        structure_seq = []
    return {
        "problem_type": cfg.get("problem_type"),
        "num_labels": cfg.get("num_labels"),
        "metrics": [x for x in metrics if isinstance(x, str)],
        "sequence_column_name": cfg.get("sequence_column_name"),
        "label_column_name": cfg.get("label_column_name"),
        "structure_seq": [x for x in structure_seq if isinstance(x, str)],
        "pdb_dir": cfg.get("pdb_dir"),
    }


@router.post("/dataset/preview")
async def dataset_preview(payload: DatasetPreviewRequest):
    constant = _safe_read_json(_CONSTANT_PATH)
    dataset_configs = constant.get("dataset_configs", {})

    try:
        if payload.dataset_selection == "Pre-defined":
            if payload.dataset_config not in dataset_configs:
                raise HTTPException(status_code=400, detail=f"Unknown dataset config: {payload.dataset_config}")
            cfg = _safe_read_json(Path(dataset_configs[payload.dataset_config]))
            dataset_name = cfg.get("dataset", "")
            dataset = load_dataset(dataset_name)
            display_name = dataset_name
            column_options: List[str] = []
        else:
            train_file = str(payload.train_file or "").strip()
            valid_file = str(payload.valid_file or "").strip()
            test_file = str(payload.test_file or "").strip()

            file_mode = bool(train_file or valid_file or test_file)
            if file_mode:
                def _safe_len(path: str) -> int:
                    if not path:
                        return 0
                    try:
                        file_path = _resolve_client_file_path(path)
                    except HTTPException:
                        return 0
                    if not file_path.exists() or not file_path.is_file():
                        return 0
                    ext = file_path.suffix.lower()
                    try:
                        if ext == ".csv":
                            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                                return max(sum(1 for _ in f) - 1, 0)
                        if ext == ".tsv":
                            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                                return max(sum(1 for _ in f) - 1, 0)
                        if ext in {".xlsx", ".xls"}:
                            return len(pd.read_excel(file_path))
                    except Exception:
                        return 0
                    return 0

                preview_path = train_file or valid_file or test_file
                preview_df = _read_tabular_df(preview_path, sample_n=3)
                columns = [str(c) for c in list(preview_df.columns)]
                rows: List[Dict[str, Any]] = []
                for _, row in preview_df.iterrows():
                    item: Dict[str, Any] = {}
                    for k, v in row.to_dict().items():
                        text = str(v)
                        if len(text) > 120:
                            text = text[:120] + "..."
                        item[str(k)] = text
                    rows.append(item)
                return {
                    "dataset_name": preview_path,
                    "stats": {
                        "train": _safe_len(train_file),
                        "validation": _safe_len(valid_file),
                        "test": _safe_len(test_file),
                    },
                    "preview": {"columns": columns, "rows": rows},
                    "column_options": columns,
                }

            if not payload.dataset_custom:
                raise HTTPException(status_code=400, detail="Custom dataset path is required")
            dataset = load_dataset(payload.dataset_custom)
            display_name = payload.dataset_custom
            column_options = []

        split = "train" if "train" in dataset else list(dataset.keys())[0]
        split_data = dataset[split]
        sample_n = min(3, len(split_data))
        rows: List[Dict[str, Any]] = []
        columns: List[str] = []
        if sample_n > 0:
            samples = split_data.select(range(sample_n))
            columns = list(samples[0].keys())
            for sample in samples:
                row: Dict[str, Any] = {}
                for k, v in sample.items():
                    text = str(v)
                    if len(text) > 120:
                        text = text[:120] + "..."
                    row[k] = text
                rows.append(row)

        return {
            "dataset_name": display_name,
            "stats": {
                "train": len(dataset["train"]) if "train" in dataset else 0,
                "validation": len(dataset["validation"]) if "validation" in dataset else 0,
                "test": len(dataset["test"]) if "test" in dataset else 0,
            },
            "preview": {"columns": columns, "rows": rows},
            "column_options": column_options or columns,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to preview dataset: {exc}") from exc


@router.post("/training/preview")
async def training_preview(payload: GenericArgsRequest):
    constant = _safe_read_json(_CONSTANT_PATH)
    args = _normalize_training_args(payload.args, constant)
    return {"command": redact_path_text(preview_command(args)), "args": args}


@router.post("/evaluation/preview")
async def evaluation_preview(payload: GenericArgsRequest):
    constant = _safe_read_json(_CONSTANT_PATH)
    args = _normalize_evaluation_args(payload.args, constant)
    return {"command": redact_path_text(preview_eval_command(args)), "args": args}


@router.post("/predict/preview")
async def predict_preview(payload: GenericArgsRequest):
    constant = _safe_read_json(_CONSTANT_PATH)
    args = _normalize_predict_args(payload.args, constant)
    return {
        "command": redact_path_text(preview_predict_command(args, payload.args.get("prediction_mode") == "batch")),
        "args": args,
    }


@router.post("/training/start")
async def training_start(payload: GenericArgsRequest):
    constant = _safe_read_json(_CONSTANT_PATH)
    args = _normalize_training_args(payload.args, constant)
    return StreamingResponse(_task_stream("training", "src/train.py", args), media_type="text/event-stream")


@router.post("/evaluation/start")
async def evaluation_start(payload: GenericArgsRequest):
    constant = _safe_read_json(_CONSTANT_PATH)
    args = _normalize_evaluation_args(payload.args, constant)
    return StreamingResponse(_task_stream("evaluation", "src/evaluate.py", args), media_type="text/event-stream")


@router.post("/predict/start")
async def predict_start(payload: GenericArgsRequest):
    constant = _safe_read_json(_CONSTANT_PATH)
    args = _normalize_predict_args(payload.args, constant)
    return StreamingResponse(_task_stream("predict", "src/predict.py", args), media_type="text/event-stream")


@router.post("/training/abort")
async def training_abort():
    return {"aborted": _abort_process("training")}


@router.post("/evaluation/abort")
async def evaluation_abort():
    return {"aborted": _abort_process("evaluation")}


@router.post("/predict/abort")
async def predict_abort():
    return {"aborted": _abort_process("predict")}


@router.get("/artifacts/download")
async def download_artifact(path: str):
    target = _resolve_client_file_path(path, allow_ckpt=True)
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    if not ensure_within_roots(target, _ALLOWED_FILE_ROOTS):
        raise HTTPException(status_code=403, detail="Access denied")
    return FileResponse(path=str(target), filename=target.name, media_type="application/octet-stream")
