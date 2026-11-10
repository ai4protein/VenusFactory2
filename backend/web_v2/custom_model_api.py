from backend.web_v2.sse_events import encode_event
import json
import math
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from datasets import load_dataset
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, Request, Response, Depends
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from mcp_tools.tools.runtime_tool_policy import assert_local_feature, is_online_mode
from mcp_tools.common.command import preview_command, preview_eval_command, preview_predict_command
from mcp_tools.common.common_utils import (
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


ONLINE_TRAINING_MODEL = "facebook/esm2_t6_8M_UR50D"


def _require_training_model(args: Dict[str, Any]) -> None:
    if is_online_mode() and args.get("plm_model") != ONLINE_TRAINING_MODEL:
        raise HTTPException(403, "Online training supports ESM2-8M only.")
    initial = args.get("initial_model_path")
    if initial:
        config = _load_model_config(str(initial))
        backbone = config.get("plm_model")
        if (is_online_mode() and backbone != ONLINE_TRAINING_MODEL) or (backbone and backbone != args.get("plm_model")):
            raise HTTPException(400, "Choose a checkpoint with the same protein model. Online training requires ESM2-8M.")


def _require_local_custom_model(feature: str) -> None:
    try:
        assert_local_feature(feature)
    except RuntimeError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


from backend.web_v2.job_access import browser_owner, custom_request_scope, guard_artifact
from backend.web_v2.job_records import JobRecords, page_jobs
from backend.web_v2.training_progress import TrainingProgress, parse_training_line

router = APIRouter(dependencies=[Depends(custom_request_scope)], prefix="/api/custom-model", tags=["custom-model-v2"])

_CONSTANT_PATH = Path(__file__).resolve().parents[2] / "src" / "constant.json"
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
    guard_artifact(path)
    if is_online_mode():
        from backend.web_v2.job_access import request_context, request_owners
        request = request_context.get()
        if request is not None:
            try:
                JobRecords().check_artifact(path, request_owners(request), require_grant=True)
            except FileNotFoundError:
                raise HTTPException(404, "File not found") from None
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
    if max_depth < 1:
        return []
    result: List[str] = []
    for current, directories, _ in os.walk(root, topdown=True, followlinks=False):
        depth = len(Path(current).relative_to(root).parts)
        descend = []
        for name in directories:
            path = Path(current) / name
            if path.is_symlink():
                continue
            try:
                guard_artifact(path)
            except HTTPException:
                continue
            result.append(_to_project_rel(path))
            if depth + 1 < max_depth:
                descend.append(name)
        # Prune before traversal: hidden and out-of-depth trees are never scanned.
        directories[:] = descend
    return sorted(result)


def _scan_models_in_folder(folder_path: str) -> List[Dict[str, str]]:
    resolved_folder = _resolve_project_path(folder_path) if folder_path else None
    if not resolved_folder or not resolved_folder.is_dir():
        return []
    _assert_allowed_path(resolved_folder)
    folder = resolved_folder
    models: List[Dict[str, str]] = []
    for pt_file in sorted(folder.rglob("*.pt")):
        try: guard_artifact(pt_file)
        except HTTPException: continue
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


def _read_tabular_df(file_path: str, sample_n: int | None = 3) -> pd.DataFrame:
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
    return list(_read_tabular_df(file_path, sample_n=0).columns)


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
    return encode_event(event, data, ensure_ascii=True)


def _write_test_metrics_csv(output_dir: str, metrics: Dict[str, float]) -> str:
    try:
        # Keep metrics in the job's granted output directory. Replaying a
        # durable log updates this same file instead of creating orphan copies.
        out_dir = (_resolve_project_path(output_dir) if output_dir else
                   get_web_v2_area_dir("results", tool="custom_model", run_id=build_run_id_utc()))
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "test_results_metrics.csv"
        lines = ["Metric,Value"]
        for key, value in sorted(metrics.items()):
            lines.append(f"{key},{value:.6f}")
        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(csv_path)
    except Exception:
        return ""


def _training_metrics_download(path: str) -> str:
    from urllib.parse import urlencode
    if not path:
        return ""
    # The storage root may be configured outside the repository. Keep that
    # identity resolvable instead of falling back to a basename.
    value = os.path.relpath(Path(path).resolve(), _PROJECT_ROOT)
    return "/api/custom-model/artifacts/download?" + urlencode({"path": value})


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


def _uploaded_training_stats(args: Dict[str, Any]) -> Dict[str, int]:
    """Validate uploaded data before enqueueing and show the actual planned split sizes."""
    from mcp_tools.models.factory.training.data.split_policy import holdout_indices, assert_disjoint_sequences
    frames = {name: _read_tabular_df(args[field], sample_n=None)
              for name, field in (("train", "train_file"), ("validation", "valid_file"), ("test", "test_file"))
              if args.get(field)}
    if "train" not in frames:
        raise HTTPException(400, "Upload a training table first.")
    seq, label = args.get("sequence_column_name", "aa_seq"), args.get("label_column_name", "label")
    try:
        for name, frame in frames.items():
            if frame.empty: raise ValueError(f"The {name} table is empty.")
            for column in (seq, label):
                if column not in frame: raise ValueError(f"Column '{column}' is missing from the {name} table.")
                if frame[column].isna().any(): raise ValueError(f"Column '{column}' has empty values in the {name} table.")
        missing = [name for name in ("validation", "test") if name not in frames]
        if missing:
            original = frames["train"]
            labels = original[label].tolist() if args.get("problem_type") == "single_label_classification" else None
            indices = holdout_indices(original[seq].tolist(), missing, labels)
            for name, rows in indices.items(): frames[name] = original.iloc[rows]
        assert_disjoint_sequences({name: frame[seq] for name, frame in frames.items()})
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {name: len(frame) for name, frame in frames.items()}


def _check_online_dataset_id(value: str) -> None:
    if is_online_mode() and value:
        if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*(?:/[A-Za-z0-9][A-Za-z0-9_.-]*)?", value)
                or _resolve_project_path(value).exists()):
            raise HTTPException(400, "Use a dataset ID or upload a table.")


def _normalize_training_args(payload: Dict[str, Any], constant: Dict[str, Any]) -> Dict[str, Any]:
    plm_models = constant.get("plm_models", {})
    dataset_configs = constant.get("dataset_configs", {})

    plm_display = payload.get("plm_model", "")
    if plm_display not in plm_models:
        raise HTTPException(status_code=400, detail=f"Unknown plm_model: {plm_display}")

    _require_training_model({"plm_model": plm_models[plm_display]})
    _check_online_dataset_id(str(payload.get("dataset_custom") or "").strip())
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
        if (valid_file or test_file) and not train_file:
            raise HTTPException(400, "Add a training table before validation or test data.")
        if not dataset_custom and not train_file:
            raise HTTPException(400, "Upload a training table or choose a dataset.")
        supplied = [p for p in (train_file, valid_file, test_file) if p]
        if len(set(supplied)) != len(supplied):
            raise HTTPException(400, "Use different files for training, validation and test data.")
        effective_dataset = dataset_custom or train_file
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

    if args.get("train_file"):
        _uploaded_training_stats(args)

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

    if payload.get("training_mode") == "Continue Training":
        checkpoint = str(payload.get("transfer_model_dropdown") or "").strip()
        if not checkpoint:
            raise HTTPException(400, "Select a checkpoint to continue training.")
        args["initial_model_path"] = str(_resolve_client_file_path(checkpoint))
    _require_training_model(args)
    for name in ("num_epochs", "max_seq_len", "gradient_accumulation_steps", "batch_size", "batch_token", "learning_rate"):
        if name in args:
            try: value = float(args[name])
            except (TypeError, ValueError): raise HTTPException(400, f"Enter a valid positive value for {name}.")
            if not math.isfinite(value) or value <= 0 or (name != "learning_rate" and not value.is_integer()):
                raise HTTPException(400, f"Enter a valid positive value for {name}.")
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
    return cmd


def _queue_owner(request: Request, response: Response):
    return browser_owner(request, response, 'custom')

def _register_task(job):
    records=JobRecords()
    args=job['metadata']['args']
    task=job['metadata']['task']
    title=str(args.get('plm_model') or args.get('model_name') or task)
    title=Path(title).name
    meta=records.register('custom',job['id'],job['owner'],job['kind'],title,job['created'])
    records.protect(get_web_v2_area_dir('results',tool='custom_model'))
    for name in ['artifact_root','output_root']:
        if job['metadata'].get(name):records.grant(job['metadata'][name],'custom',job['owner'])
    # Previous queue jobs already use their own run directory.
    output=args.get('output_dir','')
    if task=='training' and Path(output).name.startswith('run_'):
        records.grant(_resolve_project_path(output),'custom',job['owner'])
    return meta

def _task_public(identity,owner):
    from backend.web_v2.job_queue import JobQueue
    job=_owned_task(identity,owner)
    return {**JobQueue().public(job['id'],owner),**_register_task(job)}

def _copy_response_headers(result,response):
    result.raw_headers.extend((k,v) for k,v in response.raw_headers if k.lower() not in {b'content-length',b'content-type'})
    result.headers['Cache-Control']='private, no-store'
    result.headers['Vary']='Cookie'
    return result

def _create_task(task,script,args,owner):
    from backend.web_v2.job_queue import launch
    args=dict(args)
    from mcp_tools.tools.capability_policy import CapabilityStore
    capability_tool = 'train_protein_model' if task == 'training' else 'protein_model_predict'
    CapabilityStore().assert_call(capability_tool, args)
    if task == "training": _require_training_model(args)
    run_name='run_'+uuid.uuid4().hex[:16]
    if task=='training':args['output_dir']=str(Path(args.get('output_dir') or 'ckpt/demo')/run_name)
    work=get_web_v2_area_dir('results',tool='custom_model')/run_name
    work.mkdir(parents=True,exist_ok=True)
    if task == 'training' and is_online_mode(): args['output_dir'] = str(work / 'checkpoints')
    command=_build_exec_cmd(script,args)
    env=dict(os.environ,TEMP_OUTPUTS_DIR=str(work))
    timeout=int(os.getenv('VENUS_QUEUE_CUSTOM_TIMEOUT','86400'))
    if task != 'training':
        from mcp_tools.models.factory.runners.checkpoint import prepare
        spec=prepare(command)
        command=spec.argv;env={**spec.env,'TEMP_OUTPUTS_DIR':str(work)};timeout=min(timeout,spec.timeout)
    resource='cpu' if str(args.get('device','')).lower()=='cpu' else 'gpu'
    identity,_=launch(command,owner=owner,kind='custom:'+task,resource=resource,cwd=_PROJECT_ROOT,env=env,
        timeout=timeout,priority=10 if task=='predict' else 0,
        metadata={'capability_tool':capability_tool,'capability_arguments':args,'task':task,'args':args,'artifact_root':str(work.resolve()),'output_root':str(_resolve_project_path(args['output_dir'])) if task=='training' else ''})
    from backend.web_v2.job_queue import JobQueue
    _register_task(JobQueue().get(identity,owner))
    return identity

def _task_stream(identity,owner):
    from backend.web_v2.job_queue import JobQueue, TERMINAL
    q=JobQueue();job=_owned_task(identity,owner);identity=job['id'];task=job['metadata']['task'];args=job['metadata']['args']
    train_state=TrainingProgress() if task=='training' else None
    csv_path=''
    spec=json.loads((q.files(identity)/'command.json').read_text())
    yield _sse('start',{'run_id':identity,'command':redact_path_text(' '.join(spec['command'])),'task':task,**_register_task(job)})
    log=q.files(identity)/'worker.log';log.touch(exist_ok=True)
    # Replaying this durable log rebuilds training curves after browser/server reconnect.
    with log.open(errors='replace') as f:
        last=None
        while True:
            job=_task_public(identity,owner)
            state=(job['status'],job['position'],job['cancel_requested'])
            if state!=last:
                yield _sse('queue',job);last=state
            while True:
                line=f.readline()
                if not line:break
                line=line.rstrip('\n')
                yield _sse('log',{'line':redact_path_text(line)})
                if train_state is not None:
                    for evt in parse_training_line(line,train_state):
                        if evt['event']=='test_results':
                            csv_path=_write_test_metrics_csv(str(args.get('output_dir','')),train_state.test_metrics)
                            evt['data']['csv_download_url']=_training_metrics_download(csv_path)
                        yield _sse(evt['event'],evt['data'])
                        if evt['event']=='training_metrics':
                            epochs=evt['data'].get('epochs',[]);total=int(args.get('num_epochs',0) or 0)
                            if epochs and total:
                                epoch=int(epochs[-1])+1
                                yield _sse('progress',{'progress':min(.98,epoch/total),'message':f'Epoch {epoch}/{total}'})
            if job['status'] in TERMINAL:break
            time.sleep(.3)
    done={'success':job['status']=='completed','aborted':job['status']=='cancelled','message':job['message'],
          'run_id':identity,'final_progress':1.0 if job['status']=='completed' else 0.0}
    if train_state is not None:
        done['training_metrics']=train_state.metrics()
        if csv_path:
            done['csv_download_url']=_training_metrics_download(csv_path)
            done['test_metrics']=train_state.test_metrics
    yield _sse('done',done)

def _queued_stream(task,script,args,owner,response):
    try:identity=_create_task(task,script,args,owner)
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc
    return _copy_response_headers(StreamingResponse(_task_stream(identity,owner),media_type='text/event-stream'),response)

def _owned_task(identity,owner,task=None):
    from backend.web_v2.job_queue import JobQueue
    try:
        if identity.upper().startswith('VF-'):identity=JobRecords().resolve('custom',identity,owner)
        job=JobQueue().get(identity,owner)
        if not job['kind'].startswith('custom:') or (task and job['kind']!='custom:'+task):raise FileNotFoundError(identity)
        return job
    except FileNotFoundError:raise HTTPException(404,'Job not found')

def _abort_task(task,identity,owner):
    from backend.web_v2.job_queue import JobQueue
    job=_owned_task(identity,owner,task)
    JobQueue().cancel(job['id'],owner)
    return {'aborted':True}

@router.get('/session')
async def custom_session(owner=Depends(_queue_owner)):
    return {'ready':True}

@router.get('/jobs')
async def custom_jobs(task: str='', owner=Depends(_queue_owner)):
    from backend.web_v2.job_queue import JobQueue
    return [_task_public(j['id'],owner) for j in JobQueue().list(owner,'custom:'+task if task else None) if j['kind'].startswith('custom:')]

@router.get('/job-history')
async def custom_job_history(task: str='', query: str=Query('',max_length=200), status: str='', kind: str='',
                             offset: int=Query(0,ge=0), limit: int=Query(20,ge=1,le=100), owner=Depends(_queue_owner)):
    await custom_jobs(task,owner)  # Index legacy tasks using their existing owner.
    jobs=[_task_public(r['identity'],owner) for r in JobRecords().entries('custom',owner) if not task or r['kind']=='custom:'+task]
    return page_jobs(jobs,query=query,status=status,kind=kind,offset=offset,limit=limit)

@router.get('/jobs/{identity}')
async def custom_job(identity: str,owner=Depends(_queue_owner)):
    return _task_public(identity,owner)

@router.post('/jobs/{identity}/stream')
async def custom_job_stream(identity: str, response: Response, owner=Depends(_queue_owner)):
    _owned_task(identity,owner)
    return _copy_response_headers(StreamingResponse(_task_stream(identity,owner),media_type='text/event-stream'),response)

@router.post('/jobs/{identity}/cancel')
async def custom_job_cancel(identity: str, owner=Depends(_queue_owner)):
    from backend.web_v2.job_queue import JobQueue
    job=_owned_task(identity,owner)
    JobQueue().cancel(job['id'],owner)
    return _task_public(job['id'],owner)

@router.post('/jobs/{identity}/retry')
async def custom_job_retry(identity: str, owner=Depends(_queue_owner)):
    old=_owned_task(identity,owner);task=old['metadata']['task']
    if old['status'] not in {'failed','interrupted','cancelled'}:raise HTTPException(409,'Only stopped tasks can retry')
    if task!='training':_require_local_custom_model('custom_model_'+task)
    script={'training':'src/train.py','evaluation':'src/evaluate.py','predict':'src/predict.py'}[task]
    identity=_create_task(task,script,old['metadata']['args'],owner)
    return _task_public(identity,owner)


class GenericArgsRequest(BaseModel):
    args: Dict[str, Any] = Field(default_factory=dict)


class DatasetPreviewRequest(BaseModel):
    dataset_selection: str = "Custom"
    dataset_config: str = ""
    dataset_custom: str = ""
    train_file: str = ""
    valid_file: str = ""
    test_file: str = ""
    training: bool = False
    sequence_column_name: str = "aa_seq"
    label_column_name: str = "label"
    problem_type: str = "single_label_classification"


class PredictBatchTextRequest(BaseModel):
    text: str = ""
    filename: str = "batch_input.fasta"


@router.get("/meta")
async def get_meta():
    data = _safe_read_json(_CONSTANT_PATH)
    return {
        "plm_models": data.get("plm_models", {}),
        "training_plm_models": {name: model for name, model in data.get("plm_models", {}).items()
                                if not is_online_mode() or model == ONLINE_TRAINING_MODEL},
        "online_training": is_online_mode(),
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
async def model_folders(root: str = Query(default="ckpt"), max_depth: int = Query(default=5, ge=1, le=10), owner=Depends(_queue_owner)):
    if is_online_mode():
        from backend.web_v2.job_queue import JobQueue
        return {"folders": [_to_project_rel(Path(j['metadata']['output_root']))
                            for item in JobQueue().list(owner, 'custom:training')
                            for j in [JobQueue().get(item['id'], owner)]
                            if j['metadata'].get('output_root') and Path(j['metadata']['output_root']).is_dir()]}
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
async def upload_custom_model_file(file: UploadFile = File(...), owner=Depends(_queue_owner)):
    filename = os.path.basename(file.filename or f"custom-model-{uuid.uuid4().hex}.csv")
    ext = Path(filename).suffix.lower()
    if ext not in {".csv", ".tsv", ".xlsx", ".xls"}:
        raise HTTPException(status_code=400, detail="Only .csv/.tsv/.xlsx/.xls files are supported.")
    run_id = build_run_id_utc()
    upload_dir = get_web_v2_area_dir("uploads", tool="custom_model", run_id=run_id)
    JobRecords().grant(upload_dir, 'custom', owner)
    save_path = upload_dir / make_web_v2_upload_name(1, filename)
    from backend.web_v2.upload_limits import save_upload
    size = await save_upload(file, save_path)
    headers = _read_tabular_columns(str(save_path))
    create_run_manifest(
        run_id=run_id,
        tool="custom_model",
        status="uploaded",
        inputs=[{"path": str(save_path), "name": filename, "size": size}],
    )
    return {"file_path": to_web_v2_public_path(save_path), "name": filename, "suffix": ext, "columns": headers, "run_id": run_id}


@router.post("/predict/upload")
async def upload_predict_input_file(file: UploadFile = File(...), owner=Depends(_queue_owner)):
    filename = os.path.basename(file.filename or f"predict-input-{uuid.uuid4().hex}.fasta")
    ext = Path(filename).suffix.lower()
    if ext not in {".csv", ".tsv", ".xlsx", ".xls", ".fasta", ".fa", ".txt"}:
        raise HTTPException(status_code=400, detail="Only .csv/.tsv/.xlsx/.xls/.fasta/.fa/.txt files are supported.")
    run_id = build_run_id_utc()
    upload_dir = get_web_v2_area_dir("uploads", tool="custom_model", run_id=run_id)
    JobRecords().grant(upload_dir, 'custom', owner)
    save_path = upload_dir / make_web_v2_upload_name(1, filename)
    from backend.web_v2.upload_limits import save_upload
    size = await save_upload(file, save_path)
    headers: List[str] = []
    if ext in {".csv", ".tsv", ".xlsx", ".xls"}:
        headers = _read_tabular_columns(str(save_path))
    create_run_manifest(
        run_id=run_id,
        tool="custom_model",
        status="uploaded",
        inputs=[{"path": str(save_path), "name": filename, "size": size}],
    )
    return {"file_path": to_web_v2_public_path(save_path), "name": filename, "suffix": ext, "columns": headers, "run_id": run_id}


@router.post("/predict/text-upload")
async def upload_predict_text(payload: PredictBatchTextRequest, owner=Depends(_queue_owner)):
    text = str(payload.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="FASTA text cannot be empty.")
    filename = os.path.basename(str(payload.filename or "batch_input.fasta").strip()) or "batch_input.fasta"
    suffix = Path(filename).suffix.lower()
    if suffix not in {".fasta", ".fa", ".txt"}:
        filename = f"{Path(filename).stem}.fasta"
    run_id = build_run_id_utc()
    upload_dir = get_web_v2_area_dir("uploads", tool="custom_model", run_id=run_id)
    JobRecords().grant(upload_dir, 'custom', owner)
    save_path = upload_dir / make_web_v2_upload_name(1, filename)
    from backend.web_v2.upload_limits import check_upload_size
    check_upload_size(len(text.encode("utf-8")))
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
                    "stats": _uploaded_training_stats(payload.model_dump()) if payload.training else {
                        "train": _safe_len(train_file),
                        "validation": _safe_len(valid_file),
                        "test": _safe_len(test_file),
                    },
                    "preview": {"columns": columns, "rows": rows},
                    "column_options": columns,
                }

            if not payload.dataset_custom:
                raise HTTPException(status_code=400, detail="Custom dataset path is required")
            _check_online_dataset_id(payload.dataset_custom)
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
    _require_local_custom_model("custom_model_evaluation")
    constant = _safe_read_json(_CONSTANT_PATH)
    args = _normalize_evaluation_args(payload.args, constant)
    return {"command": redact_path_text(preview_eval_command(args)), "args": args}


@router.post("/predict/preview")
async def predict_preview(payload: GenericArgsRequest):
    _require_local_custom_model("custom_model_predict")
    constant = _safe_read_json(_CONSTANT_PATH)
    args = _normalize_predict_args(payload.args, constant)
    return {
        "command": redact_path_text(preview_predict_command(args, payload.args.get("prediction_mode") == "batch")),
        "args": args,
    }


@router.post("/training/start")
async def training_start(payload: GenericArgsRequest, response: Response, owner=Depends(_queue_owner)):
    constant = _safe_read_json(_CONSTANT_PATH)
    args = _normalize_training_args(payload.args, constant)
    return _queued_stream("training", "src/train.py", args, owner, response)


@router.post("/evaluation/start")
async def evaluation_start(payload: GenericArgsRequest, response: Response, owner=Depends(_queue_owner)):
    _require_local_custom_model("custom_model_evaluation")
    constant = _safe_read_json(_CONSTANT_PATH)
    args = _normalize_evaluation_args(payload.args, constant)
    return _queued_stream("evaluation", "src/evaluate.py", args, owner, response)


@router.post("/predict/start")
async def predict_start(payload: GenericArgsRequest, response: Response, owner=Depends(_queue_owner)):
    _require_local_custom_model("custom_model_predict")
    constant = _safe_read_json(_CONSTANT_PATH)
    args = _normalize_predict_args(payload.args, constant)
    return _queued_stream("predict", "src/predict.py", args, owner, response)


@router.post("/training/abort")
async def training_abort(job_id: str, owner=Depends(_queue_owner)):
    return _abort_task("training",job_id,owner)


@router.post("/evaluation/abort")
async def evaluation_abort(job_id: str, owner=Depends(_queue_owner)):
    return _abort_task("evaluation",job_id,owner)


@router.post("/predict/abort")
async def predict_abort(job_id: str, owner=Depends(_queue_owner)):
    return _abort_task("predict",job_id,owner)


@router.get("/artifacts/download")
async def download_artifact(path: str):
    target = _resolve_client_file_path(path, allow_ckpt=True)
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    if not ensure_within_roots(target, _ALLOWED_FILE_ROOTS):
        raise HTTPException(status_code=403, detail="Access denied")
    return FileResponse(path=str(target), filename=target.name, media_type="application/octet-stream", headers={'Cache-Control':'private, no-store','Vary':'Cookie'})
