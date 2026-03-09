"""
Finetuned operations: predict_protein_function using task (通俗命名) + model; adapter_path from ckpt/{dataset}/{model_folder}.
Ref: src/constant.json dataset_mapping_function, model_adapter_mapping_function.
Success: status, file_info, content_preview, biological_metadata, execution_context.
Error: status "error", error { type, message, suggestion }, file_info null.
"""

import csv
import json
import os
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repo root on path when script is run directly (avoid "relative import with no known parent package")
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_PREVIEW_LEN = 500
_SOURCE = "Predict_Finetuned"
# Backend module names: protein function allowed set; _GET_MODULE_KEYS = all loadable modules for _get_module
_MODEL_KEYS = ("esm2")
# ---------- Residue (residue_mapping_function, model_residue_mapping_function) ----------
_RESIDUE_MODEL_KEYS = ("esm2")  # model_residue_mapping_function has no ProtBert
_GET_MODULE_KEYS = ("protbert", "prott5", "ankh", "esm2")
# Default base path for adapter: ckpt/{dataset_name}/{model_adapter_folder}, e.g. ckpt/DeepET_Topt/ankh-large
_CKPT_BASE = "ckpt"

_CONSTANT_PATH = Path(__file__).resolve().parent.parent.parent.parent / "constant.json"


def _load_constant() -> Dict[str, Any]:
    """Load constant.json web_ui section (dataset_mapping_function, model_adapter_mapping_function, model_mapping_function)."""
    if not _CONSTANT_PATH.exists():
        return {}
    with open(_CONSTANT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("web_ui", {})


def _get_task_choices() -> List[str]:
    """Task names (通俗命名) from constant.json dataset_mapping_function."""
    const = _load_constant()
    mapping = const.get("dataset_mapping_function", {})
    return list(mapping.keys()) if mapping else []


def _get_model_choices() -> List[str]:
    """Model display names from constant.json model_mapping_function (e.g. Ankh-large, ESM2-650M)."""
    const = _load_constant()
    mapping = const.get("model_mapping_function", {})
    return list(mapping.keys()) if mapping else []




def _get_residue_task_choices() -> List[str]:
    """Residue task names (通俗命名) from constant.json residue_mapping_function (Activity Site, Binding Site, etc.)."""
    const = _load_constant()
    mapping = const.get("residue_mapping_function", {})
    return list(mapping.keys()) if mapping else []


def _residue_task_to_dataset(task: str) -> Optional[str]:
    """Map residue task to first dataset name. e.g. Activity Site -> VenusX_Res_Act_MP90."""
    const = _load_constant()
    mapping = const.get("residue_mapping_function", {})
    names = mapping.get(task) if isinstance(mapping.get(task), list) else ([mapping.get(task)] if mapping.get(task) else None)
    return names[0] if names else None


def _get_residue_model_choices() -> List[str]:
    """Model names for residue prediction from model_residue_mapping_function (ESM2-650M, Ankh-large, ProtT5-xl-uniref50)."""
    const = _load_constant()
    mapping = const.get("model_residue_mapping_function", {})
    return list(mapping.keys()) if mapping else []


def _model_name_to_residue_key(model_name: str) -> str:
    """Normalize model_name to residue module key using model_residue_mapping_function."""
    const = _load_constant()
    func = const.get("model_residue_mapping_function", {})
    key = func.get(model_name) or func.get(model_name.strip())
    if key:
        return key.lower().strip()
    mn = model_name.lower().strip()
    return mn if mn in _RESIDUE_MODEL_KEYS else mn


def _resolve_residue_adapter_path(task: str, model_name: str, adapter_path: Optional[str], ckpt_base: str) -> str:
    """Resolve adapter path for residue prediction: ckpt_base/{residue_dataset}/{adapter_folder}."""
    if adapter_path and Path(adapter_path).exists():
        return adapter_path
    dataset = _residue_task_to_dataset(task)
    if not dataset:
        return adapter_path or ""
    model_key = _model_name_to_residue_key(model_name)
    if model_key not in _RESIDUE_MODEL_KEYS:
        return adapter_path or ""
    folder = _model_key_to_adapter_folder(model_key)
    return os.path.join(ckpt_base, dataset, folder)


def _task_to_dataset(task: str) -> Optional[str]:
    """Map task (通俗命名) to first dataset name. e.g. Optimal Temperature -> DeepET_Topt."""
    const = _load_constant()
    mapping = const.get("dataset_mapping_function", {})
    names = mapping.get(task) if isinstance(mapping.get(task), list) else ([mapping.get(task)] if mapping.get(task) else None)
    return names[0] if names else None


def _model_name_to_key(model_name: str) -> str:
    """Normalize model_name (e.g. Ankh-large, ESM2-650M) to module key (ankh, esm2)."""
    const = _load_constant()
    func = const.get("model_mapping_function", {})
    key = func.get(model_name) or func.get(model_name.strip())
    if key:
        return key.lower().strip()
    # Already a key
    mn = model_name.lower().strip()
    return mn if mn in _MODEL_KEYS else mn


def _model_key_to_adapter_folder(model_key: str) -> str:
    """e.g. ankh -> ankh-large from model_adapter_mapping_function."""
    const = _load_constant()
    func = const.get("model_adapter_mapping_function", {})
    return func.get(model_key, model_key)


def _resolve_adapter_path(task: str, model_name: str, adapter_path: Optional[str], ckpt_base: str) -> str:
    """Resolve adapter path: if adapter_path given use it; else ckpt_base/{dataset_name}/{adapter_folder}."""
    if adapter_path and Path(adapter_path).exists():
        return adapter_path
    dataset = _task_to_dataset(task)
    if not dataset:
        return adapter_path or ""
    model_key = _model_name_to_key(model_name)
    if model_key not in _MODEL_KEYS:
        return adapter_path or ""
    folder = _model_key_to_adapter_folder(model_key)
    return os.path.join(ckpt_base, dataset, folder)


def _error_response(error_type: str, message: str, suggestion: Optional[str] = None) -> str:
    out: Dict[str, Any] = {
        "status": "error",
        "error": {"type": error_type, "message": message},
        "file_info": None,
    }
    if suggestion:
        out["error"]["suggestion"] = suggestion
    return json.dumps(out, ensure_ascii=False)


def _download_success_response(
    file_path: str,
    content_preview: Optional[str] = None,
    biological_metadata: Optional[Dict[str, Any]] = None,
    compute_time_ms: int = 0,
    source: str = _SOURCE,
) -> str:
    path = Path(file_path)
    file_size = path.stat().st_size if path.exists() else 0
    fmt = path.suffix.lstrip(".").lower() or "csv"
    out: Dict[str, Any] = {
        "status": "success",
        "file_info": {
            "file_path": str(path.resolve()) if path.exists() else file_path,
            "file_name": path.name,
            "file_size": file_size,
            "format": fmt,
        },
        "content_preview": (content_preview or "")[:_PREVIEW_LEN],
        "biological_metadata": biological_metadata or {},
        "execution_context": {"compute_time_ms": compute_time_ms, "source": source},
    }
    return json.dumps(out, ensure_ascii=False)


def _get_module(model_key: str):
    """Import parse_fasta, load_model_and_tokenizer, predict and tokenizer input style for the given model key (protbert/prott5/ankh/esm2)."""
    model_key = model_key.lower().strip()
    if model_key not in _GET_MODULE_KEYS:
        raise ValueError(f"model_name must map to one of {_GET_MODULE_KEYS}, got {model_key!r}")
    # Use absolute import so this works when script is run as __main__ (e.g. python src/tools/predict/finetuned/fintuned_operations.py)
    if model_key == "protbert":
        from src.tools.predict.finetuned import protbert as mod
        def tokenize_for_model(seq: str, tokenizer):
            formatted = " ".join(list(seq))
            return tokenizer([formatted], return_tensors="pt", padding=True, truncation=True)
    else:
        if model_key == "prott5":
            from src.tools.predict.finetuned import prott5 as mod
        elif model_key == "ankh":
            from src.tools.predict.finetuned import ankh as mod
        else:
            from src.tools.predict.finetuned import esm2 as mod
        def tokenize_for_model(seq: str, tokenizer):
            return tokenizer([seq], return_tensors="pt", padding=True, truncation=True)
    return mod, tokenize_for_model


def predict_protein_function(
    fasta_file: str,
    task: str,
    model_name: str = "Ankh-large",
    adapter_path: Optional[str] = None,
    ckpt_base: str = _CKPT_BASE,
    output_file: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> str:
    """
    Run finetuned protein function prediction. task = 通俗命名 (e.g. Solubility, Optimal Temperature).
    adapter_path can be omitted: then built as ckpt_base/{dataset_name}/{model_adapter_folder}, e.g. ckpt/DeepET_Topt/ankh-large.
    """
    t0 = time.perf_counter()
    if not fasta_file or not Path(fasta_file).exists():
        return _error_response("ValidationError", f"File not found: {fasta_file}", suggestion="Check fasta_file path.")
    choices = _get_task_choices()
    if task not in choices:
        return _error_response("ValidationError", f"task must be one of: {choices}.", suggestion="Use a task from constant.json dataset_mapping_function.")
    model_key = _model_name_to_key(model_name)
    if model_key not in _MODEL_KEYS:
        return _error_response("ValidationError", f"model_name must map to one of {_MODEL_KEYS}.", suggestion="Use e.g. Ankh-large, ESM2-650M, ProtBert, ProtT5-xl-uniref50.")
    resolved_adapter = _resolve_adapter_path(task, model_name, adapter_path, ckpt_base)
    if not resolved_adapter or not Path(resolved_adapter).exists():
        return _error_response("ValidationError", f"Adapter path not found: {resolved_adapter}.", suggestion="Set adapter_path or ensure ckpt_base/task/model dir exists (e.g. ckpt/DeepET_Topt/ankh-large).")

    try:
        mod, tokenize_for_model = _get_module(model_key)
        parse_fasta = mod.parse_fasta
        load_model_and_tokenizer = mod.load_model_and_tokenizer
        predict = mod.predict

        config_path = os.path.join(resolved_adapter, "lr5e-4_bt12k_ga8.json")
        default_out = f"predict_protein_function_{task.replace(' ', '_')}_{model_key}.csv"
        out_path = output_file or (os.path.join(out_dir, default_out) if out_dir else default_out)
        args = Namespace(adapter_path=resolved_adapter, output_csv=out_path)
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                for k, v in config.items():
                    setattr(args, k, v)
        if not hasattr(args, "problem_type"):
            args.problem_type = "single_label_classification"

        model, plm_model, tokenizer, device = load_model_and_tokenizer(args)
        sequences_to_predict = parse_fasta(fasta_file)
        if not sequences_to_predict:
            return _error_response("ComputeError", "No sequences found in FASTA.", suggestion="Check FASTA format.")

        all_results: List[Dict[str, Any]] = []
        for header, sequence in sequences_to_predict:
            aa_inputs = tokenize_for_model(sequence, tokenizer)
            data_dict = {
                "aa_seq_input_ids": aa_inputs["input_ids"],
                "aa_seq_attention_mask": aa_inputs["attention_mask"],
            }
            result_data = predict(model, data_dict, device, args, plm_model)
            output_row = {"header": header, "sequence": sequence, **result_data}
            all_results.append(output_row)

        content_str = json.dumps(all_results, ensure_ascii=False, indent=2)
        meta: Dict[str, Any] = {
            "fasta_file": fasta_file,
            "task": task,
            "dataset_name": _task_to_dataset(task),
            "adapter_path": resolved_adapter,
            "model_name": model_name,
            "sequence_count": len(all_results),
        }

        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
            if all_results:
                fieldnames = list(all_results[0].keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in all_results:
                    row_serialized = {k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()}
                    writer.writerow(row_serialized)

        return _download_success_response(
            out_path,
            content_preview=content_str[:_PREVIEW_LEN],
            biological_metadata=meta,
            compute_time_ms=int((time.perf_counter() - t0) * 1000),
            source=_SOURCE,
        )
    except Exception as e:
        return _error_response(
            "ComputeError",
            str(e),
            suggestion="Check fasta_file, task, model_name, adapter_path or ckpt_base (e.g. ckpt/DeepET_Topt/ankh-large), and deps (torch, transformers).",
        )


def predict_residue_function(
    fasta_file: str,
    task: str,
    model_name: str = "ESM2-650M",
    adapter_path: Optional[str] = None,
    ckpt_base: str = _CKPT_BASE,
    output_file: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> str:
    """
    Run finetuned residue-level function prediction (Activity Site, Binding Site, Conserved Site, Motif).
    task = 通俗命名 from residue_mapping_function; adapter_path default ckpt_base/{residue_dataset}/{model_folder}.
    """
    t0 = time.perf_counter()
    if not fasta_file or not Path(fasta_file).exists():
        return _error_response("ValidationError", f"File not found: {fasta_file}", suggestion="Check fasta_file path.")
    residue_choices = _get_residue_task_choices()
    if task not in residue_choices:
        return _error_response("ValidationError", f"task must be one of: {residue_choices}.", suggestion="Use a task from constant.json residue_mapping_function.")
    model_key = _model_name_to_residue_key(model_name)
    if model_key not in _RESIDUE_MODEL_KEYS:
        return _error_response("ValidationError", f"model_name must map to one of {_RESIDUE_MODEL_KEYS}.", suggestion="Use e.g. ESM2-650M, Ankh-large, ProtT5-xl-uniref50 (see model_residue_mapping_function).")
    resolved_adapter = _resolve_residue_adapter_path(task, model_name, adapter_path, ckpt_base)
    if not resolved_adapter or not Path(resolved_adapter).exists():
        return _error_response("ValidationError", f"Adapter path not found: {resolved_adapter}.", suggestion="Set adapter_path or ensure ckpt_base/residue_dataset/model dir exists (e.g. ckpt/VenusX_Res_Act_MP90/ankh-large).")

    try:
        mod, tokenize_for_model = _get_module(model_key)
        parse_fasta = mod.parse_fasta
        load_model_and_tokenizer = mod.load_model_and_tokenizer
        predict = mod.predict

        config_path = os.path.join(resolved_adapter, "lr5e-4_bt12k_ga8.json")
        default_out = f"predict_residue_function_{task.replace(' ', '_')}_{model_key}.csv"
        out_path = output_file or (os.path.join(out_dir, default_out) if out_dir else default_out)
        args = Namespace(adapter_path=resolved_adapter, output_csv=out_path)
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                for k, v in config.items():
                    setattr(args, k, v)
        if not hasattr(args, "problem_type"):
            args.problem_type = "residue_single_label_classification"

        model, plm_model, tokenizer, device = load_model_and_tokenizer(args)
        sequences_to_predict = parse_fasta(fasta_file)
        if not sequences_to_predict:
            return _error_response("ComputeError", "No sequences found in FASTA.", suggestion="Check FASTA format.")

        all_results: List[Dict[str, Any]] = []
        for header, sequence in sequences_to_predict:
            aa_inputs = tokenize_for_model(sequence, tokenizer)
            data_dict = {
                "aa_seq_input_ids": aa_inputs["input_ids"],
                "aa_seq_attention_mask": aa_inputs["attention_mask"],
            }
            result_data = predict(model, data_dict, device, args, plm_model)
            output_row = {"header": header, "sequence": sequence, **result_data}
            all_results.append(output_row)

        content_str = json.dumps(all_results, ensure_ascii=False, indent=2)
        meta: Dict[str, Any] = {
            "fasta_file": fasta_file,
            "task": task,
            "dataset_name": _residue_task_to_dataset(task),
            "adapter_path": resolved_adapter,
            "model_name": model_name,
            "sequence_count": len(all_results),
        }

        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
            if all_results:
                fieldnames = list(all_results[0].keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in all_results:
                    row_serialized = {k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()}
                    writer.writerow(row_serialized)

        return _download_success_response(
            out_path,
            content_preview=content_str[:_PREVIEW_LEN],
            biological_metadata=meta,
            compute_time_ms=int((time.perf_counter() - t0) * 1000),
            source=_SOURCE,
        )
    except Exception as e:
        return _error_response(
            "ComputeError",
            str(e),
            suggestion="Check fasta_file, task (residue), model_name, adapter_path or ckpt_base (e.g. ckpt/VenusX_Res_Act_MP90/ankh-large), and deps.",
        )


def _slug(s: str) -> str:
    """Safe filename slug from task or model name."""
    return s.replace(" ", "_").replace("-", "_").replace("/", "_")


if __name__ == "__main__":
    import argparse

    task_choices = _get_task_choices()
    model_choices = _get_model_choices()
    residue_task_choices = _get_residue_task_choices()
    residue_model_choices = _get_residue_model_choices()
    parser = argparse.ArgumentParser(
        description="Finetuned operations: predict_protein_function + predict_residue_function. --test runs all (protein + residue) tasks × models.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test all protein + residue (task, model) pairs; write one result file per pair under out_dir.",
    )
    parser.add_argument("--fasta_file", type=str, default="example/database/P60002.fasta", help="FASTA file for prediction tests. If omitted, only resolve adapter paths and write skip entries.")
    parser.add_argument("--task", type=str, default=None, choices=task_choices if task_choices else None, help="If set with --model_name, run only this protein (task, model) pair.")
    parser.add_argument("--model_name", type=str, default=None, choices=model_choices if model_choices else None, help="If set with --task, run only this protein pair.")
    parser.add_argument("--residue_task", type=str, default=None, choices=residue_task_choices if residue_task_choices else None, help="If set with --residue_model, run only this residue (task, model) pair.")
    parser.add_argument("--residue_model", type=str, default=None, choices=residue_model_choices if residue_model_choices else None, help="If set with --residue_task, run only this residue pair.")
    parser.add_argument("--adapter_path", type=str, default=None, help="Adapter dir (optional). If omitted, use ckpt_base/{dataset}/{model_folder}.")
    parser.add_argument("--ckpt_base", type=str, default=_CKPT_BASE, help="Base dir for adapter path. Default: ckpt.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="example/predict/finetuned",
        help="Output directory for test result JSONs. Default: example/predict/finetuned.",
    )
    args = parser.parse_args()

    if not args.test:
        print("Use --test to run operations tests (all tasks × all models).")
        exit(0)

    os.makedirs(args.out_dir, exist_ok=True)
    fasta_file = args.fasta_file
    out_dir = args.out_dir

    def _print_result(name: str, res: str) -> None:
        obj = json.loads(res)
        print(f"  {name}: status={obj.get('status')}")
        if obj.get("status") == "success":
            if obj.get("file_info"):
                print(f"    file_info: {obj['file_info']}")
            if obj.get("biological_metadata"):
                print(f"    biological_metadata: {obj['biological_metadata']}")
        if obj.get("status") == "error" and obj.get("error"):
            print(f"    error: {obj['error']}")

    # Single protein (task, model) test
    if args.task and args.model_name:
        print(f"=== Single test (protein): task={args.task!r}, model_name={args.model_name!r} ===")
        res = predict_protein_function(
            fasta_file=fasta_file or "",
            task=args.task,
            model_name=args.model_name,
            adapter_path=args.adapter_path,
            ckpt_base=args.ckpt_base,
            out_dir=out_dir,
        )
        _print_result("predict_protein_function(...)", res)
        sample_path = os.path.join(out_dir, f"predict_protein_{_slug(args.task)}_{_slug(args.model_name)}.json")
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write(res)
        print(f"  saved to {sample_path}")
        print("Done.")
        exit(0)

    # Single residue (task, model) test
    if args.residue_task and args.residue_model:
        print(f"=== Single test (residue): task={args.residue_task!r}, model_name={args.residue_model!r} ===")
        res = predict_residue_function(
            fasta_file=fasta_file or "",
            task=args.residue_task,
            model_name=args.residue_model,
            adapter_path=args.adapter_path,
            ckpt_base=args.ckpt_base,
            out_dir=out_dir,
        )
        _print_result("predict_residue_function(...)", res)
        sample_path = os.path.join(out_dir, f"predict_residue_{_slug(args.residue_task)}_{_slug(args.residue_model)}.json")
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write(res)
        print(f"  saved to {sample_path}")
        print("Done.")
        exit(0)

    # Test all datasets × all models
    tasks = _get_task_choices()
    models = _get_model_choices()
    if not tasks or not models:
        print("No tasks or models from constant.json (dataset_mapping_function / model_mapping_function).")
        exit(1)

    print(f"=== Test all datasets ({len(tasks)}) × all models ({len(models)}) ===")
    print(f"  tasks: {tasks}")
    print(f"  models: {models}")
    print(f"  out_dir: {out_dir}")
    if fasta_file:
        print(f"  fasta_file: {fasta_file} (will run prediction where adapter exists)")
    else:
        print("  fasta_file: not set (will only resolve adapter paths and write skip/resolve results)")

    for task in tasks:
        for model_name in models:
            label = f"{_slug(task)}_{_slug(model_name)}"
            resolved = _resolve_adapter_path(task, model_name, args.adapter_path, args.ckpt_base)
            adapter_exists = resolved and Path(resolved).exists()

            if fasta_file and adapter_exists:
                res = predict_protein_function(
                    fasta_file=fasta_file,
                    task=task,
                    model_name=model_name,
                    adapter_path=args.adapter_path,
                    ckpt_base=args.ckpt_base,
                    output_file=os.path.join(out_dir, f"predict_protein_{label}.csv"),
                    out_dir=out_dir,
                )
                out_path = os.path.join(out_dir, f"predict_protein_{label}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(res)
                _print_result(f"[{task}] [{model_name}]", res)
            else:
                if not adapter_exists:
                    reason = "adapter_path not found"
                elif not fasta_file:
                    reason = "dry run (no fasta_file)"
                else:
                    reason = "skipped"
                skip = {
                    "status": "skipped",
                    "task": task,
                    "model_name": model_name,
                    "resolved_adapter_path": resolved,
                    "adapter_exists": adapter_exists,
                    "reason": reason,
                }
                out_path = os.path.join(out_dir, f"predict_protein_{label}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(skip, f, ensure_ascii=False, indent=2)
                print(f"  [protein] [{task}] [{model_name}]: skipped — {skip['reason']}; resolved={resolved!r}")

    # Test all residue tasks × residue models
    residue_tasks = _get_residue_task_choices()
    residue_models = _get_residue_model_choices()
    if residue_tasks and residue_models:
        print(f"=== Test all residue tasks ({len(residue_tasks)}) × residue models ({len(residue_models)}) ===")
        print(f"  residue_tasks: {residue_tasks}")
        print(f"  residue_models: {residue_models}")
        for task in residue_tasks:
            for model_name in residue_models:
                label = f"residue_{_slug(task)}_{_slug(model_name)}"
                resolved = _resolve_residue_adapter_path(task, model_name, args.adapter_path, args.ckpt_base)
                adapter_exists = resolved and Path(resolved).exists()

                if fasta_file and adapter_exists:
                    res = predict_residue_function(
                        fasta_file=fasta_file,
                        task=task,
                        model_name=model_name,
                        adapter_path=args.adapter_path,
                        ckpt_base=args.ckpt_base,
                        output_file=os.path.join(out_dir, f"predict_residue_{_slug(task)}_{_slug(model_name)}.csv"),
                        out_dir=out_dir,
                    )
                    out_path = os.path.join(out_dir, f"predict_residue_{_slug(task)}_{_slug(model_name)}.json")
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(res)
                    _print_result(f"[residue] [{task}] [{model_name}]", res)
                else:
                    if not adapter_exists:
                        reason = "adapter_path not found"
                    elif not fasta_file:
                        reason = "dry run (no fasta_file)"
                    else:
                        reason = "skipped"
                    skip = {
                        "status": "skipped",
                        "task": task,
                        "model_name": model_name,
                        "resolved_adapter_path": resolved,
                        "adapter_exists": adapter_exists,
                        "reason": reason,
                    }
                    out_path = os.path.join(out_dir, f"predict_residue_{_slug(task)}_{_slug(model_name)}.json")
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(skip, f, ensure_ascii=False, indent=2)
                    print(f"  [residue] [{task}] [{model_name}]: skipped — {reason}; resolved={resolved!r}")

    print(f"Done. Output under {out_dir}")
