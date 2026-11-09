"""Explicit dataset configuration and queued Factory training/prediction."""
from mcp_tools.tools.capability_policy import guard_operation
import json
import os
import re
import time
import uuid
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from mcp_tools.common.common_utils import get_save_path
from mcp_tools.models.factory.training.session_training import workspace_root, scoped_path, validate_config, training_progress, stage_dataset
from mcp_tools.common.command import build_command_list, build_predict_command_list


# Shared backbone identifiers for configuration validation.
CONSTANT_PATH = str(Path(__file__).resolve().parents[4] / "src/constant.json")
if os.path.exists(CONSTANT_PATH):
    with open(CONSTANT_PATH, "r", encoding="utf-8") as f:
        CONSTANT = json.load(f)
        PLM_MODELS = CONSTANT.get("plm_models", {})
else:
    PLM_MODELS = {}


def detect_sequence_and_label_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """Detect the most likely amino acid sequence and label columns in a DataFrame."""
    # Initialize with default column names
    aa_seq_column = 'aa_seq'
    label_column = 'label'

    # Check if the default columns exist
    if 'aa_seq' in df.columns and 'label' in df.columns:
        return 'aa_seq', 'label'

    # Look for common sequence column names
    sequence_column_candidates = ['sequence', 'protein_sequence', 'aa_sequence', 'amino_acid_sequence', 'seq', 'protein_seq']
    for col in sequence_column_candidates:
        if col in df.columns:
            aa_seq_column = col
            break

    # If still not found, try to identify by content (amino acid sequences)
    if aa_seq_column == 'aa_seq' and 'aa_seq' not in df.columns:
        for col in df.columns:
            # Skip if column has numeric data
            if pd.api.types.is_numeric_dtype(df[col]):
                continue

            # Check if column contains amino acid sequences
            sample = df[col].dropna().astype(str).iloc[0] if not df[col].empty else ""
            if len(sample) > 10 and set(sample.upper()).issubset(set("ACDEFGHIKLMNPQRSTVWY")):
                aa_seq_column = col
                break

    # Look for common label column names
    label_column_candidates = ['label', 'target', 'class', 'y', 'output', 'property', 'value']
    for col in label_column_candidates:
        if col in df.columns:
            label_column = col
            break

    # If still not found, use the first non-sequence column that's not the sequence
    if label_column == 'label' and 'label' not in df.columns:
        for col in df.columns:
            if col != aa_seq_column:
                label_column = col
                break

    return aa_seq_column, label_column

def download_and_process_huggingface_dataset(dataset_path: str) -> Tuple[pd.DataFrame, str]:
    """Download and process a dataset from Hugging Face.

    Accepts:
      * ``user/name`` — direct HF dataset slug
      * VenusFactory-bundled metadata JSON (e.g. ``data/DeepSol/DeepSol_HF.json``)
        which contains ``{"dataset": "AI4Protein/DeepSol", ...}`` — the slug is
        extracted and forwarded to ``load_dataset``.
      * local dataset directory — passed through to ``load_dataset`` as-is.
    """
    try:
        from datasets import load_dataset

        resolved_slug = dataset_path

        # VenusFactory bundled HF metadata JSON: parse and pull out the slug.
        if dataset_path.endswith(".json") and os.path.exists(dataset_path):
            try:
                with open(dataset_path, "r", encoding="utf-8") as _f:
                    _meta = json.load(_f)
                slug = _meta.get("dataset") if isinstance(_meta, dict) else None
                if slug and isinstance(slug, str) and "/" in slug:
                    resolved_slug = slug
            except Exception:
                pass

        # Check if dataset_path is a valid Hugging Face dataset path
        if '/' in resolved_slug:
            # It's a Hugging Face dataset path like 'username/dataset_name'
            dataset = load_dataset(resolved_slug)
        else:
            # It might be a local path or a built-in dataset
            dataset = load_dataset(resolved_slug)

        # Convert to DataFrame - typically the first split is 'train'
        if 'train' in dataset:
            df = dataset['train'].to_pandas()
        else:
            # If no 'train' split, use the first available split
            first_split = list(dataset.keys())[0]
            df = dataset[first_split].to_pandas()

        # Save to a temporary CSV file
        temp_dir = get_save_path("MCP_Server", "TempDatasets")
        temp_csv_path = temp_dir / f"hf_dataset_{uuid.uuid4().hex[:8]}.csv"
        df.to_csv(temp_csv_path, index=False)

        return df, str(temp_csv_path)
    except Exception as e:
        raise ValueError(f"Error downloading or processing Hugging Face dataset: {str(e)}")

def process_csv_and_generate_config(csv_file: Optional[str] = None, valid_csv_file: Optional[str] = None,
                                   test_csv_file: Optional[str] = None, output_name: str = "custom_training_config",
                                   dataset_path: Optional[str] = None, user_overrides: Optional[Dict] = None,
                                   user_requirements: Optional[str] = None) -> str:
    try:
        import math
        for value in (csv_file, valid_csv_file, test_csv_file):
            if value: scoped_path(value)
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", output_name):
            raise ValueError("Use a simple configuration name")

        if user_requirements:
            raise ValueError('Convert user_requirements to explicit parameters (for example plm_model, num_epochs and learning_rate); this tool does not invoke an LLM')
        if user_overrides is not None and not isinstance(user_overrides, dict):
            raise ValueError('parameters must be an object of explicit training settings')
        # Handle Hugging Face dataset if provided
        if dataset_path and not csv_file:
            df, csv_file = download_and_process_huggingface_dataset(dataset_path)
        else:
            # Read CSV file
            df = pd.read_csv(csv_file)

        # Detect sequence and label columns
        aa_seq_column, label_column = detect_sequence_and_label_columns(df)

        # Check if the detected columns exist
        if aa_seq_column not in df.columns or label_column not in df.columns:
            return json.dumps({
                "success": False,
                "error": f"Could not identify valid sequence and label columns in the dataset. Please ensure your data has protein sequences and labels."
            }, ensure_ascii=False)

        # Validate additional files if provided
        valid_samples = 0
        test_samples = 0
        if valid_csv_file:
            try:
                valid_df = pd.read_csv(valid_csv_file)
                valid_samples = len(valid_df)
            except Exception as e:
                return json.dumps({
                    "success": False,
                    "error": f"Error reading validation file: {str(e)}"
                }, ensure_ascii=False)

        if test_csv_file:
            try:
                test_df = pd.read_csv(test_csv_file)
                test_samples = len(test_df)
            except Exception as e:
                return json.dumps({
                    "success": False,
                    "error": f"Error reading test file: {str(e)}"
                }, ensure_ascii=False)

        # Create a copy of the DataFrame with standardized column names for analysis
        analysis_df = df.copy()
        analysis_df.rename(columns={aa_seq_column: 'aa_seq', label_column: 'label'}, inplace=True)

        user_config = user_overrides or {}
        analysis = analyze_dataset_for_ai(analysis_df, valid_csv_file or test_csv_file)
        default_config = get_default_config(analysis)
        if not isinstance(user_config, dict) or set(user_config)-set(default_config):
            raise ValueError('Use documented training parameters: '+', '.join(sorted(default_config)))
        final_params = merge_configs(user_config, {}, default_config)
        if final_params['problem_type'] != analysis['data_type']:
            raise ValueError('The requested problem_type does not match the dataset labels')
        for key in ['num_epochs','batch_size','max_seq_len','gradient_accumulation_steps']:
            if type(final_params[key]) is not int or final_params[key] < 1:
                raise ValueError(key+' must be a positive integer')
        for key in ['patience','warmup_steps','num_workers']:
            if type(final_params[key]) is not int or final_params[key] < 0:
                raise ValueError(key+' must be a non-negative integer')
        for key in ['learning_rate','max_grad_norm']:
            if type(final_params[key]) not in (int,float) or not math.isfinite(final_params[key]) or final_params[key] <= 0:
                raise ValueError(key+' must be a finite positive number')

        # Add detected column names to the configuration
        final_params['sequence_column_name'] = aa_seq_column
        final_params['label_column_name'] = label_column

        config = create_comprehensive_config(csv_file, valid_csv_file, test_csv_file, final_params, analysis)
        if workspace_root():
            config["output_dir"] = str(get_save_path("training_pipeline", "models") / uuid.uuid4().hex[:12])
            config["dataset_custom"] = stage_dataset(csv_file, valid_csv_file, test_csv_file, aa_seq_column, label_column, final_params['problem_type'])
            for key, name in [('train_file','train'),('valid_file','validation'),('test_file','test')]:
                config[key] = str(Path(config['dataset_custom']) / (name+'.csv'))
        config = validate_config(config)
        config_dir = get_save_path("training_pipeline", "configs")
        timestamp = int(time.time())
        config_path = os.path.join(config_dir, f"{output_name}_{timestamp}.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        # Return JSON format for consistency with other tools
        result = {
            "success": True,
            "message": "Training configuration generated successfully!",
            "config_path": config_path,
            "config_name": f"{output_name}_{timestamp}.json",
            "resolved_parameters": final_params,
            "configuration_source": "explicit_parameters_and_defaults",
            "dataset_info": {
                "train_samples": len(df),
                "valid_samples": valid_samples,
                "test_samples": test_samples,
                "num_labels": config['num_labels'],
                "problem_type": final_params.get("problem_type", "unknown"),
                "detected_columns": {
                    "sequence_column": aa_seq_column,
                    "label_column": label_column
                },
                "data_source": "Hugging Face" if dataset_path else "CSV File"
            }
        }
        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Error processing CSV: {str(e)}"
        }, ensure_ascii=False)


def merge_configs(user_config: dict, ai_config: dict, default_config: dict) -> dict:
    merged = default_config.copy()
    merged.update(ai_config)
    merged.update(user_config)
    return merged

def analyze_dataset_for_ai(df: pd.DataFrame, test_csv_file: Optional[str] = None) -> dict:
    """Analyze dataset to provide context for AI parameter selection"""

    def classify_task_heuristic(df: pd.DataFrame) -> str:
        """Classify task type based on label characteristics using heuristic rules"""

        label_data = df['label']
        sample_labels = label_data.head(50).tolist()  # Sample for analysis
        is_residue_level = False

        for i in range(min(10, len(df))):
            label_str = str(df.iloc[i]['label'])
            seq_len = len(df.iloc[i]['aa_seq'])

            clean_label = label_str.replace(',', '').replace(' ', '').replace('[', '').replace(']', '')

            if len(clean_label) >= seq_len * 0.8:  # Allow some tolerance
                is_residue_level = True
                break

            if ',' in label_str and len(label_str.split(',')) >= seq_len * 0.8:
                is_residue_level = True
                break

        is_regression = False

        for label in sample_labels:
            label_str = str(label)

            if is_residue_level:
                # For residue-level, parse the sequence of values
                if ',' in label_str:
                    values = label_str.replace('[', '').replace(']', '').split(',')
                else:
                    values = list(label_str.replace('[', '').replace(']', ''))

                # Check if values are continuous (floats)
                try:
                    float_values = [float(v.strip()) for v in values if v.strip()]
                    # If we have decimal numbers, it's regression
                    if any('.' in str(v) for v in values if v.strip()):
                        is_regression = True
                        break
                    # If values have wide range, might be regression
                    if len(float_values) > 0 and (max(float_values) - min(float_values) > 10):
                        is_regression = True
                        break
                except ValueError:
                    # If can't convert to float, it's classification
                    continue
            else:
                # For protein-level, check the single label value
                try:
                    float_val = float(label_str)
                    # If it's a decimal number, it's regression
                    if '.' in label_str:
                        is_regression = True
                        break
                    # If integer range is large, might be regression
                    if abs(float_val) > 10:
                        is_regression = True
                        break
                except ValueError:
                    # If can't convert to float, it's classification
                    continue

        # Step 3: For classification, check if it's multi-label
        is_multi_label = False
        if not is_regression and not is_residue_level:
            # Check for multi-label indicators in protein-level classification
            for label in sample_labels:
                label_str = str(label)
                if any(sep in label_str for sep in [',', ';', '|', '&', '+']):
                    is_multi_label = True
                    break
                words = label_str.split()
                if len(words) > 1 and not any(char.isdigit() for char in label_str):
                    is_multi_label = True
                    break

        # Step 4: Return the classification
        if is_residue_level:
            if is_regression:
                return "residue_regression"
            else:
                return "residue_single_label_classification"
        else:
            if is_regression:
                return "regression"
            elif is_multi_label:
                return "multi_label_classification"
            else:
                return "single_label_classification"

    label_data = df['label']

    task_type = classify_task_heuristic(df)

    analysis = {
        "total_samples": int(len(df)),
        "unique_labels": int(df['label'].nunique()),
        "label_distribution": {str(k): int(v) for k, v in df['label'].value_counts().to_dict().items()},
        "sequence_stats": {
            "mean_length": float(df['aa_seq'].str.len().mean()),
            "min_length": int(df['aa_seq'].str.len().min()),
            "max_length": int(df['aa_seq'].str.len().max()),
            "std_length": float(df['aa_seq'].str.len().std())
        },
        "data_type": task_type,  # Heuristic-determined task type
        "class_balance": "balanced" if df['label'].value_counts().std() < df['label'].value_counts().mean() * 0.5 else "imbalanced"
    }

    if test_csv_file and os.path.exists(test_csv_file):
        test_df = pd.read_csv(test_csv_file)
        analysis["test_samples"] = int(len(test_df))
        analysis["has_test_set"] = True
    else:
        analysis["has_test_set"] = False

    return analysis

def convert_to_serializable(obj):
    """Convert pandas/numpy types to JSON serializable types"""
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj


def get_default_config(analysis: dict) -> dict:
    """Fallback default configuration"""
    is_regression = analysis['data_type'] == 'regression'
    return {
        "plm_model": "ESM2-8M",
        "problem_type": analysis['data_type'],
        "training_method": "freeze",
        "learning_rate": 5e-4,
        "num_epochs": 20,
        "batch_size": 16,
        "max_seq_len": min(512, int(analysis['sequence_stats']['max_length'] * 1.2)),
        "patience": 10,
        "pooling_method": "mean",
        "scheduler": None,
        "monitored_metrics": "spearman_corr" if is_regression else "accuracy",
        "monitored_strategy": "max",
        "gradient_accumulation_steps": 1,
        "warmup_steps": 0,
        "max_grad_norm": 1.0,
        "num_workers": 1
    }

def create_comprehensive_config(csv_file: str, valid_csv_file: Optional[str], test_csv_file: Optional[str], params: dict, analysis: dict) -> dict:
    """Create complete training configuration matching 1.py requirements with train/valid/test split"""
    is_regression = analysis['data_type'] == 'regression'
    dataset_directory = os.path.dirname(csv_file)

    # Determine metrics based on problem type
    if is_regression:
        metrics_list = ["mse", "spearman_corr"]
    else:
        metrics_list = ["accuracy", "mcc", "f1", "precision", "recall", "auroc"]

    # Get sequence and label column names from params or use defaults
    sequence_column_name = params.get("sequence_column_name", "aa_seq")
    label_column_name = params.get("label_column_name", "label")

    config = {
        # Dataset configuration
        "dataset_selection": "Custom",
        "dataset_custom": dataset_directory,
        "problem_type": params["problem_type"],
        "num_labels": 1 if is_regression else analysis['unique_labels'],
        "metrics": metrics_list,
        "sequence_column_name": sequence_column_name,
        "label_column_name": label_column_name,

        # Model and training method from final params
        "plm_model": params["plm_model"],
        "training_method": params["training_method"],
        "pooling_method": params["pooling_method"],

        # Batch mode configuration
        "batch_mode": "Batch Size Mode",
        "batch_size": int(params["batch_size"]),

        # Training parameters from final params
        "learning_rate": float(params["learning_rate"]),
        "num_epochs": int(params["num_epochs"]),
        "max_seq_len": int(params["max_seq_len"]),
        "patience": int(params["patience"]),

        # Advanced parameters
        "gradient_accumulation_steps": int(params.get("gradient_accumulation_steps", 1)),
        "warmup_steps": int(params.get("warmup_steps", 0)),
        "scheduler": params.get("scheduler"),
        "max_grad_norm": float(params.get("max_grad_norm", 1.0)),
        "num_workers": int(params.get("num_workers", 1)),

        # Monitoring
        "monitored_metrics": params["monitored_metrics"],
        "monitored_strategy": params["monitored_strategy"],

        # Output
        "output_model_name": f"model_{Path(csv_file).stem}.pt",
        "output_dir": f"ckpt/{Path(csv_file).stem}",

        "wandb_enabled": False,

        # LoRA parameters (with defaults)
        "lora_r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "lora_target_modules": "query,key,value",
    }

    if test_csv_file:
        config["test_file"] = test_csv_file

    # Final conversion to ensure everything is serializable
    return convert_to_serializable(config)

@guard_operation('train_protein_model')
def run_train_tool(config_path: str) -> str:
    """Train a protein language model using a configuration file. This tool executes the training process and streams the training logs."""
    try:

        config_path = scoped_path(config_path)
        if not os.path.exists(config_path):
            return json.dumps({
                "success": False,
                "error": f"Configuration file not found: {config_path}"
            }, ensure_ascii=False)

        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            config = validate_config(json.load(f))
        if workspace_root() and not config.get('train_file'):
            raise ValueError('Scoped training requires an explicit train_file CSV')

        # Extract only valid training parameters based on args.py
        train_config = {}

        # Model parameters - map PLM model name to full path
        if "plm_model" in config:
            plm_model = config["plm_model"]
            # If it's a short name (e.g., "ESM2-8M"), map it to full path
            if plm_model in PLM_MODELS:
                train_config["plm_model"] = PLM_MODELS[plm_model]
            else:
                # Already a full path or not in mapping, use as-is
                train_config["plm_model"] = plm_model
        if "pooling_method" in config:
            train_config["pooling_method"] = config["pooling_method"]
        if "training_method" in config:
            train_config["training_method"] = config["training_method"]

        # Dataset parameters
        for key in ('train_file', 'valid_file', 'test_file'):
            if config.get(key): train_config[key] = config[key]
        dataset_selection = config.get("dataset_selection", "Custom")
        if dataset_selection == "Pre-defined":
            if "dataset_config" in config:
                train_config["dataset_config"] = config["dataset_config"]
        else:
            # Custom dataset
            if "dataset_custom" in config:
                train_config["dataset"] = config["dataset_custom"]
            if "problem_type" in config:
                train_config["problem_type"] = config["problem_type"]
            if "num_labels" in config:
                train_config["num_labels"] = config["num_labels"]
            if "metrics" in config:
                metrics = config["metrics"]
                if isinstance(metrics, list):
                    train_config["metrics"] = ",".join(metrics)
                else:
                    train_config["metrics"] = metrics

        # Column names (for both predefined and custom)
        if "sequence_column_name" in config:
            train_config["sequence_column_name"] = config["sequence_column_name"]
        if "label_column_name" in config:
            train_config["label_column_name"] = config["label_column_name"]

        # Training parameters
        if "learning_rate" in config:
            train_config["learning_rate"] = config["learning_rate"]
        if "num_epochs" in config:
            train_config["num_epochs"] = config["num_epochs"]
        if "max_seq_len" in config:
            train_config["max_seq_len"] = config["max_seq_len"]
        if "gradient_accumulation_steps" in config:
            train_config["gradient_accumulation_steps"] = config["gradient_accumulation_steps"]
        if "warmup_steps" in config:
            train_config["warmup_steps"] = config["warmup_steps"]
        if "scheduler" in config:
            train_config["scheduler"] = config["scheduler"]
        if "patience" in config:
            train_config["patience"] = config["patience"]
        if "num_workers" in config:
            train_config["num_workers"] = config["num_workers"]
        if "max_grad_norm" in config:
            train_config["max_grad_norm"] = config["max_grad_norm"]

        # Monitored parameters
        if "monitored_metrics" in config:
            monitored = config["monitored_metrics"]
            if isinstance(monitored, list):
                train_config["monitor"] = monitored[0] if monitored else "accuracy"
            else:
                train_config["monitor"] = monitored
        if "monitored_strategy" in config:
            train_config["monitor_strategy"] = config["monitored_strategy"]

        # Batch parameters
        batch_mode = config.get("batch_mode", "Batch Size Mode")
        if batch_mode == "Batch Size Mode" and "batch_size" in config:
            train_config["batch_size"] = config["batch_size"]
        elif batch_mode == "Batch Token Mode" and "batch_token" in config:
            train_config["batch_token"] = config["batch_token"]

        # Structure sequence (for ses-adapter)
        training_method = config.get("training_method", "freeze")
        if training_method == "ses-adapter" and "structure_seq" in config:
            structure_seq = config["structure_seq"]
            if isinstance(structure_seq, list):
                train_config["structure_seq"] = ",".join(structure_seq)
            else:
                train_config["structure_seq"] = structure_seq

        # LoRA parameters (only for LoRA-based methods)
        if training_method in ["plm-lora", "plm-qlora", "plm-adalora", "plm-dora", "plm-ia3"]:
            if "lora_r" in config:
                train_config["lora_r"] = config["lora_r"]
            if "lora_alpha" in config:
                train_config["lora_alpha"] = config["lora_alpha"]
            if "lora_dropout" in config:
                train_config["lora_dropout"] = config["lora_dropout"]
            if "lora_target_modules" in config:
                lora_modules = config["lora_target_modules"]
                if isinstance(lora_modules, list):
                    train_config["lora_target_modules"] = lora_modules
                elif isinstance(lora_modules, str):
                    # Already in correct format
                    train_config["lora_target_modules"] = lora_modules.split(",")

        # Output parameters
        if "output_model_name" in config:
            train_config["output_model_name"] = config["output_model_name"]
        if "output_dir" in config:
            train_config["output_dir"] = config["output_dir"]

        # Wandb parameters
        if config.get("wandb_enabled", False):
            train_config["wandb"] = True
            if "wandb_project" in config:
                train_config["wandb_project"] = config["wandb_project"]
            if "wandb_entity" in config:
                train_config["wandb_entity"] = config["wandb_entity"]

        # Build training command
        cmd = build_command_list(train_config)
        cmd[:2] = [sys.executable, str(Path(__file__).resolve().parents[4] / "src/train.py")]
        cmd_str = " ".join(cmd)

        # Start training process
        from mcp_tools.models.execution import environment
        process = subprocess.Popen(
            cmd,
            env=environment({}),
            cwd=Path(__file__).resolve().parents[4],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Collect logs
        logs = []
        max_log_lines = 100  # Limit log output to avoid overwhelming the chat

        training_progress([], "running")
        for line in process.stdout:
            line = line.strip()
            if line:
                logs.append(line)
                # Keep only the last max_log_lines
                if len(logs) > max_log_lines:
                    logs.pop(0)
                training_progress(logs, "running")

        # Wait for process to complete
        return_code = process.wait()
        training_progress(logs, "completed" if return_code == 0 else "failed")

        if return_code == 0:
            # Extract output model path from config
            output_dir = config.get("output_dir", "ckpt/custom_model")
            output_model = config.get("output_model_name", "model.pt")
            model_path = os.path.join(output_dir, output_model)

            result = {
                "success": True,
                "message": "Model training completed successfully!",
                "model_path": model_path,
                "output_dir": output_dir,
                "command": cmd_str,
                "logs": "\n".join(logs[-20:])  # Return last 20 lines of logs
            }
            # Local-only auto-register into ckpt/user_trained for cross-session reuse.
            try:
                from mcp_tools.tools.runtime_tool_policy import is_local_mode
                from mcp_tools.models.factory.training.model_registry import register_trained_model

                if is_local_mode() and not workspace_root():
                    reg = register_trained_model(
                        config_path,
                        output_dir=output_dir,
                        model_path=model_path if os.path.isfile(model_path) else None,
                    )
                    result["registration"] = reg
                    if reg.get("success") and reg.get("registered", True):
                        result["registered_model_id"] = reg.get("model_id")
                        result["registered_config_path"] = reg.get("config_path")
                        result["message"] = (
                            "Model training completed successfully and registered under "
                            f"ckpt/user_trained/{reg.get('model_id')} for cross-session reuse."
                        )
                    elif reg.get("success"):
                        result["candidate_model_id"] = reg.get("model_id")
                        result["message"] = "Training completed; checkpoint staged for independent research validation."
            except Exception as reg_exc:
                result["registration"] = {
                    "success": False,
                    "error": f"Auto-register failed: {reg_exc}",
                }
        else:
            result = {
                "success": False,
                "error": f"Training failed with return code {return_code}",
                "command": cmd_str,
                "logs": "\n".join(logs[-20:])
            }

        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Training error: {str(e)}"
        }, ensure_ascii=False)


@guard_operation('protein_model_predict')
def run_predict_tool(config_path: str, sequence: Optional[str] = None, csv_file: Optional[str] = None) -> str:
    """Predict protein properties using a user trained model. Can perform single sequence prediction or batch prediction from CSV file."""
    try:
        if workspace_root(): config_path = scoped_path(config_path)
        # Accept registered model_id or filesystem config path.
        try:
            from mcp_tools.models.factory.training.model_registry import resolve_registered_config_path
            config_path = resolve_registered_config_path(config_path)
        except Exception as exc:
            from mcp_tools.tools.capability_policy import CapabilityDisabled
            if isinstance(exc, CapabilityDisabled): raise
            pass

        if not os.path.exists(config_path):
            return json.dumps({
                "success": False,
                "error": f"Configuration file not found: {config_path}"
            }, ensure_ascii=False)

        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            config = validate_config(json.load(f), prediction=True)
        if csv_file: csv_file = scoped_path(csv_file)

        # Determine prediction mode
        is_batch = csv_file is not None

        # Extract only prediction-relevant parameters
        # Map PLM model name to full path
        plm_model = config.get("plm_model", "")
        if plm_model in PLM_MODELS:
            plm_model = PLM_MODELS[plm_model]

        training_method = config.get("training_method", "freeze")

        predict_config = {
            "model_path": config.get("model_path", config.get("output_dir", "") + "/" + config.get("output_model_name", "model.pt")),
            "plm_model": plm_model,
            "eval_method": training_method,
            "pooling_method": config.get("pooling_method", "mean"),
            "problem_type": config.get("problem_type", "single_label_classification"),
            "num_labels": config.get("num_labels", 2),
            "max_seq_len": config.get("max_seq_len", 1024),
            "batch_size": config.get("batch_size", 16),
        }

        # CRITICAL: Only use structure_seq if training method is ses-adapter
        # Otherwise, the model won't have the required embedding layers
        if training_method == "ses-adapter" and "structure_seq" in config:
            structure_seq = config["structure_seq"]
            if isinstance(structure_seq, list):
                predict_config["structure_seq"] = ",".join(structure_seq)
            else:
                predict_config["structure_seq"] = structure_seq

        if is_batch:
            if not os.path.exists(csv_file):
                return json.dumps({
                    "success": False,
                    "error": f"CSV file not found: {csv_file}"
                }, ensure_ascii=False)

            # Set batch prediction parameters
            predict_config["input_file"] = csv_file
            predict_config["output_dir"] = os.path.dirname(csv_file)
            predict_config["output_file"] = "predictions.csv"

        elif sequence:
            # Create temporary file for single sequence
            temp_dir = tempfile.mkdtemp(dir=get_save_path("training_pipeline", "predictions") if workspace_root() else None)
            temp_csv = os.path.join(temp_dir, "temp_sequence.csv")

            # Create CSV with sequence
            df = pd.DataFrame({"aa_seq": [sequence]})
            df.to_csv(temp_csv, index=False)

            predict_config["input_file"] = temp_csv
            predict_config["output_dir"] = temp_dir
            predict_config["output_file"] = "predictions.csv"
        else:
            return json.dumps({
                "success": False,
                "error": "Either 'sequence' or 'csv_file' must be provided"
            }, ensure_ascii=False)

        # Build prediction command
        cmd = build_predict_command_list(predict_config, is_batch=True)
        cmd_str = " ".join(cmd)

        from mcp_tools.models.factory.runners.checkpoint import prepare
        from mcp_tools.models.execution import execute
        spec = prepare(cmd)
        cmd_str = " ".join(spec.argv)
        logs = execute(spec).splitlines()[-50:]
        return_code = 0

        if return_code == 0:
            # Try to read prediction results
            output_file = os.path.join(predict_config["output_dir"], predict_config["output_file"])

            result = {
                "success": True,
                "message": "Prediction completed successfully!",
                "output_file": output_file,
                "command": cmd_str,
                "logs": "\n".join(logs[-10:])
            }

            # Try to load and preview results
            if os.path.exists(output_file):
                try:
                    df = pd.read_csv(output_file)
                    result["preview"] = df.head(10).to_dict(orient='records')
                    result["total_predictions"] = len(df)
                except Exception as e:
                    result["preview_error"] = f"Could not load results: {str(e)}"
        else:
            result = {
                "success": False,
                "error": f"Prediction failed with return code {return_code}",
                "command": cmd_str,
                "logs": "\n".join(logs[-10:])
            }

        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Prediction error: {str(e)}"
        }, ensure_ascii=False)
