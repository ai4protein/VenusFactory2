"""Prediction runners for executing model predictions."""

import os
import sys
import subprocess
import time
import pandas as pd
from pathlib import Path
from typing import Tuple

from .constants import MODEL_MAPPING_ZERO_SHOT, PROTEIN_PROPERTIES_MAP_FUNCTION
from .common_utils import get_save_path


def run_zero_shot_prediction(model_type: str, model_name: str, file_path: str) -> Tuple[str, pd.DataFrame]:
    """Run zero-shot mutation prediction."""
    try:
        timestamp = str(int(time.time()))
        sequence_dir = get_save_path("Zero_shot_result")
        output_csv = sequence_dir / f"{model_type}_{timestamp}.csv"
        script_name = MODEL_MAPPING_ZERO_SHOT.get(model_name)
        
        if not script_name:
            return f"Error: Model '{model_name}' has no script.", pd.DataFrame()

        script_path = f"src/mutation/models/{script_name}.py"
        if not os.path.exists(script_path):
            return f"Script not found: {script_path}", pd.DataFrame()
        
        file_argument = "--pdb_file" if model_type == "structure" else "--fasta_file"
        cmd = [
            sys.executable, script_path, 
            file_argument, file_path, 
            "--output_csv", str(output_csv)
        ]
        
        subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True, 
            encoding='utf-8', 
            errors='ignore'
        )

        if os.path.exists(output_csv):
            df = pd.read_csv(output_csv)
            os.remove(output_csv)
            return "Prediction completed successfully!", df
        
        return "Prediction finished but no output file was created.", pd.DataFrame()
        
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr or e.stdout or "Unknown subprocess error"
        return f"Prediction script failed: {error_msg}", pd.DataFrame()
    except Exception as e:
        return f"An unexpected error occurred: {e}", pd.DataFrame()


def run_single_function_prediction(
    dataset: str,
    model: str,
    fasta_file: str,
    model_mapping: dict,
    adapter_mapping: dict,
    output_dir: Path
) -> pd.DataFrame:
    """Run prediction for a single dataset."""
    try:
        model_key = model_mapping.get(model)
        if not model_key:
            raise ValueError(f"Model key not found for {model}")
        
        adapter_key = adapter_mapping[model_key]
        script_path = Path("src") / "property" / f"{model_key}.py"
        adapter_path = Path("ckpt") / dataset / adapter_key
        output_file = output_dir / f"temp_{dataset}_{model}.csv"
        
        if not script_path.exists() or not adapter_path.exists():
            raise FileNotFoundError(f"Required files not found: Script={script_path}, Adapter={adapter_path}")
        
        cmd = [
            sys.executable, str(script_path),
            "--fasta_file", str(Path(fasta_file)),
            "--adapter_path", str(adapter_path),
            "--output_csv", str(output_file)
        ]
        
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8',
            errors='ignore'
        )
        
        if output_file.exists():
            df = pd.read_csv(output_file)
            df["Dataset"] = dataset
            os.remove(output_file)
            return df
        else:
            raise FileNotFoundError(f"Output file not created: {output_file}")
            
    except Exception as e:
        error_detail = str(e)
        return pd.DataFrame([{"Dataset": dataset, "header": "ERROR", "sequence": error_detail}])


def run_protein_properties_prediction(task_type: str, file_path: str) -> Tuple[str, str]:
    """Run protein properties prediction."""
    try:
        timestamp = str(int(time.time()))
        properties_dir = get_save_path("Protein_properties_result")
        output_json = properties_dir / f"{task_type.replace(' ', '_').replace('(', '').replace(')', '')}_{timestamp}.json"
        script_name = PROTEIN_PROPERTIES_MAP_FUNCTION.get(task_type)
        if not script_name:
           return "", f"Error: Task '{task_type}' is not allowed"
       
        script_path = f"src/property/{script_name}.py"
        if not os.path.exists(script_path):
           return "", f"Script not found: {script_path}"

        # Determine file argument based on file extension
        file_argument = "--fasta_file" if file_path.lower().endswith((".fasta", ".fa")) else "--pdb_file"
       
        cmd_save = [
           sys.executable, script_path,
           file_argument, file_path,
           "--chain_id", "A",
           "--output_file", str(output_json)
        ]
       
        # Run the command and capture output for debugging
        result = subprocess.run(
           cmd_save,
           capture_output=True,
           text=True,
           encoding="utf-8",
           errors="ignore"
        )

        # Check if the command failed
        if result.returncode != 0:
            error_msg = f"Script execution failed (return code: {result.returncode})"
            if result.stderr:
                error_msg += f"\nStderr: {result.stderr}"
            if result.stdout:
                error_msg += f"\nStdout: {result.stdout}"
            return "", error_msg

        # Check if output file was created
        if os.path.exists(output_json):
            return str(output_json), ""
        else:
            # Provide more detailed error information
            error_msg = "JSON output file was not generated"
            if result.stdout:
                error_msg += f"\nScript output: {result.stdout}"
            if result.stderr:
                error_msg += f"\nScript errors: {result.stderr}"
            return "", error_msg

    except Exception as e:
        return "", f"Error: {str(e)}"

