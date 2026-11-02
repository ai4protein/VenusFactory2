from mcp_tools.models.factory.runners.finetuned import run_cli as run_finetuned_cli
"""Report prediction calculations, independent of retired UI pages."""
import os
import sys
import subprocess
import time
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Generator, Optional, Tuple, Union
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from mcp_tools.common.constants import *
from mcp_tools.common.common_utils import get_save_path, get_project_root, ensure_project_ckpt
from mcp_tools.common.prediction_runners import run_zero_shot_prediction
from mcp_tools.common.label_mappers import map_labels_individual
from mcp_tools.common.file_handlers import extract_sequence_from_pdb
from mcp_tools.common.visualization import generate_plotly_heatmap
from mcp_tools.common.data_processors import prepare_top_residue_heatmap_data
PROJECT_ROOT = get_project_root()
def handle_functional_residue_prediction(content, selected_chain, current_file, sequence_state, original_content):
    """Handle functional residue prediction."""
    if not content.strip():
        return "Error: Please provide protein sequence or upload a file."

    # Prepare FASTA file for function prediction
    timestamp = str(int(time.time()))
    input_data_dir = get_save_path("Report", "Upload_data")
    output_data_dir = get_save_path("Report", "Results")

    is_pdb = content.strip().startswith('ATOM') or (current_file and current_file.endswith('.pdb'))

    if is_pdb:
        # Extract sequence from PDB
        if current_file and current_file.endswith('.pdb'):
            with open(current_file, 'r') as f:
                pdb_content = f.read()
        else:
            pdb_content = content

        sequence = extract_sequence_from_pdb(pdb_content)
        sequence_length = len(sequence)
        fasta_content = f">{selected_chain}\n{sequence}"
    else:
        if current_file and current_file.endswith(('.fasta', '.fa')):
            with open(current_file, 'r') as f:
                fasta_content = f.read()
        else:
            fasta_content = f">{selected_chain}\n{content}"

        sequence = content.strip()
        sequence_length = len(sequence)

    # Save FASTA file
    fasta_file = input_data_dir / f"residue_input_{timestamp}.fasta"
    with open(fasta_file, 'w') as f:
        f.write(fasta_content)

    # Run function prediction for all tasks
    all_residue_results = []
    for task, datasets in RESIDUE_MAPPING_FUNCTION.items():
        if not datasets:
            print(f"No datasets found for task: {task}")
            continue

        for dataset in datasets:
            try:
                model_name = "ESM2-650M"
                model_key = MODEL_MAPPING_FUNCTION.get(model_name)
                adapter_key = MODEL_ADAPTER_MAPPING_FUNCTION[model_key]
                script_path = PROJECT_ROOT / 'mcp_tools/models/factory/prediction' / f"{model_key}.py"
                adapter_path = ensure_project_ckpt(PROJECT_ROOT / "ckpt" / dataset / adapter_key)
                output_file = output_data_dir / f"temp_{dataset}_{model_name}_{timestamp}.csv"

                if script_path.exists() and adapter_path.exists():
                    cmd = [
                        sys.executable, str(script_path),
                        "--fasta_file", str(fasta_file),
                        "--adapter_path", str(adapter_path),
                        "--output_csv", str(output_file)
                    ]

                    run_finetuned_cli(cmd)

                    if output_file.exists():
                        df = pd.read_csv(output_file)
                        df["Task"] = task
                        df["Dataset"] = dataset
                        all_residue_results.append(df)
                        os.remove(output_file)

            except Exception as e:
                print(f"Function prediction failed for {dataset}: {e}")
                continue

    if not all_residue_results:
        return "Function prediction failed: No results generated."

    raw_residue_df = pd.concat(all_residue_results, ignore_index=True)
    result = "# Functional Residue  Results \n"

    # Process each prediction result
    for _, row in raw_residue_df.iterrows():
        dataset = row.get("Dataset", "Unknown")
        task = row.get("Task", "Unknown")
        result += f"## {task} \n"

        # Get sequence from the row
        row_sequence = row.get('sequence', sequence)
        if isinstance(row_sequence, str) and len(row_sequence) > 0:
            current_sequence = row_sequence
        else:
            current_sequence = sequence

        predictions_str = row.get('predicted_class', '')
        probabilities_str = row.get('probabilities', '')

        if predictions_str and predictions_str != '':
            try:
                # Parse predictions and probabilities similar to expand_residue_predictions
                import json
                predictions = json.loads(predictions_str) if isinstance(predictions_str, str) else predictions_str
                probabilities = json.loads(probabilities_str) if isinstance(probabilities_str, str) else probabilities_str

                # Handle nested list format
                if isinstance(predictions, list) and len(predictions) > 0 and isinstance(predictions[0], list):
                    predictions = predictions[0]
                if isinstance(probabilities, list) and len(probabilities) > 0 and isinstance(probabilities[0], list):
                    probabilities = probabilities[0]

                if isinstance(predictions, list) and isinstance(probabilities, list) and len(predictions) > 0:
                    # Process each residue
                    functional_residues = []
                    non_functional_residues = []
                    residue_details = []

                    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                        if i >= len(current_sequence):
                            break

                        residue = current_sequence[i]
                        position = i + 1  # 1-based indexing

                        # Handle different probability formats
                        if isinstance(prob, list):
                            max_prob = max(prob)
                            predicted_label = prob.index(max_prob)
                        else:
                            max_prob = prob
                            predicted_label = pred

                        residue_details.append({
                            'position': position,
                            'residue': residue,
                            'predicted_label': predicted_label,
                            'probability': max_prob
                        })

                        # Classify as functional or non-functional
                        if predicted_label == 1:  # Functional residue
                            functional_residues.append((position, residue, max_prob))
                        else:  # Non-functional residue
                            non_functional_residues.append((position, residue, max_prob))

                    # Generate report for this dataset
                    total_residues = len(residue_details)
                    functional_count = len(functional_residues)

                    result += f"- Functional residues: {functional_count} ({functional_count/total_residues:.1%})\n"
                    result += f"- Non-functional residues: {len(non_functional_residues)} ({len(non_functional_residues)/total_residues:.1%})\n\n"

                    if functional_residues:
                        result += "\n"

                        formatted_residues = [f"{res}{pos}" for pos, res, prob in functional_residues]
                        chunk_size = 20  # Show 20 residues per line
                        for i in range(0, len(formatted_residues), chunk_size):
                            chunk = formatted_residues[i:i + chunk_size]
                            result += ", ".join(chunk) + "\n"

                else:
                    result += "Prediction data format not recognized or empty.\n\n"

            except Exception as e:
                result += f"Error parsing prediction results: {str(e)}\n"
                result += f"Raw predictions: {str(predictions_str)[:100]}...\n"
                result += f"Raw probabilities: {str(probabilities_str)[:100]}...\n\n"
        else:
            result += "No prediction data available.\n\n"

    return result

def handle_individual_function_prediction(content, selected_chain, current_file, sequence_state, original_content):
    """Handle individual function prediction."""
    if not content.strip():
        return "Error: Please provide protein sequence or upload a file."

    timestamp = str(int(time.time()))
    input_data_dir = get_save_path("Report", "Upload_data")
    output_data_dir = get_save_path("Report", "Results")

    is_pdb = content.strip().startswith('ATOM') or (current_file and current_file.endswith('.pdb'))

    if is_pdb:
        # Extract sequence from PDB
        if current_file and current_file.endswith('.pdb'):
            with open(current_file, 'r') as f:
                pdb_content = f.read()
        else:
            pdb_content = content

        sequence = extract_sequence_from_pdb(pdb_content)
        fasta_content = f">{selected_chain}\n{sequence}"
    else:
        if current_file and current_file.endswith(('.fasta', '.fa')):
            with open(current_file, 'r') as f:
                fasta_content = f.read()
        else:
            fasta_content = f">{selected_chain}\n{content}"

    # Save FASTA file
    fasta_file = input_data_dir / f"function_input_{timestamp}.fasta"
    with open(fasta_file, 'w') as f:
        f.write(fasta_content)

    # Run function prediction for all tasks
    all_function_results = []

    for task, datasets in DATASET_MAPPING_FUNCTION.items():
        dataset = datasets[0]
        try:
            model_name = "ESM2-650M"
            model_key = MODEL_MAPPING_FUNCTION.get(model_name)
            adapter_key = MODEL_ADAPTER_MAPPING_FUNCTION[model_key]
            script_path = PROJECT_ROOT / 'mcp_tools/models/factory/prediction' / f"{model_key}.py"
            adapter_path = ensure_project_ckpt(PROJECT_ROOT / "ckpt" / dataset / adapter_key)
            output_file = output_data_dir / f"temp_{dataset}_{model_name}_{timestamp}.csv"

            if script_path.exists() and adapter_path.exists():
                cmd = [
                    sys.executable, str(script_path),
                    "--fasta_file", str(fasta_file),
                    "--adapter_path", str(adapter_path),
                    "--output_csv", str(output_file)
                ]
                run_finetuned_cli(cmd)
                if output_file.exists():
                    df = pd.read_csv(output_file)
                    df["Dataset"] = dataset
                    all_function_results.append(df)
                    os.remove(output_file)
            else:
                print("script_path does not exist")

        except Exception as e:
            print(f"Function prediction failed for {dataset}: {e}")
            continue

    if not all_function_results:
        return "Function prediction failed: No results generated."

    # Process and format results
    raw_function_df = pd.concat(all_function_results, ignore_index=True)
    # Define regression tasks max-min values for denormalization
    REGRESSION_TASKS_FUNCTION_MAX_MIN = {
        "Stability": [40.1995166, 66.8968874],
        "Optimum temperature": [2, 120]
    }
    # Generate function prediction report in table format
    # Create table header
    result = ""
    result += "| 🔬 **Task** | 📊 **Result** | 🎯 **Confidence** |\n"
    result += "|:------------|:--------------|:------------------|\n"

    # Process each prediction result
    for _, row in raw_function_df.iterrows():
        dataset = row.get('Dataset', 'Unknown')
        current_task = DATASET_TO_TASK_MAP.get(dataset, '')

        # Map prediction values to text labels
        predicted_class = map_labels_individual(row, current_task, REGRESSION_TASKS_FUNCTION_MAX_MIN)

        # Get confidence score
        confidence_score = row.get('probabilities', 0.5)
        if isinstance(confidence_score, str) and confidence_score not in ['N/A', '']:
            try:
                if confidence_score.startswith('[') and confidence_score.endswith(']'):
                    prob_str = confidence_score.strip('[]')
                    probs = [float(x.strip()) for x in prob_str.split(',')]
                    confidence_score = max(probs)
                else:
                    confidence_score = float(confidence_score)
            except (ValueError, IndexError):
                confidence_score = 0.5

        if isinstance(confidence_score, (int, float)):
            confidence_val = confidence_score if confidence_score <= 1 else confidence_score/100
        else:
            confidence_val = 0.5

        # Determine task name and format result
        task_name = ""
        result_text = ""
        confidence_text = ""

        if dataset in ["DeepSol", "DeepSoluE", "ProtSolM"]:
            task_name = "**Solubility**"
            result_text = f"`{predicted_class.capitalize()}`"
            confidence_text = f"{confidence_val:.3f}"

        elif dataset in ["DeepLocBinary", "DeepLocMulti"]:
            task_name = "**Subcellular Localization**"
            result_text = f"`{predicted_class.capitalize()}`"
            confidence_text = f"{confidence_val:.3f}"

        elif dataset == "MetalIonBinding":
            task_name = "**Metal Ion Binding**"
            result_text = f"`{predicted_class.capitalize()}`"
            confidence_text = f"{confidence_val:.3f}"

        elif dataset == "Thermostability":
            task_name = "**Thermal Stability**"
            result_text = f"`{predicted_class}`"
            confidence_text = "**N/A**"  # Regression task

        elif dataset == "SortingSignal":
            task_name = "**Signal Peptide**"
            result_text = f"`{predicted_class.capitalize()}`"
            confidence_text = f"{confidence_val:.3f}"

        elif dataset == "DeepET_Topt":
            task_name = "**Optimal Temperature**"
            result_text = f"`{predicted_class}`"
            confidence_text = "**N/A**"  # Regression task

        # Add table row
        if task_name:  # Only add if we have a valid task
            result += f"| {task_name} | {result_text} | {confidence_text} |\n"


    return result

def handle_individual_mutation_prediction(content, selected_chain, current_file, sequence_state, original_content):
    """Handle individual mutation prediction."""
    if not content.strip():
        return "Error: Please provide protein sequence or upload a file."

    timestamp = str(int(time.time()))
    input_data_dir = get_save_path("Report", "Upload_data")

    is_pdb = content.strip().startswith('ATOM') or (current_file and current_file.endswith('.pdb'))

    if is_pdb:
        input_file = input_data_dir / f"input_pdb_{timestamp}.pdb"
        if current_file and current_file.endswith('.pdb'):
            with open(current_file, 'r') as f:
                pdb_content = f.read()
        else:
            pdb_content = content

        with open(input_file, 'w') as f:
            f.write(pdb_content)

        model_name = "ESM-IF1"
        model_type = "structure"
    else:
        input_file = input_data_dir / f"input_fasta_{timestamp}.fasta"
        if current_file and current_file.endswith(('.fasta', '.fa')):
            with open(current_file, 'r') as f:
                fasta_content = f.read()
        else:
            fasta_content = f">{selected_chain}\n{content}"

        with open(input_file, 'w') as f:
            f.write(fasta_content)

        model_name = "ESM2-650M"
        model_type = "sequence"

    try:
        status, raw_df = run_zero_shot_prediction(model_type, model_name, str(input_file))

        if raw_df.empty:
            return f"Mutation prediction failed: {status}"

        top_20_df = raw_df.head(20)
        mutation_count = len(raw_df)

        result = f"# 📊 Mutation Prediction Results\n\n"
        result += f"**Total mutations analyzed: {mutation_count}**\n\n"

        # Create side-by-side table
        result += "| 🏆 **Top 10 Mutations (Rank 1-10)** | 🥈 **Next 10 Mutations (Rank 11-20)** |\n"
        result += "|:-------------------------------------|:---------------------------------------|\n"

        # Split data
        top_10_df = top_20_df.head(10)
        next_10_df = top_20_df.tail(10) if len(top_20_df) > 10 else None

        # Create rows for side-by-side display
        max_rows = 10
        for i in range(max_rows):
            # Left column (Top 10)
            if i < len(top_10_df):
                left_row = top_10_df.iloc[i]
                left_mutant = left_row['mutant']
                left_score = left_row['esm2_score']

                # Parse mutation
                match = re.match(r'([A-Z])(\d+)([A-Z])', left_mutant)
                if match:
                    wild_type, position, mutation = match.groups()
                    left_details = f"{wild_type}→{mutation} (pos {position})"
                else:
                    left_details = left_mutant

                # left_content = f"**#{i+1}** `{left_mutant}` <br/>Score: **{left_score:.4f}** <br/>{left_details}"
                left_content = f"**#{i+1}** `{left_mutant}`"
            else:
                left_content = ""

            # Right column (Next 10)
            if next_10_df is not None and i < len(next_10_df):
                right_row = next_10_df.iloc[i]
                right_mutant = right_row['mutant']
                right_score = right_row['esm2_score']

                # Parse mutation
                match = re.match(r'([A-Z])(\d+)([A-Z])', right_mutant)
                if match:
                    wild_type, position, mutation = match.groups()
                    right_details = f"{wild_type}→{mutation} (pos {position})"
                else:
                    right_details = right_mutant

                # right_content = f"**#{i+11}** `{right_mutant}` <br/>Score: **{right_score:.4f}** <br/>{right_details}"
                right_content = f"**#{i+11}** `{right_mutant}`"
            else:
                right_content = ""

            # Add table row
            result += f"| {left_content} | {right_content} |\n"

        return result

    except Exception as e:
        return f"Error in mutation prediction: {str(e)}"
