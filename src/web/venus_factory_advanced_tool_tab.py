import os
import sys
import subprocess
import time
import json
import logging
import logging
import shutil
import torch
import numpy as np
import gradio as gr
import pandas as pd
import plotly.graph_objects as go

from transformers import T5Tokenizer, T5EncoderModel
from dotenv import load_dotenv
from pathlib import Path
from gradio_molecule3d import Molecule3D
from typing import Dict, Any, List, Generator, Optional, Tuple
from web.utils.constants import *
from web.utils.common_utils import *
from web.utils.file_handlers import *
from web.utils.ai_helpers import *
from web.utils.data_processors import *
from web.utils.visualization import *
from web.utils.prediction_runners import *
from web.utils.venusmine import *
from web.utils.label_mappers import map_labels
from web.utils.ui_helpers import (
    create_progress_html,
    create_status_html,
    handle_paste_fasta_detect,
    handle_paste_pdb_detect,
    update_dataset_choices_fixed,
)
from web.venus_factory_quick_tool_tab import *

load_dotenv()


RCSB_REPS =  [
    {
      "model": 0,
      "chain": "",
      "resname": "",
      "style": "cartoon",
      "color": "chain",
      "around": 0,
      "byres": False,
    }
  ]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def truncate_sequence(seq):
    """Truncate sequence if it's a string longer than 30 characters."""
    if isinstance(seq, str) and len(seq) > 30:
        return seq[:]
    return seq


def format_confidence(row, task):
    """Format confidence score for display."""
    score = row["Confidence Score"]
    predicted_class = row["Predicted Class"]
    
    if isinstance(score, (float, int)) and score != 'N/A':
        return round(float(score), 2)
    elif isinstance(score, str) and score not in ['N/A', '']:
        try:
            if score.startswith('[') and score.endswith(']'):
                prob_str = score.strip('[]')
                probs = [float(x.strip()) for x in prob_str.split(',')]
                
                current_task = DATASET_TO_TASK_MAP.get(row.get('Dataset', ''), task)
                
                if current_task in REGRESSION_TASKS_FUNCTION:
                    return predicted_class
                else:
                    labels_key = ("DeepLocMulti" if row.get('Dataset') == "DeepLocMulti" 
                                 else "DeepLocBinary" if row.get('Dataset') == "DeepLocBinary" 
                                 else current_task)
                    labels = LABEL_MAPPING_FUNCTION.get(labels_key, [])
                    
                    if labels and predicted_class in labels:
                        pred_index = labels.index(predicted_class)
                        if 0 <= pred_index < len(probs):
                            return round(probs[pred_index], 2)
                    
                    return round(max(probs), 2)
            else:
                return round(float(score), 2)
        except (ValueError, IndexError):
            return score
    return score


def handle_venus_pdb_upload(file):
    """Handle PDB file upload for VenusMine viewer."""
    return file if file else None


# create_progress_html and create_status_html are now imported from utils.ui_helpers


def handle_mutation_prediction_advance(
    function_selection: str, 
    file_obj: Any, 
    enable_ai: bool, 
    ai_model: str, 
    user_api_key: str, 
    model_name:str,
    progress=gr.Progress()
) -> Generator:
    try:
        import requests
        requests.post("/api/stats/track", json={"module": "mutation_prediction"})
    except Exception:
        pass
    """Handle mutation prediction workflow."""
    if not file_obj or not function_selection:
        yield (
            "❌ Error: Function and file are required.", 
            None, None, gr.update(visible=False), None, 
            gr.update(visible=False), None, 
            "Please select a function and upload a file."
        )
        return

    if isinstance(file_obj, str):
        file_path = file_obj
    else:
        file_path = file_obj.name
    if file_path.lower().endswith(".pdb"):
        if model_name:
            model_type = "structure"
        else:
            model_name, model_type = "ESM-IF1", "structure"
    elif file_path.lower().endswith((".fasta", ".fa")):
        if model_name:
            model_type = "sequence"
        else:
            model_name, model_type = "ESM2-650M", "sequence"

        processed_file_path = process_fasta_file(file_path)
        if processed_file_path != file_path:
            file_path = processed_file_path
            yield (
                "⚠️ Multi-sequence FASTA detected. Using only the first sequence for prediction.",
                None, None, gr.update(visible=False), None,
                gr.update(visible=False), None,
                "Processing first sequence only..."
            )
    else:
        yield (
            "❌ Error: Unsupported file type.", 
            None, None, gr.update(visible=False), None, 
            gr.update(visible=False), None, 
            "Please upload a .fasta, .fa, or .pdb file."
        )
        return

    progress(0.1, desc="Running prediction...")
    yield (
        f"⏳ Running prediction...", 
        None, None, gr.update(visible=False), None, 
        gr.update(visible=False), None, 
        "Prediction in progress..."
    )
    
    status, raw_df = run_zero_shot_prediction(model_type, model_name, file_path)
    progress(0.7, desc="Processing results...")
    if raw_df.empty:
        yield (
            status, 
            go.Figure(layout={'title': 'No results generated'}), 
            pd.DataFrame(), gr.update(visible=False), None, 
            gr.update(visible=False), None, 
            "No results to analyze."
        )
        return
    
    score_col = next((c for c in raw_df.columns if 'score' in c.lower()), raw_df.columns[1])
    
    display_df = pd.DataFrame()
    display_df['Mutant'] = raw_df['mutant']
    display_df['Prediction Rank'] = range(1, len(raw_df) + 1)
    
    min_s, max_s = raw_df[score_col].min(), raw_df[score_col].max()
    if max_s == min_s:
        scaled_scores = pd.Series([0.0] * len(raw_df))
    else:
        scaled_scores = -1 + 2 * (raw_df[score_col] - min_s) / (max_s - min_s)
    display_df['Prediction Score'] = scaled_scores.round(2)

    df_for_heatmap = raw_df.copy()
    df_for_heatmap['Prediction Rank'] = range(1, len(df_for_heatmap) + 1)

    total_residues = get_total_residues_count(df_for_heatmap)
    data_tuple = prepare_top_residue_heatmap_data(df_for_heatmap)
    
    if data_tuple[0] is None:
        yield (
            status, 
            go.Figure(layout={'title': 'Score column not found'}), 
            display_df, gr.update(visible=False), None, 
            gr.update(visible=False), display_df, 
            "Score column not found."
        )
        return

    summary_fig = generate_plotly_heatmap(*data_tuple[:4])
    expert_analysis = "<div style='height: 300px; display: flex; align-items: center; justify-content: center; color: #666;'>Analysis will appear here once prediction is complete...</div>"

    ai_summary = "AI Analysis disabled. Enable in settings to generate a report."
    if enable_ai:
        progress(0.8, desc="Generating AI summary...")
        yield (
            f"✅ Prediction complete. 🤖 Generating AI summary...", 
            summary_fig, display_df, gr.update(visible=False), None, 
            gr.update(visible=total_residues > 20), display_df, 
            expert_analysis
        )
        
        api_key = get_api_key(ai_model, user_api_key)
        if not api_key:
            ai_summary = "❌ No API key found."
        else:
            ai_config = AIConfig(
                api_key, ai_model, 
                AI_MODELS[ai_model]["api_base"], 
                AI_MODELS[ai_model]["model"]
            )
            prompt = generate_mutation_ai_prompt(display_df, model_name, function_selection)
            ai_summary = call_ai_api(ai_config, prompt)
            expert_analysis = format_expert_response(ai_summary)
        progress(0.9, desc="Finalizing AI analysis...")
    else:
        progress(1.0, desc="Complete!")
    
    timestamp = str(int(time.time()))
    session_dir = get_save_path("Zero_Shot", "Result")
    csv_path = session_dir/ f"mut_res_{timestamp}.csv"
    heatmap_path = session_dir/ f"mut_map_{timestamp}.html"
    
    display_df.to_csv(csv_path, index=False)
    summary_fig.write_html(heatmap_path)
    
    files_to_tar = {
        str(csv_path): "prediction_results.csv", 
        str(heatmap_path): "prediction_heatmap.html"
    }

    if not ai_summary.startswith("❌") and not ai_summary.startswith("AI Analysis"):
        report_path = session_dir / f"ai_report_{timestamp}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(ai_summary)
        files_to_tar[str(report_path)] = "AI_Analysis_Report.md"

    tar_path = session_dir / f"pred_mut_{timestamp}.tar.gz"
    tar_path_str = create_tar_archive(files_to_tar, str(tar_path))

    final_status = status if not enable_ai else "✅ Prediction and AI analysis complete!"
    progress(1.0, desc="Complete!")
    yield (
        final_status, summary_fig, display_df, 
        gr.update(visible=True, value=tar_path_str), tar_path_str, 
        gr.update(visible=total_residues > 20), display_df, expert_analysis
    )

def handle_protein_function_prediction_chat(
    task: str,
    fasta_file: str,
    model_name: str,
    datasets: List[str],
    enable_ai: bool,
    ai_model: str,
    user_api_key: Optional[str] = None
) -> Generator:
    try:
        import requests
        requests.post("/api/stats/track", json={"module": "mutation_prediction"})
    except Exception:
        pass
    model = model_name if model_name else "ESM2-650M"
    final_datasets = datasets if datasets and len(datasets) > 0 else DATASET_MAPPING_FUNCTION.get(task, [])
    all_results_list = []
    timestamp = str(int(time.time()))
    function_dir = get_save_path("Protein_Function", "Result")

    for i, dataset in enumerate(final_datasets):
        try:
            model_key = MODEL_MAPPING_FUNCTION.get(model)
            if not model_key:
                raise ValueError(f"Model key not found for {model}")

            adapter_key = MODEL_ADAPTER_MAPPING_FUNCTION[model_key]
            script_path = Path("src") / "property" / f"{model_key}.py"
            adapter_path = Path("ckpt") / dataset / adapter_key
            output_file = function_dir/ f"temp_{dataset}_{model}_{timestamp}.csv"

            if not script_path.exists() or not adapter_path.exists():
                raise FileNotFoundError(f"Required files not found for dataset {dataset}")

            cmd = [sys.executable, str(script_path), "--fasta_file", str(Path(fasta_file.name)), "--adapter_path", str(adapter_path), "--output_csv", str(output_file)]
            subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8', errors='ignore')

            if output_file.exists():
                df = pd.read_csv(output_file)
                df["Dataset"] = dataset
                all_results_list.append(df)
                os.remove(output_file)
        except Exception as e:
            error_detail = e.stderr if isinstance(e, subprocess.CalledProcessError) else str(e)
            all_results_list.append(pd.DataFrame([{"Dataset": dataset, "header": "ERROR", "sequence": error_detail}]))

    if not all_results_list:
        return "⚠️ No results generated.", pd.DataFrame(), "Prediction scripts produced no output."

    final_df = pd.concat(all_results_list, ignore_index=True).fillna('N/A')
    non_voting_tasks = REGRESSION_TASKS_FUNCTION
    non_voting_datasets = ["DeepLocMulti", "DeepLocBinary"]
    is_voting_run = task not in non_voting_tasks and not any(ds in final_df['Dataset'].unique() for ds in non_voting_datasets) and len(final_df['Dataset'].unique()) > 1
    if is_voting_run:
        voted_results = []
        for header, group in final_df.groupby('header'):
            if group.empty: continue
            
            pred_col = 'prediction' if 'prediction' in group.columns else 'predicted_class'
            if pred_col not in group.columns: continue
            
            all_probs = []
            valid_rows = []
            
            for _, row in group.iterrows():
                prob_col = 'probabilities' if 'probabilities' in row else None
                if prob_col and pd.notna(row[prob_col]):
                    try:
                        # Handle string representation of list
                        if isinstance(row[prob_col], str):
                            # Remove brackets and split by comma
                            prob_str = row[prob_col].strip('[]')
                            probs = [float(x.strip()) for x in prob_str.split(',')]
                        elif isinstance(row[prob_col], list):
                            probs = row[prob_col]
                        else:
                            probs = json.loads(str(row[prob_col]))
                        
                        if isinstance(probs, list) and len(probs) > 0:
                            all_probs.append(probs)
                            valid_rows.append(row)
                    except (json.JSONDecodeError, ValueError, IndexError):
                        continue
            
            if not all_probs:
                continue
            
            # Perform soft voting: average all probability distributions
            max_len = max(len(probs) for probs in all_probs)
            normalized_probs = []
            for probs in all_probs:
                if len(probs) < max_len:
                    probs.extend([0.0] * (max_len - len(probs)))
                normalized_probs.append(probs)
            
            # Calculate average probabilities across all models
            avg_probs = np.mean(normalized_probs, axis=0)
            
            # Get the class with highest average probability
            voted_class = np.argmax(avg_probs)
            voted_confidence = avg_probs[voted_class]
            
            # Create result row
            result_row = group.iloc[0].copy()
            result_row[pred_col] = voted_class
            if 'probabilities' in result_row:
                result_row['probabilities'] = voted_confidence
            
            voted_results.append(result_row.to_frame().T)
        
        if voted_results:
            final_df = pd.concat(voted_results, ignore_index=True)
            final_df = final_df.drop(columns=['Dataset'], errors='ignore')
    display_df = final_df.copy()
    # map_labels is now imported from utils.label_mappers
    
    # Apply label mapping
    display_df["predicted_class"] = display_df.apply(lambda row: map_labels(row, task), axis=1)

    # Remove raw prediction column if it exists
    if 'prediction' in display_df.columns and 'predicted_class' in display_df.columns:
        display_df.drop(columns=['prediction'], inplace=True)
    # Simple column renaming
    if 'header' in display_df.columns:
        display_df.rename(columns={'header': 'Protein Name'}, inplace=True)
    if 'sequence' in display_df.columns:
        display_df.rename(columns={'sequence': 'Sequence'}, inplace=True)
    if 'predicted_class' in display_df.columns:
        display_df.rename(columns={'predicted_class': 'Predicted Class'}, inplace=True)
    if 'probabilities' in display_df.columns:
        display_df.rename(columns={'probabilities': 'Confidence Score'}, inplace=True)
    if 'Dataset' in display_df.columns:
        first_dataset = display_df['Dataset'].iloc[0] if len(display_df) > 0 else None
        first_task = DATASET_TO_TASK_MAP.get(first_dataset) if first_dataset else None
        
        if first_task in REGRESSION_TASKS_FUNCTION:
            if 'prediction' in display_df.columns:
                display_df.rename(columns={'prediction': 'Predicted Value'}, inplace=True)
                display_df['Predicted Value'] = display_df['Predicted Value'].round(2)
        else:
            if 'predicted_class' in display_df.columns:
                display_df.rename(columns={'predicted_class': 'Predicted Class'}, inplace=True)
            if 'probabilities' in display_df.columns:
                display_df.rename(columns={'probabilities': 'Confidence Score'}, inplace=True)
        
        display_df.rename(columns={'Dataset': 'Dataset'}, inplace=True)

    final_status = "✅ All predictions completed!"
    return final_status, display_df, ""



def handle_protein_function_prediction_advance(
    task: str, 
    fasta_file: Any, 
    enable_ai: bool, 
    ai_model: str, 
    user_api_key: Optional[str] = None, 
    model_name: Optional[str] = None, 
    datasets: Optional[List[str]] = None,
    progress=gr.Progress()
    ) -> Generator:
    try:
        import requests
        requests.post("/api/stats/track", json={"module": "mutation_prediction"})
    except Exception:
        pass
    """Handle protein function prediction workflow."""
    model = model_name if model_name else "ESM2-650M"
    if datasets is not None and len(datasets) > 0:
        final_datasets = datasets
    else:
        final_datasets = DATASET_MAPPING_FUNCTION.get(task, [])
    if not all([task, datasets, fasta_file]):
        yield (
            "❌ Error: Task, Datasets, and FASTA file are required.", 
            pd.DataFrame(), None, gr.update(visible=False), 
            "Please provide all required inputs."
        )
        return
    progress(0.1, desc="Running prediction...")
    yield (
        f"🚀 Starting predictions with {model}...", 
        pd.DataFrame(), None, gr.update(visible=False), 
        "AI analysis will appear here..."
    )
    
    all_results_list = []

    timestamp = str(int(time.time()))
    function_dir = get_save_path("Protein_Function", "Result")

    for i, dataset in enumerate(final_datasets):
        yield (
            f"⏳ Running prediction...", 
            pd.DataFrame(), None, gr.update(visible=False), 
            "AI analysis will appear here..."
        )
        
        try:
            model_key = MODEL_MAPPING_FUNCTION.get(model)
            if not model_key:
                raise ValueError(f"Model key not found for {model}")
            
            adapter_key = MODEL_ADAPTER_MAPPING_FUNCTION[model_key]
            script_path = Path("src") / "property" / f"{model_key}.py"
            adapter_path = Path("ckpt") / dataset / adapter_key
            output_file = function_dir / f"temp_{dataset}_{model}_{timestamp}.csv"
            
            if not script_path.exists() or not adapter_path.exists():
                raise FileNotFoundError(f"Required files not found: Script={script_path}, Adapter={adapter_path}")
            
            cmd = [sys.executable, str(script_path), "--fasta_file", str(Path(fasta_file.name)), "--adapter_path", str(adapter_path), "--output_csv", str(output_file)]
            subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8', errors='ignore')
            
            if output_file.exists():
                df = pd.read_csv(output_file) 
                df["Dataset"] = dataset
                all_results_list.append(df)
                os.remove(output_file)
        except Exception as e:
            error_detail = e.stderr if isinstance(e, subprocess.CalledProcessError) else str(e)
            print(f"Failed to process '{dataset}': {error_detail}")
            all_results_list.append(pd.DataFrame([{"Dataset": dataset, "header": "ERROR", "sequence": error_detail}]))
    
    if not all_results_list:
        yield "⚠️ No results generated.", pd.DataFrame(), None, gr.update(visible=False), "No results to analyze."
        return
    progress(0.7, desc="Processing results...")
    final_df = pd.concat(all_results_list, ignore_index=True).fillna('N/A')
    
    plot_fig = generate_plots_for_all_results(final_df)
    display_df = final_df.copy()

    display_df["predicted_class"] = display_df.apply(lambda row: map_labels(row, task), axis=1)
    if 'prediction' in display_df.columns:
        display_df.drop(columns=['prediction'], inplace=True)

    rename_map = {
        'header': "Protein Name", 
        'sequence': "Sequence", 
        'predicted_class': "Predicted Class",
        'probabilities': "Confidence Score", 
        'Dataset': "Dataset"
    }
    display_df.rename(columns=rename_map, inplace=True)
    
    if "Sequence" in display_df.columns:
        display_df["Sequence"] = display_df["Sequence"].apply(truncate_sequence)

    if "Confidence Score" in display_df.columns and "Predicted Class" in display_df.columns:
        display_df["Confidence Score"] = display_df.apply(lambda row: format_confidence(row, task), axis=1)

    ai_summary = "AI Analysis disabled. Enable in settings to generate a report."
    expert_analysis = "<div style='height: 300px; display: flex; align-items: center; justify-content: center; color: #666;'>Analysis will appear here once prediction is complete...</div>"

    if enable_ai:
        progress(0.8, desc="Generating AI summary...")
        yield "🤖 Generating AI summary...", display_df, plot_fig, gr.update(visible=False), expert_analysis
        api_key = get_api_key(ai_model, user_api_key)
        if not api_key: 
            ai_summary = f"❌ No API key found for {ai_model}."
        else:
            ai_config = AIConfig(api_key, ai_model, AI_MODELS[ai_model]["api_base"], AI_MODELS[ai_model]["model"])
            prompt = generate_ai_summary_prompt(display_df, task, model)
            ai_summary = call_ai_api(ai_config, prompt)
            expert_analysis = format_expert_response(ai_summary)
        progress(0.9, desc="Finalizing AI analysis...")
    else:
        progress(1.0, desc="Complete!")
    
    tar_path_str = ""
    try:
        timestamp = str(int(time.time()))
        archive_dir = get_save_path("Protein_Function", "Downloads_zip")

        processed_df_for_save = display_df.copy()
        result_path = archive_dir / f"Result_{timestamp}.csv"
        processed_df_for_save.to_csv(result_path, index=False)

        if plot_fig and hasattr(plot_fig, 'data') and plot_fig.data: 
            plot_path = archive_dir / "results_plot.html"
            plot_fig.write_html(str(plot_path))
        else:
            plot_path = None

        if not ai_summary.startswith("❌") and not ai_summary.startswith("AI Analysis"):
            report_path = archive_dir / f"AI_Report_{timestamp}.md"
            with open(report_path, 'w', encoding='utf-8') as f: 
                f.write(f"# AI Report\n\n{ai_summary}")
        else:
            report_path = None

        files_to_tar = {str(result_path): result_path.name}
        if plot_path and plot_path.exists():
            files_to_tar[str(plot_path)] = plot_path.name
        if report_path and report_path.exists():
            files_to_tar[str(report_path)] = report_path.name

        tar_path = function_dir / f"func_pred_{timestamp}.tar.gz"
        tar_path_str = create_tar_archive(files_to_tar, str(tar_path))
    except Exception as e: 
        print(f"Error creating tar.gz file: {e}")
        tar_path_str = ""

    final_status = "✅ All predictions completed!"
    if enable_ai and not ai_summary.startswith("❌"): 
        final_status += " AI analysis included."
    progress(1.0, desc="Complete!")
    yield final_status, display_df, plot_fig, gr.update(visible=True, value=tar_path_str), expert_analysis


def handle_protein_residue_function_prediction_chat(
    task: str,
    fasta_file: Any,
    enable_ai: bool,
    ai_model: str,
    user_api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    progress=gr.Progress()
) -> Generator:
    try:
        import requests
        requests.post("/api/stats/track", json={"module": "function_analysis"})
    except Exception:
        pass

    """Handle protein residue function prediction workflow."""
    model = model_name if model_name else "ESM2-650M"

    if not all([task, fasta_file]):
        yield(
           "❌ Error: Task and FASTA file are required.", 
            pd.DataFrame(), None,
            gr.update(visible=False), 
            "Please provide all required inputs.",
            "AI Analysis disabled." 
        )
        return

    progress(0.1, desc="Running Prediction...")
    yield(
        f"🚀 Starting predictions with {model}...", 
        pd.DataFrame(), None,
        gr.update(visible=False), 
        "AI analysis will appear here...",
        "AI Analysis disabled."
    )

    all_results_list = []
    timestamp = str(int(time.time()))
    residue_save_dir = get_save_path("Protein_Function", "Residue_save")

    yield(
        f"⏳ Running prediction...", 
        pd.DataFrame(), None,
        gr.update(visible=False), 
        "AI analysis will appear here...",
        "AI Analysis disabled."
    )
    
    model_key = MODEL_MAPPING_FUNCTION.get(model)
    if not model_key:
        raise ValueError(f"Model key not found for {model}")
    
    adapter_key = MODEL_ADAPTER_MAPPING_FUNCTION[model_key]

    # Get residue task dataset
    datasets = RESIDUE_MAPPING_FUNCTION.get(task, [])
    if not datasets:
        raise ValueError(f"No datasets found for task: {task}")
    
    for dataset in datasets:
        script_path = Path("src") / "property" / f"{model_key}.py"
        adapter_path = Path("ckpt") / dataset / adapter_key
        output_file = residue_save_dir/ f"{dataset}_{model}_{timestamp}.csv"

        if not script_path.exists() or not adapter_path.exists():
            raise FileNotFoundError(f"Required files not found: Script={script_path}, Adapter={adapter_path}")
        if isinstance(fasta_file, str):
            file_path = fasta_file
        else:
            file_path = fasta_file.name
        
        cmd = [sys.executable, str(script_path), "--fasta_file", str(Path(file_path)), "--adapter_path", str(adapter_path), "--output_csv", str(output_file)]
        subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8', errors='ignore')
        
        if output_file.exists():
            df = pd.read_csv(output_file) 
            df["Task"] = task
            df["Dataset"] = dataset
            all_results_list.append(df)
            os.remove(output_file)

    
    if all_results_list:
        combined_df = pd.concat(all_results_list, ignore_index=True)
        final_df = expand_residue_predictions(combined_df)
    else:
        final_df = pd.DataFrame()
    download_path = residue_save_dir/ f"prediction_results.csv"
    final_df.to_csv(download_path, index=False)
    display_df = final_df.copy()
    column_rename = {
        'index': 'Position',
        'residue': 'Residue',
        'predicted_label': 'Predicted Label',
        'probability': 'Probability',
    }
    display_df.rename(columns=column_rename, inplace=True)
    if 'Probability' in display_df.columns:
        display_df['Probability'] = display_df['Probability'].round(3)
    yield (
            "🤖 Expert is analyzing results...", 
            display_df, None,
            gr.update(visible=False), 
            None,
            "AI Analysis in progress..."
        )

def handle_VenusMine(
    pdb_file: Any,
    protect_start: int,
    protect_end: int,
    mmseqs_threads: int,
    mmseqs_iterations: int,
    mmseqs_max_seqs: int,
    cluster_min_seq_id: float,
    cluster_threads: int,
    top_n_threshold: int,
    evalue_threshold: float,
    progress=None
) -> Generator[Tuple[str, Any, Any, Any, Any, Any, Any, str], None, None]:
    """
    Returns:
        Generator yielding tuples of:
        (log_text, tree_image_path, labels_df, tree_download_btn, labels_download_btn, 
         zip_download_btn, tree_progress_html, status_indicator_html)
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    start_time = time.time()
    
    mmseqs_database_path = "/home/lrzhang/VenusFactory/dataset/CATH.fasta"

    if not pdb_file:
        yield (
            "❌ Error: Please upload a PDB file first.",
            None,
            pd.DataFrame(),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value="", visible=False),
            create_status_html("❌ Missing PDB file", "#dc3545")
        )
        return
    
    # Create session-specific directory with timestamp
    from datetime import datetime
    now = datetime.now()
    session_dir = get_save_path("VenusMine", "Result")
    session_timestamp = now.strftime("%Y%m%d_%H%M%S")
    log_content = "🚀 Initializing VenusMine pipeline...\n"
    log_content += f"{'='*30}\n"
    log_content += f"Session ID: {session_timestamp}\n"
    log_content += f"Output Directory: {session_dir}\n"
    log_content += f"Protected Region: {protect_start}-{protect_end}\n"
    log_content += f"{'='*30}\n\n"
    
    yield (
        log_content,
        None,
        pd.DataFrame(),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(value="", visible=False),
        create_status_html("🔵 Initializing...", "#0d6efd")
    )

    try:
        # ==================== Step 1: Setup Directories ====================
        log_content += "📁 Step 1/9: Setting up directories...\n"
        yield (
            log_content,
            None,
            pd.DataFrame(),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value="", visible=False),
            create_status_html("🔵 Step 1/9 - Setup", "#0d6efd")
        )
        protein_name = "protein"
        foldseek_dir = get_save_path("VenusMine", "FoldSeek_Search")
        
        logger.info(f"Working directory: {foldseek_dir}")
        log_content += f"   ✓ Working directory created: {foldseek_dir}\n\n"

        pdb_file_path = foldseek_dir / f"{protein_name}.pdb"
        if isinstance(pdb_file, str):
            shutil.copy(pdb_file, pdb_file_path)
        else:
            with open(pdb_file_path, 'w') as f:
                content = pdb_file.read() if hasattr(pdb_file, 'read') else str(pdb_file)
                f.write(content)
    
        # ==================== Step 2: FoldSeek Search ====================
        log_content += "🔍 Step 2/9: Running FoldSeek structural search...\n"
        yield (
            log_content,
            None,
            pd.DataFrame(),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value="", visible=False),
            create_status_html("⏳ Running FoldSeek search...", "#0d6efd")
        )
        
        downloaded_files = download_foldseek_m8(str(pdb_file_path), foldseek_dir)
        logger.info(f"FoldSeek finished, results in {foldseek_dir}")
        log_content += f"   ✓ FoldSeek search completed\n"
        log_content += f"   ✓ Downloaded {len(downloaded_files)} alignment files\n\n"

        # ==================== Step 3: Parse FoldSeek Alignments ====================
        log_content += "📝 Step 3/9: Parsing FoldSeek alignments...\n"
        yield (
            log_content,
            None,
            pd.DataFrame(),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value="", visible=False),
            create_status_html("⏳ Parsing alignments...", "#0d6efd")
        )
        
        filename_list = [
            os.path.join(str(foldseek_dir), "alis_afdb-proteome.m8"),
            os.path.join(str(foldseek_dir), "alis_afdb-swissprot.m8"),
            os.path.join(str(foldseek_dir), "alis_afdb50.m8"),
            os.path.join(str(foldseek_dir), "alis_cath50.m8"),
            os.path.join(str(foldseek_dir), "alis_gmgcl_id.m8"),
            os.path.join(str(foldseek_dir), "alis_mgnify_esm30.m8"),
            os.path.join(str(foldseek_dir), "alis_pdb100.m8")
        ]
        alignments_collection = []
        alignments_dbname = [
            "afdb_proteome", "afdb_swissprot", "afdb50", "cath50",
            "gmgcl_id", "mgnify_esm30", "pdb100"]
        
        # Parse the FoldSeek result files
        for filename in filename_list:
            parser = FoldSeekAlignmentParser(filename)
            alignments = parser.parse()
            alignments_collection.append(alignments)

        total_alignments = 0
        # print results
        for i in range(len(alignments_dbname)):
            log_content += "Database: {}, found alignments: {}\n".format(alignments_dbname[i], len(alignments_collection[i]))
            total_alignments += len(alignments_collection[i])



        foldseek_dir_path = Path(foldseek_dir)
        output_file = foldseek_dir_path / f"{foldseek_dir_path.name}_foldseek.fasta"
        f_out = open(output_file, "w")
        count = 0

        for alignments_index in range(len(alignments_dbname)):
            alignments = alignments_collection[alignments_index]
            alignments_db = alignments_dbname[alignments_index]

            for alignment in alignments:
                # consider protect region
                if alignment.qstart <= protect_start and alignment.qend >= protect_end:
                    f_out.write(">" + alignments_db + " " + alignment.tseqid.split(" ")[0] + "\n")
                    f_out.write(alignment.tseq + "\n")
                    count += 1
        
        f_out.close()
        
        log_content += f"   ✓ Found {count} sequences matching protected region\n"
        log_content += f"   ✓ Sequences saved to: {output_file.name}\n\n"
        
        # Check if we have sequences to search
        if count == 0:
            log_content += "   ⚠️ No sequences found matching the protected region criteria.\n"
            log_content += "   💡 Try adjusting the protected region parameters (start/end positions).\n"
            yield (
                log_content,
                None,
                pd.DataFrame(),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(value="⚠️ No sequences found"),
                gr.update(visible=True, value="⚠️ *No sequences matched protected region*"),
                gr.update(value="**Status:** ⚠️ No sequences to search")
            )
            return

        # ==================== Step 4: MMseqs Search ====================
        log_content += "🔎 Step 4/9: Running MMseqs2 sequence search...\n"
        yield (
            log_content,
            None,
            pd.DataFrame(),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value="", visible=False),
            create_status_html("⏳ Running MMseqs2 search...", "#0d6efd")
        )
        
        # Check if mmseqs is installed
        mmseqs_check = subprocess.run(['which', 'mmseqs'], capture_output=True, text=True)
        
        
        # Check if database exists
        if not Path(mmseqs_database_path).exists():
            log_content += f"   ❌ Database file not found: {mmseqs_database_path}\n"
            yield (
                log_content,
                None,
                pd.DataFrame(),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(value="❌ Database not found"),
                gr.update(visible=True, value="❌ *Database file missing*"),
                gr.update(value="**Status:** ❌ Database not found")
            )
            return
        
        mmseqs_prefix = foldseek_dir / f"{protein_name}_MEER"
        mmseqs_tsv = Path(str(mmseqs_prefix) + ".tsv")
        tmp_dir = get_save_path("VenusMine", "tmp")

        cmd_search = [
            "mmseqs", "easy-search", 
            str(output_file), 
            str(mmseqs_database_path), 
            str(mmseqs_tsv), 
            str(tmp_dir),
            "--format-output", "query,target,pident,fident,nident,alnlen,bits,tseq,evalue",
            "--threads", str(mmseqs_threads),
            "--num-iterations", str(mmseqs_iterations),
            "--max-seqs", str(mmseqs_max_seqs),
            "--search-type", "1"
        ]
        
        log_content += f"   • Database: {Path(mmseqs_database_path).name}\n"
        log_content += f"   • Threads: {mmseqs_threads}, Iterations: {mmseqs_iterations}\n"
        log_content += f"   • Max sequences: {mmseqs_max_seqs}\n"
        
        try:
            result = subprocess.run(cmd_search, capture_output=True, text=True, check=True)
            log_content += f"   ✓ MMseqs search completed successfully\n"
        except subprocess.CalledProcessError as e:
            log_content += f"   ❌ MMseqs search failed with error:\n"
            log_content += f"   {e.stderr[:500]}\n"
            yield (
                log_content,
                None,
                pd.DataFrame(),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(value="❌ MMseqs failed"),
                gr.update(visible=True, value="❌ *MMseqs search failed*"),
                gr.update(value="**Status:** ❌ Failed - MMseqs error")
            )
            return
        
        if not mmseqs_tsv.exists():
            log_content += "   ❌ MMseqs output file not created.\n"
            yield (
                log_content,
                None,
                pd.DataFrame(),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(value="❌ MMseqs failed"),
                gr.update(visible=True, value="❌ *MMseqs search failed*"),
                gr.update(value="**Status:** ❌ Failed - MMseqs error")
            )
            return
        
        log_content += f"   ✓ MMseqs search completed\n"
        log_content += f"   ✓ Results saved to: {mmseqs_tsv.name}\n\n"

        # Convert TSV to FASTA
        log_content += "   • Converting TSV to FASTA format...\n"
        mmseqs_fasta = Path(str(mmseqs_prefix) + ".fasta")
        
        try:
            header, data = read_tsv(str(mmseqs_tsv))
            log_content += f"   • Loaded TSV file successfully, {len(data)} entries\n"
            write_fasta_from_tsv(header, data, str(mmseqs_fasta), evalue=None)
            log_content += f"   ✓ FASTA file created: {mmseqs_fasta.name}\n\n"
        except Exception as e:
            log_content += f"   ❌ TSV to FASTA conversion failed: {str(e)[:200]}\n"
            yield (
                log_content,
                None,
                pd.DataFrame(),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(value="❌ Conversion failed"),
                gr.update(visible=True, value="❌ *Format conversion failed*"),
                gr.update(value="**Status:** ❌ Failed - Conversion error")
            )
            return
        
        if not mmseqs_fasta.exists():
            log_content += "   ❌ FASTA file was not created.\n"
            yield (
                log_content,
                None,
                pd.DataFrame(),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(value="❌ Conversion failed"),
                gr.update(visible=True, value="❌ *Format conversion failed*"),
                gr.update(value="**Status:** ❌ Failed - Conversion error")
            )
            return

        # ==================== Step 5: MMseqs Clustering ====================
        log_content += "🧮 Step 5/9: Clustering sequences (removing redundancy)...\n"
        yield (
            log_content,
            None,
            pd.DataFrame(),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value="", visible=False),
            create_status_html("⏳ Clustering sequences...", "#0d6efd")
        )
        
        cluster_out = foldseek_dir / f"{protein_name}_clus2_NR_out"
        log_content += f"   • Min sequence identity: {cluster_min_seq_id}\n"
        log_content += f"   • Threads: {cluster_threads}\n"
        
        try:
            result = subprocess.run(
                ["mmseqs", "easy-cluster", str(mmseqs_fasta), str(cluster_out), str(tmp_dir),
                 "--min-seq-id", str(cluster_min_seq_id), "--threads", str(cluster_threads)],
                capture_output=True, text=True, check=True
            )
            log_content += f"   ✓ Clustering completed\n"
        except subprocess.CalledProcessError as e:
            log_content += f"   ❌ Clustering failed: {e.stderr[:200]}\n"
            yield (
                log_content,
                None,
                pd.DataFrame(),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(value="❌ Clustering failed"),
                gr.update(visible=True, value="❌ *Clustering failed*"),
                gr.update(value="**Status:** ❌ Failed - Clustering error")
            )
            return
        
        cluster_rep_fasta = foldseek_dir / f"{protein_name}_clus2_NR_out_rep_seq.fasta"
        if not cluster_rep_fasta.exists():
            log_content += "   ❌ Cluster representative file not created.\n"
            yield (
                log_content,
                None,
                pd.DataFrame(),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(value="❌ Clustering failed"),
                gr.update(visible=True, value="❌ *Clustering failed*"),
                gr.update(value="**Status:** ❌ Failed - Clustering error")
            )
            return
        
        log_content += f"   ✓ Representative sequences: {cluster_rep_fasta.name}\n\n"

        # ==================== Step 6: ProstT5 Embedding ====================
        log_content += "🧬 Step 6/9: Computing ProstT5 embeddings for discovered sequences...\n"
        yield (
            log_content,
            None,
            pd.DataFrame(),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value="", visible=False),
            create_status_html("⏳ Computing embeddings...", "#0d6efd")
        )
        
        log_content += f"   • Loading ProstT5 model on cuda...\n"
        tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False)
        model = T5EncoderModel.from_pretrained("Rostlab/ProstT5").to("cuda")
        model = model.half()

        data = parse_fasta(str(cluster_rep_fasta))
        batch_size = 16
        log_content += f"   • Processing {len(data)} sequences (batch_size={batch_size})...\n"
        
        seq_labels, seq_strs, rep = calculate_representation(model, tokenizer, data, logger, start_time, batch_size=batch_size)
        
        result_pkl = foldseek_dir / "result.pkl"
        save_representation(seq_labels, seq_strs, rep, str(result_pkl))
        
        log_content += f"   ✓ Embeddings computed for {len(seq_labels)} sequences\n"
        log_content += f"   ✓ Saved to: {result_pkl.name}\n\n"
        
        del model, tokenizer
        torch.cuda.empty_cache()

        # ==================== Step 7: Reference (PDB) Embedding ====================
        log_content += "🎯 Step 7/9: Computing embeddings for reference protein...\n"
        yield (
            log_content,
            None,
            pd.DataFrame(),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value="", visible=False),
            create_status_html("⏳ Processing reference...", "#0d6efd")
        )
        
        ref_sequences = extract_sequence_from_pdb(str(pdb_file_path))
        ref_pkl = foldseek_dir / "refseq.pkl"
        
        log_content += f"   • Loading ProstT5 model...\n"
        tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False)
        model = T5EncoderModel.from_pretrained("Rostlab/ProstT5").to("cuda")
        model = model.half()
        
        # Reference sequences are usually few, use batch_size=1
        ref_labels, ref_strs, ref_rep = calculate_representation(model, tokenizer, ref_sequences, logger, start_time, batch_size=1)
        save_representation(ref_labels, ref_strs, ref_rep, str(ref_pkl))
        
        log_content += f"   ✓ Reference embeddings computed\n"
        log_content += f"   ✓ Saved to: {ref_pkl.name}\n\n"
        
        del model, tokenizer

        torch.cuda.empty_cache()

        # ==================== Step 8: EC Dataset Embedding ====================
        log_content += "📚 Step 8/9: Processing EC enzyme classification dataset...\n"
        yield (
            log_content,
            None,
            pd.DataFrame(),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value="", visible=False),
            create_status_html("⏳ Processing EC dataset...", "#0d6efd")
        )
        
        ec_fasta = Path("data/VenusMine/ec_dataset.fasta")
        ec_pkl = Path("data/VenusMine/ec.pkl")
        ec_csv = Path("data/VenusMine/ec_dataset.csv")
        
        if not ec_pkl.exists():
            log_content += f"   • EC embeddings not found, computing...\n"
            tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False)
            model = T5EncoderModel.from_pretrained("Rostlab/ProstT5").to("cuda")
            model = model.half()
            
            data = parse_fasta(str(ec_fasta))
            batch_size = 16 if len(data) > 100 else 8
            log_content += f"   • Processing {len(data)} EC sequences (batch_size={batch_size})...\n"
            ec_labels, ec_strs, ec_rep = calculate_representation(model, tokenizer, data, logger, start_time, batch_size=batch_size)
            save_representation(ec_labels, ec_strs, ec_rep, str(ec_pkl))
            
            log_content += f"   ✓ EC embeddings computed and cached\n\n"
            
            del model, tokenizer
            torch.cuda.empty_cache()
        else:
            log_content += f"   ✓ Using cached EC embeddings\n\n"

        # ==================== Step 9: Build Tree ====================
        log_content += "🌳 Step 9/9: Building phylogenetic tree and visualization...\n"
        yield (
            log_content,
            None,
            pd.DataFrame(),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value=create_progress_html("Building phylogenetic tree..."), visible=True),
            create_status_html("🔵 Step 9/9 - Tree construction", "#0d6efd")
        )
        
        log_content += f"   • Top N threshold: {top_n_threshold}\n"
        log_content += f"   • E-value threshold: {evalue_threshold}\n"
        
        plot_path, label_path = build_and_visualize_tree(
            refseq_pkl=str(ref_pkl),
            discovered_pkl=str(result_pkl),
            ec_pkl=str(ec_pkl),
            ec_csv=str(ec_csv),
            top_n_threshold=top_n_threshold,
            output_dir=session_dir
        )
        
        log_content += f"   ✓ Tree visualization created\n"
        log_content += f"   ✓ Plot saved to: {Path(plot_path).name}\n"
        log_content += f"   ✓ Labels saved to: {Path(label_path).name}\n\n"

        # ==================== Final: Package Results ====================
        log_content += "📦 Packaging results...\n"
        
        # 读取标签数据
        labels_df = pd.DataFrame()
        if label_path and Path(label_path).exists():
            try:
                labels_df = pd.read_csv(label_path, sep='\t')  # TSV file
                log_content += f"   ✓ Loaded {len(labels_df)} sequence labels\n"
            except Exception as e:
                log_content += f"   ⚠ Could not read labels: {e}\n"
        
        # 创建 tar.gz 文件
        tar_path = None
        try:
            results_dir = get_save_path("VenusMine", "Result")
            files_to_tar = {}
            if plot_path and Path(plot_path).exists():
                files_to_tar[str(plot_path)] = Path(plot_path).name
            if label_path and Path(label_path).exists():
                files_to_tar[str(label_path)] = Path(label_path).name

            if files_to_tar:
                tar_path = results_dir / f"venusmine_results_{session_timestamp}.tar.gz"
                create_tar_archive(files_to_tar, str(tar_path))
                log_content += f"   ✓ Results packaged to: {tar_path.name}\n"
            else:
                log_content += "   ⚠ No files available to package\n"
        except Exception as e:
            log_content += f"   ⚠ Could not create tar.gz: {e}\n"
        
        # 最终日志
        log_content += f"\n{'='*60}\n"
        log_content += f"✅ VenusMine Pipeline Completed Successfully!\n"
        log_content += f"{'='*60}\n\n"
        log_content += f"📊 Summary Statistics:\n"
        log_content += f"   • Total sequences discovered: {len(labels_df)}\n"
        log_content += f"   • Processing time: {time.time() - start_time:.2f} seconds\n"
        log_content += f"   • Results directory: {session_dir}\n"
        log_content += f"\n{'='*60}\n"
        
        # 最终输出
        yield (
            log_content,
            str(plot_path) if plot_path else None,  # 返回图片路径
            labels_df,
            gr.update(visible=True, value=str(plot_path)),
            gr.update(visible=True, value=str(label_path)),
            gr.update(visible=True, value=str(tar_path) if tar_path else None),
            gr.update(value="", visible=False),  # 隐藏进度指示器
            create_status_html("✅ Completed successfully!", "#198754")
        )
    
    except Exception as e:
        # 错误处理
        error_log = f"\n{'='*60}\n"
        error_log += f"❌ Pipeline Error\n"
        error_log += f"{'='*60}\n\n"
        error_log += f"Error Type: {type(e).__name__}\n"
        error_log += f"Error Message: {str(e)}\n\n"
        
        import traceback
        error_log += f"Full Traceback:\n{traceback.format_exc()}\n"
        error_log += f"{'='*60}\n"
        
        final_log = log_content + error_log
        
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        
        yield (
            final_log,
            None,
            pd.DataFrame(),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value="", visible=False),
            create_status_html(f"❌ Error: {type(e).__name__}", "#dc3545")
        )

def create_advanced_tool_tab(constant: Dict[str, Any]) -> Dict[str, Any]:
    sequence_models = ["VenusPLM", "ESM2-650M", "ESM-1v", "ESM-1b"]
    structure_models = ["VenusREM (foldseek-based)", "ProSST-2048", "ProtSSN", "ESM-IF1", "SaProt", "MIF-ST", "VenusREM"]
    function_models = list(MODEL_MAPPING_FUNCTION.keys())
    residue_function_models = list(MODEL_RESIDUE_MAPPING_FUNCTION.keys())

    with gr.Blocks() as demo:
        with gr.Tabs():
            with gr.TabItem("Intelligent Directed Evolution"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2):
                         with gr.Tabs():
                            with gr.TabItem("🧬 Sequence-based Model"):
                                gr.Markdown("### Model Configuration")
                                seq_function_dd = gr.Dropdown(choices=DATASET_MAPPING_ZERO_SHOT, label="Select Protein Function", value=DATASET_MAPPING_ZERO_SHOT[0], visible=False)
                                seq_model_dd = gr.Dropdown(choices=sequence_models, label="Select Sequence-based Model", value=sequence_models[0])
                                gr.Markdown("**Data Input**")
                                with gr.Tabs():
                                    with gr.TabItem("Upload FASTA File"):
                                        seq_file_upload = gr.File(label="Upload FASTA file", file_types=[".fasta", ".fa"])
                                        seq_file_example = gr.Examples(examples=[["./download/P60002.fasta"]], inputs=seq_file_upload, label="Click example to load")
                                    with gr.TabItem("Paste FASTA Content"):
                                        seq_paste_content_input = gr.Textbox(label="Paste FASTA Content", placeholder="Paste FASTA content here...", lines=8, max_lines=15)
                                        with gr.Row():
                                            seq_paste_content_btn = gr.Button("🔍 Detect Content", variant="primary", size="m")
                                            seq_paste_clear_btn = gr.Button("🗑️ Clear", variant="primary", size="m")
                                
                                seq_protein_display = gr.Textbox(label="Uploaded Protein Sequence", interactive=False, lines=3, max_lines=7)
                                seq_sequence_selector = gr.Dropdown(label="Select Chain", choices=["Sequence_1"], value="Sequence_1", visible=False, allow_custom_value=True)
                                seq_original_file_path_state = gr.State("")
                                seq_original_paste_content_state = gr.State("")
                                seq_selected_sequence_state = gr.State("Sequence_1")
                                seq_sequence_state = gr.State({})
                                seq_current_file_state = gr.State("")

                                gr.Markdown("### Configure AI Analysis (Optional)")
                                with gr.Accordion("AI Settings", open=True):
                                    enable_ai_zshot_seq = gr.Checkbox(label="Enable AI Summary", value=False)
                                    with gr.Group(visible=False) as ai_box_zshot_seq:
                                        ai_model_seq_zshot = gr.Dropdown(
                                            choices=list(AI_MODELS.keys()), 
                                            value="DeepSeek", 
                                            label="Select AI Model"
                                        )
                                        ai_status_seq_zshot = gr.Markdown(
                                            value="✓ Using provided API Key" if os.getenv("DEEPSEEK_API_KEY") else "⚠ No API Key found in .env file",
                                            visible=True
                                        )
                                        api_key_in_seq_zshot = gr.Textbox(
                                            label="API Key", 
                                            type="password", 
                                            placeholder="Enter your API Key if needed",
                                            visible=not bool(os.getenv("DEEPSEEK_API_KEY"))
                                        )                           
                                seq_predict_btn = gr.Button("🚀 Start Prediction (Sequence)", variant="primary")

                            with gr.TabItem("🏗️ Structure-based Model"):
                                gr.Markdown("### Model Configuration")
                                struct_function_dd = gr.Dropdown(choices=DATASET_MAPPING_ZERO_SHOT, label="Select Protein Function", value=DATASET_MAPPING_ZERO_SHOT[0], visible=False)
                                struct_model_dd = gr.Dropdown(choices=structure_models, label="Select Structure-based Model", value=structure_models[0])
                                gr.Markdown("**Data Input**")
                                with gr.Tabs():
                                    with gr.TabItem("Upload PDB File"):
                                        struct_file_upload = gr.File(label="Upload PDB File", file_types=[".pdb"])
                                        struct_file_example = gr.Examples(examples=[["./download/alphafold2_structures/A0A0C5B5G6.pdb"]], inputs=struct_file_upload, label="Click example to load")
                                    with gr.TabItem("Paste PDB Content"):
                                        struct_paste_content_input = gr.Textbox(label="Paste PDB Content", placeholder="Paste PDB content here...", lines=8, max_lines=15)
                                        with gr.Row():
                                            struct_paste_content_btn = gr.Button("🔍 Detect Content", variant="secondary", size="sm")
                                            struct_paste_clear_btn = gr.Button("🗑️ Clear", variant="secondary", size="sm")
                                    
                                struct_protein_display = gr.Textbox(label="Uploaded Protein Sequence", interactive=False, lines=3, max_lines=7)
                                struct_chain_selector = gr.Dropdown(label="Select Chain", choices=["A"], value="A", visible=False, allow_custom_value=True)
                                struct_original_file_path_state = gr.State("")
                                struct_original_paste_content_state = gr.State("")
                                struct_selected_chain_state = gr.State("A")
                                struct_chains_state = gr.State({})
                                struct_current_file_state = gr.State("")
                                gr.Markdown("### Configure AI Analysis (Optional)")
                                with gr.Accordion("AI Settings", open=True):
                                    enable_ai_zshot_stru = gr.Checkbox(label="Enable AI Summary", value=False)
                                    with gr.Group(visible=False) as ai_box_zshot_stru:
                                        ai_model_stru_zshot = gr.Dropdown(
                                            choices=list(AI_MODELS.keys()), 
                                            value="DeepSeek", 
                                            label="Select AI Model"
                                        )
                                        ai_status_stru_zshot = gr.Markdown(
                                            value="✓ Using provided API Key" if os.getenv("DEEPSEEK_API_KEY") else "⚠ No API Key found in .env file",
                                            visible=True
                                        )
                                        api_key_in_stru_zshot = gr.Textbox(
                                            label="API Key", 
                                            type="password", 
                                            placeholder="Enter your API Key if needed",
                                            visible=not bool(os.getenv("DEEPSEEK_API_KEY"))
                                        )
                                struct_predict_btn = gr.Button("🚀 Start Prediction (Structure)", variant="primary")
                    
                    with gr.Column(scale=3):
                        gr.Markdown("### Results")
                        zero_shot_status_box = gr.Textbox(label="Status", interactive=False)
                        with gr.Tabs():
                            with gr.TabItem("📊 Raw Results"):
                                zero_shot_df_out = gr.DataFrame(label="Raw Data")
                            with gr.TabItem("📈 Prediction Heatmap"):
                                with gr.Row(visible=False) as zero_shot_view_controls:
                                    expand_btn = gr.Button("Show Complete Heatmap", size="sm", visible=False)
                                    collapse_btn = gr.Button("Show Summary View", size="sm", visible=False)
                                zero_shot_plot_out = gr.Plot(label="Heatmap")
                            with gr.TabItem("👨‍🔬 AI Expert Analysis"):
                                zero_shot_ai_expert_html = gr.HTML(
                                    value="<div style='height: 300px; display: flex; align-items: center; justify-content: center; color: #666;'>AI analysis will appear here...</div>",
                                    label="👨‍🔬 AI Expert Analysis"
                                )
                        zero_shot_download_btn = gr.DownloadButton("💾 Download Results", visible=False)
                        zero_shot_full_data_state = gr.State()
                        zero_shot_download_path_state = gr.State()

            with gr.TabItem("Protein Function Prediction"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2):
                        gr.Markdown("**Model Configuration**")
                        adv_func_model_dd = gr.Dropdown(choices=function_models, label="Select Model", value="ESM2-650M")
                        adv_func_task_dd = gr.Dropdown(choices=list(DATASET_MAPPING_FUNCTION.keys()), label="Select Task", value="Solubility")
                        all_possible_datasets = []
                        for datasets_list in DATASET_MAPPING_FUNCTION.values():
                            all_possible_datasets.extend(datasets_list)
                        all_possible_datasets = sorted(list(set(all_possible_datasets)))
                        default_datasets_for_solubility = DATASET_MAPPING_FUNCTION.get("Solubility", [])
                        adv_func_dataset_cbg = gr.CheckboxGroup(label="Select Datasets", choices=default_datasets_for_solubility, value=default_datasets_for_solubility)
                        adv_func_dataset_cbg_chat = gr.CheckboxGroup(choices=all_possible_datasets, value=all_possible_datasets, visible=False)
                        
                        gr.Markdown("**Data Input**")
                        with gr.Tabs():
                            with gr.TabItem("Upload FASTA File"):
                                function_fasta_upload = gr.File(label="Upload FASTA file", file_types=[".fasta", ".fa"])
                                function_fasta_example = gr.Examples(examples=[["./download/P60002.fasta"]], inputs=function_fasta_upload, label="Click example to load")
                            with gr.TabItem("Paste FASTA Content"):
                                function_paste_content_input = gr.Textbox(label="Paste FASTA Content", placeholder="Paste FASTA content here...", lines=8, max_lines=15)
                                with gr.Row():
                                    function_paste_content_btn = gr.Button("🔍 Detect Content", variant="primary", size="m")
                                    function_paste_clear_btn = gr.Button("🗑️ Clear", variant="primary", size="m")
                            
                        function_protein_display = gr.Textbox(label="Uploaded Protein", interactive=False, lines=3, max_lines=7)
                        function_protein_chat_btn = gr.Button("Chat API Trigger", visible=False)
                        function_protein_selector = gr.Dropdown(label="Select Chain", choices=["Sequence_1"], value="Sequence_1", visible=False, allow_custom_value=True)
                        function_original_file_path_state = gr.State("")
                        function_original_paste_content_state = gr.State("")
                        function_selected_sequence_state = gr.State("Sequence_1")
                        function_sequence_state = gr.State({})
                        function_current_file_state = gr.State("")
                        
                        gr.Markdown("### Configure AI Analysis (Optional)")
                        with gr.Accordion("AI Settings", open=True):
                            enable_ai_func = gr.Checkbox(label="Enable AI Summary", value=False)
                            with gr.Group(visible=False) as ai_box_func:
                                ai_model_seq_func = gr.Dropdown(
                                    choices=list(AI_MODELS.keys()), 
                                    label="Select AI Model", 
                                    value="DeepSeek"
                                )
                                ai_status_seq_func = gr.Markdown(
                                    value="✓ Using provided API Key" if os.getenv("DEEPSEEK_API_KEY") else "⚠ No API Key found in .env file",
                                    visible=True
                                )
                                api_key_in_seq_func = gr.Textbox(
                                    label="API Key", 
                                    type="password", 
                                    placeholder="Enter your API Key if needed",
                                    visible=not bool(os.getenv("DEEPSEEK_API_KEY"))
                                )
                        adv_func_predict_btn = gr.Button("🚀 Start Prediction (Advanced)", variant="primary")

                    with gr.Column(scale=3):
                        gr.Markdown("### Results")
                        function_status_textbox = gr.Textbox(label="Status", interactive=False)
                        with gr.Tabs():
                            with gr.TabItem("📊 Raw Results"):
                                function_results_df = gr.DataFrame(label="Prediction Data", column_widths=["20%", "20%", "20%", "20%", "20%"])
                            with gr.TabItem("📈 Prediction Plots"):
                                function_results_plot = gr.Plot(label="Confidence Scores")
                            with gr.TabItem("👨‍🔬 AI Expert Analysis"):
                                function_ai_expert_html = gr.HTML(
                                    value="<div style='height: 300px; display: flex; align-items: center; justify-content: center; color: #666;'>AI analysis will appear here...</div>",
                                    label="👨‍🔬 AI Expert Analysis"
                                )
                        function_download_btn = gr.DownloadButton("💾 Download Results", visible=False)
                
            with gr.TabItem("Functional Residue Prediction"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2):
                        gr.Markdown("### Model Configuration")
                        adv_residue_function_model_dd = gr.Dropdown(choices=residue_function_models, label="Select Model", value="ESM2-650M")
                        adv_residue_function_task_dd = gr.Dropdown(choices=list(RESIDUE_MAPPING_FUNCTION.keys()), label="Select Task", value="Activity Site")
                        gr.Markdown("**Data Input**")
                        with gr.Tabs():
                            with gr.TabItem("Upload FASTA File"):
                                adv_residue_function_fasta_upload = gr.File(label="Upload Fasta file", file_types=[".fasta", ".fa"])
                                adv_residue_function_file_exmaple = gr.Examples(examples=[["./download/P60002.fasta"]], inputs=adv_residue_function_fasta_upload, label="Click example to load")
                            with gr.TabItem("Paste FASTA Content"):
                                adv_residue_function_paste_content_input = gr.Textbox(label="Paste FASTA Content", placeholder="Paste FASTA content here...", lines=8, max_lines=15)
                                with gr.Row():
                                    adv_residue_function_paste_content_btn = gr.Button("🔍 Detect & Save Content", variant="primary", size="m")
                                    adv_residue_function_paste_clear_btn = gr.Button("🗑️ Clear", variant="primary", size="m")
                        
                        adv_residue_function_protein_display = gr.Textbox(label="Uploaded Protein Sequence", interactive=False, lines=3, max_lines=7)
                        adv_residue_function_protein_chat_btn = gr.Button("Chat API Trigger", visible=False)
                        adv_residue_function_selector = gr.Dropdown(label="Select Chain", choices=["Sequence_1"], value="Sequence_1", visible=False, allow_custom_value=True)
                        adv_residue_function_original_file_path_state = gr.State("")
                        adv_residue_function_original_paste_content_state = gr.State("")
                        adv_residue_function_selected_sequence_state = gr.State("Sequence_1")
                        adv_residue_function_sequence_state = gr.State({})
                        adv_residue_function_current_file_state = gr.State("")

                        gr.Markdown("### Configure AI  Analysis (Optional)")
                        with gr.Accordion("AI Settings", open=True):
                            enable_ai_residue_function = gr.Checkbox(label="Enable AI Summary", value=False)
                            with gr.Accordion("AI  Settings", open=True):
                                with gr.Group(visible=False) as ai_box_residue_function:
                                    ai_model_dd_residue_function = gr.Dropdown(
                                        choices=list(AI_MODELS.keys()),
                                        label="Select AI Model",
                                        value="DeepSeek"
                                    )
                                    ai_status_residue_function = gr.Markdown(
                                        value="✓ Using provided API Key" if os.getenv("DEEPSEEK_API_KEY") else "⚠ No API Key found in .env file",
                                        visible=True
                                    )
                                    api_key_in_residue_function = gr.Textbox(
                                        label="API Key",
                                        type="password",
                                        placeholder="Ener your API Key if needed",
                                        visible=not bool(os.getenv("DEEPSEEK_API_KEY"))
                                    )
                        adv_residue_function_predict_btn = gr.Button("🚀 Start Prediction", variant="primary")
                    
                    with gr.Column(scale=3):
                        gr.Markdown("### Results")
                        adv_residue_function_status_textbox = gr.Textbox(label="Status", interactive=False)
                        with gr.Tabs():
                            with gr.TabItem("📊 Raw Results"):
                                adv_residue_function_results_df = gr.DataFrame(label="Prediction Data", column_widths=["20%", "20%", "20%", "20%", "20%"])
                            with gr.TabItem("📈 Prediction Heatmap"):
                                adv_residue_function_plot_out = gr.Plot(label="Heatmap")
                            with gr.TabItem("👨‍🔬 AI Expert Analysis"):
                                adv_residue_function_ai_expert_html = gr.HTML(
                                    value="<div style='height: 300px; display: flex; align-items: center; justify-content: center; color: #666;'>AI analysis will appear here...</div>",
                                    label="👨‍🔬 AI Expert Analysis"
                                )
                        adv_residue_function_download_btn = gr.DownloadButton("💾 Download Results", visible=False)


            with gr.TabItem("VenusMine"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2):
                        gr.Markdown("### 📁 Input Configuration")
                        
                        venus_pdb_upload = gr.File(label="Upload PDB Structure", file_types=[".pdb"], type="filepath")
                        
                        with gr.Accordion("⚙️ Advanced Parameters", open=False):
                            with gr.Group():
                                gr.Markdown("**Protected Region**")
                                with gr.Row():
                                    venus_protect_start = gr.Number(label="Start Position", value=1, minimum=1, step=1)
                                    venus_protect_end = gr.Number(label="End Position", value=100, minimum=1, step=1)
                            
                            with gr.Group():
                                gr.Markdown("**MMseqs2 Search Parameters**")
                                venus_mmseqs_threads = gr.Slider(label="Threads", minimum=1, maximum=100, value=96, step=1)
                                venus_mmseqs_iterations = gr.Slider(label="Iterations", minimum=1, maximum=10, value=3, step=1)
                                venus_mmseqs_max_seqs = gr.Slider(label="Max Sequences", minimum=100, maximum=5000, value=100, step=100)
                            
                            with gr.Group():
                                gr.Markdown("**Clustering Parameters**")
                                venus_cluster_min_seq_id = gr.Slider(label="Min Sequence Identity", minimum=0.1, maximum=1.0, value=0.5, step=0.05)
                                venus_cluster_threads = gr.Slider(label="Threads", minimum=1, maximum=100, value=96, step=1)
                            
                            with gr.Group():
                                gr.Markdown("**Tree Building Parameters**")
                                venus_top_n = gr.Slider(label="Top N Results", minimum=1, maximum=10000, value=10, step=1)
                                venus_evalue = gr.Number(label="E-value Threshold", value=1e-5)
                        
                        venus_start_btn = gr.Button("🚀 Start VenusMine Pipeline", variant="primary", size="lg")
                        
                        # Status indicator
                        venus_status_indicator = gr.HTML(value="<div style='text-align: center; padding: 10px;'><span style='color: #666;'>Ready to start</span></div>")

                    with gr.Column(scale=5):
                        gr.Markdown("### 📈 Pipeline Results")
                        with gr.Tabs() as venus_result_tabs:
                            with gr.TabItem("🔬 Structure Visualization"):
                                venus_pdb_viewer = Molecule3D(label="Structure Viewer", reps=RCSB_REPS, height=400)
                            with gr.TabItem("🌳 Phylogenetic Tree"):
                                # 进度指示器
                                venus_tree_progress = gr.HTML(value="", visible=False)
                                venus_tree_image = gr.Image(label="Evolutionary Tree", type="filepath", visible=True)
                                venus_tree_download_btn = gr.DownloadButton("📊 Download Tree Image", visible=False)
                                
                            with gr.TabItem("🏷️ Sequence Labels"):
                                venus_labels_df = gr.DataFrame(
                                    label="Discovered Sequences",
                                    interactive=False, wrap=True)
                                venus_labels_download_btn = gr.DownloadButton("📄 Download Labels (TSV)", visible=False)
                            
                            with gr.TabItem("📦 Complete Results"):
                                gr.Markdown("Download all results in a single ZIP file")
                                venus_full_zip_btn = gr.DownloadButton("📦 Download Complete Results", visible=False)
                            
                            with gr.TabItem("📋 Processing Log"):
                                venus_log_output = gr.Textbox(
                                    label="Real-time Processing Log", lines=20, max_lines=25,
                                    interactive=False, autoscroll=True)

        # clear_paste_content_pdb and clear_paste_content_fasta are imported from file_handlers
        
        # update_dataset_choices_fixed is now imported from utils.ui_helpers
        
        
        enable_ai_zshot_seq.change(fn=toggle_ai_section, inputs=enable_ai_zshot_seq, outputs=ai_box_zshot_seq)
        enable_ai_zshot_stru.change(fn=toggle_ai_section, inputs=enable_ai_zshot_stru, outputs=ai_box_zshot_stru)
        enable_ai_func.change(fn=toggle_ai_section, inputs=enable_ai_func, outputs=ai_box_func)
        enable_ai_residue_function.change(fn=toggle_ai_section, inputs=enable_ai_residue_function, outputs=ai_box_residue_function)

        ai_model_stru_zshot.change(
            fn=on_ai_model_change,
            inputs=ai_model_stru_zshot,
            outputs=[api_key_in_stru_zshot, ai_status_stru_zshot]
        )
        ai_model_seq_zshot.change(
            fn=on_ai_model_change,
            inputs=ai_model_seq_zshot,
            outputs=[api_key_in_seq_zshot, ai_status_seq_zshot]
        )
        ai_model_seq_func.change(
            fn=on_ai_model_change,
            inputs=ai_model_seq_func,
            outputs=[api_key_in_seq_func, ai_status_seq_func]
        )
        ai_model_dd_residue_function.change(
            fn=on_ai_model_change,
            inputs=ai_model_dd_residue_function,
            outputs=[api_key_in_seq_func, ai_status_residue_function]
        )
        
        seq_file_upload.upload(
            fn=handle_file_upload, 
            inputs=seq_file_upload, 
            outputs=[seq_protein_display, seq_sequence_selector, seq_sequence_state, seq_selected_sequence_state, seq_original_file_path_state, seq_current_file_state]
        )

        seq_file_upload.change(
            fn=handle_file_upload, 
            inputs=seq_file_upload, 
            outputs=[seq_protein_display, seq_sequence_selector, seq_sequence_state, seq_selected_sequence_state, seq_original_file_path_state, seq_current_file_state]
        )

        seq_paste_clear_btn.click(
            fn=clear_paste_content_fasta,
            outputs=[seq_paste_content_input, seq_protein_display, seq_sequence_selector, seq_sequence_state, seq_selected_sequence_state, seq_original_file_path_state]
        )

        seq_paste_content_btn.click(
            fn=handle_paste_fasta_detect,
            inputs=seq_paste_content_input,
            outputs=[seq_protein_display, seq_sequence_selector, seq_sequence_state, seq_selected_sequence_state, seq_original_file_path_state, seq_original_paste_content_state]
        )

        seq_sequence_selector.change(
            fn=handle_sequence_change_unified,
            inputs=[seq_sequence_selector, seq_sequence_state, seq_original_file_path_state, seq_original_paste_content_state],
            outputs=[seq_protein_display, seq_current_file_state]
        )


        struct_file_upload.upload(
            fn=handle_file_upload, 
            inputs=struct_file_upload, 
            outputs=[struct_protein_display, struct_chain_selector, struct_chains_state, struct_selected_chain_state, struct_original_file_path_state, struct_current_file_state]
        )

        struct_file_upload.change(
            fn=handle_file_upload, 
            inputs=struct_file_upload, 
            outputs=[struct_protein_display, struct_chain_selector, struct_chains_state, struct_selected_chain_state, struct_original_file_path_state, struct_current_file_state]
        )
        
        struct_paste_clear_btn.click(
            fn=clear_paste_content_pdb,
            outputs=[struct_paste_content_input, struct_protein_display, struct_chain_selector, struct_chains_state, struct_selected_chain_state, struct_original_file_path_state]
        )
        
        struct_paste_content_btn.click(
            fn=handle_paste_pdb_detect,
            inputs=struct_paste_content_input,
            outputs=[struct_protein_display, struct_chain_selector, struct_chains_state, struct_selected_chain_state, struct_original_file_path_state, struct_original_paste_content_state]
        )

        struct_chain_selector.change(
            fn=handle_sequence_change_unified,
            inputs=[struct_chain_selector, struct_chains_state, struct_original_file_path_state, struct_original_paste_content_state],
            outputs=[struct_protein_display, struct_current_file_state] 
        )

        function_fasta_upload.upload(
            fn=handle_file_upload, 
            inputs=function_fasta_upload, 
            outputs=[function_protein_display, function_protein_selector, function_sequence_state, function_selected_sequence_state, function_original_file_path_state, function_current_file_state]
        )
        function_fasta_upload.change(
            fn=handle_file_upload, 
            inputs=function_fasta_upload, 
            outputs=[function_protein_display, function_protein_selector, function_sequence_state, function_selected_sequence_state, function_original_file_path_state, function_current_file_state]
        )
        function_paste_clear_btn.click(
            fn=clear_paste_content_fasta,
            outputs=[function_paste_content_input, function_protein_display, function_protein_selector, function_sequence_state, function_selected_sequence_state, function_original_file_path_state]
        )

        function_paste_content_btn.click(
            fn=handle_paste_fasta_detect,
            inputs=function_paste_content_input,
            outputs=[function_protein_display, function_protein_selector, function_sequence_state, function_selected_sequence_state, function_original_file_path_state, function_original_paste_content_state]
        )

        function_protein_selector.change(
            fn=handle_sequence_change_unified,
            inputs=[function_protein_selector, function_sequence_state, function_original_file_path_state, function_original_paste_content_state],
            outputs=[function_protein_display, function_current_file_state]
        )
        adv_func_task_dd.change(
            fn=update_dataset_choices_fixed,
            inputs=adv_func_task_dd, 
            outputs=adv_func_dataset_cbg
        )
        
        adv_residue_function_fasta_upload.upload(
            fn=handle_file_upload,
            inputs=adv_residue_function_fasta_upload,
            outputs=[adv_residue_function_protein_display, adv_residue_function_selector, adv_residue_function_sequence_state, adv_residue_function_selected_sequence_state, adv_residue_function_original_file_path_state, adv_residue_function_current_file_state]
        )
        adv_residue_function_fasta_upload.change(
            fn=handle_file_upload,
            inputs=adv_residue_function_fasta_upload,
            outputs=[adv_residue_function_protein_display, adv_residue_function_selector, adv_residue_function_sequence_state, adv_residue_function_selected_sequence_state, adv_residue_function_original_file_path_state, adv_residue_function_current_file_state]
        )
        adv_residue_function_paste_clear_btn.click(
            fn=clear_paste_content_fasta,
            outputs=[adv_residue_function_paste_content_input, adv_residue_function_protein_display, adv_residue_function_selector, adv_residue_function_sequence_state, adv_residue_function_selected_sequence_state, adv_residue_function_original_file_path_state]
        )
        adv_residue_function_paste_content_btn.click(
            fn=handle_paste_fasta_detect,
            inputs=adv_residue_function_paste_content_input,
            outputs=[adv_residue_function_protein_display, adv_residue_function_selector, adv_residue_function_sequence_state, adv_residue_function_selected_sequence_state, adv_residue_function_original_file_path_state, adv_residue_function_original_paste_content_state]
        )
        adv_residue_function_selector.change(
            fn=handle_sequence_change_unified,
            inputs=[adv_residue_function_selector, adv_residue_function_sequence_state, adv_residue_function_original_file_path_state, adv_residue_function_original_paste_content_state],
            outputs=[adv_residue_function_protein_display, adv_residue_function_current_file_state]
        )
        adv_residue_function_predict_btn.click(
            fn=handle_protein_residue_function_prediction,
            inputs=[adv_residue_function_task_dd, adv_residue_function_fasta_upload, enable_ai_residue_function, ai_model_dd_residue_function, api_key_in_residue_function, adv_residue_function_model_dd],
            outputs=[adv_residue_function_status_textbox, adv_residue_function_results_df, adv_residue_function_plot_out, adv_residue_function_download_btn, adv_residue_function_ai_expert_html, gr.State()]
        )
        adv_residue_function_protein_chat_btn.click(
            fn=handle_protein_residue_function_prediction_chat,
            inputs=[adv_residue_function_task_dd, adv_residue_function_fasta_upload, enable_ai_residue_function, ai_model_dd_residue_function, api_key_in_residue_function, adv_residue_function_model_dd],
            outputs=[adv_residue_function_status_textbox, adv_residue_function_results_df, adv_residue_function_plot_out, adv_residue_function_download_btn, adv_residue_function_ai_expert_html, gr.State()]
        )
        seq_predict_btn.click(
            fn=handle_mutation_prediction_advance, 
            inputs=[seq_function_dd, seq_file_upload, enable_ai_zshot_seq, ai_model_seq_zshot, api_key_in_seq_zshot, seq_model_dd],
            outputs=[zero_shot_status_box, zero_shot_plot_out, zero_shot_df_out, zero_shot_download_btn, zero_shot_download_path_state, zero_shot_view_controls, zero_shot_full_data_state, zero_shot_ai_expert_html],
            show_progress=True
        )

        struct_predict_btn.click(
            fn=handle_mutation_prediction_advance, 
            inputs=[struct_function_dd, struct_file_upload, enable_ai_zshot_stru, ai_model_stru_zshot, api_key_in_stru_zshot, struct_model_dd], 
            outputs=[zero_shot_status_box, zero_shot_plot_out, zero_shot_df_out, zero_shot_download_btn, zero_shot_download_path_state, zero_shot_view_controls, zero_shot_full_data_state, zero_shot_ai_expert_html],
            show_progress=True
        )

        adv_func_predict_btn.click(
            fn=handle_protein_function_prediction_advance,
            inputs=[adv_func_task_dd, function_fasta_upload, enable_ai_func, ai_model_seq_func, api_key_in_seq_func, adv_func_model_dd, adv_func_dataset_cbg],
            outputs=[function_status_textbox, function_results_df, function_results_plot, function_download_btn, function_ai_expert_html],
            show_progress=True
        )
        
        function_protein_chat_btn.click(
            fn=handle_protein_function_prediction_chat,
            inputs=[adv_func_task_dd, function_fasta_upload, adv_func_model_dd, adv_func_dataset_cbg_chat, enable_ai_func, ai_model_seq_func, api_key_in_seq_func],
            outputs=[function_status_textbox, function_results_df, function_ai_expert_html]
        )

        # handle_venus_pdb_upload is now defined outside create_advanced_tool_tab
        
        venus_pdb_upload.change(
            fn=handle_venus_pdb_upload,
            inputs=[venus_pdb_upload],
            outputs=[venus_pdb_viewer]
        )

        venus_start_btn.click(
            fn=handle_VenusMine,
            inputs=[venus_pdb_upload, venus_protect_start, venus_protect_end,
                venus_mmseqs_threads, venus_mmseqs_iterations, venus_mmseqs_max_seqs,
                venus_cluster_min_seq_id, venus_cluster_threads, venus_top_n, venus_evalue
            ],
            outputs=[
                venus_log_output, 
                venus_tree_image, 
                venus_labels_df,
                venus_tree_download_btn, 
                venus_labels_download_btn, 
                venus_full_zip_btn,
                venus_tree_progress,
                venus_status_indicator
            ],
            show_progress=True
        )
    return demo