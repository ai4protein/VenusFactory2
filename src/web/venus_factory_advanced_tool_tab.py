import gradio as gr
import pandas as pd
import os
import sys
import tempfile
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Dict, Any, List, Generator, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import requests
from dataclasses import dataclass
import re
import json
from web.venus_factory_quick_tool_tab import *
# --- Constants and Mappings ---

# For Zero-Shot Mutation Prediction
MODEL_MAPPING_ZERO_SHOT_SEQUENCE = {
    "ESM2-650M": "esm2",
    "ESM-1v": "esm1v",

}
MODEL_MAPPING_ZERO_SHOT_STRUCTURE = {
    "ESM-IF1": "esmif1",
    "SaProt": "saprot",
    "MIF-ST": "mifst",
    "ProSST-2048": "prosst",
    "ProSSN": "prossn"
}

DATASET_MAPPING_ZERO_SHOT = [
    "Activity",
    "Binding",
    "Expression",
    "Organismal Fitness",
    "Stability"
]

# For Protein Function Prediction
MODEL_MAPPING_FUNCTION = {
    "ESM2-650M": "esm2",
    "Ankh-large": "ankh",
    "ProtBert-uniref50": "protbert",
    "ProtT5-xl-uniref50": "prott5",
}

MODEL_ADAPTER_MAPPING_FUNCTION = {
    "esm2": "esm2_t33_650M_UR50D",
    "ankh": "ankh-large",
    "protbert": "prot_bert",
    "prott5": "prot_t5_xl_uniref50",
}

DATASET_MAPPING_FUNCTION = {
    "Solubility": ["DeepSol", "DeepSoluE", "ProtSolM"],
    "Localization": ["DeepLocBinary", "DeepLocMulti"],
    "Metal ion binding": ["MetalIonBinding"],
    "Stability": ["Thermostability"],
    "Sorting signal": ["SortingSignal"],
    "Optimum temperature": ["DeepET_Topt"]
}

LABEL_MAPPING_FUNCTION = {
    "Solubility": ["Insoluble", "Soluble"],
    "DeepLocBinary": ["Membrane", "Soluble"],
    "DeepLocMulti": [
        "Cytoplasm", "Nucleus", "Extracellular", "Mitochondrion", "Cell membrane",
        "Endoplasmic reticulum", "Plastid", "Golgi apparatus", "Lysosome/Vacuole", "Peroxisome"
    ],
    "Metal ion binding": ["Non-binding", "Binding"],
    "SortingSignal": ['No signal', 'Signal']
}

COLOR_MAP_FUNCTION = {
    "Soluble": "#3B82F6", "Insoluble": "#EF4444", "Membrane": "#F59E0B",
    "Cytoplasm": "#10B981", "Nucleus": "#8B5CF6", "Extracellular": "#F97316",
    "Mitochondrion": "#EC4899", "Cell membrane": "#6B7280", "Endoplasmic reticulum": "#84CC16",
    "Plastid": "#06B6D4", "Golgi apparatus": "#A78BFA", "Lysosome/Vacuole": "#FBBF24", "Peroxisome": "#34D399",
    "Metal ion binding": "#3B82F6", "Non-binding": "#EF4444",
    "Signal": "#3B82F6", "No signal": "#EF4444",
    "Default": "#9CA3AF"
}

REGRESSION_TASKS_FUNCTION = ["Stability", "Optimum temperature"]
REGRESSION_TASKS_FUNCTION_MAX_MIN = {
    "Stability": [40.1995166, 66.8968874],
    "Optimum temperature": [2, 120]
}

DATASET_TO_TASK_MAP = {
    dataset: task for task, datasets in DATASET_MAPPING_FUNCTION.items() for dataset in datasets
}


AI_MODELS = {
    "DeepSeek": {
        "api_base": "https://api.deepseek.com/v1", 
        "model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY"
    },
    "ChatGPT": {
        "api_base": "https://api.openai.com/v1",
        "model": "gpt-4o-mini", 
        "env_key": None
    },
    "Gemini": {
        "api_base": "https://generativelanguage.googleapis.com/v1beta",
        "model": "gemini-1.5-flash",
        "env_key": None
    }
}

def parse_fasta_file(file_path: str) -> str:
    if not file_path: return ""
    try:
        with open(file_path, "r") as f:
            return f.read()
    except Exception as e:
        return f"Error reading FASTA file: {e}"

def parse_pdb_for_sequence(file_path: str) -> str:
    aa_map = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }
    
    sequence = []
    seen_residues = set()
    chain = None
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                chain_id = line[21]
                if chain is None:
                    chain = chain_id
                if chain_id != chain:
                    break
                
                res_id = (chain_id, int(line[22:26]))
                if res_id not in seen_residues:
                    res_name = line[17:20].strip()
                    if res_name in aa_map:
                        sequence.append(aa_map[res_name])
                        seen_residues.add(res_id)
    
    return "".join(sequence)

def update_dataset_choices(task: str) -> gr.CheckboxGroup:
    datasets = DATASET_MAPPING_FUNCTION.get(task, [])
    return gr.update(choices=datasets, value=datasets)


def handle_file_upload(file_obj: Any) -> str:
    if not file_obj:
        return ""
    file_path = file_obj.name
    if file_path.lower().endswith((".fasta", ".fa")):
        return parse_fasta_file(file_path)
    elif file_path.lower().endswith(".pdb"):
        return parse_pdb_for_sequence(file_path)
    else:
        return "Unsupported file type. Please upload a .fasta, .fa, or .pdb file."

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

    file_path = file_obj.name
    
    # # Determine model and type
    # if model_name:
    #     model_type = "structure" if model_name == "ESM-IF1" else "sequence"
    # else:
    #     if file_path.lower().endswith((".fasta", ".fa")):
    #         model_name, model_type = "ESM2-650M", "sequence"

    if file_path.lower().endswith((".fasta", ".fa")):
        model_name, model_type = "ESM2-650M", "sequence"

        # Process FASTA file to keep only the first sequence
        processed_file_path = process_fasta_file(file_path)
        if processed_file_path != file_path:
            file_path = processed_file_path
            yield (
                "⚠️ Multi-sequence FASTA detected. Using only the first sequence for prediction.",
                None, None, gr.update(visible=False), None,
                gr.update(visible=False), None,
                "Processing first sequence only..."
            )
    elif file_path.lower().endswith(".pdb"):
        model_name, model_type = "ESM-IF1", "structure"
    else:
        yield (
            "❌ Error: Unsupported file type.", 
            None, None, gr.update(visible=False), None, 
            gr.update(visible=False), None, 
            "Please upload a .fasta, .fa, or .pdb file."
        )
        return

    # Start prediction
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

    # Handle AI analysis
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
    # Create download files
    temp_dir = Path("temp_outputs")
    temp_dir.mkdir(exist_ok=True)
    timestamp = int(time.time())
    
    csv_path = temp_dir / f"mut_res_{timestamp}.csv"
    heatmap_path = temp_dir / f"mut_map_{timestamp}.html"
    
    display_df.to_csv(csv_path, index=False)
    summary_fig.write_html(heatmap_path)
    
    files_to_zip = {
        str(csv_path): "prediction_results.csv", 
        str(heatmap_path): "prediction_heatmap.html"
    }
    
    if not ai_summary.startswith("❌") and not ai_summary.startswith("AI Analysis"):
        report_path = temp_dir / f"ai_report_{timestamp}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(ai_summary)
        files_to_zip[str(report_path)] = "AI_Analysis_Report.md"

    zip_path = temp_dir / f"pred_mut_{timestamp}.zip"
    zip_path_str = create_zip_archive(files_to_zip, str(zip_path))

    final_status = status if not enable_ai else "✅ Prediction and AI analysis complete!"
    progress(1.0, desc="Complete!")
    yield (
        final_status, summary_fig, display_df, 
        gr.update(visible=True, value=zip_path_str), zip_path_str, 
        gr.update(visible=total_residues > 20), display_df, expert_analysis
    )

def handle_protein_function_prediction_chat(
    task: str,
    fasta_file: str,
    model_name: str,
    datasets: List[str],
    enable_ai: bool,
    ai_model: str,
    user_api_key: str
) -> Generator:
    try:
        import requests
        requests.post("/api/stats/track", json={"module": "mutation_prediction"})
    except Exception:
        pass
    model = model_name if model_name else "ESM2-650M"
    final_datasets = datasets if datasets and len(datasets) > 0 else DATASET_MAPPING_FUNCTION.get(task, [])
    all_results_list = []
    temp_dir = Path("temp_outputs")
    temp_dir.mkdir(exist_ok=True)

    # Create a temporary file to run predictions
    for i, dataset in enumerate(final_datasets):
        try:
            model_key = MODEL_MAPPING_FUNCTION.get(model)
            if not model_key:
                raise ValueError(f"Model key not found for {model}")

            adapter_key = MODEL_ADAPTER_MAPPING_FUNCTION[model_key]
            script_path = Path("src") / "property" / f"{model_key}.py"
            adapter_path = Path("ckpt") / dataset / adapter_key
            output_file = temp_dir / f"temp_{dataset}_{model}.csv"

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
    
    display_df = final_df.copy()
    if 'prediction' in display_df.columns:
        display_df.drop(columns=['prediction'], inplace=True)
    rename_map = {
        'header': "Protein Name", 'sequence': "Sequence", 'predicted_class': "Predicted Class",
        'probabilities': "Confidence Score", 'Dataset': "Dataset"
    }
    display_df.rename(columns=rename_map, inplace=True)

    if enable_ai:
        api_key = get_api_key(ai_model, user_api_key)
        if not api_key: 
            ai_summary = f"❌ No API key found for {ai_model}."
        else:
            ai_config = AIConfig(api_key, ai_model, AI_MODELS[ai_model]["api_base"], AI_MODELS[ai_model]["model"])
            prompt = generate_expert_analysis_prompt(display_df, task)
            ai_summary = call_ai_api(ai_config, prompt)
            expert_analysis = format_expert_response(ai_summary)

    final_status = "✅ All predictions completed!"
    if enable_ai and not ai_summary.startswith("❌"):
        final_status += " AI analysis included."
    return final_status, display_df, expert_analysis



def handle_protein_function_prediction_advance(
    task: str, 
    fasta_file: Any, 
    enable_ai: bool, 
    ai_model: str, 
    user_api_key: str, 
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
        print(f"DEBUG: Using provided datasets: {final_datasets}")
    else:
        final_datasets = DATASET_MAPPING_FUNCTION.get(task, [])
        print(f"DEBUG: Using default datasets for {task}: {final_datasets}")
    
    print(f"DEBUG: Final datasets to process: {final_datasets}")

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
    temp_dir = Path("temp_outputs")
    temp_dir.mkdir(exist_ok=True)

    # Run predictions for each dataset
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
            output_file = temp_dir / f"temp_{dataset}_{model}.csv"
            
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
    # Concatenate all results and keep Dataset column
    final_df = pd.concat(all_results_list, ignore_index=True).fillna('N/A')
    
    plot_fig = generate_plots_for_all_results(final_df)
    display_df = final_df.copy()

    # Map prediction values to text labels BEFORE renaming columns
    def map_labels(row):
        current_task = DATASET_TO_TASK_MAP.get(row.get('Dataset', ''), task)
        
        # For regression tasks, return the numeric value as is
        if current_task in REGRESSION_TASKS_FUNCTION: 
            scaled_value = row.get("prediction")
            if pd.notna(scaled_value) and scaled_value != 'N/A' and current_task in REGRESSION_TASKS_FUNCTION_MAX_MIN:
                try:
                    scaled_value = float(scaled_value)
                    min_val, max_val = REGRESSION_TASKS_FUNCTION_MAX_MIN[current_task]
                    original_value = scaled_value * (max_val - min_val) + min_val
                    return round(original_value, 2)
                
                except (ValueError, TypeError):
                    return scaled_value

            return scaled_value

        
        # For classification tasks, map to text labels
        labels_key = ("DeepLocMulti" if row.get('Dataset') == "DeepLocMulti" 
                     else "DeepLocBinary" if row.get('Dataset') == "DeepLocBinary" 
                     else current_task)
        labels = LABEL_MAPPING_FUNCTION.get(labels_key)
        
        pred_val = row.get("prediction", row.get("predicted_class"))
        if pred_val is None or pred_val == "N/A":
            return "N/A"
            
        try:
            pred_val = int(float(pred_val))
            if labels and 0 <= pred_val < len(labels): 
                return labels[pred_val]
        except (ValueError, TypeError):
            pass
        
        return str(pred_val)

    # Apply label mapping
    if "prediction" in display_df.columns:
        display_df["predicted_class"] = display_df.apply(map_labels, axis=1)
    elif "predicted_class" in display_df.columns:
        display_df["predicted_class"] = display_df.apply(map_labels, axis=1)

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
    
    # Truncate sequence display
    if "Sequence" in display_df.columns:
        display_df["Sequence"] = display_df["Sequence"].apply(lambda x: x[:] if isinstance(x, str) and len(x) > 30 else x)

    # Format confidence score to 2 decimal places
    if "Confidence Score" in display_df.columns and "Predicted Class" in display_df.columns:
        def format_confidence(row):
            score = row["Confidence Score"]
            predicted_class = row["Predicted Class"]
            
            if isinstance(score, (float, int)) and score != 'N/A':
                return round(float(score), 2)
            elif isinstance(score, str) and score not in ['N/A', '']:
                try:
                    if score.startswith('[') and score.endswith(']'):
                        prob_str = score.strip('[]')
                        probs = [float(x.strip()) for x in prob_str.split(',')]
                        
                        # Get the current task for this row
                        current_task = DATASET_TO_TASK_MAP.get(row.get('Dataset', ''), task)
                        
                        # For regression, return the actual value
                        if current_task in REGRESSION_TASKS_FUNCTION:
                            return predicted_class
                        else:
                            # For classification, find the index of the predicted class
                            labels_key = ("DeepLocMulti" if row.get('Dataset') == "DeepLocMulti" 
                                         else "DeepLocBinary" if row.get('Dataset') == "DeepLocBinary" 
                                         else current_task)
                            labels = LABEL_MAPPING_FUNCTION.get(labels_key, [])
                            
                            if labels and predicted_class in labels:
                                pred_index = labels.index(predicted_class)
                                # Return the confidence for the predicted class
                                if 0 <= pred_index < len(probs):
                                    return round(probs[pred_index], 2)
                            
                            # If we can't find the index, return the max probability
                            return round(max(probs), 2)
                    else:
                        return round(float(score), 2)
                except (ValueError, IndexError):
                    return score
            return score
        
        display_df["Confidence Score"] = display_df.apply(format_confidence, axis=1)

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
    
    # Create download zip with processed results
    zip_path_str = ""
    try:
        ts = int(time.time())
        zip_dir = temp_dir / f"download_{ts}"
        zip_dir.mkdir()
        
        # Save the processed results with Dataset column
        processed_df_for_save = display_df.copy()
        processed_df_for_save.to_csv(zip_dir / "Result.csv", index=False)
        
        if plot_fig and hasattr(plot_fig, 'data') and plot_fig.data: 
            plot_fig.write_html(str(zip_dir / "results_plot.html"))
        
        if not ai_summary.startswith("❌") and not ai_summary.startswith("AI Analysis"):
            with open(zip_dir / "AI_Report.md", 'w', encoding='utf-8') as f: 
                f.write(f"# AI Report\n\n{ai_summary}")
        
        zip_path = temp_dir / f"func_pred_{ts}.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for file in zip_dir.glob("*"): 
                zf.write(file, file.name)
        zip_path_str = str(zip_path)
    except Exception as e: 
        print(f"Error creating zip file: {e}")

    final_status = "✅ All predictions completed!"
    if enable_ai and not ai_summary.startswith("❌"): 
        final_status += " AI analysis included."
    progress(1.0, desc="Complete!")
    yield final_status, display_df, plot_fig, gr.update(visible=True, value=zip_path_str), expert_analysis


def create_advanced_tool_tab(constant: Dict[str, Any]) -> Dict[str, Any]:
    sequence_models = list(MODEL_MAPPING_ZERO_SHOT_SEQUENCE.keys())
    structure_models = list(MODEL_MAPPING_ZERO_SHOT_STRUCTURE.keys())
    function_models = list(MODEL_MAPPING_FUNCTION.keys())

    with gr.Blocks() as demo:
        with gr.Tabs():
            with gr.TabItem("Directed Evolution: AI-Powered Mutation Prediction"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2):
                         with gr.Tabs():
                            with gr.TabItem("🧬 Sequence-based Model"):
                                gr.Markdown("### Model Configuration")
                                seq_function_dd = gr.Dropdown(choices=DATASET_MAPPING_ZERO_SHOT, label="Select Protein Function", value=DATASET_MAPPING_ZERO_SHOT[0])
                                seq_model_dd = gr.Dropdown(choices=sequence_models, label="Select Sequence-based Model", value=sequence_models[0] if sequence_models else None)
                                seq_file_upload = gr.File(label="Upload FASTA file", file_types=[".fasta", ".fa"])
                                seq_file_example = gr.Examples(
                                    examples=[["./download/P60002.fasta"]],
                                    inputs=seq_file_upload,
                                    label="Click example to load"
                                )
                                seq_protein_display = gr.Textbox(label="Uploaded Protein Sequence", interactive=False, lines=3, max_lines=7)
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
                                struct_function_dd = gr.Dropdown(choices=DATASET_MAPPING_ZERO_SHOT, label="Select Protein Function", value=DATASET_MAPPING_ZERO_SHOT[0])
                                struct_model_dd = gr.Dropdown(choices=structure_models, label="Select Structure-based Model", value=structure_models[0] if structure_models else None)
                                struct_file_upload = gr.File(label="Upload PDB file", file_types=[".pdb"])
                                struct_file_example = gr.Examples(
                                    examples=[["./download/alphafold2_structures/A0A0C5B5G6.pdb"]],
                                    inputs=struct_file_upload,
                                    label="Click example to load"
                                )
                                struct_protein_display = gr.Textbox(label="Uploaded Protein Sequence", interactive=False, lines=3, max_lines=7)
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
                            with gr.TabItem("📈 Prediction Heatmap"):
                                with gr.Row(visible=False) as zero_shot_view_controls:
                                    expand_btn = gr.Button("Show Complete Heatmap", size="sm")
                                    collapse_btn = gr.Button("Show Summary View", size="sm", visible=False)
                                zero_shot_plot_out = gr.Plot(label="Heatmap")
                            with gr.TabItem("📊 Raw Results"):
                                zero_shot_df_out = gr.DataFrame(label="Raw Data")
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
                        adv_func_model_dd = gr.Dropdown(choices=function_models, label="Select Model", value="ESM2-650M")
                        adv_func_task_dd = gr.Dropdown(choices=list(DATASET_MAPPING_FUNCTION.keys()), label="Select Task", value="Solubility")
                        all_possible_datasets = []
                        for datasets_list in DATASET_MAPPING_FUNCTION.values():
                            all_possible_datasets.extend(datasets_list)
                        all_possible_datasets = sorted(list(set(all_possible_datasets))) # Remove duplicates and sort

                        default_datasets_for_solubility = DATASET_MAPPING_FUNCTION.get("Solubility", [])

                        adv_func_dataset_cbg = gr.CheckboxGroup(label="Select Datasets", 
                                                                choices=default_datasets_for_solubility,
                                                                value=default_datasets_for_solubility)
                        adv_func_dataset_cbg_chat = gr.CheckboxGroup(
                                                        choices=all_possible_datasets,
                                                        value=all_possible_datasets,
                                                        visible=False)
                        function_fasta_upload = gr.File(label="Upload FASTA file", file_types=[".fasta", ".fa"])
                        function_fasta_example = gr.Examples(
                            examples=[["./download/P60002.fasta"]],
                            inputs=function_fasta_upload,
                            label="Click example to load"
                        )
                        function_protein_display = gr.Textbox(label="Uploaded Protein", interactive=False, lines=3, max_lines=7)
                        function_protein_chat_btn = gr.Button("Chat API Trigger", visible=False)

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
        def update_dataset_choices_fixed(task):
            choices = DATASET_MAPPING_FUNCTION.get(task, [])
            return gr.CheckboxGroup(choices=choices, value=choices)
        
        enable_ai_zshot_seq.change(fn=toggle_ai_section, inputs=enable_ai_zshot_seq, outputs=ai_box_zshot_seq)
        enable_ai_zshot_stru.change(fn=toggle_ai_section, inputs=enable_ai_zshot_stru, outputs=ai_box_zshot_stru)
        enable_ai_func.change(fn=toggle_ai_section, inputs=enable_ai_func, outputs=ai_box_func)
        
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
        
        seq_file_upload.upload(fn=parse_fasta_file, inputs=seq_file_upload, outputs=seq_protein_display)
        seq_file_upload.change(fn=parse_fasta_file, inputs=seq_file_upload, outputs=seq_protein_display)
        
        struct_file_upload.upload(fn=parse_pdb_for_sequence, inputs=struct_file_upload, outputs=struct_protein_display)
        struct_file_upload.change(fn=parse_pdb_for_sequence, inputs=struct_file_upload, outputs=struct_protein_display)
        
        function_fasta_upload.upload(fn=handle_file_upload, inputs=function_fasta_upload, outputs=function_protein_display)
        function_fasta_upload.change(fn=handle_file_upload, inputs=function_fasta_upload, outputs=function_protein_display)
        
        adv_func_task_dd.change(
            fn=update_dataset_choices_fixed,
            inputs=[adv_func_task_dd], 
            outputs=[adv_func_dataset_cbg]
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

    return demo