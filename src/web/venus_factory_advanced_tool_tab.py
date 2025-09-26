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
from .utils.paste_content_handler import process_pasted_content
from web.venus_factory_quick_tool_tab import *
# --- Constants and Mappings ---

MODEL_MAPPING_ZERO_SHOT = {
    "ProSST-2048": "prosst",
    "ProtSSN": "protssn",
    "VenusPLM": "venusplm",
    "ESM2-650M": "esm2",
    "ESM-1v": "esm1v",
    "ESM-1b": "esm1b",
    "ESM-IF1": "esmif1",
    "SaProt": "saprot",
    "MIF-ST": "mifst"
}

DATASET_MAPPING_ZERO_SHOT = [
    "Activity",
    "Binding",
    "Expression",
    "Organismal Fitness",
    "Stability"
]

MODEL_MAPPING_FUNCTION = {
    "ESM2-650M": "esm2",
    "Ankh-large": "ankh",
    "ProtBert": "protbert",
    "ProtT5-xl-uniref50": "prott5",
}

MODEL_RESIDUE_MAPPING_FUNCTION = {
    "ESM2-650M": "esm2",
    "Ankh-large": "ankh",
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
    if not file_path: return ""
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

def run_zero_shot_prediction(model_type: str, model_name: str, file_path: str) -> Tuple[str, pd.DataFrame]:
    try:
        temp_dir = Path("temp_outputs")
        temp_dir_ = temp_dir / "Zero_shot_result"
        timestamp = str(int(time.time()))
        sequence_dir = temp_dir_ / timestamp
        sequence_dir.mkdir(parents=True, exist_ok=True)
        output_csv = sequence_dir / f"{model_type}.csv"
        script_name = MODEL_MAPPING_ZERO_SHOT.get(model_name)
        
        
        if not script_name:
            return f"Error: Model '{model_name}' not found in model mapping.", pd.DataFrame()

        script_path = f"src/mutation/models/{script_name}.py"
            
        if not os.path.exists(script_path):
            return f"Script not found: {script_path}", pd.DataFrame()
        
        file_argument = "--pdb_file" if file_path.lower().endswith(".pdb") else "--fasta_file"
        
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
    temp_dir = Path("temp_outputs")
    temp_dir_ = temp_dir / "Zero_shot_result"
    timestamp = str(int(time.time()))
    session_dir = temp_dir_ / timestamp
    session_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = session_dir/ f"mut_res.csv"
    heatmap_path = session_dir/ f"mut_map.html"
    
    display_df.to_csv(csv_path, index=False)
    summary_fig.write_html(heatmap_path)
    
    files_to_zip = {
        str(csv_path): "prediction_results.csv", 
        str(heatmap_path): "prediction_heatmap.html"
    }
    
    if not ai_summary.startswith("❌") and not ai_summary.startswith("AI Analysis"):
        report_path = session_dir / f"ai_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(ai_summary)
        files_to_zip[str(report_path)] = "AI_Analysis_Report.md"

    zip_path = session_dir / f"pred_mut.zip"
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
    temp_dir_ = temp_dir / "Protein_Function"
    timestamp = str(int(time.time()))
    function_dir = temp_dir_ / timestamp
    function_dir.mkdir(parents=True, exist_ok=True)

    for i, dataset in enumerate(final_datasets):
        try:
            model_key = MODEL_MAPPING_FUNCTION.get(model)
            if not model_key:
                raise ValueError(f"Model key not found for {model}")

            adapter_key = MODEL_ADAPTER_MAPPING_FUNCTION[model_key]
            script_path = Path("src") / "property" / f"{model_key}.py"
            adapter_path = Path("ckpt") / dataset / adapter_key
            output_file = function_dir/ f"temp_{dataset}_{model}.csv"

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
    temp_dir = Path("temp_outputs")
    temp_dir_ = temp_dir / "Protein_Function"
    timestamp = str(int(time.time()))
    function_dir = temp_dir_ / timestamp
    function_dir.mkdir(parents=True, exist_ok=True)

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
            output_file = function_dir / f"temp_{dataset}_{model}.csv"
            
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

    def map_labels(row):
        current_task = DATASET_TO_TASK_MAP.get(row.get('Dataset', ''), task)
        
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

        if row.get('Dataset') == 'SortingSignal':
            predictions_str = row['predicted_class']
            predictions = json.loads(predictions_str)
            if all(p == 0 for p in predictions):
                return "No signal"
            # Get labels for SortingSignal
            signal_labels = ["CH", 'GPI', "MT", "NES", "NLS", "PTS", "SP", "TM", "TH"]
            active_labels = []
            # Find indices where prediction is 1 (active labels)
            for i, pred in enumerate(predictions):
                if pred == 1:
                    active_labels.append(signal_labels[i])
            print(predictions)
            # Return concatenated labels or "None" if no active labels
            return "_".join(active_labels) if active_labels else "None"
        
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
    
    if "Sequence" in display_df.columns:
        display_df["Sequence"] = display_df["Sequence"].apply(lambda x: x[:] if isinstance(x, str) and len(x) > 30 else x)

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
    
    zip_path_str = ""
    try:
        ts = int(time.time())
        temp_dir = Path("temp_outputs")
        temp_dir_ = temp_dir / "Downloads_zip"
        timestamp = str(int(time.time()))
        zip_dir = temp_dir_ / timestamp
        zip_dir.mkdir(parents=True, exist_ok=True)

        processed_df_for_save = display_df.copy()
        processed_df_for_save.to_csv(zip_dir / "Result.csv", index=False)
        
        if plot_fig and hasattr(plot_fig, 'data') and plot_fig.data: 
            plot_fig.write_html(str(zip_dir / "results_plot.html"))
        
        if not ai_summary.startswith("❌") and not ai_summary.startswith("AI Analysis"):
            with open(zip_dir / "AI_Report.md", 'w', encoding='utf-8') as f: 
                f.write(f"# AI Report\n\n{ai_summary}")
        
        zip_path = function_dir / f"func_pred_{ts}.zip"
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


def handle_protein_residue_function_prediction_chat(
    task: str,
    fasta_file: Any,
    enable_ai: bool,
    ai_model: str,
    user_api_key: str,
    model_name: str,
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
    temp_dir = Path("temp_outputs")
    temp_dir_ = temp_dir /  "Residue_save"
    timestamp = str(int(time.time()))
    residue_save_dir = temp_dir_ / timestamp
    residue_save_dir.mkdir(parents=True, exist_ok=True)

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
        output_file = residue_save_dir/ f"{dataset}_{model}.csv"

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

def create_advanced_tool_tab(constant: Dict[str, Any]) -> Dict[str, Any]:
    sequence_models = ["VenusPLM","ESM2-650M", "ESM-1v", "ESM-1b"]
    structure_models = [ "ProSST-2048", "ProtSSN", "ESM-IF1", "SaProt", "MIF-ST"]
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

        def clear_paste_content_pdb():
            return "No file selected", "No file selected", gr.update(choices=["A"], value="A", visible=False), {}, "A", ""

        def clear_paste_content_fasta():
            return "No file selected", "No file selected", gr.update(choices=["Sequence_1"], value="Sequence_1", visible=False), {}, "Sequence_1", ""
        
        def update_dataset_choices_fixed(task):
            choices = DATASET_MAPPING_FUNCTION.get(task, [])
            return gr.CheckboxGroup(choices=choices, value=choices)
        
        def toggle_ai_section_simple(is_checked: bool):
            return gr.update(visible=is_checked)
        
        def on_ai_model_change_simple(ai_provider: str) -> tuple:
            if ai_provider == "DeepSeek":
                return gr.update(visible=False), gr.update(visible=True)
            else:
                return gr.update(visible=True), gr.update(visible=False)
        
        enable_ai_zshot_seq.change(fn=toggle_ai_section_simple, inputs=enable_ai_zshot_seq, outputs=ai_box_zshot_seq)
        enable_ai_zshot_stru.change(fn=toggle_ai_section_simple, inputs=enable_ai_zshot_stru, outputs=ai_box_zshot_stru)
        enable_ai_func.change(fn=toggle_ai_section_simple, inputs=enable_ai_func, outputs=ai_box_func)
        enable_ai_residue_function.change(fn=toggle_ai_section_simple, inputs=enable_ai_residue_function, outputs=ai_box_residue_function)

        ai_model_stru_zshot.change(
            fn=on_ai_model_change_simple,
            inputs=ai_model_stru_zshot,
            outputs=[api_key_in_stru_zshot, ai_status_stru_zshot]
        )
        ai_model_seq_zshot.change(
            fn=on_ai_model_change_simple,
            inputs=ai_model_seq_zshot,
            outputs=[api_key_in_seq_zshot, ai_status_seq_zshot]
        )
        ai_model_seq_func.change(
            fn=on_ai_model_change_simple,
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

        def handle_paste_fasta_detect(fasta_content):
            result = parse_fasta_paste_content(fasta_content)
            return result

        seq_paste_content_btn.click(
            fn=handle_paste_fasta_detect,
            inputs=seq_paste_content_input,
            outputs=[seq_protein_display, seq_sequence_selector, seq_sequence_state, seq_selected_sequence_state, seq_original_file_path_state, seq_original_paste_content_state]
        )

        def handle_sequence_change_unified(selected_chain, chains_dict, original_file_path, original_paste_content):
            # Check for None or empty file path
            if not original_file_path:
                return "No file selected", ""
            
            if original_file_path.endswith('.fasta'):
                if original_paste_content:
                    return handle_paste_sequence_selection(selected_chain, chains_dict, original_paste_content)
                else:
                    return handle_fasta_sequence_change(selected_chain, chains_dict, original_file_path)
            elif original_file_path.endswith('.pdb'):
                if original_paste_content:
                    return handle_paste_chain_selection(selected_chain, chains_dict, original_paste_content)
                else:
                    return handle_pdb_chain_change(selected_chain, chains_dict, original_file_path)
            else:
                # Default case for no file selected
                return "No file selected", ""

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
        
        def handle_paste_detect(pdb_content):
            result = parse_pdb_paste_content(pdb_content)
            return result + (pdb_content,) 

        struct_paste_content_btn.click(
            fn=handle_paste_detect,
            inputs=struct_paste_content_input,
            outputs=[struct_protein_display, struct_chain_selector, struct_chains_state, struct_selected_chain_state, struct_original_file_path_state, struct_original_paste_content_state]
        )

        def handle_chain_change_unified(selected_chain, chains_dict, original_file_path, original_paste_content):
            # Check for None or empty file path
            if not original_file_path:
                return "No file selected", ""
                
            if original_paste_content:
                return handle_paste_chain_selection(selected_chain, chains_dict, original_paste_content)
            else:
                return handle_pdb_chain_change(selected_chain, chains_dict, original_file_path)

        struct_chain_selector.change(
            fn=handle_chain_change_unified,
            inputs=[struct_chain_selector, struct_chains_state, struct_original_file_path_state, struct_original_paste_content_state],
            outputs=[struct_protein_display, struct_file_upload] 
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
            outputs=[function_protein_display, function_fasta_upload]
        )
        adv_func_task_dd.change(
            fn=update_dataset_choices_fixed,
            inputs=[adv_func_task_dd], 
            outputs=[adv_func_dataset_cbg]
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
            outputs=[adv_residue_function_protein_display, adv_residue_function_fasta_upload]
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

    return demo