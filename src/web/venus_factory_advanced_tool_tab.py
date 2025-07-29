import gradio as gr
import pandas as pd
import os
import sys
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
MODEL_MAPPING_ZERO_SHOT = {
    "ESM2-650M": "esm2",
    "ESM-IF1": "esmif1",
    "ESM-1v": "esm1v",
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
    "Binding": ["MetalIonBinding"],
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
    "Binding": ["Non-binding", "Binding"],
    "SortingSignal": ['No signal', 'Signal']
}

COLOR_MAP_FUNCTION = {
    "Soluble": "#3B82F6", "Insoluble": "#EF4444", "Membrane": "#F59E0B",
    "Cytoplasm": "#10B981", "Nucleus": "#8B5CF6", "Extracellular": "#F97316",
    "Mitochondrion": "#EC4899", "Cell membrane": "#6B7280", "Endoplasmic reticulum": "#84CC16",
    "Plastid": "#06B6D4", "Golgi apparatus": "#A78BFA", "Lysosome/Vacuole": "#FBBF24", "Peroxisome": "#34D399",
    "Binding": "#3B82F6", "Non-binding": "#EF4444",
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
        "model": "deepseek-chat"
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

def handle_protein_function_prediction_advance(
    task: str, 
    fasta_file: Any, 
    enable_ai: bool, 
    ai_model: str, 
    user_api_key: str, 
    model_override: Optional[str] = None, 
    datasets_override: Optional[List[str]] = None
    ) -> Generator:
    """Handle protein function prediction workflow."""
    model = model_override if model_override else "ESM2-650M"
    datasets = (datasets_override if datasets_override is not None 
               else DATASET_MAPPING_FUNCTION.get(task, []))

    if not all([task, datasets, fasta_file]):
        yield (
            "❌ Error: Task, Datasets, and FASTA file are required.", 
            pd.DataFrame(), None, gr.update(visible=False), 
            "Please provide all required inputs."
        )
        return

    yield (
        f"🚀 Starting predictions with {model}...", 
        pd.DataFrame(), None, gr.update(visible=False), 
        "AI analysis will appear here..."
    )
    
    all_results_list = []
    temp_dir = Path("temp_outputs")
    temp_dir.mkdir(exist_ok=True)

    # Run predictions for each dataset
    for i, dataset in enumerate(datasets):
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
    if enable_ai:
        yield "🤖 Generating AI summary...", display_df, plot_fig, gr.update(visible=False), "🤖 AI is analyzing..."
        api_key = get_api_key(ai_model, user_api_key)
        if not api_key: 
            ai_summary = f"❌ No API key found for {ai_model}."
        else:
            ai_config = AIConfig(api_key, ai_model, AI_MODELS[ai_model]["api_base"], AI_MODELS[ai_model]["model"])
            prompt = generate_ai_summary_prompt(display_df, task, model)
            ai_summary = call_ai_api(ai_config, prompt)
    
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
    
    yield final_status, display_df, plot_fig, gr.update(visible=True, value=zip_path_str), ai_summary


def create_advanced_tool_tab(constant: Dict[str, Any]) -> Dict[str, Any]:
    sequence_models = list(MODEL_MAPPING_ZERO_SHOT.keys())
    structure_models = [k for k, v in MODEL_MAPPING_ZERO_SHOT.items() if v == 'esmif1'] # Example filter for structure models
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
                                enable_ai_zshot = gr.Checkbox(label="Enable AI Summary", value=False)
                                with gr.Group(visible=False) as ai_box_zshot:
                                    ai_model_dd_zshot = gr.Dropdown(choices=list(AI_MODELS.keys()), value="DeepSeek", label="Select AI Model")
                                    api_key_in_zshot = gr.Textbox(label="API Key (Optional)", type="password", placeholder="Leave blank for env var")
                                seq_predict_btn = gr.Button("🚀 Start Prediction (Sequence)", variant="secondary")

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
                                enable_ai_zshot = gr.Checkbox(label="Enable AI Summary", value=False)
                                with gr.Group(visible=False) as ai_box_zshot:
                                    ai_model_dd_zshot = gr.Dropdown(choices=list(AI_MODELS.keys()), value="DeepSeek", label="Select AI Model")
                                    api_key_in_zshot = gr.Textbox(label="API Key (Optional)", type="password", placeholder="Leave blank for env var")
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
                            with gr.TabItem("🤖 AI Analysis"):
                                zero_shot_ai_out = gr.Textbox(label="AI Analysis Report", value="AI analysis will appear here...", lines=20, interactive=False, show_copy_button=True)
                        
                        zero_shot_download_btn = gr.DownloadButton("💾 Download Results", visible=False)
                        zero_shot_full_data_state = gr.State()
                        zero_shot_download_path_state = gr.State()

            with gr.TabItem("Protein Function Prediction"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2):
                        adv_func_model_dd = gr.Dropdown(choices=function_models, label="Select Model", value="ESM2-650M")
                        adv_func_task_dd = gr.Dropdown(choices=list(DATASET_MAPPING_FUNCTION.keys()), label="Select Task", value="Solubility")
                        default_datasets = DATASET_MAPPING_FUNCTION.get("Solubility", [])
                        adv_func_dataset_cbg = gr.CheckboxGroup(label="Select Datasets", 
                                                                choices=default_datasets, 
                                                                value=default_datasets)
                        function_fasta_upload = gr.File(label="Upload FASTA file", file_types=[".fasta", ".fa"])
                        function_fasta_example = gr.Examples(
                            examples=[["./download/P60002.fasta"]],
                            inputs=function_fasta_upload,
                            label="Click example to load"
                        )
                        function_protein_display = gr.Textbox(label="Uploaded Protein", interactive=False, lines=3, max_lines=7)

                        gr.Markdown("### Configure AI Analysis (Optional)")
                        with gr.Accordion("AI Settings", open=True):
                            enable_ai_func = gr.Checkbox(label="Enable AI Summary", value=False)
                            with gr.Group(visible=False) as ai_box_func:
                                ai_model_dd_func = gr.Dropdown(choices=list(AI_MODELS.keys()), label="Select AI Model", value="DeepSeek")
                                api_key_in_func = gr.Textbox(label="API Key (Optional)", type="password", placeholder="Leave blank to use environment variable")
                        adv_func_predict_btn = gr.Button("🚀 Start Prediction (Advanced)", variant="primary")

                    with gr.Column(scale=3):
                        gr.Markdown("### Results")
                        function_status_textbox = gr.Textbox(label="Status", interactive=False)
                        with gr.Tabs():
                            with gr.TabItem("📊 Raw Results"):
                                function_results_df = gr.DataFrame(label="Prediction Data", column_widths=["20%", "20%", "20%", "20%", "20%"])
                            with gr.TabItem("📈 Prediction Plots"):
                                function_results_plot = gr.Plot(label="Confidence Scores")
                            with gr.TabItem("🤖 AI Analysis"):
                                function_ai_summary_output = gr.Textbox(label="AI Analysis Report", value="AI analysis will appear here...", lines=20, interactive=False, show_copy_button=True)
                        
                        function_download_btn = gr.DownloadButton("💾 Download Results", visible=False)

        enable_ai_zshot.change(fn=toggle_ai_section, inputs=enable_ai_zshot, outputs=ai_box_zshot)

        seq_file_upload.upload(fn=parse_fasta_file, inputs=seq_file_upload, outputs=seq_protein_display)
        seq_file_upload.change(fn=parse_fasta_file, inputs=seq_file_upload, outputs=seq_protein_display)
        seq_predict_btn.click(
            fn=handle_mutation_prediction, 
            inputs=[seq_function_dd, seq_file_upload, enable_ai_zshot, ai_model_dd_zshot, api_key_in_zshot, seq_model_dd],
            outputs=[zero_shot_status_box, zero_shot_plot_out, zero_shot_df_out, zero_shot_download_btn, zero_shot_download_path_state, zero_shot_view_controls, zero_shot_full_data_state, zero_shot_ai_out]
        )

        struct_file_upload.upload(fn=parse_pdb_for_sequence, inputs=struct_file_upload, outputs=struct_protein_display)
        struct_file_upload.change(fn=parse_pdb_for_sequence, inputs=struct_file_upload, outputs=struct_protein_display)
        struct_predict_btn.click(
            fn=handle_mutation_prediction, 
            inputs=[struct_function_dd, struct_file_upload, enable_ai_zshot, ai_model_dd_zshot, api_key_in_zshot, struct_model_dd], 
            outputs=[zero_shot_status_box, zero_shot_plot_out, zero_shot_df_out, zero_shot_download_btn, zero_shot_download_path_state, zero_shot_view_controls, zero_shot_full_data_state, zero_shot_ai_out]
        )

        enable_ai_func.change(fn=toggle_ai_section, inputs=enable_ai_func, outputs=ai_box_func)
        
        function_fasta_upload.upload(fn=handle_file_upload, inputs=function_fasta_upload, outputs=function_protein_display)
        function_fasta_upload.change(fn=handle_file_upload, inputs=function_fasta_upload, outputs=function_protein_display)
        
        adv_func_task_dd.change(fn=update_dataset_choices, inputs=adv_func_task_dd, outputs=adv_func_dataset_cbg)

        adv_func_predict_btn.click(
            fn=handle_protein_function_prediction_advance,
            inputs=[adv_func_task_dd, function_fasta_upload, enable_ai_func, ai_model_dd_func, api_key_in_func, adv_func_model_dd, adv_func_dataset_cbg],
            outputs=[function_status_textbox, function_results_df, function_results_plot, function_download_btn, function_ai_summary_output]
        )

    return {}