import gradio as gr
import pandas as pd
import os
import sys
import subprocess
import time
import zipfile
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Generator, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from dataclasses import dataclass
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# --- 常量与映射 ---
MODEL_MAPPING = {
    "ESM2-650M": "esm2",
    "Ankh-large": "ankh",
    "ProtBert-uniref50": "protbert",
    "ProtT5-xl-uniref50": "prott5",
}

MODEL_ADAPTER_MAPPING = {
    "esm2": "esm2_t33_650M_UR50D",
    "ankh": "ankh-large",
    "protbert": "prot_bert",
    "prott5": "prot_t5_xl_uniref50",
}

DATASET_MAPPING = {
    "Solubility": ["DeepSol", "DeepSoluE", "ProtSolM"],
    "Localization": ["DeepLocBinary", "DeepLocMulti"],
    "Binding": ["MetalIonBinding"],
    "Stability": ["Thermostability"],
    "Sorting signal": ["SortingSignal"],
    "Optimum temperature": ["DeepET_Topt"]
}

LABEL_MAPPING = {
    "Solubility": ["Insoluble", "Soluble"],
    "DeepLocBinary": ["Membrane", "Soluble"],
    "DeepLocMulti": [
        "Cytoplasm", "Nucleus", "Extracellular", "Mitochondrion", "Cell membrane", 
        "Endoplasmic reticulum", "Plastid", "Golgi apparatus", "Lysosome/Vacuole", "Peroxisome"
    ],
    "Binding": ["Non-binding", "Binding"],
    "SortingSignal": ['No signal', 'Signal']
}

COLOR_MAP = {
    "Soluble": "#3B82F6", "Insoluble": "#EF4444", "Membrane": "#F59E0B",
    "Cytoplasm": "#10B981", "Nucleus": "#8B5CF6", "Extracellular": "#F97316",
    "Mitochondrion": "#EC4899", "Cell membrane": "#6B7280", "Endoplasmic reticulum": "#84CC16",
    "Plastid": "#06B6D4", "Golgi apparatus": "#A78BFA", "Lysosome/Vacuole": "#FBBF24", "Peroxisome": "#34D399",
    "Binding": "#3B82F6", "Non-binding": "#EF4444",
    "Signal": "#3B82F6", "No signal": "#EF4444",
    "Default": "#9CA3AF"
}

REGRESSION_TASKS = ["Stability", "Optimum temperature"]
DATASET_TO_TASK_MAP = {
    dataset: task for task, datasets in DATASET_MAPPING.items() for dataset in datasets
}

AI_MODELS = {
    "DeepSeek": {
        "api_base": "https://api.deepseek.com/v1",
        "model": "deepseek-chat"
    }
}


@dataclass
class AIConfig:
    api_key: str
    model_name: str
    api_base: str
    model: str

def get_api_key(ai_provider: str, user_input_key: str = "") -> Optional[str]:
    if user_input_key and user_input_key.strip():
        return user_input_key.strip()
    
    env_var_map = {"DeepSeek": "DEEPSEEK_API_KEY"}
    env_var_name = env_var_map.get(ai_provider)
    if env_var_name and os.getenv(env_var_name):
        return os.getenv(env_var_name)
        
        
    return None

def call_ai_api(config: AIConfig, prompt: str) -> str:
    headers = {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"}
    data = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": "You are a professional bioinformatics analyst. Provide clear, scientific analysis of protein prediction results for biologists and researchers without a deep computational background."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3, "max_tokens": 2000
    }
    try:
        response = requests.post(f"{config.api_base}/chat/completions", headers=headers, json=data, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ API call failed: {str(e)}"

def generate_ai_summary_prompt(results_df: pd.DataFrame, task: str, model: str) -> str:
    prompt = f"Please provide a comprehensive biological interpretation of the protein function prediction results for the task '{task}' using the '{model}' model. The raw data is as follows in JSON format:\n\n{results_df.to_json(orient='records')}\n\n"
    prompt += "Structure your analysis with an Executive Summary, Detailed Analysis per Sequence, Biological Significance, and Recommendations for Next Steps."
    prompt += "Do not output Markdown formatting, only text formatting. Your role is to do the interpretation of the results, don't appear to be interactive"
    return prompt

def sanitize_filename(name: str) -> str:
    name = re.split(r'[|\s/]', name)[0]
    return re.sub(r'[^\w\-. ]', '_', name)

def generate_plots_for_all_results(results_df: pd.DataFrame, dataset_to_task_map: Dict) -> go.Figure:
    """
    REWRITTEN: Generates a robust Plotly figure with a simplified and stable subplot layout
    to prevent the "squished plot" issue.
    """
    plot_df = results_df[
        (results_df['header'] != "ERROR") &
        (results_df['Dataset'].apply(lambda d: dataset_to_task_map.get(d) not in REGRESSION_TASKS))
    ].copy()

    if plot_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title_text="<b>No Visualization Available</b>",
            title_x=0.5,
            xaxis={"visible": False}, 
            yaxis={"visible": False},
            annotations=[{
                "text": "No classification data to display.<br>Try selecting a classification task.",
                "xref": "paper", "yref": "paper", 
                "x": 0.5, "y": 0.5,
                "showarrow": False, 
                "font": {"size": 18, "color": "#6B7280"},
                "xanchor": "center", "yanchor": "middle"
            }],
            paper_bgcolor="white",
            plot_bgcolor="white"
        )
        return fig

    sequences = plot_df['header'].unique()
    datasets = plot_df['Dataset'].unique()
    
    n_sequences = len(sequences)
    n_datasets = len(datasets)
    
    # --- Simplified Subplot Title Generation ---
    subplot_titles = []
    for seq in sequences:
        for dataset in datasets:
            # If there are multiple sequences, combine sequence and dataset names in the title
            if n_sequences > 1:
                title = f"{seq[:15]}<br>{dataset[:20]}"
            # If only one sequence, just use the dataset name
            else:
                title = f"{dataset[:25]}"
            subplot_titles.append(title)
    
    fig = make_subplots(
        rows=n_sequences, 
        cols=n_datasets,
        subplot_titles=subplot_titles,
        vertical_spacing=0.25, # Increased vertical spacing
    )

    for r_idx, seq_header in enumerate(sequences):
        for c_idx, dataset in enumerate(datasets):
            row_data = plot_df[(plot_df['header'] == seq_header) & (plot_df['Dataset'] == dataset)]
            if row_data.empty: continue
            
            row = row_data.iloc[0]
            prob_col_name = next((col for col in row.index if 'probab' in col.lower()), None)
            if not prob_col_name or pd.isna(row[prob_col_name]): continue

            try:
                confidences = json.loads(row[prob_col_name]) if isinstance(row[prob_col_name], str) else row[prob_col_name]
                if not isinstance(confidences, list): continue
                
                task = dataset_to_task_map.get(dataset)
                labels_key = "DeepLocMulti" if dataset == "DeepLocMulti" else "DeepLocBinary" if dataset == "DeepLocBinary" else task
                labels = LABEL_MAPPING.get(labels_key, [f"Class {k}" for k in range(len(confidences))])
                colors = [COLOR_MAP.get(lbl, COLOR_MAP["Default"]) for lbl in labels]
                
                plot_data = sorted(zip(labels, confidences, colors), key=lambda x: x[1], reverse=True)
                sorted_labels, sorted_confidences, sorted_colors = zip(*plot_data)
                
                fig.add_trace(go.Bar(
                    x=sorted_labels, 
                    y=sorted_confidences, 
                    marker_color=sorted_colors,
                ), row=r_idx + 1, col=c_idx + 1)
                
                # Set consistent Y-axis range for all subplots
                fig.update_yaxes(range=[0, 1], row=r_idx + 1, col=c_idx + 1)
                
            except Exception as e:
                print(f"Plotting error for {seq_header}/{dataset}: {e}")

    # If only one sequence, add its name to the main title
    main_title = "<b>Prediction Confidence Scores</b>"
    if n_sequences == 1:
        main_title += f"<br><sub>Sequence: {sequences[0][:80]}</sub>"

    fig.update_layout(
        title=dict(text=main_title, x=0.5, font=dict(size=18)),
        showlegend=False,
        height=350 * n_sequences + 100, # Allocate extra height for titles
        font=dict(family="sans-serif", size=12),
        paper_bgcolor="white",
        plot_bgcolor="#F9FAFB",
    )
    
    # Set Y-axis title for the first column of EACH row
    for r in range(1, n_sequences + 1):
        fig.update_yaxes(title_text="Confidence", row=r, col=1)

    return fig

def create_protein_function_tab(constant: Dict[str, Any] = None) -> Dict[str, Any]:
    
    def parse_fasta_file(file_path: str) -> str:
        if not file_path: return ""
        try:
            with open(file_path, 'r', encoding='utf-8') as f: return f.read()
        except Exception as e: return f"Error parsing FASTA: {e}"

    def display_protein_sequence_from_fasta(file_obj: Any) -> str:
        return parse_fasta_file(file_obj.name) if file_obj else ""

    def update_dataset_choices(task: str) -> gr.CheckboxGroup:
        if not task: return gr.update(choices=[], value=[])
        choices = DATASET_MAPPING.get(task, [])
        return gr.update(choices=choices, value=choices)

    def toggle_ai_section(enable_ai: bool):

        return gr.update(visible=enable_ai)

    def track_usage(module):
        """追踪功能使用次数"""
        try:
            import requests
            requests.post("http://localhost:8000/api/stats/track", 
                         json={"module": module, "timestamp": datetime.now().isoformat()})
        except Exception as e:
            print(f"Failed to track usage: {e}")
            # 统计失败不影响主功能
    
    def handle_prediction(model: str, task: str, datasets: List[str], fasta_file: Any, 
                      enable_ai: bool, ai_model: str, user_api_key: str) -> Generator:
        if not all([model, task, datasets, fasta_file]):
            yield "❌ Error: All fields are required.", pd.DataFrame(), None, gr.update(visible=False), ""
            return

        # 追踪功能使用
        track_usage("function_analysis")

        yield "🚀 Starting predictions...", pd.DataFrame(), None, gr.update(visible=False), "AI analysis will appear here..."
        
        all_results_list, temp_dir = [], Path("temp_outputs")
        temp_dir.mkdir(exist_ok=True)

        for i, dataset in enumerate(datasets):
            status_msg = f"⏳ Running prediction for '{dataset}' ({i+1}/{len(datasets)})..."
            yield status_msg, None, None, gr.update(visible=False), "AI analysis will appear here..."
            try:
                model_key = MODEL_MAPPING[model]
                adapter_key = MODEL_ADAPTER_MAPPING[model_key] 
                script_path = Path("src") / "property" / f"{model_key}.py"
                adapter_path = Path("ckpt") / dataset / adapter_key
                output_file = temp_dir / f"temp_{dataset}_{model}.csv"
                if not script_path.exists() or not adapter_path.exists():
                    raise FileNotFoundError(f"Required files not found for {dataset}")
                
                cmd = [
                    sys.executable, str(script_path.absolute()),
                    "--fasta_file", str(Path(fasta_file.name).absolute()),
                    "--adapter_path", str(adapter_path.absolute()),
                    "--output_csv", str(output_file.absolute()),
                ]
                subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8', errors='ignore')
                # 改成job harsh id
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
        
        raw_final_df = pd.concat(all_results_list, ignore_index=True).fillna('N/A')
        
        plot_fig = generate_plots_for_all_results(raw_final_df, DATASET_TO_TASK_MAP)
        
        display_df = raw_final_df.copy()
        
        rename_map = {}
        if 'header' in display_df.columns: rename_map['header'] = "Header"
        if 'sequence' in display_df.columns: rename_map['sequence'] = "Sequence"
        if 'prediction' in display_df.columns: rename_map['prediction'] = "Predicted Value / Class"
        if 'predicted_class' in display_df.columns: rename_map['predicted_class'] = "Predicted Value / Class"
        if 'probabilities' in display_df.columns: rename_map['probabilities'] = "Confidence Scores"
        display_df.rename(columns=rename_map, inplace=True)
        if "Predicted Value / Class" in display_df.columns:
            def map_labels(row):
                current_task = DATASET_TO_TASK_MAP.get(row['Dataset'], task)
                if current_task in REGRESSION_TASKS:
                    return row["Predicted Value / Class"]
                
                labels_key = "DeepLocMulti" if row['Dataset'] == "DeepLocMulti" else "DeepLocBinary" if row['Dataset'] == "DeepLocBinary" else current_task
                labels = LABEL_MAPPING.get(labels_key)
                pred_val = pd.to_numeric(row["Predicted Value / Class"], errors='coerce')
                
                if labels and pd.notna(pred_val) and 0 <= int(pred_val) < len(labels):
                    return labels[int(pred_val)]
                return row["Predicted Value / Class"]
                
            display_df["Predicted Value / Class"] = display_df.apply(map_labels, axis=1)

        ai_summary = "AI analysis was not enabled."
        if enable_ai:
            yield "🤖 Generating AI summary...", display_df, plot_fig, gr.update(visible=False), "🤖 AI is analyzing the results..."
            api_key = get_api_key(ai_model, user_api_key)
            if not api_key:
                ai_summary = f"❌ No API key found for {ai_model}. Please provide one or set the DEEPSEEK_API_KEY environment variable."
            else:
                ai_config = AIConfig(api_key, ai_model, AI_MODELS[ai_model]["api_base"], AI_MODELS[ai_model]["model"])
                prompt = generate_ai_summary_prompt(display_df, task, model)
                ai_summary = call_ai_api(ai_config, prompt)
        
        zip_path_str = ""
        try:
            zip_temp_dir = temp_dir / f"download_{int(time.time())}"
            zip_temp_dir.mkdir()
            for seq_header in raw_final_df['header'].unique():
                if seq_header == "ERROR": continue
                sanitized_name = sanitize_filename(seq_header)
                seq_df = raw_final_df[raw_final_df['header'] == seq_header]
                seq_df.to_csv(zip_temp_dir / f"{sanitized_name}_result.csv", index=False)
                seq_fig = generate_plots_for_all_results(seq_df, DATASET_TO_TASK_MAP)
                if seq_fig.data:
                    seq_fig.write_html(str(zip_temp_dir / f"{sanitized_name}_fig.html"))
            if not ai_summary.startswith("❌") and not ai_summary.startswith("AI analysis was not enabled"):
                with open(zip_temp_dir / "AI_Analysis_Report.md", 'w', encoding='utf-8') as f:
                    f.write(f"# AI Analysis Report\n\n{ai_summary}")
            zip_path = temp_dir / "function_prediction_results.zip"
            with zipfile.ZipFile(zip_path, 'w') as zf:
                for file in zip_temp_dir.glob("*"): zf.write(file, file.name)
            zip_path_str = str(zip_path)
        except Exception as e:
            print(f"Error creating zip file: {e}")

        final_status = "✅ All predictions completed!"
        if enable_ai and not ai_summary.startswith("❌"):
            final_status += " AI analysis included."
        
        yield final_status, display_df, plot_fig, gr.update(visible=True, value=zip_path_str), ai_summary

    gr.Markdown("## ⚡️ Protein Function Prediction with AI Analysis")
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### Model Configuration")
            model_dd = gr.Dropdown(choices=list(MODEL_MAPPING.keys()), label="Select Model", value="ESM2-650M")
            task_dd = gr.Dropdown(choices=list(DATASET_MAPPING.keys()), label="Select Task", value="Solubility")
            
            with gr.Accordion("Advanced Options (Customize Datasets & AI Analysis)", open=False):
                gr.Markdown("--- \n**Dataset Selection**\nYou can choose the datasets according to your needs. By default, all are selected.")
                default_datasets = DATASET_MAPPING.get("Solubility", [])
                dataset_cbg = gr.CheckboxGroup(label="Select Datasets", 
                                                choices=default_datasets, 
                                                value=default_datasets)
                gr.Markdown("--- \n**AI Analysis**")
                enable_ai_summary = gr.Checkbox(label="Enable AI Summary", value=False)
                with gr.Group(visible=True) as ai_box:
                    ai_model_dropdown = gr.Dropdown(choices=list(AI_MODELS.keys()), label="Select AI Model", value="DeepSeek")
                    api_key_input = gr.Textbox(label="API Key (Optional)", type="password", placeholder="Leave blank to use environment variable")
            
            fasta_file_upload = gr.File(label="Upload FASTA file", file_types=[".fasta", ".fa"])
            fasta_protein_display = gr.Textbox(label="Uploaded Protein", interactive=False, lines=7, max_lines=10)
            predict_btn = gr.Button("🚀 Start Prediction", variant="primary")
        
        with gr.Column(scale=3):
            gr.Markdown("### Prediction Results")
            status_textbox = gr.Textbox(label="Status", interactive=False, lines=2)
            with gr.Tabs():
                with gr.TabItem("📊 Raw Results"):
                    results_df = gr.DataFrame(label="Prediction Data")
                with gr.TabItem("📈 Prediction Plots"):
                    results_plot = gr.Plot(label="Confidence Scores")
                with gr.TabItem("🤖 AI Analysis"):
                    ai_summary_output = gr.Textbox(
                        label="AI Analysis Report",
                        value="AI analysis will appear here...",
                        lines=20, 
                        interactive=False, 
                        show_copy_button=True 
                    )
            download_btn = gr.DownloadButton("💾 Download Results", visible=False)

    # Event Listeners
    fasta_file_upload.upload(fn=display_protein_sequence_from_fasta, inputs=fasta_file_upload, outputs=fasta_protein_display)
    task_dd.change(fn=update_dataset_choices, inputs=task_dd, outputs=dataset_cbg)
    enable_ai_summary.change(fn=toggle_ai_section, inputs=enable_ai_summary, outputs=ai_box)
    
    predict_btn.click(
        fn=handle_prediction, 
        inputs=[model_dd, task_dd, dataset_cbg, fasta_file_upload, enable_ai_summary, ai_model_dropdown, api_key_input], 
        outputs=[status_textbox, results_df, results_plot, download_btn, ai_summary_output]
    )
    
    return {}

if __name__ == "__main__":
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        create_protein_function_tab()
    demo.launch()
