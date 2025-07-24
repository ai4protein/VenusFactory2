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

# --- Constants and Mappings ---

# For Zero-Shot Mutation Prediction
MODEL_MAPPING_ZERO_SHOT = {
    "ESM2-650M": "esm2",
    "ESM-IF1": "esmif1",
    "ESM-1v": "esm1v",
    "SaProt": "saprot",
    "MIF-ST": "mifst",
    "ProSST-2048": "prosst",
    "ProtSSN": "protssn"
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
DATASET_TO_TASK_MAP = {
    dataset: task for task, datasets in DATASET_MAPPING_FUNCTION.items() for dataset in datasets
}

AI_MODELS = {
    "DeepSeek": {
        "api_base": "https://api.deepseek.com/v1",
        "model": "deepseek-chat"
    }
}

# --- General Settings & AI Functions ---
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
            {"role": "system", "content": "You are an expert protein scientist. Provide clear, structured, and insightful analysis based on the data provided. Do not ask interactive questions."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 2000
    }
    try:
        response = requests.post(f"{config.api_base}/chat/completions", headers=headers, json=data, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ API call failed: {str(e)}"

def generate_mutation_ai_prompt(results_df: pd.DataFrame, model_name: str, function_selection: str) -> Optional[str]:
    if 'mutant' not in results_df.columns:
        return "Error: 'mutant' column not found in results."
    mutant_col = 'mutant'
    
    score_col = next((col for col in results_df.columns if 'score' in col.lower()), None)
    if not score_col:
        if len(results_df.columns) > 1:
            score_col = results_df.columns[1]
        else:
            return "Error: Score column not found in results."

    num_rows = len(results_df)
    top_count = max(1, int(num_rows * 0.05)) if num_rows >= 5 else num_rows
    lowest_count = max(1, int(num_rows * 0.05)) if num_rows >= 5 else 0

    top_mutations = results_df.head(top_count)
    lowest_mutations = results_df.tail(lowest_count) if lowest_count > 0 else pd.DataFrame()

    top_mutations_str = top_mutations[[mutant_col, score_col]].to_string(index=False)
    lowest_mutations_str = lowest_mutations[[mutant_col, score_col]].to_string(index=False) if not lowest_mutations.empty else "N/A"

    prompt = f"""
        Please act as an expert protein engineer and analyze the following mutation prediction results generated by the '{model_name}' model for the function '{function_selection}'.
        A deep mutational scan was performed. The results are sorted from most beneficial to least beneficial based on the '{score_col}'. Below are the top 5% and bottom 5% of mutations.

        ### Top 5% Predicted Mutations (Potentially Most Beneficial):
        ```
        {top_mutations_str}
        ```

        ### Bottom 5% Predicted Mutations (Potentially Most Detrimental):
        ```
        {lowest_mutations_str}
        ```

        ### Your Analysis Task:
        Based on this data, provide a structured scientific analysis report that includes the following sections:
        1.  **Executive Summary**: Briefly summarize the key findings. Are there clear hotspot regions for beneficial mutations?
        2.  **Analysis of Beneficial Mutations**: Discuss the top mutations. What biochemical properties might these mutations be altering to improve '{function_selection}'?
        3.  **Analysis of Detrimental Mutations & Sequence Conservation**: Discuss the most harmful mutations. What do these suggest about functionally critical residues for '{function_selection}'?
        4.  **Recommendations for Experimentation**: Suggest 3-5 specific point mutations that are the most promising candidates for experimental validation. Justify your choices.

        Provide a concise, clear, and insightful report in a professional scientific tone.
        """
    return prompt

def generate_ai_summary_prompt(results_df: pd.DataFrame, task: str, model: str) -> str:
    prompt = f"Please provide a comprehensive biological interpretation of the protein function prediction results for the task '{task}' using the '{model}' model. The raw data is as follows in JSON format:\n\n{results_df.to_json(orient='records')}\n\n"
    prompt += "Structure your analysis with an Executive Summary, Detailed Analysis per Sequence, Biological Significance, and Recommendations for Next Steps."
    prompt += "Do not output Markdown formatting, only text formatting. Your role is to do the interpretation of the results, don't appear to be interactive."
    return prompt

# --- File Parsing and Helper Functions ---

def parse_fasta_file(file_path: str) -> str:
    if not file_path: return ""
    try:
        with open(file_path, 'r') as f:
            return "".join([l.strip() for l in f if not l.startswith('>')])
    except Exception as e:
        return f"Error reading FASTA file: {e}"

def parse_pdb_for_sequence(file_path: str) -> str:
    aa_code_map = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}
    sequence, seen_residues, current_chain = [], set(), None
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith("ATOM"):
                    chain_id = line[21]
                    if current_chain is None: current_chain = chain_id
                    if chain_id != current_chain: break # Only read the first chain
                    res_name, res_seq_num = line[17:20].strip(), int(line[22:26])
                    residue_id = (chain_id, res_seq_num)
                    if residue_id not in seen_residues:
                        if res_name in aa_code_map:
                            sequence.append(aa_code_map[res_name])
                            seen_residues.add(residue_id)
        return "".join(sequence)
    except Exception as e:
        return f"Error reading PDB file: {e}"

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

def sanitize_filename(name: str) -> str:
    name = re.split(r'[|\s/]', name)[0]
    return re.sub(r'[^\w\-. ]', '_', name)

def get_total_residues_count(df: pd.DataFrame) -> int:
    if 'mutant' not in df.columns:
        return 0
    try:
        # Extract numeric part (position) from the 'mutant' string (e.g., A123G -> 123)
        positions = df['mutant'].str.extract(r'(\d+)').dropna()
        if positions.empty:
            return 0
        return positions[0].astype(int).nunique()
    except Exception:
        return 0

def toggle_ai_section(is_checked: bool):
    return gr.update(visible=is_checked)

def create_zip_archive(files_to_zip: Dict[str, str], zip_filename: str) -> str:
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zf:
        for src, arc in files_to_zip.items():
            if os.path.exists(src):
                zf.write(src, arcname=arc)
    return zip_filename

def update_dataset_choices(task: str) -> gr.CheckboxGroup:
    datasets = DATASET_MAPPING_FUNCTION.get(task, [])
    return gr.CheckboxGroup.update(choices=datasets, value=datasets)


# --- Zero-Shot Prediction Logic ---

def run_zero_shot_prediction(model_type: str, model_name: str, file_path: str) -> Tuple[str, pd.DataFrame]:
    try:
        output_csv = f"temp_{model_type}_{int(time.time())}.csv"
        # Ensure the model name from the UI maps to the script name
        script_name = MODEL_MAPPING_ZERO_SHOT.get(model_name)
        if not script_name:
            return f"Error: Model '{model_name}' does not have a corresponding script.", pd.DataFrame()

        script_path = f"src/mutation/models/{script_name}.py"
        file_argument = "--pdb_file" if model_type == "structure" else "--fasta_file"
        cmd = [sys.executable, script_path, file_argument, file_path, "--output_csv", output_csv]

        # This assumes the scripts exist at the specified path.
        if not os.path.exists(script_path):
             return f"Prediction script not found at: {script_path}", pd.DataFrame()

        subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8', errors='ignore')

        if os.path.exists(output_csv):
            df = pd.read_csv(output_csv)
            os.remove(output_csv) # Clean up temp file
            return "Prediction completed successfully!", df
        return "Prediction completed but no output file was generated.", pd.DataFrame()
    except subprocess.CalledProcessError as e:
        error_detail = e.stderr if e.stderr else e.stdout
        return f"Prediction script failed with error: {error_detail}", pd.DataFrame()
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}", pd.DataFrame()


def prepare_plotly_heatmap_data(df: pd.DataFrame, max_residues: Optional[int] = None) -> Tuple:
    score_col = next((col for col in df.columns if 'score' in col.lower()), None)
    if score_col is None: return (None,) * 6
    
    # Filter for valid mutation format (e.g., A123G)
    valid_mutations_df = df[df['mutant'].apply(lambda m: isinstance(m, str) and bool(re.match(r'^[A-Z]\d+[A-Z]$', m)) and m[0] != m[-1])].copy()
    if valid_mutations_df.empty: return ([], [], np.array([[]]), np.array([[]]), np.array([[]]), score_col)

    valid_mutations_df['rank'] = valid_mutations_df[score_col].rank(method='min', ascending=False).astype(int)
    valid_mutations_df['inverted_rank_bin'] = 11 - np.ceil(valid_mutations_df['rank'] / (len(valid_mutations_df) / 10)).clip(upper=10)
    valid_mutations_df['position'] = valid_mutations_df['mutant'].str[1:-1].astype(int)
    
    sorted_positions = sorted(valid_mutations_df['position'].unique())
    if max_residues is not None:
        sorted_positions = sorted_positions[:max_residues]
        valid_mutations_df = valid_mutations_df[valid_mutations_df['position'].isin(sorted_positions)]

    x_labels = list("ACDEFGHIKLMNPQRSTVWY")
    x_map = {lbl: i for i, lbl in enumerate(x_labels)}
    wt_map = {pos: mut[0] for pos, mut in zip(valid_mutations_df['position'], valid_mutations_df['mutant'])}
    y_labels = [f"{wt_map.get(pos, '?')}{pos}" for pos in sorted_positions]
    y_map = {pos: i for i, pos in enumerate(sorted_positions)}
    
    z_data, rank_matrix, score_matrix = (np.full((len(y_labels), len(x_labels)), np.nan) for _ in range(3))
    
    for _, row in valid_mutations_df.iterrows():
        pos, mut_aa = row['position'], row['mutant'][-1]
        if pos in y_map and mut_aa in x_map:
            y_idx, x_idx = y_map[pos], x_map[mut_aa]
            z_data[y_idx, x_idx] = row['inverted_rank_bin']
            rank_matrix[y_idx, x_idx] = row['rank']
            score_matrix[y_idx, x_idx] = round(row[score_col], 3)
            
    return x_labels, y_labels, z_data, rank_matrix, score_matrix, score_col

def generate_plotly_heatmap(x_labels: List, y_labels: List, z_data: np.ndarray, rank_data: np.ndarray, score_data: np.ndarray, is_partial: bool = False, total_residues: Optional[int] = None) -> go.Figure:
    if z_data is None or z_data.size == 0: return go.Figure().update_layout(title="No data to display")
    
    num_residues = len(y_labels)
    dynamic_height = max(400, min(8000, 30 * num_residues + 150))
    custom_data = np.stack((rank_data, score_data), axis=-1)
    
    fig = go.Figure(data=go.Heatmap(
        z=z_data, x=x_labels, y=y_labels,
        customdata=custom_data,
        hovertemplate="<b>Position</b>: %{y}<br><b>Mutation to</b>: %{x}<br><b>Rank</b>: %{customdata[0]}<br><b>Score</b>: %{customdata[1]}<extra></extra>",
        colorscale='RdYlGn_r', zmin=1, zmax=10, showscale=True,
        colorbar={'title': 'Rank Percentile', 'tickvals': [10, 6, 1], 'ticktext': ['Top 10%', 'Top 50%', 'Lowest 10%']}
    ))
    
    title_text = "Prediction Heatmap"
    if is_partial and total_residues and total_residues > num_residues:
        title_text += f" (Showing first {num_residues} of {total_residues} residues)"
        
    fig.update_layout(title=title_text, xaxis_title='Mutant Amino Acid', yaxis_title='Residue Position', height=dynamic_height, yaxis_autorange='reversed')
    return fig

def handle_mutation_prediction(function_selection: str, file_obj: Any, enable_ai: bool, ai_model: str, user_api_key: str, model_name_override: Optional[str] = None) -> Generator:
    if not file_obj or not function_selection:
        yield "❌ Error: Function and file are required.", None, None, gr.update(visible=False), None, gr.update(visible=False), None, "Please select a function and upload a file."
        return

    file_path = file_obj.name
    if model_name_override:
        model_name = model_name_override
        model_type = "structure" if model_name == "ESM-IF1" else "sequence"
    else:
        # Automatic Model Selection based on file type
        if file_path.lower().endswith((".fasta", ".fa")):
            model_type = "sequence"
            model_name = "ESM2-650M"
        elif file_path.lower().endswith(".pdb"):
            model_type = "structure"
            model_name = "ESM-IF1"
        else:
            yield "❌ Error: Unsupported file type.", None, None, gr.update(visible=False), None, gr.update(visible=False), None, "Please upload a .fasta, .fa, or .pdb file."
            return

    yield f"⏳ Running {model_type} prediction with {model_name}...", None, None, gr.update(visible=False), None, gr.update(visible=False), None, "Prediction in progress..."
    status, df = run_zero_shot_prediction(model_type, model_name, file_path)
    
    if df.empty:
        yield status, go.Figure(layout={'title': 'No results generated'}), pd.DataFrame(), gr.update(visible=False), None, gr.update(visible=False), None, "No results to analyze."
        return
    
    total_residues = get_total_residues_count(df)
    data_tuple = prepare_plotly_heatmap_data(df, max_residues=40)
    
    if data_tuple[0] is None:
        yield status, go.Figure(layout={'title': 'Score column not found'}), df, gr.update(visible=False), None, gr.update(visible=False), df, "Score column not found."
        return

    plot_data = data_tuple[:5]
    summary_fig = generate_plotly_heatmap(*plot_data, is_partial=True, total_residues=total_residues)
    
    ai_summary = "For a complete analysis of the prediction results, activate the AI Analysis feature. Simply go to AI Settings, select Enable AI Summary, and enter a valid API Key."
    if enable_ai:
        yield f"✅ Prediction complete. 🤖 Generating AI summary...", summary_fig, df, gr.update(visible=False), None, gr.update(visible=total_residues > 40), df, "🤖 AI is analyzing the results..."
        api_key = get_api_key(ai_model, user_api_key)
        if not api_key:
            ai_summary = "❌ No API key found. Please provide one or set the environment variable."
        else:
            ai_config = AIConfig(api_key, ai_model, AI_MODELS[ai_model]["api_base"], AI_MODELS[ai_model]["model"])
            prompt = generate_mutation_ai_prompt(df, model_name, function_selection)
            ai_summary = call_ai_api(ai_config, prompt)
    
    full_data_tuple = prepare_plotly_heatmap_data(df)
    full_plot_data = full_data_tuple[:5]
    full_fig = generate_plotly_heatmap(*full_plot_data, is_partial=False, total_residues=total_residues)
    
    temp_dir = Path("temp_outputs"); temp_dir.mkdir(exist_ok=True)
    run_timestamp = int(time.time())
    csv_path = temp_dir / f"temp_mutation_results_{run_timestamp}.csv"
    df.to_csv(csv_path, index=False)
    heatmap_path = temp_dir / f"temp_mutation_heatmap_{run_timestamp}.html"
    full_fig.write_html(heatmap_path)
    
    files_to_zip = {str(csv_path): "prediction_results.csv", str(heatmap_path): "prediction_heatmap.html"}
    if not ai_summary.startswith("❌") and not ai_summary.startswith("AI analysis"):
        report_path = temp_dir / f"temp_ai_report_{run_timestamp}.md"
        with open(report_path, 'w', encoding='utf-8') as f: f.write(ai_summary)
        files_to_zip[str(report_path)] = "AI_Analysis_Report.md"

    zip_path = temp_dir / f"prediction_mutation_results_{run_timestamp}.zip"
    zip_path_str = create_zip_archive(files_to_zip, str(zip_path))

    final_status = status if not enable_ai else "✅ Prediction and AI analysis complete!"
    yield final_status, summary_fig, df, gr.update(visible=True, value=zip_path_str), zip_path_str, gr.update(visible=total_residues > 40), df, ai_summary

def expand_heatmap(full_df):
    if full_df is None or full_df.empty: return go.Figure(), gr.update(visible=True), gr.update(visible=False)
    data_tuple = prepare_plotly_heatmap_data(full_df)
    plot_data = data_tuple[:5]
    fig = generate_plotly_heatmap(*plot_data, is_partial=False, total_residues=get_total_residues_count(full_df))
    return fig, gr.update(visible=False), gr.update(visible=True)

def collapse_heatmap(full_df):
    if full_df is None or full_df.empty: return go.Figure(), gr.update(visible=True), gr.update(visible=False)
    data_tuple = prepare_plotly_heatmap_data(full_df, max_residues=40)
    plot_data = data_tuple[:5]
    fig = generate_plotly_heatmap(*plot_data, is_partial=True, total_residues=get_total_residues_count(full_df))
    return fig, gr.update(visible=True), gr.update(visible=False)

# --- Protein Function Prediction Logic ---

def generate_plots_for_all_results(results_df: pd.DataFrame) -> go.Figure:
    plot_df = results_df[
        (results_df['header'] != "ERROR") &
        (results_df['Dataset'].apply(lambda d: DATASET_TO_TASK_MAP.get(d) not in REGRESSION_TASKS_FUNCTION))
    ].copy()

    if plot_df.empty:
        fig = go.Figure()
        fig.update_layout(title_text="<b>No Visualization Available</b>", title_x=0.5, xaxis={"visible": False}, yaxis={"visible": False},
                          annotations=[{"text": "The current task does not support results analysis display.", "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5, "showarrow": False}])
        return fig

    sequences = plot_df['header'].unique()
    datasets = plot_df['Dataset'].unique()
    n_sequences, n_datasets = len(sequences), len(datasets)
    
    subplot_titles = [f"{seq[:15]}<br>{ds[:20]}" if n_sequences > 1 else f"{ds[:25]}" for seq in sequences for ds in datasets]
    
    fig = make_subplots(rows=n_sequences, cols=n_datasets, subplot_titles=subplot_titles, vertical_spacing=0.25)

    for r_idx, seq_header in enumerate(sequences, 1):
        for c_idx, dataset in enumerate(datasets, 1):
            row_data = plot_df[(plot_df['header'] == seq_header) & (plot_df['Dataset'] == dataset)]
            if row_data.empty: continue
            
            row = row_data.iloc[0]
            prob_col_name = next((col for col in row.index if 'probab' in col.lower()), None)
            if not prob_col_name or pd.isna(row[prob_col_name]): continue

            try:
                confidences_raw = row[prob_col_name]
                confidences = json.loads(confidences_raw) if isinstance(confidences_raw, str) else confidences_raw
                if not isinstance(confidences, list): continue
                
                task = DATASET_TO_TASK_MAP.get(dataset)
                labels_key = "DeepLocMulti" if dataset == "DeepLocMulti" else "DeepLocBinary" if dataset == "DeepLocBinary" else task
                labels = LABEL_MAPPING_FUNCTION.get(labels_key, [f"Class {k}" for k in range(len(confidences))])
                colors = [COLOR_MAP_FUNCTION.get(lbl, COLOR_MAP_FUNCTION["Default"]) for lbl in labels]
                
                plot_data = sorted(zip(labels, confidences, colors), key=lambda x: x[1], reverse=True)
                sorted_labels, sorted_confidences, sorted_colors = zip(*plot_data)
                
                fig.add_trace(go.Bar(x=sorted_labels, y=sorted_confidences, marker_color=sorted_colors), row=r_idx, col=c_idx)
                fig.update_yaxes(range=[0, 1], row=r_idx, col=c_idx)
            except Exception as e:
                print(f"Plotting error for {seq_header}/{dataset}: {e}")

    main_title = "<b>Prediction Confidence Scores</b>"
    if n_sequences == 1: main_title += f"<br><sub>Sequence: {sequences[0][:80]}</sub>"
    fig.update_layout(title=dict(text=main_title, x=0.5), showlegend=False, height=max(400, 350 * n_sequences + 100))
    for r in range(1, n_sequences + 1): fig.update_yaxes(title_text="Confidence", row=r, col=1)
    return fig

def handle_protein_function_prediction(task: str, fasta_file: Any, enable_ai: bool, ai_model: str, user_api_key: str, model_override: Optional[str] = None, datasets_override: Optional[List[str]] = None) -> Generator:
    model = model_override if model_override else "ESM2-650M"
    datasets = datasets_override if datasets_override is not None else DATASET_MAPPING_FUNCTION.get(task, [])

    if not all([task, datasets, fasta_file]):
        yield "❌ Error: Task, Datasets, and FASTA file are required.", pd.DataFrame(), None, gr.update(visible=False), ""
        return

    yield f"🚀 Starting predictions with {model}...", pd.DataFrame(), None, gr.update(visible=False), "AI analysis will appear here..."
    
    all_results_list, temp_dir = [], Path("temp_outputs")
    temp_dir.mkdir(exist_ok=True)

    for i, dataset in enumerate(datasets):
        yield f"⏳ Running prediction for '{dataset}' ({i+1}/{len(datasets)})...", None, None, gr.update(visible=False), "AI analysis will appear here..."
        try:
            model_key = MODEL_MAPPING_FUNCTION.get(model)
            if not model_key: raise ValueError(f"Model key not found for {model}")
            
            adapter_key = MODEL_ADAPTER_MAPPING_FUNCTION[model_key]
            script_path = Path("src") / "property" / f"{model_key}.py"
            adapter_path = Path("ckpt") / dataset / adapter_key
            output_file = temp_dir / f"temp_{dataset}_{model}.csv"
            
            if not script_path.exists() or not adapter_path.exists():
                raise FileNotFoundError(f"Required files not found for {dataset} (Script: {script_path}, Adapter: {adapter_path})")
            
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
    
    raw_final_df = pd.concat(all_results_list, ignore_index=True).fillna('N/A')
    plot_fig = generate_plots_for_all_results(raw_final_df)
    
    display_df = raw_final_df.copy()
    rename_map = {'header': "Header", 'sequence': "Sequence", 'prediction': "Predicted Value / Class", 'predicted_class': "Predicted Value / Class", 'probabilities': "Confidence Scores"}
    display_df.rename(columns=rename_map, inplace=True)

    if "Predicted Value / Class" in display_df.columns:
        def map_labels(row):
            current_task = DATASET_TO_TASK_MAP.get(row['Dataset'], task)
            if current_task in REGRESSION_TASKS_FUNCTION: return row["Predicted Value / Class"]
            
            labels_key = "DeepLocMulti" if row['Dataset'] == "DeepLocMulti" else "DeepLocBinary" if row['Dataset'] == "DeepLocBinary" else current_task
            labels = LABEL_MAPPING_FUNCTION.get(labels_key)
            pred_val = pd.to_numeric(row["Predicted Value / Class"], errors='coerce')
            
            if labels and pd.notna(pred_val) and 0 <= int(pred_val) < len(labels):
                return labels[int(pred_val)]
            return row["Predicted Value / Class"]
        display_df["Predicted Value / Class"] = display_df.apply(map_labels, axis=1)

    ai_summary = "For a complete analysis of the prediction results, activate the AI Analysis feature. Simply go to AI Settings, select Enable AI Summary, and enter a valid API Key."
    if enable_ai:
        yield "🤖 Generating AI summary...", display_df, plot_fig, gr.update(visible=False), "🤖 AI is analyzing the results..."
        api_key = get_api_key(ai_model, user_api_key)
        if not api_key:
            ai_summary = f"❌ No API key found for {ai_model}."
        else:
            ai_config = AIConfig(api_key, ai_model, AI_MODELS[ai_model]["api_base"], AI_MODELS[ai_model]["model"])
            prompt = generate_ai_summary_prompt(display_df, task, model)
            ai_summary = call_ai_api(ai_config, prompt)
    
    zip_path_str = ""
    try:
        zip_temp_dir = temp_dir / f"download_{int(time.time())}"; zip_temp_dir.mkdir()
        raw_final_df.to_csv(zip_temp_dir / "full_results.csv", index=False)
        if plot_fig.data: plot_fig.write_html(str(zip_temp_dir / "results_plot.html"))
        if not ai_summary.startswith("❌") and not ai_summary.startswith("AI analysis"):
            with open(zip_temp_dir / "AI_Analysis_Report.md", 'w', encoding='utf-8') as f: f.write(f"# AI Analysis Report\n\n{ai_summary}")
        
        zip_path = temp_dir / "function_prediction_results.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for file in zip_temp_dir.glob("*"): zf.write(file, file.name)
        zip_path_str = str(zip_path)
    except Exception as e:
        print(f"Error creating zip file: {e}")

    final_status = "✅ All predictions completed!"
    if enable_ai and not ai_summary.startswith("❌"): final_status += " AI analysis included."
    
    yield final_status, display_df, plot_fig, gr.update(visible=True, value=zip_path_str), ai_summary

def create_easy_use_tab(constant: Dict[str, Any]) -> Dict[str, Any]:
    # Note: The 'constant' argument is not used in the original code, but kept for signature consistency.
    # Model choices for advanced dropdowns are derived from the main constant dictionaries.
    sequence_models = list(MODEL_MAPPING_ZERO_SHOT.keys())
    structure_models = [k for k, v in MODEL_MAPPING_ZERO_SHOT.items() if v == 'esmif1'] # Example filter for structure models
    function_models = list(MODEL_MAPPING_FUNCTION.keys())

    with gr.Blocks() as demo:
        with gr.Tabs():
            with gr.TabItem("Directed Evolution: AI-Powered Mutation Prediction"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2):
                        # --- Accordion for Easy vs. Advanced Use ---
                        with gr.Accordion("Quick Analysis: Obtain protein mutation prediction with default settings", open=True):
                            gr.Markdown("### Model Configuration")
                            zero_shot_function_dd = gr.Dropdown(choices=DATASET_MAPPING_ZERO_SHOT, label="Select Protein Function", value=DATASET_MAPPING_ZERO_SHOT[0])
                            
                            gr.Markdown("### Configure AI Analysis (Optional)")
                            enable_ai_zshot = gr.Checkbox(label="Enable AI Summary", value=False)
                            with gr.Group(visible=False) as ai_box_zshot:
                                ai_model_dd_zshot = gr.Dropdown(choices=list(AI_MODELS.keys()), value="DeepSeek", label="Select AI Model")
                                api_key_in_zshot = gr.Textbox(label="API Key (Optional)", type="password", placeholder="Leave blank for env var")

                            easy_zshot_file_upload = gr.File(label="Upload Protein File (.fasta, .fa, .pdb)", file_types=[".fasta", ".fa", ".pdb"])
                            easy_zshot_protein_display = gr.Textbox(label="Uploaded Protein Sequence", interactive=False, lines=5, max_lines=7)
                            easy_zshot_predict_btn = gr.Button("🚀 Start Prediction (Easy)", variant="primary")

                        with gr.Accordion("Advanced Settings: Choose the most suitable protein analysis approach for your needs", open=False):
                            with gr.Tabs():
                                with gr.TabItem("🧬 Sequence-based Model"):
                                    gr.Markdown("### Model Configuration")
                                    seq_function_dd = gr.Dropdown(choices=DATASET_MAPPING_ZERO_SHOT, label="Select Protein Function", value=DATASET_MAPPING_ZERO_SHOT[0])
                                    seq_model_dd = gr.Dropdown(choices=sequence_models, label="Select Sequence-based Model", value=sequence_models[0] if sequence_models else None)
                                    seq_file_upload = gr.File(label="Upload FASTA file", file_types=[".fasta", ".fa"])
                                    seq_protein_display = gr.Textbox(label="Uploaded Protein Sequence", interactive=False, lines=5, max_lines=7)
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
                                    struct_protein_display = gr.Textbox(label="Uploaded Protein Sequence", interactive=False, lines=5, max_lines=7)
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
                        with gr.Accordion("Quick Analysis: Obtain protein mutation prediction with default settings", open=True):
                            gr.Markdown("### Model Configuration")
                            easy_func_task_dd = gr.Dropdown(choices=list(DATASET_MAPPING_FUNCTION.keys()), label="Select Task", value="Solubility")
                            base_function_fasta_upload = gr.File(label="Upload FASTA file", file_types=[".fasta", ".fa"])
                            base_function_protein_display = gr.Textbox(label="Uploaded Protein Sequence", interactive=False, lines=5, max_lines=7)

                            gr.Markdown("### Configure AI Analysis (Optional)")
                            with gr.Accordion("AI Settings", open=True):
                                enable_ai_func = gr.Checkbox(label="Enable AI Summary", value=False)
                                with gr.Group(visible=False) as ai_box_func:
                                    ai_model_dd_func = gr.Dropdown(choices=list(AI_MODELS.keys()), label="Select AI Model", value="DeepSeek")
                                    api_key_in_func = gr.Textbox(label="API Key (Optional)", type="password", placeholder="Leave blank to use environment variable")
                                easy_func_predict_btn = gr.Button("🚀 Start Prediction (Easy)", variant="primary")
                            
                        with gr.Accordion("Advanced Settings: Choose the most suitable protein analysis approach for your needs", open=False):
                            adv_func_model_dd = gr.Dropdown(choices=function_models, label="Select Model", value="ESM2-650M")
                            adv_func_task_dd = gr.Dropdown(choices=list(DATASET_MAPPING_FUNCTION.keys()), label="Select Task", value="Solubility")
                            default_datasets = DATASET_MAPPING_FUNCTION.get("Solubility", [])
                            adv_func_dataset_cbg = gr.CheckboxGroup(label="Select Datasets", 
                                                                    choices=default_datasets, 
                                                                    value=default_datasets)
                            function_fasta_upload = gr.File(label="Upload FASTA file", file_types=[".fasta", ".fa"])
                            function_protein_display = gr.Textbox(label="Uploaded Protein", interactive=False, lines=7, max_lines=10)

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
                                function_results_df = gr.DataFrame(label="Prediction Data")
                            with gr.TabItem("📈 Prediction Plots"):
                                function_results_plot = gr.Plot(label="Confidence Scores")
                            with gr.TabItem("🤖 AI Analysis"):
                                function_ai_summary_output = gr.Textbox(label="AI Analysis Report", value="AI analysis will appear here...", lines=20, interactive=False, show_copy_button=True)
                        
                        function_download_btn = gr.DownloadButton("💾 Download Results", visible=False)

        # --- Event Handlers ---

        # --- Directed Evolution Tab Event Handlers ---
        enable_ai_zshot.change(fn=toggle_ai_section, inputs=enable_ai_zshot, outputs=ai_box_zshot)
        
        easy_zshot_file_upload.upload(fn=handle_file_upload, inputs=easy_zshot_file_upload, outputs=easy_zshot_protein_display)
        easy_zshot_predict_btn.click(
            fn=handle_mutation_prediction,
            inputs=[zero_shot_function_dd, easy_zshot_file_upload, enable_ai_zshot, ai_model_dd_zshot, api_key_in_zshot],
            outputs=[zero_shot_status_box, zero_shot_plot_out, zero_shot_df_out, zero_shot_download_btn, zero_shot_download_path_state, zero_shot_view_controls, zero_shot_full_data_state, zero_shot_ai_out]
        )

        seq_file_upload.upload(fn=parse_fasta_file, inputs=seq_file_upload, outputs=seq_protein_display)
        seq_predict_btn.click(
            fn=handle_mutation_prediction, 
            inputs=[seq_function_dd, seq_file_upload, enable_ai_zshot, ai_model_dd_zshot, api_key_in_zshot, seq_model_dd],
            outputs=[zero_shot_status_box, zero_shot_plot_out, zero_shot_df_out, zero_shot_download_btn, zero_shot_download_path_state, zero_shot_view_controls, zero_shot_full_data_state, zero_shot_ai_out]
        )

        struct_file_upload.upload(fn=parse_pdb_for_sequence, inputs=struct_file_upload, outputs=struct_protein_display)
        struct_predict_btn.click(
            fn=handle_mutation_prediction, 
            inputs=[struct_function_dd, struct_file_upload, enable_ai_zshot, ai_model_dd_zshot, api_key_in_zshot, struct_model_dd], 
            outputs=[zero_shot_status_box, zero_shot_plot_out, zero_shot_df_out, zero_shot_download_btn, zero_shot_download_path_state, zero_shot_view_controls, zero_shot_full_data_state, zero_shot_ai_out]
        )

        expand_btn.click(fn=expand_heatmap, inputs=[zero_shot_full_data_state], outputs=[zero_shot_plot_out, expand_btn, collapse_btn])
        collapse_btn.click(fn=collapse_heatmap, inputs=[zero_shot_full_data_state], outputs=[zero_shot_plot_out, expand_btn, collapse_btn])

        # --- Protein Function Prediction Tab Event Handlers ---
        enable_ai_func.change(fn=toggle_ai_section, inputs=enable_ai_func, outputs=ai_box_func)
        
        base_function_fasta_upload.upload(fn=handle_file_upload, inputs=base_function_fasta_upload, outputs=base_function_protein_display)
        function_fasta_upload.upload(fn=handle_file_upload, inputs=function_fasta_upload, outputs=function_protein_display)
        
        # Easy Use Button
        easy_func_predict_btn.click(
        fn=lambda task, file, ai, model, key: (yield from handle_protein_function_prediction(task, file, ai, model, key, model_override="ESM2-650M", datasets_override=DATASET_MAPPING_FUNCTION.get(task, []))),
        inputs=[easy_func_task_dd, base_function_fasta_upload, enable_ai_func, ai_model_dd_func, api_key_in_func],
        outputs=[function_status_textbox, function_results_df, function_results_plot, function_download_btn, function_ai_summary_output]
    )
        # Advanced Use Button and Controls
        adv_func_task_dd.change(fn=update_dataset_choices, inputs=adv_func_task_dd, outputs=adv_func_dataset_cbg)

        adv_func_predict_btn.click(
            fn=handle_protein_function_prediction,
            inputs=[adv_func_task_dd, function_fasta_upload, enable_ai_func, ai_model_dd_func, api_key_in_func, adv_func_model_dd, adv_func_dataset_cbg],
            outputs=[function_status_textbox, function_results_df, function_results_plot, function_download_btn, function_ai_summary_output]
        )

    return {}