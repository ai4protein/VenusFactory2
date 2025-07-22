import gradio as gr
import os
import sys
import pandas as pd
import subprocess
import json
import time
import zipfile
from typing import Dict, Any, Tuple, List
import plotly.graph_objects as go
import numpy as np


def run_zero_shot_prediction(model_type: str, model_name: str, file_path: str) -> Tuple[str, pd.DataFrame]:
    """
    Executes a prediction script as a subprocess.
    """
    try:
        output_csv = f"temp_{model_type}_{int(time.time())}.csv"
        model_name_map = {
            "ESM-1v": "esm1v", "ESM2-650M": "esm2", "SaProt": "saprot",
            "ESM-IF1": "esmif1", "MIF-ST": "mifst", "ProSST-2048": "prosst",
            "ProSSN": "protssn"
        }
        script_path = f"src/mutation/models/{model_name_map[model_name]}.py"

        if model_type == "structure":
            file_argument = "--pdb_file"
        else:
            file_argument = "--fasta_file"

        cmd = [sys.executable, script_path, file_argument, file_path, "--output_csv", output_csv]
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if process.returncode == 0:
            if os.path.exists(output_csv):
                return "Prediction completed successfully!", pd.read_csv(output_csv)
            else:
                return "Prediction completed but no output file was generated.", pd.DataFrame()
        else:
            return f"Prediction failed: {process.stderr or 'Unknown error'}", pd.DataFrame()

    except Exception as e:
        return f"Error running prediction: {e}", pd.DataFrame()


def prepare_plotly_heatmap_data(df: pd.DataFrame, max_residues: int = None) -> Tuple:
    """
    Prepares data for Plotly heatmap. Y-axis is sorted by residue's natural sequence order.
    If max_residues is specified, only shows the first max_residues positions.
    """
    score_col = next((col for col in df.columns if 'score' in col.lower()), None)
    if score_col is None:
        return None, None, None, None, None, None

    valid_mutations_df = df[df['mutant'].apply(lambda m: len(m) > 2 and m[0] != m[-1] and m[1:-1].isdigit())].copy()
    if valid_mutations_df.empty:
        return [], [], np.array([[]]), np.array([[]]), np.array([[]]), score_col

    valid_mutations_df['rank'] = valid_mutations_df[score_col].rank(method='min', ascending=False).astype(int)
    total_mutations = len(valid_mutations_df)
    valid_mutations_df['rank_bin'] = np.ceil(valid_mutations_df['rank'] / (total_mutations / 10)).clip(upper=10)
    valid_mutations_df['inverted_rank_bin'] = 11 - valid_mutations_df['rank_bin']
    valid_mutations_df['position'] = valid_mutations_df['mutant'].str[1:-1].astype(int)

    # Sort positions in natural order
    sorted_positions = sorted(valid_mutations_df['position'].unique())
    
    # Apply max_residues limit if specified
    if max_residues is not None:
        sorted_positions = sorted_positions[:max_residues]
        # Filter the dataframe to only include these positions
        valid_mutations_df = valid_mutations_df[valid_mutations_df['position'].isin(sorted_positions)]

    x_labels = list("ACDEFGHIKLMNPQRSTVWY")
    x_map = {label: i for i, label in enumerate(x_labels)}
    
    wt_map = {int(mut[1:-1]): mut[0] for mut in valid_mutations_df['mutant']}
    y_labels = [f"{wt_map.get(pos, '?')}{pos}" for pos in sorted_positions]
    y_map = {pos: i for i, pos in enumerate(sorted_positions)}

    num_y = len(y_labels)
    num_x = len(x_labels)
    z_data = np.full((num_y, num_x), np.nan)
    rank_matrix = np.full((num_y, num_x), np.nan)
    score_matrix = np.full((num_y, num_x), np.nan)

    for _, row in valid_mutations_df.iterrows():
        pos = row['position']
        mut_aa = row['mutant'][-1]
        if pos in y_map and mut_aa in x_map:
            y_idx = y_map[pos]
            x_idx = x_map[mut_aa]
            z_data[y_idx, x_idx] = row['inverted_rank_bin']
            rank_matrix[y_idx, x_idx] = row['rank']
            score_matrix[y_idx, x_idx] = round(row[score_col], 3)
            
    return x_labels, y_labels, z_data, rank_matrix, score_matrix, score_col


def generate_plotly_heatmap(x_labels: List, y_labels: List, z_data: np.ndarray, 
                          rank_data: np.ndarray, score_data: np.ndarray, 
                          is_partial: bool = False, total_residues: int = None) -> go.Figure:
    """
    Generates a Plotly heatmap with adaptive height and natural y-axis order.
    """
    if z_data is None or z_data.size == 0:
        return go.Figure().update_layout(title="No data available to display heatmap")

    # Calculate plot height - smaller for partial view
    num_residues = len(y_labels)
    if is_partial:
        dynamic_height = max(400, min(1200, 25 * num_residues + 150))
    else:
        dynamic_height = max(500, min(4000, 25 * num_residues + 200))

    custom_data = np.stack((rank_data, score_data), axis=-1)

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=x_labels,
        y=y_labels,
        customdata=custom_data,
        hovertemplate=(
            "<b>Position</b>: %{y}<br>"
            "<b>Mutation to</b>: %{x}<br>"
            "<b>Rank</b>: %{customdata[0]}<br>"
            "<b>Score</b>: %{customdata[1]}"
            "<extra></extra>"
        ),
        colorscale='RdYlGn_r',
        zmin=1,
        zmax=10,
        showscale=True,  # Always show colorbar
        colorbar={
            'title': 'Rank Percentile',
            'tickvals': [10, 6, 1],
            'ticktext': ['Top 10%', 'Top 50%', 'Lowest 10%']
        }
    ))

    # Add title indicating partial or full view
    title_text = "Prediction Heatmap"
    if is_partial and total_residues:
        title_text += f" (Showing first {num_residues} of {total_residues} residues)"
    elif not is_partial and total_residues:
        title_text += f" (Complete view - {total_residues} residues)"

    fig.update_layout(
        title=title_text,
        xaxis_title='Mutant Amino Acid',
        yaxis_title='Residue Position and Wild Type',
        height=dynamic_height,
        yaxis_autorange='reversed'
    )
    return fig


def get_total_residues_count(df: pd.DataFrame) -> int:
    """
    Get total number of unique residue positions from the dataframe
    """
    if df.empty:
        return 0
    
    valid_mutations_df = df[df['mutant'].apply(lambda m: len(m) > 2 and m[0] != m[-1] and m[1:-1].isdigit())].copy()
    if valid_mutations_df.empty:
        return 0
    
    valid_mutations_df['position'] = valid_mutations_df['mutant'].str[1:-1].astype(int)
    return len(valid_mutations_df['position'].unique())


def create_zip_archive(files_to_zip: Dict[str, str], zip_filename: str) -> str:
    """
    Creates a zip archive from a dictionary of source paths and desired archive names.
    """
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for src_path, arc_name in files_to_zip.items():
            if os.path.exists(src_path):
                zipf.write(src_path, arcname=arc_name)
    return zip_filename


def create_zero_shot_tab(constant: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sets up the Gradio UI.
    """
    sequence_models = list(constant.get("zero_shot_sequence_model", {}).keys())
    structure_models = list(constant.get("zero_shot_structure_model", {}).keys())

    def parse_fasta_file(file_path: str) -> str:
        if not file_path: return ""
        try:
            with open(file_path, 'r') as f:
                sequences = [line.strip() for line in f if not line.startswith('>')]
            return "".join(sequences)
        except Exception as e:
            return f"Error parsing FASTA: {e}"

    def parse_pdb_for_sequence(file_path: str) -> str:
        aa_code_map = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 
            'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 
            'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }
        sequence, seen_residues, current_chain = [], set(), None
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if line.startswith("ATOM"):
                        chain_id = line[21]
                        if current_chain is None: current_chain = chain_id
                        if chain_id != current_chain: break
                        res_name, res_seq_num = line[17:20].strip(), int(line[22:26])
                        residue_id = (chain_id, res_seq_num)
                        if residue_id not in seen_residues:
                            if res_name in aa_code_map:
                                sequence.append(aa_code_map[res_name])
                                seen_residues.add(residue_id)
            return "".join(sequence)
        except Exception as e:
            return "Error: Could not read sequence from PDB file."

    def display_protein_sequence_from_fasta(file_obj: Any) -> str:
        return parse_fasta_file(file_obj.name) if file_obj else ""

    def display_protein_sequence_from_pdb(file_obj: Any) -> str:
        return parse_pdb_for_sequence(file_obj.name) if file_obj else ""

    gr.Markdown("## ⚡️ Protein Mutation Prediction")

    with gr.Tabs():
        with gr.TabItem("🧬 Sequence-based Model"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=2):
                    gr.Markdown("### Model Configuration")
                    seq_model_dropdown = gr.Dropdown(choices=sequence_models, label="Select Sequence-based Model", value=sequence_models[0] if sequence_models else None)
                    seq_file = gr.File(label="Upload FASTA sequence file (.fasta, .fa)", file_types=[".fasta", ".fa"], type="filepath")
                    seq_protein_display = gr.Textbox(label="Uploaded Protein", interactive=False, lines=7, max_lines=10, placeholder="Protein sequence will appear here after upload...")
                    seq_predict_btn = gr.Button("🚀 Start Prediction", variant="primary", interactive=False)
                with gr.Column(scale=3):
                    gr.Markdown("### Prediction Results")
                    seq_status = gr.Textbox(label="Status", interactive=False, value="Please select a model and upload a file.")
                    
                    with gr.Tabs():
                        with gr.TabItem("Prediction Heatmap"):
                            # View control buttons now inside the heatmap tab
                            seq_view_controls = gr.Row(visible=False)
                            with seq_view_controls:
                                seq_expand_btn = gr.Button("📈 Show Complete Heatmap", variant="secondary", size="sm")
                                seq_collapse_btn = gr.Button("📉 Show Summary View (First 40 residues)", variant="secondary", size="sm", visible=False)
                            
                            seq_plot = gr.Plot(label="Prediction Heatmap")
                        with gr.TabItem("Raw Results"):
                            seq_dataframe = gr.DataFrame(label="Raw Prediction Data")
                    seq_download_btn = gr.DownloadButton("💾 Download Results", visible=False)

        with gr.TabItem("🏗️ Structure-based Model"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=2):
                    gr.Markdown("### Model Configuration")
                    struct_model_dropdown = gr.Dropdown(choices=structure_models, label="Select Structure-based Model", value=structure_models[0] if structure_models else None)
                    struct_file = gr.File(label="Upload PDB structure file (.pdb)", file_types=[".pdb"], type="filepath")
                    struct_protein_display = gr.Textbox(label="Uploaded Protein Sequence (from PDB)", interactive=False, lines=7, max_lines=10, placeholder="Protein sequence from PDB will appear here...")
                    struct_predict_btn = gr.Button("🚀 Start Prediction", variant="primary", interactive=False)
                with gr.Column(scale=3):
                    gr.Markdown("### Prediction Results")
                    struct_status = gr.Textbox(label="Status", interactive=False, value="Please select a model and upload a file.")
                    
                    with gr.Tabs():
                        with gr.TabItem("Prediction Heatmap"):
                            # View control buttons now inside the heatmap tab
                            struct_view_controls = gr.Row(visible=False)
                            with struct_view_controls:
                                struct_expand_btn = gr.Button("📈 Show Complete Heatmap", variant="secondary", size="sm")
                                struct_collapse_btn = gr.Button("📉 Show Summary View (First 40 residues)", variant="secondary", size="sm", visible=False)
                            
                            struct_plot = gr.Plot(label="Prediction Heatmap")
                        with gr.TabItem("Raw Results"):
                            struct_dataframe = gr.DataFrame(label="Raw Prediction Data")
                    struct_download_btn = gr.DownloadButton("💾 Download Results", visible=False)

    # State variables to store full data
    temp_seq_zip_path = gr.State(None)
    temp_struct_zip_path = gr.State(None)
    seq_full_data = gr.State(None)  # Store full prediction data
    struct_full_data = gr.State(None)  # Store full prediction data

    def update_predict_button_status(model, file_obj):
        return gr.update(interactive=bool(model and file_obj and hasattr(file_obj, 'name')))

    def update_status_predicting():
        """Update status to show prediction is running"""
        return "🔄 Predicting... Please wait."
    
    def handle_prediction(model_type: str, model_name: str, file_obj: Any):
        """Handle prediction and return initial summary view"""
        if not model_name or not file_obj:
            return ("Error: Check inputs", go.Figure(), pd.DataFrame(), 
                   gr.update(visible=False), None, gr.update(visible=False), None)
    
        status, df = run_zero_shot_prediction(model_type, model_name, file_obj.name)
        
        if df.empty:
            return (status, go.Figure().update_layout(title="No results generated"), pd.DataFrame(), 
                   gr.update(visible=False), None, gr.update(visible=False), None)
        
        # Get total residue count
        total_residues = get_total_residues_count(df)
        
        # Generate summary view (first 40 residues)
        x_labels, y_labels, z_data, rank_data, score_data, score_col = prepare_plotly_heatmap_data(df, max_residues=40)
        
        if score_col is None:
            csv_path = f"temp_{model_type}_result_{int(time.time())}.csv"
            df.to_csv(csv_path, index=False)
            return (status, go.Figure().update_layout(title="Score column not found"), df, 
                   gr.update(visible=True, value=csv_path), csv_path, 
                   gr.update(visible=False), df)

        # Generate summary heatmap
        fig = generate_plotly_heatmap(x_labels, y_labels, z_data, rank_data, score_data, 
                                    is_partial=True, total_residues=total_residues)
        
        # Create download files
        run_timestamp = int(time.time())
        csv_path = f"temp_{model_type}_results_{run_timestamp}.csv"
        df.to_csv(csv_path, index=False)
        heatmap_path = f"temp_{model_type}_heatmap_{run_timestamp}.html"
        fig.write_html(heatmap_path)
        zip_path = f"prediction_{model_type}_results_{run_timestamp}.zip"
        create_zip_archive({csv_path: "prediction_results.csv", heatmap_path: "prediction_heatmap.html"}, zip_path)

        # Show expand button if there are more than 40 residues
        show_controls = total_residues > 40

        return (status, fig, df, gr.update(visible=True, value=zip_path), zip_path, 
               gr.update(visible=show_controls), df)

    def expand_heatmap(full_df, model_type):
        """Show complete heatmap"""
        if full_df is None or full_df.empty:
            return go.Figure().update_layout(title="No data available"), gr.update(visible=True), gr.update(visible=False)
        
        total_residues = get_total_residues_count(full_df)
        x_labels, y_labels, z_data, rank_data, score_data, score_col = prepare_plotly_heatmap_data(full_df)
        
        if score_col is None:
            return go.Figure().update_layout(title="Score column not found"), gr.update(visible=True), gr.update(visible=False)
        
        fig = generate_plotly_heatmap(x_labels, y_labels, z_data, rank_data, score_data, 
                                    is_partial=False, total_residues=total_residues)
        
        return fig, gr.update(visible=False), gr.update(visible=True)

    def collapse_heatmap(full_df, model_type):
        """Show summary heatmap (first 40 residues)"""
        if full_df is None or full_df.empty:
            return go.Figure().update_layout(title="No data available"), gr.update(visible=True), gr.update(visible=False)
        
        total_residues = get_total_residues_count(full_df)
        x_labels, y_labels, z_data, rank_data, score_data, score_col = prepare_plotly_heatmap_data(full_df, max_residues=40)
        
        if score_col is None:
            return go.Figure().update_layout(title="Score column not found"), gr.update(visible=True), gr.update(visible=False)
        
        fig = generate_plotly_heatmap(x_labels, y_labels, z_data, rank_data, score_data, 
                                    is_partial=True, total_residues=total_residues)
        
        return fig, gr.update(visible=True), gr.update(visible=False)

    # Event handlers
    seq_file.upload(fn=display_protein_sequence_from_fasta, inputs=[seq_file], outputs=[seq_protein_display])
    seq_file.upload(fn=update_predict_button_status, inputs=[seq_model_dropdown, seq_file], outputs=[seq_predict_btn])
    seq_model_dropdown.change(fn=update_predict_button_status, inputs=[seq_model_dropdown, seq_file], outputs=[seq_predict_btn])
    
    # First update status to show prediction is starting
    seq_predict_btn.click(
        fn=update_status_predicting,
        inputs=[],
        outputs=[seq_status]
    ).then(
        fn=lambda model, file: handle_prediction("sequence", model, file), 
        inputs=[seq_model_dropdown, seq_file], 
        outputs=[seq_status, seq_plot, seq_dataframe, seq_download_btn, temp_seq_zip_path, seq_view_controls, seq_full_data]
    )
    
    seq_expand_btn.click(
        fn=lambda df: expand_heatmap(df, "sequence"),
        inputs=[seq_full_data],
        outputs=[seq_plot, seq_expand_btn, seq_collapse_btn]
    )
    
    seq_collapse_btn.click(
        fn=lambda df: collapse_heatmap(df, "sequence"),
        inputs=[seq_full_data],
        outputs=[seq_plot, seq_expand_btn, seq_collapse_btn]
    )

    struct_file.upload(fn=display_protein_sequence_from_pdb, inputs=[struct_file], outputs=[struct_protein_display])
    struct_file.upload(fn=update_predict_button_status, inputs=[struct_model_dropdown, struct_file], outputs=[struct_predict_btn])
    struct_model_dropdown.change(fn=update_predict_button_status, inputs=[struct_model_dropdown, struct_file], outputs=[struct_predict_btn])
    
    # First update status to show prediction is starting
    struct_predict_btn.click(
        fn=update_status_predicting,
        inputs=[],
        outputs=[struct_status]
    ).then(
        fn=lambda model, file: handle_prediction("structure", model, file), 
        inputs=[struct_model_dropdown, struct_file], 
        outputs=[struct_status, struct_plot, struct_dataframe, struct_download_btn, temp_struct_zip_path, struct_view_controls, struct_full_data]
    )
    
    struct_expand_btn.click(
        fn=lambda df: expand_heatmap(df, "structure"),
        inputs=[struct_full_data],
        outputs=[struct_plot, struct_expand_btn, struct_collapse_btn]
    )
    
    struct_collapse_btn.click(
        fn=lambda df: collapse_heatmap(df, "structure"),
        inputs=[struct_full_data],
        outputs=[struct_plot, struct_expand_btn, struct_collapse_btn]
    )

    return {}