
import gradio as gr
import os
import sys
import pandas as pd
import subprocess
import json
import re
import time
import threading
from queue import Queue
from typing import Dict, Any, Tuple

def run_zero_shot_prediction(model_type: str, model_name: str, file_path: str) -> Tuple[str, pd.DataFrame]:
    """
    Run zero-shot prediction using the specified model and file.
    """
    try:
        output_csv = f"temp_{model_type}_{int(time.time())}.csv"
        model_name_map = {
            "ESM-1v": "esm1v", "ESM2-650M": "esm2", "SaProt": "saprot",
            "ESM-IF1": "esmif1", "MIF-ST": "mifst", "ProSST-2048": "prosst",
            "ProSSN": "protsst"
        }
        script_path = f"src/mutation/models/{model_name_map[model_name]}.py"

        # --- CHANGED PART ---
        # Determine the correct file argument based on the model type
        if model_type == "structure":
            file_argument = "--pdb_file"
        else:  # Default to sequence
            file_argument = "--fasta_file"

        # Construct the command with the correct file argument
        cmd = [sys.executable, script_path, file_argument, file_path, "--output_csv", output_csv]
        # --- END OF CHANGE ---
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
# --- MODIFIED FUNCTION ---
def create_zero_shot_tab(constant: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates and returns the UI components for the Gradio Zero-Shot Prediction tab.
    """
    sequence_models = list(constant.get("zero_shot_sequence_model", {}).keys())
    structure_models = list(constant.get("zero_shot_structure_model", {}).keys())
    
    # --- Helper function to parse FASTA file ---
    def parse_fasta_file(file_path: str) -> str:
        # ... (This function is from the previous step and remains unchanged)
        if not file_path: return ""
        try:
            with open(file_path, 'r') as f:
                sequences = [line.strip() for line in f if not line.startswith('>')]
            return "".join(sequences)
        except Exception as e:
            return f"Error parsing FASTA: {e}"

    # --- NEW HELPER FUNCTION to parse PDB file ---
    def parse_pdb_for_sequence(file_path: str) -> str:
        """Reads a PDB file and returns the amino acid sequence of the first chain."""
        if not file_path:
            return ""
        
        # 3-letter to 1-letter amino acid code mapping
        aa_code_map = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }
        
        sequence = []
        seen_residues = set()
        current_chain = None

        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if line.startswith("ATOM"):
                        chain_id = line[21]
                        
                        # Start processing the first chain found
                        if current_chain is None:
                            current_chain = chain_id
                        
                        # If we move to a new chain, stop.
                        if chain_id != current_chain:
                            break

                        res_name = line[17:20].strip()
                        res_seq_num = int(line[22:26])
                        
                        # Unique identifier for a residue is its number and chain
                        residue_id = (chain_id, res_seq_num)

                        if residue_id not in seen_residues:
                            if res_name in aa_code_map:
                                sequence.append(aa_code_map[res_name])
                                seen_residues.add(residue_id)
            return "".join(sequence)
        except Exception as e:
            print(f"Error parsing PDB file: {e}")
            return "Error: Could not read sequence from PDB file."

    # --- Event Handlers for UI ---
    def display_protein_sequence_from_fasta(file_obj: Any) -> str:
        return parse_fasta_file(file_obj.name) if file_obj else ""
        
    def display_protein_sequence_from_pdb(file_obj: Any) -> str:
        return parse_pdb_for_sequence(file_obj.name) if file_obj else ""

    # ... other inner functions are unchanged ...

    # with gr.Tab("Zero-Shot Prediction"):
    gr.Markdown("## ⚡️ Protein Mutation Prediction")

    with gr.Tabs():
        # --- Sequence-based Model Tab ---
        with gr.TabItem("🧬 Sequence-based Model"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=2):
                    # ... (UI components for sequence tab are unchanged)
                    gr.Markdown("### Model Configuration")
                    seq_model_dropdown = gr.Dropdown(choices=sequence_models, label="Select Sequence-based Model", value=sequence_models[0] if sequence_models else None)
                    seq_file = gr.File(label="Upload FASTA sequence file (.fasta, .fa)", file_types=[".fasta", ".fa"], type="filepath")
                    seq_protein_display = gr.Textbox(label="Uploaded Protein", interactive=False, lines=7, max_lines=10, placeholder="Protein sequence will appear here after upload...")
                    seq_predict_btn = gr.Button("🚀 Start Prediction", variant="primary", interactive=False)
                with gr.Column(scale=3):
                    # ... (Results part is unchanged)
                    gr.Markdown("### Prediction Results")
                    seq_status = gr.Textbox(label="Status", interactive=False, value="Please select a model and upload a file.")
                    seq_result = gr.DataFrame(label="Prediction Results", headers=["Sequence_ID", "Prediction_Score"], value=[["...", ""]])
                    seq_download_btn = gr.DownloadButton("💾 Download Results (CSV)", visible=False)

        # --- Structure-based Model Tab ---
        with gr.TabItem("🏗️ Structure-based Model"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=2):
                    gr.Markdown("### Model Configuration")
                    struct_model_dropdown = gr.Dropdown(choices=structure_models, label="Select Structure-based Model", value=structure_models[0] if structure_models else None)
                    struct_file = gr.File(label="Upload PDB structure file (.pdb)", file_types=[".pdb"], type="filepath")
                    
                    # CHANGED: Textbox now displays the protein sequence from PDB
                    struct_protein_display = gr.Textbox(
                        label="Uploaded Protein Sequence (from PDB)",
                        interactive=False,
                        lines=7,
                        max_lines=10,
                        placeholder="Protein sequence from PDB will appear here..."
                    )
                    struct_predict_btn = gr.Button("🚀 Start Prediction", variant="primary", interactive=False)

                with gr.Column(scale=3):
                    # ... (Results part is unchanged)
                    gr.Markdown("### Prediction Results")
                    struct_status = gr.Textbox(label="Status", interactive=False, value="Please select a model and upload a file.")
                    struct_result = gr.DataFrame(label="Prediction Results", headers=["Chain_ID", "Stability_Score"], value=[["...", ""]])
                    struct_download_btn = gr.DownloadButton("💾 Download Results (CSV)", visible=False)

    # --- Interactive Logic & Event Bindings ---
    temp_seq_csv_path = gr.State(None)
    temp_struct_csv_path = gr.State(None)

    def update_predict_button_status(model, file_obj):
        return gr.update(interactive=bool(model and file_obj and hasattr(file_obj, 'name')))
    
    # ... handle_predict functions are unchanged ...
    def handle_seq_predict(model_name, file_obj):
        # ...
        if not model_name or not file_obj:
            return "Error: Check inputs", gr.update(value=[["...", ""]]), gr.update(visible=False), None
        yield "Predicting...", gr.update(value=[["...", ""]]), gr.update(visible=False), None
        status, df = run_zero_shot_prediction("sequence", model_name, file_obj.name)
        if df.empty:
            yield status, gr.update(value=[["...", ""]]), gr.update(visible=False), None
        else:
            csv_path = f"temp_seq_result_{int(time.time())}.csv"
            df.to_csv(csv_path, index=False)
            yield status, gr.update(value=df), gr.update(visible=True, value=csv_path), csv_path
            
    def handle_struct_predict(model_name, file_obj):
        # ...
        if not model_name or not file_obj:
            return "Error: Check inputs", gr.update(value=[["...", ""]]), gr.update(visible=False), None
        yield "Predicting...", gr.update(value=[["...", ""]]), gr.update(visible=False), None
        status, df = run_zero_shot_prediction("structure", model_name, file_obj.name)
        if df.empty:
            yield status, gr.update(value=[["...", ""]]), gr.update(visible=False), None
        else:
            csv_path = f"temp_struct_result_{int(time.time())}.csv"
            df.to_csv(csv_path, index=False)
            yield status, gr.update(value=df), gr.update(visible=True, value=csv_path), csv_path

    # --- Event Listener Bindings ---
    # Sequence model listeners
    seq_file.upload(fn=display_protein_sequence_from_fasta, inputs=[seq_file], outputs=[seq_protein_display])
    seq_file.upload(fn=update_predict_button_status, inputs=[seq_model_dropdown, seq_file], outputs=[seq_predict_btn])
    seq_model_dropdown.change(fn=update_predict_button_status, inputs=[seq_model_dropdown, seq_file], outputs=[seq_predict_btn])
    seq_predict_btn.click(fn=handle_seq_predict, inputs=[seq_model_dropdown, seq_file], outputs=[seq_status, seq_result, seq_download_btn, temp_seq_csv_path])

    # Structure model listeners
    # CHANGED: The first listener now calls the new PDB parsing function
    struct_file.upload(fn=display_protein_sequence_from_pdb, inputs=[struct_file], outputs=[struct_protein_display])
    struct_file.upload(fn=update_predict_button_status, inputs=[struct_model_dropdown, struct_file], outputs=[struct_predict_btn])
    struct_model_dropdown.change(fn=update_predict_button_status, inputs=[struct_model_dropdown, struct_file], outputs=[struct_predict_btn])
    struct_predict_btn.click(fn=handle_struct_predict, inputs=[struct_model_dropdown, struct_file], outputs=[struct_status, struct_result, struct_download_btn, temp_struct_csv_path])

    return {}
