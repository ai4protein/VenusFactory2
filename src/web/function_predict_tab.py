import gradio as gr
import pandas as pd
import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List, Generator, Tuple, Union

# --- Constants and Mappings ---
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
    "solubility": ["DeepSol", "DeepSoluE", "ProtSolM"],
    "localization": ["DeepLocBinary", "DeepLocMulti"],
    "binding": ["MetalIonBinding"],
    "stability": ["Thermostability"],
}

LABEL_MAPPING = {
    "solubility": ["Insoluble", "Soluble"],
    "localization_binary": ["Membrane", "Soluble"],
    "localization_multi": [
        "Cytoplasm/Nucleus/", "Cytoplasm", "Extracellular", "Mitochondrion", 
        "Cell membrane", "Endoplasmic reticulum", "Plastid", "Golgi apparatus", 
        "Lysosome/Vacuole", "Peroxisome"
    ],
    "binding": ["Non-binding", "Binding"],
    "stability": ["Non-thermally stable", "Thermally stable"]
}


def create_protein_function_tab(constant: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Creates the Gradio UI tab for predicting protein functions using adapter-based models.
    """
    
    def parse_fasta_file(file_path: str) -> str:
        if not file_path: return ""
        try:
            with open(file_path, 'r') as f:
                return f.read()
        except Exception as e:
            return f"Error parsing FASTA: {e}"

    def display_protein_sequence_from_fasta(file_obj: Any) -> str:
        return parse_fasta_file(file_obj.name) if file_obj else ""

    def update_dataset_choices(task: str) -> gr.CheckboxGroup:
        choices = DATASET_MAPPING.get(task, [])
        return gr.update(choices=choices, value=[])

    def handle_prediction(model: str, task: str, datasets: List[str], fasta_file: Any) -> Generator[Tuple[str, Union[pd.DataFrame, None], gr.update], None, None]:
        if not all([model, task, datasets, fasta_file]):
            yield "Error: All fields are required.", pd.DataFrame(), gr.update(visible=False)
            return

        yield "Starting predictions...", pd.DataFrame(), gr.update(visible=False)

        all_results_list = []
        input_fasta_path = Path(fasta_file.name)
        temp_dir = Path("temp_outputs")
        temp_dir.mkdir(exist_ok=True)

        for i, dataset in enumerate(datasets):
            status_msg = f"Running prediction for '{dataset}' ({i+1}/{len(datasets)})..."
            yield status_msg, None, gr.update(visible=False)

            try:
                model_key = MODEL_MAPPING[model]
                adapter_key = MODEL_ADAPTER_MAPPING[model_key]
                
                script_path = Path("src") / "property" / f"{model_key}.py"
                adapter_path = Path("ckpt") / dataset / adapter_key
                output_file = temp_dir / f"temp_{dataset}_{model}.csv"

                if not script_path.exists() or not adapter_path.exists():
                    raise FileNotFoundError(f"Script or adapter not found for {dataset}")

                cmd = [
                    sys.executable, str(script_path.absolute()),
                    "--fasta_file", str(input_fasta_path.absolute()),
                    "--adapter_path", str(adapter_path.absolute()) + "\\",
                    "--output_csv", str(output_file.absolute()),
                ]
                
                process = subprocess.run(cmd, capture_output=True, text=True, check=True)

                if output_file.exists():
                    df = pd.read_csv(output_file)
                    
                    # --- MODIFIED: Robust renaming for the four-column format ---
                    rename_map = {}
                    if len(df.columns) > 1: rename_map[df.columns[1]] = "Sequence"
                    if len(df.columns) > 2: rename_map[df.columns[2]] = "Predicted Results"
                    if len(df.columns) > 3: rename_map[df.columns[3]] = "Confidence"
                    df.rename(columns=rename_map, inplace=True)

                    df["Dataset"] = dataset

                    if 'Predicted Results' in df.columns:
                        label_key = task
                        if task == "localization":
                            if dataset == "DeepLocBinary": label_key = "localization_binary"
                            elif dataset == "DeepLocMulti": label_key = "localization_multi"
                        
                        if label_key in LABEL_MAPPING:
                            labels = LABEL_MAPPING[label_key]
                            df['Predicted Results'] = pd.to_numeric(df['Predicted Results'], errors='coerce').apply(
                                lambda x: labels[int(x)] if pd.notna(x) and 0 <= x < len(labels) else 'Unknown'
                            )

                    all_results_list.append(df)
                    os.remove(output_file)

            except (FileNotFoundError, subprocess.CalledProcessError, Exception) as e:
                error_detail = e.stderr if isinstance(e, subprocess.CalledProcessError) else str(e)
                print(f"Failed to process '{dataset}': {error_detail}")
                # Update error dataframe to match new columns
                error_df = pd.DataFrame([{"Dataset": dataset, "Sequence": "ERROR", "Predicted Results": str(e), "Confidence": "N/A"}])
                all_results_list.append(error_df)
                continue
        
        if not all_results_list:
            yield "Prediction finished, but no results were generated.", pd.DataFrame(), gr.update(visible=False)
            return

        final_df = pd.concat(all_results_list, ignore_index=True)
        
        # --- MODIFIED: Reorder columns for the new four-column format ---
        cols_ordered = ["Dataset", "Sequence", "Predicted Results", "Confidence"]
        # Ensure only existing columns are selected
        final_cols = [col for col in cols_ordered if col in final_df.columns]
        # Add any other columns that might exist
        final_cols += [col for col in final_df.columns if col not in final_cols]
        final_df = final_df[final_cols]

        final_csv_path = temp_dir / "final_function_prediction.csv"
        final_df.to_csv(final_csv_path, index=False)

        yield "All predictions completed!", final_df, gr.update(visible=True, value=str(final_csv_path))

    gr.Markdown("## ⚡️ Protein Function Prediction")
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### Model Configuration")
            model_dd = gr.Dropdown(choices=list(MODEL_MAPPING.keys()), label="Select Model", value="None")
            task_dd = gr.Dropdown(choices=list(DATASET_MAPPING.keys()), label="Select Task", value="None")
            dataset_cbg = gr.CheckboxGroup(label="Select Datasets to Predict", info="Select a task above to see available datasets.", value=[])
            fasta_file_upload = gr.File(label="Upload FASTA file", file_types=[".fasta", ".fa"])
            fasta_protein_display = gr.Textbox(label="Uploaded Protein", interactive=False, lines=7, max_lines=10, placeholder="Protein sequence will appear here...")
            predict_btn = gr.Button("🚀 Start Prediction", variant="primary")
        
        with gr.Column(scale=3):
            gr.Markdown("### Prediction Results")
            status_textbox = gr.Textbox(label="Status", interactive=False, lines=2)
            # --- MODIFIED: Update headers and add column_widths ---
            results_df = gr.DataFrame(
                label="Prediction Results", 
                headers=["Dataset", "Sequence", "Predicted Results", "Confidence"],
                column_widths=["15%", "40%", "25%", "20%"]
            )
            download_btn = gr.DownloadButton("💾 Download Results (CSV)", visible=False)

    # --- Event Listeners ---
    fasta_file_upload.upload(fn=display_protein_sequence_from_fasta, inputs=fasta_file_upload, outputs=fasta_protein_display)
    task_dd.change(fn=update_dataset_choices, inputs=task_dd, outputs=dataset_cbg)
    predict_btn.click(fn=handle_prediction, inputs=[model_dd, task_dd, dataset_cbg, fasta_file_upload], outputs=[status_textbox, results_df, download_btn])
    
    return {}