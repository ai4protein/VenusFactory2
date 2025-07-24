import gradio as gr
import os
import json
import re
import zipfile
import subprocess # Import the subprocess module
import time
import threading
from pathlib import Path
from typing import Dict, Any, List, Generator, Optional, Tuple

# --- Constants ---
# Configuration file for the 3D structure viewer (Dash app)
CONFIG_FILE = "./src/web/pdb_config.txt"
# URL for the Dash 3D structure viewer application
DASH_VIEWER_URL = "http://127.0.0.1:8050"

# --- Helper Functions ---
def run_download_script(script_name: str, **kwargs) -> str:
    """
    Runs an external Python script using subprocess.

    Args:
        script_name: The name of the script to run from the 'src/crawler/' directory.
        **kwargs: Command-line arguments to pass to the script.

    Returns:
        The stdout or stderr from the script execution.
    """
    cmd = ["python", f"src/crawler/{script_name}"]
    for k, v in kwargs.items():
        if v is None:
            continue
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        elif isinstance(v, str) and v.startswith("--"):
            cmd.append(v)
        else:
            cmd.extend([f"--{k}", str(v)])
    
    try:
        # Using check=True will raise CalledProcessError on non-zero exit codes.
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return f"Download completed successfully:\n{result.stdout}"
    except subprocess.CalledProcessError as e:
        return f"Error during download script execution:\n{e.stderr}"
    except FileNotFoundError:
        return f"Error: Script not found at 'src/crawler/{script_name}'. Please check the path."
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def handle_interpro_download(method: str, id_val: str, json_val: str, out_dir: str, error: bool) -> str:
    """Handles InterPro metadata download."""
    args = {
        "interpro_id": id_val if method == "Single ID" else None,
        "interpro_json": json_val if method == "From JSON" else None,
        "out_dir": out_dir,
        "error_file": f"{out_dir}/failed.txt" if error else None
    }
    return run_download_script("metadata/download_interpro.py", **args)    

def handle_rcsb_download(method: str, id_val: str, file_val: str, out_dir: str, error: bool) -> str:
    """Handles RCSB metadata download."""
    args = {
        "pdb_id": id_val if method == "Single ID" else None,
        "pdb_id_file": file_val if method == "From File" else None,
        "out_dir": out_dir,
        "error_file": f"{out_dir}/failed.txt" if error else None
    }
    return run_download_script("metadata/download_rcsb.py", **args)

def handle_uniprot_download(method: str, id_val: str, file_val: str, out_dir: str, merge: bool, error: bool) -> str:
    """Handles UniProt sequence download."""
    args = {
        "uniprot_id": id_val if method == "Single ID" else None,
        "file": file_val if method == "From File" else None,
        "out_dir": out_dir,
        "merge": "--merge" if merge else None,
        "error_file": f"{out_dir}/failed.txt" if error else None
    }
    return run_download_script("sequence/download_uniprot_seq.py", **args)

def handle_struct_download(method: str, id_val: str, file_val: str, out_dir: str, type_val: str, unzip: bool, error: bool) -> Tuple[str, str]:
    """
    Handles RCSB structure download and updates the 3D viewer.

    Args:
        method: Download method (Single ID or From File).
        id_val: PDB ID for single download.
        file_val: File path for batch download.
        out_dir: Output directory.
        type_val: Structure file type.
        unzip: Whether to unzip downloaded files.
        error: Whether to save error file.
        
    Returns:
        A tuple containing download output message and visualization status message.
    """
    # Download the structure
    if method == "Single ID":
        download_output = run_download_script(
            "structure/download_rcsb.py",
            pdb_id=id_val,
            out_dir=out_dir,
            type=type_val,
            unzip="--unzip" if unzip else None,
            error_file=f"{out_dir}/failed.txt" if error else None
        )
        
        # Visualize the downloaded structure if successful and PDB type
        if "Download completed successfully" in download_output:
            # PDB IDs are typically lowercase in file names
            pdb_file = Path(out_dir) / f"{id_val.lower()}.{type_val}"
            if type_val == "pdb" and pdb_file.exists():
                # Signal the Dash app to update by writing the new PDB path.
                try:
                    with open(CONFIG_FILE, "w") as f:
                        f.write(str(pdb_file))
                    viz_status = f"✅ Sent '{os.path.basename(pdb_file)}' to viewer. It will refresh automatically."
                except Exception as e:
                    viz_status = f"❌ Failed to update viewer config: {str(e)}"
            else:
                viz_status = f"Cannot visualize {type_val} format or file not found: {pdb_file}"
        else:
            viz_status = "Download failed, cannot visualize."
    else: # From File method
        download_output = run_download_script(
            "structure/download_rcsb.py",
            pdb_id_file=file_val,
            out_dir=out_dir,
            type=type_val,
            unzip="--unzip" if unzip else None,
            error_file=f"{out_dir}/failed.txt" if error else None
        )
        viz_status = "Batch download completed. Select a single ID to visualize."
    return download_output, viz_status

            
def handle_af_download(method: str, id_val: str, file_val: str, out_dir: str, index_level: int, error: bool, unzip: bool) -> Tuple[str, str]:
    """
    Handles downloading AlphaFold structures and then signaling the Dash app to update.
    """
    # Step 1: Run the download script
    args = {
        "uniprot_id": id_val if method == "Single ID" else None,
        "uniprot_id_file": file_val if method == "From File" else None,
        "out_dir": out_dir,
        "index_level": index_level,
        "error_file": f"{out_dir}/failed.txt" if error else None
    }
    download_output = run_download_script("structure/download_alphafold.py", **args)

    # Step 2: If download was successful for a single ID, signal the Dash app
    viz_status = "Visualization is only available for 'Single ID' downloads."
    if method == "Single ID":
        if "Download completed successfully" in download_output:
            # AlphaFold file names have a standard format (UniProt ID.pdb)
            pdb_file = Path(out_dir) / f"{id_val}.pdb"
            if pdb_file.exists():
                # Signal the Dash app to update by writing the new PDB path.
                try:
                    with open(CONFIG_FILE, "w") as f:
                        f.write(str(pdb_file))
                    viz_status = f"✅ Sent '{os.path.basename(pdb_file)}' to viewer. It will refresh automatically."
                except Exception as e:
                    viz_status = f"❌ Failed to update viewer config: {str(e)}"
            else:
                viz_status = f"❌ Download OK, but PDB file not found: {pdb_file}"
        else:
            viz_status = "❌ Download failed. Cannot visualize."

    return download_output, viz_status

# --- Visibility Update Functions ---
def update_interpro_visibility(method: str) -> Dict[str, gr.update]:
    """Updates visibility of InterPro input fields based on selected method."""
    return {
        interpro_id: gr.update(visible=(method == "Single ID")),
        interpro_json: gr.update(visible=(method == "From JSON"))
    }

def update_rcsb_visibility(method: str) -> Dict[str, gr.update]:
    """Updates visibility of RCSB metadata input fields based on selected method."""
    return {
        rcsb_id: gr.update(visible=(method == "Single ID")),
        rcsb_file: gr.update(visible=(method == "From File"))
    }

def update_uniprot_visibility(method: str) -> Dict[str, gr.update]:
    """Updates visibility of UniProt input fields based on selected method."""
    return {
        uniprot_id: gr.update(visible=(method == "Single ID")),
        uniprot_file: gr.update(visible=(method == "From File"))
    }

def update_af_visibility(method: str) -> Dict[str, gr.update]:
    """Updates visibility of AlphaFold input fields based on selected method."""
    return {
        af_id: gr.update(visible=(method == "Single ID")),
        af_file: gr.update(visible=(method == "From File"))
    }

def update_struct_visibility(method: str) -> Dict[str, gr.update]:
    """Update visibility of RCSB structure input fields based on selected method."""
    return {
        struct_id: gr.update(visible=(method == "Single ID")),
        struct_file: gr.update(visible=(method == "From File"))
    }

# --- Main Tab Creation Function ---
def create_tool_download_tab(constant: Dict[str, Any]):
    """
    Creates the main download tab, containing sequence/metadata download and structure download functionalities.
    """
    with gr.Blocks() as download_tab_content:
        gr.Markdown("## Download Center")

        with gr.Tabs():
            # --- Sequence and Metadata Download Tab ---
            with gr.TabItem("🧬 Sequence & Metadata Download"):
                gr.Markdown("### Protein/Nucleic Acid Sequence and Annotation Information Download")
                with gr.Tabs():
                    # InterPro Metadata Download
                    with gr.TabItem("InterPro Metadata"):
                        gr.Markdown("#### Download protein domain, family, and Gene Ontology annotation metadata from InterPro")
                        with gr.Row():
                            interpro_method = gr.Radio(["Single ID", "From JSON"], label="Download Method", value="Single ID")
                        interpro_id = gr.Textbox(label="InterPro ID", value="IPR000001", visible=True, placeholder="e.g., IPR000001")
                        interpro_json = gr.Textbox(label="InterPro JSON Path", value="download/interpro_domain/interpro_json.customization", visible=False, placeholder="e.g., path/to/interpro_ids.json")
                        interpro_out = gr.Textbox(label="Output Directory", value="temp_downloads/interpro_metadata")
                        interpro_error = gr.Checkbox(label="Save error file", value=True)
                        interpro_btn = gr.Button("Download InterPro Metadata")
                        interpro_output = gr.Textbox(label="Output", interactive=False, lines=5)

                        interpro_method.change(fn=update_interpro_visibility, inputs=interpro_method, outputs=[interpro_id, interpro_json])
                        interpro_btn.click(fn=handle_interpro_download, inputs=[interpro_method, interpro_id, interpro_json, interpro_out, interpro_error], outputs=interpro_output)

                    # RCSB Metadata Download
                    with gr.TabItem("RCSB Metadata"):
                        gr.Markdown("#### Download metadata for PDB entries from RCSB")
                        with gr.Row():
                            rcsb_method = gr.Radio(["Single ID", "From File"], label="Download Method", value="Single ID")

                        rcsb_id = gr.Textbox(label="PDB ID", value="1a0j", visible=True, placeholder="e.g., 1a0j")
                        rcsb_file = gr.Textbox(label="PDB List File Path", value="download/rcsb.txt", visible=False, placeholder="e.g., path/to/pdb_ids.txt")
                        rcsb_out = gr.Textbox(label="Output Directory", value="temp_downloads/rcsb_metadata")
                        rcsb_error = gr.Checkbox(label="Save error file", value=True)
                        rcsb_btn = gr.Button("Download RCSB Metadata")
                        rcsb_output = gr.Textbox(label="Output", interactive=False, lines=5)

                        rcsb_method.change(fn=update_rcsb_visibility, inputs=rcsb_method, outputs=[rcsb_id, rcsb_file])
                        rcsb_btn.click(fn=handle_rcsb_download, inputs=[rcsb_method, rcsb_id, rcsb_file, rcsb_out, rcsb_error], outputs=rcsb_output)

                    # UniProt Sequences Download
                    with gr.TabItem("UniProt Sequences"):
                        gr.Markdown("#### Download protein sequences from UniProt in FASTA format")
                        with gr.Row():
                            uniprot_method = gr.Radio(["Single ID", "From File"], label="Download Method", value="Single ID")
                        
                        uniprot_id = gr.Textbox(label="UniProt ID", value="P00734", placeholder="e.g., P00734")
                        uniprot_file = gr.Textbox(label="UniProt ID List File Path", value="download/uniprot.txt", visible=False, placeholder="e.g., path/to/uniprot_ids.txt")
                        uniprot_out = gr.Textbox(label="Output Directory", value="temp_downloads/uniprot_sequences")
                        uniprot_merge = gr.Checkbox(label="Merge into single FASTA file", value=False)
                        uniprot_error = gr.Checkbox(label="Save error file", value=True)
                        uniprot_btn = gr.Button("Download UniProt Sequences")
                        uniprot_output = gr.Textbox(label="Output", interactive=False, lines=5)

                        uniprot_method.change(fn=update_uniprot_visibility, inputs=uniprot_method, outputs=[uniprot_id, uniprot_file])
                        uniprot_btn.click(fn=handle_uniprot_download, inputs=[uniprot_method, uniprot_id, uniprot_file, uniprot_out, uniprot_merge, uniprot_error], outputs=uniprot_output)

            # --- Structure Download Tab ---
            with gr.TabItem("⚛️ Structure Download"):
                gr.Markdown("### Protein Structure Download and Visualization")

                with gr.Tabs():
                    # RCSB PDB Structure Download
                    with gr.TabItem("RCSB PDB Structure Download"):
                        gr.Markdown("#### Download protein/nucleic acid structure files from the RCSB PDB database")
                        with gr.Row(): # Use gr.Row for left-right layout
                            with gr.Column(scale=1): # Left column for inputs
                                with gr.Group():  # Group for better visual separation
                                    struct_method = gr.Radio(
                                        choices=["Single ID", "From File"],
                                        label="Download Method",
                                        value="Single ID"
                                    )
                                    
                                    struct_id = gr.Textbox(label="PDB ID", value="1a0j", visible=True, placeholder="e.g., 1CRN, 6M0J")
                                    struct_file = gr.Textbox(label="PDB List File Path", value="download/rcsb.txt", visible=False, placeholder="e.g., path/to/pdb_ids.txt")
                                    struct_out = gr.Textbox(label="Output Directory", value="temp_downloads/rcsb_structures")
                                    
                                    struct_type = gr.Dropdown(
                                        choices=["cif", "pdb", "pdb1", "xml", "sf", "mr", "mrstr"], 
                                        value="pdb", 
                                        label="Structure Type"
                                    )
                                    
                                    with gr.Row():
                                        struct_unzip = gr.Checkbox(label="Unzip downloaded files", value=True)
                                        struct_error = gr.Checkbox(label="Save error file", value=True)
                                    
                                    struct_btn = gr.Button("Download RCSB Structure", size="lg")
                                    
                                    struct_output = gr.Textbox(label="Download Output", interactive=False, lines=4)
                                    struct_viz_status = gr.Textbox(label="Visualization Status", interactive=False)
                            
                            with gr.Column(scale=2): # Right column for visualization
                                gr.HTML(
                                    value=f"""
                                    <div style="width: 100%; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; height: 650px;">
                                        <iframe src="{DASH_VIEWER_URL}" width="100%" height="100%" frameborder="0" style="display: block;"></iframe>
                                    </div>
                                    """,
                                    label="3D Structure Viewer"
                                )

                        # Event handlers
                        struct_method.change(fn=update_struct_visibility, inputs=struct_method, outputs=[struct_id, struct_file])
                        struct_btn.click(
                            fn=handle_struct_download,
                            inputs=[struct_method, struct_id, struct_file, struct_out, struct_type, struct_unzip, struct_error],
                            outputs=[struct_output, struct_viz_status]
                        )

                    # AlphaFold2 Structure Download
                    with gr.TabItem("AlphaFold2 Structure Download"):
                        gr.Markdown("#### Download predicted protein structures from the AlphaFold database and visualize")
                        with gr.Row(): # Use gr.Row for left-right layout
                            with gr.Column(scale=1): # Left column for inputs
                                af_method = gr.Radio(["Single ID", "From File"], label="Download Method", value="Single ID")
                                af_id = gr.Textbox(label="UniProt ID", value="P00734", placeholder="e.g., P0DTD1")
                                af_file = gr.Textbox(label="UniProt ID List File Path", value="download/uniprot.txt", visible=False, placeholder="e.g., path/to/uniprot_ids.txt")
                                af_out = gr.Textbox(label="Output Directory", value="temp_downloads/alphafold2_structures")
                                af_index_level = gr.Number(label="Index Level (for multi-model AlphaFold files)", value=0, precision=0)
                                af_unzip = gr.Checkbox(label="Unzip downloaded files", value=True) # Added unzip for AlphaFold
                                af_error = gr.Checkbox(label="Save error file", value=True)
                                af_btn = gr.Button("Download AlphaFold2 Structure")
                                af_output = gr.Textbox(label="Download Output", interactive=False, lines=4)
                                af_viz_status = gr.Textbox(label="Visualization Status", interactive=False)
                            
                            with gr.Column(scale=2): # Right column for visualization
                                gr.Markdown("""
                                    <div style="
                                        display: flex;
                                        justify-content: space-between;
                                        margin-bottom: 15px;
                                        text-align: center;
                                        gap: 10px;
                                    ">
                                        <div style="flex: 1; min-width: 0;">
                                            <div style="margin-bottom: 5px;">Very high (plDDT > 90)</div>
                                            <div style="
                                                height: 10px;
                                                background-color: #0053D6;
                                                margin: 0 auto;
                                                width: 100%;
                                                max-width: 220px; /* Adjust as needed */
                                                border-radius: 6px;
                                            "></div>
                                        </div>
                                        <div style="flex: 1; min-width: 0;">
                                            <div style="margin-bottom: 5px;">Confident (90 > plDDT > 70)</div>
                                            <div style="
                                                height: 10px;
                                                background-color: #65CBF3;
                                                margin: 0 auto;
                                                width: 100%;
                                                max-width: 220px;
                                                border-radius: 6px;
                                            "></div>
                                        </div>
                                        <div style="flex: 1; min-width: 0;">
                                            <div style="margin-bottom: 5px;">Low (70 > plDDT > 50)</div>
                                            <div style="
                                                height: 10px;
                                                background-color: #FFDB13;
                                                margin: 0 auto;
                                                width: 100%;
                                                max-width: 220px;
                                                border-radius: 6px;
                                            "></div>
                                        </div>
                                        <div style="flex: 1; min-width: 0;">
                                            <div style="margin-bottom: 5px;">Very low (plDDT < 50)</div>
                                            <div style="
                                                height: 10px;
                                                background-color: #FF7D45;
                                                margin: 0 auto;
                                                width: 100%;
                                                max-width: 220px;
                                                border-radius: 6px;
                                            "></div>
                                        </div>
                                    </div>
                                    """)
                                
                                gr.HTML(
                                    value=f"""
                                    <div style="width: 100%; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; height: 500px;">
                                        <iframe src="{DASH_VIEWER_URL}" width="100%" height="100%" frameborder="0" style="display: block;"></iframe>
                                    </div>
                                    """,
                                    label="AlphaFold2 Structure Viewer"
                                )
                        
                        # Event handlers
                        af_method.change(fn=update_af_visibility, inputs=af_method, outputs=[af_id, af_file])
                        af_btn.click(
                            fn=handle_af_download,
                            inputs=[af_method, af_id, af_file, af_out, af_index_level, af_error, af_unzip],
                            outputs=[af_output, af_viz_status]
                        )

    return download_tab_content