"""
Download tool tab: Gradio UI for database download (UniProt, NCBI, RCSB, AlphaFold, InterPro).
Scripts: src/tools/search/database/{uniprot, ncbi, rcsb, alphafold, interpro}/*.py
  - uniprot_sequence.py: -i/-f, -o, -m, -e
  - ncbi_sequence.py: -i/-f, -o, -m, -e
  - rcsb_metadata.py: -i/-f (pdb_id), -o, -e
  - rcsb_structure.py: -i/-f, -o, -t, -u, -e
  - alphafold_structure.py: -i/-f, -o, -e
  - alphafold_metadata.py: -i/-f, -o, -e
  - interpro_metadata.py: -i/-f, -o, -e
  - interpro_proteins.py: --interpro_id/--interpro_json, --out_dir, --error_file
"""
import os
import time
import subprocess
import tempfile
import json
import hashlib
import tarfile
import gradio as gr
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Union
from web.utils.common_utils import get_save_path

# Maximum number of items to process in batch mode
MAX_BATCH_ITEMS = 50

# Placeholder for 3D viewer (Molecule3D not compatible with Gradio 5 frontend)
def _structure_viewer_placeholder(msg: str = "Structure viewer placeholder.") -> str:
    return f'<div style="padding:1rem;color:#666;">{msg}</div>'


def tar_task_results(task_folder: str, task_type: str) -> str:
    timestamp = str(int(time.time()))
    tar_dir_path = get_save_path(task_type, "Download_data")
    tar_path = f"{tar_dir_path}/archives/{task_type}_{timestamp}.tar.gz"
    os.makedirs(os.path.dirname(tar_path), exist_ok=True)

    with tarfile.open(tar_path, "w:gz") as tar:
        for root, dirs, files in os.walk(task_folder):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, task_folder)
                tar.add(file_path, arcname=arcname)

    return tar_path


def get_first_file_preview(task_folder: str, max_lines: int = 10) -> str:
    """Read first max_lines of the first readable text file in task_folder.
    Returns preview text or error reason if unable to display."""
    if not task_folder or not os.path.exists(task_folder):
        return "No output folder yet."
    skip_suffixes = (".tar.gz", ".gz", "_failed.txt")
    skip_names = ("truncated_input.txt", "truncated_input.json")
    try:
        for root, dirs, files in os.walk(task_folder):
            if "archives" in dirs:
                dirs.remove("archives")
            for f in sorted(files):
                if f.endswith(skip_suffixes) or f in skip_names:
                    continue
                path = os.path.join(root, f)
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as fp:
                        lines = fp.readlines()[:max_lines]
                        content = "".join(lines).rstrip()
                        if content:
                            return f"[{f}] first {max_lines} lines:\n{content}"
                except OSError as e:
                    return f"Cannot read {f}: {e}"
                except Exception as e:
                    return f"Cannot preview {f}: {str(e)[:80]}"
        return "No readable text files in output directory"
    except Exception as e:
        return f"Error: {str(e)[:80]}"

def create_download_tool_tab(constant: Dict[str, Any]):
    def run_download_script(script_path: str, **kwargs) -> str:
        """Runs an external Python script using subprocess."""
        script_full = script_path if script_path.startswith("src/") else f"src/tools/search/database/{script_path}"
        cmd = ["python", script_full]
        for k, v in kwargs.items():
            if v is None: continue
            if isinstance(v, bool):
                if v: cmd.append(f"--{k}")
            elif isinstance(v, str) and v.startswith("--"):
                cmd.append(v)
            else:
                cmd.extend([f"--{k}", str(v)])
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return f"Download completed successfully:\n{result.stdout}"
        except subprocess.CalledProcessError as e:
            return f"Error during download script execution:\n{e.stderr}"
        except FileNotFoundError:
            return f"Error: Script not found at '{script_full}'. Please check the path."
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"

    def parse_pdb_chains(pdb_file_path: str) -> List[str]:
        """Parses a PDB file to find all unique chain IDs."""
        chains = set()
        try:
            with open(pdb_file_path, 'r') as f:
                for line in f:
                    if line.startswith('ATOM'):
                        chain_id = line[21:22].strip()
                        if chain_id:
                            chains.add(chain_id)
        except Exception as e:
            print(f"Error parsing PDB chains: {e}")
        return sorted(list(chains))

    def read_uploaded_file(file_path: str, max_lines: int = 20) -> Tuple[List[str], str]:
        """Read and preview uploaded file content with batch size limitation."""
        if not file_path or not os.path.exists(file_path):
            return [], "No file uploaded"
        
        preview_text = ""
        is_truncated = False
        truncated_for_batch = False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            
            # Check if we need to truncate for batch processing
            original_count = len(lines)
            if original_count > MAX_BATCH_ITEMS:
                lines = lines[:MAX_BATCH_ITEMS]
                truncated_for_batch = True
            
            # Create preview (limited to max_lines for display)
            preview_lines = lines[:max_lines]
            preview_text += "\n".join(preview_lines)
            
            if len(lines) > max_lines:
                preview_text += f"\n... and {len(lines) - max_lines} more entries (showing first {max_lines})"
            
            # Add batch limitation warning
            if truncated_for_batch:
                preview_text += f"\n\n⚠️  BATCH LIMIT: Only processing first {MAX_BATCH_ITEMS} items out of {original_count} total items."
                preview_text += f"\nRemaining {original_count - MAX_BATCH_ITEMS} items will be ignored."
            
            return lines, preview_text
            
        except Exception as e:
            return [], f"Error reading file: {str(e)}"

    def read_json_file(file_path: str, max_items: int = 10) -> Tuple[Any, str]:
        """Read and preview uploaded JSON file content with batch size limitation."""
        if not file_path or not os.path.exists(file_path):
            return None, "No file uploaded"
        
        preview_text = ""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            truncated_for_batch = False
            original_count = 0
            
            if isinstance(data, list):
                original_count = len(data)
                if original_count > MAX_BATCH_ITEMS:
                    data = data[:MAX_BATCH_ITEMS]
                    truncated_for_batch = True
                
                for i, item in enumerate(data[:max_items]):
                    preview_text += f"{i+1}. {item}\n"
                if len(data) > max_items:
                    preview_text += f"... and {len(data) - max_items} more entries"
                    
            elif isinstance(data, dict):
                original_count = len(data)
                if original_count > MAX_BATCH_ITEMS:
                    # For dict, keep first MAX_BATCH_ITEMS keys
                    keys_to_keep = list(data.keys())[:MAX_BATCH_ITEMS]
                    data = {k: data[k] for k in keys_to_keep}
                    truncated_for_batch = True
                
                keys = list(data.keys())[:max_items]
                for key in keys:
                    preview_text += f"- {key}: {str(data[key])[:50]}...\n"
                if len(data) > max_items:
                    preview_text += f"... and {len(data) - max_items} more keys"
            else:
                preview_text = f"JSON content: {str(data)[:200]}..."
            
            # Add batch limitation warning
            if truncated_for_batch:
                preview_text += f"\n\n⚠️  BATCH LIMIT: Only processing first {MAX_BATCH_ITEMS} items out of {original_count} total items."
                preview_text += f"\nRemaining {original_count - MAX_BATCH_ITEMS} items will be ignored."
            
            return data, preview_text
            
        except Exception as e:
            return None, f"Error reading JSON file: {str(e)}"

    def save_truncated_file(original_file_path: str, lines: List[str], task_folder: str) -> str:
        """Save truncated file for batch processing."""
        if not lines:
            return original_file_path
        
        # Create truncated file in task folder
        truncated_file_path = os.path.join(task_folder, "truncated_input.txt")
        with open(truncated_file_path, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(f"{line}\n")
        return truncated_file_path

    def save_truncated_json(data: Any, task_folder: str) -> str:
        """Save truncated JSON for batch processing."""
        truncated_file_path = os.path.join(task_folder, "truncated_input.json")
        with open(truncated_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        return truncated_file_path

    def handle_interpro_download(method: str, id_val: str, file_val, user_hash: str, error: bool) -> tuple:
        """Handles InterPro entry metadata download by InterPro ID(s)."""
        timestamp = str(int(time.time()))
        task_folder = get_save_path("Download_data", "InterPro")
        file_path = None
        if method == "From File" and file_val:
            lines, preview = read_uploaded_file(file_val.name)
            if lines:
                file_path = save_truncated_file(file_val.name, lines, task_folder)

        args = {
            "interpro_id": id_val if method == "Single ID" else None,
            "interpro_id_file": file_path,
            "out_dir": task_folder,
            "error_file": f"{task_folder}/{timestamp}_failed.txt" if error else None
        }
        download_result = run_download_script("interpro/interpro_metadata.py", **args)
        tar_path = tar_task_results(task_folder, "interpro_metadata")
        preview = get_first_file_preview(task_folder)
        return preview, download_result, gr.update(visible=True, value=tar_path)

    def handle_rcsb_download(method: str, id_val: str, file_val, user_hash: str, error: bool) -> tuple:
        """Handles RCSB metadata download."""
        timestamp = str(int(time.time()))
        task_folder = get_save_path("Download_data", "RCSB")
        
        file_path = None
        if method == "From File" and file_val:
            lines, preview = read_uploaded_file(file_val.name)
            if lines:
                file_path = save_truncated_file(file_val.name, lines, task_folder)
        
        args = {
            "pdb_id": id_val if method == "Single ID" else None,
            "pdb_id_file": file_path,
            "out_dir": task_folder,
            "error_file": f"{task_folder}/{timestamp}_failed.txt" if error else None
        }
        download_result = run_download_script("rcsb/rcsb_metadata.py", **args)
        tar_path = tar_task_results(task_folder, "rcsb_metadata")
        preview = get_first_file_preview(task_folder)
        return preview, download_result, gr.update(visible=True, value=tar_path)

    def handle_uniprot_download(method: str, id_val: str, file_val, user_hash: str, merge: bool, error: bool) -> tuple:
        """Handles UniProt sequence download."""
        timestamp = str(int(time.time()))
        task_folder = get_save_path("Download_data", "UniProt")
        
        file_path = None
        if method == "From File" and file_val:
            lines, preview = read_uploaded_file(file_val.name)
            if lines:
                file_path = save_truncated_file(file_val.name, lines, task_folder)
        
        args = {
            "uniprot_id": id_val if method == "Single ID" else None,
            "file": file_path,
            "out_dir": task_folder,
            "merge": "--merge" if merge else None,
            "error_file": f"{task_folder}/{timestamp}_failed.txt" if error else None
        }
        download_result = run_download_script("uniprot/uniprot_sequence.py", **args)
        tar_path = tar_task_results(task_folder, "uniprot_sequence")
        preview = get_first_file_preview(task_folder)
        return preview, download_result, gr.update(visible=True, value=tar_path)

    def handle_ncbi_download(method: str, id_val: str, file_val, user_hash: str, merge: bool, error: bool) -> tuple:
        """Handles NCBI sequence download."""
        timestamp = str(int(time.time()))
        task_folder = get_save_path("Download_data", "NCBI")

        file_path = None
        if method == "From File" and file_val:
            lines, preview = read_uploaded_file(file_val.name)
            if lines:
                file_path = save_truncated_file(file_val.name, lines, task_folder)
        
        args = {
            "id": id_val if method == "Single ID" else None,
            "file": file_path,
            "out_dir": task_folder,
            "merge": "--merge" if merge else None,
            "error_file": f"{task_folder}/failed.txt" if error else None
        }
        download_result = run_download_script("ncbi/ncbi_sequence.py", **args)
        tar_path = tar_task_results(task_folder, "ncbi_sequence")
        preview = get_first_file_preview(task_folder)
        return preview, download_result, gr.update(visible=True, value=tar_path)

    def handle_struct_download(method: str, id_val: str, file_upload, user_hash: str, type_val: str, error: bool) -> tuple:
        """Handles RCSB structure download and updates the UI."""
        timestamp = str(int(time.time()))
        task_folder = get_save_path("Download_data", "RCSB")
        
        file_path = None
        if method == "From File" and file_upload:
            lines, preview = read_uploaded_file(file_upload.name)
            if lines:
                file_path = save_truncated_file(file_upload.name, lines, task_folder)
        
        download_output = run_download_script(
            "rcsb/rcsb_structure.py",
            pdb_id=id_val if method == "Single ID" else None,
            pdb_id_file=file_path,
            out_dir=task_folder, 
            type=type_val,
            unzip="--unzip",
            error_file=f"{task_folder}/failed.txt" if error else None
        )
        viz_status = "Download finished."
        tar_path = tar_task_results(task_folder, "rcsb_structure")
        
        # Update viewer placeholder for single ID downloads
        viewer_html = _structure_viewer_placeholder("Use the Save button to download the PDB archive.")
        if "Download completed successfully" in download_output and method == "Single ID":
            pdb_file = Path(task_folder) / f"{id_val}.{type_val}"
            if pdb_file.exists():
                viewer_html = _structure_viewer_placeholder(f"✅ Downloaded {id_val}.{type_val}. Use the Save button to get the file and open in PyMOL/Mol* etc.")
                viz_status = f"✅ Structure {id_val}.{type_val} from RCSB PDB"
            else:
                viz_status = f"✅ Downloaded {id_val} but file not found for viewing"
        elif "Download completed successfully" in download_output:
            viz_status = "Batch download complete. Select a single ID to view."
        else:
            viz_status = "Download failed."
        preview = get_first_file_preview(task_folder)
        return preview, download_output, viz_status, viewer_html, gr.update(visible=True, value=tar_path)

    def handle_af_download(method: str, id_val: str, file_upload, user_hash: str, error: bool) -> tuple:
        """Handles AlphaFold structure download and updates the UI."""
        timestamp = str(int(time.time()))
        task_folder = get_save_path("Download_data", "AlphaFold")
        
        file_path = None
        if method == "From File" and file_upload:
            lines, preview = read_uploaded_file(file_upload.name)
            if lines:
                file_path = save_truncated_file(file_upload.name, lines, task_folder)
        
        download_output = run_download_script(
            "alphafold/alphafold_structure.py",
            uniprot_id=id_val if method == "Single ID" else None,
            uniprot_id_file=file_path,
            out_dir=task_folder, error_file=f"{task_folder}/{timestamp}_failed.txt" if error else None
        )
        viz_status = "Download finished."
        tar_path = tar_task_results(task_folder, "AlphaFold_structure")
        
        # Update viewer placeholder for single ID downloads
        viewer_html = _structure_viewer_placeholder("Use the Save button to download the PDB archive.")
        if "Download completed successfully" in download_output and method == "Single ID":
            pdb_file = Path(task_folder) / f"{id_val}.pdb"
            if pdb_file.exists():
                viewer_html = _structure_viewer_placeholder(f"✅ Downloaded {id_val}.pdb. Use the Save button to get the file and open in PyMOL/Mol* etc.")
                viz_status = f"✅ Structure {id_val}.pdb from AlphaFold2"
            else:
                viz_status = f"✅ Downloaded {id_val} but file not found for viewing"
        elif "Download completed successfully" in download_output:
            viz_status = "Batch download complete. Select a single ID to view."
        else:
            viz_status = "Download failed."
        preview = get_first_file_preview(task_folder)
        return preview, download_output, viz_status, viewer_html, gr.update(visible=True, value=tar_path)

    # --- Gradio UI Layout ---
    with gr.Column():  # Use Column instead of Blocks - nested Blocks causes Component ID error in Gradio 6
        gr.Markdown("## Download Center")
        gr.Markdown(f"**Note**: Batch downloads are limited to {MAX_BATCH_ITEMS} items maximum for performance reasons.")

        with gr.Tabs():
            # --- Sequence and Metadata Download Tab ---
            with gr.Tab("🧬 Sequence Download"):
                gr.Markdown("### Protein Sequence and Annotation Information Download")
                with gr.Tabs():
                    # UniProt Sequences Download
                    with gr.Tab("UniProt Sequences"):
                        gr.Markdown("#### Download protein sequences from UniProt in FASTA format")
                        with gr.Row(equal_height=False):
                            with gr.Column(scale=2):
                                uniprot_method = gr.Radio(["Single ID", "From File"], label="Download Method", value="Single ID")
                                uniprot_id = gr.Textbox(label="UniProt ID", value="P00734", placeholder="e.g., P00734", interactive=True)
                                with gr.Column(visible=False) as uniprot_file_column:
                                    uniprot_file_upload = gr.File(label="Upload UniProt ID List", file_types=[".txt"])
                                    uniprot_file_example = gr.Examples(
                                        examples=[["./example/download/uniprot.txt"]],
                                        inputs=uniprot_file_upload,
                                        label="Click example to load"
                                    )
                                uniprot_merge = gr.Checkbox(label="Merge into single FASTA file", value=False)
                                uniprot_error = gr.Checkbox(label="Save error file", value=True)
                                uniprot_btn = gr.Button("Start Download", variant="primary")
                                uniprot_user = gr.Textbox(label="User Hash", value="default_user", visible=False)
                            with gr.Column(scale=3):
                                uniprot_preview = gr.Textbox(label="File Preview", interactive=False, lines=5, visible=False)
                                uniprot_result_preview = gr.Textbox(label="File Preview", interactive=False, lines=10, value="")
                                uniprot_output = gr.Textbox(label="Download Status", interactive=False, lines=3)
                                uniprot_download_btn = gr.DownloadButton(label="Save Downloaded Data", visible=False)
                        # Event handlers for UniProt
                        uniprot_method.change(
                            lambda method: [
                                gr.update(visible=(method == "From File")),
                                gr.update(interactive=(method == "Single ID"))
                            ],
                            inputs=uniprot_method,
                            outputs=[uniprot_file_column, uniprot_id]
                        )
                        uniprot_file_upload.change(
                            lambda f: read_uploaded_file(f.name if f else None)[1] if f else "",
                            inputs=uniprot_file_upload,
                            outputs=uniprot_preview
                        )
                        uniprot_btn.click(
                            fn=handle_uniprot_download, 
                            inputs=[uniprot_method, uniprot_id, uniprot_file_upload, uniprot_user, uniprot_merge, uniprot_error], 
                            outputs=[uniprot_result_preview, uniprot_output, uniprot_download_btn]
                        )
                    
                    # NCBI Sequences Download
                    with gr.Tab("NCBI Sequences"):
                        gr.Markdown("#### Download protein sequences from NCBI in FASTA format")
                        with gr.Row(equal_height=False):
                            with gr.Column(scale=2):
                                ncbi_method = gr.Radio(["Single ID", "From File"], label="Download Method", value="Single ID")
                                ncbi_id = gr.Textbox(label="NCBI ID", value="NP_000517.1", visible=True, placeholder="e.g., NP_000517.1", interactive=True)
                                with gr.Column(visible=False) as ncbi_file_column:
                                    ncbi_file_upload = gr.File(label="Upload NCBI ID List", file_types=[".txt"])
                                    ncbi_file_example = gr.Examples(
                                        examples=[["./example/download/ncbi.txt"]],
                                        inputs=ncbi_file_upload,
                                        label="Click example to load"
                                    )
                                ncbi_merge = gr.Checkbox(label="Merge into single FASTA file", value=False)
                                ncbi_error = gr.Checkbox(label="Save error file", value=True)
                                ncbi_btn = gr.Button("Start Download", variant="primary")
                                ncbi_user = gr.Textbox(label="User Hash", value="default_user", visible=False)
                            with gr.Column(scale=3):
                                ncbi_result_preview = gr.Textbox(label="File Preview", interactive=False, lines=10, value="")
                                ncbi_output = gr.Textbox(label="Download Status", interactive=False, lines=3)
                                ncbi_download_btn = gr.DownloadButton(label="Save Downloaded Data", visible=False)
                        # Event handlers for InterPro
                        ncbi_method.change(
                            lambda method: [
                                gr.update(visible=(method == "From File")),
                                gr.update(interactive=(method == "Single ID"))
                            ],
                            inputs=ncbi_method,
                            outputs=[ncbi_file_column, ncbi_id]
                        )
                        ncbi_btn.click(
                            fn=handle_ncbi_download, 
                            inputs=[ncbi_method, ncbi_id, ncbi_file_upload, ncbi_user, ncbi_merge, ncbi_error], 
                            outputs=[ncbi_result_preview, ncbi_output, ncbi_download_btn]
                        )
                    
               

            with gr.Tab("⚛️ Structure Download"):
                gr.Markdown("### Protein Structure Download and Visualization")
                # Main Tabs to switch between RCSB and AlphaFold
                with gr.Tabs():

                    # --- RCSB PDB Tab ---
                    with gr.Tab("RCSB PDB"):
                        # A row to create the left-right layout
                        with gr.Row(equal_height=False):
                            
                            # Left Column: Controls
                            with gr.Column(scale=2):
                                gr.Markdown("#### Download from RCSB PDB")
                                struct_method = gr.Radio(["Single ID", "From File"], label="Method", value="Single ID")
                                struct_id = gr.Textbox(label="PDB ID", value="1a0j", placeholder="e.g., 1crn", visible=True, interactive=True)
                                struct_type = gr.Dropdown(["pdb", "cif"], value="pdb", label="File Type")
                                struct_untar = gr.Checkbox(label="Untar files", value=True)
                                struct_error = gr.Checkbox(label="Save error log", value=True)
                                with gr.Column(visible=False) as struct_file_column:
                                    struct_file_upload = gr.File(label="Upload PDB ID List", file_types=[".txt"])
                                    struct_file_example = gr.Examples(
                                        examples=[["./example/download/rcsb.txt"]],
                                        inputs=struct_file_upload,
                                        label="Click example to load"
                                    )
                                    struct_preview = gr.Textbox(label="File Preview", interactive=False, lines=5)

                                struct_btn = gr.Button("Start Download", variant="primary")
                                struct_user = gr.Textbox(label="User Hash", value="default_user", visible=False) # Hidden field
                            # Right Column: Viewer and Outputs
                            with gr.Column(scale=3):
                                struct_result_preview = gr.Textbox(label="File Preview", interactive=False, lines=10, value="")
                                struct_output = gr.Textbox(label="Download Status", interactive=False, lines=3)
                                struct_viz_status = gr.Textbox(label="Visualization Status", interactive=False, value="Enter a PDB ID and click 'Download'")
                                struct_molecule_viewer = gr.HTML(
                                    value=_structure_viewer_placeholder("Download a structure to get the PDB file; use the Save button to open in an external viewer.")
                                )
                                
                                struct_download_btn = gr.DownloadButton(label="Save Downloaded Data", visible=False)

                    # --- AlphaFold2 Tab ---
                    with gr.Tab("AlphaFold2"):
                        # A row to create the left-right layout
                        with gr.Row(equal_height=False):

                            # Left Column: Controls
                            with gr.Column(scale=2):
                                gr.Markdown("#### Download from AlphaFold DB")
                                af_method = gr.Radio(["Single ID", "From File"], label="Method", value="Single ID")
                                af_id = gr.Textbox(label="UniProt ID", value="P00734", placeholder="e.g., P0DTD1", visible=True, interactive=True)
                                af_untar = gr.Checkbox(label="Untar files", value=True) # Note: AF files are not tarped, so this may not be needed
                                af_error = gr.Checkbox(label="Save error log", value=True)
                                with gr.Column(visible=False) as af_file_column:
                                    af_file_upload = gr.File(label="Upload AlphaFold DB ID List", file_types=[".txt"])
                                    af_file_example = gr.Examples(
                                        examples=[["./example/download/uniprot.txt"]],
                                        inputs=af_file_upload,
                                        label="Click example to load"
                                    )
                                    af_preview = gr.Textbox(label="File Preview", interactive=False, lines=5)

                                af_btn = gr.Button("Start Download", variant="primary")
                                af_user = gr.Textbox(label="User Hash", value="default_user", visible=False) # Hidden field
                            # Right Column: Viewer and Outputs
                            with gr.Column(scale=3):
                                af_result_preview = gr.Textbox(label="File Preview", interactive=False, lines=10, value="")
                                af_output = gr.Textbox(label="Download Status", interactive=False, lines=3)
                                af_viz_status = gr.Textbox(label="Visualization Status", interactive=False, value="Enter a UniProt ID and click 'Download'")
                                af_molecule_viewer = gr.HTML(
                                    value=_structure_viewer_placeholder("Download a structure to get the PDB file; use the Save button to open in an external viewer.")
                                )
                                af_download_btn = gr.DownloadButton(label="Save Downloaded Data", visible=False)

                        # File preview handlers
                        struct_method.change(
                            lambda method: [
                                gr.update(visible=(method == "From File")),
                                gr.update(interactive=(method == "Single ID"))
                            ],
                            inputs=struct_method, 
                            outputs=[struct_file_column, struct_id]
                        )
                        af_method.change(
                            lambda method: [
                                gr.update(visible=(method == "From File")),
                                gr.update(interactive=(method == "Single ID"))
                            ], 
                            inputs=af_method, 
                            outputs=[af_file_column, af_id]
                        )
                        struct_file_upload.change(
                            lambda f: read_uploaded_file(f.name if f else None)[1] if f else "",
                            inputs=struct_file_upload,
                            outputs=struct_preview
                        )
                        af_file_upload.change(
                            lambda f: read_uploaded_file(f.name if f else None)[1] if f else "",
                            inputs=af_file_upload,
                            outputs=af_preview
                        )
                        struct_btn.click(
                            handle_struct_download, 
                            inputs=[struct_method, struct_id, struct_file_upload, struct_user, struct_type, struct_error], 
                            outputs=[struct_result_preview, struct_output, struct_viz_status, struct_molecule_viewer, struct_download_btn]
                        )
                        af_btn.click(
                            handle_af_download, 
                            inputs=[af_method, af_id, af_file_upload, af_user, af_error], 
                            outputs=[af_result_preview, af_output, af_viz_status, af_molecule_viewer, af_download_btn]
                        )
    
            # --- Structure Download Tab ---
            with gr.Tab("🗂️ Metadata Download"):
                gr.Markdown("### Annotation Information Download")
                with gr.Tabs():
                    # RCSB Metadata Download
                    with gr.Tab("RCSB Metadata"):
                        gr.Markdown("#### Download metadata for PDB entries from RCSB")
                        with gr.Row(equal_height=False):
                            with gr.Column(scale=2):
                                rcsb_method = gr.Radio(["Single ID", "From File"], label="Download Method", value="Single ID")
                                rcsb_id = gr.Textbox(label="PDB ID", value="1a0j", visible=True, placeholder="e.g., 1a0j", interactive=True)
                                with gr.Column(visible=False) as rcsb_file_column:
                                    rcsb_file_upload = gr.File(label="Upload PDB ID List", file_types=[".txt"])
                                    rcsb_file_example = gr.Examples(
                                        examples=[["./example/download/rcsb.txt"]],
                                        inputs=rcsb_file_upload,
                                        label="Click example to load"
                                    )
                                rcsb_error = gr.Checkbox(label="Save error file", value=True)
                                rcsb_btn = gr.Button("Start Download", variant="primary")
                                rcsb_user = gr.Textbox(label="User Hash", value="default_user", visible=False)
                            with gr.Column(scale=3):
                                rcsb_result_preview = gr.Textbox(label="File Preview", interactive=False, lines=10, value="")
                                rcsb_output = gr.Textbox(label="Download Status", interactive=False, lines=3)
                                rcsb_download_btn = gr.DownloadButton(label="Save Downloaded Data", visible=False)
                        # Event handlers for RCSB
                        rcsb_method.change(
                            lambda method: [
                                gr.update(visible=(method == "From File")),
                                gr.update(interactive=(method == "Single ID"))
                            ],
                            inputs=rcsb_method,
                            outputs=[rcsb_file_column, rcsb_id]
                        )
                        rcsb_btn.click(
                            fn=handle_rcsb_download, 
                            inputs=[rcsb_method, rcsb_id, rcsb_file_upload, rcsb_user, rcsb_error], 
                            outputs=[rcsb_result_preview, rcsb_output, rcsb_download_btn]
                        )
                    
                    # InterPro Metadata Download
                    with gr.Tab("InterPro Metadata"):
                        gr.Markdown("#### Download InterPro entry/family metadata by InterPro ID (e.g. IPR000001)")
                        with gr.Row(equal_height=False):
                            with gr.Column(scale=2):
                                interpro_method = gr.Radio(["Single ID", "From File"], label="Download Method", value="Single ID")
                                interpro_id = gr.Textbox(label="InterPro ID", value="IPR000001", visible=True, placeholder="e.g., IPR000001", interactive=True)
                                with gr.Column(visible=False) as interpro_file_column:
                                    interpro_file_upload = gr.File(label="Upload InterPro ID List (one per line)", file_types=[".txt"])
                                    interpro_file_example = gr.Examples(
                                        examples=[["./example/database/interpro/interpro.txt"]],
                                        inputs=interpro_file_upload,
                                        label="Click example to load"
                                    )
                                interpro_error = gr.Checkbox(label="Save error file", value=True)
                                interpro_btn = gr.Button("Start Download", variant="primary")
                                interpro_user = gr.Textbox(label="User Hash", value="default_user", visible=False)
                            with gr.Column(scale=3):
                                interpro_preview = gr.Textbox(label="File Preview", interactive=False, lines=5, visible=False)
                                interpro_result_preview = gr.Textbox(label="File Preview", interactive=False, lines=10, value="")
                                interpro_output = gr.Textbox(label="Download Status", interactive=False, lines=3)
                                interpro_download_btn = gr.DownloadButton(label="Save Downloaded Data", visible=False)
                        # Event handlers for InterPro
                        interpro_method.change(
                            lambda method: [
                                gr.update(visible=(method == "From File")),
                                gr.update(interactive=(method == "Single ID"))
                            ],
                            inputs=interpro_method,
                            outputs=[interpro_file_column, interpro_id]
                        )
                        interpro_file_upload.change(
                            lambda f: read_uploaded_file(f.name if f else None)[1] if f else "",
                            inputs=interpro_file_upload,
                            outputs=interpro_preview
                        )
                        interpro_btn.click(
                            fn=handle_interpro_download,
                            inputs=[interpro_method, interpro_id, interpro_file_upload, interpro_user, interpro_error],
                            outputs=[interpro_result_preview, interpro_output, interpro_download_btn]
                        )
            
    return {}