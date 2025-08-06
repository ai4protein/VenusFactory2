import gradio as gr
import os
import subprocess
from typing import Dict, Any

CONFIG_FILE ="./src/web/pdb_config.txt"


def create_download_tab(constant: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates the Gradio UI components and logic for all download-related tabs.
    
    Args:
        constant: A dictionary which can be used to pass configuration constants.
        
    Returns:
        An empty dictionary, as components are defined within the Gradio Blocks scope.
    """

    def run_download_script(script_name: str, **kwargs) -> str:
        """
        Runs an external Python script using subprocess.
        
        Args:
            script_name: The name of the script to run from the 'src/crawler/' directory.
            **kwargs: Command-line arguments to pass to the script.
            
        Returns:
            The stdout or stderr from the script execution.
        """
        # Note: Ensure the path 'src/crawler/' is correct relative to where webui.py is run.
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

     # Create the main download tab
    with gr.Tab("Download"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Download Protein Data (See help for more details)")
    
        with gr.Tabs():
            with gr.Tab("InterPro Metadata"):
                with gr.Row():
                    interpro_method = gr.Radio(["Single ID", "From JSON"], label="Download Method", value="Single ID")
                
                interpro_id = gr.Textbox(label="InterPro ID", value="IPR000001", visible=True)
                interpro_json = gr.Textbox(label="InterPro JSON Path", value="download/interpro_domain/interpro_json.customization", visible=False)
                interpro_out = gr.Textbox(label="Output Directory", value="download/interpro_domain")
                interpro_error = gr.Checkbox(label="Save error file", value=True)
                interpro_btn = gr.Button("Download InterPro Data")
                interpro_output = gr.Textbox(label="Output", interactive=False, lines=5)

                def update_interpro_visibility(method):
                    return {
                        interpro_id: gr.update(visible=(method == "Single ID")),
                        interpro_json: gr.update(visible=(method == "From JSON"))
                    }
                interpro_method.change(fn=update_interpro_visibility, inputs=interpro_method, outputs=[interpro_id, interpro_json])

                def handle_interpro_download(method, id_val, json_val, out_dir, error):
                    args = {
                        "interpro_id": id_val if method == "Single ID" else None,
                        "interpro_json": json_val if method == "From JSON" else None,
                        "out_dir": out_dir,
                        "error_file": f"{out_dir}/failed.txt" if error else None
                    }
                    return run_download_script("metadata/download_interpro.py", **args)
                interpro_btn.click(fn=handle_interpro_download, inputs=[interpro_method, interpro_id, interpro_json, interpro_out, interpro_error], outputs=interpro_output)


            with gr.Tab("RCSB Metadata"):
                with gr.Row():
                    rcsb_method = gr.Radio(["Single ID", "From File"], label="Download Method", value="Single ID")

                rcsb_id = gr.Textbox(label="PDB ID", value="1a0j", visible=True)
                rcsb_file = gr.Textbox(label="PDB List File", value="download/rcsb.txt", visible=False)
                rcsb_out = gr.Textbox(label="Output Directory", value="download/rcsb_metadata")
                rcsb_error = gr.Checkbox(label="Save error file", value=True)
                rcsb_btn = gr.Button("Download RCSB Metadata")
                rcsb_output = gr.Textbox(label="Output", interactive=False, lines=5)

                def update_rcsb_visibility(method):
                    return {
                        rcsb_id: gr.update(visible=(method == "Single ID")),
                        rcsb_file: gr.update(visible=(method == "From File"))
                    }
                rcsb_method.change(fn=update_rcsb_visibility, inputs=rcsb_method, outputs=[rcsb_id, rcsb_file])

                def handle_rcsb_download(method, id_val, file_val, out_dir, error):
                    args = {
                        "pdb_id": id_val if method == "Single ID" else None,
                        "pdb_id_file": file_val if method == "From File" else None,
                        "out_dir": out_dir,
                        "error_file": f"{out_dir}/failed.txt" if error else None
                    }
                    return run_download_script("metadata/download_rcsb.py", **args)
                rcsb_btn.click(fn=handle_rcsb_download, inputs=[rcsb_method, rcsb_id, rcsb_file, rcsb_out, rcsb_error], outputs=rcsb_output)


            with gr.Tab("UniProt Sequences"):
                with gr.Row():
                    uniprot_method = gr.Radio(["Single ID", "From File"], label="Download Method", value="Single ID")
                
                uniprot_id = gr.Textbox(label="UniProt ID", value="P00734")
                uniprot_file = gr.Textbox(label="UniProt ID List File", value="download/uniprot.txt", visible=False)
                uniprot_out = gr.Textbox(label="Output Directory", value="download/uniprot_sequences")
                uniprot_merge = gr.Checkbox(label="Merge into single FASTA", value=False)
                uniprot_error = gr.Checkbox(label="Save error file", value=True)
                uniprot_btn = gr.Button("Download UniProt Sequences")
                uniprot_output = gr.Textbox(label="Output", interactive=False, lines=5)

                def update_uniprot_visibility(method):
                    return {
                        uniprot_id: gr.update(visible=(method == "Single ID")),
                        uniprot_file: gr.update(visible=(method == "From File"))
                    }
                uniprot_method.change(fn=update_uniprot_visibility, inputs=uniprot_method, outputs=[uniprot_id, uniprot_file])

                def handle_uniprot_download(method, id_val, file_val, out_dir, merge, error):
                    args = {
                        "uniprot_id": id_val if method == "Single ID" else None,
                        "file": file_val if method == "From File" else None,
                        "out_dir": out_dir,
                        "merge": "--merge" if merge else None,
                        "error_file": f"{out_dir}/failed.txt" if error else None
                    }
                    return run_download_script("sequence/download_uniprot_seq.py", **args)
                uniprot_btn.click(fn=handle_uniprot_download, inputs=[uniprot_method, uniprot_id, uniprot_file, uniprot_out, uniprot_merge, uniprot_error], outputs=uniprot_output)

            with gr.Tab("RCSB Structures"):
                with gr.Row():
                    # Left column for inputs
                    with gr.Column(scale=3):
                        with gr.Group():  # Group for better visual separation
                            struct_method = gr.Radio(
                                choices=["Single ID", "From File"],
                                label="Download Method",
                                value="Single ID"
                            )
                            
                            # Input parameters section with consistent spacing
                            with gr.Row():
                                struct_id = gr.Textbox(label="PDB ID", value="1a0j")
                            
                            with gr.Row():
                                struct_file = gr.Textbox(label="PDB List File", value="download/rcsb.txt", visible=False)
                            
                            with gr.Row():
                                struct_out = gr.Textbox(label="Output Directory", value="download/rcsb_structures")
                            
                            with gr.Row():
                                struct_type = gr.Dropdown(
                                    choices=["cif", "pdb", "pdb1", "xml", "sf", "mr", "mrstr"], 
                                    value="pdb", 
                                    label="Structure Type"
                                )
                            
                            with gr.Row():
                                with gr.Column(scale=1):
                                    struct_unzip = gr.Checkbox(label="Unzip downloaded files", value=True)
                                with gr.Column(scale=1):
                                    struct_error = gr.Checkbox(label="Save error file", value=True)
                            
                            with gr.Row():
                                struct_btn = gr.Button("Download RCSB Structures", size="lg")
                            
                            # Output section
                            struct_output = gr.Textbox(label="Download Output", interactive=False, lines=4)
                            struct_viz_status = gr.Textbox(label="Visualization Status", interactive=False)
                        
                    with gr.Column(scale=5):
                        # Visualization section with full height
                        gr.HTML("""
                            <div style="width: 1000px; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; height: 600px;">
                                <iframe src="http://127.0.0.1:8050" width="100%" height="100%" frameborder="0" style="display: block;"></iframe>
                            </div>
                            """)


            with gr.Tab("AlphaFold2 Structures"):
                with gr.Row():
                    with gr.Column(scale=2): # Left column for inputs
                        af_method = gr.Radio(["Single ID", "From File"], label="Download Method", value="Single ID")
                        af_id = gr.Textbox(label="UniProt ID", value="P00734")
                        af_file = gr.Textbox(label="UniProt ID List File", value="download/uniprot.txt", visible=False)
                        af_out = gr.Textbox(label="Output Directory", value="download/alphafold2_structures")
                        af_index_level = gr.Number(label="Index Level", value=0, precision=0)
                        af_error = gr.Checkbox(label="Save error file", value=True)
                        af_btn = gr.Button("Download and Visualize")
                        af_output = gr.Textbox(label="Download Output", interactive=False, lines=5)
                        af_viz_status = gr.Textbox(label="Visualization Status", interactive=False)
                    



                    with gr.Column(scale=3): # Right column for visualization
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
                                        width: 200%;
                                        max-width: 220px;
                                        border-radius: 6px;
                                    "></div>
                                </div>
                                <div style="flex: 1; min-width: 0;">
                                    <div style="margin-bottom: 5px;">Confident (90 > plDDT > 70)</div>
                                    <div style="
                                        height: 10px;
                                        background-color: #65CBF3;
                                        margin: 0 auto;
                                        width: 200%;
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
                                        width: 200%;
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
                                        width: 200%;
                                        max-width: 220px;
                                        border-radius: 6px;
                                    "></div>
                                </div>
                            </div>
                            """)
                        
                
                        gr.HTML("""
                            <div style="width: 1050px; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; height: 600px;">
                                <iframe src="http://127.0.0.1:8050" width="100%" height="100%" frameborder="0" style="display: block;"></iframe>
                            </div>
                            """)

                def update_af_visibility(method):
                    return {
                        af_id: gr.update(visible=(method == "Single ID")),
                        af_file: gr.update(visible=(method == "From File"))
                    }
                af_method.change(fn=update_af_visibility, inputs=af_method, outputs=[af_id, af_file])

                def handle_struct_download(method, id_val, file_val, out_dir, type_val, unzip, error):
                    """
                    Handle RCSB structure download and visualization
                    
                    Args:
                        method: Download method (Single ID or From File)
                        id_val: PDB ID for single download
                        file_val: File path for batch download
                        out_dir: Output directory
                        type_val: Structure file type
                        unzip: Whether to unzip downloaded files
                        error: Whether to save error file
                        
                    Returns:
                        Tuple containing download output, visualization status, and Plotly figure
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
                        
                        # Visualize the downloaded structure
                        if "Download completed successfully" in download_output:
                            pdb_file = f"{out_dir}/{id_val.lower()}.{type_val}"
                            if type_val == "pdb" and os.path.exists(pdb_file):
                                with open(CONFIG_FILE, "w") as f:
                                    f.write(pdb_file)
                                viz_status = f"✅ Sent '{os.path.basename(pdb_file)}' to viewer. It will refresh automatically."
                                return download_output, viz_status
                            else:
                                return download_output, f"Cannot visualize {type_val} format or file not found", None
                        else:
                            return download_output, "Download failed, cannot visualize", None
                    else:
                        download_output = run_download_script(
                            "structure/download_rcsb.py",
                            pdb_id_file=file_val,
                            out_dir=out_dir,
                            type=type_val,
                            unzip="--unzip" if unzip else None,
                            error_file=f"{out_dir}/failed.txt" if error else None
                        )
                        return download_output, "Batch download completed, select a single ID to visualize", None
                    return download_output, viz_status
                
                struct_btn.click(
                    fn=handle_struct_download,
                    inputs=[struct_method, struct_id, struct_file, struct_out, struct_type, struct_unzip, struct_error],
                    outputs=[struct_output, struct_viz_status]
                )

                def handle_af_download(method, id_val, file_val, out_dir, index_level, error):
                    """
                    Handles downloading the file and then signaling the Dash app to update.
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
                            # AlphaFold file names have a standard format
                            pdb_file = f"{out_dir}/{id_val}.pdb"
                            if os.path.exists(pdb_file):
                                # Signal the Dash app to update by writing the new PDB path.
                                with open(CONFIG_FILE, "w") as f:
                                    f.write(pdb_file)
                                viz_status = f"✅ Sent '{os.path.basename(pdb_file)}' to viewer. It will refresh automatically."
                            else:
                                viz_status = f"❌ Download OK, but PDB file not found: {pdb_file}"
                        else:
                            viz_status = "❌ Download failed. Cannot visualize."


                af_btn.click(
                    fn=handle_af_download,
                    inputs=[af_method, af_id, af_file, af_out, af_index_level, af_error],
                    outputs=[af_output, af_viz_status]
                )

    return {}