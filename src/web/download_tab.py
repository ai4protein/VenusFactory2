import gradio as gr
import os
import subprocess
from typing import Dict, Any

def create_download_tab(constant: Dict[str, Any]) -> Dict[str, Any]:
    def run_download_script(script_name: str, **kwargs) -> str:
        cmd = ["python", f"src/crawler/{script_name}"]
        for k, v in kwargs.items():
            if v is None:  # Skip None values
                continue
            if isinstance(v, bool):  # Handle boolean flags
                if v:
                    cmd.append(f"--{k}")
            elif v == "--merge":  # Handle special merge flag
                cmd.append(v)
            else:  # Handle regular arguments
                cmd.extend([f"--{k}", str(v)])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return f"Download completed successfully\n{result.stdout}"
            else:
                return f"Error during download:\n{result.stderr}"
        except Exception as e:
            return f"Failed to run download script: {str(e)}"

    

    with gr.Tab("Download"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Download Protein Data (See help for more details)")
                
        with gr.Tab("InterPro Metadata"):
            with gr.Row():
                interpro_method = gr.Radio(
                    choices=["Single ID", "From JSON"],
                    label="Download Method",
                    value="Single ID"
                )
            
            with gr.Column():
                interpro_id = gr.Textbox(label="InterPro ID", value="IPR000001")
                interpro_json = gr.Textbox(label="InterPro JSON Path", value="download/interpro_domain/interpro_json.customization", visible=False)
                interpro_out = gr.Textbox(label="Output Directory", value="download/interpro_domain")
                interpro_error = gr.Checkbox(label="Save error file", value=True)
                interpro_btn = gr.Button("Download InterPro Data")
                interpro_output = gr.Textbox(label="Output", interactive=False)

            def update_interpro_visibility(method):
                return {
                    interpro_id: gr.update(visible=(method == "Single ID")),
                    interpro_json: gr.update(visible=(method == "From JSON"))
                }
            
            interpro_method.change(
                fn=update_interpro_visibility,
                inputs=[interpro_method],
                outputs=[interpro_id, interpro_json]
            )

        with gr.Tab("RCSB Metadata"):
            with gr.Row():
                rcsb_method = gr.Radio(
                    choices=["Single ID", "From File"],
                    label="Download Method",
                    value="Single ID"
                )
            
            with gr.Column():
                rcsb_id = gr.Textbox(label="PDB ID", value="1a0j")
                rcsb_file = gr.Textbox(label="PDB List File", value="download/rcsb.txt", visible=False)
                rcsb_out = gr.Textbox(label="Output Directory", value="download/rcsb_metadata")
                rcsb_error = gr.Checkbox(label="Save error file", value=True)
                rcsb_btn = gr.Button("Download RCSB Metadata")
                rcsb_output = gr.Textbox(label="Output", interactive=False)

            def update_rcsb_visibility(method):
                return {
                    rcsb_id: gr.update(visible=(method == "Single ID")),
                    rcsb_file: gr.update(visible=(method == "From File"))
                }
            
            rcsb_method.change(
                fn=update_rcsb_visibility,
                inputs=[rcsb_method],
                outputs=[rcsb_id, rcsb_file]
            )

        with gr.Tab("UniProt Sequences"):
            with gr.Row():
                uniprot_method = gr.Radio(
                    choices=["Single ID", "From File"],
                    label="Download Method",
                    value="Single ID"
                )
            
            with gr.Column():
                uniprot_id = gr.Textbox(label="UniProt ID", value="P00734")
                uniprot_file = gr.Textbox(label="UniProt ID List File", value="download/uniprot.txt", visible=False)
                uniprot_out = gr.Textbox(label="Output Directory", value="download/uniprot_sequences")
                uniprot_merge = gr.Checkbox(label="Merge into single FASTA", value=False)
                uniprot_error = gr.Checkbox(label="Save error file", value=True)
                uniprot_btn = gr.Button("Download UniProt Sequences")
                uniprot_output = gr.Textbox(label="Output", interactive=False)

            def update_uniprot_visibility(method):
                return {
                    uniprot_id: gr.update(visible=(method == "Single ID")),
                    uniprot_file: gr.update(visible=(method == "From File"))
                }
            
            uniprot_method.change(
                fn=update_uniprot_visibility,
                inputs=[uniprot_method],
                outputs=[uniprot_id, uniprot_file]
            )

        with gr.Tab("RCSB Structures"):
            with gr.Row():
                struct_method = gr.Radio(
                    choices=["Single ID", "From File"],
                    label="Download Method",
                    value="Single ID"
                )
            
            with gr.Column():
                struct_id = gr.Textbox(label="PDB ID", value="1a0j")
                struct_file = gr.Textbox(label="PDB List File", value="download/rcsb.txt", visible=False)
                struct_out = gr.Textbox(label="Output Directory", value="download/rcsb_structures")
                struct_type = gr.Dropdown(
                    choices=["cif", "pdb", "pdb1", "xml", "sf", "mr", "mrstr"], 
                    value="pdb", 
                    label="Structure Type"
                )
                struct_unzip = gr.Checkbox(label="Unzip downloaded files", value=True)
                struct_error = gr.Checkbox(label="Save error file", value=True)
                struct_btn = gr.Button("Download RCSB Structures")
                struct_output = gr.Textbox(label="Output", interactive=False)

            def update_struct_visibility(method):
                return {
                    struct_id: gr.update(visible=(method == "Single ID")),
                    struct_file: gr.update(visible=(method == "From File"))
                }
            
            struct_method.change(
                fn=update_struct_visibility,
                inputs=[struct_method],
                outputs=[struct_id, struct_file]
            )

        with gr.Tab("AlphaFold2 Structures"):
            with gr.Row():
                af_method = gr.Radio(
                    choices=["Single ID", "From File"],
                    label="Download Method",
                    value="Single ID"
                )
            
            with gr.Column():
                af_id = gr.Textbox(label="UniProt ID", value="P00734")
                af_file = gr.Textbox(label="UniProt ID List File", value="download/uniprot.txt", visible=False)
                af_out = gr.Textbox(label="Output Directory", value="download/alphafold2_structures")
                af_index_level = gr.Number(label="Index Level", value=0, precision=0)
                af_error = gr.Checkbox(label="Save error file", value=True)
                af_btn = gr.Button("Download AlphaFold Structures")
                af_output = gr.Textbox(label="Output", interactive=False)

            def update_af_visibility(method):
                return {
                    af_id: gr.update(visible=(method == "Single ID")),
                    af_file: gr.update(visible=(method == "From File"))
                }
            
            af_method.change(
                fn=update_af_visibility,
                inputs=[af_method],
                outputs=[af_id, af_file]
            )
        
        
        def load_help_text():
            help_path = os.path.join(os.path.dirname(__file__), "download_help.md")
            with open(help_path, 'r') as f:
                return f.read()
        
        with gr.Tab("Help"):
            gr.Markdown(load_help_text())


    def handle_interpro_download(method, id_val, json_val, out_dir, error):
        if method == "Single ID":
            return run_download_script(
                "metadata/download_interpro.py",
                interpro_id=id_val,
                out_dir=out_dir,
                error_file=f"{out_dir}/failed.txt" if error else None
            )
        else:
            return run_download_script(
                "metadata/download_interpro.py",
                interpro_json=json_val,
                out_dir=out_dir,
                error_file=f"{out_dir}/failed.txt" if error else None
            )

    interpro_btn.click(
        fn=handle_interpro_download,
        inputs=[interpro_method, interpro_id, interpro_json, interpro_out, interpro_error],
        outputs=interpro_output
    )

    def handle_rcsb_download(method, id_val, file_val, out_dir, error):
        if method == "Single ID":
            return run_download_script(
                "metadata/download_rcsb.py",
                pdb_id=id_val,
                out_dir=out_dir,
                error_file=f"{out_dir}/failed.txt" if error else None
            )
        else:
            return run_download_script(
                "metadata/download_rcsb.py",
                pdb_id_file=file_val,
                out_dir=out_dir,
                error_file=f"{out_dir}/failed.txt" if error else None
            )

    rcsb_btn.click(
        fn=handle_rcsb_download,
        inputs=[rcsb_method, rcsb_id, rcsb_file, rcsb_out, rcsb_error],
        outputs=rcsb_output
    )

    def handle_uniprot_download(method, id_val, file_val, out_dir, merge, error):
        if method == "Single ID":
            return run_download_script(
                "sequence/download_uniprot_seq.py",
                uniprot_id=id_val,
                out_dir=out_dir,
                merge="--merge" if merge else None,
                error_file=f"{out_dir}/failed.txt" if error else None
            )
        else:
            return run_download_script(
                "sequence/download_uniprot_seq.py",
                file=file_val,
                out_dir=out_dir,
                merge="--merge" if merge else None,
                error_file=f"{out_dir}/failed.txt" if error else None
            )

    uniprot_btn.click(
        fn=handle_uniprot_download,
        inputs=[uniprot_method, uniprot_id, uniprot_file, uniprot_out, uniprot_merge, uniprot_error],
        outputs=uniprot_output
    )

    def handle_struct_download(method, id_val, file_val, out_dir, type_val, unzip, error):
        if method == "Single ID":
            return run_download_script(
                "structure/download_rcsb.py",
                pdb_id=id_val,
                out_dir=out_dir,
                type=type_val,
                unzip="--unzip" if unzip else None,
                error_file=f"{out_dir}/failed.txt" if error else None
            )
        else:
            return run_download_script(
                "structure/download_rcsb.py",
                pdb_id_file=file_val,
                out_dir=out_dir,
                type=type_val,
                unzip="--unzip" if unzip else None,
                error_file=f"{out_dir}/failed.txt" if error else None
            )

    struct_btn.click(
        fn=handle_struct_download,
        inputs=[struct_method, struct_id, struct_file, struct_out, struct_type, struct_unzip, struct_error],
        outputs=struct_output
    )

    def handle_af_download(method, id_val, file_val, out_dir, index_level, error):
        if method == "Single ID":
            return run_download_script(
                "structure/download_alphafold.py",
                uniprot_id=id_val,
                out_dir=out_dir,
                index_level=index_level,
                error_file=f"{out_dir}/failed.txt" if error else None
            )
        else:
            return run_download_script(
                "structure/download_alphafold.py",
                uniprot_id_file=file_val,
                out_dir=out_dir,
                index_level=index_level,
                error_file=f"{out_dir}/failed.txt" if error else None
            )

    af_btn.click(
        fn=handle_af_download,
        inputs=[af_method, af_id, af_file, af_out, af_index_level, af_error],
        outputs=af_output
    )

    return {}
