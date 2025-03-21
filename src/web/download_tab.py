import gradio as gr
import os
import subprocess
import numpy as np
import plotly.graph_objects as go
from Bio.PDB import PDBParser
import io
import base64
from typing import Dict, Any, Tuple, List, Optional

def create_download_tab(constant: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create the download tab with various options for downloading protein data.
    
    Args:
        constant: Dictionary containing constant values for the application
        
    Returns:
        Dictionary containing any state information
    """
    def run_download_script(script_name: str, **kwargs) -> str:
        """
        Run a download script with the specified arguments.
        
        Args:
            script_name: Name of the script to run
            **kwargs: Arguments to pass to the script
            
        Returns:
            Output of the script as a string
        """
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

    # Function to visualize protein structure using Plotly
    def visualize_protein_structure(pdb_file: str) -> Tuple[str, go.Figure]:
        """
        Visualize a protein structure from a PDB file using Plotly for interactive 3D visualization.
        
        Args:
            pdb_file: Path to the PDB file
            
        Returns:
            Tuple containing status message and Plotly figure
        """
        try:
            if not os.path.exists(pdb_file):
                return f"File not found: {pdb_file}", None
            
            # Parse the PDB file
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("protein", pdb_file)
            
            # Extract atom coordinates and information for all atoms
            all_atoms_x, all_atoms_y, all_atoms_z = [], [], []
            all_atoms_text = []  # For hover information
            all_atoms_color = []
            
            # Color mapping for different atom types
            color_map = {
                'C': '#333333',  # Dark gray for carbon
                'N': '#3050F8',  # Blue for nitrogen
                'O': '#FF2010',  # Red for oxygen
                'S': '#FFFF30',  # Yellow for sulfur
                'P': '#FF8000',  # Orange for phosphorus
                'H': '#E0E0E0',  # Light gray for hydrogen
                'CA': '#00FF00'  # Green for alpha carbon
            }
            
            # Extract backbone (CA atoms) for the ribbon representation
            ca_x, ca_y, ca_z = [], [], []
            ca_text = []
            
            # Track chains for coloring
            chains = {}
            chain_colors = [
                '#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD',
                '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF'
            ]
            
            # Create a Plotly figure
            fig = go.Figure()
            
            # Track the number of backbone traces for visibility control
            backbone_trace_count = 0

            # Extract coordinates and properties
            for model in structure:
                for chain in model:
                    chain_id = chain.get_id()
                    if chain_id not in chains:
                        chains[chain_id] = len(chains) % len(chain_colors)
                    
                    chain_color = chain_colors[chains[chain_id]]
                    
                    # Collect CA atoms for this chain
                    chain_ca_x, chain_ca_y, chain_ca_z = [], [], []
                    chain_ca_text = []
                    
                    for residue in chain:
                        res_id = residue.get_id()
                        res_name = residue.get_resname()
                        res_num = res_id[1]
                        
                        # Extract CA atoms for backbone trace
                        if 'CA' in residue:
                            ca = residue['CA'].get_coord()
                            chain_ca_x.append(ca[0])
                            chain_ca_y.append(ca[1])
                            chain_ca_z.append(ca[2])
                            chain_ca_text.append(f"Chain {chain_id}, {res_name} {res_num}")
                            
                            # Also add to global CA lists
                            ca_x.append(ca[0])
                            ca_y.append(ca[1])
                            ca_z.append(ca[2])
                            ca_text.append(f"Chain {chain_id}, {res_name} {res_num}")
                        
                        # Extract all atoms
                        for atom in residue:
                            coord = atom.get_coord()
                            all_atoms_x.append(coord[0])
                            all_atoms_y.append(coord[1])
                            all_atoms_z.append(coord[2])
                            
                            atom_name = atom.get_name()
                            atom_element = atom.element
                            
                            all_atoms_text.append(f"Chain {chain_id}, {res_name} {res_num}, {atom_name}")
                            
                            # Determine atom color
                            if atom_name == 'CA':
                                all_atoms_color.append(color_map.get('CA', '#808080'))
                            else:
                                all_atoms_color.append(color_map.get(atom_element, '#808080'))
                    
                    # Add this chain's CA atoms as a separate trace for better visualization
                    if chain_ca_x:
                        fig.add_trace(go.Scatter3d(
                            x=chain_ca_x,
                            y=chain_ca_y,
                            z=chain_ca_z,
                            mode='lines',
                            name=f'Chain {chain_id}',
                            line=dict(color=chain_color, width=8),  # Increased line width
                            text=chain_ca_text,
                            hoverinfo='text',
                            showlegend=True
                        ))
                        backbone_trace_count += 1
            
            # Add backbone trace (CA atoms as markers)
            fig.add_trace(go.Scatter3d(
                x=ca_x,
                y=ca_y,
                z=ca_z,
                mode='markers',
                name='Backbone',
                marker=dict(
                    size=7,  # Increased marker size
                    color='#00FF00',
                    opacity=0.8,
                    symbol='circle'
                ),
                text=ca_text,
                hoverinfo='text',
                showlegend=True
            ))
            backbone_trace_count += 1
            
            # Add all atoms as small markers
            fig.add_trace(go.Scatter3d(
                x=all_atoms_x,
                y=all_atoms_y,
                z=all_atoms_z,
                mode='markers',
                name='All Atoms',
                marker=dict(
                    size=2.5,
                    color=all_atoms_color,
                    opacity=0.6
                ),
                text=all_atoms_text,
                hoverinfo='text',
                showlegend=True,
                visible='legendonly'  # Hide by default, can be toggled in legend
            ))
            
            # Set layout properties
            pdb_id = os.path.basename(pdb_file).split('.')[0]
            fig.update_layout(
                title=dict(
                    text=f"Structure: {pdb_id}",
                    font=dict(size=20, family="Arial, sans-serif")
                ),
                scene=dict(
                    xaxis=dict(title='X (Å)', showbackground=False, showgrid=True, gridcolor='lightgray'),
                    yaxis=dict(title='Y (Å)', showbackground=False, showgrid=True, gridcolor='lightgray'),
                    zaxis=dict(title='Z (Å)', showbackground=False, showgrid=True, gridcolor='lightgray'),
                    aspectmode='data',
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    )
                ),
                margin=dict(l=0, r=0, b=0, t=40),
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="lightgray",
                    borderwidth=1
                ),
                template="plotly_white",
                height=600,  # Increase height for better visualization
                width=800    # Set width for better aspect ratio
            )
            
            # Create visibility arrays for the buttons
            # For "Backbone Only": all backbone traces visible, all atoms hidden
            backbone_only_visibility = [True] * backbone_trace_count + [False]
            # For "All Atoms": all traces visible
            all_atoms_visibility = [True] * (backbone_trace_count + 1)
            
            # Add buttons for different views
            fig.update_layout(
                updatemenus=[
                    dict(
                        type="buttons",
                        direction="right",
                        buttons=[
                            dict(
                                args=[{"visible": backbone_only_visibility}],
                                label="Backbone Only",
                                method="update"
                            ),
                            dict(
                                args=[{"visible": all_atoms_visibility}],
                                label="All Atoms",
                                method="update"
                            )
                        ],
                        pad={"r": 10, "t": 10},
                        showactive=True,
                        x=0.1,
                        xanchor="left",
                        y=1.1,
                        yanchor="top",
                        bgcolor="rgba(255, 255, 255, 0.8)",
                        bordercolor="lightgray",
                        borderwidth=1
                    ),
                ]
            )
            
            return f"Successfully visualized structure from {pdb_file}", fig
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error visualizing structure: {str(e)}\n{error_details}")
            return f"Error visualizing structure: {str(e)}", None

    # Create the main download tab
    with gr.Tab("Download"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Download Protein Data (See help for more details)")
                
        # InterPro Metadata tab
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
                """Update visibility of InterPro input fields based on selected method"""
                return {
                    interpro_id: gr.update(visible=(method == "Single ID")),
                    interpro_json: gr.update(visible=(method == "From JSON"))
                }
            
            interpro_method.change(
                fn=update_interpro_visibility,
                inputs=[interpro_method],
                outputs=[interpro_id, interpro_json]
            )

        # RCSB Metadata tab
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
                """Update visibility of RCSB input fields based on selected method"""
                return {
                    rcsb_id: gr.update(visible=(method == "Single ID")),
                    rcsb_file: gr.update(visible=(method == "From File"))
                }
            
            rcsb_method.change(
                fn=update_rcsb_visibility,
                inputs=[rcsb_method],
                outputs=[rcsb_id, rcsb_file]
            )

        # UniProt Sequences tab
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
                """Update visibility of UniProt input fields based on selected method"""
                return {
                    uniprot_id: gr.update(visible=(method == "Single ID")),
                    uniprot_file: gr.update(visible=(method == "From File"))
                }
            
            uniprot_method.change(
                fn=update_uniprot_visibility,
                inputs=[uniprot_method],
                outputs=[uniprot_id, uniprot_file]
            )

        # RCSB Structures tab
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
                
                # Right column for visualization
                with gr.Column(scale=5):
                    # Visualization section with full height
                    struct_viz = gr.Plot(label="Structure Visualization", elem_id="struct_viz_plot")

            def update_struct_visibility(method):
                """Update visibility of RCSB structure input fields based on selected method"""
                return {
                    struct_id: gr.update(visible=(method == "Single ID")),
                    struct_file: gr.update(visible=(method == "From File"))
                }
            
            struct_method.change(
                fn=update_struct_visibility,
                inputs=[struct_method],
                outputs=[struct_id, struct_file]
            )

        # AlphaFold2 Structures tab
        with gr.Tab("AlphaFold2 Structures"):
            with gr.Row():
                # Left column for inputs
                with gr.Column(scale=3):
                    with gr.Group():  # Group for better visual separation
                        af_method = gr.Radio(
                            choices=["Single ID", "From File"],
                            label="Download Method",
                            value="Single ID"
                        )
                        
                        # Input parameters section with consistent spacing
                        with gr.Row():
                            af_id = gr.Textbox(label="UniProt ID", value="P00734")
                        
                        with gr.Row():
                            af_file = gr.Textbox(label="UniProt ID List File", value="download/uniprot.txt", visible=False)
                        
                        with gr.Row():
                            af_out = gr.Textbox(label="Output Directory", value="download/alphafold2_structures")
                        
                        with gr.Row():
                            af_index_level = gr.Number(label="Index Level", value=0, precision=0)
                        
                        with gr.Row():
                            af_error = gr.Checkbox(label="Save error file", value=True)
                        
                        with gr.Row():
                            af_btn = gr.Button("Download AlphaFold Structures", size="lg")
                        
                        # Output section
                        af_output = gr.Textbox(label="Download Output", interactive=False, lines=4)
                        af_viz_status = gr.Textbox(label="Visualization Status", interactive=False)
                
                # Right column for visualization
                with gr.Column(scale=5):
                    # Visualization section with full height
                    af_viz = gr.Plot(label="Structure Visualization", elem_id="af_viz_plot")

            def update_af_visibility(method):
                """Update visibility of AlphaFold input fields based on selected method"""
                return {
                    af_id: gr.update(visible=(method == "Single ID")),
                    af_file: gr.update(visible=(method == "From File"))
                }
            
            af_method.change(
                fn=update_af_visibility,
                inputs=[af_method],
                outputs=[af_id, af_file]
            )

    # Handler functions for download buttons
    def handle_interpro_download(method, id_val, json_val, out_dir, error):
        """Handle InterPro data download"""
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
        """Handle RCSB metadata download"""
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
        """Handle UniProt sequence download"""
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
                    viz_status, viz_fig = visualize_protein_structure(pdb_file)
                    return download_output, viz_status, viz_fig
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

    struct_btn.click(
        fn=handle_struct_download,
        inputs=[struct_method, struct_id, struct_file, struct_out, struct_type, struct_unzip, struct_error],
        outputs=[struct_output, struct_viz_status, struct_viz]
    )

    def handle_af_download(method, id_val, file_val, out_dir, index_level, error):
        """
        Handle AlphaFold structure download and visualization
        
        Args:
            method: Download method (Single ID or From File)
            id_val: UniProt ID for single download
            file_val: File path for batch download
            out_dir: Output directory
            index_level: Index level for directory structure
            error: Whether to save error file
            
        Returns:
            Tuple containing download output, visualization status, and Plotly figure
        """
        # Download the structure
        if method == "Single ID":
            download_output = run_download_script(
                "structure/download_alphafold.py",
                uniprot_id=id_val,
                out_dir=out_dir,
                index_level=index_level,
                error_file=f"{out_dir}/failed.txt" if error else None
            )
            
            # Visualize the downloaded structure
            if "Download completed successfully" in download_output:
                # Try different possible file paths
                possible_paths = [
                    f"{out_dir}/AF-{id_val}-F1-model_v4.pdb",
                    f"{out_dir}/{id_val}.pdb"
                ]
                
                for pdb_file in possible_paths:
                    if os.path.exists(pdb_file):
                        viz_status, viz_fig = visualize_protein_structure(pdb_file)
                        return download_output, viz_status, viz_fig
                
                return download_output, f"PDB file not found in expected locations", None
            else:
                return download_output, "Download failed, cannot visualize", None
        else:
            download_output = run_download_script(
                "structure/download_alphafold.py",
                uniprot_id_file=file_val,
                out_dir=out_dir,
                index_level=index_level,
                error_file=f"{out_dir}/failed.txt" if error else None
            )
            return download_output, "Batch download completed, select a single ID to visualize", None

    af_btn.click(
        fn=handle_af_download,
        inputs=[af_method, af_id, af_file, af_out, af_index_level, af_error],
        outputs=[af_output, af_viz_status, af_viz]
    )

    return {}
