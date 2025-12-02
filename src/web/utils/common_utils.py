"""Common utility functions for VenusFactory."""

import os
import re
import tarfile
from pathlib import Path
from datetime import datetime
from typing import Dict
import gradio as gr


def sanitize_filename(name: str) -> str:
    """Sanitize filename for safe file operations."""
    name = re.split(r'[|\s/]', name)[0]
    return re.sub(r'[^\w\-. ]', '_', name)


def get_save_path(subdir1: str, subdir2: str | None = None) -> Path:
    """Get save path with date-based directory structure."""
    now = datetime.now()
    date_path = Path("temp_outputs") / f"{now.year}/{now.month:02d}/{now.day:02d}"

    if subdir2:
        path = date_path / subdir1 / subdir2
    else:
        path = date_path / subdir1

    path.mkdir(parents=True, exist_ok=True)
    return path


def toggle_ai_section(is_checked: bool):
    """Toggle visibility of AI configuration section."""
    return gr.update(visible=is_checked)


def create_tar_archive(files_to_archive: Dict[str, str], output_filename: str) -> str:
    """
    Create tar.gz archive with specified files.
    
    Args:
        files_to_archive: Dict mapping source paths to archive names (internal paths)
        output_filename: The output filename (e.g., 'archive.tar.gz')
    """
    with tarfile.open(output_filename, "w:gz") as tar:
        for src, arcname in files_to_archive.items():
            if os.path.exists(src):
                tar.add(src, arcname=arcname)
            else:
                print(f"Warning: File not found: {src}")
                
    return output_filename


def format_physical_chemical_properties(data: dict) -> str:
    """Format physical and chemical properties results."""
    result = ""
    result += f"Sequence length: {data['sequence_length']} aa\n"
    result += f"Molecular weight: {data['molecular_weight'] / 1000:.2f} kDa\n"
    result += f"Theoretical pI: {data['theoretical_pI']}\n"
    result += f"Aromaticity: {data['aromaticity']}\n"
    result += f"Instability index: {data['instability_index']}\n"
    
    if data['instability_index'] > 40:
        result += "  ⚠️ Predicted as unstable protein\n"
    else:
        result += "  ✅ Predicted as stable protein\n"
    
    result += f"GRAVY: {data['gravy']}\n"
    
    ssf = data['secondary_structure_fraction']
    result += f"Secondary structure prediction: Helix={ssf['helix']}, Turn={ssf['turn']}, Sheet={ssf['sheet']}\n"
    
    return result


def format_rsa_results(data: dict) -> str:
    """Format RSA calculation results."""
    result = ""
    result += f"Exposed residues: {data['exposed_residues']}\n"
    result += f"Buried residues: {data['buried_residues']}\n"
    result += f"Total residues: {data['total_residues']}\n"
    
    def get_residue_number_rsa(item):
        """Get residue number from item for sorting."""
        return int(item[0])
    
    try:
        sorted_residues = sorted(data['residue_rsa'].items(), key=get_residue_number_rsa)
    except ValueError:
        sorted_residues = sorted(data['residue_rsa'].items())
    
    for res_id, res_data in sorted_residues:
        aa = res_data['aa']
        rsa = res_data['rsa']
        location = "Exposed (surface)" if rsa >= 0.25 else "Buried (core)"
        result += f"  Residue {res_id} ({aa}): RSA = {rsa:.3f} ({location})\n"
    return result


def format_sasa_results(data: dict) -> str:
    """Format SASA calculation results."""
    result = ""
    result += f"{'Chain':<6} {'Residue':<12} {'SASA (Ų)':<15}\n"
    
    for chain_id, chain_data in sorted(data['chains'].items()):
        result += f"--- Chain {chain_id} (Total SASA: {chain_data['total_sasa']:.2f} Ų) ---\n"
        
        def get_residue_number_sasa(item):
            """Get residue number from item for sorting."""
            return int(item[0])
        
        try:
            sorted_residues = sorted(chain_data['residue_sasa'].items(), key=get_residue_number_sasa)
        except ValueError:
            sorted_residues = sorted(chain_data['residue_sasa'].items())
        
        for res_num, res_data in sorted_residues:
            res_id_str = f"{res_data['resname']}{res_num}"
            result += f"{chain_id:<6} {res_id_str:<12} {res_data['sasa']:<15.2f}\n"

    return result


def format_secondary_structure_results(data: dict) -> str:
    """Format secondary structure calculation results."""
    result = f"Successfully calculated secondary structure for chain '{data['chain_id']}'\n"
    result += f"Sequence length: {len(data['aa_sequence'])}\n"
    result += f"Helix (H): {data['ss_counts']['helix']} ({data['ss_counts']['helix']/len(data['aa_sequence'])*100:.1f}%)\n"
    result += f"Sheet (E): {data['ss_counts']['sheet']} ({data['ss_counts']['sheet']/len(data['aa_sequence'])*100:.1f}%)\n"
    result += f"Coil (C): {data['ss_counts']['coil']} ({data['ss_counts']['coil']/len(data['aa_sequence'])*100:.1f}%)\n"
    
    def get_residue_number_ss(item):
        """Get residue number from item for sorting."""
        return int(item[0])
    
    try:
        sorted_residues = sorted(data['residue_ss'].items(), key=get_residue_number_ss)
    except ValueError:
        sorted_residues = sorted(data['residue_ss'].items())
    
    for res_id, res_data in sorted_residues:
        result += f"  Residue {res_id} ({res_data['aa_seq']}): ss8: {res_data['ss8_seq']} ({res_data['ss8_name']}), ss3: {res_data['ss3_seq']}\n"
    
    result += f"aa_seq: {data['aa_sequence']}\n"
    result += f"ss8_seq: {data['ss8_sequence']}\n"
    result += f"ss3_seq: {data['ss3_sequence']}\n"
    
    return result

