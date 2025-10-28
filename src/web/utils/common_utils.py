"""Common utility functions for Venus Factory."""

import os
import re
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Dict
import gradio as gr


def sanitize_filename(name: str) -> str:
    """Sanitize filename for safe file operations."""
    name = re.split(r'[|\s/]', name)[0]
    return re.sub(r'[^\w\-. ]', '_', name)


def get_save_path(subdir: str) -> Path:
    """Get save path with date-based directory structure."""
    temp_dir = Path("temp_outputs")
    now = datetime.now()
    year = str(now.year)
    month = str(now.month).zfill(2)
    day = str(now.day).zfill(2)
    temp_dir_ = temp_dir / year / month / day / subdir
    temp_dir_.mkdir(parents=True, exist_ok=True)
    return temp_dir_


def toggle_ai_section(is_checked: bool):
    """Toggle visibility of AI configuration section."""
    return gr.update(visible=is_checked)


def create_zip_archive(files_to_zip: Dict[str, str], zip_filename: str) -> str:
    """Create ZIP archive with specified files."""
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zf:
        for src, arc in files_to_zip.items():
            if os.path.exists(src):
                zf.write(src, arcname=arc)
    return zip_filename


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
    
    try:
        sorted_residues = sorted(data['residue_rsa'].items(), key=lambda x: int(x[0]))
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
        
        try:
            sorted_residues = sorted(chain_data['residue_sasa'].items(), key=lambda x: int(x[0]))
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
    
    try:
        sorted_residues = sorted(data['residue_ss'].items(), key=lambda x: int(x[0]))
    except ValueError:
        sorted_residues = sorted(data['residue_ss'].items())
    
    for res_id, res_data in sorted_residues:
        result += f"  Residue {res_id} ({res_data['aa_seq']}): ss8: {res_data['ss8_seq']} ({res_data['ss8_name']}), ss3: {res_data['ss3_seq']}\n"
    
    result += f"aa_seq: {data['aa_sequence']}\n"
    result += f"ss8_seq: {data['ss8_sequence']}\n"
    result += f"ss3_seq: {data['ss3_sequence']}\n"
    
    return result

