"""File handling utilities for parsing and processing protein files."""

import os
import re
import time
from pathlib import Path
from typing import Any, Tuple, Dict
import gradio as gr

from .common_utils import get_save_path, sanitize_filename

def extract_sequence_from_pdb(pdb_content: str) -> str:
    """
    Extract FASTA sequence from PDB content.
    
    Args:
        pdb_content: PDB file content as string
    
    Returns:
        Extracted amino acid sequence as string
    """
    aa_map = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }
    
    sequence = []
    seen_residues = set()
    chain = None
    
    for line in pdb_content.strip().split('\n'):
        if line.startswith("ATOM"):
            chain_id = line[21]
            if chain is None:
                chain = chain_id
            if chain_id != chain:
                break
            
            res_id = (chain_id, int(line[22:26]))
            if res_id not in seen_residues:
                res_name = line[17:20].strip()
                if res_name in aa_map:
                    sequence.append(aa_map[res_name])
                    seen_residues.add(res_id)
    
    return "".join(sequence)


def parse_fasta_paste_content(fasta_content: str) -> Tuple:
    """Parse FASTA content from paste input."""
    if not fasta_content or not fasta_content.strip():
        return "No file selected", gr.update(choices=["Sequence_1"], value="Sequence_1", visible=False), {}, "Sequence_1", ""
   
    try:
        sequences = {}
        current_header = None
        current_sequence = ""
        sequence_counter = 1
       
        for line in fasta_content.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('>'):
                if current_header is not None and current_sequence:
                    sequences[current_header] = current_sequence
                
                current_header = line[1:].strip()
                current_sequence = ""
            else:
                sequence_data = ''.join(c.upper() for c in line if c.isalpha())
                
                if current_header is None:
                    current_header = f"Sequence_{sequence_counter}"
                    sequence_counter += 1
                
                current_sequence += sequence_data

        if current_header is not None and current_sequence:
            sequences[current_header] = current_sequence
       
        if not sequences:
            return "No valid protein sequences found in FASTA content", gr.update(choices=["Sequence_1"], value="Sequence_1", visible=False), {}, "Sequence_1", ""
        
        fasta_lines = []
        for header, sequence in sequences.items():
            fasta_lines.append(f">{header}")
            fasta_lines.append(sequence)
        modify_fasta_content = "\n".join(fasta_lines)
       
        sequence_choices = list(sequences.keys())
        default_sequence = sequence_choices[0]
        display_sequence = sequences[default_sequence]
        selector_visible = len(sequence_choices) > 1
        
        timestamp = str(int(time.time()))
        sequence_dir = get_save_path("Upload_Data", "Upload_Fasta")
        temp_fasta_path = os.path.join(sequence_dir, f"{sanitize_filename(default_sequence)}_{timestamp}.fasta")
        save_selected_sequence_fasta(modify_fasta_content, default_sequence, temp_fasta_path)
        return display_sequence, gr.update(choices=sequence_choices, value=default_sequence, visible=selector_visible), sequences, default_sequence, temp_fasta_path, modify_fasta_content
       
    except Exception as e:
        print(f"Error in parse_fasta_paste_content: {str(e)}")
        return f"Error parsing FASTA content: {str(e)}", gr.update(choices=["Sequence_1"], value="Sequence_1", visible=False), {}, "Sequence_1", "", ""


def save_selected_sequence_fasta(original_fasta_content: str, selected_sequence: str, output_path: str):
    """Save selected sequence from FASTA content to file."""
    sequences = {}
    current_header = None
    current_sequence = ""
    sequence_counter = 1
   
    for line in original_fasta_content.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('>'):
            if current_header is not None and current_sequence:
                sequences[current_header] = current_sequence
            
            current_header = line[1:].strip()
            current_sequence = ""
        else:
            sequence_data = ''.join(c.upper() for c in line if c.isalpha())
            
            if current_header is None:
                current_header = f"Sequence_{sequence_counter}"
                sequence_counter += 1
            
            current_sequence += sequence_data

    if current_header is not None and current_sequence:
        sequences[current_header] = current_sequence
   
    if not sequences or selected_sequence not in sequences:
        print(f"Error: Sequence '{selected_sequence}' not found in parsed sequences")
        return

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f">{selected_sequence}\n")
            f.write(sequences[selected_sequence])
    except Exception as e:
        print(f"Error saving file: {str(e)}")


def parse_pdb_paste_content(pdb_content: str) -> Tuple:
    """Parse PDB content from paste input."""
    if not pdb_content.strip():
        return "No file selected", gr.update(choices=["A"], value="A", visible=False), {}, "A", ""
    
    try:
        chains = {}
        current_chain = None
        sequence = ""
        
        for line in pdb_content.strip().split('\n'):
            if line.startswith('ATOM'):
                chain_id = line[21:22].strip()
                if chain_id == "":
                    chain_id = "A"
                
                if current_chain != chain_id:
                    if current_chain is not None and sequence:
                        chains[current_chain] = sequence
                    current_chain = chain_id
                    sequence = ""
                
                res_name = line[17:20].strip()
                if res_name in ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR']:
                    aa_map = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 
                             'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 
                             'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 
                             'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}
                    
                    res_num = int(line[22:26].strip())
                    if len(sequence) < res_num:
                        sequence += aa_map[res_name]
        
        if current_chain is not None and sequence:
            chains[current_chain] = sequence
        
        if not chains:
            return "No valid protein chains found in PDB content", gr.update(choices=["A"], value="A", visible=False), {}, "A", ""
        
        chain_choices = list(chains.keys())
        default_chain = chain_choices[0]
        display_sequence = chains[default_chain]
        selector_visible = len(chain_choices) > 1
        timestamp = str(int(time.time()))
        sequence_dir = get_save_path("Upload_Data", "Upload_PDB")
        temp_pdb_path = os.path.join(sequence_dir, f"paste_content_chain_{default_chain}_{timestamp}.pdb")
        save_selected_chain_pdb(pdb_content, default_chain, temp_pdb_path)
        return display_sequence, gr.update(choices=chain_choices, value=default_chain, visible=selector_visible), chains, default_chain, temp_pdb_path
        
    except Exception as e:
        return f"Error parsing PDB content: {str(e)}", gr.update(choices=["A"], value="A", visible=False), {}, "A", ""


def save_selected_chain_pdb(original_pdb_content: str, selected_chain: str, output_path: str):
    """Save selected chain from PDB content to file."""
    new_pdb_lines = []
    atom_counter = 1
    
    for line in original_pdb_content.strip().split('\n'):
        if line.startswith('ATOM'):
            chain_id = line[21:22].strip()
            if chain_id == "":
                chain_id = "A"
            
            if chain_id == selected_chain:
                new_line = line[:21] + 'A' + line[22:]
                new_line = f"ATOM  {atom_counter:5d}" + new_line[11:]
                new_pdb_lines.append(new_line)
                atom_counter += 1
        elif not line.startswith('ATOM'):
            new_pdb_lines.append(line)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(new_pdb_lines))


def extract_first_sequence_from_fasta_file(fasta_file_path: str) -> str:
    """Extract first sequence from FASTA file and save as new file.
    
    Uses BioPython to read FASTA and extracts the first sequence.
    Reuses save_selected_sequence_fasta() for actual sequence extraction.
    
    Args:
        fasta_file_path: Path to original FASTA file
        
    Returns:
        Path to processed FASTA file with only first sequence
    """
    try:
        from Bio import SeqIO
        
        # Read FASTA file content
        with open(fasta_file_path, 'r') as f:
            fasta_content = f.read()
        
        # Use BioPython to parse sequences
        sequences = list(SeqIO.parse(fasta_file_path, "fasta"))
        
        if not sequences:
            return fasta_file_path  # No sequences found
        
        # If only one sequence, return original file
        if len(sequences) == 1:
            return fasta_file_path
        
        # Get first sequence header
        first_seq_id = sequences[0].id
        
        # Create output path
        output_dir = get_save_path("Upload_Data", "Processed_Data")
        base_name = os.path.splitext(os.path.basename(fasta_file_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_first_seq.fasta")
        
        # Save selected sequence using existing function
        save_selected_sequence_fasta(fasta_content, first_seq_id, output_path)
        
        print(f"✓ Extracted first sequence '{first_seq_id}' from FASTA: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Warning: Could not process FASTA sequences: {e}")
        return fasta_file_path  # Return original on error


def extract_first_chain_from_pdb_file(pdb_file_path: str) -> str:
    """Extract first chain from PDB file and save as new file.
    
    Uses BioPython to read PDB structure and extracts the first chain.
    Reuses save_selected_chain_pdb() for actual chain extraction.
    
    Args:
        pdb_file_path: Path to original PDB file
        
    Returns:
        Path to processed PDB file with only first chain (renamed to chain A)
    """
    try:
        from Bio.PDB import PDBParser
        
        # Read PDB file content
        with open(pdb_file_path, 'r') as f:
            pdb_content = f.read()
        
        # Use BioPython to parse structure and get chains
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_file_path)
        
        # Get all chain IDs
        chain_ids = []
        for model in structure:
            for chain in model:
                chain_ids.append(chain.id)
            break  # Only process first model
        
        if not chain_ids:
            return pdb_file_path  # No chains found
        
        # If only one chain, return original file
        if len(chain_ids) == 1:
            return pdb_file_path
        
        # Get first chain
        first_chain = chain_ids[0]
        
        # Create output path
        output_dir = get_save_path("Upload_Data", "Processed_Data")
        base_name = os.path.splitext(os.path.basename(pdb_file_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_chain_{first_chain}.pdb")
        
        # Save selected chain using existing function
        save_selected_chain_pdb(pdb_content, first_chain, output_path)
        
        print(f"✓ Extracted first chain '{first_chain}' from PDB: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Warning: Could not process PDB chains: {e}")
        return pdb_file_path  # Return original on error


def handle_paste_chain_selection(selected_chain: str, chains_dict: Dict, original_pdb_content: str) -> Tuple[str, str]:
    """Handle chain selection from pasted PDB content."""
    if not chains_dict or selected_chain not in chains_dict:
        return "No file selected", ""
    
    if not original_pdb_content or original_pdb_content == "No file selected":
        return "No file selected", ""
    
    try:
        timestamp = str(int(time.time()))
        sequence_dir = get_save_path("Upload_Data", "Upload_PDB_Single")
        temp_pdb_path = os.path.join(sequence_dir, f"{selected_chain}_{timestamp}.pdb")
        save_selected_chain_pdb(original_pdb_content, selected_chain, temp_pdb_path)
        
        return chains_dict[selected_chain], temp_pdb_path
        
    except Exception as e:
        return f"Error processing chain selection: {str(e)}", ""


def handle_paste_sequence_selection(selected_sequence: str, sequences_dict: Dict, original_fasta_content: str) -> Tuple[str, str]:
    """Handle sequence selection from pasted FASTA content."""
    if not sequences_dict or selected_sequence not in sequences_dict:
        return "No file selected", ""
    
    if not original_fasta_content or original_fasta_content == "No file selected":
        return "No file selected", ""
    
    try:
        timestamp = str(int(time.time()))
        sequence_dir = get_save_path("Upload_Data", "Upload_Fasta")
        temp_pdb_path = os.path.join(sequence_dir, f"{selected_sequence}_{timestamp}.fasta")
        save_selected_sequence_fasta(original_fasta_content, selected_sequence, temp_pdb_path)
        
        return sequences_dict[selected_sequence], temp_pdb_path
        
    except Exception as e:
        return f"Error processing chain selection: {str(e)}", ""


def handle_pdb_chain_change(selected_chain: str, chains_dict: Dict, original_file_path: str) -> Tuple[str, Any]:
    """Handle PDB chain change from uploaded file."""
    if not chains_dict or selected_chain not in chains_dict:
        return "No file selected", ""
    
    if not original_file_path or original_file_path == "No file selected" or not os.path.exists(original_file_path):
        return "No file selected", ""
        
    try:
        with open(original_file_path, 'r') as f:
            pdb_content = f.read()
        
        new_pdb_lines = []
        atom_counter = 1
        
        for line in pdb_content.strip().split('\n'):
            if line.startswith('ATOM'):
                chain_id = line[21:22].strip()
                if chain_id == "":
                    chain_id = "A"
                
                if chain_id == selected_chain:
                    new_line = line[:21] + 'A' + line[22:]
                    new_line = f"ATOM  {atom_counter:5d}" + new_line[11:]
                    new_pdb_lines.append(new_line)
                    atom_counter += 1
            elif not line.startswith('ATOM'):
                new_pdb_lines.append(line)
        
        dir_path = os.path.dirname(original_file_path)
        base_name, extension = os.path.splitext(os.path.basename(original_file_path))
        new_filename = f"{base_name}_A{extension}"
        new_pdb_path = os.path.join(dir_path, new_filename)
        
        with open(new_pdb_path, 'w') as f:
            f.write('\n'.join(new_pdb_lines))
        
        return chains_dict[selected_chain], gr.update(value=new_pdb_path)
        
    except Exception as e:
        return f"Error processing chain selection: {str(e)}", ""


def handle_fasta_sequence_change(selected_sequence: str, sequences_dict: Dict, original_fasta_path: str) -> Tuple[str, str]:
    """Handle FASTA sequence change from uploaded file."""
    if not sequences_dict or selected_sequence not in sequences_dict:
        return "No file selected", ""
    
    if not original_fasta_path or original_fasta_path == "No file selected" or not os.path.exists(original_fasta_path):
        return "No file selected", ""
    
    try:
        with open(original_fasta_path, 'r') as f:
            lines = f.read()

        new_fasta_lines = []
        new_fasta_lines.append(">" + selected_sequence)
        new_fasta_lines.append(sequences_dict[selected_sequence])
        
        dir_path = os.path.dirname(original_fasta_path)
        base_name, extension = os.path.splitext(os.path.basename(original_fasta_path))
        new_filename = f"{base_name}_1{extension}"
        new_fasta_path = os.path.join(dir_path, new_filename)
        with open(new_fasta_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_fasta_lines))
        
        return sequences_dict[selected_sequence], new_fasta_path

    except Exception as e:
        return f"Error processing sequence selection: {str(e)}", ""


def process_pdb_file_upload(file_path: str) -> Tuple:
    """Process uploaded PDB file."""
    if not file_path:
        return "No file selected", gr.update(choices=["A"], value="A", visible=False), {}, "A", "", ""
    try:
        with open(file_path, 'r') as f:
            pdb_content = f.read()
        sequence, chain_update, chains_dict, default_chain, _ = parse_pdb_paste_content(pdb_content)
        return sequence, chain_update, chains_dict, default_chain, file_path, file_path
    except Exception as e:
        return f"Error reading PDB file: {str(e)}", gr.update(choices=["A"], value="A", visible=False), {}, "A", "", ""


def process_fasta_file_upload(file_path: str) -> Tuple:
    """Process uploaded FASTA file."""
    if not file_path:
        return "No file selected", gr.update(choices=["Sequence_1"], value="Sequence_1", visible=False), {}, "Sequence_1", "", ""
    try:
        with open(file_path, 'r') as f:
            fasta_content = f.read()
        sequence, selector_update, sequences_dict, default_sequence, file_path, modify_fasta_content = parse_fasta_paste_content(fasta_content)
        return sequence, selector_update, sequences_dict, default_sequence, file_path, file_path
    except Exception as e:
        return f"Error reading FASTA file: {str(e)}", gr.update(choices=["Sequence_1"], value="Sequence_1", visible=False), {}, "Sequence_1", "", ""


def handle_file_upload(file_obj: Any) -> Tuple:
    """Handle file upload for both FASTA and PDB files."""
    if not file_obj:
        return "No file selected", gr.update(choices=["Sequence_1"], value="Sequence_1", visible=False), {}, "Sequence_1", "", ""
    if isinstance(file_obj, str):
        file_path = file_obj
    else:
        file_path = file_obj.name
    if file_path.lower().endswith((".fasta", ".fa")):
        return process_fasta_file_upload(file_path)
    elif file_path.lower().endswith(".pdb"):
        return process_pdb_file_upload(file_path)
    else:
        return "No file selected", gr.update(choices=["Sequence_1"], value="Sequence_1", visible=False), {}, "Sequence_1", "", ""


def process_fasta_file(file_path: str) -> str:
    """Process FASTA file, keeping only the first sequence if multiple exist."""
    sequences = []
    current_seq = ""
    current_header = ""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_header and current_seq:
                    sequences.append((current_header, current_seq))
                current_header = line
                current_seq = ""
            else:
                current_seq += line

        if current_header and current_seq:
            sequences.append((current_header, current_seq))
    
    if len(sequences) <= 1:
        return file_path
    
    original_path = Path(file_path)
    timestamp = str(int(time.time()))
    fasta_dir = get_save_path("Upload_Data", "Upload_Fasta")

    new_file_path = fasta_dir / f"filtered_{original_path.name}_{timestamp}"
    
    with open(new_file_path, 'w', encoding='utf-8') as f:
        f.write(f"{sequences[0][0]}\n")
        seq = sequences[0][1]
        f.write(f"{seq}\n")
    
    return str(new_file_path)


def clear_paste_content_pdb() -> Tuple:
    """Clear pasted PDB content."""
    return "", "", gr.update(choices=["A"], value="A", visible=False), {}, "A", ""


def clear_paste_content_fasta() -> Tuple:
    """Clear pasted FASTA content."""
    return "No file selected", "No file selected", gr.update(choices=["Sequence_1"], value="Sequence_1", visible=False), {}, "Sequence_1", ""


def handle_sequence_change_unified(selected_chain: str, chains_dict: Dict, original_file_path: str, original_paste_content: str) -> Tuple[str, str]:
    """Unified handler for sequence/chain changes from both file uploads and paste."""
    if not original_file_path:
        return "No file selected", ""
    
    if original_file_path.endswith('.fasta'):
        if original_paste_content:
            return handle_paste_sequence_selection(selected_chain, chains_dict, original_paste_content)
        else:
            return handle_fasta_sequence_change(selected_chain, chains_dict, original_file_path)
    elif original_file_path.endswith('.pdb'):
        if original_paste_content:
            return handle_paste_chain_selection(selected_chain, chains_dict, original_paste_content)
        else:
            return handle_pdb_chain_change(selected_chain, chains_dict, original_file_path)
    else:
        return "No file selected", ""

