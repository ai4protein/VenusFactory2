"""File handling utilities for parsing and processing protein files."""

import os
import re
import time
from pathlib import Path
from typing import Any, Tuple, Dict

from .common_utils import get_save_path, sanitize_filename

try:
    from mcp_tools.tools.protein_io import (
        extract_first_chain_from_pdb_file as extract_first_chain_from_pdb_file,
        extract_first_sequence_from_fasta_file as extract_first_sequence_from_fasta_file,
        save_selected_chain_pdb as save_selected_chain_pdb,
        save_selected_sequence_fasta as save_selected_sequence_fasta,
    )
except ModuleNotFoundError as exc:
    if exc.name not in {"tools", 'mcp_tools.tools.protein_io'}:
        raise
    from mcp_tools.tools.protein_io import (
        extract_first_chain_from_pdb_file as extract_first_chain_from_pdb_file,
        extract_first_sequence_from_fasta_file as extract_first_sequence_from_fasta_file,
        save_selected_chain_pdb as save_selected_chain_pdb,
        save_selected_sequence_fasta as save_selected_sequence_fasta,
    )


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


def validate_and_normalize_fasta_content(fasta_content: str) -> str:
    """Validate FASTA text and normalize it into clean single-line sequences.

    Rules:
    - First non-empty line must start with '>'.
    - Every record must have a non-empty header and non-empty sequence.
    - Sequence keeps alphabetic characters only and is uppercased.
    """
    if not fasta_content or not fasta_content.strip():
        raise ValueError("FASTA content is empty.")

    raw_lines = [line.strip() for line in fasta_content.splitlines() if line.strip()]
    if not raw_lines:
        raise ValueError("FASTA content is empty.")
    if not raw_lines[0].startswith(">"):
        raise ValueError("FASTA must include header line starting with >")

    normalized_lines = []
    current_header = None
    current_sequence_parts = []

    for line in raw_lines:
        if line.startswith(">"):
            if len(line) <= 1:
                raise ValueError("FASTA header cannot be empty.")
            if current_header is not None:
                sequence = "".join(current_sequence_parts)
                if not sequence:
                    raise ValueError(f"Sequence under header '{current_header[1:]}' is empty.")
                normalized_lines.append(current_header)
                normalized_lines.append(sequence)
            current_header = line
            current_sequence_parts = []
            continue

        if current_header is None:
            raise ValueError("FASTA must include header line starting with >")

        sequence_chunk = "".join(char for char in line.upper() if char.isalpha())
        current_sequence_parts.append(sequence_chunk)

    if current_header is None:
        raise ValueError("FASTA must include header line starting with >")

    sequence = "".join(current_sequence_parts)
    if not sequence:
        raise ValueError(f"Sequence under header '{current_header[1:]}' is empty.")
    normalized_lines.append(current_header)
    normalized_lines.append(sequence)

    return "\n".join(normalized_lines) + "\n"
