"""Protein file selection without importing UI or model runtimes."""

import os

from mcp_tools.common.common_utils import get_save_path


def save_selected_sequence_fasta(original_fasta_content: str, selected_sequence: str, output_path: str):
    """Save selected sequence from FASTA content to file."""
    sequences = {}
    current_header = None
    current_sequence = ""
    sequence_counter = 1

    for line in original_fasta_content.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        if line.startswith(">"):
            if current_header is not None and current_sequence:
                sequences[current_header] = current_sequence

            current_header = line[1:].strip()
            current_sequence = ""
        else:
            sequence_data = "".join(c.upper() for c in line if c.isalpha())

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
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f">{selected_sequence}\n")
            f.write(sequences[selected_sequence])
    except Exception as e:
        print(f"Error saving file: {str(e)}")


def save_selected_chain_pdb(original_pdb_content: str, selected_chain: str, output_path: str):
    """Save selected chain from PDB content to file."""
    new_pdb_lines = []
    atom_counter = 1

    for line in original_pdb_content.strip().split("\n"):
        if line.startswith("ATOM"):
            chain_id = line[21:22].strip()
            if chain_id == "":
                chain_id = "A"

            if chain_id == selected_chain:
                new_line = line[:21] + "A" + line[22:]
                new_line = f"ATOM  {atom_counter:5d}" + new_line[11:]
                new_pdb_lines.append(new_line)
                atom_counter += 1
        elif not line.startswith("ATOM"):
            new_pdb_lines.append(line)

    with open(output_path, "w") as f:
        f.write("\n".join(new_pdb_lines))


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
        with open(fasta_file_path) as f:
            fasta_content = f.read()

        # Use BioPython to parse sequences
        sequences = list(SeqIO.parse(fasta_file_path, "fasta"))

        if not sequences:
            return fasta_file_path  # No sequences found

        # If only one sequence, return original file
        if len(sequences) == 1:
            return fasta_file_path

        # Get first sequence header
        first_seq_id = sequences[0].description

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
        return fasta_file_path


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
        with open(pdb_file_path) as f:
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
        return pdb_file_path
