"""FASTA and UID list processing (extracted from search.database.utils)."""
import os
from typing import List, Optional


def read_multi_fasta(file_path: str) -> dict:
    """
    Read a multi-FASTA file into a dictionary of header -> sequence.
    file_path: path to a fasta file.
    Returns: {header: sequence, ...}
    """
    sequences = {}
    current_sequence = ""
    header = None
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_sequence and header is not None:
                    sequences[header] = current_sequence
                    current_sequence = ""
                header = line
            else:
                current_sequence += line
        if current_sequence and header is not None:
            sequences[header] = current_sequence
    return sequences


def get_uids_from_fasta(multi_fasta_file: str, separator: str = "|", uid_index: int = 1) -> List[str]:
    """
    Extract IDs from FASTA headers. Assumes header format like >prefix|UID|... (e.g. UniProt).
    separator: field separator (default "|"); uid_index: 0-based index of the ID field (default 1).
    Returns list of UIDs.
    """
    uids = []
    with open(multi_fasta_file, "r") as f:
        for line in f:
            if line.startswith(">"):
                parts = line.strip().split(separator)
                if len(parts) > uid_index:
                    uids.append(parts[uid_index].strip())
    return uids


def make_uid_chunks(
    uid_file: str,
    chunk_dir: Optional[str] = None,
    chunk_size: int = 10000,
) -> List[str]:
    """
    Split a file of IDs (one per line) into chunk files.
    Returns list of output chunk file paths.
    """
    with open(uid_file, "r") as f:
        uids = [line.strip() for line in f if line.strip()]
    base_dir = os.path.dirname(uid_file)
    if chunk_dir is None:
        chunk_dir = os.path.join(base_dir, "chunks")
    os.makedirs(chunk_dir, exist_ok=True)
    chunk_name = os.path.basename(uid_file).rsplit(".", 1)[0]
    out_paths = []
    for i in range(0, len(uids), chunk_size):
        chunk_path = os.path.join(chunk_dir, f"{chunk_name}_{i // chunk_size}.txt")
        with open(chunk_path, "w") as f:
            f.write("\n".join(uids[i : i + chunk_size]))
        out_paths.append(chunk_path)
    return out_paths
