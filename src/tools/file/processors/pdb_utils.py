"""PDB parsing: sequences, chain A only, batch to FASTA, apo check (from search.database.utils and pdb_filter)."""
import os
from typing import List

from Bio.PDB import PDBParser, PPBuilder
from Bio.SeqUtils import seq1


def get_chain_sequences_from_pdb(pdb_file: str) -> List[dict]:
    """
    Extract chain id and sequence from PDB. Returns [{"chain": "A", "sequence": "..."}, ...].
    Uses first model only.
    """
    parser = PDBParser(QUIET=True)
    pdb_id = os.path.splitext(os.path.basename(pdb_file))[0][:4]
    structure = parser.get_structure(pdb_id, pdb_file)
    seqs = []
    for model in structure:
        for chain in model:
            residues = []
            for residue in chain:
                if residue.id[0] == " ":
                    try:
                        residues.append(seq1(residue.resname))
                    except Exception:
                        pass
            if residues:
                seqs.append({"chain": chain.id, "sequence": "".join(residues)})
        if seqs:
            break
    return seqs


def get_seq_from_pdb_chain_a(pdb_file: str) -> str:
    """Get sequence of chain A only from PDB (PPBuilder)."""
    parser = PDBParser(QUIET=True)
    name = os.path.splitext(os.path.basename(pdb_file))[0][:4]
    structure = parser.get_structure(name, pdb_file)
    ppb = PPBuilder()
    chain = structure[0]["A"]
    seq = ""
    for pp in ppb.build_peptides(chain):
        seq += pp.get_sequence()
    return seq


def get_seqs_from_pdb_dir(pdb_dir: str, out_fasta_path: str, use_chain_a_only: bool = True) -> str:
    """
    Write one FASTA file from all PDB files in pdb_dir.
    If use_chain_a_only True, uses chain A sequence per PDB; else uses first chain per model.
    Returns out_fasta_path.
    """
    from tqdm import tqdm

    pdbs = [f for f in os.listdir(pdb_dir) if f.lower().endswith(".pdb") or f.lower().endswith(".cif")]
    os.makedirs(os.path.dirname(out_fasta_path) or ".", exist_ok=True)
    with open(out_fasta_path, "w") as f:
        for pdb in tqdm(pdbs, desc="PDB to FASTA"):
            path = os.path.join(pdb_dir, pdb)
            try:
                if use_chain_a_only:
                    seq = get_seq_from_pdb_chain_a(path)
                else:
                    chains = get_chain_sequences_from_pdb(path)
                    seq = chains[0]["sequence"] if chains else ""
                if seq:
                    f.write(f">{pdb}\n{seq}\n")
            except Exception:
                continue
    return out_fasta_path


def is_apo_pdb(pdb_path: str) -> bool:
    """Return True if PDB has no hetero residues (apo structure)."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pdb", pdb_path)
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0].strip() != "":
                    return False
    return True
