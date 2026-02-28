"""
File converter MCP: MAXIT PDB/CIF conversion.
Logic in converter/maxit.py; optional OSS upload via tools.search.tools_mcp.
"""
import json
import os
from pathlib import Path
from typing import Optional

from tools.file.converter.maxit import convert as maxit_convert

# Backend: local (local path only) vs pjlab (upload via SCP, return URL)
try:
    from tools.search.tools_mcp import upload_file_to_oss_sync, DEFAULT_BACKEND
except ImportError:
    DEFAULT_BACKEND = "local"

    def upload_file_to_oss_sync(file_path: str, backend: str = None) -> Optional[str]:
        return None


STRATEGY_MAP = {
    "pdb2cif": (1, ".cif"),
    "cif2pdb": (2, ".pdb"),
    "cif2mmcif": (8, ".cif"),
}


def call_maxit_convert(
    file_path: str,
    strategy: str,
    out_dir: Optional[str] = None,
    backend: Optional[str] = None,
) -> str:
    """
    Convert structure file with MAXIT (PDB <-> CIF).
    strategy: pdb2cif, cif2pdb, cif2mmcif.
    Returns JSON with success, output path, and optional oss_url.
    """
    if strategy not in STRATEGY_MAP:
        return json.dumps({
            "success": False,
            "error": f"Unknown strategy: {strategy}. Use pdb2cif, cif2pdb, or cif2mmcif.",
        })
    if not file_path or not os.path.exists(file_path):
        return json.dumps({
            "success": False,
            "error": f"Input file not found: {file_path}",
        })
    backend = backend or DEFAULT_BACKEND
    try:
        maxit_o, postfix = STRATEGY_MAP[strategy]
        maxit_convert(file_path, maxit_o=maxit_o, out_dir=out_dir, postfix=postfix)
        # Output path: same as maxit.convert logic
        converted_basename = Path(file_path).stem + postfix
        if out_dir:
            output_path = os.path.join(out_dir, converted_basename)
        else:
            output_path = str(Path(file_path).with_suffix(postfix))
        if not os.path.exists(output_path):
            return json.dumps({
                "success": False,
                "error": f"Conversion may have failed; output not found: {output_path}",
            })
        oss_url = upload_file_to_oss_sync(output_path, backend=backend)
        return json.dumps({
            "success": True,
            "input_file": file_path,
            "output_file": output_path,
            "strategy": strategy,
            "oss_url": oss_url,
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
        })


# ---------- Processors (fasta, archive, pdb, metadata) ----------
from tools.file.processors import (
    read_multi_fasta,
    get_uids_from_fasta,
    make_uid_chunks,
    unzip_archive,
    ungzip_file,
    get_chain_sequences_from_pdb,
    get_seq_from_pdb_chain_a,
    get_seqs_from_pdb_dir,
    is_apo_pdb,
    get_uniprot_id_from_rcsb_metadata,
)


def call_read_fasta(file_path: str) -> str:
    """Parse multi-FASTA file; return JSON with headers and sequences."""
    if not file_path or not os.path.exists(file_path):
        return json.dumps({"success": False, "error": f"File not found: {file_path}"})
    try:
        sequences = read_multi_fasta(file_path)
        return json.dumps({
            "success": True,
            "file": file_path,
            "headers": list(sequences.keys()),
            "sequences": sequences,
            "count": len(sequences),
        }, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def call_extract_uids_from_fasta(
    multi_fasta_file: str,
    uid_file: Optional[str] = None,
    separator: str = "|",
    uid_index: int = 1,
) -> str:
    """Extract IDs from FASTA headers (e.g. UniProt IDs from >sp|P12345|...). Optionally write to uid_file."""
    if not multi_fasta_file or not os.path.exists(multi_fasta_file):
        return json.dumps({"success": False, "error": f"File not found: {multi_fasta_file}"})
    try:
        uids = get_uids_from_fasta(multi_fasta_file, separator=separator, uid_index=uid_index)
        if uid_file:
            os.makedirs(os.path.dirname(uid_file) or ".", exist_ok=True)
            with open(uid_file, "w") as f:
                f.write("\n".join(uids))
        return json.dumps({
            "success": True,
            "multi_fasta_file": multi_fasta_file,
            "uids": uids,
            "count": len(uids),
            "uid_file": uid_file,
        }, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def call_uid_file_to_chunks(
    uid_file: str,
    chunk_dir: Optional[str] = None,
    chunk_size: int = 10000,
) -> str:
    """Split a file of IDs (one per line) into chunk files."""
    if not uid_file or not os.path.exists(uid_file):
        return json.dumps({"success": False, "error": f"File not found: {uid_file}"})
    try:
        resolved_chunk_dir = chunk_dir or os.path.join(os.path.dirname(uid_file), "chunks")
        out_paths = make_uid_chunks(uid_file, chunk_dir=chunk_dir, chunk_size=chunk_size)
        return json.dumps({
            "success": True,
            "uid_file": uid_file,
            "chunk_dir": resolved_chunk_dir,
            "chunk_size": chunk_size,
            "chunk_files": out_paths,
            "count": len(out_paths),
        }, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def call_unzip(zip_path: str, save_folder: str) -> str:
    """Extract a zip archive to save_folder."""
    if not zip_path or not os.path.exists(zip_path):
        return json.dumps({"success": False, "error": f"File not found: {zip_path}"})
    try:
        unzip_archive(zip_path, save_folder)
        return json.dumps({
            "success": True,
            "zip_path": zip_path,
            "save_folder": save_folder,
        }, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def call_ungzip(gz_path: str, out_dir: str) -> str:
    """Decompress a .gz file into out_dir."""
    if not gz_path or not os.path.exists(gz_path):
        return json.dumps({"success": False, "error": f"File not found: {gz_path}"})
    try:
        out_path = ungzip_file(gz_path, out_dir)
        return json.dumps({
            "success": True,
            "gz_path": gz_path,
            "out_dir": out_dir,
            "output_file": out_path,
        }, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def call_pdb_chain_sequences(pdb_file: str) -> str:
    """Extract chain IDs and sequences from a PDB file (first model)."""
    if not pdb_file or not os.path.exists(pdb_file):
        return json.dumps({"success": False, "error": f"File not found: {pdb_file}"})
    try:
        seqs = get_chain_sequences_from_pdb(pdb_file)
        return json.dumps({
            "success": True,
            "pdb_file": pdb_file,
            "chains": seqs,
        }, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def call_pdb_dir_to_fasta(
    pdb_dir: str,
    out_fasta_path: str,
    use_chain_a_only: bool = True,
) -> str:
    """Write one FASTA from all PDB files in a directory (one sequence per PDB, chain A or first chain)."""
    if not pdb_dir or not os.path.isdir(pdb_dir):
        return json.dumps({"success": False, "error": f"Directory not found: {pdb_dir}"})
    try:
        get_seqs_from_pdb_dir(pdb_dir, out_fasta_path, use_chain_a_only=use_chain_a_only)
        return json.dumps({
            "success": True,
            "pdb_dir": pdb_dir,
            "out_fasta_path": out_fasta_path,
        }, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def call_pdb_is_apo(pdb_path: str) -> str:
    """Check if PDB has no hetero residues (apo structure)."""
    if not pdb_path or not os.path.exists(pdb_path):
        return json.dumps({"success": False, "error": f"File not found: {pdb_path}"})
    try:
        is_apo = is_apo_pdb(pdb_path)
        return json.dumps({
            "success": True,
            "pdb_file": pdb_path,
            "is_apo": is_apo,
        }, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def call_rcsb_metadata_uniprot_id(meta_data_file: str) -> str:
    """Get UniProt ID from RCSB PDB metadata JSON file."""
    if not meta_data_file or not os.path.exists(meta_data_file):
        return json.dumps({"success": False, "error": f"File not found: {meta_data_file}"})
    try:
        uid = get_uniprot_id_from_rcsb_metadata(meta_data_file)
        return json.dumps({
            "success": True,
            "meta_data_file": meta_data_file,
            "uniprot_id": uid,
        }, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
