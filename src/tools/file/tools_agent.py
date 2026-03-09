# file: LangChain @tool definitions; each tool calls atomic ops in converter/ and processors/, returns status JSON.

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.tools import tool
from pydantic import BaseModel, Field

from .converter.maxit import convert as maxit_convert
from .processors import (
    read_multi_fasta,
    get_uids_from_fasta,
    make_uid_chunks,
    unzip_archive,
    ungzip_file,
    get_chain_sequences_from_pdb,
    get_seqs_from_pdb_dir,
    is_apo_pdb,
    get_uniprot_id_from_rcsb_metadata,
)


def _error_response(error_type: str, message: str, suggestion: Optional[str] = None) -> str:
    """Return JSON status: status error, error { type, message, suggestion }, file_info null."""
    out: Dict[str, Any] = {
        "status": "error",
        "error": {"type": error_type, "message": message},
        "file_info": None,
    }
    if suggestion:
        out["error"]["suggestion"] = suggestion
    return json.dumps(out, ensure_ascii=False)


def _success_response(
    data: Any = None,
    file_path: Optional[str] = None,
    file_name: Optional[str] = None,
    **extra: Any,
) -> str:
    """Return JSON status: status success, file_info when path given, data/content."""
    out: Dict[str, Any] = {"status": "success", "data": data}
    if file_path:
        path = Path(file_path)
        out["file_info"] = {
            "file_path": str(path.resolve()) if path.exists() else file_path,
            "file_name": file_name or path.name,
            "file_size": path.stat().st_size if path.exists() else 0,
        }
    else:
        out["file_info"] = None
    out.update(extra)
    return json.dumps(out, ensure_ascii=False, indent=2)


# ---------- MAXIT ----------
STRATEGY_MAP = {
    "pdb2cif": (1, ".cif"),
    "cif2pdb": (2, ".pdb"),
    "cif2mmcif": (8, ".cif"),
}


class MaxitConvertInput(BaseModel):
    file_path: str = Field(..., description="Path to input structure file (.pdb or .cif). Required.")
    strategy: str = Field(
        ...,
        description="Conversion strategy: 'pdb2cif', 'cif2pdb', or 'cif2mmcif'. Requires MAXIT (RCSB) installed.",
    )
    out_dir: Optional[str] = Field(default=None, description="Output directory. If omitted, output next to input.")


@tool("maxit_structure_convert", args_schema=MaxitConvertInput)
def maxit_structure_convert_tool(
    file_path: str,
    strategy: str,
    out_dir: Optional[str] = None,
) -> str:
    """Convert protein structure file between PDB and CIF using MAXIT (RCSB). Returns status JSON."""
    if strategy not in STRATEGY_MAP:
        return _error_response("ConversionError", f"Unknown strategy: {strategy}. Use pdb2cif, cif2pdb, cif2mmcif.")
    if not file_path or not os.path.exists(file_path):
        return _error_response("ConversionError", f"Input file not found: {file_path}")
    try:
        maxit_o, postfix = STRATEGY_MAP[strategy]
        maxit_convert(file_path, maxit_o=maxit_o, out_dir=out_dir, postfix=postfix)
        converted_basename = Path(file_path).stem + postfix
        output_path = os.path.join(out_dir, converted_basename) if out_dir else str(Path(file_path).with_suffix(postfix))
        if not os.path.exists(output_path):
            return _error_response("ConversionError", f"Output not found: {output_path}")
        return _success_response(data={"strategy": strategy, "output_file": output_path}, file_path=output_path)
    except Exception as e:
        return _error_response("ConversionError", str(e), suggestion="Check file path, strategy, and MAXIT installation.")


# ---------- FASTA / UID ----------
class ReadFastaInput(BaseModel):
    file_path: str = Field(..., description="Path to a multi-FASTA file (.fasta, .fa). Required.")


@tool("read_fasta", args_schema=ReadFastaInput)
def read_fasta_tool(file_path: str) -> str:
    """Parse a multi-FASTA file and return headers and sequences as status JSON."""
    if not file_path or not os.path.exists(file_path):
        return _error_response("ReadError", f"File not found: {file_path}")
    try:
        sequences = read_multi_fasta(file_path)
        return _success_response(
            data={"headers": list(sequences.keys()), "sequences": sequences, "count": len(sequences)},
            file_path=file_path,
        )
    except Exception as e:
        return _error_response("ReadError", str(e), suggestion="Check file path and FASTA format.")


class ExtractUidsFromFastaInput(BaseModel):
    multi_fasta_file: str = Field(..., description="Path to multi-FASTA file. Required.")
    uid_file: Optional[str] = Field(default=None, description="Optional output file path for one UID per line.")
    separator: str = Field(default="|", description="Field separator in FASTA header (e.g. '|' for UniProt).")
    uid_index: int = Field(default=1, ge=0, description="0-based index of ID field after split. Default 1 for >sp|P12345|...")


@tool("extract_uids_from_fasta", args_schema=ExtractUidsFromFastaInput)
def extract_uids_from_fasta_tool(
    multi_fasta_file: str,
    uid_file: Optional[str] = None,
    separator: str = "|",
    uid_index: int = 1,
) -> str:
    """Extract IDs from FASTA headers (e.g. UniProt IDs). Optionally write to a UID list file. Returns status JSON."""
    if not multi_fasta_file or not os.path.exists(multi_fasta_file):
        return _error_response("ExtractError", f"File not found: {multi_fasta_file}")
    try:
        uids = get_uids_from_fasta(multi_fasta_file, separator=separator, uid_index=uid_index)
        if uid_file:
            os.makedirs(os.path.dirname(uid_file) or ".", exist_ok=True)
            with open(uid_file, "w") as f:
                f.write("\n".join(uids))
        return _success_response(
            data={"uids": uids, "count": len(uids), "uid_file": uid_file},
            file_path=uid_file or multi_fasta_file,
        )
    except Exception as e:
        return _error_response("ExtractError", str(e), suggestion="Check FASTA path, separator, uid_index.")


class UidFileToChunksInput(BaseModel):
    uid_file: str = Field(..., description="Path to file with one ID per line. Required.")
    chunk_dir: Optional[str] = Field(default=None, description="Output directory for chunks. Default: <uid_file_dir>/chunks.")
    chunk_size: int = Field(default=10000, ge=1, description="IDs per chunk file. Default 10000.")


@tool("uid_file_to_chunks", args_schema=UidFileToChunksInput)
def uid_file_to_chunks_tool(
    uid_file: str,
    chunk_dir: Optional[str] = None,
    chunk_size: int = 10000,
) -> str:
    """Split a file of IDs (one per line) into multiple chunk files. Returns status JSON."""
    if not uid_file or not os.path.exists(uid_file):
        return _error_response("ChunkError", f"File not found: {uid_file}")
    try:
        out_paths = make_uid_chunks(uid_file, chunk_dir=chunk_dir, chunk_size=chunk_size)
        resolved_dir = chunk_dir or os.path.join(os.path.dirname(uid_file), "chunks")
        return _success_response(
            data={"chunk_dir": resolved_dir, "chunk_size": chunk_size, "chunk_files": out_paths, "count": len(out_paths)},
            file_path=out_paths[0] if out_paths else None,
        )
    except Exception as e:
        return _error_response("ChunkError", str(e), suggestion="Check uid_file path and chunk_size.")


# ---------- Archive ----------
class UnzipInput(BaseModel):
    zip_path: str = Field(..., description="Path to .zip archive. Required.")
    save_folder: str = Field(..., description="Directory to extract into. Created if missing. Required.")


@tool("unzip_archive", args_schema=UnzipInput)
def unzip_archive_tool(zip_path: str, save_folder: str) -> str:
    """Extract a zip archive to a folder. Returns status JSON."""
    if not zip_path or not os.path.exists(zip_path):
        return _error_response("UnzipError", f"File not found: {zip_path}")
    try:
        unzip_archive(zip_path, save_folder)
        return _success_response(
            data={"zip_path": zip_path, "save_folder": save_folder},
            file_path=save_folder,
        )
    except Exception as e:
        return _error_response("UnzipError", str(e), suggestion="Check zip_path and save_folder.")


class UngzipInput(BaseModel):
    gz_path: str = Field(..., description="Path to .gz file. Required.")
    out_dir: str = Field(..., description="Output directory for decompressed file. Required.")


@tool("ungzip_file", args_schema=UngzipInput)
def ungzip_file_tool(gz_path: str, out_dir: str) -> str:
    """Decompress a .gz file into the given directory. Returns status JSON."""
    if not gz_path or not os.path.exists(gz_path):
        return _error_response("UngzipError", f"File not found: {gz_path}")
    try:
        out_path = ungzip_file(gz_path, out_dir)
        return _success_response(
            data={"gz_path": gz_path, "out_dir": out_dir, "output_file": out_path},
            file_path=out_path,
        )
    except Exception as e:
        return _error_response("UngzipError", str(e), suggestion="Check gz_path and out_dir.")


# ---------- PDB ----------
class PdbChainSequencesInput(BaseModel):
    pdb_file: str = Field(..., description="Path to PDB structure file. Required.")


@tool("pdb_chain_sequences", args_schema=PdbChainSequencesInput)
def pdb_chain_sequences_tool(pdb_file: str) -> str:
    """Extract chain IDs and sequences from a PDB file (first model). Returns status JSON."""
    if not pdb_file or not os.path.exists(pdb_file):
        return _error_response("PDBError", f"File not found: {pdb_file}")
    try:
        chains = get_chain_sequences_from_pdb(pdb_file)
        return _success_response(data={"chains": chains}, file_path=pdb_file)
    except Exception as e:
        return _error_response("PDBError", str(e), suggestion="Check PDB file path and format.")


class PdbDirToFastaInput(BaseModel):
    pdb_dir: str = Field(..., description="Directory containing PDB files. Required.")
    out_fasta_path: str = Field(..., description="Output FASTA file path. Required.")
    use_chain_a_only: bool = Field(default=True, description="If True use chain A only; else first chain. Default True.")


@tool("pdb_dir_to_fasta", args_schema=PdbDirToFastaInput)
def pdb_dir_to_fasta_tool(
    pdb_dir: str,
    out_fasta_path: str,
    use_chain_a_only: bool = True,
) -> str:
    """Write one FASTA file from all PDB files in a directory (one sequence per PDB). Returns status JSON."""
    if not pdb_dir or not os.path.isdir(pdb_dir):
        return _error_response("PDBError", f"Directory not found: {pdb_dir}")
    try:
        get_seqs_from_pdb_dir(pdb_dir, out_fasta_path, use_chain_a_only=use_chain_a_only)
        return _success_response(
            data={"pdb_dir": pdb_dir, "out_fasta_path": out_fasta_path},
            file_path=out_fasta_path,
        )
    except Exception as e:
        return _error_response("PDBError", str(e), suggestion="Check pdb_dir and out_fasta_path.")


class PdbIsApoInput(BaseModel):
    pdb_path: str = Field(..., description="Path to PDB structure file. Required.")


@tool("check_pdb_apo", args_schema=PdbIsApoInput)
def check_pdb_apo_tool(pdb_path: str) -> str:
    """Check if a PDB file is apo (no hetero/ligand residues). Returns status JSON."""
    if not pdb_path or not os.path.exists(pdb_path):
        return _error_response("PDBError", f"File not found: {pdb_path}")
    try:
        is_apo = is_apo_pdb(pdb_path)
        return _success_response(data={"pdb_file": pdb_path, "is_apo": is_apo}, file_path=pdb_path)
    except Exception as e:
        return _error_response("PDBError", str(e), suggestion="Check PDB file path.")


# ---------- Metadata ----------
class RcsbMetadataUniprotIdInput(BaseModel):
    metadata_file: str = Field(..., description="Path to RCSB PDB metadata JSON file. Required.")


@tool("extract_uniprot_id_from_rcsb_metadata", args_schema=RcsbMetadataUniprotIdInput)
def extract_uniprot_id_from_rcsb_metadata_tool(metadata_file: str) -> str:
    """Extract UniProt ID from an RCSB PDB metadata JSON file. Returns status JSON."""
    if not metadata_file or not os.path.exists(metadata_file):
        return _error_response("MetadataError", f"File not found: {metadata_file}")
    try:
        uid = get_uniprot_id_from_rcsb_metadata(metadata_file)
        return _success_response(data={"metadata_file": metadata_file, "uniprot_id": uid}, file_path=metadata_file)
    except Exception as e:
        return _error_response(
            "MetadataError", str(e),
            suggestion="Check metadata_file path and JSON structure (data.entry.polymer_entities.uniprots.rcsb_id).",
        )

FILE_TOOLS = [
    maxit_structure_convert_tool,
    read_fasta_tool,
    extract_uids_from_fasta_tool,
    uid_file_to_chunks_tool,
    unzip_archive_tool,
    ungzip_file_tool,
    pdb_chain_sequences_tool,
    pdb_dir_to_fasta_tool,
    check_pdb_apo_tool,
    extract_uniprot_id_from_rcsb_metadata_tool,
]