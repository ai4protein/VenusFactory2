# file: @tool definitions for MAXIT, FASTA/UID, archive, PDB, metadata; logic in .tools_mcp

from typing import Optional

from langchain.tools import tool
from pydantic import BaseModel, Field

from .tools_mcp import (
    call_maxit_convert,
    call_read_fasta,
    call_extract_uids_from_fasta,
    call_uid_file_to_chunks,
    call_unzip,
    call_ungzip,
    call_pdb_chain_sequences,
    call_pdb_dir_to_fasta,
    call_pdb_is_apo,
    call_rcsb_metadata_uniprot_id,
    DEFAULT_BACKEND,
)


class MaxitConvertInput(BaseModel):
    file_path: str = Field(..., description="Path to input structure file (.pdb or .cif)")
    strategy: str = Field(
        ...,
        description="Conversion strategy: pdb2cif (PDB to CIF), cif2pdb (CIF to PDB), cif2mmcif (CIF to mmCIF)",
    )
    out_dir: Optional[str] = Field(None, description="Output directory; default is same as input file")
    backend: str = Field(
        default=DEFAULT_BACKEND,
        description="local: local path only; pjlab: upload result via SCP and return OSS URL",
    )


@tool("maxit_structure_convert", args_schema=MaxitConvertInput)
def maxit_structure_convert_tool(
    file_path: str,
    strategy: str,
    out_dir: Optional[str] = None,
    backend: str = None,
) -> str:
    """Convert protein structure file between PDB and CIF formats using MAXIT (RCSB). Requires maxit installed."""
    try:
        return call_maxit_convert(
            file_path=file_path,
            strategy=strategy,
            out_dir=out_dir,
            backend=backend or DEFAULT_BACKEND,
        )
    except Exception as e:
        return f"MAXIT conversion error: {str(e)}"


# ---------- FASTA / UID ----------
class ReadFastaInput(BaseModel):
    file_path: str = Field(..., description="Path to a multi-FASTA file")


@tool("read_fasta", args_schema=ReadFastaInput)
def read_fasta_tool(file_path: str) -> str:
    """Parse a multi-FASTA file and return headers and sequences as JSON."""
    try:
        return call_read_fasta(file_path)
    except Exception as e:
        return f"Read FASTA error: {str(e)}"


class ExtractUidsFromFastaInput(BaseModel):
    multi_fasta_file: str = Field(..., description="Path to multi-FASTA file (headers e.g. >sp|P12345|...)")
    uid_file: Optional[str] = Field(None, description="Optional output file path to write one UID per line")
    separator: str = Field(default="|", description="Header field separator")
    uid_index: int = Field(default=1, description="0-based index of ID field in header after split by separator")


@tool("extract_uids_from_fasta", args_schema=ExtractUidsFromFastaInput)
def extract_uids_from_fasta_tool(
    multi_fasta_file: str,
    uid_file: Optional[str] = None,
    separator: str = "|",
    uid_index: int = 1,
) -> str:
    """Extract IDs from FASTA headers (e.g. UniProt IDs). Optionally write to a UID list file."""
    try:
        return call_extract_uids_from_fasta(
            multi_fasta_file=multi_fasta_file,
            uid_file=uid_file,
            separator=separator,
            uid_index=uid_index,
        )
    except Exception as e:
        return f"Extract UIDs from FASTA error: {str(e)}"


class UidFileToChunksInput(BaseModel):
    uid_file: str = Field(..., description="Path to file with one ID per line")
    chunk_dir: Optional[str] = Field(None, description="Output directory for chunk files; default: <uid_file_dir>/chunks")
    chunk_size: int = Field(default=10000, description="Number of IDs per chunk file")


@tool("uid_file_to_chunks", args_schema=UidFileToChunksInput)
def uid_file_to_chunks_tool(
    uid_file: str,
    chunk_dir: Optional[str] = None,
    chunk_size: int = 10000,
) -> str:
    """Split a file of IDs (one per line) into multiple chunk files."""
    try:
        return call_uid_file_to_chunks(uid_file=uid_file, chunk_dir=chunk_dir, chunk_size=chunk_size)
    except Exception as e:
        return f"UID file to chunks error: {str(e)}"


# ---------- Archive ----------
class UnzipInput(BaseModel):
    zip_path: str = Field(..., description="Path to .zip file")
    save_folder: str = Field(..., description="Directory to extract into")


@tool("unzip_archive", args_schema=UnzipInput)
def unzip_archive_tool(zip_path: str, save_folder: str) -> str:
    """Extract a zip archive to a folder."""
    try:
        return call_unzip(zip_path, save_folder)
    except Exception as e:
        return f"Unzip error: {str(e)}"


class UngzipInput(BaseModel):
    gz_path: str = Field(..., description="Path to .gz file")
    out_dir: str = Field(..., description="Output directory for decompressed file")


@tool("ungzip_file", args_schema=UngzipInput)
def ungzip_file_tool(gz_path: str, out_dir: str) -> str:
    """Decompress a .gz file into the given directory."""
    try:
        return call_ungzip(gz_path, out_dir)
    except Exception as e:
        return f"Ungzip error: {str(e)}"


# ---------- PDB ----------
class PdbChainSequencesInput(BaseModel):
    pdb_file: str = Field(..., description="Path to PDB file")


@tool("pdb_chain_sequences", args_schema=PdbChainSequencesInput)
def pdb_chain_sequences_tool(pdb_file: str) -> str:
    """Extract chain IDs and sequences from a PDB file (first model)."""
    try:
        return call_pdb_chain_sequences(pdb_file)
    except Exception as e:
        return f"PDB chain sequences error: {str(e)}"


class PdbDirToFastaInput(BaseModel):
    pdb_dir: str = Field(..., description="Directory containing PDB files")
    out_fasta_path: str = Field(..., description="Output FASTA file path")
    use_chain_a_only: bool = Field(default=True, description="If True use chain A only per PDB; else first chain")


@tool("pdb_dir_to_fasta", args_schema=PdbDirToFastaInput)
def pdb_dir_to_fasta_tool(
    pdb_dir: str,
    out_fasta_path: str,
    use_chain_a_only: bool = True,
) -> str:
    """Write one FASTA file from all PDB files in a directory (one sequence per PDB)."""
    try:
        return call_pdb_dir_to_fasta(
            pdb_dir=pdb_dir,
            out_fasta_path=out_fasta_path,
            use_chain_a_only=use_chain_a_only,
        )
    except Exception as e:
        return f"PDB dir to FASTA error: {str(e)}"


class PdbIsApoInput(BaseModel):
    pdb_path: str = Field(..., description="Path to PDB file")


@tool("check_pdb_apo", args_schema=PdbIsApoInput)
def check_pdb_apo_tool(pdb_path: str) -> str:
    """Check if a PDB file is apo (no hetero/ligand residues)."""
    try:
        return call_pdb_is_apo(pdb_path)
    except Exception as e:
        return f"PDB is_apo check error: {str(e)}"


# ---------- Metadata ----------
class RcsbMetadataUniprotIdInput(BaseModel):
    meta_data_file: str = Field(..., description="Path to RCSB PDB metadata JSON file")


@tool("extract_uniprot_id_from_rcsb_metadata", args_schema=RcsbMetadataUniprotIdInput)
def extract_uniprot_id_from_rcsb_metadata_tool(meta_data_file: str) -> str:
    """Extract UniProt ID(s) from an RCSB PDB metadata JSON file."""
    try:
        return call_rcsb_metadata_uniprot_id(meta_data_file)
    except Exception as e:
        return f"RCSB metadata UniProt ID error: {str(e)}"
