# search: @tool definitions for retrieval + data fetch; logic in .tools_mcp, .database, .source
#
# Query tools (return text, no file): interpro_lookup, uniprot_sequence_query, uniprot_meta_query,
#   ncbi_sequence_query, ncbi_meta_query, rcsb_entry_query, rcsb_structure_query, alphafold_structure_query
# Download tools (save to local/OSS): uniprot_sequence_download, pdb_structure_download, ncbi_sequence_download,
#   alphafold_structure_download

import json
import os
from typing import Optional

from langchain.tools import tool
from pydantic import BaseModel, Field
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

from web.utils.common_utils import get_save_path
from .database.foldseek import get_foldseek_sequences
from .database.uniprot import query_uniprot_seq, query_uniprot_meta
from .database.ncbi import query_ncbi_seq, query_ncbi_meta
from .database.rcsb import query_rcsb_entry, query_rcsb_structure
from .database.alphafold import query_alphafold_structure
from .source.literature import literature_search
from .source.web_search import web_search
from .source.dataset_search import dataset_search
from .tools_mcp import (
    upload_file_to_oss_sync,
    call_interpro_function_query,
    get_uniprot_sequence,
    download_pdb_structure_from_id,
    download_ncbi_sequence,
    download_alphafold_structure,
    DEFAULT_BACKEND,
)


# ---------- Schemas ----------
class InterProQueryInput(BaseModel):
    uniprot_id: str = Field(..., description="UniProt ID for protein function query")


class UniProtQueryInput(BaseModel):
    uniprot_id: str = Field(..., description="UniProt ID for protein sequence query")


class PDBSequenceExtractionInput(BaseModel):
    pdb_file: str = Field(..., description="Path to local PDB file")


class PDBStructureInput(BaseModel):
    pdb_id: str = Field(..., description="PDB ID for protein structure download")
    output_format: str = Field(default="pdb", description="Output format: pdb, mmcif")
    backend: str = Field(default=DEFAULT_BACKEND, description="local: local path only; pjlab: upload via SCP")


class FoldSeekSearchInput(BaseModel):
    pdb_file_path: str = Field(..., description="Path to PDB structure file")
    protect_start: int = Field(..., description="Start position of protected region")
    protect_end: int = Field(..., description="End position of protected region")
    backend: str = Field(default=DEFAULT_BACKEND, description="local: local path only; pjlab: upload via SCP")


class NCBISequenceInput(BaseModel):
    accession_id: str = Field(..., description="NCBI accession ID")
    output_format: str = Field(default="fasta", description="Output format: fasta, genbank")
    backend: str = Field(default=DEFAULT_BACKEND, description="local: local path only; pjlab: upload via SCP")


class RCSBEntryQueryInput(BaseModel):
    pdb_id: str = Field(..., description="PDB ID for RCSB entry metadata query")


class RCSBStructureQueryInput(BaseModel):
    pdb_id: str = Field(..., description="PDB ID for RCSB structure query")
    file_type: str = Field(default="pdb", description="Structure format: pdb, cif, pdb1, xml")


class NCBIQueryInput(BaseModel):
    ncbi_id: str = Field(..., description="NCBI accession ID")
    db: str = Field(default="protein", description="NCBI database: protein, nuccore")


class AlphaFoldStructureInput(BaseModel):
    uniprot_id: str = Field(..., description="UniProt ID for AlphaFold structure download")
    output_format: str = Field(default="pdb", description="Output format: pdb, mmcif")
    backend: str = Field(default=DEFAULT_BACKEND, description="local: local path only; pjlab: upload via SCP")


class LiteratureSearchInput(BaseModel):
    query: str = Field(..., description="Search query for academic literature")
    max_results: int = Field(5, description="Maximum number of results")
    source: str = Field("arxiv", description="Source: arxiv, pubmed, biorxiv, semantic_scholar, or all")


class DatasetSearchInput(BaseModel):
    query: str = Field(..., description="Search query for datasets")
    max_results: int = Field(5, description="Maximum number of results")
    source: str = Field("github", description="Source: github, hugging_face, or all")


class WebSearchInput(BaseModel):
    query: str = Field(..., description="Search query for web search")
    max_results: int = Field(5, description="Maximum number of results")
    source: str = Field("tavily", description="Source: duckduckgo, tavily, or all")


# ---------- Data tools ----------
@tool("interpro_lookup", args_schema=InterProQueryInput)
def interpro_lookup_tool(uniprot_id: str) -> str:
    """Look up InterPro domain/function annotations by UniProt ID."""
    try:
        return call_interpro_function_query(uniprot_id)
    except Exception as e:
        return f"InterPro query error: {str(e)}"


@tool("uniprot_sequence_query", args_schema=UniProtQueryInput)
def uniprot_sequence_query_tool(uniprot_id: str) -> str:
    """Query UniProt sequence by ID (returns FASTA text, no file download)."""
    try:
        return query_uniprot_seq(uniprot_id)
    except Exception as e:
        return f"UniProt sequence query error: {str(e)}"


@tool("uniprot_meta_query", args_schema=UniProtQueryInput)
def uniprot_meta_query_tool(uniprot_id: str) -> str:
    """Query UniProt entry metadata by ID (returns JSON text, no file download)."""
    try:
        return query_uniprot_meta(uniprot_id)
    except Exception as e:
        return f"UniProt meta query error: {str(e)}"


@tool("ncbi_sequence_query", args_schema=NCBIQueryInput)
def ncbi_sequence_query_tool(ncbi_id: str, db: str = "protein") -> str:
    """Query NCBI sequence by accession (returns FASTA text, no file download)."""
    try:
        return query_ncbi_seq(ncbi_id, db=db)
    except Exception as e:
        return f"NCBI sequence query error: {str(e)}"


@tool("ncbi_meta_query", args_schema=NCBIQueryInput)
def ncbi_meta_query_tool(ncbi_id: str, db: str = "protein") -> str:
    """Query NCBI entry metadata by accession (returns GenBank text, no file download)."""
    try:
        return query_ncbi_meta(ncbi_id, db=db)
    except Exception as e:
        return f"NCBI meta query error: {str(e)}"


@tool("rcsb_entry_query", args_schema=RCSBEntryQueryInput)
def rcsb_entry_query_tool(pdb_id: str) -> str:
    """Query RCSB PDB entry metadata by PDB ID (returns JSON text, no file download)."""
    try:
        return query_rcsb_entry(pdb_id)
    except Exception as e:
        return f"RCSB entry query error: {str(e)}"


@tool("rcsb_structure_query", args_schema=RCSBStructureQueryInput)
def rcsb_structure_query_tool(pdb_id: str, file_type: str = "pdb") -> str:
    """Query RCSB PDB structure content by PDB ID (returns structure text, no file download)."""
    try:
        return query_rcsb_structure(pdb_id, file_type=file_type)
    except Exception as e:
        return f"RCSB structure query error: {str(e)}"


@tool("alphafold_structure_query", args_schema=AlphaFoldStructureInput)
def alphafold_structure_query_tool(uniprot_id: str) -> str:
    """Query AlphaFold structure metadata by UniProt ID (returns JSON text, no file download)."""
    try:
        return query_alphafold_structure(uniprot_id)
    except Exception as e:
        return f"AlphaFold structure query error: {str(e)}"


@tool("uniprot_sequence_download", args_schema=UniProtQueryInput)
def uniprot_sequence_download_tool(uniprot_id: str) -> str:
    """Download protein sequence from UniProt by UniProt ID."""
    try:
        out = get_uniprot_sequence(uniprot_id)
        return json.dumps(out, ensure_ascii=False) if isinstance(out, dict) else out
    except Exception as e:
        return f"UniProt sequence download error: {str(e)}"


@tool("pdb_structure_download", args_schema=PDBStructureInput)
def pdb_structure_download_tool(pdb_id: str, output_format: str = "pdb", backend: str = None) -> str:
    """Download protein structure from RCSB PDB using PDB ID."""
    try:
        return download_pdb_structure_from_id(pdb_id, output_format, backend=backend or DEFAULT_BACKEND)
    except Exception as e:
        return f"PDB structure download error: {str(e)}"


@tool("pdb_sequence_extraction", args_schema=PDBSequenceExtractionInput)
def pdb_sequence_extraction_tool(pdb_file: str) -> str:
    """Extract protein sequence(s) from a local PDB file using Biopython."""
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("pdb_struct", pdb_file)
        sequences = []
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
                    sequences.append({"chain": chain.id, "sequence": "".join(residues)})
        if not sequences:
            return json.dumps({"success": False, "error": "No protein sequences found in PDB file."})
        return json.dumps({"success": True, "pdb_file": pdb_file, "sequences": sequences}, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


@tool("foldseek_search", args_schema=FoldSeekSearchInput)
def foldseek_search_tool(pdb_file_path: str, protect_start: int, protect_end: int, backend: str = None) -> str:
    """Search for protein structures using FoldSeek."""
    try:
        output_dir = get_save_path("FoldSeek", "Download_data")
        fasta_file, total_sequences = get_foldseek_sequences(
            pdb_file_path, protect_start, protect_end, output_dir=output_dir
        )
        fasta_file_oss_url = upload_file_to_oss_sync(str(fasta_file), backend=backend or DEFAULT_BACKEND)
        return json.dumps({
            "success": True,
            "fasta_file": str(fasta_file),
            "fasta_file_oss_url": str(fasta_file_oss_url) if fasta_file_oss_url else None,
            "total_sequences": total_sequences,
        })
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


@tool("ncbi_sequence_download", args_schema=NCBISequenceInput)
def ncbi_sequence_download_tool(accession_id: str, output_format: str = "fasta", backend: str = None) -> str:
    """Download protein or nucleotide sequences from NCBI database using accession ID."""
    try:
        return download_ncbi_sequence(accession_id, output_format, backend=backend or DEFAULT_BACKEND)
    except Exception as e:
        return f"NCBI sequence download error: {str(e)}"


@tool("alphafold_structure_download", args_schema=AlphaFoldStructureInput)
def alphafold_structure_download_tool(uniprot_id: str, output_format: str = "pdb", backend: str = None) -> str:
    """Download protein structures from AlphaFold database using UniProt ID."""
    try:
        return download_alphafold_structure(uniprot_id, output_format, backend=backend or DEFAULT_BACKEND)
    except Exception as e:
        return f"AlphaFold structure download error: {str(e)}"


# ---------- Search tools ----------
@tool("literature_search", args_schema=LiteratureSearchInput)
def literature_search_tool(query: str, max_results: int = 5, source: str = "pubmed") -> str:
    """Search for academic literature using arXiv and PubMed."""
    try:
        refs = literature_search(query, max_results=max_results, source=source)
        return json.dumps({"success": True, "references": refs}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


@tool("dataset_search", args_schema=DatasetSearchInput)
def dataset_search_tool(query: str, max_results: int = 5, source: str = "github") -> str:
    """Search for datasets using GitHub and Hugging Face."""
    try:
        datasets = dataset_search(query, max_results=max_results, source=source)
        return json.dumps({"success": True, "datasets": datasets}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


@tool("web_search", args_schema=WebSearchInput)
def web_search_tool(query: str, max_results: int = 5, source: str = "tavily") -> str:
    """Search the web using DuckDuckGo and Tavily."""
    try:
        results = web_search(query, max_results=max_results, source=source)
        return json.dumps({"success": True, "results": results}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
