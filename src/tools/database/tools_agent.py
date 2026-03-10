import json
import os
import sys
sys.path.append(os.getcwd())
from typing import List, Optional, Literal
from langchain.tools import tool
from pydantic import BaseModel, Field
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from src.web.utils.common_utils import get_save_path
from .alphafold import (
    download_alphafold_structure_by_uniprot_id,
    download_alphafold_metadata_by_uniprot_id,
)
from .brenda import (
    download_brenda_km_values_by_ec_number,
    download_brenda_reactions_by_ec_number,
    download_brenda_enzymes_by_substrate,
    download_brenda_compare_organisms_by_ec_number,
    download_brenda_environmental_parameters_by_ec_number,
    download_brenda_kinetic_data_by_ec_number,
    download_brenda_pathway_report,
)
from .chembl import (
    download_chembl_molecule_by_id,
    download_chembl_similarity_by_smiles,
    download_chembl_substructure_by_smiles,
    download_chembl_drug_by_id,
)
from .foldseek import download_foldseek_results_by_pdb_file
from .kegg import (
    download_kegg_info_by_database,
    download_kegg_list_by_database,
    download_kegg_find_by_database,
    download_kegg_entry_by_id,
    download_kegg_conv_by_id,
    download_kegg_link_by_id,
    download_kegg_ddi_by_id,
)
from .interpro import (
    download_interpro_metadata_by_id,
    download_interpro_annotations_by_uniprot_id,
    download_interpro_proteins_by_id,
    download_interpro_uniprot_list_by_id,
)
from .ncbi import (
    download_ncbi_sequence,
    download_ncbi_metadata,
    download_ncbi_blast,
    download_ncbi_clinvar_variants,
    download_ncbi_gene_by_id,
    download_ncbi_gene_by_symbol,
    download_ncbi_batch_lookup_by_symbols,
)
from .rcsb import (
    download_rcsb_entry_metadata_by_pdb_id,
    download_rcsb_structure_by_pdb_id,
)
from .uniprot import (
    download_uniprot_search_by_query,
    download_uniprot_retrieve_by_id,
    download_uniprot_mapping,
    download_uniprot_seq_by_id,
    download_uniprot_meta_by_id,
)


# AlphaFold Database Tools
class AlphaFoldStructureDownloadInput(BaseModel):
    uniprot_id: str = Field(..., description="UniProt accession for AlphaFold structure (e.g. P04040). Required.")
    out_dir: str = Field(..., description="Output directory for AlphaFold structure. Required.")
    format: str = Field(default="pdb", choices=["pdb", "cif"], description="Structure format: 'pdb' (default) or 'cif'.")
    version: str = Field(default="v6", choices=["v1", "v2", "v4", "v6"], description="AlphaFold DB version: v1, v2, v4, or v6. Default v6.")
    fragment: int = Field(default=1, ge=1, description="Fragment index for multi-fragment entries (1-based). Default 1.")

@tool("download_alphafold_structure_by_uniprot_id", args_schema=AlphaFoldStructureDownloadInput)
def download_alphafold_structure_by_uniprot_id_tool(
    uniprot_id: str,
    out_dir: str,
    format: str = "pdb",
    version: str = "v6",
    fragment: int = 1
) -> str:
    """Download AlphaFold structure by UniProt ID. Returns JSON: {success, file_path} where file_path is the path to the downloaded structure file."""
    try:
        return download_alphafold_structure_by_uniprot_id(uniprot_id, out_dir, format=format, version=version, fragment=fragment)
    except Exception as e:
        return f"Download AlphaFold structure error: {str(e)}"

class AlphaFoldMetadataDownloadInput(BaseModel):
    uniprot_id: str = Field(..., description="UniProt accession for AlphaFold metadata (e.g. P04040). Required.")
    out_dir: str = Field(..., description="Output directory for AlphaFold metadata. Required.")

@tool("download_alphafold_metadata_by_uniprot_id", args_schema=AlphaFoldMetadataDownloadInput)
def download_alphafold_metadata_by_uniprot_id_tool(
    uniprot_id: str,
    out_dir: str
) -> str:
    """Download AlphaFold metadata by UniProt ID. Returns JSON: {success, file_path} where file_path is the path to the downloaded metadata file."""
    try:
        return download_alphafold_metadata_by_uniprot_id(uniprot_id, out_dir)
    except Exception as e:
        return f"Download AlphaFold metadata error: {str(e)}"
    
# ---------- BRENDA Database Tools (download only) ----------
# All return JSON: {success, file_path[, error]}. Require BRENDA_EMAIL and BRENDA_PASSWORD in environment.
class BrendaDownloadKmInput(BaseModel):
    ec_number: str = Field(..., description="EC number. Required.")
    out_path: str = Field(..., description="Output file path (.json or .txt).")
    organism: str = Field(default="*", description="Organism filter or '*'.")
    substrate: str = Field(default="*", description="Substrate filter or '*'.")

@tool("download_brenda_km_values_by_ec_number", args_schema=BrendaDownloadKmInput)
def download_brenda_km_values_by_ec_number_tool(
    ec_number: str, out_path: str, organism: str = "*", substrate: str = "*"
) -> str:
    """Download BRENDA Km values by EC number to file. Returns JSON: {success, file_path}."""
    try:
        return download_brenda_km_values_by_ec_number(ec_number, out_path, organism=organism, substrate=substrate)
    except Exception as e:
        return f"Download BRENDA Km values by EC number error: {str(e)}"

# --- Download: Reactions by EC number ---
class BrendaDownloadReactionsInput(BaseModel):
    ec_number: str = Field(..., description="EC number. Required.")
    out_path: str = Field(..., description="Output file path (.json or .txt).")
    organism: str = Field(default="*", description="Organism filter or '*'.")

@tool("download_brenda_reactions_by_ec_number", args_schema=BrendaDownloadReactionsInput)
def download_brenda_reactions_by_ec_number_tool(ec_number: str, out_path: str, organism: str = "*") -> str:
    """Download BRENDA reactions by EC number to file. Returns JSON: {success, file_path}."""
    try:
        return download_brenda_reactions_by_ec_number(ec_number, out_path, organism=organism)
    except Exception as e:
        return f"Download BRENDA reactions by EC number error: {str(e)}"

# --- Download: Enzymes by substrate ---
class BrendaDownloadEnzymesBySubstrateInput(BaseModel):
    substrate: str = Field(..., description="Substrate name.")
    out_path: str = Field(..., description="Output JSON file path.")
    limit: int = Field(default=50, ge=1, le=500, description="Max enzymes. Default 50.")

@tool("download_brenda_enzymes_by_substrate", args_schema=BrendaDownloadEnzymesBySubstrateInput)
def download_brenda_enzymes_by_substrate_tool(substrate: str, out_path: str, limit: int = 50) -> str:
    """Download BRENDA enzyme-by-substrate search results to JSON file. Returns JSON: {success, file_path}."""
    try:
        return download_brenda_enzymes_by_substrate(substrate, out_path, limit=limit)
    except Exception as e:
        return f"Download BRENDA enzymes by substrate error: {str(e)}"

# --- Download: Compare organisms by EC number ---
class BrendaDownloadCompareOrganismsInput(BaseModel):
    ec_number: str = Field(..., description="EC number. Required.")
    organisms: List[str] = Field(..., description="List of organism names.")
    out_path: str = Field(..., description="Output JSON file path.")

@tool("download_brenda_compare_organisms_by_ec_number", args_schema=BrendaDownloadCompareOrganismsInput)
def download_brenda_compare_organisms_by_ec_number_tool(ec_number: str, organisms: List[str], out_path: str) -> str:
    """Download BRENDA organism comparison by EC number to JSON. Returns JSON: {success, file_path}."""
    try:
        return download_brenda_compare_organisms_by_ec_number(ec_number, organisms, out_path)
    except Exception as e:
        return f"Download BRENDA compare organisms by EC number error: {str(e)}"

# --- Download: Environmental parameters by EC number ---
class BrendaDownloadEnvironmentalParametersInput(BaseModel):
    ec_number: str = Field(..., description="EC number. Required.")
    out_path: str = Field(..., description="Output JSON file path.")

@tool("download_brenda_environmental_parameters_by_ec_number", args_schema=BrendaDownloadEnvironmentalParametersInput)
def download_brenda_environmental_parameters_by_ec_number_tool(ec_number: str, out_path: str) -> str:
    """Download BRENDA environmental parameters by EC number to JSON. Returns JSON: {success, file_path}."""
    try:
        return download_brenda_environmental_parameters_by_ec_number(ec_number, out_path)
    except Exception as e:
        return f"Download BRENDA environmental parameters by EC number error: {str(e)}"

# --- Download: Kinetic data by EC number ---
class BrendaDownloadKineticDataInput(BaseModel):
    ec_number: str = Field(..., description="EC number. Required.")
    out_path: str = Field(..., description="Output file path (e.g. .json or .csv).")
    format: str = Field(default="json", description="Export format: 'json' or 'csv'. Default json.")

@tool("download_brenda_kinetic_data_by_ec_number", args_schema=BrendaDownloadKineticDataInput)
def download_brenda_kinetic_data_by_ec_number_tool(
    ec_number: str, out_path: str, format: str = "json"
) -> str:
    """Download BRENDA kinetic data export by EC number to file. Returns JSON: {success, file_path}."""
    try:
        return download_brenda_kinetic_data_by_ec_number(ec_number, out_path, format=format)
    except Exception as e:
        return f"Download BRENDA kinetic data by EC number error: {str(e)}"

# --- Download: Pathway report (from pathway data) ---
class BrendaDownloadPathwayReportInput(BaseModel):
    pathway: dict = Field(..., description="Pathway data dict (e.g. from query_brenda_pathway_by_product).")
    out_path: str = Field(..., description="Output report file path (e.g. .txt).")

@tool("download_brenda_pathway_report", args_schema=BrendaDownloadPathwayReportInput)
def download_brenda_pathway_report_tool(pathway: dict, out_path: str) -> str:
    """Generate and save BRENDA pathway report from pathway data to file. Returns JSON: {success, file_path}."""
    try:
        return download_brenda_pathway_report(pathway, out_path)
    except Exception as e:
        return f"Download BRENDA pathway report error: {str(e)}"

# ---------- ChEMBL Database Tools ----------
# All return rich JSON: status, content/file_info, content_preview, biological_metadata, execution_context.
class ChemblMoleculeDownloadInput(BaseModel):
    mol_id: str = Field(..., description="ChEMBL molecule ID (e.g. CHEMBL25, CHEMBL100). Required.")
    out_path: str = Field(..., description="Output JSON file path. Required.")

@tool("download_chembl_molecule_by_id", args_schema=ChemblMoleculeDownloadInput)
def download_chembl_molecule_by_id_tool(mol_id: str, out_path: str) -> str:
    """Download ChEMBL molecule JSON by ChEMBL ID to file. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    try:
        return download_chembl_molecule_by_id(mol_id, out_path)
    except Exception as e:
        return f"Download ChEMBL molecule by ID error: {str(e)}"

class ChemblSimilarityDownloadInput(BaseModel):
    smiles: str = Field(..., description="SMILES string of the query molecule for Tanimoto similarity search. Required.")
    out_path: str = Field(..., description="Output JSON file path. Required.")
    threshold: int = Field(default=70, ge=0, le=100, description="Tanimoto similarity threshold 0–100. Default 70.")
    max_results: Optional[int] = Field(default=None, ge=1, le=5000, description="Max number of results to return (default 500, cap 5000). Omit for default.")

@tool("download_chembl_similarity_by_smiles", args_schema=ChemblSimilarityDownloadInput)
def download_chembl_similarity_by_smiles_tool(
    smiles: str,
    out_path: str,
    threshold: int = 70,
    max_results: Optional[int] = None,
) -> str:
    """Download ChEMBL similarity search results to JSON file. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    try:
        return download_chembl_similarity_by_smiles(smiles, out_path, threshold=threshold, max_results=max_results)
    except Exception as e:
        return f"Download ChEMBL similarity by SMILES error: {str(e)}"

class ChemblSubstructureDownloadInput(BaseModel):
    smiles: str = Field(..., description="SMILES substructure to search for in ChEMBL molecules. Required.")
    out_path: str = Field(..., description="Output JSON file path. Required.")
    max_results: Optional[int] = Field(default=None, ge=1, le=5000, description="Max number of results to return (default 500, cap 5000). Omit for default.")

@tool("download_chembl_substructure_by_smiles", args_schema=ChemblSubstructureDownloadInput)
def download_chembl_substructure_by_smiles_tool(
    smiles: str,
    out_path: str,
    max_results: Optional[int] = None,
) -> str:
    """Download ChEMBL substructure search results to JSON file. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    try:
        return download_chembl_substructure_by_smiles(smiles, out_path, max_results=max_results)
    except Exception as e:
        return f"Download ChEMBL substructure by SMILES error: {str(e)}"

class ChemblDrugDownloadInput(BaseModel):
    chembl_id: str = Field(..., description="ChEMBL drug/molecule ID (e.g. CHEMBL25). Required.")
    out_path: str = Field(..., description="Output JSON file path. Required.")
    max_results: Optional[int] = Field(default=None, ge=1, le=5000, description="Max number of mechanism/indication records (default 500, cap 5000). Omit for default.")

@tool("download_chembl_drug_by_id", args_schema=ChemblDrugDownloadInput)
def download_chembl_drug_by_id_tool(
    chembl_id: str,
    out_path: str,
    max_results: Optional[int] = None,
) -> str:
    """Download ChEMBL drug info (drug, mechanisms, indications) to JSON file. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    try:
        return download_chembl_drug_by_id(chembl_id, out_path, max_results=max_results)
    except Exception as e:
        return f"Download ChEMBL drug by ID error: {str(e)}"

# foldseek

class FoldSeekSearchInput(BaseModel):
    pdb_file_path: str = Field(..., description="Absolute or relative path to the PDB structure file to use as query.")
    protect_start: int = Field(..., description="Start position (1-based inclusive) of the protected region to mask in the structure.", ge=1,)
    protect_end: int = Field(..., description="End position (1-based inclusive) of the protected region to mask.", ge=1,)
    out_dir: str = Field(default=None, description="Output directory for FoldSeek results. If not provided, will use default path.",)

@tool("download_foldseek_results_by_pdb_file", args_schema=FoldSeekSearchInput)
def download_foldseek_results_by_pdb_file_tool(
    pdb_file_path: str,
    protect_start: int,
    protect_end: int,
    out_dir: str = None,
) -> str:
    """Download FoldSeek results by PDB file (submit + wait + download pipeline). Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    try:
        out_dir = get_save_path("FoldSeek", "Download_data") if out_dir is None else out_dir
        return download_foldseek_results_by_pdb_file(pdb_file_path, protect_start, protect_end, out_dir=out_dir)
    except Exception as e:
        return f"Download FoldSeek results by PDB file error: {str(e)}"

# ---------- InterPro Database Tools (download only) ----------
# All return rich JSON: status, file_info, content_preview, biological_metadata, execution_context.

# --- Download: InterPro entry metadata by InterPro ID ---
class InterProMetadataDownloadInput(BaseModel):
    interpro_id: str = Field(..., description="InterPro entry ID (e.g. IPR001557). Required.")
    out_dir: str = Field(..., description="Output directory for metadata JSON file. Required.")

@tool("download_interpro_metadata_by_id", args_schema=InterProMetadataDownloadInput)
def download_interpro_metadata_by_id_tool(interpro_id: str, out_dir: str) -> str:
    """Download InterPro entry/family metadata by InterPro ID to JSON file. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    try:
        return download_interpro_metadata_by_id(interpro_id, out_dir)
    except Exception as e:
        return f"Download InterPro entry metadata by ID error: {str(e)}"

# --- Download: InterPro annotations by UniProt ID ---
class InterProAnnotationsDownloadInput(BaseModel):
    uniprot_id: str = Field(..., description="UniProt accession ID (e.g. P40925). Required.")
    out_dir: str = Field(..., description="Output directory for annotation JSON file. Required.")

@tool("download_interpro_annotations_by_uniprot_id", args_schema=InterProAnnotationsDownloadInput)
def download_interpro_annotations_by_uniprot_id_tool(uniprot_id: str, out_dir: str) -> str:
    """Download InterPro domain/function annotations and GO terms by UniProt ID to JSON file. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    try:
        return download_interpro_annotations_by_uniprot_id(uniprot_id, out_dir)
    except Exception as e:
        return f"Download InterPro annotations by UniProt ID error: {str(e)}"

# --- Download: InterPro family proteins by InterPro ID ---
class InterProProteinsDownloadInput(BaseModel):
    interpro_id: str = Field(..., description="InterPro entry ID (e.g. IPR001557). Required.")
    out_dir: str = Field(..., description="Output directory for protein detail/meta/uids files. Required.")
    max_results: Optional[int] = Field(default=None, ge=1, le=10000, description="Max number of proteins to download. Omit for all reviewed proteins.")

@tool("download_interpro_proteins_by_id", args_schema=InterProProteinsDownloadInput)
def download_interpro_proteins_by_id_tool(
    interpro_id: str,
    out_dir: str,
    max_results: Optional[int] = None,
) -> str:
    """Download reviewed protein list for an InterPro family (detail.json, meta.json, uids.txt). Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    try:
        return download_interpro_proteins_by_id(interpro_id, out_dir, max_results=max_results)
    except Exception as e:
        return f"Download InterPro proteins by ID error: {str(e)}"

# --- Download: UniProt ID list by InterPro ID ---
class InterProUniprotListDownloadInput(BaseModel):
    interpro_id: str = Field(..., description="InterPro entry ID (e.g. IPR001557). Required.")
    out_dir: str = Field(..., description="Output directory for chunked UniProt ID text files. Required.")
    protein_name: str = Field(default="", description="Prefix for output filenames. Defaults to InterPro ID if empty.")
    chunk_size: int = Field(default=5000, ge=1, description="Number of accessions per output file. Default 5000.")
    filter_name: Optional[str] = Field(default=None, description="Optional InterPro sub-filter name for the API query.")
    page_size: int = Field(default=200, ge=1, le=200, description="API page size for paginated fetching. Default 200.")
    max_results: Optional[int] = Field(default=None, ge=1, le=100000, description="Max number of UniProt accessions to fetch. Omit for all.")

@tool("download_interpro_uniprot_list_by_id", args_schema=InterProUniprotListDownloadInput)
def download_interpro_uniprot_list_by_id_tool(
    interpro_id: str,
    out_dir: str,
    protein_name: str = "",
    chunk_size: int = 5000,
    filter_name: Optional[str] = None,
    page_size: int = 200,
    max_results: Optional[int] = None,
) -> str:
    """Download UniProt accession list for an InterPro entry to chunked text files. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    try:
        return download_interpro_uniprot_list_by_id(
            interpro_id, out_dir, protein_name=protein_name, chunk_size=chunk_size,
            filter_name=filter_name, page_size=page_size, max_results=max_results,
        )
    except Exception as e:
        return f"Download UniProt ID list by InterPro ID error: {str(e)}"

# KEGG
class KeggDownloadInfoInput(BaseModel):
    database: str = Field(..., description="KEGG database name (e.g. pathway, compound, gene, genome, ko). Required.",)
    out_path: str = Field(..., description="Output file path to save KEGG info result (e.g. /path/to/kegg_info_pathway.txt). Required.",)

class KeggDownloadListInput(BaseModel):
    database: str = Field(..., description="KEGG database name (e.g. pathway, compound, gene). Required.",)
    out_path: str = Field(..., description="Output file path to save KEGG list result. Required.",)
    org_or_ids: Optional[str] = Field(default=None, description="Optional organism code (e.g. hsa, eco) or entry IDs to filter the list.",)

class KeggDownloadFindInput(BaseModel):
    database: str = Field(..., description="KEGG database to search (e.g. compound, pathway, gene). Required.",)
    query: str = Field(..., description="Search query string. Required.",)
    out_path: str = Field(..., description="Output file path to save KEGG find result. Required.",)
    option: Optional[str] = Field(default=None, description="Optional search option (e.g. formula, exact_mass, mol_weight for compound DB).",)

class KeggDownloadEntryInput(BaseModel):
    entry_id: str = Field(..., description="KEGG entry ID (e.g. hsa:7535, C00001, path:hsa04010). Required.",)
    out_path: str = Field(..., description="Output file path to save KEGG entry data. Required.",)
    format: Optional[str] = Field(default=None, description="Optional output format (e.g. aaseq, ntseq, mol, kcf, image, json, kgml).",)

class KeggDownloadConvInput(BaseModel):
    target_db: str = Field(..., description="Target database for ID conversion (e.g. ncbi-geneid, ncbi-proteinid, uniprot). Required.",)
    source_id: str = Field(..., description="Source KEGG ID(s) to convert (e.g. hsa:7535). Required.",)
    out_path: str = Field(..., description="Output file path to save conversion result. Required.",)

class KeggDownloadLinkInput(BaseModel):
    target_db: str = Field(..., description="Target KEGG database for cross-reference (e.g. pathway, enzyme, compound). Required.",)
    source_id: str = Field(..., description="Source KEGG ID(s) for cross-reference lookup (e.g. hsa:7535). Required.",)
    out_path: str = Field(..., description="Output file path to save link result. Required.",)

class KeggDownloadDdiInput(BaseModel):
    drug_id: str = Field(..., description="KEGG drug ID for drug-drug interaction query (e.g. D00001). Required.",)
    out_path: str = Field(..., description="Output file path to save DDI result. Required.",)

@tool("download_kegg_info_by_database", args_schema=KeggDownloadInfoInput)
def download_kegg_info_by_database_tool(database: str, out_path: str) -> str:
    """Download KEGG database info/statistics by database name to file. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context. Academic use only."""
    try:
        return download_kegg_info_by_database(database, out_path)
    except Exception as e:
        return f"Download KEGG database info by database name error: {str(e)}"

@tool("download_kegg_list_by_database", args_schema=KeggDownloadListInput)
def download_kegg_list_by_database_tool(database: str, out_path: str, org_or_ids: Optional[str] = None) -> str:
    """Download KEGG entry list by database name to file. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context. Academic use only."""
    try:
        return download_kegg_list_by_database(database, out_path, org_or_ids=org_or_ids)
    except Exception as e:
        return f"Download KEGG entry list by database name error: {str(e)}"

@tool("download_kegg_find_by_database", args_schema=KeggDownloadFindInput)
def download_kegg_find_by_database_tool(database: str, query: str, out_path: str, option: Optional[str] = None) -> str:
    """Download KEGG search results by database and query string to file. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context. Academic use only."""
    try:
        return download_kegg_find_by_database(database, query, out_path, option=option)
    except Exception as e:
        return f"Download KEGG search results by database and query string error: {str(e)}"

@tool("download_kegg_entry_by_id", args_schema=KeggDownloadEntryInput)
def download_kegg_entry_by_id_tool(entry_id: str, out_path: str, format: Optional[str] = None) -> str:
    """Download KEGG entry data by entry ID (e.g. hsa:7535, C00001) to file. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context. Academic use only."""
    try:
        return download_kegg_entry_by_id(entry_id, out_path, format=format)
    except Exception as e:
        return f"Download KEGG entry data by entry ID error: {str(e)}"

@tool("download_kegg_conv_by_id", args_schema=KeggDownloadConvInput)
def download_kegg_conv_by_id_tool(target_db: str, source_id: str, out_path: str) -> str:
    """Download KEGG ID conversion result (KEGG to/from external DB) to file. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context. Academic use only."""
    try:
        return download_kegg_conv_by_id(target_db, source_id, out_path)
    except Exception as e:
        return f"Download KEGG ID conversion result by ID error: {str(e)}"

@tool("download_kegg_link_by_id", args_schema=KeggDownloadLinkInput)
def download_kegg_link_by_id_tool(target_db: str, source_id: str, out_path: str) -> str:
    """Download KEGG cross-reference links by ID to file. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context. Academic use only."""
    try:
        return download_kegg_link_by_id(target_db, source_id, out_path)
    except Exception as e:
        return f"Download KEGG cross-reference links by ID error: {str(e)}"

@tool("download_kegg_ddi_by_id", args_schema=KeggDownloadDdiInput)
def download_kegg_ddi_by_id_tool(drug_id: str, out_path: str) -> str:
    """Download KEGG drug-drug interaction data by drug ID to file. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context. Academic use only."""
    try:
        return download_kegg_ddi_by_id(drug_id, out_path)
    except Exception as e:
        return f"Download KEGG drug-drug interaction data by drug ID error: {str(e)}"

# NCBI
class NcbiSequenceDownloadInput(BaseModel):
    ncbi_id: str = Field(..., description="NCBI accession ID (e.g. NP_000483.1). Required.")
    out_path: str = Field(..., description="Output file path to save FASTA. Required.")
    db: str = Field(default="protein", description="NCBI database to search ('protein' or 'nuccore'). Default 'protein'.")

class NcbiMetadataDownloadInput(BaseModel):
    ncbi_id: str = Field(..., description="NCBI accession ID (e.g. NP_000483.1). Required.")
    out_path: str = Field(..., description="Output file path to save GenBank/XML. Required.")
    db: str = Field(default="protein", description="NCBI database ('protein' or 'nuccore'). Default 'protein'.")
    rettype: str = Field(default="gb", description="Return format (e.g. 'gb', 'fasta'). Default 'gb'.")

from .ncbi.ncbi_blast import BLAST_PROGRAMS, BLAST_DATABASES
class NcbiBlastDownloadInput(BaseModel):
    sequence: str = Field(..., description="Protein or nucleotide sequence to BLAST. Required.")
    out_path: str = Field(..., description="Output file path to save BLAST XML. Required.")
    program: Literal[BLAST_PROGRAMS] = Field(default="blastp", description="BLAST program.", choices=BLAST_PROGRAMS)
    database: Literal[BLAST_DATABASES] = Field(default="swissprot", description="BLAST database.", choices=BLAST_DATABASES)
    hitlist_size: int = Field(default=50, description="Max hits. Default 50.")
    alignments: int = Field(default=25, description="Max alignments. Default 25.")
    format_type: str = Field(default="XML", description="Output format. Default 'XML'.")
    entrez_query: Optional[str] = Field(default=None, description="Optional Entrez query filter.")

class NcbiClinvarVariantsDownloadInput(BaseModel):
    term: str = Field(..., description="ClinVar search term (e.g. BRCA1[gene]). Required.")
    out_path: str = Field(..., description="Output file path to save JSON. Required.")
    retmax: int = Field(default=20, le=500, description="Max variants to fetch. Default 20.")

class NcbiGeneByIdDownloadInput(BaseModel):
    gene_id: str = Field(..., description="NCBI Gene ID (e.g. 672). Required.")
    out_path: str = Field(..., description="Output file path to save JSON. Required.")

class NcbiGeneBySymbolDownloadInput(BaseModel):
    symbol: str = Field(..., description="Gene symbol (e.g. BRCA1). Required.")
    taxon: str = Field(..., description="Taxon/organism (e.g. human, mouse). Required.")
    out_path: str = Field(..., description="Output file path to save JSON. Required.")

class NcbiBatchLookupBySymbolsDownloadInput(BaseModel):
    gene_symbols: List[str] = Field(..., description="List of gene symbols (e.g. ['BRCA1', 'TP53']). Required.")
    organism: str = Field(..., description="Organism (e.g. human). Required.")
    out_path: str = Field(..., description="Output file path to save JSON. Required.")

@tool("download_ncbi_sequence", args_schema=NcbiSequenceDownloadInput)
def download_ncbi_sequence_tool(ncbi_id: str, out_path: str, db: str = "protein") -> str:
    """Download NCBI sequence by accession to file. Returns rich JSON with file_info."""
    try:
        return download_ncbi_sequence(ncbi_id, out_path, db=db)
    except Exception as e:
        return f"Download NCBI sequence by accession error: {str(e)}"

@tool("download_ncbi_metadata", args_schema=NcbiMetadataDownloadInput)
def download_ncbi_metadata_tool(ncbi_id: str, out_path: str, db: str = "protein", rettype: str = "gb") -> str:
    """Download NCBI metadata by accession to file. Returns rich JSON with file_info."""
    try:
        return download_ncbi_metadata(ncbi_id, out_path, db=db, rettype=rettype)
    except Exception as e:
        return f"Download NCBI metadata by accession error: {str(e)}"

@tool("download_ncbi_blast", args_schema=NcbiBlastDownloadInput)
def download_ncbi_blast_tool(sequence: str, out_path: str, program: str = "blastp", database: str = "swissprot", hitlist_size: int = 50, alignments: int = 25, format_type: str = "XML", entrez_query: Optional[str] = None) -> str:
    """Submit sequence to NCBI BLAST and download XML. Returns rich JSON with file_info."""
    try:
        return download_ncbi_blast(sequence, out_path, program=program, database=database, hitlist_size=hitlist_size, alignments=alignments, format_type=format_type, entrez_query=entrez_query)
    except Exception as e:
        return f"Submit sequence to NCBI BLAST and download XML error: {str(e)}"

@tool("download_ncbi_clinvar_variants", args_schema=NcbiClinvarVariantsDownloadInput)
def download_ncbi_clinvar_variants_tool(term: str, out_path: str, retmax: int = 20) -> str:
    """Search and download ClinVar variants by term to JSON. Returns rich JSON with file_info."""
    try:
        return download_ncbi_clinvar_variants(term, out_path, retmax=retmax)
    except Exception as e:
        return f"Search and download ClinVar variants by term error: {str(e)}"

@tool("download_ncbi_gene_by_id", args_schema=NcbiGeneByIdDownloadInput)
def download_ncbi_gene_by_id_tool(gene_id: str, out_path: str) -> str:
    """Download NCBI Gene data by Gene ID to JSON. Returns rich JSON with file_info."""
    try:
        return download_ncbi_gene_by_id(gene_id, out_path)
    except Exception as e:
        return f"Download NCBI Gene data by Gene ID error: {str(e)}"

@tool("download_ncbi_gene_by_symbol", args_schema=NcbiGeneBySymbolDownloadInput)
def download_ncbi_gene_by_symbol_tool(symbol: str, taxon: str, out_path: str) -> str:
    """Download NCBI Gene data by Gene Symbol to JSON. Returns rich JSON with file_info."""
    try:
        return download_ncbi_gene_by_symbol(symbol, taxon, out_path)
    except Exception as e:
        return f"Download NCBI Gene data by Gene Symbol error: {str(e)}"

@tool("download_ncbi_batch_lookup_by_symbols", args_schema=NcbiBatchLookupBySymbolsDownloadInput)
def download_ncbi_batch_lookup_by_symbols_tool(gene_symbols: List[str], organism: str, out_path: str) -> str:
    """Download NCBI Gene batch lookup by symbols to JSON. Returns rich JSON with file_info."""
    try:
        return download_ncbi_batch_lookup_by_symbols(gene_symbols, organism, out_path)
    except Exception as e:
        return f"Download NCBI Gene batch lookup by symbols error: {str(e)}"

# RCSB PDB
class RCSBEntryDownloadInput(BaseModel):
    pdb_id: str = Field(..., description="RCSB PDB entry ID (e.g. 4HHB). Required.")
    out_path: str = Field(..., description="Output JSON file path. Required.")

class RCSBStructureDownloadInput(BaseModel):
    pdb_id: str = Field(..., description="RCSB PDB entry ID (e.g. 4HHB). Required.")
    out_dir: str = Field(..., description="Output directory for the structure file. Required.")
    file_type: str = Field(default="pdb", description="File type to download: 'pdb', 'cif', 'xml'. Default 'pdb'.")

@tool("download_rcsb_entry_metadata_by_pdb_id", args_schema=RCSBEntryDownloadInput)
def download_rcsb_entry_metadata_by_pdb_id_tool(pdb_id: str, out_path: str) -> str:
    """Download RCSB PDB entry metadata by PDB ID to JSON file. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    try:
        from .rcsb import download_rcsb_entry_metadata_by_pdb_id
        return download_rcsb_entry_metadata_by_pdb_id(pdb_id, out_path)
    except Exception as e:
        return f"Download RCSB PDB entry metadata by PDB ID error: {str(e)}"

@tool("download_rcsb_structure_by_pdb_id", args_schema=RCSBStructureDownloadInput)
def download_rcsb_structure_by_pdb_id_tool(pdb_id: str, out_dir: str, file_type: str = "pdb") -> str:
    """Download RCSB PDB structure file by PDB ID. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    try:
        from .rcsb import download_rcsb_structure_by_pdb_id
        return download_rcsb_structure_by_pdb_id(pdb_id, out_dir, file_type=file_type)
    except Exception as e:
        return f"Download RCSB PDB structure file by PDB ID error: {str(e)}"

# ---------- STRING Database Tools ----------
class StringMapIdsDownloadInput(BaseModel):
    identifiers: str = Field(..., description="Comma-separated gene or protein IDs (e.g. BRCA1, TP53). Required.")
    out_dir: str = Field(..., description="Output directory to save map_ids.tsv. Required.")
    species: int = Field(default=9606, description="NCBI taxonomy ID (9606 = human). Default 9606.")
    limit: int = Field(default=1, description="Max matches per identifier. Default 1.")
    echo_query: int = Field(default=1, description="Include query term in output. Default 1.")
    filename: str = Field(default="map_ids.tsv", description="Output filename. Default 'map_ids.tsv'.")

@tool("download_string_map_ids", args_schema=StringMapIdsDownloadInput)
def download_string_map_ids_tool(identifiers: str, out_dir: str, species: int = 9606, limit: int = 1, echo_query: int = 1, filename: str = "map_ids.tsv") -> str:
    """Download STRING map_ids results to TSV file. Returns rich JSON format."""
    try:
        from .string import download_string_map_ids
        ids_list = [x.strip() for x in identifiers.split(",") if x.strip()]
        return download_string_map_ids(ids_list if len(ids_list) > 1 else identifiers.strip(), out_dir, species=species, limit=limit, echo_query=echo_query, filename=filename)
    except Exception as e:
        return f"Download STRING map_ids results to TSV file error: {str(e)}"

class StringNetworkDownloadInput(BaseModel):
    identifiers: str = Field(..., description="Comma-separated gene or protein IDs. Required.")
    out_dir: str = Field(..., description="Output directory to save network.tsv. Required.")
    species: int = Field(default=9606, description="NCBI taxonomy ID. Default 9606.")
    required_score: int = Field(default=400, description="Confidence threshold 0-1000. Default 400.")
    network_type: str = Field(default="functional", description="Network type: functional or physical. Default functional.")
    add_nodes: int = Field(default=0, description="Add N most connected proteins (0-10). Default 0.")
    filename: str = Field(default="network.tsv", description="Output filename. Default 'network.tsv'.")

@tool("download_string_network", args_schema=StringNetworkDownloadInput)
def download_string_network_tool(identifiers: str, out_dir: str, species: int = 9606, required_score: int = 400, network_type: str = "functional", add_nodes: int = 0, filename: str = "network.tsv") -> str:
    """Download STRING PPI network to TSV file. Returns rich JSON format."""
    try:
        from .string import download_string_network
        ids_list = [x.strip() for x in identifiers.split(",") if x.strip()]
        return download_string_network(ids_list if len(ids_list) > 1 else identifiers.strip(), out_dir, species=species, required_score=required_score, network_type=network_type, add_nodes=add_nodes, filename=filename)
    except Exception as e:
        return f"Download STRING PPI network to TSV file error: {str(e)}"

class StringNetworkImageDownloadInput(BaseModel):
    identifiers: str = Field(..., description="Comma-separated gene or protein IDs. Required.")
    out_dir: str = Field(..., description="Output directory for network PNG image. Required.")
    species: int = Field(default=9606, description="NCBI taxonomy ID. Default 9606.")
    required_score: int = Field(default=400, description="Confidence threshold. Default 400.")
    network_flavor: str = Field(default="evidence", description="Image style: evidence, confidence, or actions. Default evidence.")
    add_nodes: int = Field(default=0, description="Add N most connected proteins. Default 0.")
    filename: str = Field(default="network.png", description="Output filename. Default 'network.png'.")

@tool("download_string_network_image", args_schema=StringNetworkImageDownloadInput)
def download_string_network_image_tool(identifiers: str, out_dir: str, species: int = 9606, required_score: int = 400, network_flavor: str = "evidence", add_nodes: int = 0, filename: str = "network.png") -> str:
    """Download STRING network as a PNG image. Returns rich JSON format."""
    try:
        from .string import download_string_network_image
        ids_list = [x.strip() for x in identifiers.split(",") if x.strip()]
        return download_string_network_image(ids_list if len(ids_list) > 1 else identifiers.strip(), out_dir, species=species, required_score=required_score, network_flavor=network_flavor, add_nodes=add_nodes, filename=filename)
    except Exception as e:
        return f"Download STRING network as a PNG image error: {str(e)}"

class StringInteractionPartnersDownloadInput(BaseModel):
    identifiers: str = Field(..., description="Comma-separated gene or protein IDs. Required.")
    out_dir: str = Field(..., description="Output directory to save interaction partners. Required.")
    species: int = Field(default=9606, description="NCBI taxonomy ID. Default 9606.")
    required_score: int = Field(default=400, description="Confidence threshold. Default 400.")
    limit: int = Field(default=10, description="Max partners per protein. Default 10.")
    filename: str = Field(default="interaction_partners.tsv", description="Output filename. Default 'interaction_partners.tsv'.")

@tool("download_string_interaction_partners", args_schema=StringInteractionPartnersDownloadInput)
def download_string_interaction_partners_tool(identifiers: str, out_dir: str, species: int = 9606, required_score: int = 400, limit: int = 10, filename: str = "interaction_partners.tsv") -> str:
    """Download STRING interaction partners to TSV file. Returns rich JSON format."""
    try:
        from .string import download_string_interaction_partners
        ids_list = [x.strip() for x in identifiers.split(",") if x.strip()]
        return download_string_interaction_partners(ids_list if len(ids_list) > 1 else identifiers.strip(), out_dir, species=species, required_score=required_score, limit=limit, filename=filename)
    except Exception as e:
        return f"Download STRING interaction partners to TSV file error: {str(e)}"

class StringEnrichmentDownloadInput(BaseModel):
    identifiers: str = Field(..., description="Comma-separated gene or protein IDs. Required.")
    out_dir: str = Field(..., description="Output directory to save enrichment results. Required.")
    species: int = Field(default=9606, description="NCBI taxonomy ID. Default 9606.")
    filename: str = Field(default="enrichment.tsv", description="Output filename. Default 'enrichment.tsv'.")

@tool("download_string_enrichment", args_schema=StringEnrichmentDownloadInput)
def download_string_enrichment_tool(identifiers: str, out_dir: str, species: int = 9606, filename: str = "enrichment.tsv") -> str:
    """Download STRING functional enrichment (GO/KEGG/Pfam) to TSV file. Returns rich JSON format."""
    try:
        from .string import download_string_enrichment
        ids_list = [x.strip() for x in identifiers.split(",") if x.strip()]
        return download_string_enrichment(ids_list if len(ids_list) > 1 else identifiers.strip(), out_dir, species=species, filename=filename)
    except Exception as e:
        return f"Download STRING functional enrichment (GO/KEGG/Pfam) to TSV file error: {str(e)}"

class StringPpiEnrichmentDownloadInput(BaseModel):
    identifiers: str = Field(..., description="Comma-separated gene or protein IDs. Required.")
    out_dir: str = Field(..., description="Output directory to save PPI enrichment. Required.")
    species: int = Field(default=9606, description="NCBI taxonomy ID. Default 9606.")
    required_score: int = Field(default=400, description="Confidence threshold. Default 400.")
    filename: str = Field(default="ppi_enrichment.json", description="Output filename. Default 'ppi_enrichment.json'.")

@tool("download_string_ppi_enrichment", args_schema=StringPpiEnrichmentDownloadInput)
def download_string_ppi_enrichment_tool(identifiers: str, out_dir: str, species: int = 9606, required_score: int = 400, filename: str = "ppi_enrichment.json") -> str:
    """Download STRING PPI network enrichment stats to JSON file. Returns rich JSON format."""
    try:
        from .string import download_string_ppi_enrichment
        ids_list = [x.strip() for x in identifiers.split(",") if x.strip()]
        return download_string_ppi_enrichment(ids_list if len(ids_list) > 1 else identifiers.strip(), out_dir, species=species, required_score=required_score, filename=filename)
    except Exception as e:
        return f"Download STRING PPI network enrichment stats to JSON file error: {str(e)}"

class StringHomologyDownloadInput(BaseModel):
    identifiers: str = Field(..., description="Comma-separated gene or protein IDs. Required.")
    out_dir: str = Field(..., description="Output directory to save homology results. Required.")
    species: int = Field(default=9606, description="NCBI taxonomy ID. Default 9606.")
    filename: str = Field(default="homology.tsv", description="Output filename. Default 'homology.tsv'.")

@tool("download_string_homology", args_schema=StringHomologyDownloadInput)
def download_string_homology_tool(identifiers: str, out_dir: str, species: int = 9606, filename: str = "homology.tsv") -> str:
    """Download STRING homology and similarity scores to TSV file. Returns rich JSON format."""
    try:
        from .string import download_string_homology
        ids_list = [x.strip() for x in identifiers.split(",") if x.strip()]
        return download_string_homology(ids_list if len(ids_list) > 1 else identifiers.strip(), out_dir, species=species, filename=filename)
    except Exception as e:
        return f"Download STRING homology and similarity scores to TSV file error: {str(e)}"


# Uniprot
class UniprotSearchByQueryInput(BaseModel):
    query: str = Field(..., description="Search query string. Required.")
    out_path: str = Field(..., description="Output file path. Required.")
    frmt: str = Field(default="tsv", description="Format (e.g., tsv, fasta, json, excel). Default 'tsv'.")
    columns: Optional[str] = Field(default=None, description="Comma-separated column names for TSV format.")
    limit: Optional[int] = Field(default=100, description="Max entries to download. Omit for default (100). Max 500 suggested.")
    database: str = Field(default="uniprotkb", description="UniProt database to search (e.g., uniprotkb, uniref, unipar). Default 'uniprotkb'.")

@tool("download_uniprot_search_by_query", args_schema=UniprotSearchByQueryInput)
def download_uniprot_search_by_query_tool(query: str, out_path: str, frmt: str = "tsv", columns: Optional[str] = None, limit: Optional[int] = 100, database: str = "uniprotkb", **filters) -> str:
    """Download UniProt search results. Returns rich JSON format."""
    try:
        from .uniprot import download_uniprot_search_by_query
        return download_uniprot_search_by_query(query=query, out_path=out_path, frmt=frmt, columns=columns, limit=limit, database=database, **filters)
    except Exception as e:
        return f"Download UniProt search results error: {str(e)}"

class UniprotRetrieveByIdInput(BaseModel):
    uniprot_id: str = Field(..., description="UniProtID or accession (e.g. P51451). Required.")
    out_path: str = Field(..., description="Output file path. Required.")
    frmt: str = Field(default="fasta", description="Download format (e.g., fasta, json, txt, xml). Default 'fasta'.")

@tool("download_uniprot_retrieve_by_id", args_schema=UniprotRetrieveByIdInput)
def download_uniprot_retrieve_by_id_tool(uniprot_id: str, out_path: str, frmt: str = "fasta") -> str:
    """Download single entry from UniProt. Returns rich JSON format."""
    try:
        from .uniprot import download_uniprot_retrieve_by_id
        return download_uniprot_retrieve_by_id(uniprot_id=uniprot_id, out_path=out_path, frmt=frmt)
    except Exception as e:
        return f"Download single entry from UniProt error: {str(e)}"

class UniprotMappingInput(BaseModel):
    fr: str = Field(..., description="From database name/ID (e.g., 'UniProtKB_AC-ID'). Required.")
    to: str = Field(..., description="To database name/ID (e.g., 'KEGG'). Required.")
    query: str = Field(..., description="Query ID(s), comma-separated if multiple. Required.")
    out_path: str = Field(..., description="Output JSON file path. Required.")

@tool("download_uniprot_mapping", args_schema=UniprotMappingInput)
def download_uniprot_mapping_tool(fr: str, to: str, query: str, out_path: str) -> str:
    """Download mapped IDs across databases via UniProt ID Mapping. Returns rich JSON format."""
    try:
        from .uniprot import download_uniprot_mapping
        return download_uniprot_mapping(fr=fr, to=to, query=query, out_path=out_path)
    except Exception as e:
        return f"Download mapped IDs across databases via UniProt ID Mapping error: {str(e)}"

class UniprotSeqByIdInput(BaseModel):
    uniprot_id: str = Field(..., description="UniProt accession ID (e.g. P40925). Required.")
    out_path: str = Field(..., description="Output FASTA file path. Required.")

@tool("download_uniprot_seq_by_id", args_schema=UniprotSeqByIdInput)
def download_uniprot_seq_by_id_tool(uniprot_id: str, out_path: str) -> str:
    """Download sequence (FASTA) from Uniprot. Returns rich JSON format."""
    try:
        from .uniprot import download_uniprot_seq_by_id
        return download_uniprot_seq_by_id(uniprot_id=uniprot_id, out_path=out_path)
    except Exception as e:
        return f"Download sequence (FASTA) from Uniprot error: {str(e)}"

class UniprotMetaByIdInput(BaseModel):
    uniprot_id: str = Field(..., description="UniProt accession ID (e.g. P40925). Required.")
    out_path: str = Field(..., description="Output JSON file path. Required.")

@tool("download_uniprot_meta_by_id", args_schema=UniprotMetaByIdInput)
def download_uniprot_meta_by_id_tool(uniprot_id: str, out_path: str) -> str:
    """Download metadata (JSON) from Uniprot. Returns rich JSON format."""
    try:
        from .uniprot import download_uniprot_meta_by_id
        return download_uniprot_meta_by_id(uniprot_id=uniprot_id, out_path=out_path)
    except Exception as e:
        return f"Download metadata (JSON) from Uniprot error: {str(e)}"

# DATABASE_TOOLS: by-ID fetch (UniProt, NCBI, RCSB, AlphaFold, InterPro)
DATABASE_TOOLS = [
    # AlphaFold
    download_alphafold_structure_by_uniprot_id_tool,
    download_alphafold_metadata_by_uniprot_id_tool,
    # BRENDA
    download_brenda_km_values_by_ec_number_tool,
    download_brenda_reactions_by_ec_number_tool,
    download_brenda_enzymes_by_substrate_tool,
    download_brenda_compare_organisms_by_ec_number_tool,
    download_brenda_environmental_parameters_by_ec_number_tool,
    download_brenda_kinetic_data_by_ec_number_tool,
    download_brenda_pathway_report_tool,
    # ChEMBL
    download_chembl_molecule_by_id_tool,
    download_chembl_similarity_by_smiles_tool,
    download_chembl_substructure_by_smiles_tool,
    download_chembl_drug_by_id_tool,
    # FoldSeek
    download_foldseek_results_by_pdb_file_tool,
    # InterPro
    download_interpro_metadata_by_id_tool,
    download_interpro_annotations_by_uniprot_id_tool,
    download_interpro_proteins_by_id_tool,
    download_interpro_uniprot_list_by_id_tool,
    # KEGG
    download_kegg_info_by_database_tool,
    download_kegg_list_by_database_tool,
    download_kegg_find_by_database_tool,
    download_kegg_entry_by_id_tool,
    download_kegg_conv_by_id_tool,
    download_kegg_link_by_id_tool,
    download_kegg_ddi_by_id_tool,
    # NCBI
    download_ncbi_sequence_tool,
    download_ncbi_metadata_tool,
    download_ncbi_blast_tool,
    download_ncbi_clinvar_variants_tool,
    download_ncbi_gene_by_id_tool,
    download_ncbi_gene_by_symbol_tool,
    download_ncbi_batch_lookup_by_symbols_tool,
    # RCSB
    download_rcsb_entry_metadata_by_pdb_id_tool,
    download_rcsb_structure_by_pdb_id_tool,
    # STRING
    download_string_map_ids_tool,
    download_string_network_tool,
    download_string_network_image_tool,
    download_string_interaction_partners_tool,
    download_string_enrichment_tool,
    download_string_ppi_enrichment_tool,
    download_string_homology_tool,
    # Uniprot
    download_uniprot_search_by_query_tool,
    download_uniprot_retrieve_by_id_tool,
    download_uniprot_mapping_tool,
    download_uniprot_seq_by_id_tool,
    download_uniprot_meta_by_id_tool,
]