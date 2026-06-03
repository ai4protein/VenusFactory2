import json
import sys
from pathlib import Path

_REPO_ROOT = next((p for p in Path(__file__).absolute().parents if (p / "src").is_dir()), Path(__file__).absolute().parent)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from typing import List, Optional, Literal
from langchain.tools import tool
from pydantic import BaseModel, Field
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from src.web.utils.common_utils import get_save_path
from .alphafold import (
    download_alphafold_structure_by_uniprot_id,
    download_alphafold_metadata_by_uniprot_id,
    analyze_alphafold_plddt_by_metadata_file,
    analyze_alphafold_pae_by_pae_file,
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
from .clustalo import download_clustalo_msa_by_fasta
from .openalex import (
    download_openalex_entries_by_query,
    download_openalex_entry_by_id,
)
from .arxiv import download_arxiv_paper_by_id
from .biorxiv import download_biorxiv_by_doi
from .seq_search import (
    download_mmseqs2_homologs_by_sequence,
    download_blast_homologs_by_sequence,
)
from src.tools.visualize.pymol import (
    render_protein_structure,
    superpose_two_structures,
)
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
    translate_ncbi_cds_to_protein,
    search_ncbi_protein_by_gene_and_organism,
    download_pubmed_abstracts_by_pmids,
)
from .rcsb import (
    download_rcsb_entry_metadata_by_pdb_id,
    download_rcsb_structure_by_pdb_id,
    download_rcsb_search_by_query,
)
from .uniprot import (
    download_uniprot_search_by_query,
    download_uniprot_retrieve_by_id,
    download_uniprot_mapping,
    download_uniprot_seq_by_id,
    download_uniprot_meta_by_id,
    download_uniprot_sparql_by_query,
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

class AlphaFoldAnalyzePlddtInput(BaseModel):
    metadata_path: str = Field(..., description="Path to a previously downloaded AlphaFold metadata JSON (output of download_alphafold_metadata_by_uniprot_id). Required.")

@tool("analyze_alphafold_plddt_by_metadata_file", args_schema=AlphaFoldAnalyzePlddtInput)
def analyze_alphafold_plddt_by_metadata_file_tool(metadata_path: str) -> str:
    """Analyze AlphaFold pLDDT confidence fractions from a local metadata JSON. Returns rich JSON: {status, content, biological_metadata {uniprot_id, global_plddt, fractions, conclusion}}."""
    try:
        return analyze_alphafold_plddt_by_metadata_file(metadata_path)
    except Exception as e:
        return f"Analyze AlphaFold pLDDT error: {str(e)}"

class AlphaFoldAnalyzePaeInput(BaseModel):
    pae_path: str = Field(..., description="Path to an AlphaFold PAE (predicted aligned error) JSON file. Required.")
    distance_cutoff: float = Field(default=7.0, description="Avg-PAE cutoff (Å) for joining residues into the same sub-domain. Default 7.0.")
    min_domain_size: int = Field(default=40, ge=1, description="Minimum residues a sub-domain must have. Default 40.")

@tool("analyze_alphafold_pae_by_pae_file", args_schema=AlphaFoldAnalyzePaeInput)
def analyze_alphafold_pae_by_pae_file_tool(pae_path: str, distance_cutoff: float = 7.0, min_domain_size: int = 40) -> str:
    """Analyze an AlphaFold PAE matrix and detect global domain boundaries. Returns rich JSON: {status, content, biological_metadata {matrix_shape, mean_pae, domains[], conclusion}}."""
    try:
        return analyze_alphafold_pae_by_pae_file(pae_path, distance_cutoff=distance_cutoff, min_domain_size=min_domain_size)
    except Exception as e:
        return f"Analyze AlphaFold PAE error: {str(e)}"

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

# ---------- Clustal Omega (EBI) MSA Tools (download only) ----------

class ClustalOmegaMsaDownloadInput(BaseModel):
    fasta_path: str = Field(..., description="Path to a FASTA file containing 2-4000 sequences, ≤ 4 MB. Required.")
    out_dir: str = Field(..., description="Output directory for the MSA result file (<stem>_msa.fasta). Required.")
    email: Optional[str] = Field(default=None, description="Contact email for the EBI job. Falls back to env USER_EMAIL or noreply@venusfactory.cn.")
    poll_interval: float = Field(default=10.0, ge=1.0, description="Seconds between status polls. Default 10.")
    timeout_secs: int = Field(default=15 * 60, ge=10, description="Maximum total wait time. Default 900 s.")

@tool("download_clustalo_msa_by_fasta", args_schema=ClustalOmegaMsaDownloadInput)
def download_clustalo_msa_by_fasta_tool(
    fasta_path: str,
    out_dir: str,
    email: Optional[str] = None,
    poll_interval: float = 10.0,
    timeout_secs: int = 15 * 60,
) -> str:
    """Run EBI Clustal Omega multiple sequence alignment on a FASTA file. Returns rich JSON: {status, file_info, content_preview, biological_metadata {input_sequences, aligned_sequences, job_id, email}}."""
    try:
        return download_clustalo_msa_by_fasta(
            fasta_path, out_dir, email=email, poll_interval=poll_interval, timeout_secs=timeout_secs
        )
    except Exception as e:
        return f"Download Clustal Omega MSA error: {str(e)}"

# ---------- Sequence Similarity Search Tools (MMseqs2 + BLAST) ----------

class Mmseqs2HomologSearchInput(BaseModel):
    sequence_or_fasta_path: str = Field(..., description="Either a raw amino-acid sequence or a path to a FASTA file. Required.")
    out_dir: str = Field(..., description="Output directory for the hits JSON file. Required.")
    include_mgnify: bool = Field(default=False, description="Also include hits from the mgnify/environmental database. Default False.")
    poll_interval: float = Field(default=10.0, ge=1.0, description="Seconds between status polls. Default 10.")
    timeout_secs: int = Field(default=15 * 60, ge=10, description="Maximum wall-clock wait. Default 900 s.")

@tool("download_mmseqs2_homologs_by_sequence", args_schema=Mmseqs2HomologSearchInput)
def download_mmseqs2_homologs_by_sequence_tool(
    sequence_or_fasta_path: str,
    out_dir: str,
    include_mgnify: bool = False,
    poll_interval: float = 10.0,
    timeout_secs: int = 15 * 60,
) -> str:
    """Submit a protein sequence to ColabFold MMseqs2 and download parsed homologue hits as JSON. Returns rich JSON envelope; preview is a top-10 markdown table."""
    try:
        return download_mmseqs2_homologs_by_sequence(
            sequence_or_fasta_path, out_dir,
            include_mgnify=include_mgnify, poll_interval=poll_interval, timeout_secs=timeout_secs,
        )
    except Exception as e:
        return f"MMseqs2 homologue search error: {str(e)}"

class BlastHomologSearchInput(BaseModel):
    sequence_or_fasta_path: str = Field(..., description="Either a raw amino-acid sequence or a path to a FASTA file. Required.")
    out_dir: str = Field(..., description="Output directory for the hits JSON file. Required.")
    database: str = Field(default="uniprotkb_swissprot", description="EBI BLAST database (or comma-separated list). Examples: uniprotkb, uniprotkb_swissprot, uniref90, pdb. Default uniprotkb_swissprot.")
    email: Optional[str] = Field(default=None, description="Contact email for the EBI job. Falls back to env USER_EMAIL.")
    poll_interval: float = Field(default=30.0, ge=1.0, description="Seconds between status polls. Default 30.")
    timeout_secs: int = Field(default=15 * 60, ge=10, description="Maximum wall-clock wait. Default 900 s.")

@tool("download_blast_homologs_by_sequence", args_schema=BlastHomologSearchInput)
def download_blast_homologs_by_sequence_tool(
    sequence_or_fasta_path: str,
    out_dir: str,
    database: str = "uniprotkb_swissprot",
    email: Optional[str] = None,
    poll_interval: float = 30.0,
    timeout_secs: int = 15 * 60,
) -> str:
    """Submit a protein sequence to EBI NCBI BLAST and download parsed UniProt/PDB hits as JSON. Returns rich JSON envelope; preview is a top-10 markdown table."""
    try:
        return download_blast_homologs_by_sequence(
            sequence_or_fasta_path, out_dir,
            database=database, email=email, poll_interval=poll_interval, timeout_secs=timeout_secs,
        )
    except Exception as e:
        return f"BLAST homologue search error: {str(e)}"

# ---------- OpenAlex (scholarly works / authors / institutions / topics) ----------

class OpenAlexEntriesQueryInput(BaseModel):
    entity_type: str = Field(..., description="OpenAlex entity type: works, authors, sources, institutions, topics, concepts, domains, fields, subfields, sdgs, countries, continents, languages, keywords, publishers, funders. Required.")
    out_dir: str = Field(..., description="Output directory for the OpenAlex JSON page response. Required.")
    search: Optional[str] = Field(default=None, description="Free-text keyword (e.g. 'AlphaFold').")
    filter_expr: Optional[str] = Field(default=None, description="OpenAlex filter expression (e.g. 'authorships.author.id:A123,publication_year:>2020').")
    sort: Optional[str] = Field(default=None, description="OpenAlex sort expression (e.g. 'cited_by_count:desc', 'publication_date:desc').")
    per_page: int = Field(default=25, ge=1, le=200, description="Results per page. Default 25, max 200.")
    page: int = Field(default=1, ge=1, description="1-based page index. Default 1.")
    timeout: int = Field(default=30, ge=5, description="HTTP timeout in seconds.")

@tool("download_openalex_entries_by_query", args_schema=OpenAlexEntriesQueryInput)
def download_openalex_entries_by_query_tool(
    entity_type: str, out_dir: str,
    search: Optional[str] = None, filter_expr: Optional[str] = None, sort: Optional[str] = None,
    per_page: int = 25, page: int = 1, timeout: int = 30,
) -> str:
    """Search OpenAlex for scholarly entities. Save a single page of JSON to disk. Returns rich JSON envelope; biological_metadata contains total_count + next_page hint."""
    try:
        return download_openalex_entries_by_query(
            entity_type, out_dir,
            search=search, filter_expr=filter_expr, sort=sort,
            per_page=per_page, page=page, timeout=timeout,
        )
    except Exception as e:
        return f"OpenAlex search error: {str(e)}"

class OpenAlexEntryByIdInput(BaseModel):
    entity_type: str = Field(..., description="OpenAlex entity type (works, authors, sources, institutions, topics, ...). Required.")
    openalex_id: str = Field(..., description="Short OpenAlex ID (e.g. W3177828909, A5089215617) or a full openalex.org URL. Required.")
    out_dir: str = Field(..., description="Output directory for the entity JSON. Required.")
    timeout: int = Field(default=30, ge=5, description="HTTP timeout in seconds.")

@tool("download_openalex_entry_by_id", args_schema=OpenAlexEntryByIdInput)
def download_openalex_entry_by_id_tool(entity_type: str, openalex_id: str, out_dir: str, timeout: int = 30) -> str:
    """Fetch a single OpenAlex entity by ID and save the JSON. Returns rich JSON envelope; biological_metadata extracts title, cited_by_count, year, DOI when applicable."""
    try:
        return download_openalex_entry_by_id(entity_type, openalex_id, out_dir, timeout=timeout)
    except Exception as e:
        return f"OpenAlex entry-by-id error: {str(e)}"

# ---------- arXiv paper download ----------

class ArxivPaperDownloadInput(BaseModel):
    arxiv_id: str = Field(..., description="arXiv id (e.g. '2106.04559', '2106.04559v2', 'hep-th/9510017'). Required.")
    out_dir: str = Field(..., description="Output directory for the downloaded file. Required.")
    format: str = Field(default="pdf", description="One of: 'pdf' (default), 'html', 'source' (tar.gz). HTML only exists for newer papers.")
    timeout: int = Field(default=60, ge=5, description="HTTP timeout in seconds.")

@tool("download_arxiv_paper_by_id", args_schema=ArxivPaperDownloadInput)
def download_arxiv_paper_by_id_tool(arxiv_id: str, out_dir: str, format: str = "pdf", timeout: int = 60) -> str:
    """Download an arXiv paper as PDF / HTML / source tarball to out_dir. Returns rich JSON envelope; file_info.file_path is the saved file."""
    try:
        return download_arxiv_paper_by_id(arxiv_id, out_dir, format=format, timeout=timeout)
    except Exception as e:
        return f"arXiv download error: {str(e)}"

# ---------- bioRxiv / medRxiv per-DOI fetch ----------

class BiorxivByDoiInput(BaseModel):
    doi: str = Field(..., description="Bare DOI (e.g. '10.1101/2023.05.16.541025') or DOI URL. Required.")
    out_dir: str = Field(..., description="Output directory for the JSON. Required.")
    server: str = Field(default="biorxiv", description="'biorxiv' or 'medrxiv'. Default 'biorxiv'.")
    include_abstract: bool = Field(default=True, description="If False, abstract is stripped from saved JSON. Default True.")
    timeout: int = Field(default=30, ge=5, description="HTTP timeout in seconds.")

@tool("download_biorxiv_by_doi", args_schema=BiorxivByDoiInput)
def download_biorxiv_by_doi_tool(doi: str, out_dir: str, server: str = "biorxiv", include_abstract: bool = True, timeout: int = 30) -> str:
    """Fetch a single bioRxiv/medRxiv preprint by DOI. Returns rich JSON envelope; file_info.file_path → JSON with all versions + the latest record."""
    try:
        return download_biorxiv_by_doi(doi, out_dir, server=server, include_abstract=include_abstract, timeout=timeout)
    except Exception as e:
        return f"bioRxiv-by-doi error: {str(e)}"

# ---------- Visualization Tools (PyMOL headless rendering) ----------

class PymolRenderProteinStructureInput(BaseModel):
    pdb_path: str = Field(..., description="Path to a PDB or mmCIF structure file. Required.")
    out_dir: str = Field(..., description="Output directory for the rendered PNG + PSE session. Required.")
    color_by: str = Field(default="plddt", description="Coloring strategy: 'plddt' (B-factor as pLDDT spectrum), 'bfactor' (general), 'chain', or 'ss' (secondary structure).")
    width: int = Field(default=1200, ge=200, description="Output PNG width in pixels.")
    height: int = Field(default=900, ge=200, description="Output PNG height in pixels.")
    dpi: int = Field(default=150, ge=50, description="Output PNG DPI.")
    timeout_secs: int = Field(default=5 * 60, ge=10, description="PyMOL subprocess timeout.")

@tool("render_protein_structure", args_schema=PymolRenderProteinStructureInput)
def render_protein_structure_tool(
    pdb_path: str,
    out_dir: str,
    color_by: str = "plddt",
    width: int = 1200,
    height: int = 900,
    dpi: int = 150,
    timeout_secs: int = 5 * 60,
) -> str:
    """Render a protein structure to PNG via headless PyMOL (OSMesa). Returns rich JSON envelope including the PNG and PSE session paths."""
    try:
        return render_protein_structure(
            pdb_path, out_dir,
            color_by=color_by, width=width, height=height, dpi=dpi, timeout_secs=timeout_secs,
        )
    except Exception as e:
        return f"PyMOL render error: {str(e)}"

class PymolSuperposeInput(BaseModel):
    pdb_a: str = Field(..., description="Mobile structure path (gets aligned ONTO pdb_b). Required.")
    pdb_b: str = Field(..., description="Reference structure path. Required.")
    out_dir: str = Field(..., description="Output directory. Required.")
    width: int = Field(default=1200, ge=200)
    height: int = Field(default=900, ge=200)
    dpi: int = Field(default=150, ge=50)
    timeout_secs: int = Field(default=5 * 60, ge=10)

@tool("superpose_two_structures", args_schema=PymolSuperposeInput)
def superpose_two_structures_tool(
    pdb_a: str,
    pdb_b: str,
    out_dir: str,
    width: int = 1200,
    height: int = 900,
    dpi: int = 150,
    timeout_secs: int = 5 * 60,
) -> str:
    """Superpose pdb_a onto pdb_b via PyMOL cealign, render the alignment to PNG, return RMSD. Rich JSON envelope; biological_metadata.rmsd_angstroms is the alignment RMSD."""
    try:
        return superpose_two_structures(
            pdb_a, pdb_b, out_dir, width=width, height=height, dpi=dpi, timeout_secs=timeout_secs,
        )
    except Exception as e:
        return f"PyMOL superpose error: {str(e)}"

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

class NcbiCdsTranslateInput(BaseModel):
    accession: str = Field(..., description="NCBI nuccore accession (e.g. NM_000518 for HBB mRNA). Required.")
    out_dir: str = Field(..., description="Output directory for the FASTA file. Required.")
    target_length: int = Field(default=0, ge=0, description="Pick the CDS translation closest to this length (residues). 0 = longest. Default 0.")
    timeout: int = Field(default=60, ge=5, description="HTTP timeout in seconds.")

@tool("translate_ncbi_cds_to_protein", args_schema=NcbiCdsTranslateInput)
def translate_ncbi_cds_to_protein_tool(accession: str, out_dir: str, target_length: int = 0, timeout: int = 60) -> str:
    """Fetch a CDS-translated protein FASTA for an NCBI nuccore accession (uses efetch rettype=fasta_cds_aa). Picks the translation closest to target_length, or the longest. Returns rich JSON envelope."""
    try:
        return translate_ncbi_cds_to_protein(accession, out_dir, target_length=target_length, timeout=timeout)
    except Exception as e:
        return f"NCBI CDS translate error: {str(e)}"

class NcbiSearchByGeneOrganismInput(BaseModel):
    gene: str = Field(..., description="Gene symbol (e.g. TP53). Required.")
    organism: str = Field(..., description="Organism scientific name (e.g. 'Homo sapiens'). Required.")
    out_dir: str = Field(..., description="Output directory for the multi-FASTA + JSON summary. Required.")
    target_length: int = Field(default=0, ge=0, description="Narrow by sequence length window (±25 around target). 0 = no length filter. Default 0.")
    retmax: int = Field(default=10, ge=1, le=200, description="Max number of hits to fetch. Default 10.")
    timeout: int = Field(default=60, ge=5, description="HTTP timeout in seconds.")

@tool("search_ncbi_protein_by_gene_and_organism", args_schema=NcbiSearchByGeneOrganismInput)
def search_ncbi_protein_by_gene_and_organism_tool(
    gene: str, organism: str, out_dir: str, target_length: int = 0, retmax: int = 10, timeout: int = 60,
) -> str:
    """Search NCBI Protein DB by gene symbol + organism (with optional length filter), fetch all hits as a multi-FASTA. Returns rich JSON envelope; biological_metadata.summary_path points at the per-hit metadata JSON."""
    try:
        return search_ncbi_protein_by_gene_and_organism(
            gene, organism, out_dir, target_length=target_length, retmax=retmax, timeout=timeout,
        )
    except Exception as e:
        return f"NCBI protein search by gene+organism error: {str(e)}"

class PubmedAbstractsBatchInput(BaseModel):
    pmids: List[str] = Field(..., description="List of PubMed IDs (PMIDs). Max 200 per call. Required.")
    out_path: str = Field(..., description="Output JSON file path. Required.")
    timeout: int = Field(default=60, ge=5, description="HTTP timeout in seconds.")

@tool("download_pubmed_abstracts_by_pmids", args_schema=PubmedAbstractsBatchInput)
def download_pubmed_abstracts_by_pmids_tool(pmids: List[str], out_path: str, timeout: int = 60) -> str:
    """Fetch full title + authors + journal + structured abstract + DOI for a batch of PubMed PMIDs (one efetch call). Returns rich JSON envelope; file_info.file_path → JSON with `{requested_pmids, articles[]}`."""
    try:
        return download_pubmed_abstracts_by_pmids(pmids, out_path, timeout=timeout)
    except Exception as e:
        return f"PubMed batch abstract fetch error: {str(e)}"

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

class RCSBSearchInput(BaseModel):
    query: str = Field(..., description="JSON string of an RCSB Search API v2 query. Either a query block ({type, service, parameters}) or a full request payload ({query, ...}). Required.")
    out_dir: str = Field(..., description="Output directory for the response JSON. Required.")
    return_type: str = Field(default="entry", description="One of: entry, assembly, polymer_entity, non_polymer_entity, polymer_instance, mol_definition. Default 'entry' (PDB IDs).")
    page_start: Optional[int] = Field(default=None, description="Start index for pagination. If None and rows is None, all hits are returned.")
    rows: Optional[int] = Field(default=None, description="Number of results to return per page.")
    sort_by: Optional[str] = Field(default=None, description="Attribute to sort by, e.g. 'score' or 'rcsb_accession_info.initial_release_date'.")
    sort_direction: Optional[str] = Field(default=None, description="'asc' or 'desc'; used with sort_by.")
    count_only: bool = Field(default=False, description="Only return total_count, not the full result list. Cheaper for cardinality checks.")
    timeout: int = Field(default=60, ge=5, description="HTTP timeout in seconds.")

@tool("download_rcsb_search_by_query", args_schema=RCSBSearchInput)
def download_rcsb_search_by_query_tool(
    query: str,
    out_dir: str,
    return_type: str = "entry",
    page_start: Optional[int] = None,
    rows: Optional[int] = None,
    sort_by: Optional[str] = None,
    sort_direction: Optional[str] = None,
    count_only: bool = False,
    timeout: int = 60,
) -> str:
    """Run an RCSB Search API v2 query and save the matching identifiers to JSON. Returns rich JSON envelope; biological_metadata contains total_count, return_type, and pagination echo."""
    try:
        from .rcsb import download_rcsb_search_by_query
        return download_rcsb_search_by_query(
            query, out_dir,
            return_type=return_type, page_start=page_start, rows=rows,
            sort_by=sort_by, sort_direction=sort_direction,
            count_only=count_only, timeout=timeout,
        )
    except Exception as e:
        return f"RCSB search by query error: {str(e)}"

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

class UniprotSparqlInput(BaseModel):
    query: str = Field(..., description="SPARQL query string to run against https://sparql.uniprot.org/sparql. Required.")
    out_dir: str = Field(..., description="Output directory for the SPARQL JSON response. Required.")
    timeout: int = Field(default=120, ge=5, description="HTTP timeout in seconds. Default 120.")

@tool("download_uniprot_sparql_by_query", args_schema=UniprotSparqlInput)
def download_uniprot_sparql_by_query_tool(query: str, out_dir: str, timeout: int = 120) -> str:
    """Execute a SPARQL query against sparql.uniprot.org and save the raw JSON response. Returns rich JSON envelope; biological_metadata contains row_count and head_vars."""
    try:
        from .uniprot import download_uniprot_sparql_by_query
        return download_uniprot_sparql_by_query(query, out_dir, timeout=timeout)
    except Exception as e:
        return f"UniProt SPARQL error: {str(e)}"

class HpaGeneDownloadInput(BaseModel):
    gene_name: str = Field(..., description="Target gene symbol (e.g. EGFR, TP53).")
    out_path: str = Field(..., description="Output JSON file path.")

@tool("download_hpa_protein_by_gene", args_schema=HpaGeneDownloadInput)
def download_hpa_protein_by_gene_tool(gene_name: str, out_path: str) -> str:
    """Download Human Protein Atlas primary protein info for a gene to JSON file. Returns foundational metadata: Ensembl ID, UniProt accessions, gene description, protein class (e.g. Predicted secreted proteins, CD markers), and prognostic summary. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    try:
        from .hpa.hpa_operations import download_hpa_protein_by_gene
        return download_hpa_protein_by_gene(gene_name, out_path)
    except Exception as e:
        return f"Download HPA protein by gene error: {str(e)}"

@tool("download_hpa_subcellular_location_by_gene", args_schema=HpaGeneDownloadInput)
def download_hpa_subcellular_location_by_gene_tool(gene_name: str, out_path: str) -> str:
    """Download Human Protein Atlas immunofluorescence subcellular location data for a gene to JSON file. Returns subcellular compartments (e.g. Nucleoli, Plasma membrane) and automatically resolved localization_type (Secreted, Membrane, Intracellular, Unknown) based on biological heuristics. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    try:
        from .hpa.hpa_operations import download_hpa_subcellular_location_by_gene
        return download_hpa_subcellular_location_by_gene(gene_name, out_path)
    except Exception as e:
        return f"Download HPA subcellular location by gene error: {str(e)}"

@tool("download_hpa_tissue_expression_by_gene", args_schema=HpaGeneDownloadInput)
def download_hpa_tissue_expression_by_gene_tool(gene_name: str, out_path: str) -> str:
    """Download Human Protein Atlas macroscopic RNA tissue expression data for a gene to JSON file. Returns tissue specificity categories (e.g. Tissue enriched) and top expressed tissues based on nTPM values. Good for macroscopic somatic expression mapping. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    try:
        from .hpa.hpa_operations import download_hpa_tissue_expression_by_gene
        return download_hpa_tissue_expression_by_gene(gene_name, out_path)
    except Exception as e:
        return f"Download HPA tissue expression by gene error: {str(e)}"

@tool("download_hpa_single_cell_type_by_gene", args_schema=HpaGeneDownloadInput)
def download_hpa_single_cell_type_by_gene_tool(gene_name: str, out_path: str) -> str:
    """Download Human Protein Atlas RNA single cell type specificity data for a gene to JSON file. Returns per-cell-type nCPM expression values and cell type enrichment category (e.g. Astrocytes for GFAP, Hepatocytes for ALB). Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    try:
        from .hpa.hpa_operations import download_hpa_single_cell_type_by_gene
        return download_hpa_single_cell_type_by_gene(gene_name, out_path)
    except Exception as e:
        return f"Download HPA single cell type by gene error: {str(e)}"

@tool("download_hpa_blood_expression_by_gene", args_schema=HpaGeneDownloadInput)
def download_hpa_blood_expression_by_gene_tool(gene_name: str, out_path: str) -> str:
    """Download Human Protein Atlas blood cell expression and serum concentration data for any gene. Returns per-blood-cell-type nTPM expression (T cells, B cells, NK cells, monocytes, neutrophils, etc.), blood lineage specificity, and measured serum/plasma protein concentration (immunoassay and mass spectrometry). Useful for antibody design, biomarker discovery, and immune cell targeting. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    try:
        from .hpa.hpa_operations import download_hpa_blood_expression_by_gene
        return download_hpa_blood_expression_by_gene(gene_name, out_path)
    except Exception as e:
        return f"Download HPA blood expression by gene error: {str(e)}"

# DATABASE_TOOLS: by-ID fetch (UniProt, NCBI, RCSB, AlphaFold, InterPro)
DATABASE_TOOLS = [
    # AlphaFold
    download_alphafold_structure_by_uniprot_id_tool,
    download_alphafold_metadata_by_uniprot_id_tool,
    analyze_alphafold_plddt_by_metadata_file_tool,
    analyze_alphafold_pae_by_pae_file_tool,
    # Clustal Omega
    download_clustalo_msa_by_fasta_tool,
    # Sequence similarity search
    download_mmseqs2_homologs_by_sequence_tool,
    download_blast_homologs_by_sequence_tool,
    # OpenAlex (scholarly)
    download_openalex_entries_by_query_tool,
    download_openalex_entry_by_id_tool,
    # arXiv (literature download)
    download_arxiv_paper_by_id_tool,
    # bioRxiv (per-DOI)
    download_biorxiv_by_doi_tool,
    # PyMOL visualization
    render_protein_structure_tool,
    superpose_two_structures_tool,
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
    translate_ncbi_cds_to_protein_tool,
    search_ncbi_protein_by_gene_and_organism_tool,
    download_pubmed_abstracts_by_pmids_tool,
    # RCSB
    download_rcsb_entry_metadata_by_pdb_id_tool,
    download_rcsb_search_by_query_tool,
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
    download_uniprot_sparql_by_query_tool,
    # HPA
    download_hpa_protein_by_gene_tool,
    download_hpa_subcellular_location_by_gene_tool,
    download_hpa_tissue_expression_by_gene_tool,
    download_hpa_single_cell_type_by_gene_tool,
    download_hpa_blood_expression_by_gene_tool,
]