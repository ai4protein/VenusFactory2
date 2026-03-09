"""
# ==========================================
# Database MCP outer layer (tools/database/tools_mcp.py)
# ==========================================
# Uses FastMCP; each tool calls core download logic directly (no Agent layer).
# Explicit tool names for the model; each tool returns the core status dict as-is.
"""
import os
import sys

sys.path.insert(0, os.getcwd())

from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

# ========== Core layer imports (pure download logic, same as tools_api) ==========
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
from .string import (
    download_string_map_ids,
    download_string_network,
    download_string_network_image,
    download_string_interaction_partners,
    download_string_enrichment,
    download_string_ppi_enrichment,
    download_string_homology,
)
from .uniprot import (
    download_uniprot_search_by_query,
    download_uniprot_retrieve_by_id,
    download_uniprot_mapping,
    download_uniprot_seq_by_id,
    download_uniprot_meta_by_id,
)
from src.web.utils.common_utils import get_save_path


mcp = FastMCP("Venus_Bio_MCP")


# ---------- AlphaFold ----------
@mcp.tool(name="download_alphafold_structure")
def mcp_download_alphafold_structure(
    uniprot_id: str,
    out_dir: str,
    format: str = "pdb",
    version: str = "v6",
    fragment: int = 1,
) -> str:
    """Download AlphaFold 3D structure by UniProt ID."""
    result = download_alphafold_structure_by_uniprot_id(
        uniprot_id, out_dir, format=format, version=version, fragment=fragment
    )
    return result


@mcp.tool(name="download_alphafold_metadata")
def mcp_download_alphafold_metadata(uniprot_id: str, out_dir: str) -> str:
    """Download AlphaFold metadata by UniProt ID."""
    result = download_alphafold_metadata_by_uniprot_id(uniprot_id, out_dir)
    return result


# ---------- BRENDA ----------
@mcp.tool(name="download_brenda_km_values")
def mcp_download_brenda_km_values(
    ec_number: str,
    out_path: str,
    organism: str = "*",
    substrate: str = "*",
) -> str:
    """Download BRENDA Km values by EC number to file."""
    result = download_brenda_km_values_by_ec_number(
        ec_number, out_path, organism=organism, substrate=substrate
    )
    return result


@mcp.tool(name="download_brenda_reactions")
def mcp_download_brenda_reactions(
    ec_number: str, out_path: str, organism: str = "*"
) -> str:
    """Download BRENDA reactions by EC number to file."""
    result = download_brenda_reactions_by_ec_number(
        ec_number, out_path, organism=organism
    )
    return result


@mcp.tool(name="download_brenda_enzymes_by_substrate")
def mcp_download_brenda_enzymes_by_substrate(
    substrate: str, out_path: str, limit: int = 50
) -> str:
    """Download BRENDA enzyme-by-substrate search results to file."""
    result = download_brenda_enzymes_by_substrate(
        substrate, out_path, limit=limit
    )
    return result


@mcp.tool(name="download_brenda_compare_organisms")
def mcp_download_brenda_compare_organisms(
    ec_number: str, organisms: List[str], out_path: str
) -> str:
    """Download BRENDA organism comparison by EC number to file."""
    result = download_brenda_compare_organisms_by_ec_number(
        ec_number, organisms, out_path
    )
    return result


@mcp.tool(name="download_brenda_environmental_parameters")
def mcp_download_brenda_environmental_parameters(
    ec_number: str, out_path: str
) -> str:
    """Download BRENDA environmental parameters by EC number to file."""
    result = download_brenda_environmental_parameters_by_ec_number(
        ec_number, out_path
    )
    return result


@mcp.tool(name="download_brenda_kinetic_data")
def mcp_download_brenda_kinetic_data(
    ec_number: str, out_path: str, format: str = "json"
) -> str:
    """Download BRENDA kinetic data by EC number to file."""
    result = download_brenda_kinetic_data_by_ec_number(
        ec_number, out_path, format=format
    )
    return result


@mcp.tool(name="download_brenda_pathway_report")
def mcp_download_brenda_pathway_report(pathway: Dict[str, Any], out_path: str) -> str:
    """Generate and save BRENDA pathway report from pathway data to file."""
    result = download_brenda_pathway_report(pathway, out_path)
    return result


# ---------- ChEMBL ----------
@mcp.tool(name="download_chembl_molecule")
def mcp_download_chembl_molecule(mol_id: str, out_path: str) -> str:
    """Download ChEMBL molecule by ID to file."""
    result = download_chembl_molecule_by_id(mol_id, out_path)
    return result


@mcp.tool(name="download_chembl_similarity")
def mcp_download_chembl_similarity(
    smiles: str,
    out_path: str,
    threshold: int = 70,
    max_results: Optional[int] = None,
) -> str:
    """Download ChEMBL similarity search results by SMILES to file."""
    result = download_chembl_similarity_by_smiles(
        smiles, out_path, threshold=threshold, max_results=max_results
    )
    return result


@mcp.tool(name="download_chembl_substructure")
def mcp_download_chembl_substructure(
    smiles: str, out_path: str, max_results: Optional[int] = None
) -> str:
    """Download ChEMBL substructure search results by SMILES to file."""
    result = download_chembl_substructure_by_smiles(
        smiles, out_path, max_results=max_results
    )
    return result


@mcp.tool(name="download_chembl_drug")
def mcp_download_chembl_drug(
    chembl_id: str, out_path: str, max_results: Optional[int] = None
) -> str:
    """Download ChEMBL drug info (mechanisms, indications) by ID to file."""
    result = download_chembl_drug_by_id(
        chembl_id, out_path, max_results=max_results
    )
    return result


# ---------- FoldSeek ----------
@mcp.tool(name="download_foldseek_results")
def mcp_download_foldseek_results(
    pdb_file_path: str,
    protect_start: int,
    protect_end: int,
    out_dir: Optional[str] = None,
) -> str:
    """Run FoldSeek search with PDB file and download results (submit + wait + download)."""
    out_dir = out_dir or str(get_save_path("FoldSeek", "Download_data"))
    result = download_foldseek_results_by_pdb_file(
        pdb_file_path, protect_start, protect_end, out_dir=out_dir
    )
    return result


# ---------- InterPro ----------
@mcp.tool(name="download_interpro_metadata")
def mcp_download_interpro_metadata(interpro_id: str, out_dir: str) -> str:
    """Download InterPro entry metadata by InterPro ID to file."""
    result = download_interpro_metadata_by_id(interpro_id, out_dir)
    return result


@mcp.tool(name="download_interpro_annotations")
def mcp_download_interpro_annotations(uniprot_id: str, out_dir: str) -> str:
    """Download InterPro annotations by UniProt ID to file."""
    result = download_interpro_annotations_by_uniprot_id(uniprot_id, out_dir)
    return result


@mcp.tool(name="download_interpro_proteins")
def mcp_download_interpro_proteins(
    interpro_id: str, out_dir: str, max_results: Optional[int] = None
) -> str:
    """Download InterPro family protein list by InterPro ID to file."""
    result = download_interpro_proteins_by_id(
        interpro_id, out_dir, max_results=max_results
    )
    return result


@mcp.tool(name="download_interpro_uniprot_list")
def mcp_download_interpro_uniprot_list(
    interpro_id: str,
    out_dir: str,
    protein_name: str = "",
    chunk_size: int = 5000,
    filter_name: Optional[str] = None,
    page_size: int = 200,
    max_results: Optional[int] = None,
) -> str:
    """Download UniProt ID list for an InterPro entry to chunked files."""
    result = download_interpro_uniprot_list_by_id(
        interpro_id, out_dir,
        protein_name=protein_name, chunk_size=chunk_size,
        filter_name=filter_name, page_size=page_size, max_results=max_results,
    )
    return result


# ---------- KEGG ----------
@mcp.tool(name="download_kegg_info")
def mcp_download_kegg_info(database: str, out_path: str) -> str:
    """Download KEGG database info/statistics by database name to file."""
    result = download_kegg_info_by_database(database, out_path)
    return result


@mcp.tool(name="download_kegg_list")
def mcp_download_kegg_list(
    database: str, out_path: str, org_or_ids: Optional[str] = None
) -> str:
    """Download KEGG entry list by database name to file."""
    result = download_kegg_list_by_database(
        database, out_path, org_or_ids=org_or_ids
    )
    return result


@mcp.tool(name="download_kegg_find")
def mcp_download_kegg_find(
    database: str, query: str, out_path: str, option: Optional[str] = None
) -> str:
    """Download KEGG find/search results by database and query to file."""
    result = download_kegg_find_by_database(
        database, query, out_path, option=option
    )
    return result


@mcp.tool(name="download_kegg_entry")
def mcp_download_kegg_entry(
    entry_id: str, out_path: str, format: Optional[str] = None
) -> str:
    """Download KEGG entry data by entry ID to file."""
    result = download_kegg_entry_by_id(entry_id, out_path, format=format)
    return result


@mcp.tool(name="download_kegg_conv")
def mcp_download_kegg_conv(target_db: str, source_id: str, out_path: str) -> str:
    """Download KEGG ID conversion result to file."""
    result = download_kegg_conv_by_id(target_db, source_id, out_path)
    return result


@mcp.tool(name="download_kegg_link")
def mcp_download_kegg_link(
    target_db: str, source_id: str, out_path: str
) -> str:
    """Download KEGG cross-reference links to file."""
    result = download_kegg_link_by_id(target_db, source_id, out_path)
    return result


@mcp.tool(name="download_kegg_ddi")
def mcp_download_kegg_ddi(drug_id: str, out_path: str) -> str:
    """Download KEGG drug-drug interaction data by drug ID to file."""
    result = download_kegg_ddi_by_id(drug_id, out_path)
    return result


# ---------- NCBI ----------
@mcp.tool(name="download_ncbi_sequence")
def mcp_download_ncbi_sequence(
    ncbi_id: str, out_path: str, db: str = "protein"
) -> str:
    """Download NCBI sequence by accession to file."""
    result = download_ncbi_sequence(ncbi_id, out_path, db=db)
    return result


@mcp.tool(name="download_ncbi_metadata")
def mcp_download_ncbi_metadata(
    ncbi_id: str, out_path: str, db: str = "protein", rettype: str = "gb"
) -> str:
    """Download NCBI metadata by accession to file."""
    result = download_ncbi_metadata(
        ncbi_id, out_path, db=db, rettype=rettype
    )
    return result


@mcp.tool(name="download_ncbi_blast")
def mcp_download_ncbi_blast(
    sequence: str,
    out_path: str,
    program: str = "blastp",
    database: str = "swissprot",
    hitlist_size: int = 50,
    alignments: int = 25,
    format_type: str = "XML",
    entrez_query: Optional[str] = None,
) -> str:
    """Submit sequence to NCBI BLAST and download results to file."""
    result = download_ncbi_blast(
        sequence, out_path,
        program=program, database=database,
        hitlist_size=hitlist_size, alignments=alignments,
        format_type=format_type, entrez_query=entrez_query,
    )
    return result


@mcp.tool(name="download_ncbi_clinvar_variants")
def mcp_download_ncbi_clinvar_variants(
    term: str, out_path: str, retmax: int = 20
) -> str:
    """Search and download ClinVar variants by term to file."""
    result = download_ncbi_clinvar_variants(term, out_path, retmax=retmax)
    return result


@mcp.tool(name="download_ncbi_gene_by_id")
def mcp_download_ncbi_gene_by_id(gene_id: str, out_path: str) -> str:
    """Download NCBI Gene data by Gene ID to file."""
    result = download_ncbi_gene_by_id(gene_id, out_path)
    return result


@mcp.tool(name="download_ncbi_gene_by_symbol")
def mcp_download_ncbi_gene_by_symbol(
    symbol: str, taxon: str, out_path: str
) -> str:
    """Download NCBI Gene data by gene symbol and taxon to file."""
    result = download_ncbi_gene_by_symbol(symbol, taxon, out_path)
    return result


@mcp.tool(name="download_ncbi_batch_lookup")
def mcp_download_ncbi_batch_lookup(
    gene_symbols: List[str], organism: str, out_path: str
) -> str:
    """Download NCBI Gene batch lookup by symbols to file."""
    result = download_ncbi_batch_lookup_by_symbols(
        gene_symbols, organism, out_path
    )
    return result


# ---------- RCSB ----------
@mcp.tool(name="download_rcsb_metadata")
def mcp_download_rcsb_metadata(pdb_id: str, out_path: str) -> str:
    """Download RCSB PDB entry metadata by PDB ID to file."""
    result = download_rcsb_entry_metadata_by_pdb_id(pdb_id, out_path)
    return result


@mcp.tool(name="download_rcsb_structure")
def mcp_download_rcsb_structure(
    pdb_id: str, out_dir: str, file_type: str = "pdb"
) -> str:
    """Download RCSB PDB structure file by PDB ID."""
    result = download_rcsb_structure_by_pdb_id(
        pdb_id, out_dir, file_type=file_type
    )
    return result


# ---------- STRING ----------
def _string_ids(identifiers: str):
    ids = [x.strip() for x in identifiers.split(",") if x.strip()]
    return ids if len(ids) > 1 else (ids[0] if ids else identifiers.strip())


@mcp.tool(name="download_string_map_ids")
def mcp_download_string_map_ids(
    identifiers: str,
    out_dir: str,
    species: int = 9606,
    limit: int = 1,
    echo_query: int = 1,
    filename: str = "map_ids.tsv",
) -> str:
    """Download STRING map_ids results (gene/protein ID mapping) to file."""
    ids = _string_ids(identifiers)
    result = download_string_map_ids(
        ids, out_dir,
        species=species, limit=limit, echo_query=echo_query, filename=filename,
    )
    return result


@mcp.tool(name="download_string_network")
def mcp_download_string_network(
    identifiers: str,
    out_dir: str,
    species: int = 9606,
    required_score: int = 400,
    network_type: str = "functional",
    add_nodes: int = 0,
    filename: str = "network.tsv",
) -> str:
    """Download STRING PPI network to file."""
    ids = _string_ids(identifiers)
    result = download_string_network(
        ids, out_dir,
        species=species, required_score=required_score,
        network_type=network_type, add_nodes=add_nodes, filename=filename,
    )
    return result


@mcp.tool(name="download_string_network_image")
def mcp_download_string_network_image(
    identifiers: str,
    out_dir: str,
    species: int = 9606,
    required_score: int = 400,
    network_flavor: str = "evidence",
    add_nodes: int = 0,
    filename: str = "network.png",
) -> str:
    """Download STRING network as PNG image to file."""
    ids = _string_ids(identifiers)
    result = download_string_network_image(
        ids, out_dir,
        species=species, required_score=required_score,
        network_flavor=network_flavor, add_nodes=add_nodes, filename=filename,
    )
    return result


@mcp.tool(name="download_string_interaction_partners")
def mcp_download_string_interaction_partners(
    identifiers: str,
    out_dir: str,
    species: int = 9606,
    required_score: int = 400,
    limit: int = 10,
    filename: str = "interaction_partners.tsv",
) -> str:
    """Download STRING interaction partners to file."""
    ids = _string_ids(identifiers)
    result = download_string_interaction_partners(
        ids, out_dir,
        species=species, required_score=required_score,
        limit=limit, filename=filename,
    )
    return result


@mcp.tool(name="download_string_enrichment")
def mcp_download_string_enrichment(
    identifiers: str,
    out_dir: str,
    species: int = 9606,
    filename: str = "enrichment.tsv",
) -> str:
    """Download STRING functional enrichment (GO/KEGG/Pfam) to file."""
    ids = _string_ids(identifiers)
    result = download_string_enrichment(
        ids, out_dir, species=species, filename=filename
    )
    return result


@mcp.tool(name="download_string_ppi_enrichment")
def mcp_download_string_ppi_enrichment(
    identifiers: str,
    out_dir: str,
    species: int = 9606,
    required_score: int = 400,
    filename: str = "ppi_enrichment.json",
) -> str:
    """Download STRING PPI network enrichment stats to file."""
    ids = _string_ids(identifiers)
    result = download_string_ppi_enrichment(
        ids, out_dir,
        species=species, required_score=required_score, filename=filename,
    )
    return result


@mcp.tool(name="download_string_homology")
def mcp_download_string_homology(
    identifiers: str,
    out_dir: str,
    species: int = 9606,
    filename: str = "homology.tsv",
) -> str:
    """Download STRING homology/similarity results to file."""
    ids = _string_ids(identifiers)
    result = download_string_homology(
        ids, out_dir, species=species, filename=filename
    )
    return result


# ---------- UniProt ----------
@mcp.tool(name="download_uniprot_search")
def mcp_download_uniprot_search(
    query: str,
    out_path: str,
    frmt: str = "tsv",
    columns: Optional[str] = None,
    limit: Optional[int] = 100,
    database: str = "uniprotkb",
) -> str:
    """Download UniProt search results to file."""
    result = download_uniprot_search_by_query(
        query=query, out_path=out_path, frmt=frmt,
        columns=columns, limit=limit, database=database,
    )
    return result


@mcp.tool(name="download_uniprot_entry")
def mcp_download_uniprot_entry(
    uniprot_id: str, out_path: str, frmt: str = "fasta"
) -> str:
    """Download single UniProt entry by ID to file."""
    result = download_uniprot_retrieve_by_id(
        uniprot_id=uniprot_id, out_path=out_path, frmt=frmt
    )
    return result


@mcp.tool(name="download_uniprot_mapping")
def mcp_download_uniprot_mapping(
    fr: str, to: str, query: str, out_path: str
) -> str:
    """Download UniProt ID mapping result to file."""
    result = download_uniprot_mapping(
        fr=fr, to=to, query=query, out_path=out_path
    )
    return result


@mcp.tool(name="download_uniprot_sequence")
def mcp_download_uniprot_sequence(uniprot_id: str, out_path: str) -> str:
    """Download UniProt sequence (FASTA) by ID to file."""
    result = download_uniprot_seq_by_id(
        uniprot_id=uniprot_id, out_path=out_path
    )
    return result


@mcp.tool(name="download_uniprot_metadata")
def mcp_download_uniprot_metadata(uniprot_id: str, out_path: str) -> str:
    """Download UniProt metadata (JSON) by ID to file."""
    result = download_uniprot_meta_by_id(
        uniprot_id=uniprot_id, out_path=out_path
    )
    return result
