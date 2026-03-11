"""
Aggregate tools_agent: mutation, predict, search, train, file, skill.
Hub for tools_agent submodules; only import and get_tools/FASTAPI_TOOLS/MCP_TOOLS, no tool implementation.
Each series is a single list, then concatenated (no duplicate tool references).

Usage: get_tools() -> all tools; get_pi_tools() -> search-only; FASTAPI_TOOLS/MCP_TOOLS -> filtered subsets.

Tool inventory (all @tool from mutation, predict, search, train, file, skill):
  MUTATION (2): zero_shot_mutation_sequence_prediction, zero_shot_mutation_structure_prediction
  PREDICT (4): protein_property_prediction, protein_function_prediction, functional_residue_prediction, esmfold_structure_prediction
  SEARCH (6): query_literature_by_keywords, query_dataset_by_keywords, query_web_by_keywords, query_fda_by_keywords, query_foldseek_search_by_pdb_file, query_sequence_from_pdb_file
  DATABASE (32): by-ID (InterPro, UniProt, NCBI, RCSB, AlphaFold query/download) + BRENDA, ChEMBL, FDA, KEGG, STRING, ClinVar, UniProt search, NCBI BLAST
  TRAIN (4): generate_training_config, train_protein_model, protein_model_predict, agent_generated_code
  FILE (10): maxit_structure_convert, read_fasta, extract_uids_from_fasta, uid_file_to_chunks, unzip_archive, ungzip_file, pdb_chain_sequences, pdb_dir_to_fasta, check_pdb_apo, extract_uniprot_id_from_rcsb_metadata
  SKILL (1–2): read_skill, python_repl (if langchain_experimental installed)
"""
from langchain.tools import BaseTool
from tools.database.tools_agent import (
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
    # HPA
    download_hpa_protein_by_gene_tool,
    download_hpa_subcellular_location_by_gene_tool,
    download_hpa_tissue_expression_by_gene_tool,
    download_hpa_single_cell_type_by_gene_tool,
    download_hpa_blood_expression_by_gene_tool,
)

from tools.mutation.tools_agent import (
    zero_shot_mutation_sequence_prediction_tool,
    zero_shot_mutation_structure_prediction_tool,
)
from tools.predict.tools_agent import (
    calculate_physchem_from_fasta_tool,
    calculate_rsa_from_pdb_tool,
    calculate_sasa_from_pdb_tool,
    calculate_ss_from_pdb_tool,
    predict_protein_function_tool,
    predict_residue_function_tool,
    predict_structure_esmfold_tool,
)
from tools.search.tools_agent import (
    query_arxiv_tool,
    query_biorxiv_tool,
    query_duckduckgo_tool,
    query_fda_tool,
    query_github_tool,
    query_hugging_face_tool,
    query_pubmed_tool,
    query_semantic_scholar_tool,
    query_tavily_tool,
)
from tools.train.tools_agent import (
    generate_training_config_tool,
    train_protein_model_tool,
    agent_generated_code_tool,
    protein_model_predict_tool,
)
from tools.file.tools_agent import (
    maxit_structure_convert_tool,
    read_fasta_tool,
    extract_uids_from_fasta_tool,
    uid_file_to_chunks_tool,
    unzip_archive_tool,
    ungzip_file_tool,
    get_seq_from_pdb_chain_a_tool,
    pdb_chain_sequences_tool,
    pdb_dir_to_fasta_tool,
    check_pdb_apo_tool,
    extract_uniprot_id_from_rcsb_metadata_tool,
)
from tools.skill.tools_agent import read_skill_tool, get_python_repl_tool

# ---------------------------------------------------------------------------
# Per-series lists (each tool appears in exactly one list)
# ---------------------------------------------------------------------------

MUTATION_TOOLS: list[BaseTool] = [
    zero_shot_mutation_sequence_prediction_tool,
    zero_shot_mutation_structure_prediction_tool,
]

PREDICT_TOOLS: list[BaseTool] = [
    calculate_physchem_from_fasta_tool,
    calculate_rsa_from_pdb_tool,
    calculate_sasa_from_pdb_tool,
    calculate_ss_from_pdb_tool,
    predict_protein_function_tool,
    predict_residue_function_tool,
    predict_structure_esmfold_tool,
]

SEARCH_TOOLS: list[BaseTool] = [
    query_arxiv_tool,
    query_biorxiv_tool,
    query_duckduckgo_tool,
    query_fda_tool,
    query_github_tool,
    query_hugging_face_tool,
    query_pubmed_tool,
    query_semantic_scholar_tool,
    query_tavily_tool,
]

DATABASE_TOOLS: list[BaseTool] = [
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
    # HPA
    download_hpa_protein_by_gene_tool,
    download_hpa_subcellular_location_by_gene_tool,
    download_hpa_tissue_expression_by_gene_tool,
    download_hpa_single_cell_type_by_gene_tool,
    download_hpa_blood_expression_by_gene_tool,
]

TRAIN_TOOLS: list[BaseTool] = [
    generate_training_config_tool,
    train_protein_model_tool,
    protein_model_predict_tool,
    agent_generated_code_tool,
]

FILE_TOOLS: list[BaseTool] = [
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

# Skill tools: read_skill (CB/MLS see skills), optional python_repl (execute code; output/plots visible in chat)
_py_repl = get_python_repl_tool()
SKILL_TOOLS: list[BaseTool] = [read_skill_tool] + ([_py_repl] if _py_repl else [])

# ---------------------------------------------------------------------------
# Concatenated full list and subsets
# ---------------------------------------------------------------------------

ALL_TOOLS: list[BaseTool] = (
    MUTATION_TOOLS
    + PREDICT_TOOLS
    + SEARCH_TOOLS
    + TRAIN_TOOLS
    + FILE_TOOLS
    + SKILL_TOOLS
    + DATABASE_TOOLS
)

# PI can execute only search tools (no download / train / file / etc.). Used for PI report and PI answer chains.
PI_SEARCH_TOOL_NAMES = frozenset({
    "query_arxiv_tool",
    "query_biorxiv_tool",
    "query_duckduckgo_tool",
    "query_fda_tool",
    "query_github_tool",
    "query_hugging_face_tool",
    "query_pubmed_tool",
    "query_semantic_scholar_tool",
    "query_tavily_tool",
})

# FASTAPI: all except train_protein_model, protein_model_predict, query_dataset_by_keywords, query_web_by_keywords, query_foldseek_search_by_pdb_file, esmfold_structure_prediction
FASTAPI_TOOL_NAMES = frozenset(t.name for t in ALL_TOOLS) - {
    "train_protein_model",
    "protein_model_predict",
    "query_dataset_by_keywords",
    "query_web_by_keywords",
    "query_foldseek_search_by_pdb_file",
    "esmfold_structure_prediction",
}

# MCP: all except generate_training_config, train_protein_model, protein_model_predict, agent_generated_code
MCP_TOOL_NAMES = frozenset(t.name for t in ALL_TOOLS) - {
    "generate_training_config",
    "train_protein_model",
    "protein_model_predict",
    "agent_generated_code",
}


def get_tools() -> list[BaseTool]:
    """Full list of tools for the chat agent (Planner/Worker)."""
    return ALL_TOOLS


def get_pi_tools() -> list[BaseTool]:
    """Tools that PI is allowed to use (search only: literature, dataset, web, foldseek). No download or other tools."""
    return [t for t in ALL_TOOLS if t.name in PI_SEARCH_TOOL_NAMES]


FASTAPI_TOOLS = tuple(t for t in ALL_TOOLS if t.name in FASTAPI_TOOL_NAMES)
MCP_TOOLS = tuple(t for t in ALL_TOOLS if t.name in MCP_TOOL_NAMES)


__all__ = [
    "get_tools",
    "get_pi_tools",
    "PI_SEARCH_TOOL_NAMES",
    "ALL_TOOLS",
    "SKILL_TOOLS",
    "DATABASE_TOOLS",
    "MUTATION_TOOLS",
    "PREDICT_TOOLS",
    "SEARCH_TOOLS",
    "TRAIN_TOOLS",
    "FILE_TOOLS",
    "FASTAPI_TOOLS",
    "MCP_TOOLS",
    "FASTAPI_TOOL_NAMES",
    "MCP_TOOL_NAMES",
    # AlphaFold
    "download_alphafold_structure_by_uniprot_id_tool",
    "download_alphafold_metadata_by_uniprot_id_tool",
    # BRENDA
    "download_brenda_km_values_by_ec_number_tool",
    "download_brenda_reactions_by_ec_number_tool",
    "download_brenda_enzymes_by_substrate_tool",
    "download_brenda_compare_organisms_by_ec_number_tool",
    "download_brenda_environmental_parameters_by_ec_number_tool",
    "download_brenda_kinetic_data_by_ec_number_tool",
    "download_brenda_pathway_report_tool",
    # ChEMBL
    "download_chembl_molecule_by_id_tool",
    "download_chembl_similarity_by_smiles_tool",
    "download_chembl_substructure_by_smiles_tool",
    "download_chembl_drug_by_id_tool",
    # FoldSeek
    "download_foldseek_results_by_pdb_file_tool",
    # InterPro
    "download_interpro_metadata_by_id_tool",
    "download_interpro_annotations_by_uniprot_id_tool",
    "download_interpro_proteins_by_id_tool",
    "download_interpro_uniprot_list_by_id_tool",
    # KEGG
    "download_kegg_info_by_database_tool",
    "download_kegg_list_by_database_tool",
    "download_kegg_find_by_database_tool",
    "download_kegg_entry_by_id_tool",
    "download_kegg_conv_by_id_tool",
    "download_kegg_link_by_id_tool",
    "download_kegg_ddi_by_id_tool",
    # NCBI
    "download_ncbi_sequence_tool",
    "download_ncbi_metadata_tool",
    "download_ncbi_blast_tool",
    "download_ncbi_clinvar_variants_tool",
    "download_ncbi_gene_by_id_tool",
    "download_ncbi_gene_by_symbol_tool",
    "download_ncbi_batch_lookup_by_symbols_tool",
    # RCSB
    "download_rcsb_entry_metadata_by_pdb_id_tool",
    "download_rcsb_structure_by_pdb_id_tool",
    # STRING
    "download_string_map_ids_tool",
    "download_string_network_tool",
    "download_string_network_image_tool",
    "download_string_interaction_partners_tool",
    "download_string_enrichment_tool",
    "download_string_ppi_enrichment_tool",
    "download_string_homology_tool",
    # Uniprot
    "download_uniprot_search_by_query_tool",
    "download_uniprot_retrieve_by_id_tool",
    "download_uniprot_mapping_tool",
    "download_uniprot_seq_by_id_tool",
    "download_uniprot_meta_by_id_tool",
    # Mutation
    "zero_shot_mutation_sequence_prediction_tool",
    "zero_shot_mutation_structure_prediction_tool",
    # Predict
    "calculate_physchem_from_fasta_tool",
    "calculate_rsa_from_pdb_tool",
    "calculate_sasa_from_pdb_tool",
    "calculate_ss_from_pdb_tool",
    "predict_protein_function_tool",
    "predict_residue_function_tool",
    "predict_structure_esmfold_tool",
    # Search
    "query_arxiv_tool",
    "query_biorxiv_tool",
    "query_duckduckgo_tool",
    "query_fda_tool",
    "query_github_tool",
    "query_hugging_face_tool",
    "query_pubmed_tool",
    "query_semantic_scholar_tool",
    "query_tavily_tool",
    "download_foldseek_results_by_pdb_file_tool",
    # Train
    "generate_training_config_tool",
    "train_protein_model_tool",
    "protein_model_predict_tool",
    "agent_generated_code_tool",
    # File
    "maxit_structure_convert_tool",
    "read_fasta_tool",
    "extract_uids_from_fasta_tool",
    "uid_file_to_chunks_tool",
    "unzip_archive_tool",
    "ungzip_file_tool",
    "get_seq_from_pdb_chain_a_tool",
    "pdb_chain_sequences_tool",
    "pdb_dir_to_fasta_tool",
    "check_pdb_apo_tool",
    "extract_uniprot_id_from_rcsb_metadata_tool",
    "read_skill_tool",
]
