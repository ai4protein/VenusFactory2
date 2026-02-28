"""
Aggregate tools_agent: mutation, predict, search, train, file.
Hub for tools_agent submodules; only import and get_tools/FASTAPI_TOOLS/MCP_TOOLS, no tool implementation.
Each series is a single list, then concatenated (no duplicate tool references).

Usage: get_tools() -> all tools; get_pi_tools() -> search-only; FASTAPI_TOOLS/MCP_TOOLS -> filtered subsets.
"""
from langchain.tools import BaseTool

from tools.mutation.tools_agent import (
    zero_shot_mutation_sequence_prediction_tool,
    zero_shot_mutation_structure_prediction_tool,
)
from tools.predict.tools_agent import (
    protein_property_prediction_tool,
    protein_function_prediction_tool,
    functional_residue_prediction_tool,
    esmfold_structure_prediction_tool,
)
from tools.search.tools_agent import (
    literature_search_tool,
    dataset_search_tool,
    web_search_tool,
    foldseek_search_tool,
    interpro_lookup_tool,
    uniprot_sequence_query_tool,
    uniprot_meta_query_tool,
    ncbi_sequence_query_tool,
    ncbi_meta_query_tool,
    rcsb_entry_query_tool,
    rcsb_structure_query_tool,
    alphafold_structure_query_tool,
    uniprot_sequence_download_tool,
    pdb_structure_download_tool,
    pdb_sequence_extraction_tool,
    ncbi_sequence_download_tool,
    alphafold_structure_download_tool,
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
    pdb_chain_sequences_tool,
    pdb_dir_to_fasta_tool,
    check_pdb_apo_tool,
    extract_uniprot_id_from_rcsb_metadata_tool,
)

# ---------------------------------------------------------------------------
# Per-series lists (each tool appears in exactly one list)
# ---------------------------------------------------------------------------

MUTATION_TOOLS: list[BaseTool] = [
    zero_shot_mutation_sequence_prediction_tool,
    zero_shot_mutation_structure_prediction_tool,
]

PREDICT_TOOLS: list[BaseTool] = [
    protein_property_prediction_tool,
    protein_function_prediction_tool,
    functional_residue_prediction_tool,
]

SEARCH_TOOLS: list[BaseTool] = [
    literature_search_tool,
    dataset_search_tool,
    web_search_tool,
    foldseek_search_tool,
    interpro_lookup_tool,
    uniprot_sequence_query_tool,
    uniprot_meta_query_tool,
    ncbi_sequence_query_tool,
    ncbi_meta_query_tool,
    rcsb_entry_query_tool,
    rcsb_structure_query_tool,
    alphafold_structure_query_tool,
    uniprot_sequence_download_tool,
    pdb_structure_download_tool,
    pdb_sequence_extraction_tool,
    ncbi_sequence_download_tool,
    alphafold_structure_download_tool,
    esmfold_structure_prediction_tool,
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

# ---------------------------------------------------------------------------
# Concatenated full list and subsets
# ---------------------------------------------------------------------------

ALL_TOOLS: list[BaseTool] = (
    MUTATION_TOOLS
    + PREDICT_TOOLS
    + SEARCH_TOOLS
    + TRAIN_TOOLS
    + FILE_TOOLS
)

# PI can execute only search tools (no download / train / file / etc.). Used for PI report and PI answer chains.
PI_SEARCH_TOOL_NAMES = frozenset({
    "literature_search",
    "dataset_search",
    "web_search",
    "foldseek_search",
})

# FASTAPI: all except train_protein_model, protein_model_predict, dataset_search, web_search, foldseek_search, esmfold_structure_prediction
FASTAPI_TOOL_NAMES = frozenset(t.name for t in ALL_TOOLS) - {
    "train_protein_model",
    "protein_model_predict",
    "dataset_search",
    "web_search",
    "foldseek_search",
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
    "MUTATION_TOOLS",
    "PREDICT_TOOLS",
    "SEARCH_TOOLS",
    "TRAIN_TOOLS",
    "FILE_TOOLS",
    "FASTAPI_TOOLS",
    "MCP_TOOLS",
    "FASTAPI_TOOL_NAMES",
    "MCP_TOOL_NAMES",
    "zero_shot_mutation_sequence_prediction_tool",
    "zero_shot_mutation_structure_prediction_tool",
    "protein_property_prediction_tool",
    "protein_function_prediction_tool",
    "functional_residue_prediction_tool",
    "literature_search_tool",
    "dataset_search_tool",
    "web_search_tool",
    "foldseek_search_tool",
    "interpro_lookup_tool",
    "uniprot_sequence_query_tool",
    "uniprot_meta_query_tool",
    "ncbi_sequence_query_tool",
    "ncbi_meta_query_tool",
    "rcsb_entry_query_tool",
    "rcsb_structure_query_tool",
    "alphafold_structure_query_tool",
    "uniprot_sequence_download_tool",
    "pdb_structure_download_tool",
    "pdb_sequence_extraction_tool",
    "ncbi_sequence_download_tool",
    "alphafold_structure_download_tool",
    "esmfold_structure_prediction_tool",
    "generate_training_config_tool",
    "train_protein_model_tool",
    "protein_model_predict_tool",
    "agent_generated_code_tool",
    "maxit_structure_convert_tool",
    "read_fasta_tool",
    "extract_uids_from_fasta_tool",
    "uid_file_to_chunks_tool",
    "unzip_archive_tool",
    "ungzip_file_tool",
    "pdb_chain_sequences_tool",
    "pdb_dir_to_fasta_tool",
    "check_pdb_apo_tool",
    "extract_uniprot_id_from_rcsb_metadata_tool",
]
