import json
import os
import re
import requests
import tempfile
import shutil
import time
import base64
import numpy as np
import gradio as gr
import uuid
import subprocess
import sys
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv
from gradio_client import Client, handle_file
import pandas as pd
from langchain.tools import tool
from pydantic import BaseModel, Field, validator
from web.utils.literature import literature_search

load_dotenv()

class ZeroShotSequenceInput(BaseModel):
    """Input for zero-shot sequence mutation prediction"""
    sequence: Optional[str] = Field(None, description="Protein sequence in single letter amino acid code")
    fasta_file: Optional[str] = Field(None, description="Path to FASTA file")
    model_name: str = Field(default="ESM2-650M", description="Model name: ESM-1v, ESM2-650M, ESM-1b, VenusPLM")

class ZeroShotStructureInput(BaseModel):
    """Input for zero-shot structure mutation prediction"""
    structure_file: str = Field(..., description="Path to PDB structure file")
    model_name: str = Field(default="ESM-IF1", description="Model name: SaProt, ProtSSN, ESM-IF1, MIF-ST, ProSST-2048")

class FunctionPredictionInput(BaseModel):
    """Input for protein function prediction"""
    sequence: Optional[str] = Field(None, description="Protein sequence in single letter amino acid code")
    fasta_file: Optional[str] = Field(None, description="Path to FASTA file")
    model_name: str = Field(default="ESM2-650M", description="Model name for function prediction")
    task: str = Field(default="Solubility", description="Task: Solubility, Subcellular Localization, Membrane Protein, Metal ion binding, Stability, Sortingsignal, Optimum temperature, Kcat, Optimal PH, Immunogenicity Prediction - Virus, Immunogenicity Prediction - Bacteria, Immunogenicity Prediction - Tumor")

class ResidueFunctionPredictionInput(BaseModel):
    """Input for functional residue prediction"""
    sequence: Optional[str] = Field(None, description="Protein sequence in single letter amino acid code")
    fasta_file: Optional[str] = Field(None, description="Path to FASTA file")
    model_name: str = Field(default="ESM2-650M", description="Model name for function prediction")
    task: str = Field(default="Activity Site", description="Task: Activity Site, Binding Site, Conserved Site, Motif")

class InterProQueryInput(BaseModel):
    """Input for InterPro database query"""
    uniprot_id: str = Field(..., description="UniProt ID for protein function query")

class UniProtQueryInput(BaseModel):
    """Input for UniProt database query"""
    uniprot_id: str = Field(..., description="UniProt ID for protein sequence query")


class PDBSequenceExtractionInput(BaseModel):
    """Input for extracting sequence from a local PDB file"""
    pdb_file: str = Field(..., description="Path to local PDB file")

class CSVTrainingConfigInput(BaseModel):
    """Input for CSV training config generation"""
    csv_file: str = Field(..., description="Path to CSV file with training data")
    test_csv_file: Optional[str] = Field(None, description="Optional path to test CSV file")
    output_name: str = Field(default="custom_training_config", description="Name for the generated config")

class ProteinPropertiesInput(BaseModel):
    """Input for protein properties generation"""
    sequence: Optional[str] = Field(None, description="Protein sequence in single letter amino acid code")
    fasta_file: Optional[str] = Field(None, description="Path to PDB structure file or fasta file")
    task_name: str = Field(default="Physical and chemical properties", description="Task name: Physical and chemical properties, Relative solvent accessible surface area (PDB only), SASA value (PDB only), Secondary structure (PDB only)")

class CodeExecutionInput(BaseModel):
    """Input for AI-generated code execution"""
    task_description: str = Field(..., description="Description of the task to be accomplished")
    input_files: List[str] = Field(default=[], description="List of input file paths")

class NCBISequenceInput(BaseModel):
    """Input for NCBI sequence download"""
    accession_id: str = Field(..., description="NCBI accession ID (e.g., NP_001123456, NM_001234567)")
    output_format: str = Field(default="fasta", description="Output format: fasta, genbank")

class AlphaFoldStructureInput(BaseModel):
    """Input for AlphaFold structure download"""
    uniprot_id: str = Field(..., description="UniProt ID for AlphaFold structure download")
    output_format: str = Field(default="pdb", description="Output format: pdb, mmcif")

class PDBStructureInput(BaseModel):
    """Input for PDB database structure download"""
    pdb_id: str = Field(..., description="PDB ID for protein structure download")
    output_format: str = Field(default="pdb", description="Output format: pdb, mmcif")

class LiteratureSearchInput(BaseModel):
    """Input for literature search"""
    query: str = Field(..., description="Search query")
    max_results: int = Field(5, description="Maximum number of results to return")


# Langchain Tools
@tool("zero_shot_sequence_prediction", args_schema=ZeroShotSequenceInput)
def zero_shot_sequence_prediction_tool(sequence: Optional[str] = None, fasta_file: Optional[str] = None, model_name: str = "ESM2-650M") -> str:
    """Predict beneficial mutations using sequence-based zero-shot models. Use for mutation prediction with protein sequences."""
    try:
        if fasta_file:
            if not os.path.exists(fasta_file):
                return f"Error: FASTA file not found at path: {fasta_file}"
            return call_zero_shot_sequence_prediction(
                fasta_file=fasta_file, model_name=model_name
                )
        elif sequence:
            return call_zero_shot_sequence_prediction(
                sequence=sequence, model_name=model_name
                )
        else:
            return "Error: Either sequence or fasta_file must be provided"
    except Exception as e:
        return f"Zero-shot sequence prediction error: {str(e)}"

@tool("zero_shot_structure_prediction", args_schema=ZeroShotStructureInput)
def zero_shot_structure_prediction_tool(structure_file: str, model_name: str = "ESM-IF1") -> str:
    """Predict beneficial mutations using structure-based zero-shot models. Use for mutation prediction with PDB structure files."""
    try:
        actual_file_path = structure_file
        try:
            import json
            if structure_file.startswith('{') and structure_file.endswith('}'):
                file_info = json.loads(structure_file)
                if isinstance(file_info, dict) and 'file_path' in file_info:
                    actual_file_path = file_info['file_path']
        except (json.JSONDecodeError, KeyError, TypeError):
            # If not JSON or doesn't have file_path, use original value
            pass
        
        if not os.path.exists(actual_file_path):
            return f"Error: Structure file not found at path: {actual_file_path}"
        return call_zero_shot_structure_prediction_from_file(actual_file_path, model_name)
    except Exception as e:
        return json.dumps({"success": False, "error": f"Zero-shot structure prediction error: {str(e)}"}, ensure_ascii=False)

@tool("protein_function_prediction", args_schema=FunctionPredictionInput)
def protein_function_prediction_tool(sequence: Optional[str] = None, fasta_file: Optional[str] = None, model_name: str = "ESM2-650M", task: str = "Solubility") -> str:
    """Predict protein functions like solubility, localization, metal ion binding, stability, sorting signal, and optimum temperature."""
    try:
        if fasta_file and os.path.exists(fasta_file):
            return call_protein_function_prediction(
                fasta_file=fasta_file, model_name=model_name, task=task
                )
        elif sequence:
            return call_protein_function_prediction(
                sequence=sequence, model_name=model_name, task=task
                )
        else:
            return "Error: Either sequence or fasta_file must be provided"
    except Exception as e:
        return f"Function protein prediction error: {str(e)}"

@tool("functional_residue_prediction", args_schema=ResidueFunctionPredictionInput)
def functional_residue_prediction_tool(sequence: Optional[str] = None, fasta_file: Optional[str] = None, model_name: str = "ESM2-650M", task: str = "Activate") -> str:
    try:
        if fasta_file and os.path.exists(fasta_file):
            return call_functional_residue_prediction(
                fasta_file=fasta_file, model_name=model_name, task=task
                )
        elif sequence:
            return call_functional_residue_prediction(
                sequence=sequence, model_name=model_name, task=task
                )
        else:
            return "Error: Either sequence or fasta_file must be procided"
    except Exception as e:
        return f"Functional residue prediction error: {str(e)}"

@tool("interpro_query", args_schema=InterProQueryInput)
def interpro_query_tool(uniprot_id: str) -> str:
    """Query InterPro database for protein function annotations and GO terms using UniProt ID."""
    try:
        return call_interpro_function_query(uniprot_id)
    except Exception as e:
        return f"InterPro query error: {str(e)}"

@tool("UniProt_query", args_schema=UniProtQueryInput)
def uniprot_query_tool(uniprot_id: str) -> str:
    """Query UniProt database for protein sequence"""
    try:
        return get_uniprot_sequence(uniprot_id)
    except Exception as e:
        return f"UniProt query error: {str(e)}"

@tool("PDB_structure_download", args_schema=PDBStructureInput)
def pdb_structure_download_tool(pdb_id: str, output_format: str = "pdb") -> str:
    """Download protein structure from PDB database using PDB ID."""
    try:
        return download_pdb_structure_from_id(pdb_id, output_format)
    except Exception as e:
        return f"PDB structure download error: {str(e)}"

@tool("PDB_sequence_extraction", args_schema=PDBSequenceExtractionInput)
def PDB_sequence_extraction_tool(pdb_file: str) -> str:
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

@tool("generate_training_config", args_schema=CSVTrainingConfigInput)
def generate_training_config_tool(csv_file: str, test_csv_file: Optional[str] = None, output_name: str = "custom_training_config") -> str:
    """Generate training JSON configuration from CSV files containing protein sequences and labels."""
    try:
        return process_csv_and_generate_config(csv_file, test_csv_file, output_name)
    except Exception as e:
        return f"Training config generation error: {str(e)}"

@tool("protein_properties_generation", args_schema=ProteinPropertiesInput)
def protein_properties_generation_tool(sequence: Optional[str] = None, fasta_file: Optional[str] = None, task_name = "Physical and chemical properties" ) -> str:
    """Predict the protein phyical, chemical, and structure properties."""
    try:
        if fasta_file:
            if not os.path.exists(fasta_file):
                return f"Error: FASTA file not found at path: {fasta_file}"
            return call_protein_properties_prediction(
                fasta_file=fasta_file, task_name=task_name
                )
        elif sequence:
            return call_protein_properties_prediction(
                sequence=sequence, task_name=task_name
                )
        else:
            return f"Error: Structure file not found at path: {fasta_file}"
        
    except Exception as e:
        return f"Protein properties prediction error: {str(e)}"

@tool("ai_code_execution", args_schema=CodeExecutionInput)
def ai_code_execution_tool(task_description: str, input_files: List[str] = []) -> str:
    """Generate and execute Python code based on task description. Use for data processing, splitting, analysis tasks."""
    try:
        return generate_and_execute_code(task_description, input_files)
    except Exception as e:
        return f"Code execution error: {str(e)}"

@tool("ncbi_sequence_download", args_schema=NCBISequenceInput)
def ncbi_sequence_download_tool(accession_id: str, output_format: str = "fasta") -> str:
    """Download protein or nucleotide sequences from NCBI database using accession ID."""
    try:
        return download_ncbi_sequence(accession_id, output_format)
    except Exception as e:
        return f"NCBI sequence download error: {str(e)}"

@tool("alphafold_structure_download", args_schema=AlphaFoldStructureInput)
def alphafold_structure_download_tool(uniprot_id: str, output_format: str = "pdb") -> str:
    """Download protein structures from AlphaFold database using UniProt ID."""
    try:
        return download_alphafold_structure(uniprot_id, output_format)
    except Exception as e:
        return f"AlphaFold structure download error: {str(e)}"


@tool("literature_search", args_schema=LiteratureSearchInput)
def literature_search_tool(query: str, max_results: int = 5) -> str:
    """
    Search for literature using MCP Tools: arXiv / PubMed / Google

    Args:
        query: The query to search for.
        max_results: The maximum number of results to return.

    Returns:
        A JSON string containing the search results.
    """
    try:
        refs = literature_search(query, max_results=max_results)
        return json.dumps({"success": True, "references": refs}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})

def call_zero_shot_sequence_prediction(
    sequence: str = None,
    fasta_file: str = None,
    model_name: str = "ESM2-650M",
    api_key: str = None
) -> str:
    """
    Call VenusFactory zero-shot sequence-based mutation prediction API.
    If fasta_file is provided, use it directly; otherwise, use sequence (writes to temp fasta).
    """
    try:
        if fasta_file:
            fasta_path = fasta_file
            temp_fasta_created = False
        elif sequence:
            temp_fasta = tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False)
            temp_fasta.write(f">temp_sequence\n{sequence}\n")
            temp_fasta.close()
            fasta_path = temp_fasta.name
            temp_fasta_created = True
        else:
            return "Zero-shot sequence prediction error: No sequence or fasta_file provided."

        client = Client("http://localhost:7860/")
        result = client.predict(
            function_selection="Activity",
            file_obj=handle_file(fasta_path),
            enable_ai=False,
            ai_model="DeepSeek",
            user_api_key=api_key,
            model_name=model_name,
            api_name="/handle_mutation_prediction_base"
        )

        if 'temp_fasta_created' in locals() and temp_fasta_created:
            os.unlink(fasta_path)

        # Limit mutation results to first 200 entries to avoid long context
        raw_result = result[2]
        try:
            import json
            # Check if raw_result is already a dict or needs to be parsed
            if isinstance(raw_result, dict):
                result_data = raw_result
            else:
                result_data = json.loads(raw_result)
            
            # Handle the data format with 'data' field containing mutations
            if isinstance(raw_result, dict) and 'data' in raw_result:
                mutations_data = raw_result['data']
                total_mutations = len(mutations_data)
                if total_mutations > 1000:
                    top_50 = mutations_data[:500]
                    bottom_50 = mutations_data[-500:]
                    separator_row = ['...', '...', '...']
                    combined_data = top_50 + [separator_row] + bottom_50
                    raw_result['data'] = combined_data
                    raw_result['total_mutations'] = total_mutations
                    raw_result['displayed_mutations'] = 1000
                    raw_result['note'] = (f"Showing top 500 most beneficial and bottom 500 least beneficial mutations "
                                          f"out of {total_mutations} total to avoid long context. "
                                          f"Results are separated by '...'.")
        
            return json.dumps(raw_result, indent=2)
        except (json.JSONDecodeError, KeyError, TypeError):
            # If not JSON or doesn't have expected structure, return as is
            return raw_result
    except Exception as e:
        return f"Zero-shot sequence prediction error: {str(e)}"

def call_zero_shot_structure_prediction_from_file(structure_file: str, model_name: str = "ESM-IF1", api_key: str = None) -> str:
    """Call VenusFactory zero-shot structure-based mutation prediction API"""
    try:
        client = Client("http://localhost:7860/")
        result = client.predict(
            function_selection="Activity",
            file_obj=handle_file(structure_file),
            enable_ai=False,
            ai_model="DeepSeek",
            user_api_key=api_key,
            model_name=model_name,
            api_name="/handle_mutation_prediction_base"
        )
        # Limit mutation results to first 200 entries to avoid long context
        raw_result = result[2]
        try:
            import json
            result_data = json.loads(raw_result)
            
            # Handle the data format with 'data' field containing mutations
            if isinstance(raw_result, dict) and 'data' in raw_result:
                mutations_data = raw_result['data']
                total_mutations = len(mutations_data)
                if total_mutations > 100:
                    top_50 = mutations_data[:50]
                    bottom_50 = mutations_data[-50:]
                    separator_row = ['...', '...', '...']
                    combined_data = top_50 + [separator_row] + bottom_50
                    raw_result['data'] = combined_data
                    raw_result['total_mutations'] = total_mutations
                    raw_result['displayed_mutations'] = 100
                    raw_result['note'] = (f"Showing top 50 most beneficial and bottom 50 least beneficial mutations "
                                          f"out of {total_mutations} total to avoid long context. "
                                          f"Results are separated by '...'.")
        
            return json.dumps(raw_result, indent=2)
        except (json.JSONDecodeError, KeyError, TypeError):
            # If not JSON or doesn't have expected structure, return as is
            return raw_result
    except Exception as e:
        return f"Zero-shot structure prediction error: {str(e)}"

def call_protein_function_prediction(
    sequence: str = None,
    fasta_file: str = None,
    model_name: str = "ProtT5-xl-uniref50",
    task: str = "Solubility",
    api_key: str = None
) -> str:
    """
    Call VenusFactory protein function prediction API.
    If fasta_file is provided, use it; otherwise, use sequence (writes to temp fasta).
    """
    try:
        dataset_mapping = {
            "Solubility": ["DeepSol", "DeepSoluE", "ProtSolM"],
            "Subcellular Localization": ["DeepLocMulti"],
            "Membrane Protein": ["DeepLocBinary"],
            "Metal Ion Binding": ["MetalIonBinding"], 
            "Stability": ["Thermostability"],
            "Sortingsignal": ["SortingSignal"], 
            "Optimal Temperature": ["DeepET_Topt"],
            "Kcat": ["DLKcat"],
            "Optimal PH": ["EpHod"],
            "Immunogenicity Prediction - Virus": ["VenusVaccine_VirusBinary"],
            "Immunogenicity Prediction - Bacteria": ["VenusVaccine_BacteriaBinary"],
            "Immunogenicity Prediction - Tumor": ["VenusVaccine_TumorBinary"],

        }
        datasets = dataset_mapping.get(task, ["DeepSol"])

        temp_fasta_path = None
        if fasta_file:
            fasta_path = fasta_file
        elif sequence:
            temp_fasta = tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False)
            temp_fasta.write(f">temp_sequence\n{sequence}\n")
            temp_fasta.close()
            temp_fasta_path = temp_fasta.name
            fasta_path = temp_fasta_path
        else:
            return "Error: Either sequence or fasta_file must be provided"

        client = Client("http://localhost:7860/")
        result = client.predict(
            task=task,
            fasta_file=handle_file(fasta_path),
            model_name=model_name,
            datasets=datasets,
            enable_ai=True,
            ai_model="DeepSeek",
            user_api_key=api_key,
            api_name="/handle_protein_function_prediction_chat"
        )

        if temp_fasta_path:
            os.unlink(temp_fasta_path)

        return result[1]
    except Exception as e:
        return f"Function prediction error: {str(e)}"

def call_functional_residue_prediction(
    sequence: str = None,
    fasta_file: str = None,
    model_name: str = "ESM2-650M",
    task: str = "Activity",
    api_key: str = None
) -> str:
    """
    Call VenusFactory functional residue prediction API using either a sequence or a FASTA file.
    If fasta_file is provided, use it; otherwise, use sequence (writes to temp fasta).
    """
    try:
        temp_fasta_path = None
        if fasta_file:
            fasta_path = fasta_file
        elif sequence:
            temp_fasta = tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False)
            temp_fasta.write(f">temp_sequence\n{sequence}\n")
            temp_fasta.close()
            temp_fasta_path = temp_fasta.name
            fasta_path = temp_fasta_path
        else:
            return "Error: Either sequence or fasta_file must be provided"

        client = Client("http://localhost:7860/")
        result = client.predict(
            task=task,
            fasta_file=handle_file(fasta_path),
            enable_ai=True,
            ai_model="DeepSeek",
            user_api_key=api_key,
            model_name=model_name,
            api_name="/handle_protein_residue_function_prediction_chat"
        )

        if temp_fasta_path:
            os.unlink(temp_fasta_path)

        # Filter results to only include residues with predicted label = 1
        raw_result = result[1]
        try:
            import json
            # Check if raw_result is already a dict or needs to be parsed
            if isinstance(raw_result, dict):
                result_data = raw_result
            else:
                result_data = json.loads(raw_result)
            
            # Handle the data format with 'data' field containing residue predictions
            if isinstance(result_data, dict) and 'data' in result_data:
                all_residues = result_data['data']
                # Filter to only keep residues with predicted label = 1
                functional_residues = [residue for residue in all_residues if residue[2] == 1]
                
                if functional_residues:
                    result_data['data'] = functional_residues
                    result_data['total_residues'] = len(all_residues)
                    result_data['functional_residues'] = len(functional_residues)
                    result_data['note'] = f"Showing {len(functional_residues)} functional residues (label=1) out of {len(all_residues)} total residues"
                    return json.dumps(result_data)
                else:
                    # No functional residues found
                    result_data['data'] = []
                    result_data['total_residues'] = len(all_residues)
                    result_data['functional_residues'] = 0
                    result_data['note'] = f"No functional residues (label=1) found out of {len(all_residues)} total residues"
                    return json.dumps(result_data)
            
            return raw_result
        except (json.JSONDecodeError, KeyError, TypeError):
            # If not JSON or doesn't have expected structure, return as is
            return raw_result
    except Exception as e:
        return f"Functional residue prediction error: {str(e)}"

def download_single_interpro(uniprot_id):
    """
    Fetches InterPro entries and GO annotations for a single UniProt ID.
    """
    url = f"https://www.ebi.ac.uk/interpro/api/protein/UniProt/{uniprot_id}/entry/?extra_fields=counters&page_size=100"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Will raise an exception for bad status codes (4xx or 5xx)
        data = response.json()
    except Exception as e:
        return {
            "success": False,
            "uniprot_id": uniprot_id,
            "error_message": f"Error during API call for {uniprot_id}: {str(e)}"
        }

    metadata = data.get("metadata", {})
    interpro_entries = data.get("entries", [])
    
    result = {
        "success": True,
        "uniprot_id": uniprot_id,
        "basic_info": {
            "uniprot_id": metadata.get("accession", ""),
            "protein_name": metadata.get("name", ""),
            "length": metadata.get("length", 0),
            "gene_name": metadata.get("gene", ""),
            "organism": metadata.get("source_organism", {}),
            "source_database": metadata.get("source_database", ""),
            "in_alphafold": metadata.get("in_alphafold", False)
        },
        "interpro_entries": interpro_entries,
        "go_annotations": {
            "molecular_function": [],
            "biological_process": [],
            "cellular_component": []
        },
        "counters": metadata.get("counters", {}),
        "num_entries": len(interpro_entries)
    }

    if "go_terms" in metadata:
        for go_term in metadata["go_terms"]:
            category_name = go_term.get("category", {}).get("name", "")
            go_annotation = {
                "go_id": go_term.get("identifier", ""),
                "name": go_term.get("name", "")
            }
            
            if category_name == "molecular_function":
                result["go_annotations"]["molecular_function"].append(go_annotation)
            elif category_name == "biological_process":
                result["go_annotations"]["biological_process"].append(go_annotation)
            elif category_name == "cellular_component":
                result["go_annotations"]["cellular_component"].append(go_annotation)

    return json.dumps(result, indent=4)

def call_interpro_function_query(uniprot_id: str) -> str:
    """Query InterPro database for protein function"""
    try:
        result = download_single_interpro(uniprot_id)
        return result
    except Exception as e:
        return f"InterPro query error: {str(e)}"


def get_uniprot_sequence(uniprot_id):
    """
    Fetches protein sequence for a single UniProt ID.
    """
    url = f"https://www.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        fasta_text = response.text
        
        # Extract sequence from FASTA format (skip header line)
        lines = fasta_text.strip().split('\n')
        sequence = ''.join(lines[1:])  # Skip first line (header)
        
        return {
            "success": True,
            "uniprot_id": uniprot_id,
            "sequence": sequence
        }
        
    except Exception as e:
        return {
            "success": False,
            "uniprot_id": uniprot_id,
            "error_message": f"Error fetching sequence for {uniprot_id}: {str(e)}"
        }

def download_pdb_structure_from_id(pdb_id: str, output_format: str = "pdb") -> str:
    try:
        # Create temporary directory for output
        temp_dir = Path("temp_outputs")
        structure_dir = temp_dir / "pdb_structures"
        os.makedirs(structure_dir, exist_ok=True)

        pdb_id = pdb_id.upper()
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"

        response = requests.get(url, timeout=30)
        response.raise_for_status()
        structure_text = response.text

        expected_file = structure_dir / f"{pdb_id}.pdb"
        with open(expected_file, "w", encoding="utf-8") as f:
            f.write(structure_text)

        # Parse saved PDB and extract chain A by default
        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure(pdb_id, str(expected_file))
            seqs = []
            for model in structure:
                for chain in model:
                    if chain.id == "A":
                        residues = []
                        for residue in chain:
                            if residue.id[0] == " ":
                                try:
                                    residues.append(seq1(residue.resname))
                                except Exception:
                                    pass
                        chain_seq = "".join(residues)
                        seqs.append({"chain": chain.id, "sequence": chain_seq})
                        break
                if seqs:
                    break
            if not seqs:
                # fallback: try first chain
                for model in structure:
                    for chain in model:
                        residues = []
                        for residue in chain:
                            if residue.id[0] == " ":
                                try:
                                    residues.append(seq1(residue.resname))
                                except Exception:
                                    pass
                        chain_seq = "".join(residues)
                        if chain_seq:
                            seqs.append({"chain": chain.id, "sequence": chain_seq})
                            break
                    if seqs:
                        break

            if not seqs:
                return json.dumps({"success": False, "pdb_id": pdb_id, "error_message": "No protein sequence found in PDB file."})

            return json.dumps({
                "success": True,
                "pdb_id": pdb_id,
                "pdb_file": str(expected_file),
                "sequences": seqs,
                "message": f"PDB structure downloaded and chain A extracted: {expected_file}"
            }, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "pdb_id": pdb_id, "error_message": f"Failed to parse PDB: {str(e)}"})
    except Exception as e:
        return json.dumps({"success": False, "pdb_id": pdb_id, "error_message": f"Error downloading structure for {pdb_id}: {str(e)}"})

def download_ncbi_sequence(accession_id: str, output_format: str = "fasta") -> str:
    """Download protein or nucleotide sequences from NCBI using existing crawler script."""
    try:
        # Create temporary directory for output
        temp_dir = Path("temp_outputs")
        sequence_dir = temp_dir / "ncbi_sequences"
        os.makedirs(sequence_dir, exist_ok=True)
        
        # Determine database type based on accession ID
        db_type = "protein" if accession_id.startswith(('NP_', 'XP_', 'YP_', 'WP_')) else "nuccore"
        
        # Call the existing NCBI download script
        cmd = [
            "python", "src/crawler/sequence/download_ncbi_seq.py",
            "--id", accession_id,
            "--out_dir", str(sequence_dir),
            "--db", db_type
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Find the downloaded file
        expected_file = sequence_dir / f"{accession_id}.fasta"
        if expected_file.exists():
            return json.dumps({
                "success": True,
                "accession_id": accession_id,
                "format": output_format,
                "pdb_file": str(expected_file),
                "message": f"Sequence downloaded successfully and saved to: {expected_file}",
                "script_output": result.stdout
            })
        else:
            return json.dumps({
                "success": False,
                "accession_id": accession_id,
                "error_message": f"Download completed but file not found: {expected_file}",
                "script_output": result.stdout
            })
            
    except subprocess.CalledProcessError as e:
        return json.dumps({
            "success": False,
            "accession_id": accession_id,
            "error_message": f"Download script failed: {e.stderr}",
            "script_output": e.stdout
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "accession_id": accession_id,
            "error_message": f"Error downloading sequence for {accession_id}: {str(e)}"
        })

def download_alphafold_structure(uniprot_id: str, output_format: str = "pdb") -> str:
    """Download protein structures from AlphaFold using existing crawler script."""
    try:
        # Create temporary directory for output
        temp_dir = Path("temp_outputs")
        structure_dir = temp_dir / "alphafold_structures"
        os.makedirs(structure_dir, exist_ok=True)
        
        # Call the existing AlphaFold download script (not RCSB!)
        cmd = [
            "python", "src/crawler/structure/download_alphafold.py",
            "--uniprot_id", uniprot_id,
            "--out_dir", str(structure_dir)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Find the downloaded file (AlphaFold uses UniProt ID as filename)
        expected_file = structure_dir / f"{uniprot_id}.pdb"
        if expected_file.exists():
            # Extract confidence information from PDB file
            confidence_info = {}
            try:
                with open(expected_file, 'r') as f:
                    structure_data = f.read()
                
                confidence_scores = []
                for line in structure_data.split('\n'):
                    if line.startswith('ATOM') and 'CA' in line:
                        try:
                            confidence = float(line[60:66].strip())
                            confidence_scores.append(confidence)
                        except (ValueError, IndexError):
                            continue
                
                if confidence_scores:
                    confidence_info = {
                        "mean_confidence": round(sum(confidence_scores) / len(confidence_scores), 2),
                        "min_confidence": round(min(confidence_scores), 2),
                        "max_confidence": round(max(confidence_scores), 2),
                        "high_confidence_residues": sum(1 for score in confidence_scores if score >= 70),
                        "total_residues": len(confidence_scores)
                    }
            except Exception:
                pass  # Confidence parsing failed, but download was successful
            
            return json.dumps({
                "success": True,
                "uniprot_id": uniprot_id,
                "format": output_format,
                "pdb_file": str(expected_file),
                "confidence_info": confidence_info,
                "message": f"AlphaFold structure downloaded successfully and saved to: {expected_file}",
                "script_output": result.stdout
            })
        else:
            return json.dumps({
                "success": False,
                "uniprot_id": uniprot_id,
                "error_message": f"Download completed but file not found: {expected_file}",
                "script_output": result.stdout
            })
            
    except subprocess.CalledProcessError as e:
        return json.dumps({
            "success": False,
            "uniprot_id": uniprot_id,
            "error_message": f"Download script failed: {e.stderr}",
            "script_output": e.stdout
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "uniprot_id": uniprot_id,
            "error_message": f"Error downloading structure for {uniprot_id}: {str(e)}"
        })

def call_protein_properties_prediction(sequence: str = None, fasta_file: str = None, task_name: str = "Physical and chemical properties", api_key: str = None) -> str:
    """
    Predict protein properties from a sequence or a fasta file.
    If fasta_file is provided, use it directly; otherwise, use sequence (writes to temp fasta).
    """
    try:
        if fasta_file:
            file_path = fasta_file
            temp_fasta_created = False
        elif sequence:
            temp_fasta = tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False)
            temp_fasta.write(f">temp_sequence\n{sequence}\n")
            temp_fasta.close()
            file_path = temp_fasta.name
            temp_fasta_created = True
        else:
            return "Protein properties prediction error: No sequence or fasta_file provided."

        client = Client("http://localhost:7860/")
        result = client.predict(
            task=task_name,
            file_obj=handle_file(file_path),
            api_name="/handle_protein_properties_generation"
        )

        if 'temp_fasta_created' in locals() and temp_fasta_created:
            os.unlink(file_path)
        return result[1]
    except Exception as e:
        return f"Protein properties prediction error: {str(e)}"

def generate_and_execute_code(task_description: str, input_files: List[str] = []) -> str:
    """
    Generate and execute Python code based on task description.
    """
    script_path = None 
    try:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        
        # Validate and prepare input files
        valid_files = []
        file_info = []
        for file_path in input_files:
            if os.path.exists(file_path):
                valid_files.append(file_path)
                # Get file info for better context
                file_ext = os.path.splitext(file_path)[1]
                file_size = os.path.getsize(file_path)
                file_info.append({
                    "path": file_path,
                    "extension": file_ext,
                    "size_kb": round(file_size / 1024, 2)
                })
        
        # Determine output directory
        if valid_files:
            primary_file = valid_files[0]
            output_directory = os.path.dirname(primary_file)
        else:
            # Create temp output directory if no input files
            temp_dir = Path("temp_outputs")
            output_directory = str(temp_dir / "generated_outputs")
            os.makedirs(output_directory, exist_ok=True)

        # Enhanced prompt with more context and flexibility
        code_prompt = f"""You are an expert Python programmer specializing in bioinformatics and data processing.
Generate a complete, executable Python script to accomplish the following task.

**TASK DESCRIPTION:**
{task_description}

**INPUT FILES:**
{json.dumps(file_info, indent=2) if file_info else "No input files provided"}

**OUTPUT DIRECTORY:**
{output_directory}

**REQUIREMENTS:**
1. Generate PRODUCTION-READY code with proper error handling
2. If input files are provided, read and process them appropriately
3. Save ALL output files to: {output_directory}
   - Use descriptive filenames (e.g., 'mutant_A12R.fasta', 'analysis_results.csv')
   - Use os.path.join('{output_directory}', 'filename.ext') for all output paths
4. The script MUST be runnable standalone from command line
5. Include ALL necessary imports at the top
6. Add informative print statements for progress tracking
7. At the end, print a JSON summary with:
   - "success": true/false
   - "output_files": list of created file paths
   - "summary": brief description of what was done
   - "details": any relevant metrics or information

**AVAILABLE LIBRARIES:**
pandas, numpy, scikit-learn, biopython (Bio.SeqIO, Bio.Seq), json, os, shutil, pathlib, re, collections

**COMMON TASK PATTERNS:**

For SEQUENCE MUTATION:
- Read FASTA file using Bio.SeqIO
- Apply mutation (e.g., seq[position] = new_amino_acid)
- Save mutant to new FASTA file
- Example: A12R means position 12 (0-indexed: 11), Alanine → Arginine

For DATA SPLITTING:
- Use train_test_split from sklearn
- Maintain label distribution with stratify parameter
- Save as separate CSV files (train.csv, valid.csv, test.csv)

For FILE MODIFICATION:
- Read file, apply modifications, save with new name or overwrite
- Always backup important files before modification

For ANALYSIS:
- Generate statistics, visualizations, or reports
- Save results to JSON or CSV format

**CRITICAL:**
- Return ONLY raw Python code (no markdown, no ```python blocks, no explanations)
- Code must be immediately executable
- Use try-except blocks for robust error handling
- Final print must be valid JSON for parsing

**EXAMPLE FINAL OUTPUT:**
print(json.dumps({{
    "success": True,
    "output_files": ["/path/to/output1.fasta", "/path/to/output2.csv"],
    "summary": "Created A12R mutant from human insulin sequence",
    "details": {{"original_length": 110, "mutation": "A12R", "position": 12}}
}}))
"""

        # Call DeepSeek API
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system", 
                    "content": "You are an expert Python programmer. Generate clean, executable code without any markdown formatting."
                },
                {
                    "role": "user", 
                    "content": code_prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 3000  # Increased for more complex code
        }
        
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=60
        )
        
        if response.status_code != 200:
            return json.dumps({
                "success": False,
                "error": f"API error: {response.status_code} - {response.text}"
            })
        
        result = response.json()
        generated_code = result['choices'][0]['message']['content'].strip()
        
        # Clean up code (remove markdown if present)
        generated_code = re.sub(r'^```python\s*', '', generated_code)
        generated_code = re.sub(r'^```\s*', '', generated_code)
        generated_code = re.sub(r'\s*```$', '', generated_code)
        
        # Save generated code for debugging
        temp_script_name = f"generated_code_{uuid.uuid4().hex}.py"
        script_path = os.path.join(tempfile.gettempdir(), temp_script_name)
        
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(generated_code)
        
        # Execute the generated code
        process = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, 
            text=True,           
            timeout=120,
            cwd=output_directory  # Run in output directory for relative paths
        )

        if process.returncode == 0:
            # Try to parse JSON output from the script
            stdout = process.stdout.strip()
            try:
                # Look for JSON in the output
                json_match = re.search(r'\{.*\}', stdout, re.DOTALL)
                if json_match:
                    result_json = json.loads(json_match.group())
                    result_json["generated_code_path"] = script_path  # Keep for debugging
                    return json.dumps(result_json, indent=2)
                else:
                    # Fallback if no JSON found
                    return json.dumps({
                        "success": True,
                        "output": stdout,
                        "generated_code_path": script_path
                    }, indent=2)
            except json.JSONDecodeError:
                # If JSON parsing fails, return raw output
                return json.dumps({
                    "success": True,
                    "output": stdout,
                    "generated_code_path": script_path
                }, indent=2)
        else:
            return json.dumps({
                "success": False,
                "error": process.stderr,
                "stdout": process.stdout,
                "generated_code_path": script_path
            }, indent=2)
            
    except subprocess.TimeoutExpired:
        return json.dumps({
            "success": False,
            "error": "Code execution timed out (>120 seconds)",
            "generated_code_path": script_path
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "generated_code_path": script_path
        })


def process_csv_and_generate_config(csv_file: str, test_csv_file: Optional[str] = None, output_name: str = "custom_training_config", user_overrides: Optional[Dict] = None) -> str:
    try:
        df = pd.read_csv(csv_file)
        required_columns = ['aa_seq', 'label']
        
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            return f"Missing required columns: {missing}. Please ensure your CSV has 'aa_seq' and 'label' columns, then upload again."
        user_config = user_overrides or {}
        analysis = analyze_dataset_for_ai(df, test_csv_file)
        ai_config = generate_ai_training_config(analysis)
        print(ai_config)
        default_config = get_default_config(analysis)
        final_params = merge_configs(user_config, ai_config, default_config)
        config = create_comprehensive_config(csv_file, test_csv_file, final_params, analysis)
        temp_dir = Path("temp_outputs")
        sequence_dir = temp_dir / "training_configs"
        os.makedirs(sequence_dir, exist_ok=True)
        timestamp = int(time.time())
        config_path = os.path.join(sequence_dir, f"{output_name}_{timestamp}.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        message = f"Training configuration generated successfully! Config file: {config_path} The configuration is ready for use in the training interface."
        return message

    except Exception as e:
        return f"Error processing CSV: {str(e)}"

def merge_configs(user_config: dict, ai_config: dict, default_config: dict) -> dict:
    merged = default_config.copy()
    merged.update(ai_config)
    merged.update(user_config)
    return merged

def analyze_dataset_for_ai(df: pd.DataFrame, test_csv_file: Optional[str] = None) -> dict:
    """Analyze dataset to provide context for AI parameter selection"""
    
    def classify_task_heuristic(df: pd.DataFrame) -> str:
        """Classify task type based on label characteristics using heuristic rules"""
        
        label_data = df['label']
        sample_labels = label_data.head(50).tolist()  # Sample for analysis
        is_residue_level = False

        for i in range(min(10, len(df))):
            label_str = str(df.iloc[i]['label'])
            seq_len = len(df.iloc[i]['aa_seq'])

            clean_label = label_str.replace(',', '').replace(' ', '').replace('[', '').replace(']', '')

            if len(clean_label) >= seq_len * 0.8:  # Allow some tolerance
                is_residue_level = True
                break

            if ',' in label_str and len(label_str.split(',')) >= seq_len * 0.8:
                is_residue_level = True
                break

        is_regression = False

        for label in sample_labels:
            label_str = str(label)
            
            if is_residue_level:
                # For residue-level, parse the sequence of values
                if ',' in label_str:
                    values = label_str.replace('[', '').replace(']', '').split(',')
                else:
                    values = list(label_str.replace('[', '').replace(']', ''))
                
                # Check if values are continuous (floats)
                try:
                    float_values = [float(v.strip()) for v in values if v.strip()]
                    # If we have decimal numbers, it's regression
                    if any('.' in str(v) for v in values if v.strip()):
                        is_regression = True
                        break
                    # If values have wide range, might be regression
                    if len(float_values) > 0 and (max(float_values) - min(float_values) > 10):
                        is_regression = True
                        break
                except ValueError:
                    # If can't convert to float, it's classification
                    continue
            else:
                # For protein-level, check the single label value
                try:
                    float_val = float(label_str)
                    # If it's a decimal number, it's regression
                    if '.' in label_str:
                        is_regression = True
                        break
                    # If integer range is large, might be regression
                    if abs(float_val) > 10:
                        is_regression = True
                        break
                except ValueError:
                    # If can't convert to float, it's classification
                    continue
        
        # Step 3: For classification, check if it's multi-label
        is_multi_label = False
        if not is_regression and not is_residue_level:
            # Check for multi-label indicators in protein-level classification
            for label in sample_labels:
                label_str = str(label)
                if any(sep in label_str for sep in [',', ';', '|', '&', '+']):
                    is_multi_label = True
                    break
                words = label_str.split()
                if len(words) > 1 and not any(char.isdigit() for char in label_str):
                    is_multi_label = True
                    break
        
        # Step 4: Return the classification
        if is_residue_level:
            if is_regression:
                return "residue_regression"
            else:
                return "residue_single_label_classification"
        else:
            if is_regression:
                return "regression"
            elif is_multi_label:
                return "multi_label_classification"
            else:
                return "single_label_classification"

    label_data = df['label']

    task_type = classify_task_heuristic(df)
    
    analysis = {
        "total_samples": int(len(df)),
        "unique_labels": int(df['label'].nunique()),
        "label_distribution": {str(k): int(v) for k, v in df['label'].value_counts().to_dict().items()},
        "sequence_stats": {
            "mean_length": float(df['aa_seq'].str.len().mean()),
            "min_length": int(df['aa_seq'].str.len().min()),
            "max_length": int(df['aa_seq'].str.len().max()),
            "std_length": float(df['aa_seq'].str.len().std())
        },
        "data_type": task_type,  # Heuristic-determined task type
        "class_balance": "balanced" if df['label'].value_counts().std() < df['label'].value_counts().mean() * 0.5 else "imbalanced"
    }
   
    if test_csv_file and os.path.exists(test_csv_file):
        test_df = pd.read_csv(test_csv_file)
        analysis["test_samples"] = int(len(test_df))
        analysis["has_test_set"] = True
    else:
        analysis["has_test_set"] = False
   
    return analysis

def convert_to_serializable(obj):
    """Convert pandas/numpy types to JSON serializable types"""
    import numpy as np
    
    if isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj

def generate_ai_training_config(analysis: dict) -> dict:
    """Use DeepSeek AI to generate optimal training configuration"""
    try:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            return get_default_config(analysis)
        data = json.load(open("src/constant.json"))

        prompt = f"""You are VenusAgent, an expert in protein machine learning. Generate optimal training configuration following these STRICT rules:

            RULE 1 - USER REQUIREMENTS ARE ABSOLUTE LAW:
            If user mentions ANY specific requirement (model name, training method, etc.), you MUST use exactly what they specified. No exceptions, no "better alternatives".

            RULE 2 - EFFICIENCY FIRST FOR UNSPECIFIED PARAMETERS:
            For parameters not specified by user, choose the most efficient option that maintains good performance.

            RULE 3 - DATASET-DRIVEN OPTIMIZATION:
            - Small dataset (<1000): Use smaller models, lower learning rates (1e-5), more epochs (50-100)
            - Large dataset (>10000): Use larger models, higher learning rates (5e-5), fewer epochs (10-30)
            - Long sequences (>500): Smaller batch sizes (8-16), use gradient accumulation
            - Imbalanced data: Monitor F1 score, use patience=10

            Dataset Analysis:
            - Samples: {analysis['total_samples']}
            - Type: {analysis['data_type']}
            - Labels: {analysis['unique_labels']}
            - Balance: {analysis['class_balance']}
            - Seq length: {analysis['sequence_stats']['mean_length']:.0f} (min:{analysis['sequence_stats']['min_length']}, max:{analysis['sequence_stats']['max_length']})
            - Test set: {analysis['has_test_set']}

            Available options:
            - Models: {list(data["plm_models"].keys())}
            - Training: ["full", "freeze", "ses-adapter", "plm-lora", "plm-qlora", "plm-adalora", "plm-dora", "plm-ia3"]
            - Problem: ["single_label_classification", "multi_label_classification", "regression", "residue_single_label_classification", "residue_regression"]

            EXAMPLES:
            - User wants "ProtT5 + QLoRA" → Must use "prot-t5-xl" + "plm-qlora" (no alternatives!)
            - No user preference → Choose most efficient for dataset size

            Return ONLY valid JSON:
            {{
            "plm_model": "exact_name_from_available_options",
            "training_method": "exact_method_from_list",
            "problem_type": "auto_detected_from_data",
            "learning_rate": optimal_number,
            "num_epochs": optimal_number,
            "batch_size": optimal_number,
            "max_seq_len": {min(2048, int(analysis['sequence_stats']['max_length'] * 1.1))},
            "patience": 1-50,
            "pooling_method": ["mean", "attention1d", "light_attention"],
            "scheduler": ["linear", "cosine", "step", None],
            "monitored_metrics": ["accuracy", "recall", "precision", "f1", "mcc", "auroc", "aupr", "f1_max", "f1_positive", "f1_negative", "spearman_corr", "mse"],
            "monitored_strategy": ["max", "min"],
            "gradient_accumulation_steps": 1-32,
            "warmup_steps": 0-1000,
            "max_grad_norm": 0.1-10.0,
            "num_workers": 0-16,
            "reasoning": "explain your choices"
        }}

        Consider:
        - Small datasets (<1000): lower learning rate, more epochs, early stopping
        - Large datasets (>10000): higher learning rate, fewer epochs
        - Long sequences (>500): smaller batch size, gradient accumulation
        - Imbalanced classes: appropriate metrics (f1, mcc)
        - Regression tasks: use spearman_corr with min strategy
        
        You should return the entire path, start with temp_outputs
        """

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are a protein machine learning expert. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 800
        }
        
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        
        return get_default_config(analysis)
        
    except Exception as e:
        print(f"AI config generation failed: {e}")
        return get_default_config(analysis)

def get_default_config(analysis: dict) -> dict:
    """Fallback default configuration"""
    is_regression = analysis['data_type'] == 'regression'
    return {
        "plm_model": "ESM2-8M",
        "problem_type": analysis['data_type'],
        "training_method": "freeze",
        "learning_rate": 5e-4,
        "num_epochs": 20,
        "batch_size": 16,
        "max_seq_len": min(512, int(analysis['sequence_stats']['max_length'] * 1.2)),
        "patience": 10,
        "pooling_method": "mean",
        "scheduler": None,
        "monitored_metrics": "spearman_corr" if is_regression else "accuracy",
        "monitored_strategy": "max",
        "gradient_accumulation_steps": 1,
        "warmup_steps": 0,
        "max_grad_norm": 1.0,
        "num_workers": 1
    }

def create_comprehensive_config(csv_file: str, test_csv_file: Optional[str], params: dict, analysis: dict) -> dict:
    """Create complete training configuration matching 1.py requirements"""
    is_regression = analysis['data_type'] == 'regression'
    dataset_directory = os.path.dirname(csv_file)
    config = {
        # Dataset configuration
        "dataset_selection": "Custom Dataset",
        "dataset_custom": dataset_directory,
        "problem_type": params["problem_type"],
        "num_labels": 1 if is_regression else analysis['unique_labels'],
        "metrics": ["mse", "spearman_corr"] if is_regression else ["accuracy", "mcc", "f1"],
        
        # Model and training method from final params
        "plm_model": params["plm_model"],
        "training_method": params["training_method"],
        "pooling_method": params["pooling_method"],
        
        # Training parameters from final params
        "learning_rate": float(params["learning_rate"]),
        "num_epochs": int(params["num_epochs"]),
        "max_seq_len": int(params["max_seq_len"]),
        "patience": int(params["patience"]),
        "batch_size": int(params["batch_size"]),
        
        # Advanced parameters
        "gradient_accumulation_steps": int(params.get("gradient_accumulation_steps", 1)),
        "warmup_steps": int(params.get("warmup_steps", 0)),
        "scheduler": params.get("scheduler"),
        "max_grad_norm": float(params.get("max_grad_norm", 1.0)),
        "num_workers": int(params.get("num_workers", 1)),
        
        # Monitoring
        "monitored_metrics": params["monitored_metrics"],
        "monitored_strategy": params["monitored_strategy"],
        
        # Output
        "output_model_name": f"model_{Path(csv_file).stem}",
        "output_dir": "./outputs",
        
        "wandb_enabled": False,
    }
    
    if test_csv_file:
        config["test_file"] = test_csv_file
    
    # Final conversion to ensure everything is serializable
    return convert_to_serializable(config)