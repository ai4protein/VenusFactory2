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
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv
from gradio_client import Client, handle_file
import pandas as pd
from langchain.tools import tool
from pydantic import BaseModel, Field, validator

load_dotenv()

class ZeroShotSequenceInput(BaseModel):
    """Input for zero-shot sequence mutation prediction"""
    sequence: Optional[str] = Field(None, description="Protein sequence in single letter amino acid code")
    fasta_file: Optional[str] = Field(None, description="Path to FASTA file")
    model_name: str = Field(default="ESM2-650M", description="Model name: ESM-1v, ESM2-650M, ESM-1b")

class ZeroShotStructureInput(BaseModel):
    """Input for zero-shot structure mutation prediction"""
    structure_file: str = Field(..., description="Path to PDB structure file")
    model_name: str = Field(default="ESM-IF1", description="Model name: SaProt, ProtSSN, ESM-IF1, MIF-ST, ProSST-2048")

class FunctionPredictionInput(BaseModel):
    """Input for protein function prediction"""
    sequence: Optional[str] = Field(None, description="Protein sequence in single letter amino acid code")
    fasta_file: Optional[str] = Field(None, description="Path to FASTA file")
    model_name: str = Field(default="ESM2-650M", description="Model name for function prediction")
    task: str = Field(default="Solubility", description="Task: Solubility, Localization, Metal ion binding, Stability, Sorting signal, Optimum temperature")

class ResidueFunctionPredictionInput(BaseModel):
    """Input for functional residue prediction"""
    sequence: Optional[str] = Field(None, description="Protein sequence in single letter amino acid code")
    fasta_file: Optional[str] = Field(None, description="Path to FASTA file")
    model_name: str = Field(default="ESM2-650M", description="Model name for function prediction")
    task: str = Field(default="Activate", description="Task: Activate, Binding, Evolutionary, Motif")

class InterProQueryInput(BaseModel):
    """Input for InterPro database query"""
    uniprot_id: str = Field(..., description="UniProt ID for protein function query")

class UniProtQueryInput(BaseModel):
    """Input for UniProt database query"""
    uniprot_id: str = Field(..., description="UniProt ID for protein sequence query")

class PDBQueryInput(BaseModel):
    """Input for PDB database query"""
    pdb_id: str = Field(..., description="PDB ID for protein sequence query")

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

# Langchain Tools
@tool("zero_shot_sequence_prediction", args_schema=ZeroShotSequenceInput)
def zero_shot_sequence_prediction_tool(sequence: Optional[str] = None, fasta_file: Optional[str] = None, model_name: str = "ESM2-650M") -> str:
    """Predict beneficial mutations using sequence-based zero-shot models. Use for mutation prediction with protein sequences."""
    try:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        
        if fasta_file:
            if not os.path.exists(fasta_file):
                return f"Error: FASTA file not found at path: {fasta_file}"
            return call_zero_shot_sequence_prediction_from_file(fasta_file, model_name, api_key)
        elif sequence:
            return call_zero_shot_sequence_prediction(sequence, model_name, api_key)
        else:
            return "Error: Either sequence or fasta_file must be provided"
    except Exception as e:
        return f"Zero-shot sequence prediction error: {str(e)}"

@tool("zero_shot_structure_prediction", args_schema=ZeroShotStructureInput)
def zero_shot_structure_prediction_tool(structure_file: str, model_name: str = "ESM-IF1") -> str:
    """Predict beneficial mutations using structure-based zero-shot models. Use for mutation prediction with PDB structure files."""
    try:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not os.path.exists(structure_file):
            return f"Error: Structure file not found at path: {structure_file}"
        return call_zero_shot_structure_prediction_from_file(structure_file, model_name, api_key)
    except Exception as e:
        return f"Zero-shot structure prediction error: {str(e)}"

@tool("protein_function_prediction", args_schema=FunctionPredictionInput)
def protein_function_prediction_tool(sequence: Optional[str] = None, fasta_file: Optional[str] = None, model_name: str = "ESM2-650M", task: str = "Solubility") -> str:
    """Predict protein functions like solubility, localization, metal ion binding, stability, sorting signal, and optimum temperature."""
    try:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if fasta_file and os.path.exists(fasta_file):
            return call_protein_function_prediction_from_file(fasta_file, model_name, task, api_key)
        elif sequence:
            return call_protein_function_prediction(sequence, model_name, task, api_key)
        else:
            return "Error: Either sequence or fasta_file must be provided"
    except Exception as e:
        return f"Function protein prediction error: {str(e)}"

@tool("functional_residue_prediction", args_schema=ResidueFunctionPredictionInput)
def functional_residue_prediction_tool(sequence: Optional[str] = None, fasta_file: Optional[str] = None, model_name: str = "ESM2-650M", task: str = "Activate") -> str:
    try:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if fasta_file and os.path.exists(fasta_file):
            return call_functional_residue_prediction_from_file(fasta_file, model_name, task, api_key)
        elif sequence:
            return call_functional_residue_prediction(sequence, model_name, task, api_key)
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

@tool("PDB_query", args_schema=PDBQueryInput)
def pdb_query_tool(pdb_id: str) -> str:
    """Query PDB database for protein sequence"""
    try:
        return get_pdb_sequence(pdb_id)
    except Exception as e:
        return f"PDB query error: {str(e)}"

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
        api_key = os.getenv("DEEPSEEK_API_KEY")
        
        if fasta_file:
            if not os.path.exists(fasta_file):
                return f"Error: FASTA file not found at path: {fasta_file}"
            return call_protein_properties_prediction_from_file(fasta_file, task_name, api_key)
        elif sequence:
            return call_protein_properties_prediction(sequence, task_name, api_key)
        else:
            return f"Error: Structure file not found at path: {fasta_file}"
        
    except Exception as e:
        return f"Zero-shot structure prediction error: {str(e)}"

@tool("ai_code_execution", args_schema=CodeExecutionInput)
def ai_code_execution_tool(task_description: str, input_files: List[str] = []) -> str:
    """Generate and execute Python code based on task description. Use for data processing, splitting, analysis tasks."""
    try:
        return generate_and_execute_code(task_description, input_files)
    except Exception as e:
        return f"Code execution error: {str(e)}"

def call_zero_shot_sequence_prediction(sequence: str, model_name: str = "ESM2-650M", api_key: str = None) -> str:
    """Call VenusFactory zero-shot sequence-based mutation prediction API"""
    try:
        temp_fasta = tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False)
        temp_fasta.write(f">temp_sequence\n{sequence}\n")
        temp_fasta.close()
        
        client = Client("http://localhost:7860/")
        result = client.predict(
            function_selection="Activity",
            file_obj=handle_file(temp_fasta.name),
            enable_ai=False,
            ai_model="DeepSeek",
            user_api_key=api_key,
            model_name=model_name,
            api_name="/handle_mutation_prediction_base"
        )
        os.unlink(temp_fasta.name)
        
        return result[2]
    except Exception as e:
        return f"Zero-shot sequence prediction error: {str(e)}"

def call_zero_shot_sequence_prediction_from_file(fasta_file: str, model_name: str = "ESM2-650M", api_key: str = None) -> str:
    """Call VenusFactory zero-shot sequence prediction API using uploaded FASTA file"""
    try:
        client = Client("http://localhost:7860/")
        result = client.predict(
            function_selection="Activity",
            file_obj=handle_file(fasta_file),
            enable_ai=False,
            ai_model="DeepSeek",
            user_api_key=api_key,
            model_name=model_name,
            api_name="/handle_mutation_prediction_base"
        )
        return result[2]
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
        return result[2]
    except Exception as e:
        return f"Zero-shot structure prediction error: {str(e)}"

def call_protein_function_prediction_from_file(fasta_file: str, model_name: str = "ProtT5-xl-uniref50", task: str = "Solubility", api_key: str = None) -> str:
    """Call VenusFactory protein function prediction API using uploaded FASTA file"""
    try:
        dataset_mapping = {
            "Solubility": ["DeepSol", "DeepSoluE", "ProtSolM"],
            "Localization": ["DeepLocBinary", "DeepLocMulti"],
            "Metal ion binding": ["MetalIonBinding"],
            "Stability": ["Thermostability"],
            "Sorting signal": ["SortingSignal"],
            "Optimum temperature": ["DeepET_Topt"]
        }
        datasets = dataset_mapping.get(task, ["DeepSol"])
        
        client = Client("http://localhost:7860/")
        result = client.predict(
            task=task, 
            fasta_file=handle_file(fasta_file),
            model_name=model_name,
            datasets=datasets,
            enable_ai=True,
            ai_model="DeepSeek",
            user_api_key=api_key,
            api_name="/handle_protein_function_prediction_chat"
        )
        return result[1]
    except Exception as e:
        return f"Function prediction error: {str(e)}"

def call_protein_function_prediction(sequence: str, model_name: str = "ProtT5-xl-uniref50", task: str = "Solubility", api_key: str = None) -> str:
    """Call VenusFactory protein function prediction API with sequence"""
    try:
        temp_fasta = tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False)
        temp_fasta.write(f">temp_sequence\n{sequence}\n")
        temp_fasta.close()

        dataset_mapping = {
            "Solubility": ["DeepSol", "DeepSoluE", "ProtSolM"],
            "Localization": ["DeepLocBinary", "DeepLocMulti"],
            "Metal ion binding": ["MetalIonBinding"],
            "Stability": ["Thermostability"],
            "Sorting signal": ["SortingSignal"],
            "Optimum temperature": ["DeepET_Topt"]
        }
        datasets = dataset_mapping.get(task, ["DeepSol"])
        
        client = Client("http://localhost:7860/")
        result = client.predict(
            task=task, 
            fasta_file=handle_file(temp_fasta.name),
            model_name=model_name,
            datasets=datasets,
            enable_ai=True,
            ai_model="DeepSeek",
            user_api_key=api_key,
            api_name="/handle_protein_function_prediction_chat"
        )
        os.unlink(temp_fasta.name)
        
        return result[1]
    except Exception as e:
        return f"Function prediction error: {str(e)}"

def call_functional_residue_prediction_from_file(fasta_file: str, model_name: str = "ESM2-650M", task: str = "Activity", api_key: str = None) -> str:
    """Call VenusFactory functional residue prediction API using uploaded FASTA file"""
    try:

        client = Client("http://localhost:7860/")
        result = client.predict(
            task=task, 
            fasta_file=handle_file(fasta_file),
            enable_ai=True,
            ai_model="DeepSeek",
            user_api_key=api_key,
            model_name=model_name,
            api_name="/handle_protein_residue_function_prediction_chat"
        )
        return result[1]
    except Exception as e:
        return f"Functional residue prediction error: {str(e)}"

def call_functional_residue_prediction(sequence: str, model_name: str = "ESM2-650M", task: str = "Activity", api_key: str = None) -> str:
    """Call VenusFactory functional residue prediction API using uploaded FASTA file"""
    try:
        temp_fasta = tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False)
        temp_fasta.write(f">temp_sequence\n{sequence}\n")
        temp_fasta.close()

        client = Client("http://localhost:7860/")
        result = client.predict(
            task=task, 
            fasta_file=handle_file(temp_fasta.name),
            enable_ai=True,
            ai_model="DeepSeek",
            user_api_key=api_key,
            model_name=model_name,
            api_name="/handle_protein_residue_function_prediction_chat"
        )
        os.unlink(temp_fasta.name)
        return result[1]
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

def get_pdb_sequence(pdb_id):
    """
    Fetches protein sequences for a single PDB ID.
    """
    pdb_id = pdb_id.upper()
    url = f"https://www.rcsb.org/fasta/entry/{pdb_id}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        fasta_text = response.text
        
        sequences = []
        current_chain = None
        current_sequence = ""
        
        for line in fasta_text.strip().split('\n'):
            if line.startswith('>'):
                # Save previous sequence if exists
                if current_chain and current_sequence:
                    sequences.append({
                        "chain": current_chain,
                        "sequence": current_sequence
                    })
                
                # Parse chain info from header
                # Example: >1ABC_1|Chain A|LYSOZYME|Gallus gallus (9031)
                parts = line.split('|')
                chain_info = parts[0].replace('>', '')
                current_chain = parts[1] if len(parts) > 1 else chain_info
                current_sequence = ""
            else:
                current_sequence += line
        
        # Add last sequence
        if current_chain and current_sequence:
            sequences.append({
                "chain": current_chain,
                "sequence": current_sequence
            })
        
        return {
            "success": True,
            "pdb_id": pdb_id,
            "sequences": sequences
        }
        
    except Exception as e:
        return {
            "success": False,
            "pdb_id": pdb_id,
            "error_message": f"Error fetching sequence for {pdb_id}: {str(e)}"
        }

def call_protein_properties_prediction_from_file(fasta_file: str, task_name: str, api_key: str = None) -> str:
    try:
        client = Client("http://localhost:7860/")
        result = client.predict(
            task=task_name,
            file_obj=handle_file(fasta_file),
            api_name="/handle_protein_properties_generation"
        )
        return result[1]
    except Exception as e:
        return f"Protein properties prediction error: {str(e)}"

def call_protein_properties_prediction(sequence: str, task_name, api_key: str = None) -> str:
    try:
        temp_fasta = tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False)
        temp_fasta.write(f">temp_sequence\n{sequence}\n")
        temp_fasta.close()

        client = Client("http://localhost:7860/")
        result = client.predict(
            task=task_name,
            file_obj=handle_file(temp_fasta.name),
            api_name="/handle_protein_properties_generation"
        )
        os.unlink(temp_fasta.name)
        return result[1]
    except Exception as e:
        return f"Protein properties prediction error: {str(e)}"

def generate_and_execute_code(task_description: str, input_files: List[str]) -> str:
    script_path = None 
    try:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        valid_files = []
        for file_path in input_files:
            if os.path.exists(file_path):
                valid_files.append(file_path)
            else:
                return f"Error, file not found: {file_path}"
        
        if not valid_files:
            return "Error, there are no valuable input file。"

        primary_file = valid_files[0]

        output_directory = os.path.dirname(primary_file)

        code_prompt = f"""
        You are an expert Python programmer specializing in data processing.
        Generate a complete, executable Python script to accomplish the following task.

        Task: {task_description}
        Input File(s): {valid_files}
        Output Directory: '{output_directory}'

        Requirements:
        1. Read the primary input file: '{primary_file}'.
        2. Perform the required data processing.
        3. Save any output files to the specified output directory: '{output_directory}'. For example, use os.path.join('{output_directory}', 'train.csv'). 
        4. The script MUST be runnable from the command line and contain all necessary imports.
        5. Save the file names as train.csv, valid.csv, and test.csv in the same folder as before
        6. Print a final summary message to the console indicating success and listing the created files. This print message will be the result.

        Available libraries: pandas, numpy, scikit-learn, json, os, shutil, time.
        Generate only the raw Python code, without any markdown formatting (like ```python) or explanations.
        """

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": code_prompt}],
            "temperature": 0.1,
            "max_tokens": 2000
        }
        
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=60
        )
        
        if response.status_code != 200:
            return f"Generate code error: {response.status_code} - {response.text}"
        
        result = response.json()
        generated_code = result['choices'][0]['message']['content'].strip()
        temp_script_name = f"generated_code_{uuid.uuid4().hex}.py"
        script_path = os.path.join(tempfile.gettempdir(), temp_script_name)
        
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(generated_code)

        process = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, 
            text=True,           
            timeout=120        
        )

        if process.returncode == 0:
            return f"Success:\n---\n{process.stdout}"
        else:
            return f"Error:\n---\n{process.stderr}"
            
    except Exception as e:
        return f"Function Error: {str(e)}"
    finally:
        if script_path and os.path.exists(script_path):
            os.remove(script_path)

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
