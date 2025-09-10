import gradio as gr
import pandas as pd
import os
import sys
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Dict, Any, List, Generator, Optional, Tuple, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import requests
from dataclasses import dataclass
import re
import json
from .utils.paste_content_handler import process_pasted_content
from dotenv import load_dotenv
load_dotenv()

MODEL_MAPPING_ZERO_SHOT = {
    "ESM2-650M": "esm2", 
    "ESM-1b": "esm1b",
    "ESM-IF1": "esmif1", 
    "ESM-1v": "esm1v",
    "SaProt": "saprot", 
    "MIF-ST": "mifst", 
    "ProSST-2048": "prosst", 
    "ProtSSN": "protssn"
}

DATASET_MAPPING_ZERO_SHOT = [
    "Activity",
    "Binding",
    "Expression",
    "Organismal Fitness",
    "Stability"
]

MODEL_MAPPING_FUNCTION = {
    "ESM2-650M": "esm2", 
    "Ankh-large": "ankh",
    "ProtBert": "protbert", 
    "ProtT5-xl-uniref50": "prott5",
}

MODEL_ADAPTER_MAPPING_FUNCTION = {
    "esm2": "esm2_t33_650M_UR50D", 
    "ankh": "ankh-large",
    "protbert": "prot_bert", 
    "prott5": "prot_t5_xl_uniref50",
}

DATASET_MAPPING_FUNCTION = {
    "Solubility": ["DeepSol", "DeepSoluE", "ProtSolM"],
    "Localization": ["DeepLocBinary", "DeepLocMulti"],
    "Metal ion binding": ["MetalIonBinding"], 
    "Stability": ["Thermostability"],
    "Sortingsignal": ["SortingSignal"], 
    "Optimum temperature": ["DeepET_Topt"]
}

LABEL_MAPPING_FUNCTION = {
    "Solubility": ["Insoluble", "Soluble"],
    "DeepLocBinary": ["Membrane", "Soluble"],
    "DeepLocMulti": [
        "Cytoplasm", "Nucleus", "Extracellular", "Mitochondrion", "Cell membrane",
        "Endoplasmic reticulum", "Plastid", "Golgi apparatus", "Lysosome/Vacuole", "Peroxisome"
    ],
    "Metal ion binding": ["Non-binding", "Binding"],
    "Sortingsignal": ['No signal', "CH", 'GPI', "MT", "NES", "NLS", "PTS", "SP", "TM", "TH"]    
}

RESIDUE_MAPPING_FUNCTION = {
    "Activity Site": ["VenusX_Res_Act_MP90"],
    "Binding Site": ["VenusX_Res_BindI_MP90"],
    "Conserved Site": ["VenusX_Res_Evo_MP90"],
    "Motif": ["VenusX_Res_Motif_MP90"]
}

COLOR_MAP_FUNCTION = {
    "Soluble": "#3B82F6", "Insoluble": "#EF4444", "Membrane": "#F59E0B", 
    "Cytoplasm": "#10B981", "Nucleus": "#8B5CF6", "Extracellular": "#F97316", 
    "Mitochondrion": "#EC4899", "Cell membrane": "#6B7280", "Endoplasmic reticulum": "#84CC16", 
    "Plastid": "#06B6D4", "Golgi apparatus": "#A78BFA", "Lysosome/Vacuole": "#FBBF24", 
    "Peroxisome": "#34D399", "Binding": "#3B82F6", "Non-binding": "#EF4444", 
    "Signal": "#3B82F6", "No signal": "#EF4444", "Default": "#9CA3AF"
}

PROTEIN_PROPERTIES_FUNCTION = {
    "Physical and chemical properties",
    "Relative solvent accessible surface area (PDB only)",
    "SASA value (PDB only)",
    "Secondary structure (PDB only)"
}
PROTEIN_PROPERTIES_MAP_FUNCTION = {
    "Physical and chemical properties" : "calculate_physchem",
    "Relative solvent accessible surface area (PDB only)" : "calculate_rsa",
    "SASA value (PDB only)": "calculate_sasa",
    "Secondary structure (PDB only)" : "calculate_secondary_structure"
}

REGRESSION_TASKS_FUNCTION = ["Stability", "Optimum temperature"]
REGRESSION_TASKS_FUNCTION_MAX_MIN = {
    "Stability": [40.1995166, 66.8968874],
    "Optimum temperature": [2, 120]
}

DATASET_TO_TASK_MAP = {
    dataset: task 
    for task, datasets in DATASET_MAPPING_FUNCTION.items() 
    for dataset in datasets
}


AI_MODELS = {
    "DeepSeek": {
        "api_base": "https://api.deepseek.com/v1", 
        "model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY"
    },
    "ChatGPT": {
        "api_base": "https://api.openai.com/v1",
        "model": "gpt-4o-mini", 
        "env_key": None
    },
    "Gemini": {
        "api_base": "https://generativelanguage.googleapis.com/v1beta",
        "model": "gemini-1.5-flash",
        "env_key": None
    }
}

@dataclass
class AIConfig:
    """Configuration for AI API calls."""
    api_key: str
    ai_model_name: str
    api_base: str
    model: str

def get_api_key(ai_provider: str, user_input_key: str = "") -> Optional[str]:
    """Get API key based on provider and user input."""
    model_config = AI_MODELS.get(ai_provider, {})
    env_key = model_config.get("env_key")
    
    if env_key:
        env_api_key = os.getenv(env_key)
        if env_api_key and env_api_key.strip():
            return env_api_key.strip()
        if user_input_key and user_input_key.strip():
            return user_input_key.strip()
        return None
    
    if user_input_key and user_input_key.strip():
        return user_input_key.strip()
    return None

def call_ai_api(config: AIConfig, prompt: str) -> str:
    """Make API call to AI service."""
    if config.ai_model_name == "ChatGPT":
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert protein scientist. Provide clear, structured, and insightful analysis based on the data provided. Do not ask interactive questions."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 2000
        }
        endpoint = f"{config.api_base}/chat/completions"
        
    elif config.ai_model_name == "Gemini":
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "contents": [{
                "parts": [{
                    "text": f"You are an expert protein scientist. {prompt}"
                }]
            }],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 2000
            }
        }
        endpoint = f"{config.api_base}/models/{config.model}:generateContent?key={config.api_key}"
        
    else:
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert protein scientist. Provide clear, structured, and insightful analysis based on the data provided. Do not ask interactive questions."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 2000
        }
        endpoint = f"{config.api_base}/chat/completions"
    
    try:
        response = requests.post(
            endpoint,
            headers=headers,
            json=data,
            timeout=60
        )
        response.raise_for_status()
        response_json = response.json()
        
        if config.ai_model_name == "Gemini":
            if "candidates" in response_json and len(response_json["candidates"]) > 0:
                return response_json["candidates"][0]["content"]["parts"][0]["text"]
            else:
                return "❌ Gemini API returned empty response"
        else:
            return response_json["choices"][0]["message"]["content"]
            
    except requests.exceptions.RequestException as e:
        return f"❌ Network error: {str(e)}"
    except KeyError as e:
        return f"❌ API response format error: {str(e)}"
    except Exception as e:
        return f"❌ API call failed: {str(e)}"
    
def check_ai_config_status(ai_provider: str, user_api_key: str) -> tuple[bool, str]:
    """Check AI configuration status and return validity and message."""
    model_config = AI_MODELS.get(ai_provider, {})
    env_key = model_config.get("env_key")
    
    if env_key:
        env_api_key = os.getenv(env_key)
        if env_api_key and env_api_key.strip():
            return True, "✓ Using the Provided API Key"
        elif user_api_key and user_api_key.strip():
            return True, "✓ The server will not save your API Key"
        else:
            return False, "⚠ No API Key found in .env file"
    else:
        if user_api_key and user_api_key.strip():
            return True, "✓ Manual API Key provided"
        else:
            return False, "⚠ Manual API Key required"  

def on_ai_model_change(ai_provider: str) -> tuple:
    """Handle AI model selection change."""
    model_config = AI_MODELS.get(ai_provider, {})
    env_key = model_config.get("env_key")
    
    if env_key:
        env_api_key = os.getenv(env_key)
        if env_api_key and env_api_key.strip():
            status_msg = "✓ Using API Key from .env file"
            show_input = False
            placeholder = ""
        else:
            status_msg = "⚠ No API Key available. Please enter manually below:"
            show_input = True
            placeholder = "Enter your DeepSeek API Key"
    else:
        status_msg = f"⚠ Manual API Key required for {ai_provider}"
        show_input = True
        if ai_provider == "ChatGPT":
            placeholder = "Enter your OpenAI API Key (sk-...)"
        elif ai_provider == "Gemini":
            placeholder = "Enter your Google AI API Key"
        else:
            placeholder = f"Enter your {ai_provider} API Key"
    
    return (
        gr.update(visible=show_input, placeholder=placeholder, value=""),
        gr.update(value=status_msg)
    )


def generate_expert_analysis_prompt_residue(results_df: pd.DataFrame, task: str) -> str:
    """
    Generates a prompt for an LLM to analyze residue-level prediction results.
    """
    prompt = f"""
        You are a senior protein biochemist and structural biologist with deep expertise in enzyme mechanisms and protein engineering. 
        A colleague has just brought you the following prediction results identifying key functional residues, specifically for the task '{task}'.
        Please analyze these residue-specific results from a practical, lab-focused perspective:

        {results_df.to_string(index=False)}

        Provide a concise, practical analysis in a single paragraph, focusing ONLY on:
        1. The predicted functional role of the individual high-scoring residues (e.g., activity site, binding site, conserved site, and motif).
        2. The significance of the confidence scores (i.e., how certain are we that a specific residue is critical for the function).
        3. Specific experimental validation steps, especially which residues are the top candidates for site-directed mutagenesis to confirm their roles.
        4. Any notable patterns or outliers (e.g., a cluster of high-scoring residues forming a potential active site, or a surprising critical residue far from the expected functional region).
        5. Do not output formatted content, just one paragraph is sufficient
        Use simple, direct language that a bench scientist would use. Do NOT mention:
        - The model used or its training data.
        - Any computational or implementation details.
        - Complex statistical concepts beyond confidence.

        Keep your response under 200 words and speak as if you are discussing next steps for an experiment with a colleague.
        """
    return prompt

def generate_expert_analysis_prompt(results_df: pd.DataFrame, task: str) -> str:
    protein_count = len(results_df)

    prompt = f"""
        You are a senior protein biochemist with extensive laboratory experience. A colleague has just shown you protein function prediction results for the task '{task}'. 
        Please analyze these results from a practical biologist's perspective:
        {results_df.to_string(index=False)}
        Provide a concise, practical analysis focusing ONLY on:
        0. The task '{task}' with "Stability" and "Optimum temperature" are regression task
        1. What the prediction results mean for each protein
        2. The biological significance of the confidence scores
        3. Practical experimental recommendations based on these predictions
        4. Any notable patterns or outliers in the results
        5. Do not output formatted content, just one paragraph is sufficient
        Use simple, clear language that a bench scientist would appreciate. Do NOT mention:
        - Training datasets or models
        - Technical implementation details  
        - Computational methods
        - Statistical concepts beyond confidence scores
        Keep your response under 200 words and speak as if you're having a conversation with a colleague in the lab.
        """
    return prompt

def format_expert_response(ai_response: str) -> str:
    """Format AI response as HTML with expert avatar and speech bubble."""
    escaped_response = ai_response.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')
    
    html = f"""
    <div style="min-height: 150px; padding: 10px; display: flex; align-items: flex-end; font-family: Arial, sans-serif;">
        <div style="margin-right: 15px; text-align: center;">
            <div style="width: 60px; height: 60px; background-color: #4A90E2; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 24px; color: white; border: 3px solid #357ABD;">
                👨‍🔬
            </div>
            <div style="font-size: 12px; color: #666; margin-top: 5px; font-weight: bold;">
                Expert
            </div>
        </div>
        <div style="flex: 1; position: relative;">
            <div style="background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 15px; padding: 15px; position: relative; max-width: 90%; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="position: absolute; left: -10px; bottom: 20px; width: 0; height: 0; border-top: 10px solid transparent; border-bottom: 10px solid transparent; border-right: 10px solid #f8f9fa;"></div>
                <div style="position: absolute; left: -11px; bottom: 20px; width: 0; height: 0; border-top: 10px solid transparent; border-bottom: 10px solid transparent; border-right: 10px solid #dee2e6;"></div>
                <div style="font-size: 15px; line-height: 1.4; color: #333;">
                    {escaped_response}
                </div>
            </div>
        </div>
    </div>
    """
    return html

def generate_mutation_ai_prompt(results_df: pd.DataFrame, model_name: str, function_selection: str) -> str:
    """Generate AI prompt for mutation analysis."""
    if 'Mutant' not in results_df.columns:
        return "Error: 'Mutant' column not found."
    
    score_col = next(
        (col for col in results_df.columns if 'Prediction Rank' in col.lower()), 
        results_df.columns[1] if len(results_df.columns) > 1 else None
    )
    
    if not score_col:
        return "Error: Score column not found."

    num_rows = len(results_df)
    top_count = max(20, int(num_rows * 0.05)) if num_rows >= 5 else num_rows
    top_mutations_str = results_df.head(top_count)[['Mutant', score_col]].to_string(index=False)
    
    return f"""
        Please act as an expert protein engineer and analyze the following mutation prediction results generated by the '{model_name}' model for the function '{function_selection}'.
        A deep mutational scan was performed. The results are sorted from most beneficial to least beneficial based on the '{score_col}'. Below are the top 5% of mutations.

        ### Top 5% Predicted Mutations (Potentially Most Beneficial):
        ```
        {top_mutations_str}
        ```

        ### Your Analysis Task:
        Based on this data, provide a structured scientific analysis report that includes the following sections:
        1. Executive Summary: Briefly summarize the key findings. Are there clear hotspot regions?
        2. Analysis of Beneficial Mutations: Discuss the top mutations and their potential biochemical impact on '{function_selection}'.
        3. Analysis of Detrimental Mutations: What do the most harmful mutations suggest about critical residues?
        4. Recommendations for Experimentation: Suggest 3-5 specific mutations for validation, with justifications.
        5. Do not output formatted content, just one paragraph is sufficient
        Provide a concise, clear, and insightful report in a professional scientific tone, summarize the above content into 1-2 paragraphs and output unformatted content.
        """

def generate_ai_summary_prompt(results_df: pd.DataFrame, task: str, model: str) -> str:
    """Generate AI prompt for function prediction summary."""
    prompt = f"""
        You are a senior protein biochemist with extensive laboratory experience. A colleague has just shown you protein function prediction results for the task '{task}' on model '{model}'. 
        Please analyze these results from a practical biologist's perspective:
        {results_df.to_string(index=False)}
        Provide a concise, practical analysis focusing ONLY on:
        0. The task '{task}' with "Stability" and "Optimum temperature" are regression task
        1. What the prediction results mean for each protein
        2. The biological significance of the confidence scores
        3. Practical experimental recommendations based on these predictions
        4. Any notable patterns or outliers in the results
        5. Do not output formatted content, just one paragraph is sufficient
        Use simple, clear language that a bench scientist would appreciate. Do NOT mention:
        - Training datasets or models
        - Technical implementation details  
        - Computational methods
        - Statistical concepts beyond confidence scores
        Keep your response under 200 words and speak as if you're having a conversation with a colleague in the lab.
        """
    return prompt

def parse_fasta_paste_content(fasta_content):
    if not fasta_content or not fasta_content.strip():
        return "No file selected", gr.update(choices=["Sequence 1"], value="Sequence 1", visible=False), {}, "Sequence 1", ""
    
    try:
        sequences = {}
        current_header = None
        current_sequence = ""
        
        for line in fasta_content.strip().split('\n'):
            line = line.strip()
            if not line:
                continue

            if line.startswith('>'):
                if current_header is not None and current_sequence:
                    sequences[current_header] = current_sequence
                
                current_header = line[1:].strip()
                if not current_header:
                    current_header = f"Sequence_{len(sequences) + 1}"
                current_sequence = ""
            else:
                if current_header is None:
                    current_header = "Sequence_1"
                clean_line = ''.join(c.upper() for c in line if c.isalpha())
                current_sequence += clean_line
        
        if current_header is not None and current_sequence:
            sequences[current_header] = current_sequence
        
        if not sequences:
            return "No valid protein sequences found in FASTA content", gr.update(choices=["Sequence 1"], value="Sequence 1", visible=False), {}, "Sequence 1", ""

        sequence_choices = list(sequences.keys())
        default_sequence = sequence_choices[0]
        display_sequence = sequences[default_sequence]
        selector_visible = len(sequence_choices) > 1
        temp_dir = Path("temp_outputs")
        sequence_dir = temp_dir / "Fasta"
        sequence_dir.mkdir(parents=True, exist_ok=True)
        temp_fasta_path = os.path.join(sequence_dir, f"paste_content_seq_{sanitize_filename(default_sequence)}.fasta")
        save_selected_sequence_fasta(fasta_content, default_sequence, temp_fasta_path)
        return display_sequence, gr.update(choices=sequence_choices, value=default_sequence, visible=selector_visible), sequences, default_sequence, temp_fasta_path
        
    except Exception as e:
        print(f"Error in parse_fasta_paste_content: {str(e)}")
        return f"Error parsing FASTA content: {str(e)}", gr.update(choices=["Sequence 1"], value="Sequence 1", visible=False), {}, "Sequence 1", ""

def save_selected_sequence_fasta(original_fasta_content, selected_sequence, output_path):
    new_fasta_lines = []
    lines = original_fasta_content.strip().split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('>'):
            header = line[1:].strip()
            if header == selected_sequence:
                new_fasta_lines.append(line)
                new_fasta_lines.append(lines[i+1])
                break
            else:
                i += 1
        else:
            i += 1
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_fasta_lines))

def handle_paste_sequence_selection(selected_sequence, sequences_dict, original_fasta_content_from_state):
    if not sequences_dict or selected_sequence not in sequences_dict:
        return "No file selected", ""
    
    # Check if the content is valid
    if not original_fasta_content_from_state or original_fasta_content_from_state == "No file selected":
        return "No file selected", ""
    
    try:
        temp_dir = Path("temp_outputs")
        sequence_dir = temp_dir / "Fasta"
        sequence_dir.mkdir(parents=True, exist_ok=True)
        temp_pdb_path = os.path.join(sequence_dir, f"paste_content_seq_{selected_sequence}.fasta")
        save_selected_sequence_fasta(original_fasta_content_from_state, selected_sequence, temp_pdb_path)
        
        return sequences_dict[selected_sequence], temp_pdb_path
        
    except Exception as e:
        return f"Error processing chain selection: {str(e)}", ""

def handle_fasta_sequence_change(selected_sequence, sequences_dict, original_fasta_path):
    if not sequences_dict or selected_sequence not in sequences_dict:
        return "No file selected", ""
    
    # Check if the file path is valid and exists
    if not original_fasta_path or original_fasta_path == "No file selected" or not os.path.exists(original_fasta_path):
        return "No file selected", ""
    
    try:
        with open(original_fasta_path, 'r') as f:
            lines = f.read()

        new_fasta_lines = []
        new_fasta_lines.append(">"+selected_sequence)
        new_fasta_lines.append(sequences_dict[selected_sequence])
        
        dir_path = os.path.dirname(original_fasta_path)
        base_name, extension = os.path.splitext(os.path.basename(original_fasta_path))
        new_filename = f"{base_name}_1{extension}"
        new_fasta_path = os.path.join(dir_path, new_filename)
        with open(new_fasta_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_fasta_lines))
        
        return sequences_dict[selected_sequence], new_fasta_path

    except Exception as e:
        return f"Error processing sequence selection: {str(e)}", ""

def parse_pdb_paste_content(pdb_content):
    if not pdb_content.strip():
        return "No file selected", gr.update(choices=["A"], value="A", visible=False), {}, "A", ""
    
    try:
        chains = {}
        current_chain = None
        sequence = ""
        
        for line in pdb_content.strip().split('\n'):
            if line.startswith('ATOM'):
                chain_id = line[21:22].strip()
                if chain_id == "":
                    chain_id = "A"
                
                if current_chain != chain_id:
                    if current_chain is not None and sequence:
                        chains[current_chain] = sequence
                    current_chain = chain_id
                    sequence = ""
                
                res_name = line[17:20].strip()
                if res_name in ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR']:
                    aa_map = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 
                             'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 
                             'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 
                             'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}
                    
                    res_num = int(line[22:26].strip())
                    if len(sequence) < res_num:
                        sequence += aa_map[res_name]
        
        if current_chain is not None and sequence:
            chains[current_chain] = sequence
        
        if not chains:
            return "No valid protein chains found in PDB content", gr.update(choices=["A"], value="A", visible=False), {}, "A", ""
        
        chain_choices = list(chains.keys())
        default_chain = chain_choices[0]
        display_sequence = chains[default_chain]
        selector_visible = len(chain_choices) > 1
        temp_dir = Path("temp_outputs")
        sequence_dir = temp_dir / "Fasta"
        sequence_dir.mkdir(parents=True, exist_ok=True)
        temp_pdb_path = os.path.join(sequence_dir, f"paste_content_chain_{default_chain}.pdb")
        save_selected_chain_pdb(pdb_content, default_chain, temp_pdb_path)
        return display_sequence, gr.update(choices=chain_choices, value=default_chain, visible=selector_visible), chains, default_chain, temp_pdb_path
        
    except Exception as e:
        return f"Error parsing PDB content: {str(e)}", gr.update(choices=["A"], value="A", visible=False), {}, "A", ""

def save_selected_chain_pdb(original_pdb_content, selected_chain, output_path):
    new_pdb_lines = []
    atom_counter = 1
    
    for line in original_pdb_content.strip().split('\n'):
        if line.startswith('ATOM'):
            chain_id = line[21:22].strip()
            if chain_id == "":
                chain_id = "A"
            
            if chain_id == selected_chain:
                new_line = line[:21] + 'A' + line[22:]
                new_line = f"ATOM  {atom_counter:5d}" + new_line[11:]
                new_pdb_lines.append(new_line)
                atom_counter += 1
        elif not line.startswith('ATOM'):
            new_pdb_lines.append(line)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(new_pdb_lines))

def handle_paste_chain_selection(selected_chain, chains_dict, original_pdb_content_from_state):
    if not chains_dict or selected_chain not in chains_dict:
        return "No file selected", ""
    
    # Check if the content is valid
    if not original_pdb_content_from_state or original_pdb_content_from_state == "No file selected":
        return "No file selected", ""
    
    try:
        temp_dir = Path("temp_outputs")
        sequence_dir = temp_dir / "PDB"
        sequence_dir.mkdir(parents=True, exist_ok=True)
        temp_pdb_path = os.path.join(sequence_dir, f"paste_content_chain_{selected_chain}.pdb")
        save_selected_chain_pdb(original_pdb_content_from_state, selected_chain, temp_pdb_path)
        
        return chains_dict[selected_chain], temp_pdb_path
        
    except Exception as e:
        return f"Error processing chain selection: {str(e)}", ""

def handle_pdb_chain_change(selected_chain, chains_dict, original_file_path):
    if not chains_dict or selected_chain not in chains_dict:
        return "No file selected", ""
    
    # Check if the file path is valid and exists
    if not original_file_path or original_file_path == "No file selected" or not os.path.exists(original_file_path):
        return "No file selected", ""
        
    try:
        with open(original_file_path, 'r') as f:
            pdb_content = f.read()
        
        new_pdb_lines = []
        atom_counter = 1
        
        for line in pdb_content.strip().split('\n'):
            if line.startswith('ATOM'):
                chain_id = line[21:22].strip()
                if chain_id == "":
                    chain_id = "A"
                
                if chain_id == selected_chain:
                    new_line = line[:21] + 'A' + line[22:]
                    new_line = f"ATOM  {atom_counter:5d}" + new_line[11:]
                    new_pdb_lines.append(new_line)
                    atom_counter += 1
            elif not line.startswith('ATOM'):
                new_pdb_lines.append(line)
        
        dir_path = os.path.dirname(original_file_path)
        base_name, extension = os.path.splitext(os.path.basename(original_file_path))
        new_filename = f"{base_name}_A{extension}"
        new_pdb_path = os.path.join(dir_path, new_filename)
        
        with open(new_pdb_path, 'w') as f:
            f.write('\n'.join(new_pdb_lines))
        
        return chains_dict[selected_chain], new_pdb_path
        
    except Exception as e:
        return f"Error processing chain selection: {str(e)}", ""

def process_pdb_file_upload(file_path):
    if not file_path:
        return "No file selected", gr.update(choices=["A"], value="A", visible=False), {}, "A", "", ""
    try:
        with open(file_path, 'r') as f:
            pdb_content = f.read()
        sequence, chain_update, chains_dict, default_chain, _ = parse_pdb_paste_content(pdb_content)
        return sequence, chain_update, chains_dict, default_chain, file_path, file_path
    except Exception as e:
        return f"Error reading PDB file: {str(e)}", gr.update(choices=["A"], value="A", visible=False), {}, "A", "", ""

def process_fasta_file_upload(file_path):
    if not file_path:
        return "No file selected", gr.update(choices=["Sequence 1"], value="Sequence 1", visible=False), {}, "Sequence 1", "", ""
    try:
        with open(file_path, 'r') as f:
            fasta_content = f.read()
        sequence, selector_update, sequences_dict, default_sequence, file_path = parse_fasta_paste_content(fasta_content)
        return sequence, selector_update, sequences_dict, default_sequence, file_path, file_path
    except Exception as e:
        return f"Error reading FASTA file: {str(e)}", gr.update(choices=["Sequence 1"], value="Sequence 1", visible=False), {}, "Sequence 1", "", ""
        
def handle_file_upload(file_obj: Any) -> str:
    if not file_obj:
        return "No file selected", gr.update(choices=["Sequence 1"], value="Sequence 1", visible=False), {}, "Sequence 1", "", ""
    if isinstance(file_obj, str):
        file_path = file_obj
    else:
        file_path = file_obj.name
    if file_path.lower().endswith((".fasta", ".fa")):
        return process_fasta_file_upload(file_path)
    elif file_path.lower().endswith(".pdb"):
        return process_pdb_file_upload(file_path)
    else:
        return "No file selected", gr.update(choices=["Sequence 1"], value="Sequence 1", visible=False), {}, "Sequence 1", "", ""

def _read_fasta_file(file_path: str) -> str:
    with open(file_path, "r") as f:
        return f.read()

def _read_pdb_file(file_path: str) -> str:
    """Extract amino acid sequence from PDB file."""
    aa_map = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }
    
    sequence = []
    seen_residues = set()
    chain = None
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                chain_id = line[21]
                if chain is None:
                    chain = chain_id
                if chain_id != chain:
                    break
                
                res_id = (chain_id, int(line[22:26]))
                if res_id not in seen_residues:
                    res_name = line[17:20].strip()
                    if res_name in aa_map:
                        sequence.append(aa_map[res_name])
                        seen_residues.add(res_id)
    
    return "".join(sequence)

def sanitize_filename(name: str) -> str:
    """Sanitize filename for safe file operations."""
    name = re.split(r'[|\s/]', name)[0]
    return re.sub(r'[^\w\-. ]', '_', name)

def get_total_residues_count(df: pd.DataFrame) -> int:
    """Get total number of unique residue positions from mutation data."""
    if 'mutant' not in df.columns:
        return 0
    
    try:
        positions = df['mutant'].str.extract(r'(\d+)').dropna()
        return positions[0].astype(int).nunique() if not positions.empty else 0
    except Exception:
        return 0

def toggle_ai_section(is_checked: bool):
    """Toggle visibility of AI configuration section."""
    return gr.update(visible=is_checked)

def create_zip_archive(files_to_zip: Dict[str, str], zip_filename: str) -> str:
    """Create ZIP archive with specified files."""
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zf:
        for src, arc in files_to_zip.items():
            if os.path.exists(src):
                zf.write(src, arcname=arc)
    return zip_filename

def update_dataset_choices(task: str) -> gr.CheckboxGroup:
    """Update dataset choices based on selected task."""
    datasets = DATASET_MAPPING_FUNCTION.get(task, [])
    return gr.CheckboxGroup.update(choices=datasets, value=datasets)


def run_zero_shot_prediction(model_type: str, model_name: str, file_path: str) -> Tuple[str, pd.DataFrame]:
    """Run zero-shot mutation prediction."""
    try:
        temp_dir = Path("temp_outputs")
        temp_dir_ = temp_dir / "Zero_shot_result"
        timestamp = str(int(time.time()))
        sequence_dir = temp_dir_ / timestamp
        sequence_dir.mkdir(parents=True, exist_ok=True)
        output_csv = sequence_dir / f"{model_type}.csv"
        script_name = MODEL_MAPPING_ZERO_SHOT.get(model_name)
        
        if not script_name:
            return f"Error: Model '{model_name}' has no script.", pd.DataFrame()

        script_path = f"src/mutation/models/{script_name}.py"
        if not os.path.exists(script_path):
            return f"Script not found: {script_path}", pd.DataFrame()
        
        file_argument = "--pdb_file" if model_type == "structure" else "--fasta_file"
        cmd = [
            sys.executable, script_path, 
            file_argument, file_path, 
            "--output_csv", output_csv
        ]
        
        subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True, 
            encoding='utf-8', 
            errors='ignore'
        )

        if os.path.exists(output_csv):
            df = pd.read_csv(output_csv)
            os.remove(output_csv)
            return "Prediction completed successfully!", df
        
        return "Prediction finished but no output file was created.", pd.DataFrame()
        
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr or e.stdout or "Unknown subprocess error"
        return f"Prediction script failed: {error_msg}", pd.DataFrame()
    except Exception as e:
        return f"An unexpected error occurred: {e}", pd.DataFrame()

def prepare_top_residue_heatmap_data(df: pd.DataFrame) -> Tuple:
    """Prepare data for heatmap visualization."""
    score_col = next((col for col in df.columns if 'score' in col.lower()), None)
    rank_col = "Prediction Rank"
    if score_col is None:
        return (None,) * 5

    valid_df = df[
        df['mutant'].apply(
            lambda m: isinstance(m, str) and 
            re.match(r'^[A-Z]\d+[A-Z]$', m) and 
            m[0] != m[-1]
        )
    ].copy()
    
    if valid_df.empty:
        return ([], [], np.array([[]]), np.array([[]]), score_col)

    min_score, max_score = valid_df[score_col].min(), valid_df[score_col].max()
    if max_score == min_score:
        valid_df['scaled_score'] = 0.0
    else:
        valid_df['scaled_score'] = -1 + 2 * (valid_df[score_col] - min_score) / (max_score - min_score)
    
    valid_df['position'] = valid_df['mutant'].str[1:-1].astype(int)

    effect_scores = valid_df.groupby('position')['scaled_score'].mean()
    sorted_positions = effect_scores.sort_values(ascending=False)
    top_positions = sorted_positions.head(20).index if len(sorted_positions) > 20 else sorted_positions.index
    top_df = valid_df[valid_df['position'].isin(top_positions)]
    x_labels = list("ACDEFGHIKLMNPQRSTVWY")
    x_map = {label: i for i, label in enumerate(x_labels)}
    wt_map = {pos: mut[0] for pos, mut in zip(top_df['position'], top_df['mutant'])}
    y_labels = [f"{wt_map.get(pos, '?')}{pos}" for pos in top_positions]
    y_map = {pos: i for i, pos in enumerate(top_positions)}
    
    z_data = np.full((len(y_labels), len(x_labels)), np.nan)
    score_matrix = np.full((len(y_labels), len(x_labels)), np.nan)
    rank_matrix = np.full((len(y_labels), len(x_labels)), np.nan)

    for _, row in top_df.iterrows():
        pos, mut_aa = row['position'], row['mutant'][-1]
        if pos in y_map and mut_aa in x_map:
            y_idx, x_idx = y_map[pos], x_map[mut_aa]
            z_data[y_idx, x_idx] = row['scaled_score']
            score_matrix[y_idx, x_idx] = round(row[score_col], 3)
            rank_matrix[y_idx, x_idx] = row[rank_col]
            
    return x_labels, y_labels, z_data, score_matrix, rank_matrix

def generate_plotly_heatmap(x_labels: List, y_labels: List, z_data: np.ndarray, score_data: np.ndarray) -> go.Figure:
    """Generate Plotly heatmap visualization."""
    if z_data is None or z_data.size == 0:
        return go.Figure().update_layout(title="Not enough data for heatmap")

    num_residues = len(y_labels)
    fig = go.Figure(data=go.Heatmap(
        z=z_data, 
        x=x_labels, 
        y=y_labels, 
        customdata=score_data,
        hovertemplate=(
            "<b>Position</b>: %{y}<br>"
            "<b>Mutation</b>: %{x}<br>"
            "<b>Effect</b>: %{z:.2f}"
            "<extra></extra>"
        ),
        colorscale='Blues', 
        zmin=-1, 
        zmax=1, 
        showscale=True,
        colorbar={
            'title': 'Normalized Effect', 
            'tickvals': [-1, 0, 1], 
            'ticktext': ['Low', 'Neutral', 'High']
        }
    ))
    
    fig.update_layout( 
        xaxis_title='Mutant Amino Acid', 
        yaxis_title='Residue Position (by impact)',
        height=max(400, 30 * num_residues + 150), 
        yaxis_autorange='reversed'
    )
    
    return fig


def process_fasta_file(file_path: str) -> str:
    sequences = []
    current_seq = ""
    current_header = ""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_header and current_seq:
                    sequences.append((current_header, current_seq))
                current_header = line
                current_seq = ""
            else:
                current_seq += line

        if current_header and current_seq:
            sequences.append((current_header, current_seq))
    
    if len(sequences) <= 1:
        return file_path
    

    original_path = Path(file_path)
    temp_dir = Path("temp_outputs")
    fasra_dir = temp_dir / "Fasta"
    fasra_dir.mkdir(parents=True, exist_ok=True)

    new_file_path = fasra_dir / f"filtered_{original_path.name}"
    
    with open(new_file_path, 'w', encoding='utf-8') as f:
        f.write(f"{sequences[0][0]}\n")
        seq = sequences[0][1]
        f.write(f"{seq}\n")
    
    return str(new_file_path)

def handle_mutation_prediction_base(
    function_selection: str, 
    file_obj: Any, 
    enable_ai: bool, 
    ai_model: str, 
    user_api_key: str, 
    model_name: Optional[str] = None,
    progress=gr.Progress()
) -> Generator:
    try:
        import requests
        requests.post("/api/stats/track", json={"module": "function_analysis"})
    except Exception:
        pass
    """Handle mutation prediction workflow."""
    if not file_obj or not function_selection:
        yield (
            "❌ Error: Function and file are required.", 
            None, None, gr.update(visible=False), None, 
            gr.update(visible=False), None, 
            "Please select a function and upload a file."
        )
        return

    if isinstance(file_obj, str):
        file_path = file_obj
    else:
        file_path = file_obj.name

    if file_path.lower().endswith((".fasta", ".fa")):
        if model_name and model_name in ["ESM2-650M", "ESM-1v", "ESM-1b"]:
            model_type = "sequence"
        else:
            model_name, model_type = "ESM2-650M", "sequence"

        processed_file_path = process_fasta_file(file_path)
        if processed_file_path != file_path:
            file_path = processed_file_path
            yield (
                "⚠️ Multi-sequence FASTA detected. Using only the first sequence for prediction.",
                None, None, gr.update(visible=False), None,
                gr.update(visible=False), None,
                "Processing first sequence only..."
            )
    elif file_path.lower().endswith(".pdb"):
        if model_name and model_name in ["ESM-IF1", "SaProt", "MIF-ST", "ProSST-2048", "ProtSSN"]:
            model_type = "structure"
        else:
            model_name, model_type = "ESM-IF1", "structure"
    else:
        yield (
            "❌ Error: Unsupported file type.", 
            None, None, gr.update(visible=False), None, 
            gr.update(visible=False), None, 
            "Please upload a .fasta, .fa, or .pdb file."
        )
        return

    yield (
        f"⏳ Running prediction...", 
        None, None, gr.update(visible=False), None, 
        gr.update(visible=False), None, 
        "Prediction in progress..."
    )
    progress(0.1, desc="Running prediction...")
    status, raw_df = run_zero_shot_prediction(model_type, model_name, file_path)
    progress(0.7, desc="Processing results...")
    if raw_df.empty:
        yield (
            status, 
            go.Figure(layout={'title': 'No results generated'}), 
            pd.DataFrame(), gr.update(visible=False), None, 
            gr.update(visible=False), None, 
            "No results to analyze."
        )
        return
    
    score_col = next((c for c in raw_df.columns if 'score' in c.lower()), raw_df.columns[1])
    
    display_df = pd.DataFrame()
    display_df['Mutant'] = raw_df['mutant']
    display_df['Prediction Rank'] = range(1, len(raw_df) + 1)
    
    min_s, max_s = raw_df[score_col].min(), raw_df[score_col].max()
    if max_s == min_s:
        scaled_scores = pd.Series([0.0] * len(raw_df))
    else:
        scaled_scores = -1 + 2 * (raw_df[score_col] - min_s) / (max_s - min_s)
    display_df['Prediction Score'] = scaled_scores.round(2)

    df_for_heatmap = raw_df.copy()
    df_for_heatmap['Prediction Rank'] = range(1, len(df_for_heatmap) + 1)

    total_residues = get_total_residues_count(df_for_heatmap)
    data_tuple = prepare_top_residue_heatmap_data(df_for_heatmap)
    
    if data_tuple[0] is None:
        yield (
            status, 
            go.Figure(layout={'title': 'Score column not found'}), 
            display_df, gr.update(visible=False), None, 
            gr.update(visible=False), display_df, 
            "Score column not found."
        )
        return

    summary_fig = generate_plotly_heatmap(*data_tuple[:4])
    
    ai_summary = "AI Analysis disabled. Enable in settings to generate a report."

    expert_analysis = "<div style='height: 300px; display: flex; align-items: center; justify-content: center; color: #666;'>Analysis will appear here once prediction is complete...</div>"

    if enable_ai:
        progress(0.8, desc="Generating AI summary...")
        yield (
            f"🤖 Expert is analyzing results...", 
            summary_fig, display_df, gr.update(visible=False), None, 
            gr.update(visible=total_residues > 20), display_df, 
            expert_analysis
        )
        
        api_key = get_api_key(ai_model, user_api_key)
        if not api_key:
            ai_summary = "❌ No API key found."
        else:
            ai_config = AIConfig(
                api_key, ai_model, 
                AI_MODELS[ai_model]["api_base"], 
                AI_MODELS[ai_model]["model"]
            )
            prompt = generate_mutation_ai_prompt(display_df, model_name, function_selection)
            ai_summary = call_ai_api(ai_config, prompt)
            expert_analysis = format_expert_response(ai_summary)
        progress(0.9, desc="Finalizing AI analysis...")
    else:
        progress(1.0, desc="Complete!")
    
    
    temp_dir = Path("temp_outputs")
    temp_dir_ = temp_dir / "Zero_shot_result"
    timestamp = str(int(time.time()))
    heatmap_dir = temp_dir_ / timestamp
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    csv_path = heatmap_dir / f"mut_res.csv"
    heatmap_path = heatmap_dir / f"mut_map.html"
    
    display_df.to_csv(csv_path, index=False)
    summary_fig.write_html(heatmap_path)
    
    files_to_zip = {
        str(csv_path): "prediction_results.csv", 
        str(heatmap_path): "prediction_heatmap.html"
    }
    
    if not ai_summary.startswith("❌") and not ai_summary.startswith("AI Analysis"):
        report_path = heatmap_dir / f"ai_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(ai_summary)
        files_to_zip[str(report_path)] = "AI_Analysis_Report.md"

    zip_path = heatmap_dir / f"pred_mut.zip"
    zip_path_str = create_zip_archive(files_to_zip, str(zip_path))

    final_status = status if not enable_ai else "✅ Prediction and AI analysis complete!"
    progress(1.0, desc="Complete!")
    yield (
        final_status, summary_fig, display_df, 
        gr.update(visible=True, value=zip_path_str), zip_path_str, 
        gr.update(visible=total_residues > 20), display_df, expert_analysis
    )

def generate_plots_for_all_results(results_df: pd.DataFrame) -> go.Figure:
    """Generate plots for function prediction results with consistent Dardana font and academic styling."""
    # Filter data
    plot_df = results_df[
        (results_df['header'] != "ERROR") & 
        (results_df['Dataset'].apply(
            lambda d: DATASET_TO_TASK_MAP.get(d) not in REGRESSION_TASKS_FUNCTION
        ))
    ].copy()

    if plot_df.empty:
        return go.Figure().update_layout(
            title_text="<b>No Visualization Available</b>", 
            title_x=0.5,
            font_family="Dardana",
            title_font_size=14,
            xaxis={"visible": False}, 
            yaxis={"visible": False},
            annotations=[{
                "text": "This task does not support visualization.", 
                "xref": "paper", "yref": "paper", 
                "x": 0.5, "y": 0.5, 
                "showarrow": False,
                "font": {"family": "Dardana", "size": 12}
            }]
        )

    sequences = plot_df['header'].unique()
    datasets = plot_df['Dataset'].unique()
    n_seq, n_data = len(sequences), len(datasets)
    
    titles = [
        f"{seq[:15]}<br>{ds[:20]}" if n_seq > 1 else f"{ds[:25]}" 
        for seq in sequences for ds in datasets
    ]
    
    fig = make_subplots(
        rows=n_seq, cols=n_data, 
        subplot_titles=titles, 
        vertical_spacing=0.25 if n_seq > 1 else 0.15
    )

    FONT_STYLE = dict(family="Dardana", size=12)
    AXIS_TITLE_STYLE = dict(family="Dardana", size=11)
    TICK_FONT_STYLE = dict(family="Dardana", size=10)
    BAR_WIDTH = 0.7 

    for r_idx, seq in enumerate(sequences, 1):
        for c_idx, ds in enumerate(datasets, 1):
            row_data = plot_df[(plot_df['header'] == seq) & (plot_df['Dataset'] == ds)]
            if row_data.empty:
                continue
            
            row = row_data.iloc[0]
            prob_col = next((col for col in row.index if 'probab' in col.lower()), None)
            
            if not prob_col or pd.isna(row[prob_col]):
                continue

            try:
                confidences = (json.loads(row[prob_col]) if isinstance(row[prob_col], str) 
                             else row[prob_col])
                
                if not isinstance(confidences, list):
                    continue
                
                task = DATASET_TO_TASK_MAP.get(ds)
                labels_key = ("DeepLocMulti" if ds == "DeepLocMulti" 
                            else "DeepLocBinary" if ds == "DeepLocBinary" 
                            else task)
                
                labels = LABEL_MAPPING_FUNCTION.get(
                    labels_key, 
                    [f"Class {k}" for k in range(len(confidences))]
                )
                
                colors = [
                    COLOR_MAP_FUNCTION.get(lbl, COLOR_MAP_FUNCTION["Default"]) 
                    for lbl in labels
                ]
                
                plot_data = sorted(
                    zip(labels, confidences, colors), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                sorted_labels, sorted_conf, sorted_colors = zip(*plot_data)
                
                fig.add_trace(
                    go.Bar(
                        x=sorted_labels, 
                        y=sorted_conf, 
                        marker_color=sorted_colors,
                        width=BAR_WIDTH,
                        marker_line_width=0,
                        hovertemplate="<b>%{x}</b><br>Confidence: %{y:.2f}<extra></extra>"
                    ), 
                    row=r_idx, col=c_idx
                )
                
                # Axis styling
                fig.update_xaxes(
                    tickfont=TICK_FONT_STYLE,
                    row=r_idx, col=c_idx
                )
                fig.update_yaxes(
                    range=[0, 1], 
                    title_text="Confidence" if c_idx == 1 else "",
                    title_font=AXIS_TITLE_STYLE,
                    tickfont=TICK_FONT_STYLE,
                    row=r_idx, col=c_idx
                )
                
            except Exception as e:
                print(f"Plotting error for {seq}/{ds}: {e}")

    # Global layout adjustments
    main_title = "<b>Prediction Confidence Scores</b>"
    if n_seq == 1:
        main_title += f"<br><sub>{sequences[0][:80]}</sub>"
    
    fig.update_layout(
        title=dict(
            text=main_title, 
            x=0.5,
            font=dict(family="Dardana", size=16)
        ), 
        showlegend=False,
        font=FONT_STYLE,
        height=max(400, 300 * n_seq + 100),
        margin=dict(l=50, r=50, b=80, t=100),
        plot_bgcolor='white',
        paper_bgcolor='white',
        bargap=0.2 
    )
    
    # Subplot title styling
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(family="Dardana", size=12)
    
    return fig

def handle_protein_function_prediction(
    task: str, 
    fasta_file: Any, 
    enable_ai: bool, 
    ai_model: str, 
    user_api_key: str, 
    model_name: Optional[str] = None, 
    datasets: Optional[List[str]] = None,
    progress=gr.Progress()
    ) -> Generator:
    try:
        import requests
        requests.post("/api/stats/track", json={"module": "function_analysis"})
    except Exception:
        pass
    """Handle protein function prediction workflow."""
    model = model_name if model_name else "ESM2-650M"
    datasets = (datasets if datasets is not None 
               else DATASET_MAPPING_FUNCTION.get(task, []))

    if not all([task, datasets, fasta_file]):
        yield (
            "❌ Error: Task, Datasets, and FASTA file are required.", 
            pd.DataFrame(), 
            gr.update(visible=False), 
            "Please provide all required inputs.",
            "AI Analysis disabled."
        )
        return

    yield (
        f"🚀 Starting predictions with {model}...", 
        pd.DataFrame(), 
        gr.update(visible=False), 
        "AI analysis will appear here...",
        "AI Analysis disabled."
    )
    
    all_results_list = []
    temp_dir = Path("temp_outputs")
    temp_dir_ = temp_dir / "Protein_Function"
    timestamp = str(int(time.time()))
    function_dir = temp_dir_ / timestamp
    function_dir.mkdir(parents=True, exist_ok=True)


    # Run predictions for each dataset
    progress(0.1, desc="Running prediction...")
    for i, dataset in enumerate(datasets):
        yield (
            f"⏳ Running prediction...", 
            pd.DataFrame(), 
            gr.update(visible=False), 
            "AI analysis will appear here...",
            "AI Analysis disabled."
        )
        
        try:
            model_key = MODEL_MAPPING_FUNCTION.get(model)
            if not model_key:
                raise ValueError(f"Model key not found for {model}")
            
            adapter_key = MODEL_ADAPTER_MAPPING_FUNCTION[model_key]
            script_path = Path("src") / "property" / f"{model_key}.py"
            adapter_path = Path("ckpt") / dataset / adapter_key
            output_file = function_dir / f"temp_{dataset}_{model}.csv"
            
            if not script_path.exists() or not adapter_path.exists():
                raise FileNotFoundError(f"Required files not found: Script={script_path}, Adapter={adapter_path}")
            if isinstance(fasta_file, str):
                file_path = fasta_file
            else:
                file_path = fasta_file.name
            cmd = [sys.executable, str(script_path), "--fasta_file", str(Path(file_path)), "--adapter_path", str(adapter_path), "--output_csv", str(output_file)]
            subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8', errors='ignore')
            
            if output_file.exists():
                df = pd.read_csv(output_file) 
                df["Dataset"] = dataset
                all_results_list.append(df)
                os.remove(output_file)
        except Exception as e:
            error_detail = e.stderr if isinstance(e, subprocess.CalledProcessError) else str(e)
            print(f"Failed to process '{dataset}': {error_detail}")
            all_results_list.append(pd.DataFrame([{"Dataset": dataset, "header": "ERROR", "sequence": error_detail}]))
    progress(0.7, desc="Processing results...")
    if not all_results_list:
        yield (
            "⚠️ No results generated.", 
            pd.DataFrame(), 
            gr.update(visible=False), 
            "No results to analyze.",
            "AI Analysis disabled."
        )
        return
    
    raw_final_df = pd.concat(all_results_list, ignore_index=True).fillna('N/A')
    
    final_df = raw_final_df.copy()
    non_voting_tasks = REGRESSION_TASKS_FUNCTION
    non_voting_datasets = ["DeepLocMulti", "DeepLocBinary"]
    is_voting_run = task not in non_voting_tasks and not any(ds in raw_final_df['Dataset'].unique() for ds in non_voting_datasets) and len(raw_final_df['Dataset'].unique()) > 1

    if is_voting_run:
        yield (
            "🤝 Performing soft voting on prediction results...", 
            pd.DataFrame(), 
            gr.update(visible=False), 
            "Aggregating results...",
            "AI Analysis disabled."
        )
        voted_results = []
        for header, group in raw_final_df.groupby('header'):
            if group.empty: continue
            
            # Get the column name for predictions (could be 'prediction' or 'predicted_class')
            pred_col = 'prediction' if 'prediction' in group.columns else 'predicted_class'
            if pred_col not in group.columns: continue
            
            # Collect all probability distributions
            all_probs = []
            valid_rows = []
            
            for _, row in group.iterrows():
                prob_col = 'probabilities' if 'probabilities' in row else None
                if prob_col and pd.notna(row[prob_col]):
                    try:
                        # Handle string representation of list
                        if isinstance(row[prob_col], str):
                            # Remove brackets and split by comma
                            prob_str = row[prob_col].strip('[]')
                            probs = [float(x.strip()) for x in prob_str.split(',')]
                        elif isinstance(row[prob_col], list):
                            probs = row[prob_col]
                        else:
                            probs = json.loads(str(row[prob_col]))
                        
                        if isinstance(probs, list) and len(probs) > 0:
                            all_probs.append(probs)
                            valid_rows.append(row)
                    except (json.JSONDecodeError, ValueError, IndexError):
                        continue
            
            if not all_probs:
                continue
            
            # Perform soft voting: average all probability distributions
            # Ensure all probability arrays have the same length
            max_len = max(len(probs) for probs in all_probs)
            normalized_probs = []
            for probs in all_probs:
                if len(probs) < max_len:
                    # Pad with zeros if needed
                    probs.extend([0.0] * (max_len - len(probs)))
                normalized_probs.append(probs)
            
            # Calculate average probabilities across all models
            avg_probs = np.mean(normalized_probs, axis=0)
            
            # Get the class with highest average probability
            voted_class = np.argmax(avg_probs)
            voted_confidence = avg_probs[voted_class]
            
            # Create result row
            result_row = group.iloc[0].copy()
            result_row[pred_col] = voted_class
            if 'probabilities' in result_row:
                result_row['probabilities'] = voted_confidence
            
            voted_results.append(result_row.to_frame().T)
        
        if voted_results:
            final_df = pd.concat(voted_results, ignore_index=True)
            # Remove Dataset column for voting results
            final_df = final_df.drop(columns=['Dataset'], errors='ignore')

    # Generate plot but don't return it in the yield
    plot_fig = generate_plots_for_all_results(raw_final_df)
    display_df = final_df.copy()

    # Map prediction values to text labels BEFORE renaming columns
    def map_labels(row):
        current_task = DATASET_TO_TASK_MAP.get(row.get('Dataset', ''), task)
        
        # For regression tasks, return the numeric value as is
        if current_task in REGRESSION_TASKS_FUNCTION: 
            scaled_value = row.get("prediction")
            if pd.notna(scaled_value) and scaled_value != 'N/A' and current_task in REGRESSION_TASKS_FUNCTION_MAX_MIN:
                try:
                    scaled_value = float(scaled_value)
                    min_val, max_val = REGRESSION_TASKS_FUNCTION_MAX_MIN[current_task]
                    original_value = scaled_value * (max_val - min_val) + min_val
                    return round(original_value, 2)
                
                except (ValueError, TypeError):
                    return scaled_value

            return scaled_value

        if row.get('Dataset') == 'SortingSignal':
                predictions_str = row['predicted_class']
                predictions = json.loads(predictions_str)
                if all(p == 0 for p in predictions):
                    return "No signal"
                # Get labels for SortingSignal
                signal_labels = ["CH", 'GPI', "MT", "NES", "NLS", "PTS", "SP", "TM", "TH"]
                active_labels = []
                # Find indices where prediction is 1 (active labels)
                for i, pred in enumerate(predictions):
                    if pred == 1:
                        active_labels.append(signal_labels[i])
                print(predictions)
                # Return concatenated labels or "None" if no active labels
                return "_".join(active_labels) if active_labels else "None"

        # For classification tasks, map to text labels
        labels_key = ("DeepLocMulti" if row.get('Dataset') == "DeepLocMulti" 
                     else "DeepLocBinary" if row.get('Dataset') == "DeepLocBinary" 
                     else current_task)
        labels = LABEL_MAPPING_FUNCTION.get(labels_key)
        
        pred_val = row.get("prediction", row.get("predicted_class"))
        if pred_val is None or pred_val == "N/A":
            return "N/A"
            
        try:
            pred_val = int(float(pred_val))
            if labels and 0 <= pred_val < len(labels): 
                return labels[pred_val]
        except (ValueError, TypeError):
            pass
        
        return str(pred_val)

    # Apply label mapping
    if "prediction" in display_df.columns:
        display_df["predicted_class"] = display_df.apply(map_labels, axis=1)
    elif "predicted_class" in display_df.columns:
        display_df["predicted_class"] = display_df.apply(map_labels, axis=1)

    if 'prediction' in display_df.columns:
        display_df.drop(columns=['prediction'], inplace=True)

    rename_map = {
        'header': "Protein Name", 
        'sequence': "Sequence", 
        'predicted_class': "Predicted Class", # This now correctly renames the only remaining prediction column
        'probabilities': "Confidence Score", 
        'Dataset': "Dataset"
    }
    display_df.rename(columns=rename_map, inplace=True)
    
    # Truncate sequence display
    if "Sequence" in display_df.columns:
        display_df["Sequence"] = display_df["Sequence"].apply(lambda x: x[:]  if isinstance(x, str) and len(x) > 30 else x)

    # Format confidence score to 2 decimal places
    if "Confidence Score" in display_df.columns and "Predicted Class" in display_df.columns:
        def format_confidence(row):
            score = row["Confidence Score"]
            predicted_class = row["Predicted Class"]
            
            if isinstance(score, (float, int)) and score != 'N/A':
                return round(float(score), 2)
            elif isinstance(score, str) and score not in ['N/A', '']:
                try:
                    if score.startswith('[') and score.endswith(']'):
                        prob_str = score.strip('[]')
                        probs = [float(x.strip()) for x in prob_str.split(',')]
                        
                        # Get the original prediction index before text mapping
                        # We need to reverse map the text label back to index
                        pred_index = None
                        current_task = task if is_voting_run else row.get('Dataset', task)
                        
                        # Find the reverse mapping
                        if current_task in REGRESSION_TASKS_FUNCTION:
                            # For regression, return the actual value
                            return predicted_class
                        else:
                            # For classification, find the index of the predicted class
                            labels_key = task if is_voting_run else ("DeepLocMulti" if row.get('Dataset') == "DeepLocMulti" else "DeepLocBinary" if row.get('Dataset') == "DeepLocBinary" else current_task)
                            labels = LABEL_MAPPING_FUNCTION.get(labels_key, [])
                            
                            if labels and predicted_class in labels:
                                pred_index = labels.index(predicted_class)
                            
                            # Return the confidence for the predicted class
                            if pred_index is not None and 0 <= pred_index < len(probs):
                                return round(probs[pred_index], 2)
                            else:
                                # If we can't find the index, return the max probability
                                return round(max(probs), 2)
                    else:
                        return round(float(score), 2)
                except (ValueError, IndexError):
                    return score
            return score
        
        display_df["Confidence Score"] = display_df.apply(format_confidence, axis=1)

    ai_summary = "AI Analysis disabled. Enable in settings to generate a report."
    ai_response = "AI Analysis disabled."  # Initialize ai_response variable

    expert_analysis = "<div style='height: 300px; display: flex; align-items: center; justify-content: center; color: #666;'>Analysis will appear here once prediction is complete...</div>"

    if enable_ai:
        progress(0.8, desc="Generating AI summary...")
        yield (
            "🤖 Expert is analyzing results...", 
            display_df, 
            gr.update(visible=False), 
            expert_analysis,
            "AI Analysis in progress..."
        )
        api_key = get_api_key(ai_model, user_api_key)
        if not api_key: 
            ai_summary = f"❌ No API key found for {ai_model}."
        else:
            ai_config = AIConfig(api_key, ai_model, AI_MODELS[ai_model]["api_base"], AI_MODELS[ai_model]["model"])
            prompt = generate_expert_analysis_prompt(display_df, task)
            ai_response = call_ai_api(ai_config, prompt)
            expert_analysis = format_expert_response(ai_response)
        progress(0.9, desc="Finalizing AI analysis...")
    else:
        progress(1.0, desc="Complete!")
    
    # Create download zip with processed results
    zip_path_str = ""
    try:
        temp_dir = Path("temp_outputs")
        temp_dir_ = temp_dir / "Function_download"
        timestamp = str(int(time.time()))
        zip_dir = temp_dir_ / timestamp
        zip_dir.mkdir(parents=True, exist_ok=True)
        
        # Save only the processed results
        processed_df_for_save = display_df.copy()
        processed_df_for_save.to_csv(zip_dir  / "Result.csv", index=False)
        
        # Save plot as HTML file (optional)
        if plot_fig and hasattr(plot_fig, 'data') and plot_fig.data: 
            plot_fig.write_html(str(zip_dir/ "results_plot.html"))
        
        if not ai_summary.startswith("❌") and not ai_summary.startswith("AI Analysis"):
            with open(zip_dir / "AI_Report.md", 'w', encoding='utf-8') as f: 
                f.write(f"# AI Expert Analysis\n\n{ai_summary}")
        
        zip_path = temp_dir / f"func_pred_{ts}.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for file in zip_dir.glob("*"): 
                zf.write(file, file.name)
        zip_path_str = str(zip_path)
    except Exception as e: 
        print(f"Error creating zip file: {e}")

    final_status = "✅ All predictions completed!"
    if is_voting_run: 
        final_status += " Results were aggregated using soft voting."
    if enable_ai and not ai_summary.startswith("❌"): 
        final_status += " AI analysis included."
    progress(1.0, desc="Complete!")
    yield (
        final_status, 
        display_df, 
        gr.update(visible=True, value=zip_path_str) if zip_path_str else gr.update(visible=False), 
        expert_analysis, ai_response
    )


def generate_plots_for_residue_results(results_df: pd.DataFrame, task: str = "Functional Prediction") -> go.Figure:
    """Generate plots for residue prediction results with consistent styling."""
    if results_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data to visualize",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, family="Arial")
        )
        return fig
    
    residues = results_df['residue'].tolist()
    probabilities = results_df['probability'].tolist()
    positions = results_df['index'].tolist() if 'index' in results_df.columns else list(range(1, len(residues) + 1))
    predicted_labels = results_df['predicted_label'].tolist() if 'predicted_label' in results_df.columns else [1 if p >= 0.5 else 0 for p in probabilities]
    functional_residues = [i for i, label in enumerate(predicted_labels) if label == 1]

    fig = go.Figure()

    sequence_length = len(residues)
    y_position = 0

    fig.add_trace(go.Scatter(
        x=[1, sequence_length],
        y=[y_position, y_position],
        mode='lines',
        line=dict(color='lightgray', width=12),
        showlegend=False,
        hoverinfo='skip'
    ))

    if functional_residues:
        segments = []
        current_segment = [functional_residues[0]]

        for i in range(1, len(functional_residues)):
            if functional_residues[i] == functional_residues[i-1] + 1:
                current_segment.append(functional_residues[i])
            else:
                segments.append(current_segment)
                current_segment = [functional_residues[i]]
        segments.append(current_segment)

        for i, segment in enumerate(segments):
            start_pos = positions[segment[0]]
            end_pos = positions[segment[-1]]

            segment_residues = [residues[j] for j in segment]
            segment_probs = [probabilities[j] for j in segment]
            hover_text = f"Task: {task}<br>Range: {start_pos}-{end_pos}<br>Residues: {''.join(segment_residues)}<br>Avg Probability: {np.mean(segment_probs):.3f}"
            
            fig.add_trace(go.Scatter(
                x=[start_pos, end_pos],
                y=[y_position, y_position],
                mode='lines',
                line=dict(color='red', width=12),
                showlegend=False,
                hovertemplate=hover_text + "<extra></extra>",
                name=f"Functional Region {i+1}"
            ))

    interval = max(1, sequence_length // 10)
    position_labels = list(range(1, sequence_length + 1, interval))
    if position_labels[-1] != sequence_length:
        position_labels.append(sequence_length)
    
    fig.update_layout(
        title=dict(
            text=f"<b>Functional Residue Prediction</b> - {task}",
            x=0.02, y=0.95, xanchor='left', yanchor='top',
            font=dict(size=14, family="Arial", color="black")
        ),
        xaxis=dict(
            title="Residue Position", 
            tickmode='array', 
            tickvals=position_labels,
            ticktext=[str(pos) for pos in position_labels],
            tickfont=dict(size=10), 
            showgrid=False,
            zeroline=False, 
            range=[0, sequence_length + 1]
        ),
        yaxis=dict(
            title="", 
            showticklabels=False, 
            showgrid=False,
            zeroline=False, 
            range=[-0.5, 0.5]
        ),
        showlegend=False, 
        height=150, 
        margin=dict(l=50, r=50, t=50, b=40),
        plot_bgcolor='white', 
        paper_bgcolor='white', 
        font=dict(family="Arial", size=10), 
        hovermode='closest'
    )
    return fig

def expand_residue_predictions(df):
    expanded_rows = []
    
    for _, row in df.iterrows():
        header = row['header']
        sequence = row['sequence']
        task = row['Task']
        dataset = row['Dataset']
        try:
            predictions = json.loads(row['prediction'])
            probabilities = json.loads(row['probabilities']) if isinstance(row['probabilities'], str) else row['probabilities']
            
            if isinstance(predictions[0], list):
                predictions = predictions[0]
            if isinstance(probabilities[0], list):
                probabilities = probabilities[0]
            for i, (residue, pred, prob) in enumerate(zip(sequence, predictions, probabilities)):
                if isinstance(prob, list):
                    max_prob = max(prob)
                    predicted_label = prob.index(max_prob)
                else:
                    max_prob = prob
                    predicted_label = pred
                
                expanded_rows.append({
                    'index': i,
                    'residue': residue,
                    'predicted_label': predicted_label,
                    'probability': max_prob,
                })
                
        except Exception as e:
            print(f"Error processing row for {header}: {e}")
            continue
    
    return pd.DataFrame(expanded_rows)

def handle_protein_residue_function_prediction(
    task: str,
    fasta_file: Any,
    enable_ai: bool,
    ai_model: str,
    user_api_key: str,
    model_name: Optional[str] = None,
    progress=gr.Progress()
) -> Generator:
    try:
        import requests
        requests.post("/api/stats/track", json={"module": "function_analysis"})
    except Exception:
        pass

    """Handle protein residue function prediction workflow."""
    model = model_name if model_name else "ESM2-650M"

    if not all([task, fasta_file]):
        yield(
           "❌ Error: Task and FASTA file are required.", 
            pd.DataFrame(), None,
            gr.update(visible=False), 
            "Please provide all required inputs.",
            "AI Analysis disabled." 
        )
        return

    all_results_list = []
    temp_dir = Path("temp_outputs")
    temp_dir_ = temp_dir /  "Residue_save"
    timestamp = str(int(time.time()))
    residue_save_dir = temp_dir_ / timestamp
    residue_save_dir.mkdir(parents=True, exist_ok=True)

    progress(0.1, desc="Running Prediction...")
    yield(
        f"🚀 Starting predictions with {model}...", 
        pd.DataFrame(), None,
        gr.update(visible=False), 
        "AI analysis will appear here...",
        "AI Analysis disabled."
    )

    
    # Run predictions for each dataset
    progress(0.1, desc="Running Predicrtion...")
    yield(
        f"⏳ Running prediction...", 
        pd.DataFrame(), None,
        gr.update(visible=False), 
        "AI analysis will appear here...",
        "AI Analysis disabled."
    )
    
    try:
        model_key = MODEL_MAPPING_FUNCTION.get(model)
        if not model_key:
            raise ValueError(f"Model key not found for {model}")
        
        adapter_key = MODEL_ADAPTER_MAPPING_FUNCTION[model_key]

        # Get residue task dataset
        datasets = RESIDUE_MAPPING_FUNCTION.get(task, [])
        if not datasets:
            raise ValueError(f"No datasets found for task: {task}")
        
        for dataset in datasets:
            script_path = Path("src") / "property" / f"{model_key}.py"
            adapter_path = Path("ckpt") / dataset / adapter_key
            output_file = residue_save_dir/ f"{dataset}_{model}.csv"

            if not script_path.exists() or not adapter_path.exists():
                raise FileNotFoundError(f"Required files not found: Script={script_path}, Adapter={adapter_path}")
            if isinstance(fasta_file, str):
                file_path = fasta_file
            else:
                file_path = fasta_file.name
            
            cmd = [sys.executable, str(script_path),
                 "--fasta_file", str(Path(file_path)), 
                 "--adapter_path", str(adapter_path), 
                 "--output_csv", str(output_file)]
            subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8', errors='ignore')
            
            if output_file.exists():
                df = pd.read_csv(output_file) 
                df["Task"] = task
                df["Dataset"] = dataset
                all_results_list.append(df)
                os.remove(output_file)

    except Exception as e:
        error_detail = e.stderr if isinstance(e, subprocess.CalledProcessError) else str(e)
        print(f"Failed to process '{task}': {error_detail}")
        all_results_list.append(pd.DataFrame([{"Task": task, "header": "ERROR", "residue": error_detail, "probability": 0}]))
        yield (
            f"❌ Prediction failed: {error_detail}",
            pd.DataFrame(), None,
            gr.update(visible=False),
            "An error occurred during prediction.",
            "AI Analysis disabled.",
        )
        return
    
    progress(0.7, desc="Processing results...")
    if all_results_list:
        combined_df = pd.concat(all_results_list, ignore_index=True)
        final_df = expand_residue_predictions(combined_df)
    else:
        final_df = pd.DataFrame()
    download_path = residue_save_dir / f"prediction_results.csv"
    final_df.to_csv(download_path, index=False)

    progress(0.8, desc="Creating visualization...")

    try:
        residue_plot = generate_plots_for_residue_results(final_df, task)
        plot_update = gr.update(value=residue_plot, visible=True)
    except Exception as e:
        print(f"Visualization error: {e}")
        plot_update = gr.update(visible=False)
    
    display_df = final_df.copy()
    column_rename = {
        'index': 'Position',
        'residue': 'Residue',
        'predicted_label': 'Predicted Label',
        'probability': 'Probability',
    }
    display_df.rename(columns=column_rename, inplace=True)
    if 'Probability' in display_df.columns:
        display_df['Probability'] = display_df['Probability'].round(3)
    ai_summary = "AI Analysis disabled. Enable in settings to generate a report."
    ai_response = "AI Analysis disabled."  # Initialize ai_response variable

    expert_analysis = "<div style='height: 300px; display: flex; align-items: center; justify-content: center; color: #666;'>Analysis will appear here once prediction is complete...</div>"

    if enable_ai:
        progress(0.8, desc="Generate AI summary...")
        yield (
            "🤖 Expert is analyzing results...", 
            display_df, None,
            gr.update(visible=False), 
            expert_analysis,
            "AI Analysis in progress..."
        )
        api_key = get_api_key(ai_model, user_api_key)
        if not api_key: 
            ai_summary = f"❌ No API key found for {ai_model}."
        else:
            ai_config = AIConfig(api_key, ai_model, AI_MODELS[ai_model]["api_base"], AI_MODELS[ai_model]["model"])
            prompt = generate_expert_analysis_prompt_residue(display_df, task)
            ai_response = call_ai_api(ai_config, prompt)
            expert_analysis = format_expert_response(ai_response)
        progress(0.9, desc="Finalizing AI analysis...")
    else:
        progress(1.0, desc="Complete!")
    
    final_status = "✅ All predictions completed!"
    progress(1.0, desc="Complete!")
    yield(
        final_status,
        display_df,
        residue_plot,
        gr.update(visible=True, value=download_path),
        expert_analysis,
        ai_response
    )


def run_protein_properties_prediction(task_type: str, file_path: str) -> Tuple[str, str]:
    """Run protein properties prediction"""
    try:
       # Ensure temp_outputs directory exists
        temp_dir = Path("temp_outputs")
        temp_dir_ = temp_dir / "Protein_properties"
        timestamp = str(int(time.time()))
        properties_dir = temp_dir_ / timestamp
        properties_dir.mkdir(parents=True, exist_ok=True)
        output_json = properties_dir / f"{task_type.replace(' ', '_').replace('(', '').replace(')', '')}.json"
       
        script_name = PROTEIN_PROPERTIES_MAP_FUNCTION.get(task_type)
        if not script_name:
           return "", f"Error: Task '{task_type}' is not allowed"
       
        script_path = f"src/property/{script_name}.py"
        if not os.path.exists(script_path):
           return "", f"Script not found: {script_path}"

        # Determine file argument based on file extension
        file_argument = "--fasta_file" if file_path.lower().endswith((".fasta", ".fa")) else "--pdb_file"
       
        cmd_save = [
           sys.executable, script_path,
           file_argument, file_path,
           "--chain_id", "A",
           "--output_file", str(output_json)
        ]
       
        # Run the command and capture output for debugging
        result = subprocess.run(
           cmd_save,
           capture_output=True,
           text=True,
           encoding="utf-8",
           errors="ignore"
        )

        # Check if the command failed
        if result.returncode != 0:
            error_msg = f"Script execution failed (return code: {result.returncode})"
            if result.stderr:
                error_msg += f"\nStderr: {result.stderr}"
            if result.stdout:
                error_msg += f"\nStdout: {result.stdout}"
            return "", error_msg

        # Check if output file was created
        if os.path.exists(output_json):
            return output_json, ""
        else:
            # Provide more detailed error information
            error_msg = "JSON output file was not generated"
            if result.stdout:
                error_msg += f"\nScript output: {result.stdout}"
            if result.stderr:
                error_msg += f"\nScript errors: {result.stderr}"
            return "", error_msg

    except Exception as e:
        return "", f"Error: {str(e)}"

def format_physical_chemical_properties(data: dict) -> str:
    """Format physical and chemical properties results"""
    result = ""
    result += f"Sequence length: {data['sequence_length']} aa\n"
    result += f"Molecular weight: {data['molecular_weight'] / 1000:.2f} kDa\n"
    result += f"Theoretical pI: {data['theoretical_pI']}\n"
    result += f"Aromaticity: {data['aromaticity']}\n"
    result += f"Instability index: {data['instability_index']}\n"
    
    if data['instability_index'] > 40:
        result += "  ⚠️ Predicted as unstable protein\n"
    else:
        result += "  ✅ Predicted as stable protein\n"
    
    result += f"GRAVY: {data['gravy']}\n"
    
    ssf = data['secondary_structure_fraction']
    result += f"Secondary structure prediction: Helix={ssf['helix']}, Turn={ssf['turn']}, Sheet={ssf['sheet']}\n"
    
    return result


def format_rsa_results(data: dict) -> str:
    """Format RSA calculation results"""
    result = ""
    result += f"Exposed residues: {data['exposed_residues']}\n"
    result += f"Buried residues: {data['buried_residues']}\n"
    result += f"Total residues: {data['total_residues']}\n"
    
    # Sort residues by number for display
    try:
        sorted_residues = sorted(data['residue_rsa'].items(), key=lambda x: int(x[0]))
    except ValueError:
        sorted_residues = sorted(data['residue_rsa'].items())
    
    for res_id, res_data in sorted_residues:
        aa = res_data['aa']
        rsa = res_data['rsa']
        location = "Exposed (surface)" if rsa >= 0.25 else "Buried (core)"
        result += f"  Residue {res_id} ({aa}): RSA = {rsa:.3f} ({location})\n"
    return result


def format_sasa_results(data: dict) -> str:
    """Format SASA calculation results"""
    result = ""
    result += f"{'Chain':<6} {'Residue':<12} {'SASA (Ų)':<15}\n"
    
    for chain_id, chain_data in sorted(data['chains'].items()):
        result += f"--- Chain {chain_id} (Total SASA: {chain_data['total_sasa']:.2f} Ų) ---\n"
        
        # Sort residues by number for display
        try:
            sorted_residues = sorted(chain_data['residue_sasa'].items(), key=lambda x: int(x[0]))
        except ValueError:
            sorted_residues = sorted(chain_data['residue_sasa'].items())
        
        for res_num, res_data in sorted_residues:
            res_id_str = f"{res_data['resname']}{res_num}"
            result += f"{chain_id:<6} {res_id_str:<12} {res_data['sasa']:<15.2f}\n"

    return result


def format_secondary_structure_results(data: dict) -> str:
    """Format secondary structure calculation results"""
    result = f"Successfully calculated secondary structure for chain '{data['chain_id']}'\n"
    result += f"Sequence length: {len(data['aa_sequence'])}\n"
    result += f"Helix (H): {data['ss_counts']['helix']} ({data['ss_counts']['helix']/len(data['aa_sequence'])*100:.1f}%)\n"
    result += f"Sheet (E): {data['ss_counts']['sheet']} ({data['ss_counts']['sheet']/len(data['aa_sequence'])*100:.1f}%)\n"
    result += f"Coil (C): {data['ss_counts']['coil']} ({data['ss_counts']['coil']/len(data['aa_sequence'])*100:.1f}%)\n"
    
    # Sort residues by number for display
    try:
        sorted_residues = sorted(data['residue_ss'].items(), key=lambda x: int(x[0]))
    except ValueError:
        sorted_residues = sorted(data['residue_ss'].items())
    
    for res_id, res_data in sorted_residues:
        result += f"  Residue {res_id} ({res_data['aa_seq']}): ss8: {res_data['ss8_seq']} ({res_data['ss8_name']}), ss3: {res_data['ss3_seq']}\n"
    
    result += f"aa_seq: {data['aa_sequence']}\n"
    result += f"ss8_seq: {data['ss8_sequence']}\n"
    result += f"ss3_seq: {data['ss3_sequence']}\n"
    
    return result


def handle_protein_properties_generation(
    task: str,
    file_obj: Any,
    progress = gr.Progress()
    ) -> Generator:
    """
    Handle protein properties generation with progress updates
    """
    try:
        if not file_obj or not task:
            yield (
                "❌ Error: Function and file are required.", 
                None, gr.update(visible=False), None, 
                "Please select a function and upload a file."
            )
            return
        
        if isinstance(file_obj, str):
            file_path = file_obj
        else:
            file_path = file_obj.name
        
        if file_path.lower().endswith((".fasta", ".fa")):
            if "PDB only" in task:
                yield (
                    "❌ Error: Unsupported task.", 
                    None, gr.update(visible=False), None, 
                    f"Task '{task}' requires a PDB file, but a FASTA file was provided."
                )
                return
        
        yield (
            f"⏳ Running prediction...", 
            None, gr.update(visible=False), None, 
            "Prediction in progress..."     
        )
        progress(0.1, desc="Running prediction...")
        output_json_path, error_message = run_protein_properties_prediction(task, file_path)
        progress(0.7, desc="Processing results...")
        
        if error_message:
            yield (f"❌ Error: {error_message}", None, gr.update(visible=False),
                None, error_message
            )
            return
        
        progress(0.9, desc="Finalizing results...")
        
        # Load and format results based on task type
        formatted_result = ""
        if output_json_path and os.path.exists(output_json_path):
            try:
                with open(output_json_path, 'r') as f:
                    results_data = json.load(f)
                
                # Format results based on the task type
                if task == "Physical and chemical properties":
                    formatted_result = format_physical_chemical_properties(results_data)
                elif task == "Relative solvent accessible surface area (PDB only)":
                    formatted_result = format_rsa_results(results_data)
                elif task == "SASA value (PDB only)":
                    formatted_result = format_sasa_results(results_data)
                elif task == "Secondary structure (PDB only)":
                    formatted_result = format_secondary_structure_results(results_data)
                else:
                    formatted_result = f"Results saved to: {os.path.basename(output_json_path)}"
                    
            except Exception as e:
                formatted_result = f"Results saved to: {os.path.basename(output_json_path)}\n(Error formatting display: {str(e)})"
        
        download_visible = bool(output_json_path and os.path.exists(output_json_path))
        download_update = gr.update(visible=download_visible,
                                    value = output_json_path if download_visible else None)
        final_status = f"✅ {task} completed successfully!"
        if output_json_path:
            final_status += f" Results saved to {os.path.basename(output_json_path)}"
        
        progress(1.0, desc="Complete!")

        yield (final_status, formatted_result, download_update,
                output_json_path if download_visible else None,
                "Prediction completed successfully.")

    except Exception as e:
        yield (
            f"❌ Unexpected error: {str(e)}", None, gr.update(visible=False),
            None, f"An unexpected error occurred: {str(e)}"
        )



def create_quick_tool_tab(constant: Dict[str, Any]) -> Dict[str, Any]:
    zero_shot_model = list(MODEL_MAPPING_ZERO_SHOT.keys())
    with gr.Blocks() as demo:
        with gr.Tabs():
            with gr.TabItem("Intelligent Directed Evolution"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2):
                        gr.Markdown("### Model Configuration")
                        zero_shot_function_dd = gr.Dropdown(choices=DATASET_MAPPING_ZERO_SHOT, label="Select Protein Function", value=DATASET_MAPPING_ZERO_SHOT[0])
                        zero_shot_model_dd = gr.Dropdown(choices=zero_shot_model, label="Select Structure-based Model", visible=False)
                        gr.Markdown("**Data Input**")
                        with gr.Tabs():
                            with gr.TabItem("Upload Protein File (.fasta, .fa, .pdb)"):
                                easy_zshot_file_upload = gr.File(label="Upload Protein File (.fasta, .fa, .pdb)", file_types=[".fasta", ".fa", ".pdb"])
                                easy_zshot_file_example = gr.Examples(examples=[["./download/P60002.fasta"]], inputs=easy_zshot_file_upload, label="Click example to load")
                            with gr.TabItem("Paste Protein Content"):
                                easy_zshot_paste_content_input =  gr.Textbox(label="Paste Protein Content", placeholder="Paste protein content here...", lines=8, max_lines=15)
                                with gr.Row():
                                    easy_zshot_paste_content_btn = gr.Button("🔍 Detect Content", variant="primary", size="m")
                                    easy_zshot_paste_clear_btn = gr.Button("🗑️ Clear", variant="primary", size="m")

                        easy_zshot_protein_display = gr.Textbox(label="Uploaded Protein Sequence", interactive=False, lines=3, max_lines=7)
                        easy_zshot_sequence_selector = gr.Dropdown(label="Select Chain", choices=["Sequence 1"], value="Sequence 1", visible=False, allow_custom_value=True)
                        easy_zshot_original_file_path_state = gr.State("")
                        easy_zshot_original_paste_content_state = gr.State("")
                        easy_zshot_selected_sequence_state = gr.State("Sequence 1")
                        easy_zshot_sequence_state = gr.State({})
                        easy_zshot_current_file_state = gr.State("")
                        
                        gr.Markdown("### Configure AI Analysis (Optional)")
                        with gr.Accordion("AI Settings", open=True):
                            enable_ai_zshot = gr.Checkbox(label="Enable AI Summary", value=False)
                            with gr.Group(visible=False) as ai_box_zshot:
                                ai_model_dd_zshot = gr.Dropdown(
                                    choices=list(AI_MODELS.keys()), 
                                    value="DeepSeek", 
                                    label="Select AI Model"
                                )
                                ai_status_zshot = gr.Markdown(
                                    value="✓ Using provided API Key" if os.getenv("DEEPSEEK_API_KEY") else "⚠ No API Key found in .env file",
                                    visible=True
                                )
                                api_key_in_zshot = gr.Textbox(
                                    label="API Key", 
                                    type="password", 
                                    placeholder="Enter your API Key if needed",
                                    visible=not bool(os.getenv("DEEPSEEK_API_KEY"))
                                )
                        easy_zshot_predict_btn = gr.Button("🚀 Start Prediction", variant="primary")

                    with gr.Column(scale=3):
                        gr.Markdown("### Results")
                        zero_shot_status_box = gr.Textbox(label="Status", interactive=False)
                        
                        with gr.Tabs():
                            with gr.TabItem("📊 Raw Results"):
                                zero_shot_df_out = gr.DataFrame(label="Raw Data")
                            with gr.TabItem("📈 Prediction Heatmap"):
                                zero_shot_plot_out = gr.Plot(label="Heatmap")
                            with gr.TabItem("👨‍🔬 AI Expert Analysis"):
                                # function_results_plot = gr.Plot(label="Confidence Scores")
                                zero_shot_ai_expert_html = gr.HTML(
                                    value="<div style='height: 300px; display: flex; align-items: center; justify-content: center; color: #666;'>AI analysis will appear here...</div>",
                                    label="👨‍🔬 AI Expert Analysis"
                                )                        
                        zero_shot_download_btn = gr.DownloadButton("💾 Download Results", visible=False)
                        zero_shot_download_path_state = gr.State()
                        zero_shot_view_controls = gr.State() # Placeholder for potential future controls
                        zero_shot_full_data_state = gr.State()

            with gr.TabItem("Protein Function Prediction"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2):
                        gr.Markdown("**Model Configuration**")
                        easy_func_task_dd = gr.Dropdown(choices=list(DATASET_MAPPING_FUNCTION.keys()), label="Select Task", value="Solubility")
                        gr.Markdown("**Data Input**")
                        with gr.Tabs():
                            with gr.TabItem("Upload FASTA File"):
                                base_function_fasta_upload = gr.File(label="Upload FASTA file", file_types=[".fasta", ".fa"])
                                base_function_file_example = gr.Examples(examples=[["./download/P60002.fasta"]], inputs=base_function_fasta_upload, label="Click example to load")
                            with gr.TabItem("Paste FASTA Content"):
                                base_func_paste_content_input = gr.Textbox(label="Paste FASTA Content", placeholder="Paste FASTA content here...", lines=8, max_lines=15)
                                with gr.Row():
                                    base_func_paste_content_btn = gr.Button("🔍 Detect & Save Content", variant="primary", size="m")
                                    base_func_paste_clear_btn = gr.Button("🗑️ Clear", variant="primary", size="m")
                        
                        base_function_protein_display = gr.Textbox(label="Uploaded Protein Sequence", interactive=False, lines=3, max_lines=7)
                        base_function_selector = gr.Dropdown(label="Select Chain", choices=["Sequence 1"], value="Sequence 1", visible=False, allow_custom_value=True)
                        base_function_original_file_path_state = gr.State("")
                        base_function_original_paste_content_state = gr.State("")
                        base_function_selected_sequence_state = gr.State("Sequence 1")
                        base_function_sequence_state = gr.State({})
                        base_function_current_file_state = gr.State("")

                        gr.Markdown("### Configure AI Analysis (Optional)")
                        with gr.Accordion("AI Settings", open=True):
                            enable_ai_func = gr.Checkbox(label="Enable AI Summary", value=False)
                            with gr.Accordion("AI Settings", open=True):
                                with gr.Group(visible=False) as ai_box_func:
                                    ai_model_dd_func = gr.Dropdown(
                                        choices=list(AI_MODELS.keys()), 
                                        label="Select AI Model", 
                                        value="DeepSeek"
                                    )
                                    ai_status_func = gr.Markdown(
                                        value="✓ Using provided API Key" if os.getenv("DEEPSEEK_API_KEY") else "⚠ No API Key found in .env file",
                                        visible=True
                                    )
                                    api_key_in_func = gr.Textbox(
                                        label="API Key", 
                                        type="password", 
                                        placeholder="Enter your API Key if needed",
                                        visible=not bool(os.getenv("DEEPSEEK_API_KEY"))
                                    )
                        easy_func_predict_btn = gr.Button("🚀 Start Prediction", variant="primary")
                
                    with gr.Column(scale=3):
                        gr.Markdown("### Results")
                        function_status_textbox = gr.Textbox(label="Status", interactive=False)
                        with gr.Tabs():
                            with gr.TabItem("📊 Raw Results"):
                                function_results_df = gr.DataFrame(label="Prediction Data", column_widths=["20%", "20%", "20%", "20%", "20%"])
                            with gr.TabItem("👨‍🔬 AI Expert Analysis"):
                                function_ai_expert_html = gr.HTML(
                                    value="<div style='height: 300px; display: flex; align-items: center; justify-content: center; color: #666;'>AI analysis will appear here...</div>",
                                    label="👨‍🔬 AI Expert Analysis"
                                )
                        function_download_btn = gr.DownloadButton("💾 Download Results", visible=False)

            with gr.TabItem("Functional Residue Prediction"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2):
                        gr.Markdown("### Model Configuration")
                        base_residue_function_task_dd = gr.Dropdown(choices=list(RESIDUE_MAPPING_FUNCTION.keys()), label="Select Task", value="Activity Site")
                        gr.Markdown("**Data Input**")
                        with gr.Tabs():
                            with gr.TabItem("Upload FASTA File"):
                                base_residue_function_fasta_upload = gr.File(label="Upload Fasta file", file_types=[".fasta", ".fa"])
                                base_residue_function_file_exmaple = gr.Examples(examples=[["./download/P60002.fasta"]], inputs=base_residue_function_fasta_upload, label="Click example to load")
                            with gr.TabItem("Paste FASTA Content"):
                                base_residue_function_paste_content_input = gr.Textbox(label="Paste FASTA Content", placeholder="Paste FASTA content here...", lines=8, max_lines=15)
                                with gr.Row():
                                    base_residue_function_paste_content_btn = gr.Button("🔍 Detect & Save Content", variant="primary", size="m")
                                    base_residue_function_paste_clear_btn = gr.Button("🗑️ Clear", variant="primary", size="m")
                        
                        base_residue_function_protein_display = gr.Textbox(label="Uploaded Protein Sequence", interactive=False, lines=3, max_lines=7)
                        base_residue_function_selector = gr.Dropdown(label="Select Chain", choices=["Sequence 1"], value="Sequence 1", visible=False, allow_custom_value=True)
                        base_residue_function_original_file_path_state = gr.State("")
                        base_residue_function_original_paste_content_state = gr.State("")
                        base_residue_function_selected_sequence_state = gr.State("Sequence 1")
                        base_residue_function_sequence_state = gr.State({})
                        base_residue_function_current_file_state = gr.State("")

                        gr.Markdown("### Configure AI  Analysis (Optional)")
                        with gr.Accordion("AI Settings", open=True):
                            enable_ai_residue_function = gr.Checkbox(label="Enable AI Summary", value=False)
                            with gr.Accordion("AI  Settings", open=True):
                                with gr.Group(visible=False) as ai_box_residue_function:
                                    ai_model_dd_residue_function = gr.Dropdown(
                                        choices=list(AI_MODELS.keys()),
                                        label="Select AI Model",
                                        value="DeepSeek"
                                    )
                                    ai_status_residue_function = gr.Markdown(
                                        value="✓ Using provided API Key" if os.getenv("DEEPSEEK_API_KEY") else "⚠ No API Key found in .env file",
                                        visible=True
                                    )
                                    api_key_in_residue_function = gr.Textbox(
                                        label="API Key",
                                        type="password",
                                        placeholder="Ener your API Key if needed",
                                        visible=not bool(os.getenv("DEEPSEEK_API_KEY"))
                                    )
                        base_residue_function_predict_btn = gr.Button("🚀 Start Prediction", variant="primary")
                    
                    with gr.Column(scale=3):
                        gr.Markdown("### Results")
                        base_residue_function_status_textbox = gr.Textbox(label="Status", interactive=False)
                        with gr.Tabs():
                            with gr.TabItem("📊 Raw Results"):
                                base_residue_function_results_df = gr.DataFrame(label="Prediction Data", column_widths=["20%", "20%", "20%", "20%", "20%"])
                            with gr.TabItem("📈 Prediction Heatmap"):
                                base_residue_function_plot_out = gr.Plot(label="Heatmap")
                            with gr.TabItem("👨‍🔬 AI Expert Analysis"):
                                base_residue_function_ai_expert_html = gr.HTML(
                                    value="<div style='height: 300px; display: flex; align-items: center; justify-content: center; color: #666;'>AI analysis will appear here...</div>",
                                    label="👨‍🔬 AI Expert Analysis"
                                )
                        base_residue_function_download_btn = gr.DownloadButton("💾 Download Results", visible=False)


            with gr.TabItem("Physicochemical Property Analysis"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2):
                        gr.Markdown("**Task Configuration**")
                        protein_properties_choices = list(PROTEIN_PROPERTIES_FUNCTION)
                        protein_properties_task_dd = gr.Dropdown(choices=protein_properties_choices, label="Select Properties of Protein", value=protein_properties_choices[0])
                        
                        gr.Markdown("**Data Input**")
                        with gr.Tabs():
                            with gr.TabItem("Upload Protein File (.fasta, .fa, .pdb)"):
                                protein_properties_file_upload = gr.File(label="Upload Protein File (.fasta, .fa, .pdb)", file_types=[".fasta", ".fa", ".pdb"])
                                
                                protein_properties_file_example = gr.Examples(examples=[["./download/P60002.fasta"]], inputs=protein_properties_file_upload, label="Click example to load")

                            with gr.TabItem("Paste Protein Content"):
                                protein_properties_paste_content_input = gr.Textbox(label="Pasta Protein Content", placeholder="Paste protein content here...", lines=8, max_lines=15)
                                with gr.Row():
                                    protein_properties_paste_content_btn = gr.Button("🔍 Detect Content", variant="primary", size="m")
                                    protein_properties_paste_clear_btn = gr.Button("🗑️ Clear", variant="primary", size="m")
                        
                        
                        protein_properties_protein_display = gr.Textbox(label="Uploaded Protein Sequence", interactive=False, lines=3, max_lines=7)
                        protein_properties_sequence_selector = gr.Dropdown(label="Select Chain", choices=["Sequence 1"], value="Sequence 1", visible=False, allow_custom_value=True)
                        protein_properties_original_file_path_state = gr.State("")
                        protein_properties_original_paste_content_state = gr.State("")
                        protein_properties_selected_sequence_state = gr.State("Sequence 1")
                        protein_properties_sequence_state = gr.State({})
                        protein_properties_current_file_state = gr.State("")

                        protein_properties_predict_btn = gr.Button("🚀 Start Prediction", variant="primary")
                    with gr.Column(scale=3):
                        gr.Markdown("### Results")
                        protein_properties_status_box = gr.Textbox(label="Status", interactive=False)
                        
                        with gr.Tabs():
                            with gr.TabItem("📊 Raw Results"):
                                protein_properties_result_out = gr.Textbox(
                                                                label="Raw Data", 
                                                                interactive=False, 
                                                                lines=15
                                                            )
                        
                        protein_properties_download_btn = gr.DownloadButton("💾 Download Results", visible=False)
                        protein_properties_path_state = gr.State()
                        protein_properties_view_controls = gr.State()
                        protein_properties_full_data_state = gr.State()
        
        def clear_paste_content_pdb():
            return "", "", gr.update(choices=["A"], value="A", visible=False), {}, "A", ""

        def clear_paste_content_fasta():
            return "No file selected", "No file selected", gr.update(choices=["Sequence 1"], value="Sequence 1", visible=False), {}, "Sequence 1", ""

        def toggle_ai_section_simple(is_checked: bool):
            return gr.update(visible=is_checked)
        
        def on_ai_model_change_simple(ai_provider: str) -> tuple:
            if ai_provider == "DeepSeek":
                return gr.update(visible=False), gr.update(visible=True)
            else:
                return gr.update(visible=True), gr.update(visible=False)
        
        def handle_paste_fasta_detect(fasta_content):
            result = parse_fasta_paste_content(fasta_content)
            return result + (fasta_content, )
        
        enable_ai_zshot.change(fn=toggle_ai_section, inputs=enable_ai_zshot, outputs=ai_box_zshot)
        enable_ai_func.change(fn=toggle_ai_section, inputs=enable_ai_func, outputs=ai_box_func)

        ai_model_dd_zshot.change(
            fn=on_ai_model_change,
            inputs=ai_model_dd_zshot,
            outputs=[api_key_in_zshot, ai_status_zshot]
        )
        ai_model_dd_func.change(
            fn=on_ai_model_change,
            inputs=ai_model_dd_func,
            outputs=[api_key_in_func, ai_status_func]
        )
        enable_ai_residue_function.change(
            fn=toggle_ai_section, 
            inputs=enable_ai_residue_function, 
            outputs=ai_box_residue_function
        )

        ai_model_dd_residue_function.change(
            fn=on_ai_model_change,
            inputs=ai_model_dd_residue_function,
            outputs=[api_key_in_residue_function, ai_status_residue_function]
        )


        easy_zshot_file_upload.upload(
            fn=handle_file_upload, 
            inputs=easy_zshot_file_upload, 
            outputs=[easy_zshot_protein_display, easy_zshot_sequence_selector, easy_zshot_sequence_state, easy_zshot_selected_sequence_state, easy_zshot_original_file_path_state, easy_zshot_current_file_state]
        )
        easy_zshot_file_upload.change(
            fn=handle_file_upload, 
            inputs=easy_zshot_file_upload, 
            outputs=[easy_zshot_protein_display, easy_zshot_sequence_selector, easy_zshot_sequence_state, easy_zshot_selected_sequence_state, easy_zshot_original_file_path_state, easy_zshot_current_file_state]
        )
        easy_zshot_paste_clear_btn.click(
            fn=clear_paste_content_fasta,
            outputs=[easy_zshot_paste_content_input, easy_zshot_protein_display, easy_zshot_sequence_selector, easy_zshot_sequence_state, easy_zshot_selected_sequence_state, easy_zshot_original_file_path_state]
        )
        easy_zshot_paste_content_btn.click(
            fn=handle_paste_fasta_detect,
            inputs=easy_zshot_paste_content_input,
            outputs=[easy_zshot_protein_display, easy_zshot_sequence_selector, easy_zshot_sequence_state, easy_zshot_selected_sequence_state, easy_zshot_original_file_path_state, easy_zshot_original_paste_content_state]
        )

        def handle_sequence_change_unified(selected_chain, chains_dict, original_file_path, original_paste_content):
            # Check for None or empty file path
            if not original_file_path:
                return "No file selected", ""
            
            if original_file_path.endswith('.fasta'):
                if original_paste_content:
                    return handle_paste_sequence_selection(selected_chain, chains_dict, original_paste_content)
                else:
                    return handle_fasta_sequence_change(selected_chain, chains_dict, original_file_path)
            elif original_file_path.endswith('.pdb'):
                if original_paste_content:
                    return handle_paste_chain_selection(selected_chain, chains_dict, original_paste_content)
                else:
                    return handle_pdb_chain_change(selected_chain, chains_dict, original_file_path)
            else:
                # Default case for no file selected
                return "No file selected", ""

        easy_zshot_sequence_selector.change(
            fn=handle_sequence_change_unified,
            inputs=[easy_zshot_sequence_selector, easy_zshot_sequence_state, easy_zshot_original_file_path_state, easy_zshot_original_paste_content_state],
            outputs=[easy_zshot_protein_display, easy_zshot_current_file_state]
        )

        easy_zshot_predict_btn.click(
            fn=handle_mutation_prediction_base,
            inputs=[zero_shot_function_dd, easy_zshot_file_upload, enable_ai_zshot, ai_model_dd_zshot, api_key_in_zshot, zero_shot_model_dd],
            outputs=[zero_shot_status_box, zero_shot_plot_out, zero_shot_df_out, zero_shot_download_btn, zero_shot_download_path_state, zero_shot_view_controls, zero_shot_full_data_state, zero_shot_ai_expert_html],
            show_progress=True
        )

        base_function_fasta_upload.upload(
            fn=handle_file_upload, 
            inputs=base_function_fasta_upload, 
            outputs=[base_function_protein_display, base_function_selector, base_function_sequence_state, base_function_selected_sequence_state, base_function_original_file_path_state, base_function_current_file_state]
        )
        base_function_fasta_upload.change(
            fn=handle_file_upload, 
            inputs=base_function_fasta_upload, 
            outputs=[base_function_protein_display, base_function_selector, base_function_sequence_state, base_function_selected_sequence_state, base_function_original_file_path_state, base_function_current_file_state]
        )

        base_func_paste_clear_btn.click(
            fn=clear_paste_content_fasta,
            outputs=[base_func_paste_content_input, base_function_protein_display, base_function_selector, base_function_sequence_state, base_function_selected_sequence_state, base_function_original_file_path_state]
        )

        base_func_paste_content_btn.click(
            fn=handle_paste_fasta_detect,
            inputs=base_func_paste_content_input,
            outputs=[base_function_protein_display, base_function_selector, base_function_sequence_state, base_function_selected_sequence_state, base_function_original_file_path_state, base_function_original_paste_content_state]
        )

        base_function_selector.change(
            fn=handle_sequence_change_unified,
            inputs=[base_function_selector, base_function_sequence_state, base_function_original_file_path_state, base_function_original_paste_content_state],
            outputs=[base_function_protein_display, base_function_current_file_state]
        )

        easy_func_predict_btn.click(
            fn=handle_protein_function_prediction,
            inputs=[easy_func_task_dd, base_function_current_file_state, enable_ai_func, ai_model_dd_func, api_key_in_func],
            outputs=[function_status_textbox, function_results_df, function_download_btn, function_ai_expert_html, gr.State()], 
            show_progress=True
        )

        base_residue_function_fasta_upload.upload(
            fn=handle_file_upload,
            inputs=base_residue_function_fasta_upload,
            outputs=[base_residue_function_protein_display, base_residue_function_selector, base_residue_function_sequence_state, base_residue_function_selected_sequence_state, base_residue_function_original_file_path_state, base_residue_function_current_file_state]
        )
        base_residue_function_fasta_upload.change(
            fn=handle_file_upload,
            inputs=base_residue_function_fasta_upload,
            outputs=[base_residue_function_protein_display, base_residue_function_selector, base_residue_function_sequence_state, base_residue_function_selected_sequence_state, base_residue_function_original_file_path_state, base_residue_function_current_file_state]
        )
        base_residue_function_paste_clear_btn.click(
            fn=clear_paste_content_fasta,
            outputs=[base_residue_function_paste_content_input, base_residue_function_protein_display, base_residue_function_selector, base_residue_function_sequence_state, base_residue_function_selected_sequence_state, base_residue_function_original_file_path_state]
        )
        base_residue_function_paste_content_btn.click(
            fn=handle_paste_fasta_detect,
            inputs=base_residue_function_paste_content_input,
            outputs=[base_residue_function_protein_display, base_residue_function_selector, base_residue_function_sequence_state, base_residue_function_selected_sequence_state, base_residue_function_original_file_path_state, base_residue_function_original_paste_content_state]
        )
        base_residue_function_selector.change(
            fn=handle_sequence_change_unified,
            inputs=[base_residue_function_selector, base_residue_function_sequence_state, base_residue_function_original_file_path_state, base_residue_function_original_paste_content_state],
            outputs=[base_residue_function_protein_display, base_residue_function_current_file_state]
        )
        base_residue_function_predict_btn.click(
            fn=handle_protein_residue_function_prediction,
            inputs=[base_residue_function_task_dd, base_residue_function_current_file_state, enable_ai_residue_function, ai_model_dd_residue_function, api_key_in_residue_function],
            outputs=[base_residue_function_status_textbox, base_residue_function_results_df, base_residue_function_plot_out,base_residue_function_download_btn, base_residue_function_ai_expert_html, gr.State()],
            show_progress = True
        )

        protein_properties_file_upload.upload(
            fn=handle_file_upload,
            inputs=protein_properties_file_upload,
            outputs=[protein_properties_protein_display, protein_properties_sequence_selector, protein_properties_sequence_state, protein_properties_selected_sequence_state, protein_properties_original_file_path_state, protein_properties_current_file_state]
        )
        protein_properties_file_upload.change(
            fn=handle_file_upload,
            inputs=protein_properties_file_upload,
            outputs=[protein_properties_protein_display, protein_properties_sequence_selector, protein_properties_sequence_state, protein_properties_selected_sequence_state, protein_properties_original_file_path_state, protein_properties_current_file_state]
        )
        protein_properties_paste_clear_btn.click(
            fn=clear_paste_content_fasta,
            outputs=[protein_properties_paste_content_input, protein_properties_protein_display, protein_properties_sequence_selector, protein_properties_sequence_state, protein_properties_selected_sequence_state, protein_properties_original_file_path_state]
        )
        protein_properties_paste_content_btn.click(
            fn=handle_paste_fasta_detect,
            inputs=protein_properties_paste_content_input,
            outputs=[protein_properties_protein_display, protein_properties_sequence_selector, protein_properties_sequence_state, protein_properties_selected_sequence_state, protein_properties_original_file_path_state, protein_properties_original_paste_content_state]
        )

        protein_properties_sequence_selector.change(
            fn=handle_sequence_change_unified,
            inputs=[protein_properties_sequence_selector, protein_properties_sequence_state, protein_properties_original_file_path_state, protein_properties_original_paste_content_state],
            outputs=[protein_properties_protein_display, protein_properties_current_file_state]
        )

        protein_properties_predict_btn.click(
            fn=handle_protein_properties_generation,
            inputs=[protein_properties_task_dd, protein_properties_file_upload],
            outputs=[protein_properties_status_box, protein_properties_result_out, protein_properties_download_btn, protein_properties_path_state, gr.State()],
            show_progress=True
        )

    return {}
