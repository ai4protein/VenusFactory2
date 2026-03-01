import gradio as gr

import gradio as gr
import pandas as pd
import os
import sys
import subprocess
import time
import tarfile
from pathlib import Path
from typing import Dict, Any, List, Generator, Optional, Tuple, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import requests
import re
import json

# Import functions from the existing module
from web.quick_tool_tab import *
from web.utils.file_handlers import extract_sequence_from_pdb
from web.utils.label_mappers import map_labels_individual
from web.utils.ui_helpers import handle_paste_fasta_detect, wrap_upload_show_section, wrap_paste_show_section, wrap_clear_hide_section
from web.utils.llm_helpers import get_api_key, get_chat_base_url, call_llm_api, LLMConfig
from web.utils.common_utils import get_save_path
from web.utils.constants import LLM_MODELS
from web.utils.html_ui import load_html_template
from dotenv import load_dotenv
load_dotenv()

def generate_comprehensive_report(mutation_results: pd.DataFrame, function_results: pd.DataFrame) -> str:
    """Generate a comprehensive analysis report."""
    
    report = "# VenusFactory Comprehensive Analysis Report\n\n"
    
    # Mutation Analysis Section
    if not mutation_results.empty:
        report += "## 🧬 Mutation Prediction Analysis\n\n"
        
        # Get top mutations
        top_mutations = mutation_results.head(20)['Mutant'].tolist()
        mutation_count = len(mutation_results)
        
        report += f"Based on our deep mutational scanning analysis of {mutation_count} possible mutations, "
        report += f"VenusFactory recommends the following top 20 beneficial mutations: "
        report += ", ".join(top_mutations[:10])
        if len(top_mutations) > 10:
            report += f", and {len(top_mutations)-10} additional mutations"
        report += ".\n\n"
        
        # Analysis of mutation patterns
        if 'Mutant' in mutation_results.columns:
            positions = []
            for mutant in mutation_results['Mutant']:
                if isinstance(mutant, str) and re.match(r'^[A-Z]\d+[A-Z]$', mutant):
                    pos = int(re.findall(r'\d+', mutant)[0])
                    positions.append(pos)
            
            if positions:
                position_counts = pd.Series(positions).value_counts().head(5)
                hotspot_positions = position_counts.index.tolist()
                
                report += f"Key findings from mutation analysis:\n"
                report += f"- Identified {len(set(positions))} unique residue positions as mutation targets\n"
                report += f"- Hotspot regions detected at positions: {', '.join(map(str, hotspot_positions))}\n"
                report += f"- Top beneficial mutations show potential for significant functional improvement\n\n"
    
    # Function Prediction Section  
    if not function_results.empty:
        report += "## 🔬 Protein Function Prediction Analysis\n\n"
        
        for _, row in function_results.iterrows():
            if 'Predicted Class' in row and 'Confidence Score' in row and 'Dataset' in row:
                predicted_class = row['Predicted Class']
                confidence = row['Confidence Score']
                dataset = row.get('Dataset', 'Unknown')
                protein_name = row.get('Protein Name', 'Target protein')
                
                if isinstance(confidence, (int, float)):
                    confidence_pct = int(confidence * 100) if confidence <= 1 else int(confidence)
                else:
                    confidence_pct = 50
                
                # Generate task-specific analysis
                if dataset in ["DeepSol", "DeepSoluE", "ProtSolM"]:
                    if predicted_class == "Soluble":
                        report += f"**Solubility Analysis**: We determine that this protein is likely {predicted_class.lower()}, "
                        report += f"with a confidence probability of {confidence_pct}%. This suggests favorable conditions for "
                        report += f"expression and purification in standard laboratory conditions.\n\n"
                    else:
                        report += f"**Solubility Analysis**: Analysis indicates this protein may be {predicted_class.lower()}, "
                        report += f"with a confidence probability of {confidence_pct}%. Consider optimization strategies "
                        report += f"such as fusion tags or alternative expression systems.\n\n"
                
                if dataset in ["DeepLocBinary", "DeepLocMulti"]:
                    report += f"**Subcellular Localization**: Prediction indicates {predicted_class} localization "
                    report += f"with {confidence_pct}% confidence. This localization pattern is consistent with "
                    report += f"the protein's predicted functional role and cellular context.\n\n"
                
                if dataset == "MetalIonBinding":
                    if predicted_class == "Binding":
                        report += f"**Metal Ion Binding**: Strong evidence ({confidence_pct}% confidence) suggests this protein "
                        report += f"has metal ion binding capability. This may be crucial for structural stability "
                        report += f"and catalytic function.\n\n"
                    else:
                        report += f"**Metal Ion Binding**: Analysis suggests limited metal ion binding potential "
                        report += f"({confidence_pct}% confidence for non-binding). The protein likely maintains "
                        report += f"function through other structural mechanisms.\n\n"
                
                if dataset == "Thermostability":
                    report += f"**Thermal Stability**: Predicted thermal stability score indicates "
                    report += f"moderate to high thermal resistance. This suggests the protein can maintain "
                    report += f"structural integrity under elevated temperature conditions.\n\n"
                
                if dataset == "SortingSignal":
                    if predicted_class == "Signal":
                        report += f"**Signal Peptide**: Detected sorting signal with {confidence_pct}% confidence. "
                        report += f"This indicates the protein is likely targeted for secretion or specific "
                        report += f"subcellular compartments.\n\n"
                    else:
                        report += f"**Signal Peptide**: No significant sorting signal detected ({confidence_pct}% confidence). "
                        report += f"The protein is likely to remain in the cytoplasm or require alternative "
                        report += f"targeting mechanisms.\n\n"
                
                if dataset == "DeepET_Topt":
                    report += f"**Optimal Temperature**: Predicted optimal operating temperature provides insights "
                    report += f"into the protein's thermal adaptation and optimal experimental conditions.\n\n"
    
    # Combined Recommendations
    report += "## 💡 Integrated Recommendations\n\n"
    
    if not mutation_results.empty and not function_results.empty:
        report += "Based on the combined mutation and functional analysis:\n\n"
        report += "1. **Protein Engineering Strategy**: The identified beneficial mutations can be systematically "
        report += "introduced to enhance the protein's functional properties while maintaining structural integrity.\n\n"
        report += "2. **Experimental Validation**: We recommend prioritizing the top 5-10 mutations for experimental "
        report += "validation, focusing on those in predicted functional regions.\n\n"
        report += "3. **Functional Optimization**: The functional predictions provide a baseline for measuring "
        report += "improvement after mutation introduction.\n\n"
    elif not mutation_results.empty:
        report += "The mutation analysis provides clear targets for protein engineering experiments. "
        report += "Focus on validating the highest-scoring mutations first.\n\n"
    elif not function_results.empty:
        report += "The functional analysis provides valuable insights into the protein's properties and "
        report += "optimal experimental conditions.\n\n"
    
    report += "## 📊 Analysis Summary\n\n"
    report += f"- **Mutation variants analyzed**: {len(mutation_results) if not mutation_results.empty else 0}\n"
    report += f"- **Functional predictions**: {len(function_results) if not function_results.empty else 0}\n"
    report += f"- **Analysis completed**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    report += "*This analysis was generated by VenusFactory's integrated protein engineering pipeline. "
    report += "Results should be validated through experimental methods.*"
    
    return report

def parse_content_and_update_selector(content: str):
    """Parse content and return appropriate selector options."""
    if not content.strip():
        return "", gr.update(choices=["Sequence 1"], value="Sequence 1", visible=False), {}, "Sequence 1", ""
    
    # Check if content is PDB
    if content.strip().startswith('ATOM'):
        sequence, selector_update, sequences_dict, selected_chain, file_path = parse_pdb_paste_content(content)
        # Make selector visible if there are multiple chains
        if len(sequences_dict) > 1:
            selector_update = gr.update(choices=list(sequences_dict.keys()), value=selected_chain, visible=True)
        return sequence, selector_update, sequences_dict, selected_chain, file_path
    # Check if content is FASTA
    elif content.strip().startswith('>'):
        sequence, selector_update, sequences_dict, selected_chain, file_path = parse_fasta_paste_content(content)
        # Make selector visible if there are multiple sequences
        if len(sequences_dict) > 1:
            selector_update = gr.update(choices=list(sequences_dict.keys()), value=selected_chain, visible=True)
        return sequence, selector_update, sequences_dict, selected_chain, file_path
    else:
        # Assume it's a raw sequence
        sequence = ''.join(c.upper() for c in content if c.isalpha())
        if sequence:
            return sequence, gr.update(choices=["Sequence 1"], value="Sequence 1", visible=False), {"Sequence 1": sequence}, "Sequence 1", ""
        else:
            return "No valid sequence found", gr.update(choices=["Sequence 1"], value="Sequence 1", visible=False), {}, "Sequence 1", ""
def handle_individual_mutation_prediction(content, selected_chain, current_file, sequence_state, original_content):
    """Handle individual mutation prediction."""
    if not content.strip():
        return "Error: Please provide protein sequence or upload a file."
   
    timestamp = str(int(time.time()))
    input_data_dir = get_save_path("Report", "Upload_data")

    is_pdb = content.strip().startswith('ATOM') or (current_file and current_file.endswith('.pdb'))
   
    if is_pdb:
        input_file = input_data_dir / f"input_pdb_{timestamp}.pdb"
        if current_file and current_file.endswith('.pdb'):
            with open(current_file, 'r') as f:
                pdb_content = f.read()
        else:
            pdb_content = content
       
        with open(input_file, 'w') as f:
            f.write(pdb_content)
       
        model_name = "ESM-IF1"
        model_type = "structure"
    else:
        input_file = input_data_dir / f"input_fasta_{timestamp}.fasta"
        if current_file and current_file.endswith(('.fasta', '.fa')):
            with open(current_file, 'r') as f:
                fasta_content = f.read()
        else:
            fasta_content = f">{selected_chain}\n{content}"
       
        with open(input_file, 'w') as f:
            f.write(fasta_content)
       
        model_name = "ESM2-650M"
        model_type = "sequence"
   
    try:
        status, raw_df = run_zero_shot_prediction(model_type, model_name, str(input_file))
       
        if raw_df.empty:
            return f"Mutation prediction failed: {status}"
       
        top_20_df = raw_df.head(20)
        mutation_count = len(raw_df)
       
        result = f"# 📊 Mutation Prediction Results\n\n"
        result += f"**Total mutations analyzed: {mutation_count}**\n\n"
       
        # Create side-by-side table
        result += "| 🏆 **Top 10 Mutations (Rank 1-10)** | 🥈 **Next 10 Mutations (Rank 11-20)** |\n"
        result += "|:-------------------------------------|:---------------------------------------|\n"
        
        # Split data
        top_10_df = top_20_df.head(10)
        next_10_df = top_20_df.tail(10) if len(top_20_df) > 10 else None
        
        # Create rows for side-by-side display
        max_rows = 10
        for i in range(max_rows):
            # Left column (Top 10)
            if i < len(top_10_df):
                left_row = top_10_df.iloc[i]
                left_mutant = left_row['mutant']
                left_score = left_row['esm2_score']
                
                # Parse mutation
                match = re.match(r'([A-Z])(\d+)([A-Z])', left_mutant)
                if match:
                    wild_type, position, mutation = match.groups()
                    left_details = f"{wild_type}→{mutation} (pos {position})"
                else:
                    left_details = left_mutant
                
                # left_content = f"**#{i+1}** `{left_mutant}` <br/>Score: **{left_score:.4f}** <br/>{left_details}"
                left_content = f"**#{i+1}** `{left_mutant}`"
            else:
                left_content = ""
            
            # Right column (Next 10)
            if next_10_df is not None and i < len(next_10_df):
                right_row = next_10_df.iloc[i]
                right_mutant = right_row['mutant']
                right_score = right_row['esm2_score']
                
                # Parse mutation
                match = re.match(r'([A-Z])(\d+)([A-Z])', right_mutant)
                if match:
                    wild_type, position, mutation = match.groups()
                    right_details = f"{wild_type}→{mutation} (pos {position})"
                else:
                    right_details = right_mutant
                
                # right_content = f"**#{i+11}** `{right_mutant}` <br/>Score: **{right_score:.4f}** <br/>{right_details}"
                right_content = f"**#{i+11}** `{right_mutant}`"
            else:
                right_content = ""
            
            # Add table row
            result += f"| {left_content} | {right_content} |\n"
        
        return result
       
    except Exception as e:
        return f"Error in mutation prediction: {str(e)}"

# map_labels_individual is now imported from utils.label_mappers

def handle_individual_function_prediction(content, selected_chain, current_file, sequence_state, original_content):
    """Handle individual function prediction."""
    if not content.strip():
        return "Error: Please provide protein sequence or upload a file."
    
    timestamp = str(int(time.time()))
    input_data_dir = get_save_path("Report", "Upload_data")
    output_data_dir = get_save_path("Report", "Results")
    
    is_pdb = content.strip().startswith('ATOM') or (current_file and current_file.endswith('.pdb'))
    
    if is_pdb:
        # Extract sequence from PDB
        if current_file and current_file.endswith('.pdb'):
            with open(current_file, 'r') as f:
                pdb_content = f.read()
        else:
            pdb_content = content
        
        sequence = extract_sequence_from_pdb(pdb_content)
        fasta_content = f">{selected_chain}\n{sequence}"
    else:
        if current_file and current_file.endswith(('.fasta', '.fa')):
            with open(current_file, 'r') as f:
                fasta_content = f.read()
        else:
            fasta_content = f">{selected_chain}\n{content}"
    
    # Save FASTA file
    fasta_file = input_data_dir / f"function_input_{timestamp}.fasta"
    with open(fasta_file, 'w') as f:
        f.write(fasta_content)
    
    # Run function prediction for all tasks
    all_function_results = []

    for task, datasets in DATASET_MAPPING_FUNCTION.items():
        dataset = datasets[0]
        try:
            model_name = "ESM2-650M"
            model_key = MODEL_MAPPING_FUNCTION.get(model_name)
            adapter_key = MODEL_ADAPTER_MAPPING_FUNCTION[model_key]
            script_path = Path("src") / "tools" / "predict" / "finetuned" / f"{model_key}.py"
            adapter_path = Path("ckpt") / dataset / adapter_key
            output_file = output_data_dir / f"temp_{dataset}_{model_name}_{timestamp}.csv"
            
            if script_path.exists() and adapter_path.exists():
                cmd = [
                    sys.executable, str(script_path), 
                    "--fasta_file", str(fasta_file), 
                    "--adapter_path", str(adapter_path), 
                    "--output_csv", str(output_file)
                ]
                subprocess.run(cmd, capture_output=True, text=True, check=False, encoding='utf-8', errors='ignore')
                if output_file.exists():
                    df = pd.read_csv(output_file)
                    df["Dataset"] = dataset
                    all_function_results.append(df)
                    os.remove(output_file)
            else:
                print("script_path does not exist")
                    
        except Exception as e:
            print(f"Function prediction failed for {dataset}: {e}")
            continue

    if not all_function_results:
        return "Function prediction failed: No results generated."
    
    # Process and format results
    raw_function_df = pd.concat(all_function_results, ignore_index=True)
    # Define regression tasks max-min values for denormalization
    REGRESSION_TASKS_FUNCTION_MAX_MIN = {
        "Stability": [40.1995166, 66.8968874],
        "Optimum temperature": [2, 120]
    }
    # Generate function prediction report in table format    
    # Create table header
    result = ""
    result += "| 🔬 **Task** | 📊 **Result** | 🎯 **Confidence** |\n"
    result += "|:------------|:--------------|:------------------|\n"
    
    # Process each prediction result
    for _, row in raw_function_df.iterrows():
        dataset = row.get('Dataset', 'Unknown')
        current_task = DATASET_TO_TASK_MAP.get(dataset, '')
        
        # Map prediction values to text labels
        predicted_class = map_labels_individual(row, current_task, REGRESSION_TASKS_FUNCTION_MAX_MIN)
        
        # Get confidence score
        confidence_score = row.get('probabilities', 0.5)
        if isinstance(confidence_score, str) and confidence_score not in ['N/A', '']:
            try:
                if confidence_score.startswith('[') and confidence_score.endswith(']'):
                    prob_str = confidence_score.strip('[]')
                    probs = [float(x.strip()) for x in prob_str.split(',')]
                    confidence_score = max(probs)
                else:
                    confidence_score = float(confidence_score)
            except (ValueError, IndexError):
                confidence_score = 0.5
        
        if isinstance(confidence_score, (int, float)):
            confidence_val = confidence_score if confidence_score <= 1 else confidence_score/100
        else:
            confidence_val = 0.5
        
        # Determine task name and format result
        task_name = ""
        result_text = ""
        confidence_text = ""
        
        if dataset in ["DeepSol", "DeepSoluE", "ProtSolM"]:
            task_name = "**Solubility**"
            result_text = f"`{predicted_class.capitalize()}`"
            confidence_text = f"{confidence_val:.3f}"
        
        elif dataset in ["DeepLocBinary", "DeepLocMulti"]:
            task_name = "**Subcellular Localization**"
            result_text = f"`{predicted_class.capitalize()}`"
            confidence_text = f"{confidence_val:.3f}"
        
        elif dataset == "MetalIonBinding":
            task_name = "**Metal Ion Binding**"
            result_text = f"`{predicted_class.capitalize()}`"
            confidence_text = f"{confidence_val:.3f}"
        
        elif dataset == "Thermostability":
            task_name = "**Thermal Stability**"
            result_text = f"`{predicted_class}`"
            confidence_text = "**N/A**"  # Regression task
        
        elif dataset == "SortingSignal":
            task_name = "**Signal Peptide**"
            result_text = f"`{predicted_class.capitalize()}`"
            confidence_text = f"{confidence_val:.3f}"
        
        elif dataset == "DeepET_Topt":
            task_name = "**Optimal Temperature**"
            result_text = f"`{predicted_class}`"
            confidence_text = "**N/A**"  # Regression task
        
        # Add table row
        if task_name:  # Only add if we have a valid task
            result += f"| {task_name} | {result_text} | {confidence_text} |\n"

    
    return result

def handle_functional_residue_prediction(content, selected_chain, current_file, sequence_state, original_content):
    """Handle functional residue prediction."""
    if not content.strip():
        return "Error: Please provide protein sequence or upload a file."
    
    # Prepare FASTA file for function prediction
    timestamp = str(int(time.time()))
    input_data_dir = get_save_path("Report", "Upload_data")
    output_data_dir = get_save_path("Report", "Results")
    
    is_pdb = content.strip().startswith('ATOM') or (current_file and current_file.endswith('.pdb'))
    
    if is_pdb:
        # Extract sequence from PDB
        if current_file and current_file.endswith('.pdb'):
            with open(current_file, 'r') as f:
                pdb_content = f.read()
        else:
            pdb_content = content
        
        sequence = extract_sequence_from_pdb(pdb_content)
        sequence_length = len(sequence)
        fasta_content = f">{selected_chain}\n{sequence}"
    else:
        if current_file and current_file.endswith(('.fasta', '.fa')):
            with open(current_file, 'r') as f:
                fasta_content = f.read()
        else:
            fasta_content = f">{selected_chain}\n{content}"
        
        sequence = content.strip()
        sequence_length = len(sequence)

    # Save FASTA file
    fasta_file = input_data_dir / f"residue_input_{timestamp}.fasta"
    with open(fasta_file, 'w') as f:
        f.write(fasta_content)

    # Run function prediction for all tasks
    all_residue_results = []
    for task, datasets in RESIDUE_MAPPING_FUNCTION.items():
        if not datasets:
            print(f"No datasets found for task: {task}")
            continue

        for dataset in datasets:
            try:
                model_name = "ESM2-650M"
                model_key = MODEL_MAPPING_FUNCTION.get(model_name)
                adapter_key = MODEL_ADAPTER_MAPPING_FUNCTION[model_key]
                script_path = Path("src") / "tools" / "predict" / "finetuned" / f"{model_key}.py"
                adapter_path = Path("ckpt") / dataset / adapter_key
                output_file = output_data_dir / f"temp_{dataset}_{model_name}_{timestamp}.csv"
                
                if script_path.exists() and adapter_path.exists():
                    cmd = [
                        sys.executable, str(script_path), 
                        "--fasta_file", str(fasta_file), 
                        "--adapter_path", str(adapter_path), 
                        "--output_csv", str(output_file)
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True, check=False, encoding='utf-8', errors='ignore')
                    
                    if output_file.exists():
                        df = pd.read_csv(output_file)
                        df["Task"] = task
                        df["Dataset"] = dataset
                        all_residue_results.append(df)
                        os.remove(output_file)
                        
            except Exception as e:
                print(f"Function prediction failed for {dataset}: {e}")
                continue

    if not all_residue_results:
        return "Function prediction failed: No results generated."

    raw_residue_df = pd.concat(all_residue_results, ignore_index=True)
    result = "# Functional Residue  Results \n"

    # Process each prediction result
    for _, row in raw_residue_df.iterrows():
        dataset = row.get("Dataset", "Unknown")
        task = row.get("Task", "Unknown")
        result += f"## {task} \n"
        
        # Get sequence from the row
        row_sequence = row.get('sequence', sequence)
        if isinstance(row_sequence, str) and len(row_sequence) > 0:
            current_sequence = row_sequence
        else:
            current_sequence = sequence
            
        predictions_str = row.get('predicted_class', '')
        probabilities_str = row.get('probabilities', '')
        
        if predictions_str and predictions_str != '':
            try:
                # Parse predictions and probabilities similar to expand_residue_predictions
                import json
                predictions = json.loads(predictions_str) if isinstance(predictions_str, str) else predictions_str
                probabilities = json.loads(probabilities_str) if isinstance(probabilities_str, str) else probabilities_str
                
                # Handle nested list format
                if isinstance(predictions, list) and len(predictions) > 0 and isinstance(predictions[0], list):
                    predictions = predictions[0]
                if isinstance(probabilities, list) and len(probabilities) > 0 and isinstance(probabilities[0], list):
                    probabilities = probabilities[0]
                
                if isinstance(predictions, list) and isinstance(probabilities, list) and len(predictions) > 0:
                    # Process each residue
                    functional_residues = []
                    non_functional_residues = []
                    residue_details = []
                    
                    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                        if i >= len(current_sequence):
                            break
                            
                        residue = current_sequence[i]
                        position = i + 1  # 1-based indexing
                        
                        # Handle different probability formats
                        if isinstance(prob, list):
                            max_prob = max(prob)
                            predicted_label = prob.index(max_prob)
                        else:
                            max_prob = prob
                            predicted_label = pred
                        
                        residue_details.append({
                            'position': position,
                            'residue': residue,
                            'predicted_label': predicted_label,
                            'probability': max_prob
                        })
                        
                        # Classify as functional or non-functional
                        if predicted_label == 1:  # Functional residue
                            functional_residues.append((position, residue, max_prob))
                        else:  # Non-functional residue
                            non_functional_residues.append((position, residue, max_prob))
                    
                    # Generate report for this dataset
                    total_residues = len(residue_details)
                    functional_count = len(functional_residues)
                    
                    result += f"- Functional residues: {functional_count} ({functional_count/total_residues:.1%})\n"
                    result += f"- Non-functional residues: {len(non_functional_residues)} ({len(non_functional_residues)/total_residues:.1%})\n\n"
                    
                    if functional_residues:
                        result += "\n"

                        formatted_residues = [f"{res}{pos}" for pos, res, prob in functional_residues]
                        chunk_size = 20  # Show 20 residues per line
                        for i in range(0, len(formatted_residues), chunk_size):
                            chunk = formatted_residues[i:i + chunk_size]
                            result += ", ".join(chunk) + "\n"
                
                else:
                    result += "Prediction data format not recognized or empty.\n\n"
                    
            except Exception as e:
                result += f"Error parsing prediction results: {str(e)}\n"
                result += f"Raw predictions: {str(predictions_str)[:100]}...\n"
                result += f"Raw probabilities: {str(probabilities_str)[:100]}...\n\n"
        else:
            result += "No prediction data available.\n\n"
    
    return result

def format_property_results(results: Dict[str, Any]) -> str:
    """Format property calculation results for display using proper Markdown line breaks."""
    
    properties = results.get('properties', {})
    file_type = results.get('file_type')
    chain_id = results.get('chain_id')

    # --- Build each section as a separate f-string for clarity ---

    phys_section = ""
    if 'physicochemical' in properties:
        phys = properties['physicochemical']
        ssf = phys['secondary_structure_fraction']
        instability_text = " (Predicted Unstable)" if phys['instability_index'] > 40 else " (Predicted Stable)"
        phys_section = f"""
        ## Basic Physicochemical Properties

        **Sequence Information:**
        - Molecular Weight: {phys['molecular_weight']/1000:.3f} kDa

        **Chemical Properties:**
        - Theoretical pI: {phys['theoretical_pI']:.3f}
        - Aromaticity: {phys['aromaticity']:.3f}
        - Instability Index: {phys['instability_index']:.3f}{instability_text}
        - Grand Average of Hydropathicity {phys['gravy']:.3f}

        **Secondary Structure Prediction:**
        - Alpha Helix: {ssf['helix']:.1%}
        - Beta Turn: {ssf['turn']:.1%}
        - Beta Sheet: {ssf['sheet']:.1%}
        """

    sasa_section = ""
    if file_type == 'pdb' and 'sasa' in properties:
        sasa = properties['sasa']
        sasa_section = f"""
        ## Solvent Accessible Surface Area (SASA)

        **Chain {sasa['chain_id']} Analysis:**
        - Total SASA: {sasa['total_sasa']:.2f} Å²
        - Residues Analyzed: {sasa['residue_count']}
        - Average SASA per residue: {sasa['total_sasa']/sasa['residue_count']:.2f} Å²
        """

    rsa_section = ""
    if file_type == 'pdb' and 'rsa' in properties:
        rsa = properties['rsa']
        total_residues = rsa['exposed_residues'] + rsa['buried_residues']
        rsa_section = f"""
        ## Relative Solvent Accessibility (RSA)

        **Chain {rsa['chain_id']} Analysis:**
        - Exposed Residues: {rsa['exposed_residues']} ({rsa['exposed_residues']/total_residues:.1%})
        - Buried Residues: {rsa['buried_residues']} ({rsa['buried_residues']/total_residues:.1%})
        - Total Residues: {total_residues}
        """

    ss_section = ""
    if file_type == 'pdb' and 'secondary_structure' in properties:
        ss = properties['secondary_structure']
        counts = ss['ss_counts']
        total_len = len(ss['aa_sequence'])
        if total_len > 0:
            ss_section = f"""
                ## Secondary Structure Analysis

                **Chain {ss['chain_id']} Structure:**
                - Alpha Helix (H): {counts['helix']} residues ({counts['helix']/total_len:.1%})
                - Beta Sheet (E): {counts['sheet']} residues ({counts['sheet']/total_len:.1%})
                - Random Coil (C): {counts['coil']} residues ({counts['coil']/total_len:.1%})
                - Total Length: {total_len} residues
                """

    summary_section = f"""
        ## Analysis Summary

        - Input File Type: {file_type.upper()}
        - Chain Analyzed: {chain_id or 'N/A'}
        - Properties Calculated: {len(properties)}
        - Analysis Timestamp: {results['analysis_timestamp']}

        ---
        *Property calculations completed using VenusFactory's integrated analysis pipeline.*
        """

    # --- Combine all sections into the final report ---
    report = f"""
        # Physical & Chemical Properties Analysis
        {phys_section}
        {sasa_section}
        {rsa_section}
        {ss_section}
        {summary_section}
        """
    return report.strip()

def handle_physical_chemical_properties(content, selected_chain, current_file, sequence_state, original_content):
    """Handle physical and chemical properties analysis (placeholder)."""
    if not content.strip():
        return "Error: Please provide protein sequence or upload a file."
    
    # Prepare FASTA file for function prediction
    timestamp = str(int(time.time()))
    input_data_dir = get_save_path("Report", "Upload_data")
    output_data_dir = get_save_path("Report", "Results")

    is_pdb = content.strip().startswith('ATOM') or (current_file and current_file.endswith('.pdb'))
    
    try:
        if is_pdb:
            # Use PDB file for comprehensive analysis
            if current_file and current_file.endswith('.pdb'):
                input_file = current_file
            else:
                # Save pasted PDB content to file
                input_file = input_data_dir / f"temp_pdb_{timestamp}.pdb"
                with open(input_file, 'w') as f:
                    f.write(content)
                input_file = str(input_file)
            
            file_type = 'pdb'
        else:
            # Use FASTA file for sequence-based analysis
            if current_file and current_file.endswith(('.fasta', '.fa')):
                input_file = current_file
            else:
                # Save sequence as FASTA
                input_file = input_data_dir / f"temp_fasta_{timestamp}.fasta"
                if original_content and original_content.strip().startswith('>'):
                    fasta_content = original_content
                else:
                    sequence = content.strip()
                    fasta_content = f">{selected_chain}\n{sequence}"
                
                with open(input_file, 'w') as f:
                    f.write(fasta_content)
                input_file = str(input_file)
            
            file_type = 'fasta'
            
        # Run calculate_all_property.py as subprocess
        output_file = output_data_dir / f"property_results_{timestamp}.json"
        script_path = Path("src") / "tools" / "predict" / "features" / "calculate_all_property.py"
        cmd = [
            sys.executable, str(script_path),
            "--input_file", str(input_file),
            "--file_type", file_type,
            "--chain_id", selected_chain,
            "--output_file", str(output_file)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, encoding='utf-8', errors='ignore')
        
        if output_file.exists():
            # Load results from JSON file
            with open(output_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            # Clean up temporary files
            os.remove(output_file)
            if not current_file or not os.path.exists(current_file):
                if os.path.exists(input_file):
                    os.remove(input_file)
            
            # Format results for display
            return format_property_results(results)
        else:
            error_msg = f"Property calculation failed. stdout: {result.stdout}, stderr: {result.stderr}"
            return f"Error in property calculation: {error_msg}"
        
    except Exception as e:
        return f"Error in property calculation: {str(e)}"

# LLMConfig, get_api_key and call_llm_api from utils.llm_helpers
    

def generate_expert_analysis_report(report: str) -> str:
    """Generate a prompt for an LLM to analyze report"""
    prompt = f"""
    You are VenusFactory, 蛋白质工程的AI助手 (AI assistant for protein engineering). You are an expert proficient in computer science and biology. Your current task is to generate a more comprehensive report based on the simplified Report I provided, while maintaining the following points:
    1. Add a level-one title or emoji, which can be presented in various forms such as tables and text. The output format should be Markdown.
    2. The content of the output report needs to fully serve biologists, ensuring that they can understand the content most accurately.
    Report to Process:
    {report}    
    """
    api_key = get_api_key("DeepSeek")
    llm_config = LLMConfig(api_key, "DeepSeek", get_chat_base_url(), LLM_MODELS.get("DeepSeek", "deepseek-chat"))
    ai_response = call_llm_api(llm_config, prompt)
    return ai_response

def _md_to_pdf(md_content: str, output_path: Path) -> bool:
    """Markdown -> HTML -> PDF (via xhtml2pdf)."""
    try:
        import markdown
        from xhtml2pdf import pisa

        md = markdown.Markdown(extensions=['tables', 'fenced_code', 'toc'])
        html_content = md.convert(md_content)
        full_html = load_html_template(
            "ai_analysis_report.html",
            generated_time=time.strftime('%Y-%m-%d %H:%M:%S'),
            html_content=html_content,
        )
        with open(output_path, 'wb') as f:
            status = pisa.CreatePDF(full_html, dest=f, encoding='utf-8')
        return not status.err
    except Exception as e:
        print(f"MD to PDF failed: {e}")
        return False


def export_ai_report_to_html(ai_report_content: str):
    """Export AI analysis report to HTML file for download."""
    if not ai_report_content or ai_report_content.startswith("Error:") or ai_report_content.startswith("*"):
        return None
    try:
        import markdown

        output_dir = get_save_path("Report", "Results")
        html_path = output_dir / f"venusfactory_ai_report_{int(time.time())}.html"
        md = markdown.Markdown(extensions=['tables', 'fenced_code', 'toc'])
        html_content = md.convert(ai_report_content)
        full_html = load_html_template(
            "ai_analysis_report.html",
            generated_time=time.strftime('%Y-%m-%d %H:%M:%S'),
            html_content=html_content,
        )
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(full_html)
        return str(html_path.absolute())
    except Exception as e:
        print(f"HTML export failed: {e}")
        return None


def export_ai_report_to_pdf(ai_report_content: str):
    """Export AI analysis report to PDF file for download."""
    if not ai_report_content or ai_report_content.startswith("Error:") or ai_report_content.startswith("*"):
        return None

    output_dir = get_save_path("Report", "Results")
    pdf_path = output_dir / f"venusfactory_ai_report_{int(time.time())}.pdf"
    if _md_to_pdf(ai_report_content, pdf_path):
        return str(pdf_path.absolute())
    return None

def handle_content_change_new(content):
    """Handle content change in the main input area."""
    if content.strip():
        return parse_content_and_update_selector(content) + (content,)
    else:
        return "", gr.update(choices=["Sequence 1"], value="Sequence 1", visible=True), {}, "Sequence 1", "", ""

def handle_chain_selection_new(selected_chain, sequences_dict, current_file, original_content):
    """Handle chain/sequence selection change."""
    if not sequences_dict or selected_chain not in sequences_dict:
        return ""
    
    if current_file:
        if current_file.endswith('.fasta'):
            if original_content:
                sequence, file_path = handle_paste_sequence_selection(selected_chain, sequences_dict, original_content)
            else:
                sequence, file_path = handle_fasta_sequence_change(selected_chain, sequences_dict, current_file)
        elif current_file.endswith('.pdb'):
            if original_content:
                sequence, file_path = handle_paste_chain_selection(selected_chain, sequences_dict, original_content)
            else:
                sequence, file_path = handle_pdb_chain_change(selected_chain, sequences_dict, current_file)
        else:
            sequence = sequences_dict[selected_chain]
            file_path = current_file
        return sequence
    else:
        return sequences_dict[selected_chain]

def generate_brief_report(sequence_dict, selected_chain, current_file, original_content, selected_analyses, progress=gr.Progress()):
    """Generate a brief scientific report based on selected analyses."""
    
    if not sequence_dict or selected_chain not in sequence_dict:
        err = "Error: No valid sequence found. Please provide a protein sequence first."
        return err, err, gr.update(visible=False), gr.update(visible=False)
    
    if not selected_analyses:
        err = "Please select at least one analysis type before generating the report."
        return err, err, gr.update(visible=False), gr.update(visible=False)
    
    sequence = sequence_dict[selected_chain]
    
    # Determine content for analysis
    if original_content and original_content.strip():
        content_for_analysis = original_content
    elif sequence and sequence.strip():
        content_for_analysis = f">{selected_chain}\n{sequence}"
    else:
        err = "Error: No valid sequence content available for analysis."
        return err, err, gr.update(visible=False), gr.update(visible=False)
    
    # Generate scientific report header
    report_sections = []
    report_sections.append("# PROTEIN ANALYSIS REPORT")
    report_sections.append("")
    report_sections.append(f"**Sequence ID:** {selected_chain} ")
    report_sections.append(f"**Sequence Length:** {len(sequence)} amino acid residues ")
    report_sections.append(f"**Analysis Date:** {time.strftime('%Y-%m-%d %H:%M:%S')} ")
    report_sections.append("")
    report_sections.append("---")
    report_sections.append("")
    progress(0.1, desc="Analysising your input sequences...")
    # Execute selected analyses with improved formatting
    if "mutation" in selected_analyses:
        report_sections.append("## 🧬 Mutation Prediction Analysis")
        report_sections.append("")
        try:
            mutation_result = handle_individual_mutation_prediction(
                content_for_analysis, selected_chain, current_file, sequence_dict, original_content
            )
            if not mutation_result.startswith("Error"):
                report_sections.append(f"Deep mutational scanning identified Top 20 beneficial mutations:")
                report_sections.append("")
                report_sections.append(mutation_result)
            else:
                report_sections.append("No beneficial mutations identified in the analysis.")
                report_sections.append("")

        except Exception as e:
            report_sections.append(f"❌ Mutation analysis failed: {str(e)}")
            report_sections.append("")
    progress(0.3, desc="Mutation prediction analysis...")
    if "function" in selected_analyses:
        report_sections.append("## 🔬 Protein Function Analysis")
        report_sections.append("")
        try:
            function_result = handle_individual_function_prediction(
                content_for_analysis, selected_chain, current_file, sequence_dict, original_content
            )
            if not function_result.startswith("Error"):
                report_sections.append("")
                report_sections.append(function_result)
            else:
                report_sections.append("❌ Function analysis could not be completed.")
                report_sections.append("")
            
        except Exception as e:
            report_sections.append(f"❌ Function analysis failed: {str(e)}")
            report_sections.append("")
    progress(0.5, desc="Protein function analysis...")
    if "residue" in selected_analyses:
        report_sections.append("## 🎯 Functional Residue ")
        report_sections.append("")
        try:
            residue_result = handle_functional_residue_prediction(
                content_for_analysis, selected_chain, current_file, sequence_dict, original_content
            )
            if not residue_result.startswith("Error"):
                # Parse residue prediction results more cleanly
                lines = residue_result.split('\n')
                current_dataset = None
                
                for line in lines:
                    line = line.strip()
                    report_sections.append(line)
            else:
                report_sections.append("❌ Functional residue analysis could not be completed.")
                report_sections.append("")
        except Exception as e:
            report_sections.append(f"❌ Functional residue analysis failed: {str(e)}")
            report_sections.append("")
    progress(0.7, desc="Functional residue analysis...")
    if "properties" in selected_analyses:
        report_sections.append("## ⚗️ Physical & Chemical Properties")
        report_sections.append("")
        try:
            properties_result = handle_physical_chemical_properties(
                content_for_analysis, selected_chain, current_file, sequence_dict, original_content
            )
            if not properties_result.startswith("Error"):
                # Extract and format key properties
                lines = properties_result.split('\n')
                
                for line in lines:
                    line = line.strip()
                    if any(prop in line for prop in ["Molecular Weight:", "Theoretical pI ", "Instability Index:", "Aromaticity", "Total SASA:", "Exposed Residues:"]):
                        if line.startswith("- "):
                            report_sections.append(f"- {line[2:]}")
                        elif ":" in line:
                            prop_name, prop_value = line.split(":", 1)
                            report_sections.append(f"- {prop_name.strip()}: {prop_value.strip()}")
                
                report_sections.append("")
            else:
                report_sections.append("❌ Physical/chemical property analysis could not be completed.")
                report_sections.append("")
        except Exception as e:
            report_sections.append(f"❌ Property analysis failed: {str(e)}")
            report_sections.append("")
    progress(0.8, desc="Physical & chemical properties analysis...")
    # Add conclusion with better formatting
    report_sections.append("---")
    report_sections.append("")
    report_sections.append("## 💡 Conclusion")
    report_sections.append("")
    report_sections.append("✅ Analysis completed successfully. Results should be validated experimentally.")
    report_sections.append("")
    report_sections.append(f"*Report generated by VenusFactory v1.0 on {time.strftime('%Y-%m-%d at %H:%M:%S')}*")
    progress(0.9, desc="Generating AI-based report")
    # Add AI-based analysis
    ai_report = generate_expert_analysis_report(report_sections)
    html_path = export_ai_report_to_html(ai_report)
    pdf_path = export_ai_report_to_pdf(ai_report)
    html_btn = gr.update(visible=True, value=html_path) if html_path and os.path.exists(html_path) else gr.update(visible=False, value=None)
    pdf_btn = gr.update(visible=True, value=pdf_path) if pdf_path and os.path.exists(pdf_path) else gr.update(visible=False, value=None)

    return "\n".join(report_sections), ai_report, html_btn, pdf_btn

from web.utils.file_handlers import handle_file_upload, clear_paste_content_fasta


def create_comprehensive_tab(constant: Dict[str, Any]) -> Dict[str, Any]:
    """Create the comprehensive analysis tab. Layout mirrors quick_tool_tab: left config, right results."""
    with gr.Column():
        gr.Markdown("💡 *One-click AI protein analysis: mutation prediction, physicochemical properties, structure features, and functional residues.*")
        with gr.Row(equal_height=False):
            with gr.Column(scale=2):
                gr.Markdown("## Data Input")
                with gr.Tabs():
                    with gr.Tab("📁 Upload File"):
                        comprehensive_file_upload = gr.File(label="Protein file (.fasta, .fa, .pdb)", file_types=[".fasta", ".fa", ".pdb"])
                        comprehensive_file_example = gr.Examples(examples=[["./example/database/P60002.fasta"]], inputs=comprehensive_file_upload, label="Click example to load")
                    with gr.Tab("📋 Paste"):
                        comprehensive_paste_content_input = gr.Textbox(label="Paste protein content", placeholder="Paste FASTA/PDB or raw sequence...", lines=6, max_lines=12)
                        with gr.Row():
                            comprehensive_paste_content_btn = gr.Button("🔍 Detect", variant="primary")
                            comprehensive_paste_clear_btn = gr.Button("Clear")
                with gr.Column(visible=False) as comprehensive_sequence_section:
                    gr.Markdown("## Sequence")
                    comprehensive_protein_display = gr.Textbox(label="Uploaded Protein Sequence", interactive=False, lines=2, max_lines=5)
                    comprehensive_sequence_selector = gr.Dropdown(label="Select Chain", choices=["Sequence 1", "A"], value="Sequence 1", visible=False)
                comprehensive_original_file_path_state = gr.State("")
                comprehensive_original_paste_content_state = gr.State("")
                comprehensive_selected_sequence_state = gr.State("Sequence 1")
                comprehensive_sequence_state = gr.State({})
                comprehensive_current_file_state = gr.State("")
                gr.Markdown("## Select Analysis Types")
                analysis_selection = gr.CheckboxGroup(
                    choices=[
                        ("🧬 Mutation", "mutation"),
                        ("🔬 Function", "function"),
                        ("🎯 Residue", "residue"),
                        ("⚗️ Properties", "properties")
                    ],
                    value=[],
                    label="",
                    interactive=True
                )
                generate_btn = gr.Button("🚀 Generate Report", variant="primary")
                export_html_btn = gr.DownloadButton("📥 Export HTML", visible=False)
                export_pdf_btn = gr.DownloadButton("📥 Export PDF", visible=False)
            with gr.Column(scale=3, elem_classes=["comprehensive-output-box"]):
                output_text = gr.Markdown(value="*Your analysis results will appear here...*", label="", visible=False)
                output_text_with_ai = gr.Markdown(value="*Your AI analysis results will appear here...*", label="")
    # Connect event handlers
    comprehensive_file_upload.upload(
        fn=lambda x: wrap_upload_show_section(handle_file_upload(x)),
        inputs=comprehensive_file_upload,
        outputs=[comprehensive_protein_display, comprehensive_sequence_selector, comprehensive_sequence_state, comprehensive_selected_sequence_state, comprehensive_original_file_path_state, comprehensive_current_file_state, comprehensive_sequence_section]
    )
    comprehensive_file_upload.change(
        fn=lambda x: wrap_upload_show_section(handle_file_upload(x)),
        inputs=comprehensive_file_upload,
        outputs=[comprehensive_protein_display, comprehensive_sequence_selector, comprehensive_sequence_state, comprehensive_selected_sequence_state, comprehensive_original_file_path_state, comprehensive_current_file_state, comprehensive_sequence_section]
    )
    comprehensive_paste_clear_btn.click(
        fn=lambda: wrap_clear_hide_section(clear_paste_content_fasta()),
        outputs=[comprehensive_paste_content_input, comprehensive_protein_display, comprehensive_sequence_selector, comprehensive_sequence_state, comprehensive_selected_sequence_state, comprehensive_original_file_path_state, comprehensive_sequence_section]
    )
    comprehensive_paste_content_btn.click(
        fn=lambda c: wrap_paste_show_section(handle_paste_fasta_detect(c), truncate=6),
        inputs=comprehensive_paste_content_input,
        outputs=[comprehensive_protein_display, comprehensive_sequence_selector, comprehensive_sequence_state, comprehensive_selected_sequence_state, comprehensive_original_file_path_state, comprehensive_original_paste_content_state, comprehensive_sequence_section]
    )
    comprehensive_sequence_selector.change(
        fn=handle_sequence_change_unified,
        inputs=[comprehensive_sequence_selector, comprehensive_sequence_state, comprehensive_original_file_path_state, comprehensive_original_paste_content_state],
        outputs=[comprehensive_protein_display, comprehensive_current_file_state]
    )
    generate_btn.click(
        fn=generate_brief_report,
        inputs=[comprehensive_sequence_state, comprehensive_selected_sequence_state, comprehensive_current_file_state, comprehensive_original_paste_content_state, analysis_selection],
        outputs=[output_text, output_text_with_ai, export_html_btn, export_pdf_btn]
    )
    return {}