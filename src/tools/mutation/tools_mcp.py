"""
Evolution MCP calls: zero-shot sequence/structure mutation, protein function, functional residue prediction.
Gradio Client + optional OSS upload; OSS infra from tools.search.tools_mcp.
"""
import json
import os
import uuid
from typing import Optional

import pandas as pd
from gradio_client import Client, handle_file

from web.utils.common_utils import get_save_path
from web.utils.file_handlers import extract_first_chain_from_pdb_file
from tools.search.tools_mcp import (
    upload_file_to_oss_sync,
    get_gradio_base_url,
    DEFAULT_BACKEND,
)


def call_zero_shot_mutation_sequence_prediction(
    sequence: str = None,
    fasta_file: str = None,
    model_name: str = "ESM2-650M",
    api_key: Optional[str] = None,
    backend: str = None,
) -> str:
    """Call VenusFactory zero-shot sequence-based mutation prediction API."""
    try:
        if fasta_file:
            fasta_path = fasta_file
            temp_fasta_created = False
        elif sequence:
            temp_dir = get_save_path("MCP_Server", "TempFasta")
            temp_fasta_path = temp_dir / f"temp_sequence_{uuid.uuid4().hex[:8]}.fasta"
            with open(temp_fasta_path, 'w') as f:
                f.write(f">temp_sequence\n{sequence}\n")
            fasta_path = str(temp_fasta_path)
            temp_fasta_created = True
        else:
            return "Zero-shot sequence prediction error: No sequence or fasta_file provided."

        backend = backend or DEFAULT_BACKEND
        client = Client(get_gradio_base_url(backend))
        result = client.predict(
            function_selection="Activity",
            file_obj=handle_file(fasta_path),
            enable_ai=False,
            llm_model="DeepSeek",
            user_api_key=api_key,
            model_name=model_name,
            api_name="/handle_mutation_prediction_base"
        )

        if 'temp_fasta_created' in locals() and temp_fasta_created:
            os.unlink(fasta_path)

        update_dict = result[3]
        tar_file_path = update_dict['value']
        base_dir = get_save_path("Zero_shot", "HeatMap")
        tar_filename = os.path.basename(tar_file_path)
        tar_file_path = os.path.join(base_dir, tar_filename)

        raw_result = result[2]
        try:
            if isinstance(raw_result, dict) and 'data' in raw_result:
                mutations_data = raw_result['data']
                total_mutations = len(mutations_data)
                if total_mutations > 100:
                    top_50 = mutations_data[:50]
                    separator_row = ['...', '...', '...']
                    combined_data = top_50 + [separator_row]
                    raw_result['data'] = combined_data
                    raw_result['total_mutations'] = total_mutations
                    raw_result['displayed_mutations'] = 50
                    raw_result['note'] = (
                        f"Showing top 50 most beneficial mutations "
                        f"out of {total_mutations} total to avoid long context. "
                        f"Results are separated by '...'."
                    )
            if tar_file_path:
                try:
                    csv_filename = tar_filename.replace('pred_mut_', 'mut_res_').replace('.tar.gz', '.csv')
                    heatmap_filename = tar_filename.replace('pred_mut_', 'mut_map_').replace('.tar.gz', '.html')
                    csv_path = os.path.join(base_dir, csv_filename)
                    heatmap_path = os.path.join(base_dir, heatmap_filename)
                    if os.path.exists(csv_path):
                        raw_result['csv_path'] = csv_path
                        url = upload_file_to_oss_sync(str(csv_path), backend=backend)
                        if url is not None:
                            raw_result['csv_oss_url'] = url
                    if os.path.exists(heatmap_path):
                        raw_result['heatmap_path'] = heatmap_path
                        url = upload_file_to_oss_sync(str(heatmap_path), backend=backend)
                        if url is not None:
                            raw_result['heatmap_oss_url'] = url
                except Exception as e:
                    print(f"Warning: Could not extract file paths: {e}")
            return json.dumps(raw_result, indent=2)
        except (json.JSONDecodeError, KeyError, TypeError):
            return raw_result
    except Exception as e:
        return f"Zero-shot sequence prediction error: {str(e)}"


def call_zero_shot_mutation_structure_prediction_from_file(
    structure_file: str,
    model_name: str = "ESM-IF1",
    api_key: Optional[str] = None,
    user_output_dir: Optional[str] = None,
    backend: str = None,
) -> str:
    """Call VenusFactory zero-shot structure-based mutation prediction API."""
    backend = backend or DEFAULT_BACKEND
    try:
        processed_file = extract_first_chain_from_pdb_file(structure_file)
        client = Client(get_gradio_base_url(backend))
        result = client.predict(
            function_selection="Activity",
            file_obj=handle_file(processed_file),
            enable_ai=False,
            llm_model="DeepSeek",
            user_api_key=api_key,
            model_name=model_name,
            api_name="/handle_mutation_prediction_base"
        )

        update_dict = result[3]
        tar_file_path = update_dict['value']
        base_dir = get_save_path("Zero_shot", "HeatMap")
        tar_filename = os.path.basename(tar_file_path)
        tar_file_path = os.path.join(base_dir, tar_filename)

        raw_result = result[2]
        try:
            if isinstance(raw_result, dict) and 'data' in raw_result:
                mutations_data = raw_result['data']
                total_mutations = len(mutations_data)
                if total_mutations > 100:
                    top_50 = mutations_data[:50]
                    separator_row = ['...', '...', '...']
                    combined_data = top_50 + [separator_row]
                    raw_result['data'] = combined_data
                    raw_result['total_mutations'] = total_mutations
                    raw_result['displayed_mutations'] = 50
                    raw_result['note'] = (
                        f"Showing top 50 most beneficial mutations "
                        f"out of {total_mutations} total to avoid long context. "
                        f"Results are separated by '...'."
                    )
            if tar_file_path:
                try:
                    csv_filename = tar_filename.replace('pred_mut_', 'mut_res_').replace('.tar.gz', '.csv')
                    heatmap_filename = tar_filename.replace('pred_mut_', 'mut_map_').replace('.tar.gz', '.html')
                    csv_path = os.path.join(base_dir, csv_filename)
                    heatmap_path = os.path.join(base_dir, heatmap_filename)
                    if os.path.exists(csv_path):
                        raw_result['csv_path'] = csv_path
                        url = upload_file_to_oss_sync(str(csv_path), backend=backend)
                        if url is not None:
                            raw_result['csv_oss_url'] = url
                    if os.path.exists(heatmap_path):
                        raw_result['heatmap_path'] = heatmap_path
                        url = upload_file_to_oss_sync(str(heatmap_path), backend=backend)
                        if url is not None:
                            raw_result['heatmap_oss_url'] = url
                except Exception as e:
                    print(f"Warning: Could not extract file paths: {e}")
            return json.dumps(raw_result, indent=2)
        except (json.JSONDecodeError, KeyError, TypeError):
            return raw_result
    except Exception as e:
        return f"Zero-shot structure prediction error: {str(e)}"


