"""
Filter MCP calls: protein physchem/structure property prediction; protein function prediction; functional residue prediction.
Gradio Client; backend selects local vs pjlab service URL.
"""
import json
import os
import uuid
from typing import Optional

import pandas as pd
from gradio_client import Client, handle_file

from web.utils.common_utils import get_save_path
from tools.search.tools_mcp import (
    get_gradio_base_url,
    DEFAULT_BACKEND,
    BACKEND_LOCAL,
    upload_file_to_oss_sync,
)


def call_protein_properties_prediction(
    sequence: str = None,
    fasta_file: str = None,
    task_name: str = "Physical and chemical properties",
    api_key: Optional[str] = None,
    backend: str = BACKEND_LOCAL,
) -> str:
    """Predict protein properties from a sequence or a fasta file."""
    try:
        if fasta_file:
            file_path = fasta_file
            temp_fasta_created = False
        elif sequence:
            temp_dir = get_save_path("MCP_Server", "TempFasta")
            temp_fasta_path = temp_dir / f"temp_sequence_{uuid.uuid4().hex[:8]}.fasta"
            with open(temp_fasta_path, 'w') as f:
                f.write(f">temp_sequence\n{sequence}\n")
            file_path = str(temp_fasta_path)
            temp_fasta_created = True
        else:
            return "Protein properties prediction error: No sequence or fasta_file provided."

        backend = backend or DEFAULT_BACKEND
        client = Client(get_gradio_base_url(backend))
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


def call_protein_function_prediction(
    sequence: str = None,
    fasta_file: str = None,
    model_name: str = "ESM2-650M",
    task: str = "Solubility",
    api_key: Optional[str] = None,
    backend: str = None,
) -> str:
    """Call VenusFactory protein function prediction API (solubility, localization, metal binding, etc.)."""
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
            temp_dir = get_save_path("MCP_Server", "TempFasta")
            temp_fasta_path = temp_dir / f"temp_sequence_{uuid.uuid4().hex[:8]}.fasta"
            with open(temp_fasta_path, "w") as f:
                f.write(f">temp_sequence\n{sequence}\n")
            fasta_path = str(temp_fasta_path)
        else:
            return "Error: Either sequence or fasta_file must be provided"

        backend = backend or DEFAULT_BACKEND
        client = Client(get_gradio_base_url(backend))
        result = client.predict(
            task=task,
            fasta_file=handle_file(fasta_path),
            model_name=model_name,
            datasets=datasets,
            enable_ai=True,
            llm_model="DeepSeek",
            user_api_key=api_key,
            api_name="/handle_protein_function_prediction_chat",
        )

        if temp_fasta_path:
            os.unlink(temp_fasta_path)

        df = result[1]
        csv_path = result[2]
        csv_oss_url = upload_file_to_oss_sync(str(csv_path), backend=backend) if csv_path else None
        df["csv_path"] = csv_path
        df["csv_oss_url"] = csv_oss_url
        if isinstance(df, pd.DataFrame):
            return df.to_json(orient="records", force_ascii=False, indent=2)
        return str(df)
    except Exception as e:
        return f"Function prediction error: {str(e)}"


def call_functional_residue_prediction(
    sequence: str = None,
    fasta_file: str = None,
    model_name: str = "ESM2-650M",
    task: str = "Activity Site",
    api_key: Optional[str] = None,
    backend: str = None,
) -> str:
    """Call VenusFactory functional residue prediction API (activity/binding/conserved site, motif)."""
    try:
        temp_fasta_path = None
        if fasta_file:
            fasta_path = fasta_file
        elif sequence:
            temp_dir = get_save_path("MCP_Server", "TempFasta")
            temp_fasta_path = temp_dir / f"temp_sequence_{uuid.uuid4().hex[:8]}.fasta"
            with open(temp_fasta_path, "w") as f:
                f.write(f">temp_sequence\n{sequence}\n")
            fasta_path = str(temp_fasta_path)
        else:
            return "Error: Either sequence or fasta_file must be provided"

        backend = backend or DEFAULT_BACKEND
        client = Client(get_gradio_base_url(backend))
        result = client.predict(
            task=task,
            fasta_file=handle_file(fasta_path),
            enable_ai=True,
            llm_model="DeepSeek",
            user_api_key=api_key,
            model_name=model_name,
            api_name="/handle_protein_residue_function_prediction_chat",
        )

        if temp_fasta_path:
            os.unlink(temp_fasta_path)

        raw_result = result[1]
        try:
            if isinstance(raw_result, dict):
                result_data = raw_result
            else:
                result_data = json.loads(raw_result)
            if isinstance(result_data, dict) and "data" in result_data:
                all_residues = result_data["data"]
                functional_residues = [r for r in all_residues if r[2] == 1]
                if functional_residues:
                    result_data["data"] = functional_residues
                    result_data["total_residues"] = len(all_residues)
                    result_data["functional_residues"] = len(functional_residues)
                    result_data["note"] = f"Showing {len(functional_residues)} functional residues (label=1) out of {len(all_residues)} total residues"
                    return json.dumps(result_data)
                else:
                    result_data["data"] = []
                    result_data["total_residues"] = len(all_residues)
                    result_data["functional_residues"] = 0
                    result_data["note"] = f"No functional residues (label=1) found out of {len(all_residues)} total residues"
                    return json.dumps(result_data)
            return raw_result
        except (json.JSONDecodeError, KeyError, TypeError):
            return raw_result
    except Exception as e:
        return f"Functional residue prediction error: {str(e)}"
