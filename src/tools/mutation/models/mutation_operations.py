# Core layer: zero-shot mutation prediction (Gradio client + result shaping). Returns status dict.

import json
import os
import uuid
from typing import Any, Dict, Optional

from gradio_client import Client, handle_file

from web.utils.common_utils import get_save_path
from web.utils.file_handlers import extract_first_chain_from_pdb_file

try:
    from tools.search.tools_mcp import upload_file_to_oss_sync, get_gradio_base_url, DEFAULT_BACKEND
except ImportError:
    DEFAULT_BACKEND = "local"

    def get_gradio_base_url(backend: str) -> str:
        return os.getenv("GRADIO_BASE_URL", "http://localhost:7860").rstrip("/")

    def upload_file_to_oss_sync(file_path: str, backend: str = None) -> Optional[str]:
        return None


def _check_gradio_reachable(url: str) -> bool:
    import urllib.request
    try:
        urllib.request.urlopen(url, timeout=5)
        return True
    except Exception:
        return False


def _error_dict(error_type: str, message: str, suggestion: Optional[str] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"status": "error", "error": {"type": error_type, "message": message}, "file_info": None}
    if suggestion:
        out["error"]["suggestion"] = suggestion
    return out


def _enrich_mutation_result(raw_result: Any, tar_filename: str, backend: str) -> Dict[str, Any]:
    """Add truncated data if needed, csv_path/heatmap_path and optional oss_url."""
    if not isinstance(raw_result, dict):
        return raw_result
    if "data" in raw_result:
        mutations_data = raw_result["data"]
        total = len(mutations_data)
        if total > 100:
            raw_result["data"] = mutations_data[:50] + [["...", "...", "..."]]
            raw_result["total_mutations"] = total
            raw_result["displayed_mutations"] = 50
            raw_result["note"] = f"Showing top 50 of {total} mutations. Results separated by '...'."
    base_dir = get_save_path("Zero_shot", "HeatMap")
    csv_filename = tar_filename.replace("pred_mut_", "mut_res_").replace(".tar.gz", ".csv")
    heatmap_filename = tar_filename.replace("pred_mut_", "mut_map_").replace(".tar.gz", ".html")
    csv_path = os.path.join(base_dir, csv_filename)
    heatmap_path = os.path.join(base_dir, heatmap_filename)
    if os.path.exists(csv_path):
        raw_result["csv_path"] = csv_path
        url = upload_file_to_oss_sync(csv_path, backend=backend)
        if url:
            raw_result["csv_oss_url"] = url
    if os.path.exists(heatmap_path):
        raw_result["heatmap_path"] = heatmap_path
        url = upload_file_to_oss_sync(heatmap_path, backend=backend)
        if url:
            raw_result["heatmap_oss_url"] = url
    return raw_result


def zero_shot_mutation_sequence_prediction(
    sequence: Optional[str] = None,
    fasta_file: Optional[str] = None,
    model_name: str = "ESM2-650M",
    api_key: Optional[str] = None,
    backend: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run zero-shot sequence-based mutation prediction via Gradio. Returns status dict.
    Provide either sequence or fasta_file.
    """
    backend = backend or DEFAULT_BACKEND
    fasta_path = None
    temp_fasta_created = False
    if fasta_file and os.path.exists(fasta_file):
        fasta_path = fasta_file
    elif sequence:
        temp_dir = get_save_path("MCP_Server", "TempFasta")
        temp_dir.mkdir(parents=True, exist_ok=True)
        path = temp_dir / f"temp_sequence_{uuid.uuid4().hex[:8]}.fasta"
        path.write_text(f">temp_sequence\n{sequence}\n")
        fasta_path = str(path)
        temp_fasta_created = True
    else:
        return _error_dict("ValidationError", "Either sequence or fasta_file must be provided.")

    gradio_url = get_gradio_base_url(backend)
    if not _check_gradio_reachable(gradio_url):
        if temp_fasta_created and fasta_path and os.path.exists(fasta_path):
            try:
                os.unlink(fasta_path)
            except Exception:
                pass
        return _error_dict(
            "ConnectionError",
            f"Gradio backend is not reachable at {gradio_url}",
            suggestion=f"Start the Gradio server or set GRADIO_BASE_URL env var. Current URL: {gradio_url}",
        )

    try:
        client = Client(gradio_url)
        result = client.predict(
            function_selection="Activity",
            file_obj=handle_file(fasta_path),
            enable_ai=False,
            llm_model="DeepSeek",
            user_api_key=api_key,
            model_name=model_name,
            api_name="/handle_mutation_prediction_base",
        )
        if temp_fasta_created and fasta_path and os.path.exists(fasta_path):
            os.unlink(fasta_path)
        update_dict = result[3]
        tar_file_path = update_dict.get("value", "")
        base_dir = get_save_path("Zero_shot", "HeatMap")
        tar_filename = os.path.basename(tar_file_path)
        raw_result = result[2]
        if isinstance(raw_result, dict):
            raw_result = _enrich_mutation_result(raw_result, tar_filename, backend)
        return {"status": "success", "data": raw_result, "file_info": {"output_dir": str(base_dir)}}
    except Exception as e:
        if temp_fasta_created and fasta_path and os.path.exists(fasta_path):
            try:
                os.unlink(fasta_path)
            except Exception:
                pass
        return _error_dict("PredictionError", str(e), suggestion=f"Check sequence, model_name, and Gradio backend at {gradio_url}.")


def zero_shot_mutation_structure_prediction(
    structure_file: str,
    model_name: str = "ESM-IF1",
    api_key: Optional[str] = None,
    backend: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run zero-shot structure-based mutation prediction via Gradio. Returns status dict.
    """
    backend = backend or DEFAULT_BACKEND
    if not structure_file or not os.path.exists(structure_file):
        return _error_dict("ValidationError", f"Structure file not found: {structure_file}")

    gradio_url = get_gradio_base_url(backend)
    if not _check_gradio_reachable(gradio_url):
        return _error_dict(
            "ConnectionError",
            f"Gradio backend is not reachable at {gradio_url}",
            suggestion=f"Start the Gradio server or set GRADIO_BASE_URL env var. Current URL: {gradio_url}",
        )

    try:
        processed_file = extract_first_chain_from_pdb_file(structure_file)
        client = Client(gradio_url)
        result = client.predict(
            function_selection="Activity",
            file_obj=handle_file(processed_file),
            enable_ai=False,
            llm_model="DeepSeek",
            user_api_key=api_key,
            model_name=model_name,
            api_name="/handle_mutation_prediction_base",
        )
        update_dict = result[3]
        tar_file_path = update_dict.get("value", "")
        base_dir = get_save_path("Zero_shot", "HeatMap")
        tar_filename = os.path.basename(tar_file_path)
        raw_result = result[2]
        if isinstance(raw_result, dict):
            raw_result = _enrich_mutation_result(raw_result, tar_filename, backend)
        return {"status": "success", "data": raw_result, "file_info": {"output_dir": str(base_dir)}}
    except Exception as e:
        return _error_dict("PredictionError", str(e), suggestion="Check structure file path and Gradio backend.")
