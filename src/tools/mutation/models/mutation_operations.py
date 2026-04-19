# Core layer: zero-shot mutation prediction (direct call). Returns status dict.

import os
import time
import uuid
from typing import Any, Dict, Optional

from web.utils.common_utils import get_save_path
from web.utils.file_handlers import extract_first_chain_from_pdb_file
from web.utils.prediction_runners import run_zero_shot_prediction

DEFAULT_BACKEND = "local"


def _error_dict(error_type: str, message: str, suggestion: Optional[str] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"status": "error", "error": {"type": error_type, "message": message}, "file_info": None}
    if suggestion:
        out["error"]["suggestion"] = suggestion
    return out


def _df_to_result(df, output_dir: str) -> Dict[str, Any]:
    """Convert prediction DataFrame to result dict, save CSV."""
    records = df.to_dict(orient="split")
    data = records.get("data", [])
    total = len(data)
    result: Dict[str, Any] = {"data": data, "columns": records.get("columns", [])}
    if total > 100:
        result["data"] = data[:50] + [["...", "...", "..."]]
        result["total_mutations"] = total
        result["displayed_mutations"] = 50
        result["note"] = f"Showing top 50 of {total} mutations. Results separated by '...'."

    csv_path = os.path.join(output_dir, f"mutation_result_{int(time.time())}.csv")
    try:
        df.to_csv(csv_path, index=False)
        result["csv_path"] = csv_path
    except Exception:
        pass
    return result


def zero_shot_mutation_sequence_prediction(
    sequence: Optional[str] = None,
    fasta_file: Optional[str] = None,
    model_name: str = "ESM2-650M",
    api_key: Optional[str] = None,
    backend: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run zero-shot sequence-based mutation prediction. Returns status dict.
    Provide either sequence or fasta_file.
    """
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

    try:
        status_msg, raw_df = run_zero_shot_prediction("sequence", model_name, fasta_path)
        if raw_df.empty:
            return _error_dict("PredictionError", status_msg or "Prediction returned empty results.",
                               suggestion="Check sequence and model_name.")
        output_dir = str(get_save_path("Zero_shot", "HeatMap"))
        result_data = _df_to_result(raw_df, output_dir)
        return {"status": "success", "data": result_data, "file_info": {"output_dir": output_dir}}
    except Exception as e:
        return _error_dict("PredictionError", str(e), suggestion="Check sequence, model_name, and GPU environment.")
    finally:
        if temp_fasta_created and fasta_path and os.path.exists(fasta_path):
            try:
                os.unlink(fasta_path)
            except Exception:
                pass


def zero_shot_mutation_structure_prediction(
    structure_file: str,
    model_name: str = "ESM-IF1",
    api_key: Optional[str] = None,
    backend: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run zero-shot structure-based mutation prediction. Returns status dict.
    """
    if not structure_file or not os.path.exists(structure_file):
        return _error_dict("ValidationError", f"Structure file not found: {structure_file}")

    try:
        processed_file = extract_first_chain_from_pdb_file(structure_file)
        status_msg, raw_df = run_zero_shot_prediction("structure", model_name, processed_file)
        if raw_df.empty:
            return _error_dict("PredictionError", status_msg or "Prediction returned empty results.",
                               suggestion="Check structure file and model_name.")
        output_dir = str(get_save_path("Zero_shot", "HeatMap"))
        result_data = _df_to_result(raw_df, output_dir)
        return {"status": "success", "data": result_data, "file_info": {"output_dir": output_dir}}
    except Exception as e:
        return _error_dict("PredictionError", str(e), suggestion="Check structure file path and GPU environment.")
