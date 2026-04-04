import json
import math
import os
import mimetypes
import re
import uuid
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from web.utils.common_utils import (
    build_web_v2_download_url,
    build_run_id_utc,
    create_run_manifest,
    ensure_within_roots,
    get_web_v2_area_dir,
    make_web_v2_result_name,
    make_web_v2_upload_name,
    resolve_web_v2_client_path,
    to_web_v2_public_path,
)
from web.utils.constants import LLM_MODELS
from web.utils.llm_helpers import LLMConfig, call_llm_api, get_api_key, get_chat_base_url
from web.utils.file_handlers import validate_and_normalize_fasta_content
from web.utils.prediction_runners import run_zero_shot_prediction
from web.utils.function_prediction import prepare_top_residue_heatmap_data, generate_plotly_heatmap
try:
    from src.tools.mutation.models.mutation_operations import (
        zero_shot_mutation_sequence_prediction,
        zero_shot_mutation_structure_prediction,
        DEFAULT_BACKEND,
    )
    from src.tools.predict.finetuned.fintuned_operations import (
        predict_protein_function,
        predict_residue_function,
    )
    from src.tools.predict.features.features_operations import (
        calculate_physchem_from_fasta,
        calculate_rsa_from_pdb,
        calculate_sasa_from_pdb,
        calculate_ss_from_pdb,
    )
    from src.tools.denovo.proteinmpnn.protein_mpnn_function import proteinmpnn_design
except ModuleNotFoundError:
    from tools.mutation.models.mutation_operations import (
        zero_shot_mutation_sequence_prediction,
        zero_shot_mutation_structure_prediction,
        DEFAULT_BACKEND,
    )
    from tools.predict.finetuned.fintuned_operations import (
        predict_protein_function,
        predict_residue_function,
    )
    from tools.predict.features.features_operations import (
        calculate_physchem_from_fasta,
        calculate_rsa_from_pdb,
        calculate_sasa_from_pdb,
        calculate_ss_from_pdb,
    )
    from tools.denovo.proteinmpnn.protein_mpnn_function import proteinmpnn_design


router = APIRouter(prefix="/api/quick-tools", tags=["quick-tools-v2"])

_CONSTANT_PATH = Path(__file__).resolve().parent.parent / "constant.json"
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_FASTA_EXAMPLE = _REPO_ROOT / "example" / "database" / "P60002.fasta"
_DEFAULT_PDB_EXAMPLE = _REPO_ROOT / "example" / "database" / "alphafold" / "A0A1B0GTW7.pdb"
_ALLOWED_DOWNLOAD_EXT = {
    ".json",
    ".csv",
    ".tsv",
    ".txt",
    ".html",
    ".htm",
    ".md",
    ".tar",
    ".gz",
    ".tar.gz",
}
def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value >= minimum else default


_ONLINE_FASTA_LIMIT = _env_int("WEBUI_V2_ONLINE_FASTA_LIMIT", 50, minimum=1)
_ONLINE_SEQUENCE_DESIGN_LIMIT = _env_int("WEBUI_V2_ONLINE_SEQUENCE_DESIGN_LIMIT", 50, minimum=1)

_WEB_V2_RESULTS_ROOT = get_web_v2_area_dir("results")
_SUBCELLULAR_LABELS_DEEPLOCMULTI = [
    "Nucleus",
    "Cytoplasm",
    "Extracellular",
    "Mitochondrion",
    "Cell.membrane",
    "Endoplasmic.reticulum",
    "Plastid",
    "Golgi.apparatus",
    "Lysosome/Vacuole",
    "Peroxisome",
]
_SUBCELLULAR_LABELS_DEEPLOC2MULTI = [
    "Cytoplasm",
    "Nucleus",
    "Extracellular",
    "Cell membrane",
    "Mitochondrion",
    "Plastid",
    "Endoplasmic reticulum",
    "Lysosome/Vacuole",
    "Golgi apparatus",
    "Peroxisome",
]
_MEMBRANE_LABELS = ["M", "S"]
_SORTINGSIGNAL_LABELS = ["MT", "SP", "GPI", "NLS", "PTS", "CH", "NES", "TH", "TM"]


def _runtime_mode() -> str:
    mode = os.getenv("WEBUI_V2_MODE", "local").strip().lower()
    return mode if mode in {"local", "online"} else "local"


def _online_fasta_limit_enabled() -> bool:
    return _runtime_mode() == "online"


def _count_fasta_records(normalized_fasta: str) -> int:
    return sum(1 for line in normalized_fasta.splitlines() if line.startswith(">"))


def _parse_normalized_fasta_records(normalized_fasta: str) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    current_header = ""
    current_seq_parts: list[str] = []
    for line in normalized_fasta.splitlines():
        if line.startswith(">"):
            if current_header and current_seq_parts:
                records.append((current_header, "".join(current_seq_parts)))
            current_header = line[1:].strip() or "sequence"
            current_seq_parts = []
            continue
        if line:
            current_seq_parts.append(line.strip())
    if current_header and current_seq_parts:
        records.append((current_header, "".join(current_seq_parts)))
    return records


def _enforce_online_fasta_limit(normalized_fasta: str, source: str = "FASTA input") -> None:
    if not _online_fasta_limit_enabled():
        return
    count = _count_fasta_records(normalized_fasta)
    if count > _ONLINE_FASTA_LIMIT:
        raise HTTPException(
            status_code=400,
            detail=f"{source} contains {count} sequences. Online mode supports up to {_ONLINE_FASTA_LIMIT} sequences per run.",
        )


def _load_normalized_fasta_from_path(path: str) -> str:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="FASTA file must be UTF-8 encoded.") from exc
    try:
        return validate_and_normalize_fasta_content(text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _enforce_online_fasta_limit_for_path(path: str, source: str = "FASTA input") -> None:
    normalized = _load_normalized_fasta_from_path(path)
    _enforce_online_fasta_limit(normalized, source=source)


def _sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _stream_success(payload: Dict[str, Any], final_message: str = "Quick tool completed.") -> list[str]:
    return [
        _sse("progress", {"progress": 0.98, "message": "Finalizing output artifacts..."}),
        _sse(
            "done",
            {
                "success": True,
                "final_progress": 1.0,
                "message": final_message,
                "result_payload": payload,
            },
        ),
    ]


def _stream_error(message: str, status_code: int = 400) -> list[str]:
    return [
        _sse("error", {"message": message, "status_code": status_code}),
        _sse(
            "done",
            {
                "success": False,
                "final_progress": 0.0,
                "message": message,
            },
        ),
    ]


def _new_upload_path(tool: str, original_name: str) -> tuple[str, Path, str]:
    run_id = build_run_id_utc()
    upload_dir = get_web_v2_area_dir("uploads", tool=tool, run_id=run_id)
    filename = make_web_v2_upload_name(1, original_name)
    return run_id, upload_dir / filename, filename


def _result_dir(tool: str, run_id: str) -> Path:
    return get_web_v2_area_dir("results", tool=tool, run_id=run_id)


def _resolve_upload_input(path_value: str) -> str:
    try:
        resolved = resolve_web_v2_client_path(path_value, allowed_areas=("uploads",))
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Access denied.") from exc
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail=f"Input file not found: {path_value}")
    return str(resolved)


def _stage_result_file(source_path: Path, tool: str, run_id: str, kind: str) -> Path:
    ext = source_path.suffix.lower()
    dst = _result_dir(tool, run_id) / make_web_v2_result_name(kind, 1, ext)
    dst.write_bytes(source_path.read_bytes())
    create_run_manifest(
        run_id=run_id,
        tool=tool,
        status="completed",
        inputs=[],
        outputs=[
            {
                "path": str(dst.relative_to(_WEB_V2_RESULTS_ROOT)),
                "mime": mimetypes.guess_type(str(dst))[0] or "application/octet-stream",
                "size": dst.stat().st_size,
            }
        ],
    )
    return dst


def _cleanup_temp_source_file(source_path: Optional[Path], staged_path: Optional[Path]) -> None:
    """Remove intermediate source artifact after staging into canonical results."""
    if not source_path or not source_path.exists() or not source_path.is_file():
        return
    if staged_path and source_path.resolve() == staged_path.resolve():
        return
    if ensure_within_roots(source_path, [_WEB_V2_RESULTS_ROOT]):
        return
    try:
        source_path.unlink(missing_ok=True)
    except Exception:
        pass


def _as_json_dict(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        return {"status": "error", "error": {"type": "ParseError", "message": payload}, "file_info": None}
    return {"status": "error", "error": {"type": "TypeError", "message": "Unsupported response payload."}, "file_info": None}


def _parse_fasta_preview_rows(fasta_path: Path, max_records: int = 200) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not fasta_path.exists() or not fasta_path.is_file():
        return rows
    header = ""
    seq_parts: list[str] = []

    def _flush() -> None:
        nonlocal header, seq_parts
        if not header:
            return
        sequence = "".join(seq_parts)
        score = ""
        global_score = ""
        header_text = header[1:] if header.startswith(">") else header
        m_score = re.search(r"score=([\-0-9.]+)", header_text)
        m_global = re.search(r"global_score=([\-0-9.]+)", header_text)
        if m_score:
            score = m_score.group(1)
        if m_global:
            global_score = m_global.group(1)
        rows.append(
            {
                "header": header_text,
                "sequence": sequence,
                "length": len(sequence),
                "score": score,
                "global_score": global_score,
            }
        )
        header = ""
        seq_parts = []

    with open(fasta_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                _flush()
                header = line
                if len(rows) >= max_records:
                    break
            else:
                seq_parts.append(line)
        if len(rows) < max_records:
            _flush()
    return rows


def _normalize_key(value: str) -> str:
    return re.sub(r"[\s._-]+", "", (value or "").strip().lower())


def _label_names_for_task_and_dataset(task: str, dataset: str) -> Optional[list[str]]:
    task_key = _normalize_key(task)
    dataset_key = _normalize_key(dataset)
    if task_key == "subcellularlocalization":
        if dataset_key == "deeploc2multi":
            return _SUBCELLULAR_LABELS_DEEPLOC2MULTI
        if dataset_key == "deeplocmulti":
            return _SUBCELLULAR_LABELS_DEEPLOCMULTI
        return _SUBCELLULAR_LABELS_DEEPLOCMULTI
    if task_key in {"membraneprotein", "deeplocbinary"}:
        return _MEMBRANE_LABELS
    if task_key == "sortingsignal":
        return _SORTINGSIGNAL_LABELS
    return None


def _parse_class_index(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        m = re.match(r"^\s*(-?\d+)", value)
        if m:
            return int(m.group(1))
    return None


def _map_predicted_class_rows(rows: list[dict[str, Any]], task: str) -> tuple[list[dict[str, Any]], Dict[str, list[str]]]:
    mapped_rows: list[dict[str, Any]] = []
    label_context: Dict[str, list[str]] = {}
    for row in rows:
        next_row = dict(row)
        dataset = str(next_row.get("Dataset") or next_row.get("dataset") or next_row.get("dataset_name") or "")
        label_names = _label_names_for_task_and_dataset(task, dataset)
        if label_names and dataset:
            label_context[dataset] = label_names
        for key in ("predicted_class", "Predicted Class"):
            if key not in next_row:
                continue
            idx = _parse_class_index(next_row.get(key))
            if idx is None or not label_names or idx < 0 or idx >= len(label_names):
                continue
            next_row[key] = label_names[idx]
            next_row.setdefault("predicted_class_id", idx)
            next_row.setdefault("predicted_class_name", label_names[idx])
        mapped_rows.append(next_row)
    return mapped_rows, label_context


def _apply_protein_function_label_mapping(payload: Dict[str, Any], csv_path: Optional[Path], task: str) -> None:
    mapped_preview: list[dict[str, Any]] = []
    label_context: Dict[str, list[str]] = {}
    if csv_path and csv_path.exists() and csv_path.is_file():
        try:
            df = pd.read_csv(csv_path)
            rows = df.fillna("").to_dict(orient="records")
            mapped_rows, label_context = _map_predicted_class_rows(rows, task)
            mapped_preview = mapped_rows[:200]
            if mapped_rows:
                pd.DataFrame(mapped_rows).to_csv(csv_path, index=False)
        except Exception:
            mapped_preview = []
            label_context = {}

    content_preview = payload.get("content_preview")
    if isinstance(content_preview, str):
        try:
            raw_rows = json.loads(content_preview)
            if isinstance(raw_rows, list):
                safe_rows = [r for r in raw_rows if isinstance(r, dict)]
                if len(safe_rows) == len(raw_rows):
                    mapped_rows, content_context = _map_predicted_class_rows(safe_rows, task)
                    payload["content_preview"] = json.dumps(mapped_rows, ensure_ascii=False, indent=2)[:12000]
                    for k, v in content_context.items():
                        label_context.setdefault(k, v)
        except Exception:
            pass

    if mapped_preview:
        payload["mapped_rows_preview"] = mapped_preview
    task_default = _label_names_for_task_and_dataset(task, "")
    if task_default:
        payload["class_label_names"] = task_default
    if label_context:
        payload["class_label_names_by_dataset"] = label_context


@router.get("/meta")
async def quick_tools_meta():
    if not _CONSTANT_PATH.exists():
        raise HTTPException(status_code=404, detail="constant.json not found.")
    data = json.loads(_CONSTANT_PATH.read_text(encoding="utf-8"))
    web_ui = data.get("web_ui", {})
    return {
        "dataset_mapping_zero_shot": web_ui.get("dataset_mapping_zero_shot", []),
        "model_mapping_zero_shot": list(web_ui.get("model_mapping_zero_shot", {}).keys()),
        "dataset_mapping_function": list(web_ui.get("dataset_mapping_function", {}).keys()),
        "residue_mapping_function": list(web_ui.get("residue_mapping_function", {}).keys()),
        "protein_properties_function": web_ui.get("protein_properties_function", []),
        "llm_models": list(LLM_MODELS.keys()),
        "mode": _runtime_mode(),
        "online_fasta_limit": _ONLINE_FASTA_LIMIT,
        "online_sequence_design_limit": _ONLINE_SEQUENCE_DESIGN_LIMIT,
        "online_limit_enabled": _online_fasta_limit_enabled(),
    }


@router.post("/upload")
async def upload_quick_tool_file(file: UploadFile = File(...)):
    filename = os.path.basename(file.filename or f"quick-tools-{uuid.uuid4().hex}.txt")
    suffix = Path(filename).suffix.lower()
    if suffix not in {".fasta", ".fa", ".pdb"}:
        raise HTTPException(status_code=400, detail="Only .fasta/.fa/.pdb files are supported.")
    run_id, dst, stored_name = _new_upload_path("quick_tools", filename)
    content = await file.read()
    if suffix in {".fasta", ".fa"}:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=400, detail="FASTA file must be UTF-8 encoded.") from exc
        try:
            normalized = validate_and_normalize_fasta_content(text)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _enforce_online_fasta_limit(normalized, source="Uploaded FASTA")
        with open(dst, "w", encoding="utf-8") as out:
            out.write(normalized)
    else:
        with open(dst, "wb") as out:
            out.write(content)
    create_run_manifest(
        run_id=run_id,
        tool="quick_tools",
        status="uploaded",
        inputs=[{"path": str(dst), "name": stored_name, "size": len(content)}],
    )
    return {"file_path": to_web_v2_public_path(dst), "name": filename, "suffix": suffix, "run_id": run_id}


@router.get("/default-example")
async def quick_tool_default_example(kind: str = "fasta"):
    normalized_kind = (kind or "fasta").strip().lower()
    if normalized_kind == "pdb":
        source = _DEFAULT_PDB_EXAMPLE
    else:
        source = _DEFAULT_FASTA_EXAMPLE

    if not source.exists():
        raise HTTPException(status_code=404, detail=f"Example file not found for kind={normalized_kind}")

    suffix = source.suffix.lower()
    run_id, dst, _ = _new_upload_path("quick_tools", f"example_{source.name}")
    dst.write_bytes(source.read_bytes())
    content = ""
    if suffix in {".fasta", ".fa"}:
        content = source.read_text(encoding="utf-8")

    return {
        "file_path": to_web_v2_public_path(dst),
        "name": source.name,
        "suffix": suffix,
        "kind": normalized_kind,
        "content": content,
        "run_id": run_id,
    }


def _normalize_extension(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".tar.gz"):
        return ".tar.gz"
    return path.suffix.lower()


@router.get("/download")
async def download_quick_tool_result(file_path: str, inline: bool = False):
    try:
        path = resolve_web_v2_client_path(file_path, allowed_areas=("results",))
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied.")
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Result file not found.")
    if not ensure_within_roots(path, [_WEB_V2_RESULTS_ROOT]):
        raise HTTPException(status_code=403, detail="Access denied.")
    ext = _normalize_extension(path)
    if ext not in _ALLOWED_DOWNLOAD_EXT:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
    if inline and ext in {".html", ".htm"}:
        return FileResponse(
            path,
            media_type="text/html; charset=utf-8",
            headers={"Content-Disposition": "inline"},
        )
    return FileResponse(path, filename=path.name)


class QuickToolAiSummaryBody(BaseModel):
    tool: str = Field(..., description="Tool name, such as mutation/function/residue/properties.")
    task: str = Field(default="", description="Task name selected by user.")
    llm_provider: str = Field(default="DeepSeek", description="LLM provider name.")
    user_api_key: str = Field(default="", description="Optional user provided API key.")
    result_payload: dict = Field(default_factory=dict, description="API result JSON payload.")


@router.post("/ai-summary")
async def quick_tool_ai_summary(body: QuickToolAiSummaryBody):
    api_key = get_api_key(body.llm_provider, body.user_api_key)
    if not api_key:
        raise HTTPException(status_code=400, detail="No API key available. Set OPENAI_API_KEY or provide key.")
    if body.llm_provider not in LLM_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported LLM provider: {body.llm_provider}")

    model = LLM_MODELS[body.llm_provider]
    prompt = (
        "You are an expert protein scientist.\n"
        f"Tool: {body.tool}\n"
        f"Task: {body.task or 'N/A'}\n"
        "Analyze the following prediction output and provide:\n"
        "1) Key findings, 2) Confidence interpretation, 3) Practical wet-lab next steps.\n"
        "Keep it concise, actionable, and under 200 words.\n\n"
        f"Prediction output:\n{json.dumps(body.result_payload, ensure_ascii=False)[:12000]}"
    )

    config = LLMConfig(
        api_key=api_key,
        llm_name=body.llm_provider,
        api_base=get_chat_base_url(),
        model=model,
    )
    summary = call_llm_api(config, prompt)
    return {"summary": summary, "provider": body.llm_provider, "model": model}


class MutationRunBody(BaseModel):
    sequence: str = Field(default="", description="Protein sequence (optional if fasta_file provided).")
    fasta_file: str = Field(default="", description="FASTA file path.")
    structure_file: str = Field(default="", description="PDB file path for structure mode.")
    model_name: str = Field(default="ESM2-650M", description="Model name.")
    backend: str = Field(default=DEFAULT_BACKEND, description="Backend key.")
    api_key: str = Field(default="", description="Optional API key.")


class SequenceDesignRunBody(BaseModel):
    structure_file: str = Field(..., description="PDB file path for ProteinMPNN sequence design.")
    model_family: str = Field(default="soluble", description="soluble | vanilla | ca")
    designed_chains: List[str] = Field(default_factory=list, description="Optional designed chain ids.")
    fixed_residues_text: str = Field(default="", description="Optional fixed residues text, e.g. A12,C13 or A:12,13;B:5-8")
    num_sequences: int = Field(default=8, ge=1, le=512, description="Number of designed sequences.")
    model_name: str = Field(default="v_48_020", description="ProteinMPNN model checkpoint name.")
    backbone_noise: float = Field(default=0.2, description="Backbone Gaussian noise.")
    use_soluble_model: bool = Field(default=True, description="Use soluble-only weights.")
    ca_only: bool = Field(default=False, description="Use CA-only model.")
    temperatures: List[float] = Field(default_factory=lambda: [0.1], description="Sampling temperatures.")


def _parse_fixed_residues_text(text: str) -> dict[str, list[int]]:
    raw = (text or "").strip()
    if not raw:
        return {}
    result: dict[str, set[int]] = {}

    def _add(chain: str, start: int, end: int) -> None:
        if start <= 0 or end <= 0:
            raise HTTPException(status_code=400, detail="Invalid fixed_residues_text: residue index must be >= 1")
        if end < start:
            raise HTTPException(status_code=400, detail=f"Invalid fixed_residues_text: invalid range {start}-{end}")
        key = chain.strip().upper()
        if not re.fullmatch(r"[A-Z0-9]", key):
            raise HTTPException(status_code=400, detail=f"Invalid fixed_residues_text: invalid chain '{chain}'")
        bucket = result.setdefault(key, set())
        bucket.update(range(start, end + 1))

    if ":" in raw:
        groups = [g.strip() for g in raw.split(";") if g.strip()]
        for group in groups:
            if ":" not in group:
                raise HTTPException(status_code=400, detail=f"Invalid fixed_residues_text: expected A:12,13 in '{group}'")
            chain, positions_text = group.split(":", 1)
            if not positions_text.strip():
                raise HTTPException(status_code=400, detail=f"Invalid fixed_residues_text: missing residues for chain '{chain}'")
            for token in [t.strip() for t in positions_text.split(",") if t.strip()]:
                if "-" in token:
                    start_str, end_str = token.split("-", 1)
                    if not start_str.isdigit() or not end_str.isdigit():
                        raise HTTPException(status_code=400, detail=f"Invalid fixed_residues_text: invalid range '{token}'")
                    _add(chain, int(start_str), int(end_str))
                else:
                    if not token.isdigit():
                        raise HTTPException(status_code=400, detail=f"Invalid fixed_residues_text: invalid residue '{token}'")
                    idx = int(token)
                    _add(chain, idx, idx)
    else:
        tokens = [t.strip() for t in raw.split(",") if t.strip()]
        for token in tokens:
            m = re.fullmatch(r"([A-Za-z0-9])(\d+)(?:-(\d+))?", token)
            if not m:
                raise HTTPException(status_code=400, detail=f"Invalid fixed_residues_text token '{token}', use A12 or A12-15")
            chain = m.group(1)
            start = int(m.group(2))
            end = int(m.group(3) or start)
            _add(chain, start, end)

    return {chain: sorted(values) for chain, values in result.items()}


@router.post("/run/sequence-design")
async def run_quick_tool_sequence_design(body: SequenceDesignRunBody):
    input_path = _resolve_upload_input(body.structure_file)
    if not input_path.lower().endswith(".pdb"):
        raise HTTPException(status_code=400, detail="Sequence Design only supports .pdb input.")
    if _online_fasta_limit_enabled() and body.num_sequences > _ONLINE_SEQUENCE_DESIGN_LIMIT:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Online mode supports up to {_ONLINE_SEQUENCE_DESIGN_LIMIT} designed sequences per run. "
                f"Received {body.num_sequences}."
            ),
        )

    run_id = build_run_id_utc()
    designed_chains = [c.strip().upper() for c in body.designed_chains if str(c).strip()]
    fixed_residues = _parse_fixed_residues_text(body.fixed_residues_text)
    temperatures = body.temperatures or [0.1]
    if body.ca_only and body.use_soluble_model:
        raise HTTPException(status_code=400, detail="Invalid ProteinMPNN config: CA-only cannot be combined with Soluble.")
    if not math.isfinite(float(body.backbone_noise)):
        raise HTTPException(status_code=400, detail="backbone_noise must be a finite number.")
    try:
        fasta_path = proteinmpnn_design(
            pdb_path=input_path,
            designed_chains=designed_chains or None,
            fixed_residues=fixed_residues or None,
            num_sequences=body.num_sequences,
            temperatures=temperatures,
            omit_aas="X",
            model_name=body.model_name or "v_48_020",
            backbone_noise=float(body.backbone_noise),
            ca_only=bool(body.ca_only),
            use_soluble_model=bool(body.use_soluble_model),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ProteinMPNN sequence design failed: {exc}") from exc

    staged_path = _stage_result_file(Path(fasta_path), "quick_tools", run_id, "sequence_design")
    rows = _parse_fasta_preview_rows(staged_path)
    return {
        "status": "success",
        "message": "Sequence design completed.",
        "table": rows,
        "data": {
            "rows": rows,
            "total_sequences": len(rows),
            "fasta_path": to_web_v2_public_path(staged_path),
            "download_url": build_web_v2_download_url(staged_path),
            "effective_config": {
                "model_family": body.model_family,
                "model_name": body.model_name,
                "backbone_noise": body.backbone_noise,
                "use_soluble_model": body.use_soluble_model,
                "ca_only": body.ca_only,
                "num_sequences": body.num_sequences,
                "temperatures": temperatures,
                "fixed_residues": fixed_residues,
            },
        },
        "file_info": {
            "file_path": to_web_v2_public_path(staged_path),
            "file_name": staged_path.name,
            "format": "fasta",
            "download_url": build_web_v2_download_url(staged_path),
            "run_id": run_id,
        },
    }


@router.post("/run/mutation")
async def run_quick_tool_mutation(body: MutationRunBody):
    # Run local scripts directly (legacy quick-tool behavior) to avoid requiring an external Gradio service.
    model_type = "sequence"
    model_name = body.model_name or "ESM2-650M"
    input_path = ""

    run_id = build_run_id_utc()
    if body.structure_file:
        model_type = "structure"
        model_name = "ESM-IF1"
        input_path = _resolve_upload_input(body.structure_file)
    else:
        if body.fasta_file:
            input_path = _resolve_upload_input(body.fasta_file)
        elif body.sequence.strip():
            seq_run_id = build_run_id_utc()
            temp_dir = get_web_v2_area_dir("uploads", tool="quick_tools", run_id=seq_run_id)
            temp_path = temp_dir / make_web_v2_upload_name(1, "inline_sequence.fasta")
            try:
                normalized_fasta = validate_and_normalize_fasta_content(body.sequence.strip())
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            _enforce_online_fasta_limit(normalized_fasta, source="Directed Evolution sequence input")
            temp_path.write_text(normalized_fasta, encoding="utf-8")
            input_path = str(temp_path)
        else:
            raise HTTPException(status_code=400, detail="Either sequence/fasta_file or structure_file must be provided.")

    if not os.path.exists(input_path):
        raise HTTPException(status_code=400, detail=f"Input file not found: {input_path}")
    status = ""
    if model_type == "sequence" and input_path.lower().endswith((".fasta", ".fa")):
        normalized_fasta = _load_normalized_fasta_from_path(input_path)
        _enforce_online_fasta_limit(normalized_fasta, source="Directed Evolution FASTA")
        records = _parse_normalized_fasta_records(normalized_fasta)
        if not records:
            raise HTTPException(status_code=400, detail="No valid FASTA sequence found for Directed Evolution.")
        if len(records) != 1:
            raise HTTPException(
                status_code=400,
                detail=f"Directed Evolution supports a single protein per run. Received {len(records)} sequences.",
            )
        header, sequence = records[0]
        sequence_input_dir = get_web_v2_area_dir("uploads", tool="quick_tools", run_id=run_id) / "mutation_sequences"
        sequence_input_dir.mkdir(parents=True, exist_ok=True)
        single_input = sequence_input_dir / "seq_1.fasta"
        single_input.write_text(f">{header}\n{sequence}\n", encoding="utf-8")
        status, raw_df = run_zero_shot_prediction(model_type, model_name, str(single_input))
        if raw_df.empty:
            return {
                "status": "error",
                "error": {
                    "type": "PredictionError",
                    "message": status,
                    "suggestion": "Check sequence input and local model environment.",
                },
                "file_info": None,
            }
    else:
        status, raw_df = run_zero_shot_prediction(model_type, model_name, input_path)
        if raw_df.empty:
            return {
                "status": "error",
                "error": {
                    "type": "PredictionError",
                    "message": status,
                    "suggestion": "Check sequence/structure input and local model environment.",
                },
                "file_info": None,
            }

    score_col = next((c for c in raw_df.columns if "score" in c.lower()), raw_df.columns[1])
    display_df = pd.DataFrame()
    display_df["Mutant"] = raw_df["mutant"]
    display_df["Prediction Rank"] = range(1, len(raw_df) + 1)

    min_s, max_s = raw_df[score_col].min(), raw_df[score_col].max()
    if max_s == min_s:
        scaled_scores = pd.Series([0.0] * len(raw_df))
    else:
        scaled_scores = -1 + 2 * (raw_df[score_col] - min_s) / (max_s - min_s)
    display_df["Prediction Score"] = scaled_scores.round(3)

    result_dir = _result_dir("quick_tools", run_id)
    csv_path = result_dir / make_web_v2_result_name("metrics", 1, ".csv")
    display_df.to_csv(csv_path, index=False)
    heatmap_path: Optional[Path] = None

    # Keep v2 output compatible with legacy quick-tools heatmap artifact.
    try:
        heatmap_input_df = raw_df.copy()
        heatmap_input_df["Prediction Rank"] = range(1, len(heatmap_input_df) + 1)
        heatmap_data = prepare_top_residue_heatmap_data(heatmap_input_df)
        if heatmap_data[0] is not None:
            summary_fig = generate_plotly_heatmap(*heatmap_data[:4])
            heatmap_path = result_dir / make_web_v2_result_name("heatmap", 1, ".html")
            summary_fig.write_html(heatmap_path)
    except Exception:
        heatmap_path = None

    return {
        "status": "success",
        "data": {
            "rows": display_df.to_dict(orient="records"),
            "total_mutations": len(display_df),
            "csv_path": to_web_v2_public_path(csv_path),
            "heatmap_path": to_web_v2_public_path(heatmap_path) if heatmap_path else "",
            "csv_download_url": build_web_v2_download_url(csv_path),
            "heatmap_download_url": build_web_v2_download_url(heatmap_path) if heatmap_path else "",
            "model_type": model_type,
            "model_name": model_name,
            "message": status,
        },
        "file_info": {
            "file_path": to_web_v2_public_path(csv_path),
            "file_name": csv_path.name,
            "format": "csv",
            "heatmap_path": to_web_v2_public_path(heatmap_path) if heatmap_path else "",
            "download_url": build_web_v2_download_url(csv_path),
            "run_id": run_id,
        },
    }


class ProteinFunctionRunBody(BaseModel):
    fasta_file: str = Field(..., description="FASTA file path.")
    task: str = Field(..., description="Task name.")
    model_name: str = Field(default="ESM2-650M", description="Model name.")


@router.post("/run/protein-function")
async def run_quick_tool_protein_function(body: ProteinFunctionRunBody):
    fasta_path = _resolve_upload_input(body.fasta_file)
    _enforce_online_fasta_limit_for_path(fasta_path, source="Protein Function FASTA")
    run_id = build_run_id_utc()
    temp_output = _result_dir("quick_tools", run_id) / make_web_v2_result_name("protein_function_raw", 1, ".csv")
    result = predict_protein_function(
        fasta_file=fasta_path,
        task=body.task,
        model_name=body.model_name,
        output_file=str(temp_output),
    )
    payload = _as_json_dict(result)
    file_info = payload.get("file_info") if isinstance(payload, dict) else None
    source_path = Path(str(file_info.get("file_path", ""))) if isinstance(file_info, dict) else None
    staged_path: Optional[Path] = None
    if source_path and source_path.exists() and source_path.is_file():
        staged_path = _stage_result_file(source_path, "quick_tools", run_id, "protein_function")
        file_info["file_path"] = to_web_v2_public_path(staged_path)
        file_info["file_name"] = staged_path.name
        file_info["download_url"] = build_web_v2_download_url(staged_path)
        file_info["run_id"] = run_id
        _cleanup_temp_source_file(source_path, staged_path)
    _apply_protein_function_label_mapping(payload, staged_path or source_path, body.task)
    return payload


class ResidueFunctionRunBody(BaseModel):
    fasta_file: str = Field(..., description="FASTA file path.")
    task: str = Field(..., description="Residue task name.")
    model_name: str = Field(default="ESM2-650M", description="Model name.")


@router.post("/run/residue-function")
async def run_quick_tool_residue_function(body: ResidueFunctionRunBody):
    fasta_path = _resolve_upload_input(body.fasta_file)
    _enforce_online_fasta_limit_for_path(fasta_path, source="Functional Residue FASTA")
    run_id = build_run_id_utc()
    temp_output = _result_dir("quick_tools", run_id) / make_web_v2_result_name("residue_function_raw", 1, ".csv")
    result = predict_residue_function(
        fasta_file=fasta_path,
        task=body.task,
        model_name=body.model_name,
        output_file=str(temp_output),
    )
    payload = _as_json_dict(result)
    file_info = payload.get("file_info") if isinstance(payload, dict) else None
    source_path = Path(str(file_info.get("file_path", ""))) if isinstance(file_info, dict) else None
    if source_path and source_path.exists() and source_path.is_file():
        staged = _stage_result_file(source_path, "quick_tools", run_id, "residue_function")
        file_info["file_path"] = to_web_v2_public_path(staged)
        file_info["file_name"] = staged.name
        file_info["download_url"] = build_web_v2_download_url(staged)
        file_info["run_id"] = run_id
        _cleanup_temp_source_file(source_path, staged)
    return payload


class PropertiesRunBody(BaseModel):
    task: str = Field(..., description="Property task name.")
    file_path: str = Field(..., description="Input FASTA/PDB file path.")
    chain_id: str = Field(default="A", description="Chain id for PDB-based tasks.")


@router.post("/run/properties")
async def run_quick_tool_properties(body: PropertiesRunBody):
    safe_file_path = _resolve_upload_input(body.file_path)
    if safe_file_path.lower().endswith((".fasta", ".fa")):
        _enforce_online_fasta_limit_for_path(safe_file_path, source="Physicochemical FASTA")
    run_id = build_run_id_utc()

    def _finalize_properties_result(payload: Dict[str, Any], kind: str) -> Dict[str, Any]:
        file_info = payload.get("file_info") if isinstance(payload, dict) else None
        source_path = Path(str(file_info.get("file_path", ""))) if isinstance(file_info, dict) else None
        if source_path and source_path.exists() and source_path.is_file():
            staged = _stage_result_file(source_path, "quick_tools", run_id, kind)
            file_info["file_path"] = to_web_v2_public_path(staged)
            file_info["file_name"] = staged.name
            file_info["download_url"] = build_web_v2_download_url(staged)
            file_info["run_id"] = run_id
            _cleanup_temp_source_file(source_path, staged)
        return payload

    if body.task == "Physical and chemical properties":
        temp_output = _result_dir("quick_tools", run_id) / make_web_v2_result_name("physchem_raw", 1, ".json")
        result = calculate_physchem_from_fasta(safe_file_path, output_file=str(temp_output))
        payload = _as_json_dict(result)
        return _finalize_properties_result(payload, "physchem")
    if "Relative solvent accessible surface area" in body.task:
        temp_output = _result_dir("quick_tools", run_id) / make_web_v2_result_name("rsa_raw", 1, ".json")
        result = calculate_rsa_from_pdb(safe_file_path, chain_id=body.chain_id, output_file=str(temp_output))
        payload = _as_json_dict(result)
        return _finalize_properties_result(payload, "rsa")
    if "SASA value" in body.task:
        temp_output = _result_dir("quick_tools", run_id) / make_web_v2_result_name("sasa_raw", 1, ".json")
        result = calculate_sasa_from_pdb(safe_file_path, output_file=str(temp_output))
        payload = _as_json_dict(result)
        return _finalize_properties_result(payload, "sasa")
    if "Secondary structure" in body.task:
        temp_output = _result_dir("quick_tools", run_id) / make_web_v2_result_name("ss_raw", 1, ".json")
        result = calculate_ss_from_pdb(safe_file_path, chain_id=body.chain_id, output_file=str(temp_output))
        payload = _as_json_dict(result)
        return _finalize_properties_result(payload, "secondary_structure")
    raise HTTPException(status_code=400, detail=f"Unsupported properties task: {body.task}")


@router.post("/run/mutation/stream")
async def run_quick_tool_mutation_stream(body: MutationRunBody):
    async def event_stream():
        start = time.perf_counter()
        yield _sse("progress", {"progress": 0.08, "message": "Validating Directed Evolution input..."})
        try:
            yield _sse("progress", {"progress": 0.35, "message": "Running Directed Evolution model..."})
            payload = await run_quick_tool_mutation(body)
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            for chunk in _stream_success(payload, final_message=f"Directed Evolution completed in {elapsed_ms} ms."):
                yield chunk
        except HTTPException as exc:
            message = str(exc.detail) if exc.detail else "Directed Evolution failed."
            for chunk in _stream_error(message, status_code=exc.status_code):
                yield chunk
        except Exception as exc:
            for chunk in _stream_error(f"Directed Evolution failed: {exc}", status_code=500):
                yield chunk

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/run/sequence-design/stream")
async def run_quick_tool_sequence_design_stream(body: SequenceDesignRunBody):
    async def event_stream():
        start = time.perf_counter()
        yield _sse("progress", {"progress": 0.08, "message": "Validating structure input..."})
        try:
            yield _sse("progress", {"progress": 0.35, "message": "Running ProteinMPNN sequence design..."})
            payload = await run_quick_tool_sequence_design(body)
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            for chunk in _stream_success(payload, final_message=f"Sequence Design completed in {elapsed_ms} ms."):
                yield chunk
        except HTTPException as exc:
            message = str(exc.detail) if exc.detail else "Sequence Design failed."
            for chunk in _stream_error(message, status_code=exc.status_code):
                yield chunk
        except Exception as exc:
            for chunk in _stream_error(f"Sequence Design failed: {exc}", status_code=500):
                yield chunk

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/run/protein-function/stream")
async def run_quick_tool_protein_function_stream(body: ProteinFunctionRunBody):
    async def event_stream():
        start = time.perf_counter()
        yield _sse("progress", {"progress": 0.08, "message": "Validating FASTA input..."})
        try:
            fasta_path = _resolve_upload_input(body.fasta_file)
            normalized = _load_normalized_fasta_from_path(fasta_path)
            total = max(1, _count_fasta_records(normalized))
            yield _sse("progress", {"progress": 0.2, "message": f"Loaded {total} protein sequence(s)."})
            yield _sse("progress", {"progress": 0.45, "message": "Running Protein Function model..."})
            payload = await run_quick_tool_protein_function(body)
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            for chunk in _stream_success(payload, final_message=f"Protein Function completed in {elapsed_ms} ms."):
                yield chunk
        except HTTPException as exc:
            message = str(exc.detail) if exc.detail else "Protein Function failed."
            for chunk in _stream_error(message, status_code=exc.status_code):
                yield chunk
        except Exception as exc:
            for chunk in _stream_error(f"Protein Function failed: {exc}", status_code=500):
                yield chunk

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/run/residue-function/stream")
async def run_quick_tool_residue_function_stream(body: ResidueFunctionRunBody):
    async def event_stream():
        start = time.perf_counter()
        yield _sse("progress", {"progress": 0.08, "message": "Validating FASTA input..."})
        try:
            fasta_path = _resolve_upload_input(body.fasta_file)
            normalized = _load_normalized_fasta_from_path(fasta_path)
            total = max(1, _count_fasta_records(normalized))
            yield _sse("progress", {"progress": 0.2, "message": f"Loaded {total} protein sequence(s)."})
            yield _sse("progress", {"progress": 0.45, "message": "Running Functional Residue model..."})
            payload = await run_quick_tool_residue_function(body)
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            for chunk in _stream_success(payload, final_message=f"Functional Residue completed in {elapsed_ms} ms."):
                yield chunk
        except HTTPException as exc:
            message = str(exc.detail) if exc.detail else "Functional Residue failed."
            for chunk in _stream_error(message, status_code=exc.status_code):
                yield chunk
        except Exception as exc:
            for chunk in _stream_error(f"Functional Residue failed: {exc}", status_code=500):
                yield chunk

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/run/properties/stream")
async def run_quick_tool_properties_stream(body: PropertiesRunBody):
    async def event_stream():
        start = time.perf_counter()
        yield _sse("progress", {"progress": 0.08, "message": "Validating property input..."})
        try:
            yield _sse("progress", {"progress": 0.4, "message": "Running Physicochemical Property model..."})
            payload = await run_quick_tool_properties(body)
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            for chunk in _stream_success(payload, final_message=f"Physicochemical Property completed in {elapsed_ms} ms."):
                yield chunk
        except HTTPException as exc:
            message = str(exc.detail) if exc.detail else "Physicochemical Property failed."
            for chunk in _stream_error(message, status_code=exc.status_code):
                yield chunk
        except Exception as exc:
            for chunk in _stream_error(f"Physicochemical Property failed: {exc}", status_code=500):
                yield chunk

    return StreamingResponse(event_stream(), media_type="text/event-stream")
