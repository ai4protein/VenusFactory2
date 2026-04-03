import json
import os
import tarfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from web.advanced_tool_tab import (
    handle_VenusMine,
    handle_mutation_prediction_advance,
    handle_protein_function_prediction_advance,
)
from web.quick_tool_tab import handle_protein_residue_function_prediction
from web.utils.common_utils import (
    build_web_v2_download_url,
    build_run_id_utc,
    create_run_manifest,
    ensure_within_roots,
    get_temp_outputs_base_dir,
    get_web_v2_area_dir,
    get_web_v2_root_dir,
    make_web_v2_result_name,
    make_web_v2_upload_name,
    resolve_web_v2_client_path,
    to_web_v2_public_path,
)
from web.utils.constants import LLM_MODELS
from web.utils.file_handlers import validate_and_normalize_fasta_content
from web.utils.llm_helpers import LLMConfig, call_llm_api, get_api_key, get_chat_base_url


router = APIRouter(prefix="/api/v2/advanced-tools", tags=["advanced-tools-v2"])

_CONSTANT_PATH = Path(__file__).resolve().parent.parent / "constant.json"
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_WEB_V2_ROOT = get_web_v2_root_dir().resolve()
_TEMP_OUTPUTS_ROOT = get_temp_outputs_base_dir().resolve()
_WEB_V2_RESULTS_ROOT = get_web_v2_area_dir("results")
_STAGE_ALLOWED_SOURCE_ROOTS = [_REPO_ROOT.resolve(), _WEB_V2_ROOT, _TEMP_OUTPUTS_ROOT]
_DEFAULT_FASTA_EXAMPLE = _REPO_ROOT / "example" / "database" / "P60002.fasta"
_DEFAULT_PDB_EXAMPLE = _REPO_ROOT / "example" / "database" / "alphafold" / "A0A1B0GTW7.pdb"
_ONLINE_FASTA_LIMIT = 50
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
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
}


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


def _load_normalized_fasta_from_path(path: str) -> str:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="FASTA file must be UTF-8 encoded.") from exc
    try:
        return validate_and_normalize_fasta_content(text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _enforce_online_fasta_limit(normalized_fasta: str, source: str = "FASTA input") -> None:
    if not _online_fasta_limit_enabled():
        return
    count = _count_fasta_records(normalized_fasta)
    if count > _ONLINE_FASTA_LIMIT:
        raise HTTPException(
            status_code=400,
            detail=f"{source} contains {count} sequences. Online mode supports up to {_ONLINE_FASTA_LIMIT} sequences per run.",
        )


def _sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _stream_success(payload: Dict[str, Any], final_message: str) -> list[str]:
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


class DirectedEvolutionBody(BaseModel):
    input_mode: str = Field(default="sequence", description="sequence or structure")
    function_selection: Optional[str] = Field(default=None)
    file_path: Optional[str] = Field(default=None)
    sequence: Optional[str] = Field(default=None)
    model_name: str = Field(default="ESM2-650M")
    enable_ai: bool = Field(default=False)
    llm_provider: str = Field(default="DeepSeek")
    user_api_key: str = Field(default="")


class ProteinFunctionBody(BaseModel):
    task: str = Field(default="Solubility")
    file_path: Optional[str] = Field(default=None)
    sequence: Optional[str] = Field(default=None)
    model_name: str = Field(default="ESM2-650M")
    datasets: List[str] = Field(default_factory=list)
    enable_ai: bool = Field(default=False)
    llm_provider: str = Field(default="DeepSeek")
    user_api_key: str = Field(default="")


class FunctionalResidueBody(BaseModel):
    task: str = Field(default="Activity Site")
    file_path: Optional[str] = Field(default=None)
    sequence: Optional[str] = Field(default=None)
    model_name: str = Field(default="ESM2-650M")
    enable_ai: bool = Field(default=False)
    llm_provider: str = Field(default="DeepSeek")
    user_api_key: str = Field(default="")


class ProteinDiscoveryBody(BaseModel):
    pdb_file: str = Field(...)
    protect_start: int = Field(default=1)
    protect_end: int = Field(default=100)
    mmseqs_threads: int = Field(default=96)
    mmseqs_iterations: int = Field(default=3)
    mmseqs_max_seqs: int = Field(default=100)
    cluster_min_seq_id: float = Field(default=0.5)
    cluster_threads: int = Field(default=96)
    top_n_threshold: int = Field(default=10)
    evalue_threshold: float = Field(default=1e-5)


class AdvancedAiSummaryBody(BaseModel):
    tool: str = Field(..., description="Tool name such as directed-evolution/function/residue.")
    task: str = Field(default="", description="Task name selected by user.")
    llm_provider: str = Field(default="DeepSeek", description="LLM provider.")
    user_api_key: str = Field(default="", description="Optional user API key.")
    result_payload: Dict[str, Any] = Field(default_factory=dict, description="Merged result payload.")


def _normalize_extension(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".tar.gz"):
        return ".tar.gz"
    return path.suffix.lower()


def _extract_update_value(obj: Any) -> str:
    if isinstance(obj, dict):
        value = obj.get("value")
        return str(value) if value else ""
    return ""


def _serialize_df(df: Any) -> List[Dict[str, Any]]:
    if isinstance(df, pd.DataFrame):
        if df.empty:
            return []
        return df.fillna("").to_dict(orient="records")
    return []


def _new_upload_path(original_name: str) -> tuple[str, Path]:
    run_id = build_run_id_utc()
    upload_dir = get_web_v2_area_dir("uploads", tool="advanced_tools", run_id=run_id)
    return run_id, upload_dir / make_web_v2_upload_name(1, original_name)


def _stage_download_result(path_str: str, kind: str) -> str:
    if not path_str:
        return ""
    try:
        source = resolve_web_v2_client_path(path_str, allowed_areas=("results", "work", "uploads", "sessions", "manifests"))
    except ValueError:
        source = Path(path_str).expanduser().resolve()
    if not source.exists() or not source.is_file():
        return path_str
    if not ensure_within_roots(source, _STAGE_ALLOWED_SOURCE_ROOTS):
        return ""
    run_id = build_run_id_utc()
    result_dir = get_web_v2_area_dir("results", tool="advanced_tools", run_id=run_id)
    staged = result_dir / make_web_v2_result_name(kind, 1, source.suffix.lower())
    staged.write_bytes(source.read_bytes())
    create_run_manifest(
        run_id=run_id,
        tool="advanced_tools",
        status="completed",
        outputs=[{"path": str(staged.relative_to(_WEB_V2_RESULTS_ROOT)), "size": staged.stat().st_size}],
    )
    return to_web_v2_public_path(staged)


def _safe_download_url(path_str: str) -> str:
    return build_web_v2_download_url(path_str) if path_str else ""


def _stage_directed_evolution_heatmap(download_path: str) -> str:
    if not download_path:
        return ""
    try:
        archive = resolve_web_v2_client_path(download_path, allowed_areas=("results",))
    except ValueError:
        return ""
    if not archive.exists() or not archive.is_file():
        return ""
    if ".tar" not in archive.name.lower() and not archive.name.lower().endswith(".gz"):
        return ""

    heatmap_html = b""
    try:
        with tarfile.open(archive, "r:*") as tar:
            chosen_member = None
            fallback_mut_map_member = None
            fallback_heatmap_member = None
            fallback_html_member = None
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                base_name = Path(member.name).name.lower()
                if base_name == "prediction_heatmap.html":
                    chosen_member = member
                    break
                if not base_name.endswith((".html", ".htm")):
                    continue
                if fallback_mut_map_member is None and base_name.startswith("mut_map"):
                    fallback_mut_map_member = member
                if fallback_heatmap_member is None and "heatmap" in base_name:
                    fallback_heatmap_member = member
                if fallback_html_member is None:
                    fallback_html_member = member
            if chosen_member is None:
                chosen_member = fallback_mut_map_member or fallback_heatmap_member or fallback_html_member
            if chosen_member is None:
                # No html plot found in archive; keep API compatible by returning empty path.
                return ""
            extracted = tar.extractfile(chosen_member)
            if extracted is None:
                return ""
            heatmap_html = extracted.read()
    except (tarfile.TarError, OSError):
        return ""

    if not heatmap_html:
        return ""

    run_id = build_run_id_utc()
    result_dir = get_web_v2_area_dir("results", tool="advanced_tools", run_id=run_id)
    staged = result_dir / make_web_v2_result_name("directed_evolution_heatmap", 1, ".html")
    staged.write_bytes(heatmap_html)
    create_run_manifest(
        run_id=run_id,
        tool="advanced_tools",
        status="completed",
        outputs=[{"path": str(staged.relative_to(_WEB_V2_RESULTS_ROOT)), "size": staged.stat().st_size}],
    )
    return to_web_v2_public_path(staged)


def _ensure_fasta_path(file_path: Optional[str], sequence: Optional[str]) -> str:
    def _prepare_text_as_fasta(raw_text: str) -> str:
        text = (raw_text or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="Please provide FASTA file or sequence.")
        if text.startswith(">"):
            try:
                return validate_and_normalize_fasta_content(text)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        seq = "".join(ch for ch in text.upper() if ch.isalpha())
        if not seq:
            raise HTTPException(status_code=400, detail="Sequence is empty after normalization.")
        return f">input\n{seq}\n"

    if file_path:
        try:
            path = resolve_web_v2_client_path(file_path, allowed_areas=("uploads",))
        except ValueError as exc:
            raise HTTPException(status_code=403, detail="Access denied.") from exc
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Input file not found: {file_path}")
        suffix = path.suffix.lower()
        if suffix in {".fasta", ".fa", ".txt"}:
            try:
                source_text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                raise HTTPException(status_code=400, detail="FASTA/TXT file must be UTF-8 encoded.") from exc
            normalized = _prepare_text_as_fasta(source_text)
            run_id = build_run_id_utc()
            out_dir = get_web_v2_area_dir("uploads", tool="advanced_tools", run_id=run_id)
            out = out_dir / make_web_v2_upload_name(1, "normalized_input.fasta")
            out.write_text(normalized, encoding="utf-8")
            return str(out)
        return str(path)
    seq = (sequence or "").strip()
    normalized = _prepare_text_as_fasta(seq)
    run_id = build_run_id_utc()
    out_dir = get_web_v2_area_dir("uploads", tool="advanced_tools", run_id=run_id)
    out = out_dir / make_web_v2_upload_name(1, "inline_sequence.fasta")
    out.write_text(normalized, encoding="utf-8")
    return str(out)


def _resolve_upload_file(path_value: str) -> str:
    try:
        resolved = resolve_web_v2_client_path(path_value, allowed_areas=("uploads",))
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Access denied.") from exc
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail=f"Input file not found: {path_value}")
    return str(resolved)


def _run_generator_collect_last(gen: Any) -> Any:
    last = None
    for item in gen:
        last = item
    return last


def _merge_sequence_row(
    row: dict[str, Any],
    sequence_index: int,
    sequence_header: str,
) -> dict[str, Any]:
    merged: dict[str, Any] = {"sequence_index": sequence_index, **row}
    canonical = (
        str(merged.get("Protein Name", "") or "").strip()
        or str(merged.get("protein_name", "") or "").strip()
        or str(merged.get("header", "") or "").strip()
        or str(merged.get("sequence_header", "") or "").strip()
        or sequence_header.strip()
    )

    # Use a single canonical naming column in merged table output.
    merged["Protein Name"] = canonical
    merged.pop("protein_name", None)
    merged.pop("header", None)
    merged.pop("sequence_header", None)

    return merged


@router.get("/meta")
async def advanced_tools_meta():
    if not _CONSTANT_PATH.exists():
        raise HTTPException(status_code=404, detail="constant.json not found.")
    data = json.loads(_CONSTANT_PATH.read_text(encoding="utf-8"))
    web_ui = data.get("web_ui", {})

    return {
        "dataset_mapping_zero_shot": web_ui.get("dataset_mapping_zero_shot", []),
        "sequence_model_options": ["VenusPLM", "ESM2-650M", "ESM-1v", "ESM-1b"],
        "structure_model_options": ["VenusREM (foldseek-based)", "ProSST-2048", "ProtSSN", "ESM-IF1", "SaProt", "MIF-ST"],
        "model_mapping_function": list(web_ui.get("model_mapping_function", {}).keys()),
        "residue_model_mapping_function": list(web_ui.get("model_residue_mapping_function", {}).keys()),
        "dataset_mapping_function": web_ui.get("dataset_mapping_function", {}),
        "residue_mapping_function": web_ui.get("residue_mapping_function", {}),
        "llm_models": list(LLM_MODELS.keys()),
        "mode": _runtime_mode(),
        "online_fasta_limit": _ONLINE_FASTA_LIMIT,
        "online_limit_enabled": _online_fasta_limit_enabled(),
    }


@router.post("/upload")
async def upload_advanced_file(file: UploadFile = File(...)):
    filename = os.path.basename(file.filename or f"advanced-tools-{uuid.uuid4().hex}.txt")
    suffix = Path(filename).suffix.lower()
    if suffix not in {".fasta", ".fa", ".pdb", ".txt"}:
        raise HTTPException(status_code=400, detail="Only .fasta/.fa/.pdb/.txt files are supported.")
    run_id, dst = _new_upload_path(filename)
    content = await file.read()
    with open(dst, "wb") as out:
        out.write(content)
    create_run_manifest(
        run_id=run_id,
        tool="advanced_tools",
        status="uploaded",
        inputs=[{"path": str(dst), "name": filename, "size": len(content)}],
    )
    return {"file_path": to_web_v2_public_path(dst), "name": filename, "suffix": suffix, "run_id": run_id}


@router.get("/default-example")
async def advanced_default_example(kind: str = "fasta"):
    normalized_kind = (kind or "fasta").strip().lower()
    source = _DEFAULT_PDB_EXAMPLE if normalized_kind == "pdb" else _DEFAULT_FASTA_EXAMPLE
    if not source.exists():
        raise HTTPException(status_code=404, detail=f"Example file not found for kind={normalized_kind}")

    run_id, dst = _new_upload_path(source.name)
    dst.write_bytes(source.read_bytes())
    create_run_manifest(
        run_id=run_id,
        tool="advanced_tools",
        status="uploaded",
        inputs=[{"path": str(dst), "name": source.name, "size": source.stat().st_size}],
    )
    suffix = source.suffix.lower()
    content = source.read_text(encoding="utf-8") if suffix in {".fasta", ".fa"} else ""
    return {
        "file_path": to_web_v2_public_path(dst),
        "name": source.name,
        "suffix": suffix,
        "kind": normalized_kind,
        "content": content,
        "run_id": run_id,
    }


@router.get("/download")
async def download_advanced_result(file_path: str, inline: bool = False):
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


@router.post("/ai-summary")
async def advanced_ai_summary(body: AdvancedAiSummaryBody):
    api_key = get_api_key(body.llm_provider, body.user_api_key)
    if not api_key:
        raise HTTPException(status_code=400, detail="No API key available. Set provider key or provide user key.")
    if body.llm_provider not in LLM_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported LLM provider: {body.llm_provider}")

    model = LLM_MODELS[body.llm_provider]
    prompt = (
        "You are an expert protein scientist.\n"
        f"Tool: {body.tool}\n"
        f"Task: {body.task or 'N/A'}\n"
        "Analyze the following merged prediction output and provide:\n"
        "1) Key findings, 2) Confidence interpretation, 3) Practical wet-lab next steps.\n"
        "Keep it concise, actionable, and under 220 words.\n\n"
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


@router.post("/directed-evolution/run")
async def run_directed_evolution(body: DirectedEvolutionBody):
    if body.input_mode not in {"sequence", "structure"}:
        raise HTTPException(status_code=400, detail="input_mode must be 'sequence' or 'structure'.")

    if body.input_mode == "structure":
        if not body.file_path:
            raise HTTPException(status_code=400, detail="Structure mode requires PDB file upload.")
        file_path = _resolve_upload_file(body.file_path)
        if not file_path.lower().endswith(".pdb"):
            raise HTTPException(status_code=400, detail="Structure mode only supports .pdb input.")
    else:
        file_path = _ensure_fasta_path(body.file_path, body.sequence)
        normalized = _load_normalized_fasta_from_path(file_path)
        record_count = _count_fasta_records(normalized)
        if record_count != 1:
            raise HTTPException(
                status_code=400,
                detail="Directed Evolution supports exactly one FASTA sequence per run.",
            )

    last = _run_generator_collect_last(
        handle_mutation_prediction_advance(
            function_selection=body.function_selection or "Model-guided optimization",
            file_obj=file_path,
            enable_ai=body.enable_ai,
            llm_model=body.llm_provider,
            user_api_key=body.user_api_key,
            model_name=body.model_name,
        )
    )
    if not last:
        raise HTTPException(status_code=500, detail="No response from directed evolution handler.")

    download_path = _extract_update_value(last[3]) or (str(last[4]) if last[4] else "")
    download_path = _stage_download_result(download_path, "directed_evolution")
    heatmap_path = _stage_directed_evolution_heatmap(download_path)
    return {
        "status": str(last[0]),
        "table": _serialize_df(last[2]),
        "download_path": download_path,
        "download_url": build_web_v2_download_url(download_path) if download_path else "",
        "heatmap_path": heatmap_path,
        "heatmap_url": _safe_download_url(heatmap_path),
        "ai_summary": str(last[7]) if len(last) > 7 and last[7] else "",
    }


@router.post("/protein-function/run")
async def run_protein_function(body: ProteinFunctionBody):
    fasta_file = _ensure_fasta_path(body.file_path, body.sequence)
    if _online_fasta_limit_enabled():
        normalized = _load_normalized_fasta_from_path(fasta_file)
        _enforce_online_fasta_limit(normalized, source="Protein Function FASTA")
    meta = await advanced_tools_meta()
    datasets = body.datasets or meta.get("dataset_mapping_function", {}).get(body.task, [])
    if not datasets:
        raise HTTPException(status_code=400, detail=f"No dataset mapping found for task: {body.task}")

    last = _run_generator_collect_last(
        handle_protein_function_prediction_advance(
            task=body.task,
            fasta_file=fasta_file,
            enable_ai=body.enable_ai,
            llm_model=body.llm_provider,
            user_api_key=body.user_api_key,
            model_name=body.model_name,
            datasets=datasets,
        )
    )
    if not last:
        raise HTTPException(status_code=500, detail="No response from protein function handler.")

    download_path = _extract_update_value(last[3])
    download_path = _stage_download_result(download_path, "protein_function")
    return {
        "status": str(last[0]),
        "table": _serialize_df(last[1]),
        "download_path": download_path,
        "download_url": build_web_v2_download_url(download_path) if download_path else "",
        "ai_summary": str(last[4]) if len(last) > 4 and last[4] else "",
    }


@router.post("/functional-residue/run")
async def run_functional_residue(body: FunctionalResidueBody):
    fasta_file = _ensure_fasta_path(body.file_path, body.sequence)
    if _online_fasta_limit_enabled():
        normalized = _load_normalized_fasta_from_path(fasta_file)
        _enforce_online_fasta_limit(normalized, source="Functional Residue FASTA")

    last = _run_generator_collect_last(
        handle_protein_residue_function_prediction(
            task=body.task,
            fasta_file=fasta_file,
            enable_ai=body.enable_ai,
            llm_model=body.llm_provider,
            user_api_key=body.user_api_key,
            model_name=body.model_name,
        )
    )
    if not last:
        raise HTTPException(status_code=500, detail="No response from residue function handler.")

    download_path = _extract_update_value(last[3])
    download_path = _stage_download_result(download_path, "functional_residue")
    ai_summary = ""
    if len(last) > 5 and last[5]:
        ai_summary = str(last[5])
    elif len(last) > 4 and last[4]:
        ai_summary = str(last[4])
    return {
        "status": str(last[0]),
        "table": _serialize_df(last[1]),
        "download_path": download_path,
        "download_url": build_web_v2_download_url(download_path) if download_path else "",
        "ai_summary": ai_summary,
    }


@router.post("/protein-discovery/run")
async def run_protein_discovery(body: ProteinDiscoveryBody):
    if not body.pdb_file:
        raise HTTPException(status_code=400, detail="PDB file is required.")
    safe_pdb_file = _resolve_upload_file(body.pdb_file)

    last = _run_generator_collect_last(
        handle_VenusMine(
            pdb_file=safe_pdb_file,
            protect_start=body.protect_start,
            protect_end=body.protect_end,
            mmseqs_threads=body.mmseqs_threads,
            mmseqs_iterations=body.mmseqs_iterations,
            mmseqs_max_seqs=body.mmseqs_max_seqs,
            cluster_min_seq_id=body.cluster_min_seq_id,
            cluster_threads=body.cluster_threads,
            top_n_threshold=body.top_n_threshold,
            evalue_threshold=body.evalue_threshold,
        )
    )
    if not last:
        raise HTTPException(status_code=500, detail="No response from protein discovery handler.")

    tree_download = _extract_update_value(last[3]) if len(last) > 3 else ""
    labels_download = _extract_update_value(last[4]) if len(last) > 4 else ""
    zip_download = _extract_update_value(last[5]) if len(last) > 5 else ""
    tree_download = _stage_download_result(tree_download, "discovery_tree")
    labels_download = _stage_download_result(labels_download, "discovery_labels")
    zip_download = _stage_download_result(zip_download, "discovery_archive")
    tree_image = _stage_download_result(str(last[1]) if len(last) > 1 and last[1] else "", "discovery_tree_image")
    final_download = zip_download or tree_download or labels_download
    return {
        "status": str(last[7]) if len(last) > 7 else "Completed",
        "log": str(last[0]) if len(last) > 0 else "",
        "tree_image": tree_image,
        "tree_image_url": _safe_download_url(tree_image),
        "table": _serialize_df(last[2]) if len(last) > 2 else [],
        "download_path": final_download,
        "download_url": _safe_download_url(final_download),
        "download_tree": tree_download,
        "download_labels": labels_download,
        "download_archive": zip_download,
        "download_tree_url": _safe_download_url(tree_download),
        "download_labels_url": _safe_download_url(labels_download),
        "download_archive_url": _safe_download_url(zip_download),
    }


@router.post("/directed-evolution/run/stream")
async def run_directed_evolution_stream(body: DirectedEvolutionBody):
    async def event_stream():
        start = time.perf_counter()
        yield _sse("progress", {"progress": 0.08, "message": "Validating Directed Evolution input..."})
        try:
            yield _sse("progress", {"progress": 0.4, "message": "Running Directed Evolution model..."})
            payload = await run_directed_evolution(body)
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


@router.post("/protein-function/run/stream")
async def run_protein_function_stream(body: ProteinFunctionBody):
    async def event_stream():
        start = time.perf_counter()
        yield _sse("progress", {"progress": 0.06, "message": "Preparing FASTA input..."})
        try:
            fasta_file = _ensure_fasta_path(body.file_path, body.sequence)
            normalized = _load_normalized_fasta_from_path(fasta_file)
            _enforce_online_fasta_limit(normalized, source="Protein Function FASTA")
            records = _parse_normalized_fasta_records(normalized)
            if not records:
                raise HTTPException(status_code=400, detail="No valid FASTA sequence found.")

            total = len(records)
            yield _sse("progress", {"progress": 0.15, "message": f"Loaded {total} protein sequence(s)."})

            merged_rows: list[dict[str, Any]] = []
            failures: list[str] = []
            download_path = ""
            for idx, (header, sequence) in enumerate(records, start=1):
                yield _sse("progress", {"progress": 0.15 + 0.7 * (idx - 1) / total, "message": f"Running sequence {idx}/{total}..."})
                try:
                    item_payload = await run_protein_function(
                        ProteinFunctionBody(
                            task=body.task,
                            sequence=f">{header}\n{sequence}\n",
                            model_name=body.model_name,
                            datasets=body.datasets,
                            enable_ai=False,
                            llm_provider=body.llm_provider,
                            user_api_key=body.user_api_key,
                        )
                    )
                    table = item_payload.get("table")
                    if isinstance(table, list):
                        for row in table:
                            if isinstance(row, dict):
                                merged_rows.append(_merge_sequence_row(row, idx, header))
                    path_candidate = item_payload.get("download_path")
                    if isinstance(path_candidate, str) and path_candidate:
                        download_path = path_candidate
                except HTTPException as exc:
                    failures.append(f"[{idx}] {header}: {exc.detail}")
                except Exception as exc:
                    failures.append(f"[{idx}] {header}: {exc}")
                yield _sse("progress", {"progress": 0.15 + 0.7 * idx / total, "message": f"Completed sequence {idx}/{total}."})

            success_count = total - len(failures)
            if success_count <= 0:
                raise HTTPException(status_code=400, detail=failures[0] if failures else "All sequences failed.")

            merged_payload: Dict[str, Any] = {
                "status": f"Completed {success_count}/{total} sequences." + (f" {len(failures)} sequence(s) failed." if failures else ""),
                "table": merged_rows,
                "download_path": download_path,
                "download_url": build_web_v2_download_url(download_path) if download_path else "",
            }
            if failures:
                merged_payload["warnings"] = failures

            if body.enable_ai:
                yield _sse("progress", {"progress": 0.9, "message": "Generating AI summary..."})
                ai = await advanced_ai_summary(
                    AdvancedAiSummaryBody(
                        tool="protein-function",
                        task=body.task,
                        llm_provider=body.llm_provider,
                        user_api_key=body.user_api_key,
                        result_payload=merged_payload,
                    )
                )
                merged_payload["ai_summary"] = ai.get("summary", "")

            elapsed_ms = int((time.perf_counter() - start) * 1000)
            for chunk in _stream_success(merged_payload, final_message=f"Protein Function completed in {elapsed_ms} ms."):
                yield chunk
        except HTTPException as exc:
            message = str(exc.detail) if exc.detail else "Protein Function failed."
            for chunk in _stream_error(message, status_code=exc.status_code):
                yield chunk
        except Exception as exc:
            for chunk in _stream_error(f"Protein Function failed: {exc}", status_code=500):
                yield chunk

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/functional-residue/run/stream")
async def run_functional_residue_stream(body: FunctionalResidueBody):
    async def event_stream():
        start = time.perf_counter()
        yield _sse("progress", {"progress": 0.06, "message": "Preparing FASTA input..."})
        try:
            fasta_file = _ensure_fasta_path(body.file_path, body.sequence)
            normalized = _load_normalized_fasta_from_path(fasta_file)
            _enforce_online_fasta_limit(normalized, source="Functional Residue FASTA")
            records = _parse_normalized_fasta_records(normalized)
            if not records:
                raise HTTPException(status_code=400, detail="No valid FASTA sequence found.")

            total = len(records)
            yield _sse("progress", {"progress": 0.15, "message": f"Loaded {total} protein sequence(s)."})

            merged_rows: list[dict[str, Any]] = []
            failures: list[str] = []
            download_path = ""
            for idx, (header, sequence) in enumerate(records, start=1):
                yield _sse("progress", {"progress": 0.15 + 0.7 * (idx - 1) / total, "message": f"Running sequence {idx}/{total}..."})
                try:
                    item_payload = await run_functional_residue(
                        FunctionalResidueBody(
                            task=body.task,
                            sequence=f">{header}\n{sequence}\n",
                            model_name=body.model_name,
                            enable_ai=False,
                            llm_provider=body.llm_provider,
                            user_api_key=body.user_api_key,
                        )
                    )
                    table = item_payload.get("table")
                    if isinstance(table, list):
                        for row in table:
                            if isinstance(row, dict):
                                merged_rows.append(_merge_sequence_row(row, idx, header))
                    path_candidate = item_payload.get("download_path")
                    if isinstance(path_candidate, str) and path_candidate:
                        download_path = path_candidate
                except HTTPException as exc:
                    failures.append(f"[{idx}] {header}: {exc.detail}")
                except Exception as exc:
                    failures.append(f"[{idx}] {header}: {exc}")
                yield _sse("progress", {"progress": 0.15 + 0.7 * idx / total, "message": f"Completed sequence {idx}/{total}."})

            success_count = total - len(failures)
            if success_count <= 0:
                raise HTTPException(status_code=400, detail=failures[0] if failures else "All sequences failed.")

            merged_payload: Dict[str, Any] = {
                "status": f"Completed {success_count}/{total} sequences." + (f" {len(failures)} sequence(s) failed." if failures else ""),
                "table": merged_rows,
                "download_path": download_path,
                "download_url": build_web_v2_download_url(download_path) if download_path else "",
            }
            if failures:
                merged_payload["warnings"] = failures

            if body.enable_ai:
                yield _sse("progress", {"progress": 0.9, "message": "Generating AI summary..."})
                ai = await advanced_ai_summary(
                    AdvancedAiSummaryBody(
                        tool="functional-residue",
                        task=body.task,
                        llm_provider=body.llm_provider,
                        user_api_key=body.user_api_key,
                        result_payload=merged_payload,
                    )
                )
                merged_payload["ai_summary"] = ai.get("summary", "")

            elapsed_ms = int((time.perf_counter() - start) * 1000)
            for chunk in _stream_success(merged_payload, final_message=f"Functional Residue completed in {elapsed_ms} ms."):
                yield chunk
        except HTTPException as exc:
            message = str(exc.detail) if exc.detail else "Functional Residue failed."
            for chunk in _stream_error(message, status_code=exc.status_code):
                yield chunk
        except Exception as exc:
            for chunk in _stream_error(f"Functional Residue failed: {exc}", status_code=500):
                yield chunk

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/protein-discovery/run/stream")
async def run_protein_discovery_stream(body: ProteinDiscoveryBody):
    async def event_stream():
        start = time.perf_counter()
        yield _sse("progress", {"progress": 0.08, "message": "Validating Protein Discovery input..."})
        try:
            yield _sse("progress", {"progress": 0.4, "message": "Running VenusMine pipeline..."})
            payload = await run_protein_discovery(body)
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            for chunk in _stream_success(payload, final_message=f"Protein Discovery completed in {elapsed_ms} ms."):
                yield chunk
        except HTTPException as exc:
            message = str(exc.detail) if exc.detail else "Protein Discovery failed."
            for chunk in _stream_error(message, status_code=exc.status_code):
                yield chunk
        except Exception as exc:
            for chunk in _stream_error(f"Protein Discovery failed: {exc}", status_code=500):
                yield chunk

    return StreamingResponse(event_stream(), media_type="text/event-stream")
