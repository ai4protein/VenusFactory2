import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from web.comprehensive_tab import (
    handle_individual_mutation_prediction,
    handle_individual_function_prediction,
    handle_functional_residue_prediction,
    handle_physical_chemical_properties,
    generate_expert_analysis_report,
    export_ai_report_to_html,
    export_ai_report_to_pdf,
)
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
from web.utils.file_handlers import extract_sequence_from_pdb


router = APIRouter(prefix="/api/v2/report", tags=["report-v2"])

_REPORT_UPLOAD_DIR = get_web_v2_area_dir("uploads", tool="report")
_REPORT_RESULTS_ROOT = get_web_v2_area_dir("results")
_REPORT_DEFAULT_EXAMPLE = Path("example/database/P60002.fasta")


class ParseInputRequest(BaseModel):
    content: str = ""
    file_path: str = ""


class ParseInputResponse(BaseModel):
    sequence_map: Dict[str, str]
    selected_chain: str
    preview: str
    current_file: str = ""
    original_content: str = ""


class GenerateReportRequest(BaseModel):
    sequence_map: Dict[str, str] = Field(default_factory=dict)
    selected_chain: str = "Sequence 1"
    current_file: str = ""
    original_content: str = ""
    selected_analyses: List[str] = Field(default_factory=list)


def _secure_under(path: str, base: Path) -> bool:
    try:
        Path(path).resolve().relative_to(base.resolve())
        return True
    except Exception:
        return False


def _parse_fasta(content: str) -> Dict[str, str]:
    sequences: Dict[str, str] = {}
    current_id: Optional[str] = None
    current_seq: List[str] = []
    for line in content.splitlines():
        ln = line.strip()
        if not ln:
            continue
        if ln.startswith(">"):
            if current_id and current_seq:
                sequences[current_id] = "".join(current_seq).upper()
            current_id = ln[1:].strip() or f"Sequence {len(sequences)+1}"
            current_seq = []
        else:
            current_seq.append("".join(c for c in ln if c.isalpha()))
    if current_id and current_seq:
        sequences[current_id] = "".join(current_seq).upper()
    return {k: v for k, v in sequences.items() if v}


def _parse_input_content(content: str, file_path: str = "") -> ParseInputResponse:
    raw = content.strip()
    if not raw:
        raise HTTPException(status_code=400, detail="No valid input content found.")

    sequence_map: Dict[str, str] = {}
    selected_chain = "Sequence 1"
    preview = ""

    if raw.startswith("ATOM") or (file_path and file_path.endswith(".pdb")):
        seq = extract_sequence_from_pdb(raw)
        if not seq:
            raise HTTPException(status_code=400, detail="Could not extract sequence from PDB input.")
        selected_chain = "A"
        sequence_map = {selected_chain: seq}
        preview = seq[:300]
    elif raw.startswith(">"):
        sequence_map = _parse_fasta(raw)
        if not sequence_map:
            raise HTTPException(status_code=400, detail="No valid FASTA sequence found.")
        selected_chain = next(iter(sequence_map.keys()))
        preview = sequence_map[selected_chain][:300]
    else:
        seq = "".join(c for c in raw if c.isalpha()).upper()
        if not seq:
            raise HTTPException(status_code=400, detail="No valid amino-acid sequence found.")
        sequence_map = {"Sequence 1": seq}
        selected_chain = "Sequence 1"
        preview = seq[:300]

    return ParseInputResponse(
        sequence_map=sequence_map,
        selected_chain=selected_chain,
        preview=preview,
        current_file=file_path or "",
        original_content=raw,
    )


@router.post("/upload")
async def upload_report_file(file: UploadFile = File(...)):
    filename = os.path.basename(file.filename or f"report-{uuid.uuid4().hex}.txt")
    suffix = Path(filename).suffix.lower()
    if suffix not in {".fasta", ".fa", ".pdb"}:
        raise HTTPException(status_code=400, detail="Only .fasta/.fa/.pdb files are supported.")
    run_id = build_run_id_utc()
    upload_dir = get_web_v2_area_dir("uploads", tool="report", run_id=run_id)
    dst = upload_dir / make_web_v2_upload_name(1, filename)
    content = await file.read()
    with open(dst, "wb") as out:
        out.write(content)
    text = dst.read_text(encoding="utf-8", errors="ignore")
    parsed = _parse_input_content(text, to_web_v2_public_path(dst))
    create_run_manifest(
        run_id=run_id,
        tool="report",
        status="uploaded",
        inputs=[{"path": str(dst), "name": filename, "size": len(content)}],
    )
    return {"file_path": to_web_v2_public_path(dst), "parse": parsed.model_dump(), "run_id": run_id}


@router.post("/parse-input", response_model=ParseInputResponse)
async def parse_input(payload: ParseInputRequest):
    content = payload.content or ""
    file_path = payload.file_path or ""
    if file_path:
        try:
            fp = resolve_web_v2_client_path(file_path, allowed_areas=("uploads",))
        except ValueError:
            raise HTTPException(status_code=403, detail="Access denied.")
        if not fp.exists():
            raise HTTPException(status_code=404, detail="File not found for parsing.")
        content = fp.read_text(encoding="utf-8", errors="ignore")
    return _parse_input_content(content, to_web_v2_public_path(fp) if file_path else "")


@router.get("/default-input")
async def get_default_report_input():
    if not _REPORT_DEFAULT_EXAMPLE.exists():
        raise HTTPException(status_code=404, detail="Default example file not found.")
    content = _REPORT_DEFAULT_EXAMPLE.read_text(encoding="utf-8", errors="ignore")
    parsed = _parse_input_content(content, "")
    return {
        "name": _REPORT_DEFAULT_EXAMPLE.name,
        "content": content,
        "parse": parsed.model_dump(),
    }


@router.post("/generate")
async def generate_report(payload: GenerateReportRequest):
    sequence_map = payload.sequence_map or {}
    selected_chain = payload.selected_chain
    current_file = payload.current_file or ""
    resolved_current_file = ""
    if current_file:
        try:
            resolved_current_file = str(resolve_web_v2_client_path(current_file, allowed_areas=("uploads",)))
        except ValueError as exc:
            raise HTTPException(status_code=403, detail="Access denied.") from exc

    original_content = payload.original_content or ""
    selected_analyses = payload.selected_analyses or []

    if not sequence_map or selected_chain not in sequence_map:
        raise HTTPException(status_code=400, detail="No valid sequence selected for analysis.")
    if not selected_analyses:
        raise HTTPException(status_code=400, detail="Please select at least one analysis type.")

    sequence = sequence_map[selected_chain]
    original_raw = original_content.strip()
    is_pdb_input = bool(resolved_current_file and resolved_current_file.endswith(".pdb")) or original_raw.startswith("ATOM")
    # For FASTA/raw input without uploaded file, downstream handlers expect pure sequence.
    # Passing full FASTA text here can duplicate headers and break mutation parsing.
    content_for_analysis = original_raw if is_pdb_input else sequence

    sections: List[str] = []
    sections.append("# PROTEIN ANALYSIS REPORT")
    sections.append("")
    sections.append(f"**Sequence ID:** {selected_chain}")
    sections.append(f"**Sequence Length:** {len(sequence)} amino acid residues")
    sections.append(f"**Analysis Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    sections.append("")
    sections.append("---")
    sections.append("")

    if "mutation" in selected_analyses:
        sections.append("## 🧬 Mutation Prediction Analysis")
        sections.append("")
        try:
            mutation_result = handle_individual_mutation_prediction(
                content_for_analysis, selected_chain, resolved_current_file, sequence_map, original_content
            )
            sections.append(mutation_result)
            sections.append("")
        except Exception as exc:
            sections.append(f"❌ Mutation analysis failed: {exc}")
            sections.append("")

    if "function" in selected_analyses:
        sections.append("## 🔬 Protein Function Analysis")
        sections.append("")
        try:
            function_result = handle_individual_function_prediction(
                content_for_analysis, selected_chain, resolved_current_file, sequence_map, original_content
            )
            sections.append(function_result)
            sections.append("")
        except Exception as exc:
            sections.append(f"❌ Function analysis failed: {exc}")
            sections.append("")

    if "residue" in selected_analyses:
        sections.append("## 🎯 Functional Residue")
        sections.append("")
        try:
            residue_result = handle_functional_residue_prediction(
                content_for_analysis, selected_chain, resolved_current_file, sequence_map, original_content
            )
            sections.append(residue_result)
            sections.append("")
        except Exception as exc:
            sections.append(f"❌ Functional residue analysis failed: {exc}")
            sections.append("")

    if "properties" in selected_analyses:
        sections.append("## ⚗️ Physical & Chemical Properties")
        sections.append("")
        try:
            properties_result = handle_physical_chemical_properties(
                content_for_analysis, selected_chain, resolved_current_file, sequence_map, original_content
            )
            sections.append(properties_result)
            sections.append("")
        except Exception as exc:
            sections.append(f"❌ Property analysis failed: {exc}")
            sections.append("")

    sections.append("---")
    sections.append("")
    sections.append("## 💡 Conclusion")
    sections.append("")
    sections.append("✅ Analysis completed. Please validate critical findings experimentally.")
    sections.append("")
    sections.append(f"*Report generated by VenusFactory2 v2 at {time.strftime('%Y-%m-%d %H:%M:%S')}*")

    report_text = "\n".join(sections)
    ai_report = generate_expert_analysis_report(report_text)

    html_path = export_ai_report_to_html(ai_report)
    pdf_path = export_ai_report_to_pdf(ai_report)
    run_id = build_run_id_utc()
    result_dir = get_web_v2_area_dir("results", tool="report", run_id=run_id)
    html_stage = ""
    pdf_stage = ""
    if html_path and Path(html_path).exists():
        html_dst = result_dir / make_web_v2_result_name("report_html", 1, ".html")
        html_dst.write_bytes(Path(html_path).read_bytes())
        html_stage = str(html_dst.relative_to(_REPORT_RESULTS_ROOT))
    if pdf_path and Path(pdf_path).exists():
        pdf_dst = result_dir / make_web_v2_result_name("report_pdf", 1, ".pdf")
        pdf_dst.write_bytes(Path(pdf_path).read_bytes())
        pdf_stage = str(pdf_dst.relative_to(_REPORT_RESULTS_ROOT))
    create_run_manifest(
        run_id=run_id,
        tool="report",
        status="completed",
        outputs=[{"path": html_stage}, {"path": pdf_stage}],
    )

    return {
        "report_text": report_text,
        "ai_report": ai_report,
        "html_download_url": build_web_v2_download_url(html_stage) if html_stage else "",
        "pdf_download_url": build_web_v2_download_url(pdf_stage) if pdf_stage else "",
        "html_path": html_stage,
        "pdf_path": pdf_stage,
        "run_id": run_id,
    }


@router.post("/generate/stream")
async def generate_report_stream(payload: GenerateReportRequest):
    sequence_map = payload.sequence_map or {}
    selected_chain = payload.selected_chain
    current_file = payload.current_file or ""
    resolved_current_file = ""
    if current_file:
        try:
            resolved_current_file = str(resolve_web_v2_client_path(current_file, allowed_areas=("uploads",)))
        except ValueError as exc:
            raise HTTPException(status_code=403, detail="Access denied.") from exc
    original_content = payload.original_content or ""
    selected_analyses = payload.selected_analyses or []

    if not sequence_map or selected_chain not in sequence_map:
        raise HTTPException(status_code=400, detail="No valid sequence selected for analysis.")
    if not selected_analyses:
        raise HTTPException(status_code=400, detail="Please select at least one analysis type.")

    def sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    async def event_stream():
        try:
            sequence = sequence_map[selected_chain]
            original_raw = original_content.strip()
            is_pdb_input = bool(resolved_current_file and resolved_current_file.endswith(".pdb")) or original_raw.startswith("ATOM")
            # Keep raw PDB content; use pure sequence for FASTA/raw to avoid malformed double-header FASTA.
            content_for_analysis = original_raw if is_pdb_input else sequence
            sections: List[str] = []
            sections.append("# PROTEIN ANALYSIS REPORT")
            sections.append("")
            sections.append(f"**Sequence ID:** {selected_chain}")
            sections.append(f"**Sequence Length:** {len(sequence)} amino acid residues")
            sections.append(f"**Analysis Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
            sections.append("")
            sections.append("---")
            sections.append("")

            total_steps = len(selected_analyses) + 2  # analyses + ai + export
            done_steps = 0
            yield sse("progress", {"progress": 0.02, "message": "Initializing report task"})
            yield sse("log", {"line": f"Input ready. Selected analyses: {', '.join(selected_analyses)}"})

            if "mutation" in selected_analyses:
                yield sse("log", {"line": "Running mutation prediction analysis..."})
                mutation_result = await asyncio.to_thread(
                    handle_individual_mutation_prediction,
                    content_for_analysis,
                    selected_chain,
                    resolved_current_file,
                    sequence_map,
                    original_content,
                )
                sections.append("## 🧬 Mutation Prediction Analysis")
                sections.append("")
                sections.append(mutation_result)
                sections.append("")
                done_steps += 1
                yield sse(
                    "progress",
                    {
                        "progress": round(0.05 + 0.7 * done_steps / total_steps, 3),
                        "message": "Mutation prediction completed",
                    },
                )
                yield sse("log", {"line": "Mutation prediction completed."})

            if "function" in selected_analyses:
                yield sse("log", {"line": "Running protein function analysis..."})
                function_result = await asyncio.to_thread(
                    handle_individual_function_prediction,
                    content_for_analysis,
                    selected_chain,
                    resolved_current_file,
                    sequence_map,
                    original_content,
                )
                sections.append("## 🔬 Protein Function Analysis")
                sections.append("")
                sections.append(function_result)
                sections.append("")
                done_steps += 1
                yield sse(
                    "progress",
                    {
                        "progress": round(0.05 + 0.7 * done_steps / total_steps, 3),
                        "message": "Protein function analysis completed",
                    },
                )
                yield sse("log", {"line": "Protein function analysis completed."})

            if "residue" in selected_analyses:
                yield sse("log", {"line": "Running functional residue analysis..."})
                residue_result = await asyncio.to_thread(
                    handle_functional_residue_prediction,
                    content_for_analysis,
                    selected_chain,
                    resolved_current_file,
                    sequence_map,
                    original_content,
                )
                sections.append("## 🎯 Functional Residue")
                sections.append("")
                sections.append(residue_result)
                sections.append("")
                done_steps += 1
                yield sse(
                    "progress",
                    {
                        "progress": round(0.05 + 0.7 * done_steps / total_steps, 3),
                        "message": "Functional residue analysis completed",
                    },
                )
                yield sse("log", {"line": "Functional residue analysis completed."})

            if "properties" in selected_analyses:
                yield sse("log", {"line": "Running physical/chemical properties analysis..."})
                properties_result = await asyncio.to_thread(
                    handle_physical_chemical_properties,
                    content_for_analysis,
                    selected_chain,
                    resolved_current_file,
                    sequence_map,
                    original_content,
                )
                sections.append("## ⚗️ Physical & Chemical Properties")
                sections.append("")
                sections.append(properties_result)
                sections.append("")
                done_steps += 1
                yield sse(
                    "progress",
                    {
                        "progress": round(0.05 + 0.7 * done_steps / total_steps, 3),
                        "message": "Properties analysis completed",
                    },
                )
                yield sse("log", {"line": "Properties analysis completed."})

            sections.append("---")
            sections.append("")
            sections.append("## 💡 Conclusion")
            sections.append("")
            sections.append("✅ Analysis completed. Please validate critical findings experimentally.")
            sections.append("")
            sections.append(f"*Report generated by VenusFactory2 v2 at {time.strftime('%Y-%m-%d %H:%M:%S')}*")

            report_text = "\n".join(sections)
            yield sse("progress", {"progress": 0.82, "message": "Generating AI expert report"})
            yield sse("log", {"line": "Calling LLM for AI expert analysis..."})
            ai_report = await asyncio.to_thread(generate_expert_analysis_report, report_text)
            done_steps += 1
            yield sse("progress", {"progress": 0.9, "message": "AI expert report generated"})
            yield sse("log", {"line": "AI expert analysis completed."})

            yield sse("progress", {"progress": 0.94, "message": "Exporting HTML/PDF"})
            html_path = await asyncio.to_thread(export_ai_report_to_html, ai_report)
            pdf_path = await asyncio.to_thread(export_ai_report_to_pdf, ai_report)
            done_steps += 1
            run_id = build_run_id_utc()
            result_dir = get_web_v2_area_dir("results", tool="report", run_id=run_id)
            html_stage = ""
            pdf_stage = ""
            if html_path and Path(html_path).exists():
                html_dst = result_dir / make_web_v2_result_name("report_html", 1, ".html")
                html_dst.write_bytes(Path(html_path).read_bytes())
                html_stage = str(html_dst.relative_to(_REPORT_RESULTS_ROOT))
            if pdf_path and Path(pdf_path).exists():
                pdf_dst = result_dir / make_web_v2_result_name("report_pdf", 1, ".pdf")
                pdf_dst.write_bytes(Path(pdf_path).read_bytes())
                pdf_stage = str(pdf_dst.relative_to(_REPORT_RESULTS_ROOT))
            create_run_manifest(
                run_id=run_id,
                tool="report",
                status="completed",
                outputs=[{"path": html_stage}, {"path": pdf_stage}],
            )
            yield sse("log", {"line": "Export completed."})
            yield sse("progress", {"progress": 1.0, "message": "Report generation finished"})

            done_payload = {
                "success": True,
                "report_text": report_text,
                "ai_report": ai_report,
                "html_download_url": build_web_v2_download_url(html_stage) if html_stage else "",
                "pdf_download_url": build_web_v2_download_url(pdf_stage) if pdf_stage else "",
                "html_path": html_stage,
                "pdf_path": pdf_stage,
                "run_id": run_id,
            }
            yield sse("done", done_payload)
        except Exception as exc:
            yield sse("error", {"message": str(exc)})
            yield sse("done", {"success": False, "message": str(exc)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/download/{kind}/{filename}")
async def download_report(kind: str, filename: str, run_id: str = Query(default="")):
    if kind not in {"html", "pdf"}:
        raise HTTPException(status_code=400, detail="Invalid download type.")
    safe_name = os.path.basename(filename)
    if not safe_name:
        raise HTTPException(status_code=404, detail="Missing file name.")
    target: Optional[Path] = None
    if run_id.strip():
        run_dir = get_web_v2_area_dir("results", tool="report", run_id=run_id.strip())
        candidate = run_dir / safe_name
        if candidate.exists() and candidate.is_file():
            target = candidate
    else:
        candidates = list(_REPORT_RESULTS_ROOT.rglob(safe_name))
        if len(candidates) > 1:
            raise HTTPException(status_code=409, detail="Multiple report files found; provide run_id.")
        target = candidates[0] if candidates else None
    if not target or not target.exists() or not ensure_within_roots(target, [_REPORT_RESULTS_ROOT]):
        raise HTTPException(status_code=404, detail="Report file not found.")
    media = "text/html" if kind == "html" else "application/pdf"
    return FileResponse(path=str(target), filename=safe_name, media_type=media)
