"""Experiment report download + full-session zip export endpoints."""
import json
import zipfile
from datetime import datetime

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, Response

from web.utils.common_utils import build_run_id_utc, get_web_v2_area_dir

from web_v2.chat_api._report_files import _copy_session_files, _write_chat_history_pdf
from web_v2.chat_api._report_md import _generate_experiment_report
from web_v2.chat_api._shared import (
    _assert_session_access,
    _get_session_or_404,
    _record_access_event,
)

router = APIRouter()


@router.get("/sessions/{session_id}/report")
async def get_experiment_report(session_id: str, request: Request):
    _record_access_event(request, "/api/chat/sessions/{id}/report")
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)

    report = _generate_experiment_report(state)
    filename = f"experiment_report_{session_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    return Response(
        content=report,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/sessions/{session_id}/export")
async def export_session_bundle(session_id: str, request: Request):
    _record_access_event(request, "/api/chat/sessions/{id}/export")
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)

    run_id = build_run_id_utc()
    export_root = get_web_v2_area_dir("results", tool="chat_export", run_id=run_id)
    export_root.mkdir(parents=True, exist_ok=True)

    report_path = export_root / "final_report.md"
    report_path.write_text(_generate_experiment_report(state), encoding="utf-8")

    pdf_path = export_root / "chat_history.pdf"
    _write_chat_history_pdf(state, pdf_path)

    snapshot = {
        "session_id": state.get("session_id"),
        "model_name": getattr(state.get("llm"), "model_name", ""),
        "created_at": str(state.get("created_at", "")),
        "status": state.get("status", ""),
        "chat_mode": state.get("chat_mode", ""),
        "engine": state.get("engine", ""),
        "history": state.get("history", []),
        "conversation_log": state.get("conversation_log", []),
        "tool_executions": state.get("tool_executions", []),
        "plan": state.get("plan", []),
    }
    snapshot_path = export_root / "session_snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    _copy_session_files(state, export_root)

    zip_name = f"chat_export_{session_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    zip_path = export_root / zip_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in export_root.rglob("*"):
            if file_path == zip_path or file_path.is_dir():
                continue
            zf.write(file_path, file_path.relative_to(export_root))

    return FileResponse(
        path=str(zip_path),
        filename=zip_name,
        media_type="application/zip",
    )
