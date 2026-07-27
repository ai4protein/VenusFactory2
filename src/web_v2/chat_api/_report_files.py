"""PDF chat-history writer + session-files copier for the export endpoint.

Split from ``_report_md`` so the heavy markdown generator and the lightweight
file utilities stay independently reviewable.
"""
import shutil
from pathlib import Path
from typing import Any


def _write_chat_history_pdf(state: dict[str, Any], output_path: Path) -> None:
    history = state.get("history", []) or []
    lines: list[str] = []
    for idx, msg in enumerate(history, 1):
        kind = str(msg.get("kind", "") or "")
        # Skip empty / hollow thinking placeholders from kimi warm-up.
        content = str(msg.get("content", "") or "").strip()
        if kind == "thinking" and not content:
            continue
        if not content and str(msg.get("role", "")) == "assistant":
            continue
        role = str(msg.get("role", "unknown")).upper()
        role_id = str(msg.get("role_id", "") or "")
        suffix_parts = [p for p in (role_id, kind) if p]
        title = f"{idx}. {role}" + (f" ({', '.join(suffix_parts)})" if suffix_parts else "")
        if kind == "thinking":
            title += " [thinking]"
        lines.append(title)
        lines.extend((content or "").splitlines() or [""])
        lines.append("")

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfbase import pdfmetrics  # noqa: F401
        from reportlab.pdfbase.ttfonts import TTFont  # noqa: F401
        from reportlab.pdfgen import canvas
    except Exception as exc:
        raise RuntimeError("PDF export requires reportlab. Please install it in current environment.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output_path), pagesize=A4)
    width, height = A4
    margin = 40
    y = height - margin
    line_height = 14

    font_name = "Helvetica"
    c.setFont(font_name, 10)
    for raw in lines:
        text = raw.rstrip()
        if not text:
            y -= line_height
            if y < margin:
                c.showPage()
                c.setFont(font_name, 10)
                y = height - margin
            continue
        chunks = [text[i:i + 110] for i in range(0, len(text), 110)]
        for chunk in chunks:
            c.drawString(margin, y, chunk)
            y -= line_height
            if y < margin:
                c.showPage()
                c.setFont(font_name, 10)
                y = height - margin
    c.save()


def _copy_session_files(state: dict[str, Any], target_dir: Path) -> list[str]:
    copied: list[str] = []
    session_dir_raw = str(state.get("agent_session_dir", "") or "").strip()
    if not session_dir_raw:
        return copied
    session_dir = Path(session_dir_raw)
    if not session_dir.exists() or not session_dir.is_dir():
        return copied
    dst = target_dir / "session_files"
    shutil.copytree(session_dir, dst, dirs_exist_ok=True)
    copied.append(str(dst))
    return copied
