"""Artifact extraction + protein_ctx registration + OSS upload helpers.

The OSS upload is fire-and-forget (matches legacy behaviour: scheduled with
``asyncio.create_task`` and emits an ``artifact_uploaded`` SSE event when done).
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from logger import get_logger

from agent.chat_agent_utils import _get_output_file_path_from_raw
from agent.graph.execution.context import ExecutionContext
from web.utils.file_oss import upload_file_to_cloud_async

_logger = get_logger("agent.graph")


def extract_output_artifact_paths(tool_name: str, raw_output: Any) -> list[str]:
    """All deterministic file paths advertised by the tool's raw output.

    Mirrors the local ``_extract_output_artifact_paths`` closure from the
    original implementation, including the abs-path dedup pass.
    """
    paths: list[str] = []
    primary = _get_output_file_path_from_raw(raw_output, tool_name)
    if primary:
        paths.append(primary)
    try:
        parsed = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
        if isinstance(parsed, dict):
            direct = parsed.get("file_path")
            if isinstance(direct, str) and direct.strip():
                resolved = _get_output_file_path_from_raw(
                    json.dumps({"file_path": direct}, ensure_ascii=False), tool_name
                )
                if resolved:
                    paths.append(resolved)
            info = parsed.get("file_info")
            if isinstance(info, dict):
                fp = info.get("file_path")
                if isinstance(fp, str) and fp.strip():
                    resolved = _get_output_file_path_from_raw(
                        json.dumps({"file_info": {"file_path": fp}}, ensure_ascii=False),
                        tool_name,
                    )
                    if resolved:
                        paths.append(resolved)
    except Exception:
        pass

    unique: list[str] = []
    seen = set()
    for p in paths:
        norm = os.path.abspath(p)
        if norm in seen:
            continue
        seen.add(norm)
        unique.append(norm)
    return unique


def register_artifacts_to_context(ctx: ExecutionContext, raw_output: Any) -> None:
    """Register output files with the protein context (structures + generic)."""
    for file_path in extract_output_artifact_paths(ctx.tool_name, raw_output):
        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext in {".pdb", ".cif", ".mmcif"}:
                ctx.protein_ctx.add_structure_file(
                    file_path,
                    source=ctx.tool_name,
                    uniprot_id=ctx.protein_ctx.last_uniprot_id,
                )
            ctx.protein_ctx.add_file(file_path)
        except Exception as e:
            _logger.warning("Artifact register failed for %s: %s", file_path, e)


async def _upload_and_emit(
    file_path: str,
    kind: str,
    step_no: Any,
    sess_id: str,
    writer: Any,
) -> None:
    try:
        url = await upload_file_to_cloud_async(file_path)
        if url and writer:
            try:
                writer(
                    {
                        "type": "artifact_uploaded",
                        "kind": kind,
                        "step": step_no,
                        "session_id": sess_id,
                        "name": os.path.basename(file_path),
                        "url": url,
                    }
                )
            except Exception as _emit_err:
                _logger.warning(
                    "artifact_uploaded emit failed for %s: %s", file_path, _emit_err
                )
    except Exception as e:
        _logger.warning("OSS upload failed for %s: %s", file_path, e)


def schedule_upload(
    file_path: str,
    kind: str,
    step_no: Any,
    sess_id: str,
    writer: Any,
) -> None:
    """Schedule a fire-and-forget OSS upload + SSE emit.

    Logs any unexpected task errors via a done callback (matches legacy).
    """
    try:
        task = asyncio.create_task(
            _upload_and_emit(file_path, kind, step_no, sess_id, writer)
        )

        def _on_done(t: "asyncio.Task[Any]", _fp: str = file_path) -> None:
            try:
                exc = t.exception()
            except asyncio.CancelledError:
                return
            except Exception:
                return
            if exc is not None:
                _logger.warning(
                    "Background OSS upload task error for %s: %s", _fp, exc
                )

        task.add_done_callback(_on_done)
    except Exception as e:
        _logger.warning("Failed to schedule OSS upload for %s: %s", file_path, e)
