"""Render the per-step feedback Markdown shown in the chat UI.

Splits the legacy L2260-2436 region into focused helpers:

* :func:`build_feedback_content` — assembles the markdown body, including
  inline upload placeholders, file previews, and structure-file paths.
* :func:`schedule_artifact_uploads` — fires off background OSS uploads for the
  primary output file and any auxiliary images.
* :func:`record_execution_entry` — appends a row to ``ctx.executions``.
* :func:`dispatch_hook_on_tool_end` — fires hooks (analytics, webhooks) if
  configured.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime
from typing import Any, Optional

from langgraph.config import get_stream_writer

from logger import get_logger

from agent.chat_agent_utils import (
    _extract_image_paths_from_tool_output,
    _get_output_file_path_from_raw,
    _read_output_file_preview,
)
from agent.graph.execution.artifacts import schedule_upload
from agent.graph.execution.context import ExecutionContext
from agent.hooks import ToolResultInfo
from web.utils.common_utils import to_project_relative_path

_logger = get_logger("agent.graph")


def _ui_text(lang: str, key: str, **kwargs) -> str:
    # Re-export through the shared common module so all UI strings stay in one place.
    from agent.graph.common.ui_text import _ui_text as _impl

    return _impl(lang, key, **kwargs)


def _format_output_summary(
    ctx: ExecutionContext, last_output: Any, out_data: Any
) -> str:
    """Build the Output Summary / Preview block."""
    feedback = ""
    try:
        if isinstance(out_data, dict):
            important_keys = [
                "success",
                "error",
                "message",
                "detail",
                "traceback",
                "tool_name",
                "protein_id",
                "pdb_id",
                "uniprot_id",
                "mutation",
                "delta_delta_g",
            ]
            summary_parts = []
            for k in important_keys:
                if k in out_data:
                    val = out_data[k]
                    summary_parts.append(f"**{k}:** {val}")
            if summary_parts:
                feedback += (
                    _ui_text(ctx.ui_lang, "summary") + ", ".join(summary_parts) + "\n\n"
                )
            else:
                dump = json.dumps(out_data, ensure_ascii=False)
                output_label = "输出：" if ctx.ui_lang == "zh" else "Output: "
                feedback += f"{output_label}`{dump[:300]}...`\n\n"
        else:
            feedback += (
                _ui_text(ctx.ui_lang, "output_preview")
                + f"`{str(last_output)[:300]}...`\n\n"
            )
    except Exception:
        feedback += (
            _ui_text(ctx.ui_lang, "output_preview")
            + f"`{str(last_output)[:300]}...`\n\n"
        )
    return feedback


def build_feedback_content(
    ctx: ExecutionContext,
    last_output: Any,
    is_failure: bool,
) -> tuple[str, Optional[Any], Optional[str], Optional[Any]]:
    """Render the per-step feedback Markdown body.

    Returns ``(feedback_content, out_data, out_file, sse_writer)`` so the caller
    can reuse the parsed JSON / output path / writer without re-doing work.
    """
    try:
        out_data = json.loads(last_output) if isinstance(last_output, str) else last_output
    except Exception:
        out_data = None

    if is_failure:
        feedback_content = _ui_text(ctx.ui_lang, "step_failed", step_num=ctx.step_num)
    else:
        feedback_content = _ui_text(ctx.ui_lang, "step_done", step_num=ctx.step_num)

    feedback_content += _format_output_summary(ctx, last_output, out_data)

    raw_str = (
        last_output if isinstance(last_output, str) else json.dumps(last_output, ensure_ascii=False)
    )
    if is_failure or not isinstance(out_data, dict):
        feedback_content += _ui_text(ctx.ui_lang, "raw_output") + "\n```\n"
        feedback_content += raw_str[:2000] + ("\n...(truncated)" if len(raw_str) > 2000 else "")
        feedback_content += "\n```\n\n"

    try:
        sse_writer = get_stream_writer()
    except Exception:
        sse_writer = None

    out_file = _get_output_file_path_from_raw(last_output, ctx.tool_name)
    if out_file:
        feedback_content += _inline_upload_placeholder(ctx.ui_lang, out_file)
        preview = _read_output_file_preview(out_file)
        if preview:
            feedback_content += (
                _ui_text(ctx.ui_lang, "file_preview", name=os.path.basename(out_file))
                + f"\n```\n{preview}\n```\n\n"
            )
        _structure_exts = {".pdb", ".cif", ".mmcif", ".ent"}
        if os.path.splitext(out_file)[1].lower() in _structure_exts:
            rel_path = to_project_relative_path(out_file)
            feedback_content += f"📂 Structure file: `{rel_path}`\n\n"

    return feedback_content, out_data, out_file, sse_writer


def _inline_upload_placeholder(ui_lang: str, out_file: str) -> str:
    base = os.path.basename(out_file)
    if ui_lang == "zh":
        return f"📎 **正在上传到云端：** {base} …\n\n"
    return f"📎 **Uploading to cloud:** {base} …\n\n"


def schedule_artifact_uploads(
    ctx: ExecutionContext,
    last_output: Any,
    out_file: Optional[str],
    sse_writer: Any,
) -> str:
    """Fire-and-forget OSS uploads for the primary output and any images.

    Returns extra Markdown fragments (one per image) to append to the feedback
    body — these match the legacy "Uploading image" placeholders.
    """
    session_id_for_upload = str(ctx.state.get("session_id", ""))
    extra_md = ""
    if out_file:
        schedule_upload(out_file, "output", ctx.step_num, session_id_for_upload, sse_writer)

    try:
        img_paths = _extract_image_paths_from_tool_output(last_output, ctx.tool_name)
        for ip in img_paths:
            if ip != out_file:
                try:
                    schedule_upload(ip, "image", ctx.step_num, session_id_for_upload, sse_writer)
                    if ctx.ui_lang == "zh":
                        extra_md += f"🖼️ **正在上传图片：** {os.path.basename(ip)} …\n\n"
                    else:
                        extra_md += f"🖼️ **Uploading image:** {os.path.basename(ip)} …\n\n"
                except Exception as e:
                    _logger.warning(
                        "Failed to schedule OSS image upload for %s: %s", ip, e
                    )
    except Exception:
        pass
    return extra_md


def record_execution_entry(
    ctx: ExecutionContext,
    last_output: Any,
    oss_url: Optional[str] = None,
) -> None:
    ctx.executions.append(
        {
            "step": ctx.step_num,
            "tool_name": ctx.tool_name,
            "inputs": ctx.merged_tool_input,
            "outputs": (
                str(last_output)[:1000] + "..."
                if len(str(last_output)) > 1000
                else str(last_output)
            ),
            "oss_url": oss_url,
            "timestamp": datetime.now().isoformat(),
        }
    )


def dispatch_hook_on_tool_end(
    ctx: ExecutionContext,
    last_output: Any,
    is_failure: bool,
    failure_reason: str,
    execute_started: float,
    bg_tasks: set,
) -> None:
    """Schedule the ``on_tool_end`` hook (analytics + webhook). Best-effort."""
    from agent.graph.common.usage import _extract_usage_from_output

    try:
        input_tokens, output_tokens, total_tokens, usage_missing = _extract_usage_from_output(
            last_output
        )
        hooks = ctx.state.get("hooks")
        if hooks is None:
            return
        info = ToolResultInfo(
            tool_name=ctx.tool_name,
            tool_input=ctx.merged_tool_input,
            raw_output=last_output,
            success=not is_failure,
            error_message=failure_reason or "",
            duration_seconds=time.time() - execute_started,
            step_index=ctx.step_num,
        )
        record_task = asyncio.create_task(
            hooks.on_tool_end(
                info=info,
                session_id=str(ctx.state.get("session_id", "")),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                usage_missing=usage_missing,
                model=(
                    getattr(ctx.chains.get("llm"), "model_name", "")
                    if ctx.chains.get("llm")
                    else ""
                ),
                owner_key=str(
                    ctx.config.get("configurable", {})
                    .get("chains", {})
                    .get("owner_key", ctx.state.get("owner_key", ""))
                ),
                ip=str(ctx.state.get("client_ip", "")),
            )
        )
        bg_tasks.add(record_task)
        record_task.add_done_callback(bg_tasks.discard)
    except Exception:
        _logger.debug("hook on_tool_end dispatch failed", exc_info=True)
