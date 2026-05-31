"""Build long-form experiment report markdown from session state.

Used by the ``export`` sub-router. Holds pure functions; no router endpoints.
The PDF writer + session-files copier live in ``_report_files``.
"""
import json
import os
import re
from datetime import datetime
from typing import Any

from agent.chat_graph import _format_clarification_answers
from web.utils.common_utils import redact_path_text

from web_v2.chat_api._shared import _is_zh_text


def _generate_experiment_report(state: dict[str, Any]) -> str:
    """Build a comprehensive long-form Markdown research report from session state."""
    is_zh = _is_zh_text(state.get("last_user_text", ""))
    model_name = getattr(state.get("llm"), "model_name", "unknown")
    session_id = state.get("session_id", "unknown")
    created_at = str(state.get("created_at", ""))
    generated_at = datetime.now().isoformat()
    history = state.get("history", [])

    lines: list[str] = []

    # ── Title & Metadata ──
    lines.append("# " + ("实验研究报告" if is_zh else "Experiment Research Report"))
    lines.append("")
    lines.append(f"| {'字段' if is_zh else 'Field'} | {'值' if is_zh else 'Value'} |")
    lines.append("|---|---|")
    lines.append(f"| Session ID | `{session_id}` |")
    lines.append(f"| {'模型' if is_zh else 'Model'} | {model_name} |")
    lines.append(f"| {'创建时间' if is_zh else 'Created'} | {created_at} |")
    lines.append(f"| {'报告生成时间' if is_zh else 'Report Generated'} | {generated_at} |")
    lines.append(f"| {'状态' if is_zh else 'Status'} | {state.get('status', '')} |")
    lines.append("")

    # ── Table of Contents ──
    lines.append("## " + ("目录" if is_zh else "Table of Contents"))
    lines.append("")
    toc_items = [
        ("蛋白质上下文", "Protein Context"),
        ("研究背景与用户需求", "Background & User Request"),
        ("文献调研", "Literature Research"),
        ("调研报告草稿", "Research Report Draft"),
        ("实验设计", "Experimental Design"),
        ("实验过程与结果", "Experimental Process & Results"),
        ("图表与可视化", "Figures & Visualizations"),
        ("讨论与结论", "Discussion & Conclusion"),
    ]
    for i, (zh, en) in enumerate(toc_items, 1):
        lines.append(f"{i}. [{zh if is_zh else en}](#{i})")
    lines.append("")

    # ── 1. Protein Context ──
    lines.append("## 1. " + ("蛋白质上下文" if is_zh else "Protein Context"))
    lines.append("")
    protein_ctx = state.get("protein_context")
    if protein_ctx:
        ctx_summary = protein_ctx.get_context_summary()
        if ctx_summary and ctx_summary != "No protein data in memory":
            lines.append(redact_path_text(ctx_summary))
        else:
            lines.append("_" + ("本次实验未涉及特定蛋白质。" if is_zh else "No specific protein context in this experiment.") + "_")
    else:
        lines.append("_" + ("本次实验未涉及特定蛋白质。" if is_zh else "No specific protein context in this experiment.") + "_")
    lines.append("")

    # ── 2. Background & User Request ──
    lines.append("## 2. " + ("研究背景与用户需求" if is_zh else "Background & User Request"))
    lines.append("")
    original_text = state.get("last_user_text", "")
    if original_text:
        lines.append(redact_path_text(original_text))
        lines.append("")

    questions = state.get("clarification_questions", [])
    answers = state.get("clarification_answers", [])
    if questions and answers:
        lines.append("### " + ("需求澄清" if is_zh else "Clarification Q&A"))
        lines.append("")
        formatted = _format_clarification_answers(questions, answers)
        if formatted:
            lines.append(redact_path_text(formatted))
            lines.append("")

    # ── 3. Literature Research ──
    sections = state.get("research_sections", [])
    sub_reports = state.get("research_sub_reports", [])
    if sections or sub_reports:
        lines.append("## 3. " + ("文献调研" if is_zh else "Literature Research"))
        lines.append("")

        if sections:
            lines.append("### " + ("调研计划" if is_zh else "Research Plan"))
            lines.append("")
            for i, sec in enumerate(sections, 1):
                name = sec.get("section_name", f"Section {i}")
                focus = sec.get("focus", "")
                queries = sec.get("search_queries", [])
                lines.append(f"**{i}. {name}**")
                if focus:
                    lines.append(f"  - {'研究重点' if is_zh else 'Focus'}: {focus}")
                if queries:
                    lines.append(f"  - {'检索词' if is_zh else 'Search queries'}:")
                    for q in queries:
                        lines.append(f"    - {q}")
                lines.append("")

        if sub_reports:
            lines.append("### " + ("各节调研结果" if is_zh else "Section-by-Section Research Results"))
            lines.append("")
            for sr in sub_reports:
                lines.append(redact_path_text(sr))
                lines.append("")
                lines.append("---")
                lines.append("")

    # ── 4. Research Report Draft (PI final report) ──
    pi_report = state.get("pi_report", "")
    if pi_report:
        lines.append("## 4. " + ("调研报告草稿" if is_zh else "Research Report Draft"))
        lines.append("")
        lines.append(redact_path_text(pi_report))
        lines.append("")

    # ── 5. Experimental Design ──
    plan = state.get("plan", [])
    if plan:
        lines.append("## 5. " + ("实验设计" if is_zh else "Experimental Design"))
        lines.append("")
        pi_suggest = state.get("pi_suggest_steps", "")
        if pi_suggest:
            lines.append("### " + ("PI 建议方案" if is_zh else "PI Suggested Approach"))
            lines.append("")
            lines.append(redact_path_text(pi_suggest))
            lines.append("")

        lines.append("### " + ("最终执行计划" if is_zh else "Final Execution Plan"))
        lines.append("")
        lines.append(f"| {'步骤' if is_zh else 'Step'} | {'工具' if is_zh else 'Tool'} | {'描述' if is_zh else 'Description'} |")
        lines.append("|---|---|---|")
        for step in plan:
            step_num = step.get("step", "?")
            tool = step.get("tool_name", "?")
            desc = step.get("task_description", "").replace("\n", " ").replace("|", "\\|")
            lines.append(f"| {step_num} | `{tool}` | {desc} |")
        lines.append("")

    # ── 6. Experimental Process & Results ──
    tool_executions = state.get("tool_executions", [])
    if tool_executions:
        lines.append("## 6. " + ("实验过程与结果" if is_zh else "Experimental Process & Results"))
        lines.append("")

        for entry in tool_executions:
            step = entry.get("step", "?")
            tool = entry.get("tool_name", "unknown")
            ts = entry.get("timestamp", "")
            inputs = entry.get("inputs", {})
            outputs = entry.get("outputs", "")
            oss_url = entry.get("oss_url")

            step_plan_desc = ""
            for p in plan:
                if str(p.get("step", "")) == str(step):
                    step_plan_desc = p.get("task_description", "")
                    break

            lines.append(f"### Step {step}: `{tool}`")
            lines.append("")
            if step_plan_desc:
                lines.append(f"**{'任务目标' if is_zh else 'Objective'}:** {step_plan_desc}")
                lines.append("")
            if ts:
                lines.append(f"**{'执行时间' if is_zh else 'Timestamp'}:** {ts}")
                lines.append("")

            if inputs:
                lines.append(f"**{'输入参数' if is_zh else 'Input Parameters'}:**")
                lines.append("")
                lines.append("```json")
                lines.append(redact_path_text(json.dumps(inputs, ensure_ascii=False, indent=2)))
                lines.append("```")
                lines.append("")

            if outputs:
                output_str = redact_path_text(str(outputs))
                lines.append(f"**{'输出结果' if is_zh else 'Output'}:**")
                lines.append("")
                lines.append("```")
                lines.append(output_str[:4000] + ("..." if len(output_str) > 4000 else ""))
                lines.append("```")
                lines.append("")

            if oss_url:
                name = os.path.basename(oss_url)
                lines.append(f"**{'云端下载' if is_zh else 'Cloud Download'}:** [{name}]({oss_url})")
                lines.append("")

            # Extract per-step MLS feedback from history (cloud links, file previews, images)
            mls_feedback = _extract_step_feedback_from_history(history, step, tool)
            if mls_feedback:
                lines.append(f"**{'详细反馈' if is_zh else 'Detailed Feedback'}:**")
                lines.append("")
                lines.append(redact_path_text(mls_feedback))
                lines.append("")

            lines.append("---")
            lines.append("")

    # ── 7. Figures & Visualizations ──
    figure_links = _extract_figure_links_from_history(history)
    if figure_links:
        lines.append("## 7. " + ("图表与可视化" if is_zh else "Figures & Visualizations"))
        lines.append("")
        for i, (fig_name, fig_url) in enumerate(figure_links, 1):
            lines.append(f"### Figure {i}: {fig_name}")
            lines.append("")
            lines.append(f"![{fig_name}]({fig_url})")
            lines.append("")
            lines.append(f"[{'下载' if is_zh else 'Download'}]({fig_url})")
            lines.append("")

    # ── 8. Discussion & Conclusion ──
    lines.append("## 8. " + ("讨论与结论" if is_zh else "Discussion & Conclusion"))
    lines.append("")

    # Collect the finalizer summary (Scientific Critic / PI)
    skip_markers = (
        "iteration_prompt", "请选择下一步", "Please choose",
        "正在分析", "is analyzing", "正在汇总", "is summarizing",
        "Thinking", "思考中", "撰写小报告", "writing sub-report",
        "撰写研究草案", "writing the draft report",
        "⏳", "✍️", "📝", "sub_report_checkpoint",
        "step_checkpoint", "🔍 **第", "🔍 **Step",
    )
    final_summaries: list[str] = []
    for item in reversed(history):
        if item.get("role") != "assistant":
            continue
        content = item.get("content", "")
        role_id = item.get("role_id", "")
        if not content or len(content) < 50:
            continue
        if any(m in content for m in skip_markers):
            continue
        if role_id == "principal_investigator":
            final_summaries.append(content)
            if len(final_summaries) >= 2:
                break

    if final_summaries:
        for summary in reversed(final_summaries):
            lines.append(redact_path_text(summary))
            lines.append("")
    else:
        lines.append("_" + ("暂无总结。" if is_zh else "No conclusion available yet.") + "_")
        lines.append("")

    # Failure information
    if state.get("execution_failed"):
        lines.append("### " + ("失败信息" if is_zh else "Failure Details"))
        lines.append("")
        lines.append(f"- **{'失败步骤' if is_zh else 'Failed Step'}:** {state.get('failed_step', 'N/A')}")
        lines.append(f"- **{'失败原因' if is_zh else 'Failure Reason'}:** {state.get('failed_reason', 'Unknown')}")
        lines.append("")

    # ── Footer ──
    lines.append("---")
    lines.append("")
    lines.append(f"*{'由 VenusFactory 多智能体系统自动生成' if is_zh else 'Auto-generated by VenusFactory Multi-Agent System'}* | "
                 f"Session `{session_id[:8]}` | {generated_at}")
    lines.append("")

    return "\n".join(lines)


def _extract_step_feedback_from_history(
    history: list[dict[str, Any]], step: Any, tool_name: str
) -> str:
    """Extract the MLS feedback message for a specific execution step from conversation history."""
    step_str = str(step)
    executing_marker = f"Step {step_str}"
    found_step = False
    for item in history:
        if item.get("role") != "assistant" or item.get("role_id") != "machine_learning_specialist":
            continue
        content = item.get("content", "")
        if not found_step:
            if executing_marker in content and ("⏳" in content or "executing" in content.lower() or "正在执行" in content):
                found_step = True
            continue
        # This is the feedback message right after the executing message
        if "📎" in content or "🖼️" in content or "Cloud Download" in content or "云端下载" in content or "File Preview" in content or "文件预览" in content:
            return content
        if len(content) > 100 and (tool_name in content or "Summary" in content or "summary" in content or "输出" in content):
            return content
        break
    return ""


def _extract_figure_links_from_history(history: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Extract all generated image links from the conversation history."""
    _IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".bmp", ".tiff")
    figures: list[tuple[str, str]] = []
    seen_urls: set[str] = set()
    img_emoji_pat = re.compile(r"🖼️[^[]*\[([^\]]+)\]\(([^)]+)\)")
    md_img_pat = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
    md_link_pat = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
    for item in history:
        content = item.get("content", "")
        if not content:
            continue
        for m in img_emoji_pat.finditer(content):
            name, url = m.group(1), m.group(2)
            if url and url not in seen_urls:
                seen_urls.add(url)
                figures.append((name or "figure", url))
        for m in md_img_pat.finditer(content):
            name, url = m.group(1), m.group(2)
            if url and url not in seen_urls:
                seen_urls.add(url)
                figures.append((name or "figure", url))
        for m in md_link_pat.finditer(content):
            name, url = m.group(1), m.group(2)
            if url not in seen_urls and any(url.lower().endswith(ext) for ext in _IMAGE_EXTS):
                seen_urls.add(url)
                figures.append((name, url))
    return figures


