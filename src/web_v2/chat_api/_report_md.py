"""Build long-form experiment report markdown from session state.

Used by the ``export`` sub-router. Holds pure functions; no router endpoints.
The PDF writer + session-files copier live in ``_report_files``.

Supports two chat modes:
  - science_agent (kimi-code): conversation + tool calls (args/output)
  - science_expert (LangGraph): PI/CB/MLS/SC research pipeline report
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any

from agent.chat_graph import _format_clarification_answers
from web.utils.common_utils import redact_path_text

from web_v2.chat_api._shared import _infer_chat_mode, _is_zh_text


def _detect_chat_mode(state: dict[str, Any], history: list[dict[str, Any]]) -> str:
    """Resolve report template mode; fall back to kimi-shaped history heuristics."""
    mode = _infer_chat_mode(state)
    if mode == "science_agent":
        return mode
    engine = str(state.get("engine") or "")
    llm = state.get("llm")
    model_name = getattr(llm, "model_name", "") if llm is not None else ""
    if engine == "kimi-code" or model_name == "kimi-code":
        return "science_agent"
    # Persisted snapshots may lack llm/engine but still carry kimi message kinds.
    if any(item.get("kind") == "thinking" for item in history):
        return "science_agent"
    tool_execs = state.get("tool_executions") or []
    if tool_execs and any("args" in te and "inputs" not in te for te in tool_execs):
        return "science_agent"
    return mode


def _generate_experiment_report(state: dict[str, Any]) -> str:
    """Build a Markdown research report from session state (Agent or Expert)."""
    history = list(state.get("history", []) or [])
    chat_mode = _detect_chat_mode(state, history)
    user_text = _primary_user_text(state, history)
    is_zh = _is_zh_text(user_text or state.get("last_user_text", "") or "")

    if chat_mode == "science_agent":
        return _generate_science_agent_report(state, history, user_text, is_zh)
    return _generate_science_expert_report(state, history, user_text, is_zh)


def _primary_user_text(state: dict[str, Any], history: list[dict[str, Any]]) -> str:
    text = str(state.get("last_user_text", "") or "").strip()
    if text:
        return text
    for item in reversed(history):
        if item.get("role") == "user":
            content = str(item.get("content", "") or "").strip()
            if content:
                return content
    return ""


def _model_label(state: dict[str, Any], chat_mode: str) -> str:
    llm = state.get("llm")
    model_name = getattr(llm, "model_name", "") if llm is not None else ""
    if chat_mode == "science_agent":
        return model_name or "Science Agent (kimi-code)"
    return model_name or "unknown"


def _tool_fields(entry: dict[str, Any]) -> tuple[Any, Any, str, str]:
    """Normalize graph vs kimi tool_execution shapes."""
    tool = str(entry.get("tool_name") or entry.get("name") or "unknown")
    inputs = entry.get("inputs")
    if inputs is None:
        inputs = entry.get("args")
    if inputs is None:
        inputs = {}
    outputs = entry.get("outputs")
    if outputs is None:
        outputs = entry.get("output")
    if outputs is None:
        outputs = ""
    status = str(entry.get("status", "") or "")
    return inputs, outputs, tool, status


def _demote_md_headings(text: str, levels: int = 2) -> str:
    """Shift ATX headings down so embedded replies don't break report TOC."""
    if levels <= 0 or not text:
        return text
    prefix = "#" * levels

    def _repl(match: re.Match[str]) -> str:
        hashes = match.group(1)
        rest = match.group(2)
        # Cap at ###### 
        new_level = min(6, len(hashes) + levels)
        return "#" * new_level + rest

    return re.sub(r"^(#{1,6})(\s+.*)$", _repl, text, flags=re.MULTILINE)


def _pretty_jsonish(value: Any, limit: int = 4000) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, indent=2)
    else:
        text = str(value)
    text = redact_path_text(text)
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def _is_thinking_message(item: dict[str, Any]) -> bool:
    if item.get("kind") == "thinking" or item.get("phase") == "thinking":
        return True
    return False


def _collect_assistant_answers(history: list[dict[str, Any]]) -> list[str]:
    """Substantial assistant replies (skip thinking / ephemeral placeholders)."""
    skip_phases = {
        "thinking", "summarizing", "sub_report_writing", "draft_report_writing",
    }
    skip_markers = (
        "iteration_prompt", "请选择下一步", "Please choose",
        "正在分析", "is analyzing", "正在汇总", "is summarizing",
        "撰写小报告", "writing sub-report",
        "撰写研究草案", "writing the draft report",
        "⏳", "✍️", "📝", "sub_report_checkpoint",
        "step_checkpoint", "🔍 **第", "🔍 **Step",
    )
    answers: list[str] = []
    for item in history:
        if item.get("role") != "assistant":
            continue
        if _is_thinking_message(item) or item.get("phase") in skip_phases:
            continue
        content = str(item.get("content", "") or "").strip()
        if len(content) < 40:
            continue
        if any(m in content for m in skip_markers):
            continue
        answers.append(content)
    return answers


def _generate_science_agent_report(
    state: dict[str, Any],
    history: list[dict[str, Any]],
    user_text: str,
    is_zh: bool,
) -> str:
    session_id = state.get("session_id", "unknown")
    created_at = str(state.get("created_at", ""))
    generated_at = datetime.now().isoformat()
    model_name = _model_label(state, "science_agent")
    tool_executions = list(state.get("tool_executions", []) or [])
    answers = _collect_assistant_answers(history)
    figure_links = _extract_figure_links_from_history(history)
    figure_links.extend(_extract_figure_links_from_tool_outputs(tool_executions))
    # de-dupe figures by URL
    seen: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for name, url in figure_links:
        if url in seen:
            continue
        seen.add(url)
        deduped.append((name, url))
    figure_links = deduped

    lines: list[str] = []
    lines.append("# " + ("Science Agent 会话报告" if is_zh else "Science Agent Session Report"))
    lines.append("")
    lines.append(f"| {'字段' if is_zh else 'Field'} | {'值' if is_zh else 'Value'} |")
    lines.append("|---|---|")
    lines.append(f"| Session ID | `{session_id}` |")
    lines.append(f"| {'模式' if is_zh else 'Mode'} | Science Agent |")
    lines.append(f"| {'模型' if is_zh else 'Model'} | {model_name} |")
    lines.append(f"| {'创建时间' if is_zh else 'Created'} | {created_at} |")
    lines.append(f"| {'报告生成时间' if is_zh else 'Report Generated'} | {generated_at} |")
    lines.append(f"| {'状态' if is_zh else 'Status'} | {state.get('status', '')} |")
    lines.append("")

    lines.append("## " + ("目录" if is_zh else "Table of Contents"))
    lines.append("")
    toc = [
        ("用户请求", "User Request"),
        ("对话摘要", "Conversation"),
        ("工具调用与结果", "Tool Calls & Results"),
        ("图表与可视化", "Figures & Visualizations"),
        ("结论", "Conclusion"),
    ]
    for i, (zh, en) in enumerate(toc, 1):
        lines.append(f"{i}. [{zh if is_zh else en}](#{i})")
    lines.append("")

    # 1. User request — all user turns
    lines.append("## 1. " + ("用户请求" if is_zh else "User Request"))
    lines.append("")
    user_turns = [
        str(item.get("content", "") or "").strip()
        for item in history
        if item.get("role") == "user" and str(item.get("content", "") or "").strip()
    ]
    if user_turns:
        for i, turn in enumerate(user_turns, 1):
            if len(user_turns) > 1:
                lines.append(f"### {'轮次' if is_zh else 'Turn'} {i}")
                lines.append("")
            lines.append(redact_path_text(turn))
            lines.append("")
    elif user_text:
        lines.append(redact_path_text(user_text))
        lines.append("")
    else:
        lines.append("_" + ("暂无用户请求。" if is_zh else "No user request recorded.") + "_")
        lines.append("")

    # 2. Conversation (assistant text only)
    lines.append("## 2. " + ("对话摘要" if is_zh else "Conversation"))
    lines.append("")
    if answers:
        for i, ans in enumerate(answers, 1):
            lines.append(f"### {'回复' if is_zh else 'Reply'} {i}")
            lines.append("")
            lines.append(_demote_md_headings(redact_path_text(ans), levels=2))
            lines.append("")
    else:
        lines.append("_" + ("暂无助手回复。" if is_zh else "No assistant replies yet.") + "_")
        lines.append("")

    # 3. Tool calls
    lines.append("## 3. " + ("工具调用与结果" if is_zh else "Tool Calls & Results"))
    lines.append("")
    if tool_executions:
        for idx, entry in enumerate(tool_executions, 1):
            inputs, outputs, tool, status = _tool_fields(entry)
            ts = entry.get("timestamp") or entry.get("started_at") or ""
            lines.append(f"### {idx}. `{tool}`")
            lines.append("")
            if status:
                lines.append(f"**{'状态' if is_zh else 'Status'}:** `{status}`")
                lines.append("")
            if ts:
                lines.append(f"**{'执行时间' if is_zh else 'Timestamp'}:** {ts}")
                lines.append("")
            if inputs:
                lines.append(f"**{'输入参数' if is_zh else 'Input Parameters'}:**")
                lines.append("")
                lines.append("```json")
                lines.append(_pretty_jsonish(inputs))
                lines.append("```")
                lines.append("")
            if outputs not in ("", None):
                lines.append(f"**{'输出结果' if is_zh else 'Output'}:**")
                lines.append("")
                lines.append("```")
                lines.append(_pretty_jsonish(outputs))
                lines.append("```")
                lines.append("")
            oss_url = entry.get("oss_url")
            if oss_url:
                name = os.path.basename(str(oss_url))
                lines.append(f"**{'云端下载' if is_zh else 'Cloud Download'}:** [{name}]({oss_url})")
                lines.append("")
            lines.append("---")
            lines.append("")
    else:
        lines.append("_" + ("本次会话未记录工具调用。" if is_zh else "No tool calls recorded.") + "_")
        lines.append("")

    # 4. Figures
    lines.append("## 4. " + ("图表与可视化" if is_zh else "Figures & Visualizations"))
    lines.append("")
    if figure_links:
        for i, (fig_name, fig_url) in enumerate(figure_links, 1):
            lines.append(f"### Figure {i}: {fig_name}")
            lines.append("")
            lines.append(f"![{fig_name}]({fig_url})")
            lines.append("")
            lines.append(f"[{'下载' if is_zh else 'Download'}]({fig_url})")
            lines.append("")
    else:
        lines.append("_" + ("未检测到图表。" if is_zh else "No figures detected.") + "_")
        lines.append("")

    # 5. Conclusion = last substantial answer
    lines.append("## 5. " + ("结论" if is_zh else "Conclusion"))
    lines.append("")
    if answers:
        lines.append(_demote_md_headings(redact_path_text(answers[-1]), levels=2))
        lines.append("")
    else:
        lines.append("_" + ("暂无结论。" if is_zh else "No conclusion available yet.") + "_")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        f"*{'由 VenusFactory Science Agent 自动生成' if is_zh else 'Auto-generated by VenusFactory Science Agent'}* | "
        f"Session `{str(session_id)[:8]}` | {generated_at}"
    )
    lines.append("")
    return "\n".join(lines)


def _generate_science_expert_report(
    state: dict[str, Any],
    history: list[dict[str, Any]],
    user_text: str,
    is_zh: bool,
) -> str:
    """Legacy LangGraph PI/CB/MLS/SC long-form report."""
    model_name = _model_label(state, "science_expert")
    session_id = state.get("session_id", "unknown")
    created_at = str(state.get("created_at", ""))
    generated_at = datetime.now().isoformat()

    lines: list[str] = []

    lines.append("# " + ("Science Expert 实验研究报告" if is_zh else "Science Expert Experiment Report"))
    lines.append("")
    lines.append(f"| {'字段' if is_zh else 'Field'} | {'值' if is_zh else 'Value'} |")
    lines.append("|---|---|")
    lines.append(f"| Session ID | `{session_id}` |")
    lines.append(f"| {'模式' if is_zh else 'Mode'} | Science Expert |")
    lines.append(f"| {'模型' if is_zh else 'Model'} | {model_name} |")
    lines.append(f"| {'创建时间' if is_zh else 'Created'} | {created_at} |")
    lines.append(f"| {'报告生成时间' if is_zh else 'Report Generated'} | {generated_at} |")
    lines.append(f"| {'状态' if is_zh else 'Status'} | {state.get('status', '')} |")
    lines.append("")

    sections = state.get("research_sections", []) or []
    sub_reports = state.get("research_sub_reports", []) or []
    pi_report = state.get("pi_report", "") or ""
    plan = state.get("plan", []) or []
    tool_executions = state.get("tool_executions", []) or []
    figure_links = _extract_figure_links_from_history(history)

    toc_items: list[tuple[str, str]] = [
        ("蛋白质上下文", "Protein Context"),
        ("研究背景与用户需求", "Background & User Request"),
    ]
    if sections or sub_reports:
        toc_items.append(("文献调研", "Literature Research"))
    if pi_report:
        toc_items.append(("调研报告草稿", "Research Report Draft"))
    if plan:
        toc_items.append(("实验设计", "Experimental Design"))
    if tool_executions:
        toc_items.append(("实验过程与结果", "Experimental Process & Results"))
    if figure_links:
        toc_items.append(("图表与可视化", "Figures & Visualizations"))
    toc_items.append(("讨论与结论", "Discussion & Conclusion"))

    lines.append("## " + ("目录" if is_zh else "Table of Contents"))
    lines.append("")
    for i, (zh, en) in enumerate(toc_items, 1):
        lines.append(f"{i}. [{zh if is_zh else en}](#{i})")
    lines.append("")

    sec_n = 1

    lines.append(f"## {sec_n}. " + ("蛋白质上下文" if is_zh else "Protein Context"))
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
    sec_n += 1

    lines.append(f"## {sec_n}. " + ("研究背景与用户需求" if is_zh else "Background & User Request"))
    lines.append("")
    if user_text:
        lines.append(redact_path_text(user_text))
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
    sec_n += 1

    if sections or sub_reports:
        lines.append(f"## {sec_n}. " + ("文献调研" if is_zh else "Literature Research"))
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
        sec_n += 1

    if pi_report:
        lines.append(f"## {sec_n}. " + ("调研报告草稿" if is_zh else "Research Report Draft"))
        lines.append("")
        lines.append(redact_path_text(pi_report))
        lines.append("")
        sec_n += 1

    if plan:
        lines.append(f"## {sec_n}. " + ("实验设计" if is_zh else "Experimental Design"))
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
        sec_n += 1

    if tool_executions:
        lines.append(f"## {sec_n}. " + ("实验过程与结果" if is_zh else "Experimental Process & Results"))
        lines.append("")
        for idx, entry in enumerate(tool_executions, 1):
            inputs, outputs, tool, status = _tool_fields(entry)
            step = entry.get("step", idx)
            ts = entry.get("timestamp", "")
            oss_url = entry.get("oss_url")

            step_plan_desc = ""
            for p in plan:
                if str(p.get("step", "")) == str(step):
                    step_plan_desc = p.get("task_description", "")
                    break

            lines.append(f"### Step {step}: `{tool}`")
            lines.append("")
            if status:
                lines.append(f"**{'状态' if is_zh else 'Status'}:** `{status}`")
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
                lines.append(_pretty_jsonish(inputs))
                lines.append("```")
                lines.append("")
            if outputs not in ("", None):
                lines.append(f"**{'输出结果' if is_zh else 'Output'}:**")
                lines.append("")
                lines.append("```")
                lines.append(_pretty_jsonish(outputs))
                lines.append("```")
                lines.append("")
            if oss_url:
                name = os.path.basename(str(oss_url))
                lines.append(f"**{'云端下载' if is_zh else 'Cloud Download'}:** [{name}]({oss_url})")
                lines.append("")
            mls_feedback = _extract_step_feedback_from_history(history, step, tool)
            if mls_feedback:
                lines.append(f"**{'详细反馈' if is_zh else 'Detailed Feedback'}:**")
                lines.append("")
                lines.append(redact_path_text(mls_feedback))
                lines.append("")
            lines.append("---")
            lines.append("")
        sec_n += 1

    if figure_links:
        lines.append(f"## {sec_n}. " + ("图表与可视化" if is_zh else "Figures & Visualizations"))
        lines.append("")
        for i, (fig_name, fig_url) in enumerate(figure_links, 1):
            lines.append(f"### Figure {i}: {fig_name}")
            lines.append("")
            lines.append(f"![{fig_name}]({fig_url})")
            lines.append("")
            lines.append(f"[{'下载' if is_zh else 'Download'}]({fig_url})")
            lines.append("")
        sec_n += 1

    lines.append(f"## {sec_n}. " + ("讨论与结论" if is_zh else "Discussion & Conclusion"))
    lines.append("")

    # Prefer Scientific Critic / PI / last substantial assistant answers.
    preferred_roles = {"scientific_critic", "principal_investigator"}
    final_summaries: list[str] = []
    for item in reversed(history):
        if item.get("role") != "assistant" or _is_thinking_message(item):
            continue
        content = str(item.get("content", "") or "").strip()
        role_id = str(item.get("role_id", "") or "")
        if not content or len(content) < 50:
            continue
        if role_id in preferred_roles or (not role_id and len(content) >= 200):
            # skip checkpoint chrome
            if any(m in content for m in ("请选择下一步", "Please choose", "step_checkpoint", "⏳")):
                continue
            final_summaries.append(content)
            if len(final_summaries) >= 2:
                break

    if not final_summaries:
        # Fallback: any substantial assistant answer
        final_summaries = list(reversed(_collect_assistant_answers(history)[-2:]))

    if final_summaries:
        for summary in reversed(final_summaries):
            lines.append(redact_path_text(summary))
            lines.append("")
    else:
        lines.append("_" + ("暂无总结。" if is_zh else "No conclusion available yet.") + "_")
        lines.append("")

    if state.get("execution_failed"):
        lines.append("### " + ("失败信息" if is_zh else "Failure Details"))
        lines.append("")
        lines.append(f"- **{'失败步骤' if is_zh else 'Failed Step'}:** {state.get('failed_step', 'N/A')}")
        lines.append(f"- **{'失败原因' if is_zh else 'Failure Reason'}:** {state.get('failed_reason', 'Unknown')}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        f"*{'由 VenusFactory Science Expert 自动生成' if is_zh else 'Auto-generated by VenusFactory Science Expert'}* | "
        f"Session `{str(session_id)[:8]}` | {generated_at}"
    )
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
        if not content or _is_thinking_message(item):
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


def _extract_figure_links_from_tool_outputs(
    tool_executions: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    """Best-effort figure discovery from tool output JSON / text."""
    _IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".bmp", ".tiff")
    path_pat = re.compile(
        r"(?:['\"]?)([^\s'\"<>]+?(?:" + "|".join(re.escape(e) for e in _IMAGE_EXTS) + r"))(?:['\"]?)",
        re.IGNORECASE,
    )
    figures: list[tuple[str, str]] = []
    seen: set[str] = set()
    for entry in tool_executions:
        _, outputs, tool, _status = _tool_fields(entry)
        text = _pretty_jsonish(outputs, limit=20000)
        for m in path_pat.finditer(text):
            path = m.group(1)
            if path in seen:
                continue
            # Skip obvious false positives
            if "://" not in path and not path.startswith("/") and "temp_outputs" not in path and "sessions" not in path:
                if not any(path.lower().endswith(ext) for ext in _IMAGE_EXTS):
                    continue
            seen.add(path)
            figures.append((os.path.basename(path) or tool, path))
    return figures
