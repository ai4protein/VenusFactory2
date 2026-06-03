"""SC finalizer nodes: summarize the run and emit iteration prompt."""
from __future__ import annotations

import json
import os
import re

from langchain_core.runnables import RunnableConfig

from agent.chat_agent_utils import _tool_output_indicates_failure
from agent.graph.common.lang import _detect_ui_lang
from agent.graph.common.streaming import _stream_chain
from agent.graph.common.ui_text import _ui_text
from agent.graph.state import AgentState
from logger import get_logger

_logger = get_logger("agent.graph")


async def finalize_start_node(state: AgentState, config: RunnableConfig):
    """Show 'Summarizing' so UI updates before LLM runs."""
    history = list(state.get("history", []))
    ui_lang = state.get("ui_lang") or _detect_ui_lang(state["messages"][-1].content)
    history.append({
        "role": "assistant",
        "content": _ui_text(ui_lang, "summarizing"),
        "role_id": "scientific_critic",
        "phase": "summarizing",
    })
    return {"history": history, "ui_lang": ui_lang}


def _agent_step_failed_or_empty(outputs_raw: str) -> bool:
    """True if an agent_generated_code step had no substantive deliverable."""
    s = str(outputs_raw or "")
    if not s.strip():
        return True
    if '"success": false' in s.lower() or '"status": "error"' in s.lower():
        return True
    # Check output_files contents
    try:
        # Try to parse as JSON to read output_files
        m = re.search(r'\{.*\}', s, re.DOTALL)
        if not m:
            return False
        data = json.loads(m.group())
        ofs = data.get("output_files") or []
        if not isinstance(ofs, list) or not ofs:
            return True
        # Only no_data placeholders count as empty
        if all(isinstance(f, str) and ("no_data" in f.lower() or f.lower().endswith(".txt") and "summary" not in f.lower() and "report" not in f.lower())
               for f in ofs):
            # Heuristic: if every output is a .txt with "no_data" in the name, treat as empty
            if any("no_data" in str(f).lower() for f in ofs):
                return True
    except Exception:
        pass
    return False


def _read_file_summary(path: str, max_chars: int = 1500) -> str:
    """Read a file and return a short, SC-friendly summary string.

    Handles JSON, CSV, TSV, FASTA, and plain text. Truncates content to
    ``max_chars`` and strips long fields so SC can see real structure
    without exploding the prompt.
    """
    try:
        if not path or not os.path.exists(path):
            return f"[missing: {path}]"
        size = os.path.getsize(path)
        suffix = path.rsplit(".", 1)[-1].lower() if "." in path else ""

        if suffix == "json":
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
            if isinstance(data, dict):
                keys = list(data.keys())
                preview = {k: data[k] for k in keys[:15] if not isinstance(data[k], (list, dict))}
                struct = {k: (f"list({len(data[k])})" if isinstance(data[k], list)
                              else f"dict({len(data[k])} keys)") for k in keys[:15]
                          if isinstance(data[k], (list, dict))}
                summary = (
                    f"JSON dict, {len(keys)} top-level keys: {keys[:20]}\n"
                    f"Scalar/string preview: {json.dumps(preview, ensure_ascii=False, default=str)[:600]}\n"
                    f"Nested field shapes: {json.dumps(struct, ensure_ascii=False)[:400]}"
                )
            elif isinstance(data, list):
                summary = (
                    f"JSON list, {len(data)} items\n"
                    f"First item: {json.dumps(data[0], ensure_ascii=False, default=str)[:800]}"
                    if data else "JSON list, empty"
                )
            else:
                summary = f"JSON scalar: {str(data)[:400]}"
        elif suffix in ("csv", "tsv"):
            sep = "\t" if suffix == "tsv" else ","
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                head_lines = [f.readline().rstrip("\n") for _ in range(6)]
            # Count remaining lines
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                row_count = sum(1 for _ in f) - 1  # minus header
            summary = (
                f"{suffix.upper()}, ~{max(0,row_count)} data rows, header:\n"
                + (head_lines[0] if head_lines else "")
                + "\nFirst 5 rows:\n"
                + "\n".join(l for l in head_lines[1:6] if l)
            )
        elif suffix in ("fasta", "fa", "faa"):
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            seq_count = sum(1 for l in lines if l.startswith(">"))
            summary = (
                f"FASTA, {seq_count} sequence(s)\n"
                f"First 4 lines:\n" + "".join(lines[:4])
            )
        else:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(max_chars)
            summary = f"Text file ({size}B), first chars:\n{content}"
        return summary[:max_chars] + ("\n...(truncated)" if len(summary) > max_chars else "")
    except Exception as e:
        return f"[unreadable: {type(e).__name__}: {e}]"


def _force_embed_missing_figures(
    report: str,
    figure_artifacts: list,
    ui_lang: str,
) -> str:
    """Post-process the SC report so every produced figure is embedded.

    The SC prompt asks the LLM to insert ``![title](url)`` inline, but in
    practice the model frequently misses figures (multi-run measurement:
    roughly half are dropped). This helper deterministically appends a
    ``## Figures`` section at the end of the report with the missing
    figures so the user always sees them rendered in the chat panel.
    """
    if not figure_artifacts:
        return report

    embedded_srcs = set()
    # Detect what's already inside the report via Markdown image syntax.
    for m in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", report):
        embedded_srcs.add(m.group(1).strip())

    def _is_present(fa: dict) -> bool:
        # An image counts as embedded if either the OSS URL OR the short
        # path OR the absolute path appears in any ``![...]()`` syntax.
        for candidate in (fa.get("oss_url"), fa.get("short_path"), fa.get("abs_path")):
            if not candidate:
                continue
            for src in embedded_srcs:
                if candidate == src or candidate in src or src in candidate:
                    return True
        return False

    missing = [fa for fa in figure_artifacts if not _is_present(fa)]
    if not missing:
        return report

    is_zh = ui_lang == "zh"
    section_title = "## 图表汇总（自动嵌入）" if is_zh else "## Figures (auto-embedded)"
    intro = (
        "以下图表在本次执行中自动生成；如未在 Results 章节中提及，请将其作为补充材料参考。"
        if is_zh else
        "The following figures were generated during this run; if not already cited in the Results section, treat them as supplementary visualizations."
    )

    blocks = [section_title, "", intro, ""]
    for i, fa in enumerate(missing, 1):
        src = fa.get("oss_url") or fa.get("short_path") or fa.get("abs_path") or ""
        # Build a short, meaningful caption from the file stem
        try:
            stem = (fa.get("short_path") or fa.get("abs_path") or "").rsplit("/", 1)[-1]
            stem = stem.rsplit(".", 1)[0]
            stem = stem.replace("_", " ").replace("-", " ").strip()
        except Exception:
            stem = f"figure {i}"
        caption_word = "图" if is_zh else "Figure"
        from_step = fa.get("step", "?")
        tool_name = fa.get("tool_name", "tool")
        blocks.append(f"![{stem}]({src})")
        if is_zh:
            blocks.append(f"*{caption_word} {i}. {stem}（来自 step {from_step} `{tool_name}`）*")
        else:
            blocks.append(f"*{caption_word} {i}. {stem} (from step {from_step}, `{tool_name}`).*")
        blocks.append("")

    # Insert the section BEFORE References if present, otherwise append.
    refs_pat = re.search(r"(?:^|\n)(##\s*(?:References|参考文献)\s*\n)", report)
    section_text = "\n".join(blocks).rstrip() + "\n\n"
    if refs_pat:
        idx = refs_pat.start(1)
        return report[:idx] + section_text + report[idx:]
    return report.rstrip() + "\n\n" + section_text


async def finalize_node(state: AgentState, config: RunnableConfig):
    """Finalizer node: generates final summary using tool execution history."""
    chains = config.get("configurable", {}).get("chains", {})
    history = list(state.get("history", []))
    tool_executions = state.get("tool_executions", [])
    protein_ctx = state["protein_context"]
    user_input = state["messages"][-1].content
    ui_lang = state.get("ui_lang") or _detect_ui_lang(user_input)

    # Build the full run record for the finalizer
    analysis_log = []
    auto_remedy_blocks: list[str] = []
    figure_artifacts: list[dict] = []  # {step, path, short_path, oss_url, kind}
    IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".pdf", ".svg", ".webp")
    for i, entry in enumerate(tool_executions, 1):
        step = entry.get("step", i)
        tool_name = entry.get("tool_name", "unknown")
        inputs = entry.get("inputs", {})
        outputs = entry.get("outputs", "")
        oss_url = entry.get("oss_url")
        analysis_log.append(
            f"Step {step}: {tool_name}\n"
            f"  Input: {json.dumps(inputs, ensure_ascii=False)}\n"
            f"  Output: {str(outputs)[:2000]}\n"
            + (f"  Cloud Download: {oss_url}" if oss_url else "")
        )

        # Harvest figure-like artifacts produced by this step so SC can embed
        # them directly via Markdown ![title](path) — turning the report into
        # a paper-style figure walkthrough instead of bare file references.
        # Look at both ``output_files`` (agent_generated_code) and ``file_info``
        # (download_string_network_image / pymol_render / etc.).
        try:
            out_text = str(outputs or "")
            m_files = re.search(r'"output_files":\s*\[([^\]]*)\]', out_text)
            candidates: list[str] = []
            if m_files:
                # crude extraction: split on commas, strip quotes/whitespace
                for tok in m_files.group(1).split(","):
                    s = tok.strip().strip("'\"").strip()
                    if s:
                        candidates.append(s)
            m_fp = re.search(r'"file_path":\s*"([^"]+)"', out_text)
            if m_fp:
                candidates.append(m_fp.group(1))
            for p in candidates:
                if not isinstance(p, str):
                    continue
                if not p.lower().endswith(IMAGE_EXTS):
                    continue
                # Normalize relative → absolute, validate exists
                abs_p = p
                if not os.path.isabs(abs_p):
                    abs_p = abs_p.lstrip("./")
                    # try project root prefix
                    pr = "/inspire/hdd/global_user/tanyang-253108120165/workspace/research/VenusFactory"
                    cand = os.path.join(pr, abs_p)
                    if os.path.exists(cand):
                        abs_p = cand
                if not os.path.exists(abs_p):
                    continue
                # Short display path
                short = abs_p
                if "temp_outputs/web_v2/sessions/" in abs_p:
                    try:
                        rel = abs_p.split("temp_outputs/web_v2/sessions/", 1)[1]
                        parts = rel.split("/", 4)
                        if len(parts) >= 5:
                            short = f"~/sessions/{parts[0][:8]}/{parts[4]}"
                    except Exception:
                        pass
                figure_artifacts.append({
                    "step": step,
                    "tool_name": tool_name,
                    "abs_path": abs_p,
                    "short_path": short,
                    "oss_url": oss_url,
                    "kind": p.rsplit(".", 1)[-1].lower(),
                })
        except Exception:
            pass

        # P4 auto-remedy: when agent_generated_code failed or produced only
        # a no-data placeholder, read the upstream file(s) directly and
        # attach a structured summary to the SC context. This eliminates
        # the "建议手动检查 JSON" pattern from reports — the SC will see
        # actual content instead.
        if tool_name == "agent_generated_code" and _agent_step_failed_or_empty(outputs):
            input_files = inputs.get("input_files") or []
            if isinstance(input_files, list):
                summaries = []
                for fp in input_files[:3]:  # cap to 3 files per failed step
                    if not isinstance(fp, str):
                        continue
                    s = _read_file_summary(fp, max_chars=1500)
                    if s:
                        short_fp = fp
                        if "temp_outputs/web_v2/sessions/" in fp:
                            try:
                                rel = fp.split("temp_outputs/web_v2/sessions/", 1)[1]
                                parts = rel.split("/", 4)
                                if len(parts) >= 5:
                                    short_fp = f"~/sessions/{parts[0][:8]}/{parts[4]}"
                            except Exception:
                                pass
                        summaries.append(f"--- {short_fp} ---\n{s}")
                if summaries:
                    auto_remedy_blocks.append(
                        f"AUTO-REMEDY for failed agent_generated_code step {step}: "
                        f"the LLM script could not produce a useful deliverable, "
                        f"so the harness extracted the upstream inputs directly. "
                        f"Use these summaries to write substantive Results entries "
                        f"instead of saying 'manual review recommended'.\n\n"
                        + "\n\n".join(summaries)
                    )

    # Include PI research report and sub-reports so SC has full context
    pi_report = state.get("pi_report", "")
    sub_reports = state.get("research_sub_reports", [])
    sub_reports_text = "\n\n".join(sub_reports) if sub_reports else ""

    record_parts = [f"User request: {user_input}"]
    record_parts.append(f"Protein context: {protein_ctx.get_context_summary()}")
    if pi_report:
        record_parts.append(f"Principal Investigator research report:\n{pi_report}")
    if sub_reports_text:
        record_parts.append(f"Research sub-reports:\n{sub_reports_text}")
    if analysis_log:
        record_parts.append("Tool executions:\n" + "\n".join(analysis_log))
    else:
        record_parts.append("No tools executed.")
    # P4 auto-remedy: when agent_generated_code steps were empty/failed, the
    # block below feeds SC actual content from the upstream files so the
    # report doesn't degrade to "manual review recommended" for fields the
    # harness can read trivially.
    if auto_remedy_blocks:
        record_parts.append(
            "Upstream-file summaries (auto-remedy for failed analysis steps):\n\n"
            + "\n\n".join(auto_remedy_blocks)
        )

    # Figure inventory: feed SC the list of PNG/PDF/SVG files produced
    # during the run so it can embed each one with Markdown
    # ``![title](url)`` in the appropriate Results sub-section. The frontend
    # renders the OSS URL inline; we also include the short path so SC can
    # write a human-readable caption alongside.
    if figure_artifacts:
        lines = ["Figures produced during this run (embed inline in Results sub-sections):"]
        for i, fa in enumerate(figure_artifacts, 1):
            url = fa.get("oss_url") or fa.get("short_path")
            lines.append(
                f"  Figure {i}: step {fa['step']} {fa['tool_name']} → embed as "
                f"`![<short title>]({url})`  "
                f"(caption hint: file = `{fa['short_path']}`)"
            )
        lines.append(
            "\nMANDATORY: in the Results section, immediately after the bullet "
            "or sentence that introduces each result, insert the figure inline "
            "using the exact Markdown above. The src must be the OSS URL (when "
            "present in the inventory) so the chat panel renders the image; if "
            "no OSS URL is listed, fall back to the short path. Every figure in "
            "the inventory must appear inline once — do NOT just list the path "
            "in References."
        )
        record_parts.append("\n".join(lines))
    # Include skipped steps info
    skipped_steps = state.get("skipped_steps", [])
    if skipped_steps:
        record_parts.append(
            f"Skipped steps (failed but no downstream dependencies): {skipped_steps}\n"
            f"  Note: These steps failed but execution continued because no subsequent "
            f"steps depended on their output. Please note which steps were skipped and "
            f"their impact in your report."
        )
    # Include failure context if pipeline failed
    if state.get("execution_failed"):
        failed_step = state.get("failed_step")
        failed_reason = state.get("failed_reason", "Unknown error")
        record_parts.append(
            f"Pipeline failure:\n"
            f"  Failed at step: {failed_step}\n"
            f"  Failure reason: {failed_reason}\n"
            f"  Note: Downstream steps were skipped. Please analyze the failure cause "
            f"and suggest how to resolve it in your report."
        )
    full_run_record = "\n\n".join(record_parts)

    if history and (
        history[-1].get("phase") == "summarizing"
        or "Summarizing" in history[-1].get("content", "")
        or "正在总结" in history[-1].get("content", "")
        or "汇总" in history[-1].get("content", "")
    ):
        history.pop()

    try:
        summary = await _stream_chain(
            chains["finalizer"],
            {
                "input": user_input,
                "full_run_record": full_run_record,
                "original_input": user_input,
                "analysis_log": "\n".join(analysis_log) if analysis_log else "No analysis log available.",
                "references": "",
            },
            role_id="scientific_critic",
        )
        # Q1: deterministically force-embed any figures the SC LLM forgot.
        # We can't rely on the LLM to follow the "MANDATORY inline image"
        # rule (multiple runs proved it skips ≥50% of inventoried figures).
        # Post-process the report: for every figure not already embedded,
        # append a "## Figures (auto-embedded)" section at the end with
        # one image per missing figure and a short caption.
        summary = _force_embed_missing_figures(summary, figure_artifacts, ui_lang)
        history.append({"role": "assistant", "content": summary, "role_id": "scientific_critic"})
    except Exception as e:
        _logger.warning("Finalizer failed: %s", e)
        # Fallback: derive status directly from recorded executions.
        if tool_executions:
            summary_parts = [_ui_text(ui_lang, "final_summary_title")]
            if state.get("execution_failed"):
                summary_parts.append(
                    _ui_text(
                        ui_lang,
                        "pipeline_paused_at",
                        step=state.get("failed_step"),
                        reason=state.get("failed_reason") or ("未知错误" if ui_lang == "zh" else "Unknown error"),
                    )
                )
            else:
                summary_parts.append(_ui_text(ui_lang, "task_completed"))
            for entry in tool_executions:
                output_text = entry.get("outputs", "")
                failed, reason = _tool_output_indicates_failure(output_text)
                if failed:
                    summary_parts.append(
                        _ui_text(
                            ui_lang,
                            "tool_failed",
                            tool=entry.get("tool_name", "Tool"),
                            reason=(reason or ("错误" if ui_lang == "zh" else "error"))[:120],
                        )
                    )
                else:
                    summary_parts.append(_ui_text(ui_lang, "tool_executed", tool=entry.get("tool_name", "Tool")))
            summary = "\n".join(summary_parts)
        else:
            summary = _ui_text(ui_lang, "task_ended")
        history.append({"role": "assistant", "content": summary, "role_id": "scientific_critic"})

    history.append({
        "role": "assistant",
        "content": _ui_text(ui_lang, "iteration_prompt"),
        "role_id": "scientific_critic",
    })
    return {"history": history, "status": "waiting_for_iteration", "waiting_for": "iteration"}
