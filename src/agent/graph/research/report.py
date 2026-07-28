"""PI final research report nodes."""
from __future__ import annotations

import re

from langchain_core.runnables import RunnableConfig

from agent.graph.common.lang import _resolve_ui_lang
from agent.graph.common.streaming import _stream_chain
from agent.graph.state import AgentState


def _extract_embedded_suggest(report: str) -> str:
    """Reuse guidance already present in the draft (avoids a second LLM call)."""
    text = (report or "").strip()
    if not text:
        return ""
    m = re.search(
        r"(?is)^##\s*(Preliminary guidance|Suggested approach|Suggest(?:ed)? steps)\b.*",
        text,
        re.MULTILINE,
    )
    if not m:
        return ""
    return text[m.start():].strip()


def _synthetic_suggest_steps(user_text: str, ui_lang: str) -> str:
    """Cheap fallback for CB when a dedicated suggest-steps LLM is skipped."""
    topic = (user_text or "").strip().replace("\n", " ")
    if len(topic) > 160:
        topic = topic[:160] + "…"
    if ui_lang == "zh":
        return (
            "## Preliminary guidance\n\n"
            f"1. **Suggested capabilities** — 根据研究报告与用户请求（{topic or '…'}），"
            "规划检索/预测/分析所需能力；CB 应从 Available skills 中选择编排 skill"
            "（如 zero_shot_mutation_workflow、protein_structure_pipeline、nature_figure）"
            "并在执行前插入 read_skill。\n"
            "2. **Feasible path** — 按依赖顺序：必要时 read_skill → hub 工具 → 读产物 → 可视化。"
        )
    return (
        "## Preliminary guidance\n\n"
        f"1. **Suggested capabilities** — Based on the research draft and user request "
        f"({topic or '…'}), plan retrieval / prediction / analysis capabilities; CB should "
        "pick orchestration skills from Available skills "
        "(e.g. zero_shot_mutation_workflow, protein_structure_pipeline, nature_figure) "
        "and insert read_skill before execution.\n"
        "2. **Feasible path** — Dependency order: read_skill when needed → hub tools → "
        "read outputs → visualization where useful."
    )


async def research_report_start_node(state: AgentState, config: RunnableConfig):
    """Show 'PI is writing draft report' so UI updates before LLM runs."""
    history = list(state.get("history", []))
    ui_lang = _resolve_ui_lang(state)
    history.append({
        "role": "assistant",
        "content": "✍️ **Principal Investigator** 正在撰写研究草案（摘要、引言、相关工作、参考文献）…"
        if ui_lang == "zh" else
        "✍️ **Principal Investigator** is writing the draft report (Abstract, Introduction, Related Work, References) …",
        "role_id": "principal_investigator",
        "phase": "draft_report_writing",
    })
    return {"history": history}


async def research_report_node(state: AgentState, config: RunnableConfig):
    """PI phase 3: Aggregate sub-reports into final report and suggest steps."""
    chains = config.get("configurable", {}).get("chains", {})
    sub_reports_text = "\n\n".join(state.get("research_sub_reports", []))
    text = state["messages"][-1].content
    ui_lang = _resolve_ui_lang(state, text)
    history = list(state.get("history", []))

    if history and (
        history[-1].get("phase") == "draft_report_writing"
        or "撰写研究草案" in history[-1].get("content", "")
        or "writing the draft report" in history[-1].get("content", "").lower()
    ):
        history.pop()

    try:
        final_report = await _stream_chain(
            chains["pi_final_report"],
            {"input": text, "sub_reports": sub_reports_text},
            role_id="principal_investigator",
        )
        history.append({"role": "assistant", "content": final_report, "role_id": "principal_investigator"})
    except Exception as e:
        final_report = (
            f"生成最终研究报告失败：{e}\n\n{sub_reports_text}"
            if ui_lang == "zh"
            else f"Failed to generate final report: {e}\n\n{sub_reports_text}"
        )
        history.append({"role": "assistant", "content": final_report, "role_id": "principal_investigator"})

    # Prefer existing / embedded / synthetic suggest — avoid a second serial LLM.
    # skip-research never reaches this node; empty prior suggest is normal here.
    existing = (state.get("pi_suggest_steps") or "").strip()
    embedded = _extract_embedded_suggest(final_report)
    if existing:
        suggest_steps = existing
    elif embedded:
        suggest_steps = embedded
    else:
        suggest_steps = _synthetic_suggest_steps(text, ui_lang)

    return {
        "pi_report": final_report,
        "pi_suggest_steps": suggest_steps,
        "history": history,
        "status": "researched",
    }
