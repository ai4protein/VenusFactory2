"""PI final research report nodes."""
from __future__ import annotations

import asyncio

from langchain_core.runnables import RunnableConfig

from agent.graph.common.lang import _detect_ui_lang
from agent.graph.common.streaming import _stream_chain
from agent.graph.state import AgentState


async def research_report_start_node(state: AgentState, config: RunnableConfig):
    """Show 'PI is writing draft report' so UI updates before LLM runs."""
    history = list(state.get("history", []))
    ui_lang = state.get("ui_lang") or _detect_ui_lang(state["messages"][-1].content)
    history.append({
        "role": "assistant",
        "content": "✍️ **Principal Investigator** 正在撰写研究草案（摘要、引言、相关工作、参考文献）…"
        if ui_lang == "zh" else
        "✍️ **Principal Investigator** is writing the draft report (Abstract, Introduction, Related Work, References) …",
        "role_id": "principal_investigator",
    })
    return {"history": history}


async def research_report_node(state: AgentState, config: RunnableConfig):
    """PI phase 3: Aggregate sub-reports into final report and suggest steps."""
    chains = config.get("configurable", {}).get("chains", {})
    sub_reports_text = "\n\n".join(state.get("research_sub_reports", []))
    text = state["messages"][-1].content
    ui_lang = state.get("ui_lang") or _detect_ui_lang(text)
    history = list(state.get("history", []))

    if history and (
        "撰写研究草案" in history[-1].get("content", "")
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

    try:
        suggest_steps = await asyncio.to_thread(
            chains["pi_suggest_steps_chain"].invoke,
            {"draft_report": final_report, "input": text},
        )
    except Exception:
        suggest_steps = "执行基础分析。" if ui_lang == "zh" else "Execute basic analysis."

    return {
        "pi_report": final_report,
        "pi_suggest_steps": suggest_steps,
        "history": history,
        "status": "researched",
    }
