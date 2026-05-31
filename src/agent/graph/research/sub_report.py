"""PI per-section sub-report nodes."""
from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from agent.chat_agent_utils import _parse_sub_report_short_title
from agent.graph.common.lang import _detect_ui_lang
from agent.graph.common.streaming import _stream_chain
from agent.graph.common.ui_text import _ui_text
from agent.graph.state import AgentState


async def research_sub_report_start_node(state: AgentState, config: RunnableConfig):
    """PI phase 2b (start): Append 'PI is writing sub-report' so UI updates before LLM runs."""
    research_idx = state.get("research_idx", 0)
    sections = state.get("research_sections", [])
    history = list(state.get("history", []))
    ui_lang = state.get("ui_lang") or _detect_ui_lang(state["messages"][-1].content)
    if research_idx >= len(sections):
        return {"status": "research_steps_done"}
    section = sections[research_idx]
    history.append({
        "role": "assistant",
        "content": f"✍️ **Principal Investigator** 正在撰写小报告：**{section['section_name']}** …"
        if ui_lang == "zh" else
        f"✍️ **Principal Investigator** is writing the sub-report for: **{section['section_name']}** …",
        "role_id": "principal_investigator",
    })
    return {"history": history}


async def research_sub_report_node(state: AgentState, config: RunnableConfig):
    """PI phase 2b: Generate sub-report for the current section after searches are done."""
    chains = config.get("configurable", {}).get("chains", {})
    research_idx = state.get("research_idx", 0)
    sections = state.get("research_sections", [])
    history = list(state.get("history", []))
    ui_lang = state.get("ui_lang") or _detect_ui_lang(state["messages"][-1].content)
    current_search_results = state.get("current_search_results", [])
    sub_reports = list(state.get("research_sub_reports", []))

    if research_idx >= len(sections):
        return {"status": "research_steps_done"}

    section = sections[research_idx]
    # Join all collected results from all queries in this section with sequential numbering [1], [2], ...
    formatted_refs = []
    for i, res_item in enumerate(current_search_results, 1):
        formatted_refs.append(f"[{i}] {res_item}")

    search_results_str = (
        "\n\n".join(formatted_refs)
        if formatted_refs
        else ("该小节没有检索结果。" if ui_lang == "zh" else "No search results for this section.")
    )

    rewrite_comment = state.get("sub_report_rewrite_comment", "")
    rewrite_mode = False
    if rewrite_comment:
        rewrite_mode = True
        prev_sub_report = ""
        for item in reversed(history):
            c = item.get("content", "")
            if item.get("role_id") == "principal_investigator" and c.startswith("# "):
                prev_sub_report = c
                break
        if ui_lang == "zh":
            search_results_str = (
                f"[用户已审阅之前的小报告并提出修改意见]\n\n"
                f"之前的小报告：\n{prev_sub_report}\n\n"
                f"用户修改意见：\n{rewrite_comment}\n\n"
                f"请根据用户的修改意见重写小报告。"
            )
        else:
            search_results_str = (
                f"[User reviewed the previous sub-report and provided revision feedback]\n\n"
                f"Previous sub-report:\n{prev_sub_report}\n\n"
                f"User feedback:\n{rewrite_comment}\n\n"
                f"Please revise the sub-report based on the user's feedback."
            )

    if history and (
        "撰写小报告" in history[-1].get("content", "")
        or "writing sub-report" in history[-1].get("content", "").lower()
    ):
        history.pop()

    try:
        sub_report = await _stream_chain(
            chains["pi_sub_report"],
            {
                "section_name": section["section_name"],
                "focus": section["focus"],
                "search_results": search_results_str,
            },
            role_id="principal_investigator",
        )
        title, body = _parse_sub_report_short_title((sub_report or "").strip(), fallback_title=section["section_name"])
        history.append({"role": "assistant", "content": f"# {title}\n\n{body}", "role_id": "principal_investigator"})
        if rewrite_mode:
            sub_reports.append(f"### {title}\n\n**Sub-report (revised):**\n{body}")
        else:
            sub_reports.append(
                f"### {title}\n\n**References:**\n{search_results_str}\n\n**Sub-report:**\n{body}"
            )
    except Exception as e:
        sub_reports.append(_ui_text(ui_lang, "sub_report_failed", section=section["section_name"], error=str(e)))

    next_research_idx = research_idx + 1
    has_more_sections = next_research_idx < len(sections)

    if has_more_sections:
        remaining = len(sections) - next_research_idx
        history.append({
            "role": "assistant",
            "content": _ui_text(
                ui_lang,
                "sub_report_checkpoint",
                section=section["section_name"],
                remaining=remaining,
            ),
            "role_id": "principal_investigator",
        })
        return {
            "history": history,
            "research_sub_reports": sub_reports,
            "research_idx": next_research_idx,
            "search_idx": 0,
            "current_search_results": [],
            "sub_report_rewrite_comment": "",
            "status": "waiting_for_sub_report_review",
            "waiting_for": "sub_report_review",
        }

    return {
        "history": history,
        "research_sub_reports": sub_reports,
        "research_idx": next_research_idx,
        "search_idx": 0,
        "current_search_results": [],
        "sub_report_rewrite_comment": "",
        "status": "research_step_done",
    }
