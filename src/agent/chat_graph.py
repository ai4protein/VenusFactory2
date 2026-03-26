"""
LangGraph orchestration for the VenusFactory Agent.
Decouples agent logic (PI -> CB -> MLS) from the UI (chat_tab.py).
"""
import json
import os
import time
import asyncio
import shutil
from datetime import datetime
from typing import Dict, Any, List, Optional, TypedDict, Annotated, Sequence

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from agent.chat_agent import (
    Chat_LLM, ProteinContextManager, 
    _merge_tool_parameters_with_context, 
    get_cached_tool_result, save_cached_tool_result
)
from agent.chat_agent_utils import (
    AGENT_CHAT_MAX_TOOL_CALLS, AGENT_CHAT_MAX_MESSAGES,
    PI_SEARCH_TOOL_NAMES, _run_section_search, _parse_cb_plan,
    _parse_sub_report_short_title,
    _tool_output_indicates_failure,
    _run_mls_debug_with_tools, _run_mls_post_step_verify,
    _get_output_file_path_from_raw, _read_output_file_preview,
    _cb_post_step_check, MAX_STEP_RETRIES,
    _extract_image_paths_from_tool_output
)
from web.utils.chat_format_utils import (
    _short_topic_title, _format_search_summary, _format_search_preview
)
from agent.skills import get_skills_metadata_string
from web.utils.file_oss import upload_file_to_cloud_async


class AgentState(TypedDict):
    # Core state
    messages: List[BaseMessage]
    protein_context: ProteinContextManager
    session_id: str
    agent_session_dir: str
    
    # Process management
    pi_report: str
    pi_suggest_steps: str
    plan: List[Dict[str, Any]]
    current_step_index: int
    step_results: Dict[int, Any]
    
    # UI compatibility (for yielding partial updates)
    history: List[Dict[str, Any]]
    conversation_log: List[Dict[str, Any]]
    tool_executions: List[Dict[str, Any]]
    
    # Internal research state
    research_sections: List[Dict[str, Any]]
    research_idx: int
    search_idx: int
    current_search_results: List[str]
    research_sub_reports: List[str]
    
    # Control flags
    status: str
    error: Optional[str]


def _parse_sections(raw: str) -> List[Dict[str, Any]]:
    try:
        raw = raw.strip()
        if "```" in raw:
            start = raw.find("```")
            if "json" in raw[: start + 10]:
                start = raw.find("```") + 7
            else:
                start = raw.find("```") + 3
            end = raw.find("```", start)
            raw = raw[start:end if end > 0 else None].strip()
        i = raw.find("[")
        if i >= 0:
            depth = 0
            for j in range(i, len(raw)):
                if raw[j] == "[":
                    depth += 1
                elif raw[j] == "]":
                    depth -= 1
                    if depth == 0:
                        raw = raw[i : j + 1]
                        break
        sections_list = json.loads(raw)
        if not isinstance(sections_list, list): return []
        parsed = []
        for s in sections_list[:5]:
            sec_name = (s.get("section_name") or "Section").strip() or "Section"
            focus = (s.get("focus") or "both").strip() or "both"
            queries = s.get("search_queries") or s.get("search_query") or []
            if isinstance(queries, str): queries = [queries]
            elif not isinstance(queries, list): queries = [""]
            parsed.append({"section_name": sec_name, "search_queries": queries, "focus": focus})
        return parsed
    except:
        return []


async def research_plan_start_node(state: AgentState, config: RunnableConfig):
    """Initial node - passes through to research_plan_node for decision."""
    # Just pass through, research_plan_node will decide the path
    return {"status": "analyzing"}


async def research_plan_node(state: AgentState, config: RunnableConfig):
    """PI phase 1: Create a research plan with sections."""
    chains = config.get("configurable", {}).get("chains", {})
    protein_ctx = state["protein_context"]
    text = state["messages"][-1].content
    protein_context_summary = protein_ctx.get_context_summary()
    history = list(state.get("history", []))

    try:
        sections_out = await asyncio.to_thread(
            chains["pi_sections"].invoke,
            {"input": text, "protein_context_summary": protein_context_summary},
        )
        sections_list = _parse_sections(sections_out)
    except Exception as e:
        print(f"[PI sections] failed: {e}")
        sections_list = []

    # No research sections needed - skip directly to planning (CB will decide if tools are needed)
    if not sections_list:
        return {"status": "research_skipped", "pi_report": "", "pi_suggest_steps": ""}

    # This is a research request - show the analyzing message
    history.append({
        "role": "assistant",
        "content": "🤔 **Principal Investigator** is analyzing your request and creating a research plan...",
        "role_id": "principal_investigator",
    })

    return {
        "research_sections": sections_list,
        "research_idx": 0,
        "search_idx": 0,
        "current_search_results": [],
        "research_sub_reports": [],
        "history": history,
        "status": "research_planning_done"
    }


async def chat_start_node(state: AgentState, config: RunnableConfig):
    """Show 'PI is responding' for simple chat/greeting inputs."""
    history = list(state.get("history", []))
    history.append({
        "role": "assistant",
        "content": "🤔 Thinking...",
        "role_id": "principal_investigator",
    })
    return {"history": history}


async def chat_node(state: AgentState, config: RunnableConfig):
    """Chat mode: use pi_chat_chain for direct responses (greetings, simple questions)."""
    chains = config.get("configurable", {}).get("chains", {})
    text = state["messages"][-1].content
    history = list(state.get("history", []))

    try:
        # Use pi_chat chain for direct chat response (greetings, simple questions)
        response = await asyncio.to_thread(
            chains["pi_chat"].invoke,
            {"input": text, "chat_history": []},
        )
        # Remove the "Thinking..." placeholder before adding the actual response
        if history and "Thinking" in history[-1].get("content", ""):
            history.pop()
        history.append({"role": "assistant", "content": response, "role_id": "principal_investigator"})
    except Exception as e:
        # Remove the "Thinking..." placeholder before adding the error
        if history and "Thinking" in history[-1].get("content", ""):
            history.pop()
        history.append({"role": "assistant", "content": f"I apologize, but I encountered an error: {str(e)}", "role_id": "principal_investigator"})

    return {"history": history, "status": "completed"}


async def research_search_start_node(state: AgentState, config: RunnableConfig):
    """Show 'PI is searching' so UI updates before search runs."""
    research_idx = state.get("research_idx", 0)
    search_idx = state.get("search_idx", 0)
    sections = state.get("research_sections", [])
    history = list(state.get("history", []))
    if research_idx >= len(sections):
        return {"status": "research_steps_done"}
    section = sections[research_idx]
    queries = section.get("search_queries", [])
    if search_idx >= len(queries):
        return {}
    sq = (queries[search_idx] or "").strip()[:80]
    if sq and len(sq) == 80:
        sq = sq + "…"
    history.append({
        "role": "assistant",
        "content": f"🔍 **Principal Investigator** is searching: **{sq or '…'}** …",
        "role_id": "principal_investigator",
    })
    return {"history": history}


async def research_search_node(state: AgentState, config: RunnableConfig):
    """PI phase 2a: Process ONE search query from the current section."""
    chains = config.get("configurable", {}).get("chains", {})
    protein_ctx = state["protein_context"]
    research_idx = state.get("research_idx", 0)
    search_idx = state.get("search_idx", 0)
    sections = state.get("research_sections", [])
    history = list(state.get("history", []))
    executions = list(state.get("tool_executions", []))
    current_search_results = list(state.get("current_search_results", []))
    
    if research_idx >= len(sections) or len(protein_ctx.tool_history) >= AGENT_CHAT_MAX_TOOL_CALLS:
        return {"status": "research_steps_done"}

    section = sections[research_idx]
    queries = section.get("search_queries", [])
    
    if search_idx == 0:
        history.append({"role": "assistant", "content": f"**Section {research_idx + 1}:** {section['section_name']}", "role_id": "principal_investigator"})

    if search_idx < len(queries):
        sq = queries[search_idx]
        search_results_list, search_logged = await asyncio.to_thread(_run_section_search, sq)
        current_search_results.extend(search_results_list)
        
        for tname, tinputs, toutputs in search_logged:
            step_off = len(protein_ctx.tool_history) + 1
            protein_ctx.add_tool_call(step_off, tname, tinputs, toutputs, cached=False)
            executions.append({
                "step": step_off, "tool_name": tname, "inputs": tinputs,
                "outputs": (str(toutputs)[:1000] + "..." if len(str(toutputs)) > 1000 else str(toutputs)),
                "timestamp": datetime.now().isoformat(),
            })
            summary_msg = _format_search_summary(tname, tinputs, str(toutputs))
            preview = _format_search_preview(tname, str(toutputs))
            history.append({"role": "assistant", "content": summary_msg + ("\n\n" + preview if preview else ""), "role_id": "principal_investigator"})
    
    return {
        "history": history,
        "tool_executions": executions,
        "current_search_results": current_search_results,
        "search_idx": search_idx + 1,
        "status": "research_search_done"
    }


async def research_sub_report_start_node(state: AgentState, config: RunnableConfig):
    """PI phase 2b (start): Append 'PI is writing sub-report' so UI updates before LLM runs."""
    research_idx = state.get("research_idx", 0)
    sections = state.get("research_sections", [])
    history = list(state.get("history", []))
    if research_idx >= len(sections):
        return {"status": "research_steps_done"}
    section = sections[research_idx]
    history.append({
        "role": "assistant",
        "content": f"✍️ **Principal Investigator** is writing the sub-report for: **{section['section_name']}** …",
        "role_id": "principal_investigator",
    })
    return {"history": history}


async def research_sub_report_node(state: AgentState, config: RunnableConfig):
    """PI phase 2b: Generate sub-report for the current section after searches are done."""
    chains = config.get("configurable", {}).get("chains", {})
    research_idx = state.get("research_idx", 0)
    sections = state.get("research_sections", [])
    history = list(state.get("history", []))
    current_search_results = state.get("current_search_results", [])
    sub_reports = list(state.get("research_sub_reports", []))
    
    if research_idx >= len(sections):
        return {"status": "research_steps_done"}

    section = sections[research_idx]
    # Join all collected results from all queries in this section with sequential numbering [1], [2], ...
    formatted_refs = []
    for i, res_item in enumerate(current_search_results, 1):
        formatted_refs.append(f"[{i}] {res_item}")
    
    search_results_str = "\n\n".join(formatted_refs) if formatted_refs else "No search results for this section."

    try:
        sub_report = await asyncio.to_thread(
            chains["pi_sub_report"].invoke,
            {"section_name": section["section_name"], "focus": section["focus"], "search_results": search_results_str},
        )
        title, body = _parse_sub_report_short_title((sub_report or "").strip(), fallback_title=section["section_name"])
        history.append({"role": "assistant", "content": f"# {title}\n\n{body}", "role_id": "principal_investigator"})
        sub_reports.append(f"### {title}\n\n**References:**\n{search_results_str}\n\n**Sub-report:**\n{body}")
    except Exception as e:
        sub_reports.append(f"(Sub-report failed for {section['section_name']}: {e})")

    return {
        "history": history,
        "research_sub_reports": sub_reports,
        "research_idx": research_idx + 1,
        "search_idx": 0,
        "current_search_results": [],
        "status": "research_step_done"
    }


async def research_report_start_node(state: AgentState, config: RunnableConfig):
    """Show 'PI is writing draft report' so UI updates before LLM runs."""
    history = list(state.get("history", []))
    history.append({
        "role": "assistant",
        "content": "✍️ **Principal Investigator** is writing the draft report (Abstract, Introduction, Related Work, References) …",
        "role_id": "principal_investigator",
    })
    return {"history": history}


async def research_report_node(state: AgentState, config: RunnableConfig):
    """PI phase 3: Aggregate sub-reports into final report and suggest steps."""
    chains = config.get("configurable", {}).get("chains", {})
    sub_reports_text = "\n\n".join(state.get("research_sub_reports", []))
    text = state["messages"][-1].content
    history = list(state.get("history", []))

    try:
        final_report = await asyncio.to_thread(
            chains["pi_final_report"].invoke,
            {"input": text, "sub_reports": sub_reports_text},
        )
        history.append({"role": "assistant", "content": final_report, "role_id": "principal_investigator"})
    except Exception as e:
        final_report = f"Failed to generate final report: {e}\n\n{sub_reports_text}"
        history.append({"role": "assistant", "content": final_report, "role_id": "principal_investigator"})

    try:
        suggest_steps = await asyncio.to_thread(
            chains["pi_suggest_steps"].invoke,
            {"draft_report": final_report, "input": text},
        )
    except:
        suggest_steps = "Execute basic analysis."

    return {
        "pi_report": final_report,
        "pi_suggest_steps": suggest_steps,
        "history": history,
        "status": "researched"
    }


async def plan_start_node(state: AgentState, config: RunnableConfig):
    """Show 'CB is designing pipeline' so UI updates before LLM runs."""
    history = list(state.get("history", []))
    history.append({
        "role": "assistant",
        "content": "📋 **Computational Biologist** is designing the pipeline …",
        "role_id": "computational_biologist",
    })
    return {"history": history}


async def plan_node(state: AgentState, config: RunnableConfig):
    """CB planning node: PI report -> pipeline (JSON list)."""
    chains = config.get("configurable", {}).get("chains", {})
    pi_report = state.get("pi_report", "")
    pi_suggest_steps = state.get("pi_suggest_steps", "")
    protein_ctx = state["protein_context"]
    history = list(state.get("history", []))
    log_entries = list(state.get("conversation_log", []))

    # When research is skipped, use user's original input as the "PI report"
    if not pi_report:
        user_input = state["messages"][-1].content
        pi_report = f"User request: {user_input}\n\nNo literature research needed. Proceed directly with tool execution."
        pi_suggest_steps = pi_suggest_steps or "Execute the appropriate tool to fulfill the user's request."

    context_parts = [f"Protein context: {protein_ctx.get_context_summary()}"]
    if state.get("agent_session_dir"):
        context_parts.append(f"Default output directory: {state['agent_session_dir']}")
    protein_context_summary = "; ".join(context_parts)

    recent_tool_calls = protein_ctx.get_tool_records(limit=10)
    tool_outputs_summary = json.dumps(recent_tool_calls, ensure_ascii=False)

    # Pass tools and skills explicitly so CB always sees the full list (session_state may be the only source)
    tools_description = chains.get("tools_description") or ""
    skills_metadata = chains.get("skills_metadata") or get_skills_metadata_string()
    available_tools_list = chains.get("available_tools_list") or ""
    if not available_tools_list and chains.get("workers"):
        available_tools_list = ", ".join(chains["workers"].keys())

    cb_planner_inputs = {
        "pi_report": pi_report,
        "pi_suggest_steps": pi_suggest_steps,
        "protein_context_summary": protein_context_summary,
        "tool_outputs": tool_outputs_summary,
        "tools_description": tools_description,
        "skills_metadata": skills_metadata,
        "available_tools_list": available_tools_list,
    }

    try:
        raw_msg = await asyncio.to_thread(chains["cb_planner_raw"].invoke, cb_planner_inputs)
        content = getattr(raw_msg, "content", None) or str(raw_msg) or ""
        plan = _parse_cb_plan(content)
    except Exception as e:
        print(f"[CB planner] failed: {e}")
        plan = []

    # Filter and normalize plan
    normalized_plan = []
    for i, p in enumerate(plan):
        if not isinstance(p, dict): continue
        tname = p.get("tool_name") or p.get("tool") or ""
        if not tname or tname in PI_SEARCH_TOOL_NAMES: continue
        normalized_plan.append({
            "step": p.get("step") or (i + 1),
            "task_description": p.get("task_description") or p.get("task") or "",
            "tool_name": tname.strip(),
            "tool_input": p.get("tool_input") or p.get("input") or {}
        })

    if not normalized_plan:
        # No tools needed - this might be a greeting or simple question
        # Route to chat mode for a natural response
        return {"plan": [], "history": history, "status": "chat_mode"}

    step_lines = [f"**Step {p['step']}.** {p['task_description']}" for p in normalized_plan]
    plan_text = "📋 **Pipeline**\n\nHere's what we'll do:\n\n" + "\n\n".join(step_lines)
    history.append({"role": "assistant", "content": plan_text, "role_id": "computational_biologist"})
    log_entries.append({"role": "assistant", "content": plan_text, "role_id": "computational_biologist", "timestamp": datetime.now().isoformat()})

    return {
        "plan": normalized_plan,
        "current_step_index": 0,
        "step_results": {},
        "history": history,
        "conversation_log": log_entries,
        "status": "planned"
    }


async def execute_start_node(state: AgentState, config: RunnableConfig):
    """Show 'MLS is executing step N' so UI updates before tool runs."""
    plan = state.get("plan", [])
    idx = state.get("current_step_index", 0)
    history = list(state.get("history", []))
    if idx >= len(plan):
        return {}
    step = plan[idx]
    step_num = step.get("step", idx + 1)
    task_desc = step.get("task_description", "…")
    history.append({
        "role": "assistant",
        "content": f"⏳ **Machine Learning Specialist** is executing Step {step_num}: {task_desc} …",
        "role_id": "machine_learning_specialist",
    })
    return {"history": history}


async def execute_node(state: AgentState, config: RunnableConfig):
    """MLS execution node: executes current step in plan."""
    chains = config.get("configurable", {}).get("chains", {})
    plan = state["plan"]
    idx = state["current_step_index"]
    step = plan[idx]
    protein_ctx = state["protein_context"]
    history = list(state.get("history", []))
    log_entries = list(state.get("conversation_log", []))
    executions = list(state.get("tool_executions", []))
    step_results = dict(state.get("step_results", {}))

    step_num = step["step"]
    task_desc = step["task_description"]
    tool_name = step["tool_name"]
    tool_input = step["tool_input"]

    # Resolve dependencies
    merged_tool_input = _merge_tool_parameters_with_context(protein_ctx, tool_input)
    for key, value in list(merged_tool_input.items()):
        if isinstance(value, str) and value.startswith("dependency:"):
            try:
                parts = value.split(':')
                dep_step = int(parts[1].replace('step_', '').replace('step', ''))
                dep_out = step_results.get(dep_step, {}).get("raw_output")
                if dep_out:
                    parsed = json.loads(dep_out) if isinstance(dep_out, str) and dep_out.strip().startswith("{") else dep_out
                    
                    if len(parts) > 2:
                        field = parts[2]
                        if isinstance(parsed, dict) and field in parsed:
                            val = parsed[field]
                        else:
                            val = dep_out
                    else:
                        val = dep_out
                    
                    # Heuristic auto-extraction for paths if the expected parameter is a file or path
                    if any(k in key.lower() for k in ("path", "file")):
                        if isinstance(val, dict):
                            if "file_path" in val:
                                val = val["file_path"]
                            elif "file_info" in val and isinstance(val["file_info"], dict) and "file_path" in val["file_info"]:
                                val = val["file_info"]["file_path"]
                        elif isinstance(val, str) and val.strip().startswith("{"):
                            extracted = _get_output_file_path_from_raw(val, "previous_step")
                            if extracted:
                                val = extracted
                        
                        # Fallback: if we still have the full JSON string but needed a file, try extracting from dep_out
                        if val == dep_out and isinstance(dep_out, str) and dep_out.strip().startswith("{"):
                            extracted = _get_output_file_path_from_raw(dep_out, "previous_step")
                            if extracted: val = extracted

                    merged_tool_input[key] = val
            except: pass

    # Execution Loop with Retries
    worker = chains["workers"].get(tool_name)
    step_retry = 0
    step_done = False
    last_output = None
    cached_flag = False  # Default value

    # Get the actual tool for direct invocation
    tool = next((t for t in chains["all_tools"] if t.name == tool_name), None)

    while step_retry <= MAX_STEP_RETRIES and not step_done:
        # Cache check
        cached = get_cached_tool_result({"tool_cache": state.get("tool_cache", {})}, tool_name, merged_tool_input)
        if cached:
            raw_output = cached["outputs"]
            cached_flag = True
            step_done = True
            last_output = raw_output
        elif tool:
            # Direct tool invocation (simpler and more reliable)
            try:
                inputs_str = json.dumps(merged_tool_input, ensure_ascii=False, sort_keys=True)
                if len(inputs_str) > 500:
                    inputs_str = inputs_str[:500] + "..."
                print(f"[Execute] tool={tool_name} | input={inputs_str}")
                out = await asyncio.wait_for(asyncio.to_thread(tool.invoke, merged_tool_input), timeout=300)
                raw_output = out if isinstance(out, (str, dict)) else str(out)
                out_preview = str(raw_output)[:300] + ("..." if len(str(raw_output)) > 300 else "")
                print(f"[Result] tool={tool_name} | output_preview={out_preview}")
                save_cached_tool_result(state, tool_name, merged_tool_input, raw_output)
                cached_flag = False
                step_done = True
                last_output = raw_output
            except asyncio.TimeoutError:
                raw_output = json.dumps({"success": False, "error": "Tool execution timed out (300s)"})
                print(f"[Result] tool={tool_name} | timeout after 300s")
                cached_flag = False
                last_output = raw_output
                step_done = True  # Don't retry on timeout
            except Exception as e:
                raw_output = json.dumps({"success": False, "error": str(e)})
                print(f"[Result] tool={tool_name} | exception={e}")
                cached_flag = False
                step_retry += 1
                if step_retry > MAX_STEP_RETRIES:
                    last_output = raw_output
                    step_done = True
        else:
            # Tool not found
            last_output = json.dumps({"success": False, "error": f"Unknown tool: {tool_name}"})
            step_done = True
            cached_flag = False

    # Record result
    protein_ctx.add_tool_call(step_num, tool_name, merged_tool_input, last_output, cached=cached_flag)
    step_results[step_num] = {"raw_output": last_output}
    
    # --- Generate detailed feedback for the UI (MLS needs full context for self-check) ---
    try:
        out_data = json.loads(last_output) if isinstance(last_output, str) else last_output
        is_failure = isinstance(out_data, dict) and out_data.get("success") is False
    except Exception:
        out_data = None
        is_failure = True

    if is_failure:
        feedback_content = f"❌ **Step {step_num} failed.**\n\n"
    else:
        feedback_content = f"✅ **Step {step_num} Complete.**\n\n"

    # 1. Output summary (include error-related keys so MLS can self-check)
    try:
        if isinstance(out_data, dict):
            important_keys = [
                "success", "error", "message", "detail", "traceback", "tool_name",
                "protein_id", "pdb_id", "uniprot_id", "mutation", "delta_delta_g"
            ]
            summary_parts = []
            for k in important_keys:
                if k in out_data:
                    val = out_data[k]
                    summary_parts.append(f"**{k}:** {val}")
            if summary_parts:
                feedback_content += "Summary: " + ", ".join(summary_parts) + "\n\n"
            else:
                dump = json.dumps(out_data, ensure_ascii=False)
                feedback_content += f"Output: `{dump[:300]}...`\n\n"
        else:
            feedback_content += f"Output Preview: `{str(last_output)[:300]}...`\n\n"
    except Exception:
        feedback_content += f"Output Preview: `{str(last_output)[:300]}...`\n\n"

    # When failed or non-JSON: always show raw output for MLS self-check
    raw_str = last_output if isinstance(last_output, str) else json.dumps(last_output, ensure_ascii=False)
    if is_failure or not isinstance(out_data, dict):
        feedback_content += "**Raw output (for debugging/self-check):**\n```\n"
        feedback_content += raw_str[:2000] + ("\n...(truncated)" if len(raw_str) > 2000 else "")
        feedback_content += "\n```\n\n"

    # 2. File Hosting (OSS)
    out_file = _get_output_file_path_from_raw(last_output, tool_name)
    oss_url = None
    if out_file:
        try:
            oss_url = await upload_file_to_cloud_async(out_file)
            if oss_url:
                feedback_content += f"📎 **Cloud Download:** [{os.path.basename(out_file)}]({oss_url})\n\n"
        except Exception as e:
            print(f"OSS upload failed for {out_file}: {e}")

        # 3. File Preview
        preview = _read_output_file_preview(out_file)
        if preview:
            feedback_content += f"**File Preview ({os.path.basename(out_file)}):**\n```\n{preview}\n```\n\n"

    executions.append({
        "step": step_num, "tool_name": tool_name, "inputs": merged_tool_input,
        "outputs": (str(last_output)[:1000] + "..." if len(str(last_output)) > 1000 else str(last_output)),
        "oss_url": oss_url,
        "timestamp": datetime.now().isoformat(),
    })

    # 4. Image Hosting (plots, images)
    try:
        img_paths = _extract_image_paths_from_tool_output(last_output, tool_name)
        for ip in img_paths:
            if ip != out_file: # Avoid duplicate link
                oss_img_url = await upload_file_to_cloud_async(ip)
                if oss_img_url:
                    feedback_content += f"🖼️ **Generated Image:** [{os.path.basename(ip)}]({oss_img_url})\n\n"
    except Exception:
        pass

    history.append({"role": "assistant", "content": feedback_content, "role_id": "machine_learning_specialist"})

    return {
        "current_step_index": idx + 1,
        "step_results": step_results,
        "history": history,
        "conversation_log": log_entries,
        "tool_executions": executions,
        "status": "executing"
    }


async def finalize_start_node(state: AgentState, config: RunnableConfig):
    """Show 'Summarizing' so UI updates before LLM runs."""
    history = list(state.get("history", []))
    history.append({
        "role": "assistant",
        "content": "📝 **Principal Investigator** is summarizing the results …",
        "role_id": "principal_investigator",
    })
    return {"history": history}


async def finalize_node(state: AgentState, config: RunnableConfig):
    """Finalizer node: generates final summary using tool execution history."""
    chains = config.get("configurable", {}).get("chains", {})
    history = list(state.get("history", []))
    tool_executions = state.get("tool_executions", [])
    protein_ctx = state["protein_context"]
    user_input = state["messages"][-1].content

    # Build the full run record for the finalizer
    analysis_log = []
    for i, entry in enumerate(tool_executions, 1):
        step = entry.get("step", i)
        tool_name = entry.get("tool_name", "unknown")
        inputs = entry.get("inputs", {})
        outputs = entry.get("outputs", "")
        oss_url = entry.get("oss_url")
        analysis_log.append(
            f"Step {step}: {tool_name}\n"
            f"  Input: {json.dumps(inputs, ensure_ascii=False)}\n"
            f"  Output: {str(outputs)[:500]}\n"
            + (f"  Cloud Download: {oss_url}" if oss_url else "")
        )

    full_run_record = "\n\n".join([
        f"User request: {user_input}",
        f"Protein context: {protein_ctx.get_context_summary()}",
        f"Tool executions:\n" + "\n".join(analysis_log) if analysis_log else "No tools executed."
    ])

    try:
        summary = await asyncio.to_thread(
            chains["finalizer"].invoke,
            {
                "input": user_input,
                "full_run_record": full_run_record,
                "original_input": user_input,
                "analysis_log": "\n".join(analysis_log) if analysis_log else "No analysis log available.",
                "references": ""
            }
        )
        history.append({"role": "assistant", "content": summary, "role_id": "principal_investigator"})
    except Exception as e:
        print(f"[Finalizer] failed: {e}")
        # Fallback: generate a simple summary from tool executions
        if tool_executions:
            summary_parts = ["## Summary\n\nTask completed. Here's what was done:\n"]
            for entry in tool_executions:
                summary_parts.append(f"- **{entry.get('tool_name', 'Tool')}**: Executed successfully")
            summary = "\n".join(summary_parts)
        else:
            summary = "Task completed. See results above."
        history.append({"role": "assistant", "content": summary, "role_id": "principal_investigator"})

    return {"history": history, "status": "completed"}


def should_continue_research(state: AgentState):
    if state.get("status") == "planning_failed" or state.get("error"):
        return END
    
    research_idx = state.get("research_idx", 0)
    search_idx = state.get("search_idx", 0)
    sections = state.get("research_sections", [])
    
    if research_idx < len(sections):
        section = sections[research_idx]
        queries = section.get("search_queries", [])
        if search_idx < len(queries):
            return "research_search_start_node"
        else:
            return "research_sub_report_start_node"
    return "research_report_start_node"


def should_continue(state: AgentState):
    if state.get("status") == "planning_failed" or state.get("error"):
        return END
    
    plan = state.get("plan", [])
    current_idx = state.get("current_step_index", 0)
    
    if current_idx < len(plan):
        return "execute_start_node"
    return "finalize_start_node"


def create_agent_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("research_plan_start_node", research_plan_start_node)
    workflow.add_node("research_plan_node", research_plan_node)
    workflow.add_node("chat_start_node", chat_start_node)
    workflow.add_node("chat_node", chat_node)
    workflow.add_node("research_search_start_node", research_search_start_node)
    workflow.add_node("research_search_node", research_search_node)
    workflow.add_node("research_sub_report_start_node", research_sub_report_start_node)
    workflow.add_node("research_sub_report_node", research_sub_report_node)
    workflow.add_node("research_report_start_node", research_report_start_node)
    workflow.add_node("research_report_node", research_report_node)
    workflow.add_node("plan_start_node", plan_start_node)
    workflow.add_node("plan_node", plan_node)
    workflow.add_node("execute_start_node", execute_start_node)
    workflow.add_node("execute_node", execute_node)
    workflow.add_node("finalize_start_node", finalize_start_node)
    workflow.add_node("finalize_node", finalize_node)

    workflow.add_edge(START, "research_plan_start_node")
    workflow.add_edge("research_plan_start_node", "research_plan_node")
    workflow.add_conditional_edges(
        "research_plan_node",
        lambda s: "plan_start_node" if s.get("status") in ("research_skipped", "chat_mode") else "research_search_start_node",
    )
    # Chat mode nodes (for true greetings - currently not used, research_skipped goes to planning)
    workflow.add_edge("chat_start_node", "chat_node")
    workflow.add_edge("chat_node", END)
    workflow.add_edge("research_search_start_node", "research_search_node")
    workflow.add_conditional_edges("research_search_node", should_continue_research)
    workflow.add_edge("research_sub_report_start_node", "research_sub_report_node")
    workflow.add_edge("research_sub_report_node", "research_search_start_node")
    workflow.add_edge("research_report_start_node", "research_report_node")
    workflow.add_edge("research_report_node", "plan_start_node")
    workflow.add_edge("plan_start_node", "plan_node")
    workflow.add_conditional_edges(
        "plan_node",
        lambda s: "chat_start_node" if s.get("status") == "chat_mode" else ("finalize_start_node" if s.get("status") == "planning_failed" or not s.get("plan") else "execute_start_node"),
    )
    workflow.add_edge("execute_start_node", "execute_node")
    workflow.add_conditional_edges("execute_node", should_continue)
    workflow.add_edge("finalize_start_node", "finalize_node")
    workflow.add_edge("finalize_node", END)

    return workflow.compile()
