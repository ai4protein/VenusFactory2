import json
import os
import re
import aiohttp
import asyncio
import base64
import hashlib
import tempfile
import shutil
import time
import uuid
import numpy as np
import pandas as pd
import gradio as gr
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Mapping
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langchain_classic.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage, LLMResult
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_classic.schema.runnable import RunnablePassthrough
from langchain_classic.callbacks.base import BaseCallbackHandler
from langchain_core.prompt_values import ChatPromptValue
from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks import CallbackManagerForLLMRun

from pathlib import Path
from dotenv import load_dotenv
from gradio_client import Client, handle_file
from tools.tools_agent_hub import *
from agent.prompts import get_role_avatar_path, ROLE_DISPLAY_NAMES, MLS_SELF_CHECK_TEMPLATE
from agent.skills import get_skills_metadata_string, list_skill_ids
from agent.chat_agent import (
    initialize_session_state,
    update_llm_model,
    get_cached_tool_result,
    save_cached_tool_result,
    _merge_tool_parameters_with_context,
    _build_tools_description,
)
from web.utils.css_loader import load_css_file
import threading

from web.utils.chat_format_utils import (
    _markdown_to_html, _embed_image_paths_in_content, _get_download_btn_update, _assistant_msg,
    _avatar_data_url, _build_chat_html, _format_literature_citations, _format_literature_for_reading,
    _format_web_citations, _format_web_for_reading, _format_dataset_citations, _format_pi_steps_for_report,
    _clean_title, _short_topic_title, _format_search_preview, _format_search_summary,
    _format_mls_code_for_display, _format_backend_log, _build_conversation_panel_html, _build_log_html
)

from agent.chat_agent_utils import (
    AGENT_CHAT_MAX_MESSAGES, AGENT_CHAT_MAX_TOOL_CALLS, SEARCH_MAX_RESULTS,
    MLS_SELF_CHECK_MSG, MLS_POST_STEP_SELF_CHECK_MSG, MAX_STEP_RETRIES, TOOL_EXECUTION_TIMEOUT,
    PI_SEARCH_TOOL_NAMES, _tool_output_indicates_failure, _dedupe_references,
    extract_sequence_from_message, extract_uniprot_id_from_message, _extract_download_file_from_output,
    _get_output_file_path_from_raw, _read_output_file_preview, _extract_image_paths_from_tool_output,
    _fetch_literature_for_pi, _refine_query_for_search, _mls_debug_step, _parse_mls_debug_output,
    _run_mls_debug_executor_sync, _run_mls_debug_with_tools, _parse_mls_post_step_output,
    _run_mls_post_step_verify, _output_looks_null_or_empty, _cb_post_step_check, _try_parse_json_array,
    _find_json_array_end, _parse_cb_plan, _parse_sub_report_short_title, _extract_deepsearch_data,
    _is_search_result_empty, _translate_user_query_to_english, _run_section_search, _fetch_search_for_pi_report
)


# Load chat tab CSS from assets (cached at module level)
AGENT_TAB_CSS = load_css_file("chat_tab_layout.css")

load_dotenv()

# Cache role avatar as base64 data URL so we can embed in HTML (per-role avatar in chat)

_AVATAR_B64_CACHE: Dict[str, str] = {}


async def _run_mls_debug_with_tools(
    session_state: dict,
    step_num: int,
    task_desc: str,
    tool_name: str,
    merged_tool_input: dict,
    error_str: str,
) -> tuple:
    """Run MLS self-check; may use read_skill, python_repl, agent_generated_code, etc. Updates history with tool activity. Returns (retry_input or None, report_for_cb or None)."""
    context = (
        f"Step {step_num} failed during tool execution.\n\n"
        f"**Task:** {task_desc}\n**Tool:** {tool_name}\n**Current input:** {json.dumps(merged_tool_input, ensure_ascii=False)}\n**Error:** {error_str}\n\n"
        "You may call read_skill, python_repl, agent_generated_code or other tools to diagnose or fix. Then output exactly one JSON: {\"retry_input\": {...}} or {\"report_for_cb\": \"...\"}."
    )
    executor = session_state.get("mls_debug_executor")
    if not executor:
        return await asyncio.to_thread(
            _mls_debug_step,
            session_state["llm"],
            step_num,
            task_desc,
            tool_name,
            merged_tool_input,
            error_str,
        )
    try:
        output, intermediate_steps = await asyncio.to_thread(_run_mls_debug_executor_sync, executor, context)
        history = session_state.get("history") or []
        log_entries = session_state.get("conversation_log") or []
        for action, observation in intermediate_steps:
            tname = getattr(action, "tool", None) or (action.get("tool") if isinstance(action, dict) else None)
            tinputs = getattr(action, "tool_input", None) or (action.get("tool_input") if isinstance(action, dict) else {})
            if tname == "read_skill":
                skill_id = tinputs.get("skill_id", "") if isinstance(tinputs, dict) else ""
                msg = f"📖 **MLS self-check:** Loaded skill `{skill_id}`."
            elif tname == "python_repl":
                msg = "🔧 **MLS self-check:** Ran code (python_repl)."
            elif tname == "agent_generated_code":
                msg = "🔧 **MLS self-check:** Ran generated code."
            else:
                msg = f"🔧 **MLS self-check:** Called tool `{tname}`."
            history.append({"role": "assistant", "content": msg, "role_id": "machine_learning_specialist"})
            log_entries.append(f"MLS self-check tool: {tname}")
        session_state["history"] = history
        session_state["conversation_log"] = log_entries
        return _parse_mls_debug_output(output)
    except Exception:
        return await asyncio.to_thread(
            _mls_debug_step,
            session_state["llm"],
            step_num,
            task_desc,
            tool_name,
            merged_tool_input,
            error_str,
        )


async def _run_mls_post_step_verify(
    session_state: dict,
    step_num: int,
    task_desc: str,
    tool_name: str,
    merged_tool_input: dict,
    raw_output: Any,
) -> tuple[bool, Optional[dict], Optional[str]]:
    """Run MLS post-step self-check: verify step output before proceeding. Shows new dialog. Returns (status_ok, retry_input, report_for_cb)."""
    out_preview = str(raw_output)
    if len(out_preview) > 2000:
        out_preview = out_preview[:2000] + "\n...(truncated)"
    context = (
        f"Step {step_num} has been executed. Verify the output for technical errors (bugs, failure, wrong format).\n\n"
        f"**Task:** {task_desc}\n**Tool:** {tool_name}\n**Input:** {json.dumps(merged_tool_input, ensure_ascii=False)}\n\n"
        f"**Output:**\n{out_preview}\n\n"
        "Check: Is the output successful and usable (e.g. file path exists, result not null/empty, no nested error)? "
        "If the output has success: true but results, references, or data is null or empty, or does not match the step goal, do NOT output status ok; output {\"retry_input\": {...}} or {\"report_for_cb\": \"...\"} so the step can be re-run with different parameters, another skill, or code. "
        "You may use read_skill, python_repl, or other tools to inspect. "
        "If OK, output exactly: {\"status\": \"ok\"}. "
        "If error, null/empty result, or wrong format, output {\"retry_input\": {...}} or {\"report_for_cb\": \"...\"}."
    )
    executor = session_state.get("mls_debug_executor")
    if not executor:
        return (True, None, None)
    try:
        output, intermediate_steps = await asyncio.to_thread(_run_mls_debug_executor_sync, executor, context)
        history = session_state.get("history") or []
        log_entries = session_state.get("conversation_log") or []
        for action, observation in intermediate_steps:
            tname = getattr(action, "tool", None) or (action.get("tool") if isinstance(action, dict) else None)
            tinputs = getattr(action, "tool_input", None) or (action.get("tool_input") if isinstance(action, dict) else {})
            if tname == "read_skill":
                skill_id = tinputs.get("skill_id", "") if isinstance(tinputs, dict) else ""
                msg = f"📖 **MLS self-check (post-step):** Loaded skill `{skill_id}`."
            elif tname == "python_repl":
                msg = "🔧 **MLS self-check (post-step):** Ran code (python_repl)."
            elif tname == "agent_generated_code":
                msg = "🔧 **MLS self-check (post-step):** Ran generated code."
            else:
                msg = f"🔧 **MLS self-check (post-step):** Called tool `{tname}`."
            history.append({"role": "assistant", "content": msg, "role_id": "machine_learning_specialist"})
            log_entries.append(f"MLS post-step self-check tool: {tname}")
        session_state["history"] = history
        session_state["conversation_log"] = log_entries
        return _parse_mls_post_step_output(output)
    except Exception:
        return (True, None, None)




async def _cb_post_step_check(
    llm,
    step_num: int,
    task_desc: str,
    tool_name: str,
    raw_output: Any,
    output_file_path: Optional[str] = None,
    file_preview: Optional[str] = None,
    timeout_sec: float = 12.0,
) -> tuple[bool, str]:
    """CB verifies: (1) execution matches plan, (2) output not null/empty/weird, (3) if file produced, exists and preview correct. Returns (matches, note)."""
    try:
        out_str = str(raw_output) if raw_output is not None else ""
        if len(out_str) > 500:
            out_str = out_str[:500] + "..."
        null_or_empty = _output_looks_null_or_empty(raw_output)
        try:
            parsed = json.loads(str(raw_output)) if isinstance(raw_output, str) else raw_output
            if isinstance(parsed, dict):
                success = parsed.get("success", None)
                err = parsed.get("error", "")
                out_str = f"success={success}" + (f", error={err[:200]}" if err else "")
                if parsed.get("uniprot_id"):
                    out_str += f", uniprot_id={parsed.get('uniprot_id')}"
                if parsed.get("sequence") and isinstance(parsed["sequence"], str):
                    out_str += f", sequence length={len(parsed['sequence'])}"
                if null_or_empty:
                    out_str += "; results/references/data is null or empty"
        except Exception:
            pass
        prompt = (
            f"You are the Computational Biologist. Verify: (1) execution matches the plan, "
            f"(2) output is not null, empty, or useless for the step goal (e.g. if step was to find UniProt ID, output must contain IDs), "
            f"(3) if an output file was expected, it exists and preview looks correct.\n\n"
            f"**Planned step:** {task_desc}\n**Planned tool:** {tool_name}\n\n"
            f"**Actual tool used:** {tool_name}\n**Result summary:** {out_str}\n\n"
        )
        if null_or_empty:
            prompt += "**Note:** The result appears to have null or empty results/references/data. If the step goal required actual data (e.g. IDs, list of hits), reply MISMATCH and say output is null/empty or does not match plan; CB will ask MLS to re-execute with different parameters, another skill, or code.\n\n"
        if output_file_path:
            prompt += f"**Output file path:** {output_file_path}\n**File exists:** yes\n"
            if file_preview:
                prompt += f"**First 10 lines of file:**\n```\n{file_preview}\n```\n\nCheck that the preview is consistent with the step goal. If content looks wrong, reply MISMATCH.\n\n"
            else:
                prompt += "\n\n"
        prompt += "Reply with one line only: MATCH or MISMATCH: <brief reason>"
        response = await asyncio.wait_for(
            asyncio.to_thread(llm.invoke, [HumanMessage(content=prompt)]),
            timeout=timeout_sec,
        )
        if response is None:
            return (True, "")
        content = (response.content if hasattr(response, "content") else str(response)).strip().upper()
        if "MISMATCH" in content:
            note = content.split("MISMATCH", 1)[-1].strip(" :").strip()[:200]
            return (False, note or "Execution may deviate from plan.")
        return (True, "")
    except Exception:
        return (True, "")


async def send_message(message, session_state):
    """Async message handler with Planner-Worker-Finalizer workflow"""
    if session_state is None:
        session_state = initialize_session_state()
    # All agent execution files under year/month/day/Agent/session_id
    agent_session_dir = session_state.get("agent_session_dir")
    if not agent_session_dir:
        base_dir = os.getenv("TEMP_OUTPUTS_DIR", "temp_outputs")
        session_id = session_state.get("session_id", str(uuid.uuid4()))
        current_time = time.localtime()
        date_subdir = os.path.join(
            str(current_time.tm_year),
            f"{current_time.tm_mon:02d}",
            f"{current_time.tm_mday:02d}",
            "Agent",
        )
        agent_session_dir = os.path.join(base_dir, date_subdir, session_id)
    os.makedirs(agent_session_dir, exist_ok=True)
    UPLOAD_DIR = agent_session_dir
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    if not message or not message.get("text"):
        yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None), session_state
        return

    text = message["text"]
    files = message.get("files", [])


    # Setup file paths (Gradio 6: file_obj may be dict with "path" or string)
    file_paths = []
    if files:
        for file_obj in files:
            try:
                original_temp_path = file_obj.get("path", file_obj) if isinstance(file_obj, dict) else file_obj
                if not original_temp_path or not os.path.isfile(original_temp_path):
                    continue  # Skip dirs and invalid paths - avoids IsADirectoryError

                if os.path.exists(original_temp_path):
                    original_filename = os.path.basename(original_temp_path)
                    unique_filename = f"{original_filename}"
                    destination_path = os.path.join(UPLOAD_DIR, unique_filename)
                    shutil.copy2(original_temp_path, destination_path)
                    normalized_path = destination_path.replace('\\', '/')
                    file_paths.append(normalized_path)
                    session_state['temp_files'].append(normalized_path)
                else:
                    print(f"Warning: Gradio temp file not found at {original_temp_path}")
            except Exception as e:
                print(f"Error processing file: {e}")

    display_text = text
    if file_paths:
        file_names = ", ".join([os.path.basename(f) for f in file_paths])
        display_text += f"\n📎 *Attached: {file_names}*"
   

    # Free-use limit: max conversation messages per chat
    history_len = len(session_state.get("history") or [])
    if history_len >= AGENT_CHAT_MAX_MESSAGES:
        limit_msg = f"**Limit reached.** This chat has reached the maximum of {AGENT_CHAT_MAX_MESSAGES} messages. Start a new chat to continue."
        session_state["history"].append({"role": "assistant", "content": limit_msg, "role_id": "principal_investigator"})
        yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=True, file_count="multiple"), session_state
        return

    session_state['history'].append({"role": "user", "content": display_text})
    # LangChain: keep conversation_log for backend (conversation + tool records)
    session_state.setdefault("conversation_log", []).append({
        "role": "user", "content": display_text, "timestamp": datetime.now().isoformat()
    })

    protein_ctx = session_state['protein_context']
    sequence = extract_sequence_from_message(text)
    uniprot_id = extract_uniprot_id_from_message(text)
    if sequence:
        protein_ctx.add_sequence(sequence)
    if uniprot_id:
        protein_ctx.add_uniprot_id(uniprot_id)
    for fp in file_paths:
        protein_ctx.add_file(fp)

    # Proceed with Planner + Advisor (short memory removed)

    # Call Planner (PI: research, feasibility, plan)
    session_state['history'].append({"role": "assistant", "content": "🤔 Thinking... Creating a plan...", "role_id": "principal_investigator"})
    yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state

    # Build comprehensive context
    context_parts = []

    all_known_files = []
    if protein_ctx.files:
        for _, file_data in protein_ctx.files.items():
            if file_data.get('path'):
                all_known_files.append(file_data['path'])
    if file_paths:
        all_known_files.extend(file_paths)
    all_known_files = sorted(list(set(all_known_files)))
    if all_known_files:
        context_parts.append(f"Available uploaded files: {', '.join(all_known_files)}")
    context_parts.append(f"Default output directory for tool outputs (use for out_dir, out_path): {agent_session_dir}")
    if protein_ctx.structure_files:
        struct_info = []
        for struct_id, struct_data in protein_ctx.structure_files.items():
            struct_info.append(f"{struct_data['source']} structure: {struct_data['name']} (path: {struct_data['path']})")
        context_parts.append(f"Available structure files: {'; '.join(struct_info)}")

    context_parts.append(f"Protein context: {protein_ctx.get_context_summary()}")

    protein_context_summary = "; ".join(context_parts)

    memory = session_state['memory']
    # Include current user message so LangChain chains see this turn
    chat_history = memory.chat_memory.messages + [HumanMessage(content=display_text)]

    # Aggregate recent tool outputs and build unified tools JSON for Planner
    recent_tool_calls = getattr(protein_ctx, "tool_history", [])[-10:]
    tool_outputs_summary = []
    for call in reversed(recent_tool_calls):  # most recent first
        tool_outputs_summary.append({
            "step": call.get("step"),
            "tool": call.get("tool_name"),
            "inputs": call.get("inputs"),
            "cache_key": call.get("cache_key"),
            "cached": call.get("cached", False),
            "timestamp": call.get("timestamp").isoformat() if call.get("timestamp") else None,
            "outputs": call.get("outputs")[:500]

        })

    tool_records = protein_ctx.get_tool_records(limit=10)

    # --- Phase 1: Retrieval then PI report ---
    # Architecture: (1) Retrieval phase: search tools run (manual loop or fallback); each result is summarized in chat (English, one line). (2) Report phase: research draft then Suggest steps (new dialog) for CB/MLS.
    session_state['history'][-1] = {"role": "assistant", "content": "**Principal Investigator** is thinking…", "role_id": "principal_investigator"}
    yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
    current_tool_count = len(getattr(protein_ctx, "tool_history", []))
    pi_suggest_steps = ""
    appended_draft_and_steps = False

    if current_tool_count >= AGENT_CHAT_MAX_TOOL_CALLS:
        report_instruction = (
            "Tool call limit reached. User question: " + text + "\n\n"
            "Output a short research report: ## Abstract, ## Introduction, ## Related Work, ## Suggested approach (capabilities needed; CB will map to tools), ## Rough steps (feasible path), ## References (list all [1], [2], … at the end). No JSON."
        )
        pi_report_inputs = {
            "input": report_instruction,
            "chat_history": [],
            "protein_context_summary": protein_context_summary,
            "tool_outputs": json.dumps(tool_outputs_summary, ensure_ascii=False),
        }
        try:
            pi_report = await asyncio.to_thread(session_state["pi_report"].invoke, pi_report_inputs)
        except Exception as e:
            print(f"[PI report] fallback invoke failed: {e}")
            pi_report = "Unable to generate report (tool limit reached)."
        session_state['history'][-1] = {"role": "assistant", "content": pi_report, "role_id": "principal_investigator"}
    else:
        # PI 5-section flow: plan sections → search + sub-report → draft → Suggest steps (new dialog) for CB/MLS
        intermediate_steps = []
        pi_report = ""
        pi_suggest_steps = ""
        appended_draft_and_steps = False
        final_content_from_llm = ""
        sections_list = []
        try:
            sections_out = await asyncio.to_thread(
                session_state["pi_sections"].invoke,
                {"input": text, "protein_context_summary": protein_context_summary},
            )
            sections_raw = (sections_out or "").strip() if isinstance(sections_out, str) else ""
            if "```" in sections_raw:
                start = sections_raw.find("```")
                if "json" in sections_raw[: start + 10]:
                    start = sections_raw.find("```") + 7
                else:
                    start = sections_raw.find("```") + 3
                end = sections_raw.find("```", start)
                sections_raw = sections_raw[start:end if end > 0 else None].strip()
            i = sections_raw.find("[")
            if i >= 0:
                depth = 0
                for j in range(i, len(sections_raw)):
                    if sections_raw[j] == "[":
                        depth += 1
                    elif sections_raw[j] == "]":
                        depth -= 1
                        if depth == 0:
                            sections_raw = sections_raw[i : j + 1]
                            break
            sections_list = json.loads(sections_raw)
            if not isinstance(sections_list, list) or len(sections_list) == 0:
                sections_list = []
            else:
                sections_list = sections_list[:5]
                parsed_sections = []
                for s in sections_list:
                    sec_name = (s.get("section_name") or "Section").strip() or "Section"
                    focus = (s.get("focus") or "both").strip() or "both"
                    # Support legacy single query or new array of queries
                    queries = s.get("search_queries") or s.get("search_query") or []
                    if isinstance(queries, str):
                        queries = [queries]
                    elif not isinstance(queries, list):
                        queries = [text[:60]]
                    queries = [q.strip() or text[:60] for q in queries if isinstance(q, str)]
                    if not queries:
                        queries = [text[:60]]
                    parsed_sections.append({
                        "section_name": sec_name,
                        "search_queries": queries,
                        "focus": focus,
                    })
                sections_list = parsed_sections
        except Exception as e:
            print(f"[PI sections] parse failed: {e}")
            sections_list = []

        if sections_list:
            session_state["history"].pop()
            sub_reports_parts = []
            for idx, section in enumerate(sections_list):
                if len(protein_ctx.tool_history) >= AGENT_CHAT_MAX_TOOL_CALLS:
                    break
                section_title = section["section_name"]
                session_state["history"].append({"role": "assistant", "content": f"**Section {idx + 1}:** {section_title}", "role_id": "principal_investigator"})
                yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                session_state["history"].append({"role": "assistant", "content": "🔍 **Searching…**", "role_id": "principal_investigator"})
                yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                
                all_search_results = []
                for sq in getattr(section, "get", lambda k: section.get(k, []))("search_queries", []):
                    search_results_str, search_logged = await asyncio.to_thread(_run_section_search, sq)
                    all_search_results.append(search_results_str)
                    
                    for step_off, (tname, tinputs, toutputs) in enumerate(search_logged, start=len(protein_ctx.tool_history) + 1):
                        if len(protein_ctx.tool_history) >= AGENT_CHAT_MAX_TOOL_CALLS:
                            break
                        protein_ctx.add_tool_call(step_off, tname, tinputs, toutputs, cached=False)
                        session_state.setdefault("tool_executions", []).append({
                            "step": step_off, "tool_name": tname, "inputs": tinputs,
                            "outputs": (str(toutputs)[:1000] + "..." if len(str(toutputs)) > 1000 else str(toutputs)),
                            "timestamp": datetime.now().isoformat(),
                        })
                        summary_msg = _format_search_summary(tname, tinputs, str(toutputs))
                        preview = _format_search_preview(tname, str(toutputs))
                        content = summary_msg + ("\n\n" + preview if preview else "")
                        session_state["history"].append({"role": "assistant", "content": content, "role_id": "principal_investigator"})
                        yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                
                # Unify the multiple rounds of search results into one block for the sub-report
                search_results_str = "\n".join(all_search_results)
                
                if session_state["history"] and session_state["history"][-1].get("content") == "🔍 **Searching…**":
                    session_state["history"].pop()
                session_state["history"].append({"role": "assistant", "content": "✍️ **Writing sub-report…**", "role_id": "principal_investigator"})
                yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                try:
                    sub_report = await asyncio.to_thread(
                        session_state["pi_sub_report"].invoke,
                        {"section_name": section_title, "focus": section["focus"], "search_results": search_results_str},
                    )
                except Exception as e2:
                    sub_report = f"(Sub-report failed: {e2})"
                if session_state["history"] and session_state["history"][-1].get("content") == "✍️ **Writing sub-report…**":
                    session_state["history"].pop()
                # Use short-summary title for this sub-report (not "Sub-report" or section name)
                sub_report_title, sub_report_body = _parse_sub_report_short_title((sub_report or "").strip(), fallback_title=section_title)
                # Include references (with [1], [2], titles, links) so final report PI can see and cite them
                ref_block = f"**References:**\n{search_results_str}" if search_results_str and search_results_str.strip() else ""
                sub_block = f"**Sub-report:**\n{sub_report_body}" if sub_report_body else f"**Sub-report:**\n{(sub_report or '').strip()}"
                part = f"### {sub_report_title}\n\n{ref_block}\n\n{sub_block}" if ref_block else f"### {sub_report_title}\n\n{sub_block}"
                sub_reports_parts.append(part)
                # Show full sub-report in chat with short-summary title as heading (no truncation)
                full_sub = f"# {sub_report_title}\n\n{sub_report_body}" if sub_report_body else (sub_report or "").strip()
                session_state["history"].append({"role": "assistant", "content": full_sub, "role_id": "principal_investigator"})
                yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
            sub_reports_text = "\n\n".join(sub_reports_parts)
            session_state["history"].append({"role": "assistant", "content": "📝 **Principal Investigator** is writing the research draft…", "role_id": "principal_investigator"})
            yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
            try:
                pi_report = await asyncio.to_thread(
                    session_state["pi_final_report"].invoke,
                    {"sub_reports": sub_reports_text, "input": text},
                )
                pi_report = (pi_report or "").strip()
            except Exception as e2:
                print(f"[PI final report] invoke failed: {e2}")
                pi_report = "Report generation failed."
            # Replace bubble with actual research draft
            draft_title = (sections_list[0]["section_name"].strip() if sections_list else "") or _short_topic_title(text)
            if not draft_title:
                draft_title = "Research draft"
            draft_content = f"# {draft_title}\n\n{pi_report}"
            session_state["history"][-1] = {"role": "assistant", "content": draft_content, "role_id": "principal_investigator"}
            session_state.setdefault("conversation_log", []).append({
                "role": "assistant", "content": draft_content, "role_id": "principal_investigator",
                "timestamp": datetime.now().isoformat(),
            })
            yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
            # Generate Suggest steps (new dialog) for CB/MLS — show bubble then replace with content
            pi_suggest_steps = ""
            if pi_report and "Report generation failed" not in pi_report:
                session_state["history"].append({"role": "assistant", "content": "📋 **Principal Investigator** is generating Suggest steps…", "role_id": "principal_investigator"})
                yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                try:
                    available_tools_list = ", ".join(sorted(session_state.get("workers", {}).keys())) or "(none)"
                    available_skills_list = ", ".join(sorted(list_skill_ids())) or "(none)"
                    pi_suggest_steps = await asyncio.to_thread(
                        session_state["pi_suggest_steps"].invoke,
                        {
                            "draft_report": pi_report,
                            "input": text,
                            "available_tools_list": available_tools_list,
                            "available_skills_list": available_skills_list,
                        },
                    )
                    pi_suggest_steps = (pi_suggest_steps or "").strip()
                except Exception as e3:
                    print(f"[PI suggest steps] invoke failed: {e3}")
                if pi_suggest_steps:
                    session_state["history"][-1] = {"role": "assistant", "content": pi_suggest_steps, "role_id": "principal_investigator"}
                    session_state.setdefault("conversation_log", []).append({
                        "role": "assistant", "content": pi_suggest_steps, "role_id": "principal_investigator",
                        "timestamp": datetime.now().isoformat(),
                    })
                else:
                    session_state["history"].pop()
                yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                appended_draft_and_steps = True
        else:
            # Fallback: AgentExecutor then optional fixed search
            try:
                agent_result = await asyncio.to_thread(
                    session_state["pi_research_agent"].invoke,
                    {"input": text, "protein_context_summary": protein_context_summary},
                )
            except Exception as e:
                import traceback
                print(f"[PI research] AgentExecutor invoke failed: {e}")
                traceback.print_exc()
                agent_result = {}
            raw_steps = agent_result.get("intermediate_steps") or []
            steps_for_ui = []
            for action, observation in raw_steps:
                tname = getattr(action, "tool", None) or (action.get("tool") if isinstance(action, dict) else None)
                tinputs = getattr(action, "tool_input", None) or (action.get("tool_input", {}) if isinstance(action, dict) else {})
                obs_str = observation if isinstance(observation, str) else json.dumps(observation, ensure_ascii=False)
                if tname:
                    steps_for_ui.append((tname, tinputs, obs_str))
            if agent_result.get("output"):
                final_content_from_llm = (agent_result.get("output") or "").strip()
            if steps_for_ui:
                session_state["history"].pop()
            for step_off, (tname, tinputs, obs_str) in enumerate(steps_for_ui, start=len(protein_ctx.tool_history) + 1):
                if len(protein_ctx.tool_history) >= AGENT_CHAT_MAX_TOOL_CALLS:
                    break
                protein_ctx.add_tool_call(step_off, tname, tinputs, obs_str, cached=False)
                session_state.setdefault("tool_executions", []).append({
                    "step": step_off, "tool_name": tname, "inputs": tinputs,
                    "outputs": (obs_str[:1000] + "..." if len(obs_str) > 1000 else obs_str),
                    "timestamp": datetime.now().isoformat(),
                })
                class _Action:
                    tool = tname
                    tool_input = tinputs
                intermediate_steps.append((_Action(), obs_str))
                summary_msg = _format_search_summary(tname, tinputs, obs_str)
                preview = _format_search_preview(tname, obs_str)
                content = summary_msg + ("\n\n" + preview if preview else "")
                session_state["history"].append({"role": "assistant", "content": content, "role_id": "principal_investigator"})
                yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
            if not intermediate_steps:
                session_state["history"][-1] = {"role": "assistant", "content": "**Principal Investigator** is thinking (fallback search).", "role_id": "principal_investigator"}
                yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                search_results_str, search_logged = await asyncio.to_thread(
                    _fetch_search_for_pi_report, text, None, session_state.get("llm")
                )
                if search_logged:
                    session_state["history"].pop()
                for step_off, (tname, tinputs, toutputs) in enumerate(search_logged, start=len(protein_ctx.tool_history) + 1):
                    if len(protein_ctx.tool_history) >= AGENT_CHAT_MAX_TOOL_CALLS:
                        break
                    protein_ctx.add_tool_call(step_off, tname, tinputs, toutputs, cached=False)
                    session_state.setdefault("tool_executions", []).append({
                        "step": step_off, "tool_name": tname, "inputs": tinputs,
                        "outputs": (str(toutputs)[:1000] + "..." if len(str(toutputs)) > 1000 else str(toutputs)),
                        "timestamp": datetime.now().isoformat(),
                    })
                    summary_msg = _format_search_summary(tname, tinputs, str(toutputs))
                    preview = _format_search_preview(tname, str(toutputs))
                    content = summary_msg + ("\n\n" + preview if preview else "")
                    session_state["history"].append({"role": "assistant", "content": content, "role_id": "principal_investigator"})
                    yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                    class _Action:
                        tool = tname
                        tool_input = tinputs
                    intermediate_steps.append((_Action(), toutputs))
                if search_results_str and search_results_str != "No search results.":
                    report_instruction = (
                        "The references below are numbered [1], [2], [3], etc. You MUST cite them in your report.\n\n"
                        + search_results_str
                        + "\n\nUser question: " + text + "\n\n"
                        "Output your research report with: ## Abstract, ## Introduction, ## Related Work (cite [1],[2],…), ## Suggested approach (capabilities needed; CB will map to tools), ## Rough steps (feasible path), ## References (list all [1], [2], … at the end). No JSON."
                    )
                    try:
                        pi_report = await asyncio.to_thread(session_state["pi_report"].invoke, {
                            "input": report_instruction,
                            "chat_history": [],
                            "protein_context_summary": protein_context_summary,
                            "tool_outputs": json.dumps(tool_outputs_summary, ensure_ascii=False),
                        })
                    except Exception as e2:
                        print(f"[PI report] fallback invoke failed: {e2}")
                        pi_report = "Report generation failed."
            if intermediate_steps and not pi_report:
                search_results_str = _format_pi_steps_for_report(intermediate_steps)
                report_instruction = (
                    "The references below are numbered [1], [2], [3], etc. You MUST cite them in your report.\n\n"
                    + search_results_str
                    + "\n\nUser question: " + text + "\n\n"
                    "Output your research report with: ## Abstract, ## Introduction, ## Related Work (cite [1],[2],…), ## Suggested approach (capabilities needed; CB will map to tools), ## Rough steps (feasible path), ## References (list all [1], [2], … at the end). No JSON."
                )
                try:
                    pi_report = await asyncio.to_thread(session_state["pi_report"].invoke, {
                        "input": report_instruction,
                        "chat_history": [],
                        "protein_context_summary": protein_context_summary,
                        "tool_outputs": json.dumps(tool_outputs_summary, ensure_ascii=False),
                    })
                except Exception as e2:
                    print(f"[PI report] invoke failed: {e2}")
                    pi_report = "Report generation failed."
        if not (pi_report or pi_report.strip()):
            pi_report = (final_content_from_llm or "").strip() or "No search results; no report generated."
        # Append single report message only when we did not show draft + suggest steps (5-section) and we did not replace the last message (tool-limit path)
        if not appended_draft_and_steps and current_tool_count < AGENT_CHAT_MAX_TOOL_CALLS:
            session_state["history"].append({"role": "assistant", "content": pi_report, "role_id": "principal_investigator"})
            yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state

    session_state["memory"].save_context({"input": display_text}, {"output": pi_report})
    if current_tool_count >= AGENT_CHAT_MAX_TOOL_CALLS:
        yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
    # Rebuild tool_outputs_summary after PI research so CB sees latest tool history (including PI's search calls)
    recent_tool_calls = getattr(protein_ctx, "tool_history", [])[-10:]
    tool_outputs_summary = []
    for call in reversed(recent_tool_calls):
        tool_outputs_summary.append({
            "step": call.get("step"),
            "tool": call.get("tool_name"),
            "inputs": call.get("inputs"),
            "cache_key": call.get("cache_key"),
            "cached": call.get("cached", False),
            "timestamp": call.get("timestamp").isoformat() if call.get("timestamp") else None,
            "outputs": call.get("outputs", "")[:500] if isinstance(call.get("outputs"), str) else str(call.get("outputs", ""))[:500],
        })
    yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
    # --- Phase 2: CB planning — from PI draft + Suggest steps to concrete pipeline (JSON steps) ---
    session_state["history"].append({"role": "assistant", "content": "📋 **Computational Biologist** is planning the pipeline…", "role_id": "computational_biologist"})
    yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
    cb_planner_inputs = {
        "pi_report": pi_report,
        "pi_suggest_steps": pi_suggest_steps or "",
        "protein_context_summary": protein_context_summary,
        "tool_outputs": json.dumps(tool_outputs_summary, ensure_ascii=False),
        "skills_metadata": get_skills_metadata_string(),
    }
    try:
        raw_msg = await asyncio.to_thread(session_state["cb_planner_raw"].invoke, cb_planner_inputs)
        content = getattr(raw_msg, "content", None) or str(raw_msg) or ""
        plan = _parse_cb_plan(content)
        if not plan and content:
            print(f"[CB planner] parsed empty plan; raw length={len(content)}, snippet={content[:300]!r}")
    except Exception as e:
        print(f"[CB planner] chain invoke failed: {e}")
        plan = []
    if not isinstance(plan, list):
        plan = []

    # Normalize plan: ensure each item has step, task_description, tool_name, tool_input (CB may return different keys)
    def _normalize_step(i: int, p: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(p, dict):
            return None
        step_num = p.get("step") or p.get("Step") or (i + 1)
        task_desc = (
            p.get("task_description") or p.get("Task_description")
            or p.get("task") or p.get("Task") or p.get("description") or ""
        )
        tool_name = p.get("tool_name") or p.get("Tool_name") or p.get("tool") or p.get("Tool") or p.get("name") or ""
        if not isinstance(tool_name, str):
            tool_name = str(tool_name) if tool_name else ""
        tool_input = p.get("tool_input") or p.get("Tool_input") or p.get("input") or p.get("params") or {}
        if not tool_name:
            return None
        if not isinstance(tool_input, dict):
            tool_input = {}
        return {"step": step_num, "task_description": task_desc, "tool_name": tool_name.strip(), "tool_input": tool_input}

    plan = [_normalize_step(i, p) for i, p in enumerate(plan)]
    plan = [p for p in plan if p is not None]
    # Search steps are already done by PI in the research phase; do not run them again
    plan = [p for p in plan if (p.get("tool_name") or "") not in PI_SEARCH_TOOL_NAMES]

    # Only show "No pipeline steps" when there truly are no steps; if PI suggested steps but plan is empty, parsing failed
    def _pi_suggested_steps_like_content(s: str) -> bool:
        if not s or not s.strip():
            return False
        t = s.strip().lower()
        if "no tools" in t and "no execution" in t:
            return False
        if "无需工具" in s or "无执行步骤" in s or "不需要工具" in s:
            return False
        for token in ("step", "步骤", "tools", "工具", "suggested", "1.", "2.", "3.", "一、", "二、", "read_skill", "uniprot", "biopython", "python_repl"):
            if token in t or token in s:
                return True
        return bool(s.count("\n") >= 2)  # multi-line suggest steps likely has a list

    if not plan:
        if _pi_suggested_steps_like_content(pi_suggest_steps or ""):
            fail_msg = (
                "**Computational Biologist** could not produce a valid pipeline from the suggested steps (parse failed or no tool_name in steps). "
                "Please try again or rephrase; the research draft above remains the answer."
            )
            if content:
                print(f"[CB planner] empty plan but PI suggested steps; raw snippet: {content[:500]!r}")
            session_state["history"][-1] = {"role": "assistant", "content": fail_msg, "role_id": "computational_biologist"}
        else:
            session_state["history"][-1] = {"role": "assistant", "content": "No pipeline steps to run; the research draft above is the answer.", "role_id": "computational_biologist"}
        yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=True, file_count="multiple"), session_state
        return

    # --- CB: replace planning bubble with pipeline, then MLS executes each step ---
    step_lines = [f"**Step {p['step']}.** {p['task_description']}" for p in plan]
    plan_text = "📋 **Pipeline**\n\nHere's what we'll do:\n\n" + "\n\n".join(step_lines)
    session_state["history"][-1] = {"role": "assistant", "content": plan_text, "role_id": "computational_biologist"}
    session_state.setdefault("conversation_log", []).append({
        "role": "assistant", "content": plan_text, "role_id": "computational_biologist",
        "timestamp": datetime.now().isoformat(),
    })
    yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
    step_results = {}
    analysis_log = ""
    collected_references = []

    for step in plan:
        # Free-use limit: max tool calls per chat
        if len(getattr(protein_ctx, "tool_history", [])) >= AGENT_CHAT_MAX_TOOL_CALLS:
            session_state["history"].append({
                "role": "assistant",
                "content": f"**Tool call limit reached.** This chat has used the maximum of {AGENT_CHAT_MAX_TOOL_CALLS} tool calls. Summary below based on results so far.",
                "role_id": "computational_biologist",
            })
            yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
            break

        step_num = step["step"]
        task_desc = step["task_description"]
        tool_name = step["tool_name"]
        tool_input = step["tool_input"]

        # MLS executes pipeline steps (CB only plans; MLS runs tools)
        skill_id = (tool_input or {}).get("skill_id", "") if tool_name == "read_skill" else ""
        if tool_name == "read_skill" and skill_id:
            load_skill_msg = f"📖 **MLS loading skill:** {skill_id}"
            session_state["history"].append({"role": "assistant", "content": load_skill_msg, "role_id": "machine_learning_specialist"})
            session_state.setdefault("conversation_log", []).append({
                "role": "assistant", "content": load_skill_msg, "role_id": "machine_learning_specialist",
                "timestamp": datetime.now().isoformat(),
            })
            yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
        exec_msg = f"⏳ **Executing Step {step_num}:** {task_desc}"
        session_state['history'].append({"role": "assistant", "content": exec_msg, "role_id": "machine_learning_specialist"})
        session_state.setdefault("conversation_log", []).append({
            "role": "assistant", "content": exec_msg, "role_id": "machine_learning_specialist",
            "timestamp": datetime.now().isoformat(),
        })
        yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
        merged_tool_input = _merge_tool_parameters_with_context(protein_ctx, tool_input)
        # CRITICAL: Resolve dependencies BEFORE cache check and tool execution
        for key, value in list(merged_tool_input.items()):
            if isinstance(value, str) and value.startswith("dependency:"):
                parts = value.split(':')
                dep_step = int(parts[1].replace('step_', '').replace('step', ''))
                dep_raw_output = step_results[dep_step]['raw_output']
                if len(parts) > 2:
                    field_name = parts[2]
                    try:
                        parsed = json.loads(dep_raw_output) if isinstance(dep_raw_output, str) else dep_raw_output
                        merged_tool_input[key] = parsed.get(field_name, dep_raw_output)
                    except Exception:
                        merged_tool_input[key] = dep_raw_output
                else:
                    merged_tool_input[key] = dep_raw_output

        step_retry = 0
        step_done = False
        while step_retry <= MAX_STEP_RETRIES and not step_done:
            try:
                cached_entry = get_cached_tool_result(session_state, tool_name, merged_tool_input)

                if cached_entry:
                    raw_output = cached_entry.get("outputs", "")
                    step_results[step_num] = {'raw_output': raw_output, 'cached': True}
                    protein_ctx.add_tool_call(step_num, tool_name, merged_tool_input, raw_output, cached=True)
                    out_path_cached = _get_output_file_path_from_raw(raw_output, tool_name)
                    session_state.setdefault("tool_executions", []).append({
                        "step": step_num, "tool_name": tool_name, "inputs": merged_tool_input,
                        "outputs": (str(raw_output)[:1000] + "..." if len(str(raw_output)) > 1000 else str(raw_output)),
                        "cached": True, "timestamp": datetime.now().isoformat(),
                        "success": True, "output_file_path": out_path_cached,
                    })

                    # Collect references from cached outputs if present
                    try:
                        cached_outputs = cached_entry.get("outputs", {})
                        if isinstance(cached_outputs, dict) and cached_outputs.get("references"):
                            for rr in cached_outputs.get("references", []):
                                if rr and (rr.get("title") or rr.get("url")):
                                    collected_references.append(rr)
                    except Exception:
                        pass

                    # Every step (including cached): post-step self-check dialog
                    session_state["history"].append({
                        "role": "assistant",
                        "content": MLS_POST_STEP_SELF_CHECK_MSG.format(step_num=step_num),
                        "role_id": "machine_learning_specialist",
                    })
                    yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                    post_ok, post_retry, post_report = await _run_mls_post_step_verify(
                        session_state, step_num, task_desc, tool_name, merged_tool_input, raw_output,
                    )
                    if not post_ok:
                        if post_report:
                            session_state["history"].append({"role": "assistant", "content": f"**MLS report for CB:** {post_report}", "role_id": "machine_learning_specialist"})
                        session_state["history"].append({"role": "assistant", "content": f"**CB:** Step {step_num} (cached) verification failed. " + (post_report or "Output issue.") + " Re-plan or re-run if needed.", "role_id": "computational_biologist"})
                        yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=True), session_state
                        return

                    step_detail = f"**Step {step_num}:** {task_desc}\n\n"
                    step_detail += f"**Tool:** {tool_name} ⚡ (cached result)\n"
                    code_block = _format_mls_code_for_display(tool_name, merged_tool_input, raw_output)
                    if code_block:
                        step_detail += code_block + "\n\n"
                    step_detail += f"**Input:** {json.dumps(merged_tool_input, indent=2)}\n\n"
                    step_detail += f"**Cache Key:** {cached_entry.get('cache_key', 'N/A')}\n\n"
                    step_detail += f"**Output:**\n```\n{str(raw_output)[:500]}{'...' if len(str(raw_output)) > 500 else ''}\n```"
                    image_paths = _extract_image_paths_from_tool_output(raw_output, tool_name)
                    step_detail = _embed_image_paths_in_content(step_detail, image_paths)

                    analysis_log += f"--- Cached Analysis for Step {step_num}: {task_desc} ---\n\n"
                    analysis_log += f"Tool: {tool_name} (cached)\n"
                    analysis_log += f"Input: {json.dumps(merged_tool_input, indent=2)}\n"
                    analysis_log += f"Output: {raw_output}\n\n"

                    # Extract download file from cached output
                    try:
                        cached_data = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
                        if isinstance(cached_data, dict):
                            dl_path = _extract_download_file_from_output(tool_name, cached_data)
                            if dl_path:
                                session_state['latest_tool_output_file'] = dl_path
                    except Exception:
                        pass

                    out_path = _get_output_file_path_from_raw(raw_output, tool_name)
                    preview = _read_output_file_preview(out_path, max_lines=10) if out_path else ""
                    cb_match, cb_note = await _cb_post_step_check(session_state["llm"], step_num, task_desc, tool_name, raw_output, output_file_path=out_path, file_preview=preview if out_path else None)
                    # Always show step result (success, output, file path) so user sees execution metadata
                    step_complete_msg = f"✅ **Step {step_num} Complete (cached):** {task_desc}\n\n{step_detail}"
                    session_state['history'].append({"role": "assistant", "content": step_complete_msg, "role_id": "machine_learning_specialist"})
                    session_state.setdefault("conversation_log", []).append({
                        "role": "assistant", "content": step_complete_msg, "role_id": "machine_learning_specialist",
                        "timestamp": datetime.now().isoformat(),
                    })
                    if cb_match:
                        cb_msg = f"✓ **CB verified:** Step {step_num} complete (cached). Execution matches plan. Proceeding to next."
                        session_state['history'].append({"role": "assistant", "content": cb_msg, "role_id": "computational_biologist"})
                        session_state.setdefault("conversation_log", []).append({
                            "role": "assistant", "content": cb_msg, "role_id": "computational_biologist",
                            "timestamp": datetime.now().isoformat(),
                        })
                        yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                        step_done = True
                        break
                    # CB says cached output does not match — discuss with MLS (user already saw step result above)
                    cb_mismatch_msg = f"**CB:** Output does not match plan ({cb_note}). Re-execute with different parameters, another skill, or code."
                    session_state["history"].append({"role": "assistant", "content": cb_mismatch_msg, "role_id": "computational_biologist"})
                    session_state.setdefault("conversation_log", []).append({
                        "role": "assistant", "content": cb_mismatch_msg, "role_id": "computational_biologist",
                        "timestamp": datetime.now().isoformat(),
                    })
                    yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                    cb_retry, cb_report = await _run_mls_debug_with_tools(
                        session_state, step_num, task_desc, tool_name, merged_tool_input,
                        f"CB verification: output does not match plan. {cb_note} Re-execute with different parameters, another skill, or code.",
                    )
                    if cb_retry and step_retry < MAX_STEP_RETRIES:
                        step_retry += 1
                        merged_tool_input.update(cb_retry)
                        session_state["history"].append({"role": "assistant", "content": f"🔧 **MLS:** Retrying Step {step_num} (cached result rejected).", "role_id": "machine_learning_specialist"})
                        yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                        continue
                    if cb_report:
                        session_state["history"].append({"role": "assistant", "content": f"**MLS report for CB:** {cb_report}", "role_id": "machine_learning_specialist"})
                    session_state["history"].append({"role": "assistant", "content": f"**CB:** Step {step_num} output did not match plan. Re-plan or ask user to adjust.", "role_id": "computational_biologist"})
                    yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=True), session_state
                    return

                # Dependencies already resolved above, proceed to tool execution
                worker_executor = session_state['workers'].get(tool_name)
                if not worker_executor:
                    raise ValueError(f"Worker executor '{tool_name}' not found.")
                inputs_json_str = json.dumps(merged_tool_input, ensure_ascii=False, indent=2)

                agent_input_text = (
                    f"Execute this task: {task_desc}\n\n"
                    f"INPUT DATA (Use these parameters explicitly, DO NOT ASK for them):\n"
                    f"```json\n{inputs_json_str}\n```\n"
                    f"Command: Call the tool '{tool_name}' immediately using the data above."
                )


                print(f"\n[AgentExecutor] Step {step_num}: {tool_name}", flush=True)
                executor_result = None
                last_step_error = None
                mls_report_for_cb = None
                for attempt in range(2):
                    try:
                        executor_result = await asyncio.wait_for(
                            asyncio.to_thread(
                                worker_executor.invoke,
                                {
                                    "input": agent_input_text,
                                    **merged_tool_input
                                }
                            ),
                            timeout=TOOL_EXECUTION_TIMEOUT,
                        )
                        break
                    except asyncio.TimeoutError as e:
                        last_step_error = TimeoutError(
                            f"Tool execution timed out after {TOOL_EXECUTION_TIMEOUT}s. The tool may be slow or stuck."
                        )
                        if attempt == 0:
                            session_state["history"].append({"role": "assistant", "content": MLS_SELF_CHECK_MSG, "role_id": "machine_learning_specialist"})
                            yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                            retry_input, mls_report_for_cb = await _run_mls_debug_with_tools(
                                session_state,
                                step_num,
                                task_desc,
                                tool_name,
                                merged_tool_input,
                                str(last_step_error),
                            )
                            if session_state["history"] and session_state["history"][-1].get("content") == MLS_SELF_CHECK_MSG:
                                session_state["history"].pop()
                            if retry_input:
                                merged_tool_input.update(retry_input)
                                inputs_json_str = json.dumps(merged_tool_input, ensure_ascii=False, indent=2)
                                agent_input_text = (
                                    f"Execute this task: {task_desc}\n\n"
                                    f"INPUT DATA (Use these parameters explicitly, DO NOT ASK for them):\n"
                                    f"```json\n{inputs_json_str}\n```\n"
                                    f"Command: Call the tool '{tool_name}' immediately using the data above."
                                )
                                session_state["history"].append({"role": "assistant", "content": f"🔧 **MLS debug:** Retrying Step {step_num} with corrected parameters.", "role_id": "machine_learning_specialist"})
                                yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                                continue
                        break
                    except Exception as e:
                        last_step_error = e
                        if attempt == 0:
                            session_state["history"].append({"role": "assistant", "content": MLS_SELF_CHECK_MSG, "role_id": "machine_learning_specialist"})
                            yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                            retry_input, mls_report_for_cb = await _run_mls_debug_with_tools(
                                session_state,
                                step_num,
                                task_desc,
                                tool_name,
                                merged_tool_input,
                                str(e),
                            )
                            if session_state["history"] and session_state["history"][-1].get("content") == MLS_SELF_CHECK_MSG:
                                session_state["history"].pop()
                            if retry_input:
                                merged_tool_input.update(retry_input)
                                inputs_json_str = json.dumps(merged_tool_input, ensure_ascii=False, indent=2)
                                agent_input_text = (
                                    f"Execute this task: {task_desc}\n\n"
                                    f"INPUT DATA (Use these parameters explicitly, DO NOT ASK for them):\n"
                                    f"```json\n{inputs_json_str}\n```\n"
                                    f"Command: Call the tool '{tool_name}' immediately using the data above."
                                )
                                session_state["history"].append({"role": "assistant", "content": f"🔧 **MLS debug:** Retrying Step {step_num} with corrected parameters.", "role_id": "machine_learning_specialist"})
                                yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                                continue
                        break
                if executor_result is None and last_step_error is not None:
                    error_message = f"❌ **Error in Step {step_num}:** {task_desc}\n`{str(last_step_error)}`"
                    session_state['history'].append({"role": "assistant", "content": error_message, "role_id": "machine_learning_specialist"})
                    session_state.setdefault("conversation_log", []).append({
                        "role": "assistant", "content": error_message, "role_id": "machine_learning_specialist",
                        "timestamp": datetime.now().isoformat(),
                    })
                    session_state["history"].append({"role": "assistant", "content": MLS_SELF_CHECK_MSG, "role_id": "machine_learning_specialist"})
                    yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                    retry_input, mls_report_for_cb = await _run_mls_debug_with_tools(
                        session_state,
                        step_num,
                        task_desc,
                        tool_name,
                        merged_tool_input,
                        str(last_step_error),
                    )
                    if session_state["history"] and session_state["history"][-1].get("content") == MLS_SELF_CHECK_MSG:
                        session_state["history"].pop()
                    if retry_input and step_retry < MAX_STEP_RETRIES:
                        step_retry += 1
                        merged_tool_input.update(retry_input)
                        session_state["history"].append({"role": "assistant", "content": f"🔧 **MLS self-check:** Retrying Step {step_num} with corrected parameters.", "role_id": "machine_learning_specialist"})
                        yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                        continue
                    if mls_report_for_cb:
                        session_state['history'].append({"role": "assistant", "content": f"**MLS report for CB:** {mls_report_for_cb}", "role_id": "machine_learning_specialist"})
                        session_state['history'].append({"role": "assistant", "content": f"**CB:** Step {step_num} failed. {mls_report_for_cb} Please re-plan the pipeline or ask the user to adjust.", "role_id": "computational_biologist"})
                    else:
                        session_state['history'].append({"role": "assistant", "content": f"**CB suggestion:** Step {step_num} failed. Check parameters or retry; ask again to adjust the pipeline if needed.", "role_id": "computational_biologist"})
                    yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=True), session_state
                    return
                if executor_result is None:
                    continue

                # Extract tool output from intermediate_steps (the actual tool return value)
                raw_output = ''
                if executor_result.get('intermediate_steps'):
                    last_step = executor_result['intermediate_steps'][-1]
                    if len(last_step) >= 2:
                        tool_output = last_step[1]
                        raw_output = tool_output if isinstance(tool_output, str) else json.dumps(tool_output, ensure_ascii=False)
                    else:
                        raw_output = executor_result.get('output', '')
                else:
                    raw_output = executor_result.get('output', '')
                step_results[step_num] = {'raw_output': raw_output, 'cached': False}

                # If tool returned success: false (top-level or nested in result/data), run MLS self-check and retry or report
                try:
                    output_failed, err_str = _tool_output_indicates_failure(raw_output)
                    if output_failed:
                        err_str = err_str or "Tool reported failure or error"
                        fail_msg = f"❌ **Step {step_num} failed (tool reported error):** {task_desc}\n`{err_str[:300]}`"
                        session_state["history"].append({"role": "assistant", "content": fail_msg, "role_id": "machine_learning_specialist"})
                        session_state.setdefault("conversation_log", []).append({
                            "role": "assistant", "content": fail_msg, "role_id": "machine_learning_specialist",
                            "timestamp": datetime.now().isoformat(),
                        })
                        session_state["history"].append({"role": "assistant", "content": MLS_SELF_CHECK_MSG, "role_id": "machine_learning_specialist"})
                        yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                        retry_input, mls_report_f = await _run_mls_debug_with_tools(
                            session_state,
                            step_num,
                            task_desc,
                            tool_name,
                            merged_tool_input,
                            err_str,
                        )
                        if session_state["history"] and session_state["history"][-1].get("content") == MLS_SELF_CHECK_MSG:
                            session_state["history"].pop()
                        if retry_input and step_retry < MAX_STEP_RETRIES:
                            step_retry += 1
                            merged_tool_input.update(retry_input)
                            session_state["history"].append({"role": "assistant", "content": f"🔧 **MLS self-check:** Retrying Step {step_num} with corrected parameters.", "role_id": "machine_learning_specialist"})
                            yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                            continue
                        if mls_report_f:
                            session_state['history'].append({"role": "assistant", "content": f"**MLS report for CB:** {mls_report_f}", "role_id": "machine_learning_specialist"})
                            session_state['history'].append({"role": "assistant", "content": f"**CB:** Step {step_num} failed. {mls_report_f} Please re-plan the pipeline or ask the user to adjust.", "role_id": "computational_biologist"})
                        else:
                            session_state['history'].append({"role": "assistant", "content": f"**CB suggestion:** Step {step_num} failed. Check parameters or retry; ask again to adjust the pipeline if needed.", "role_id": "computational_biologist"})
                        yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=True), session_state
                        return
                except Exception:
                    pass

                # Every step: post-step self-check (new dialog) — verify output before proceeding
                session_state["history"].append({
                    "role": "assistant",
                    "content": MLS_POST_STEP_SELF_CHECK_MSG.format(step_num=step_num),
                    "role_id": "machine_learning_specialist",
                })
                yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                post_ok, post_retry, post_report = await _run_mls_post_step_verify(
                    session_state, step_num, task_desc, tool_name, merged_tool_input, raw_output,
                )
                if not post_ok:
                    if post_retry and step_retry < MAX_STEP_RETRIES:
                        step_retry += 1
                        merged_tool_input.update(post_retry)
                        session_state["history"].append({"role": "assistant", "content": f"🔧 **Machine Learning Specialist self-check (post-step):** Step {step_num} output issue; retrying with adjusted parameters.", "role_id": "machine_learning_specialist"})
                        yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                        continue
                    if post_report:
                        session_state["history"].append({"role": "assistant", "content": f"**Machine Learning Specialist report for Computational Biologist:** {post_report}", "role_id": "machine_learning_specialist"})
                        session_state["history"].append({"role": "assistant", "content": f"**Computational Biologist:** Step {step_num} verification failed. {post_report} Please re-plan or ask user to adjust.", "role_id": "computational_biologist"})
                    else:
                        session_state["history"].append({"role": "assistant", "content": f"**Computational Biologist:** Step {step_num} post-step check found an issue. Re-plan or retry if needed.", "role_id": "computational_biologist"})
                    yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=True), session_state
                    return

                # Display training progress if this is a training tool
                if tool_name == 'train_protein_model_tool':
                    try:
                        parsed_output = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
                        if isinstance(parsed_output, dict) and 'training_progress' in parsed_output:
                            training_progress = parsed_output['training_progress']
                            if training_progress:
                                progress_display = f"**Training Progress:**\n```\n{training_progress}\n```"
                                session_state['history'].append({"role": "assistant", "content": f"⏳ **Step {step_num}:** {task_desc}\n\n{progress_display}", "role_id": "machine_learning_specialist"})
                                yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state

                    except Exception as e:
                        print(f"Could not parse training progress: {e}")


                # Try to parse and cache the result (only if successful)
                try:
                    try:
                        parsed_output = json.loads(raw_output)
                    except Exception:
                        parsed_output = raw_output

                    cached = save_cached_tool_result(session_state, tool_name, merged_tool_input, parsed_output)
                    if not cached:
                        print(f"⚠ Step {step_num} result not cached due to execution failure")

                    try:
                        if isinstance(parsed_output, dict) and parsed_output.get("references"):
                            for rr in parsed_output.get("references", []):
                                if rr and (rr.get("title") or rr.get("url")):
                                    collected_references.append(rr)
                    except Exception:
                        pass

                except Exception as e:
                    print(f"Failed to process result: {e}")

                # Always record tool call in history
                protein_ctx.add_tool_call(step_num, tool_name, merged_tool_input, raw_output, cached=False)
                out_path_for_log = _get_output_file_path_from_raw(raw_output, tool_name)
                try:
                    parsed_for_success = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
                    step_success = (
                        parsed_for_success.get("success") is True
                        or (isinstance(parsed_for_success, dict) and parsed_for_success.get("status") == "success")
                    ) if isinstance(parsed_for_success, dict) else True
                except Exception:
                    step_success = True
                session_state.setdefault("tool_executions", []).append({
                    "step": step_num, "tool_name": tool_name, "inputs": merged_tool_input,
                    "outputs": (str(raw_output)[:1000] + "..." if len(str(raw_output)) > 1000 else str(raw_output)),
                    "cached": False, "timestamp": datetime.now().isoformat(),
                    "success": step_success, "output_file_path": out_path_for_log,
                })

                # Parse tool output to update context and download file
                try:
                    if tool_name in ['download_ncbi_sequence_by_accession', 'download_alphafold_structure_by_uniprot_id', 'download_uniprot_sequence_by_uniprot_id', 'query_interpro_by_uniprot_id',
                                     'protein_function_prediction', 'functional_residue_prediction',
                                     'protein_property_prediction', 'zero_shot_mutation_sequence_prediction', 'zero_shot_mutation_structure_prediction',
                                     'query_sequence_from_pdb_file', 'download_pdb_structure_by_pdb_id', 'query_literature_by_keywords', 'query_dataset_by_keywords', 'query_web_by_keywords',
                                     'esmfold_structure_prediction', 'query_foldseek_search_by_pdb_file']:

                        output_data = json.loads(raw_output)
                        if output_data.get('success') and 'file_path' in output_data:
                            file_path = output_data['file_path']
                            if tool_name == 'download_alphafold_structure_by_uniprot_id':
                                uniprot_id = merged_tool_input.get('uniprot_id', 'unknown')
                                protein_ctx.add_structure_file(file_path, 'alphafold', uniprot_id)
                            elif tool_name == 'download_ncbi_sequence_by_accession':
                                protein_ctx.add_file(file_path)
                        dl_path = _extract_download_file_from_output(tool_name, output_data)
                        if dl_path:
                            session_state['latest_tool_output_file'] = dl_path
                        if tool_name == 'query_literature_by_keywords' and isinstance(output_data, dict) and output_data.get('references'):
                            try:
                                for rr in output_data.get('references', []):
                                    if rr and (rr.get('title') or rr.get('url')):
                                        collected_references.append(rr)
                            except Exception:
                                pass
                except (json.JSONDecodeError, KeyError):
                    pass

                # Create detailed step result display
                step_detail = f"**Step {step_num}:** {task_desc}\n\n"
                step_detail += f"**Tool:** {tool_name}\n"
                code_block = _format_mls_code_for_display(tool_name, merged_tool_input, raw_output)
                if code_block:
                    step_detail += code_block + "\n\n"
                step_detail += f"**Input:** {json.dumps(merged_tool_input, indent=2)}\n\n"

                if tool_name == 'train_protein_model_tool':
                    try:
                        parsed_output = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
                        if isinstance(parsed_output, dict):
                            if parsed_output.get('success'):
                                step_detail += f"✅ **Training Completed Successfully!**\n\n"
                                if 'model_path' in parsed_output:
                                    step_detail += f"**Model saved to:** `{parsed_output['model_path']}`\n\n"

                                if 'training_progress' in parsed_output:
                                    step_detail += f"**Training Progress:**\n```\n{parsed_output['training_progress']}\n```\n\n"
                            else:
                                step_detail += f"❌ **Training Failed**\n\n"
                                if 'error' in parsed_output:
                                    step_detail += f"**Error:** {parsed_output['error']}\n\n"
                            if 'logs' in parsed_output:
                                step_detail += f"**Recent Logs:**\n```\n{parsed_output['logs']}\n```"
                    except Exception:
                        output_str = str(raw_output) if not isinstance(raw_output, str) else raw_output
                        step_detail += f"**Output:**\n```\n{output_str[:500]}{'...' if len(output_str) > 500 else ''}\n```"
                else:
                    output_str = str(raw_output) if not isinstance(raw_output, str) else raw_output
                    # Show more output for code/REPL tools so written code and plot paths are visible in chat
                    max_len = 3000 if tool_name in ("agent_generated_code", "python_repl") else 500
                    step_detail += f"**Output:**\n```\n{output_str[:max_len]}{'...' if len(output_str) > max_len else ''}\n```"

                image_paths = _extract_image_paths_from_tool_output(raw_output, tool_name)
                step_detail = _embed_image_paths_in_content(step_detail, image_paths)

                analysis_log += f"--- Analysis for Step {step_num}: {task_desc} ---\n\n"
                analysis_log += f"Tool: {tool_name}\n"
                analysis_log += f"Input: {json.dumps(merged_tool_input, indent=2)}\n"
                analysis_log += f"Output: {raw_output}\n\n"

                out_path = _get_output_file_path_from_raw(raw_output, tool_name)
                preview = _read_output_file_preview(out_path, max_lines=10) if out_path else ""
                cb_match, cb_note = await _cb_post_step_check(session_state["llm"], step_num, task_desc, tool_name, raw_output, output_file_path=out_path, file_preview=preview if out_path else None)
                # Always show step result (success, output, file path) so user and backend log see execution metadata
                step_complete_msg = f"✅ **Step {step_num} Complete:** {task_desc}\n\n{step_detail}"
                session_state['history'].append({"role": "assistant", "content": step_complete_msg, "role_id": "machine_learning_specialist"})
                session_state.setdefault("conversation_log", []).append({
                    "role": "assistant", "content": step_complete_msg, "role_id": "machine_learning_specialist",
                    "timestamp": datetime.now().isoformat(),
                })
                if cb_match:
                    cb_msg = f"✓ **CB verified:** Step {step_num} complete. Execution matches plan. Proceeding to next."
                    session_state['history'].append({"role": "assistant", "content": cb_msg, "role_id": "computational_biologist"})
                    session_state.setdefault("conversation_log", []).append({
                        "role": "assistant", "content": cb_msg, "role_id": "computational_biologist",
                        "timestamp": datetime.now().isoformat(),
                    })
                    yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                    step_done = True
                    break
                # CB says output does not match plan (null/empty/weird) — discuss with MLS and re-execute or report (user already saw step result above)
                cb_mismatch_msg = f"**CB:** Output does not match plan ({cb_note}). Re-execute this step with different parameters, another skill, or code."
                session_state["history"].append({"role": "assistant", "content": cb_mismatch_msg, "role_id": "computational_biologist"})
                session_state.setdefault("conversation_log", []).append({
                    "role": "assistant", "content": cb_mismatch_msg, "role_id": "computational_biologist",
                    "timestamp": datetime.now().isoformat(),
                })
                yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                cb_retry, cb_report = await _run_mls_debug_with_tools(
                    session_state, step_num, task_desc, tool_name, merged_tool_input,
                    f"CB verification: output does not match plan. {cb_note} Re-execute with different parameters, another skill, or code.",
                )
                if cb_retry and step_retry < MAX_STEP_RETRIES:
                    step_retry += 1
                    merged_tool_input.update(cb_retry)
                    session_state["history"].append({"role": "assistant", "content": f"🔧 **MLS:** Retrying Step {step_num} with adjusted parameters/skill/code.", "role_id": "machine_learning_specialist"})
                    yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                    continue
                if cb_report:
                    session_state["history"].append({"role": "assistant", "content": f"**MLS report for CB:** {cb_report}", "role_id": "machine_learning_specialist"})
                session_state["history"].append({"role": "assistant", "content": f"**CB:** Step {step_num} output did not match plan. {cb_note} Re-plan or ask user to adjust.", "role_id": "computational_biologist"})
                yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=True), session_state
                return

            except Exception as e:
                error_message = f"❌ **Error in Step {step_num}:** {task_desc}\n`{str(e)}`"
                session_state['history'].append({"role": "assistant", "content": error_message, "role_id": "machine_learning_specialist"})
                session_state.setdefault("conversation_log", []).append({
                    "role": "assistant", "content": error_message, "role_id": "machine_learning_specialist",
                    "timestamp": datetime.now().isoformat(),
                })
                session_state["history"].append({"role": "assistant", "content": MLS_SELF_CHECK_MSG, "role_id": "machine_learning_specialist"})
                yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                retry_input, mls_report_for_cb = await _run_mls_debug_with_tools(
                    session_state,
                    step_num,
                    task_desc,
                    tool_name,
                    merged_tool_input,
                    str(e),
                )
                if session_state["history"] and session_state["history"][-1].get("content") == MLS_SELF_CHECK_MSG:
                    session_state["history"].pop()
                if retry_input and step_retry < MAX_STEP_RETRIES:
                    step_retry += 1
                    merged_tool_input.update(retry_input)
                    session_state["history"].append({"role": "assistant", "content": f"🔧 **Machine Learning Specialist self-check:** Retrying Step {step_num} with corrected parameters.", "role_id": "machine_learning_specialist"})
                    yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state
                    continue
                if mls_report_for_cb:
                    session_state['history'].append({"role": "assistant", "content": f"**Machine Learning Specialist report for Computational Biologist:** {mls_report_for_cb}", "role_id": "machine_learning_specialist"})
                    session_state['history'].append({"role": "assistant", "content": f"**Computational Biologist:** Step {step_num} failed. {mls_report_for_cb} Please re-plan the pipeline or ask the user to adjust.", "role_id": "computational_biologist"})
                else:
                    session_state['history'].append({"role": "assistant", "content": f"**Computational Biologist suggestion:** Step {step_num} failed. Check parameters or retry; ask again to adjust the pipeline if needed.", "role_id": "computational_biologist"})
                yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=True), session_state
                return

    # SC summarizes only after all steps (separate dialog + avatar)
    session_state['history'].append({"role": "assistant", "content": "📄 **All steps complete. Generating final report...**", "role_id": "scientific_critic"})
    yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state

    # Build full run record so SC sees all agent outputs and tool executions
    run_parts = []
    for h in session_state.get("history") or []:
        role = h.get("role")
        role_id = h.get("role_id") or role
        content = (h.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            run_parts.append("**User:**\n" + content)
        else:
            label = ROLE_DISPLAY_NAMES.get(role_id, (role_id or "Assistant").replace("_", " ").title())
            run_parts.append(f"**{label}:**\n{content}")
    run_parts.append("\n## Tool executions")
    for t in session_state.get("tool_executions") or []:
        run_parts.append(f"\n### {t.get('step')}. {t.get('tool_name', '')}")
        run_parts.append("Input: " + json.dumps(t.get("inputs", {}), ensure_ascii=False, indent=2))
        out_str = str(t.get("outputs", ""))
        run_parts.append("Output: " + (out_str[:2000] + "..." if len(out_str) > 2000 else out_str))
    full_run_record = "\n\n---\n\n".join(run_parts)

    finalizer_inputs = {
        "original_input": text,
        "full_run_record": full_run_record,
        "analysis_log": analysis_log,
        "references": json.dumps(_dedupe_references(collected_references), ensure_ascii=False)
    }

    final_response = await asyncio.to_thread(
        session_state['finalizer'].invoke,
        finalizer_inputs
    )
    session_state['history'][-1] = {"role": "assistant", "content": final_response, "role_id": "scientific_critic"}
    session_state['memory'].save_context({"input": display_text}, {"output": final_response})
    session_state.setdefault("conversation_log", []).append({
        "role": "assistant", "content": final_response, "role_id": "scientific_critic",
        "timestamp": datetime.now().isoformat(),
    })
    yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=True, file_count="multiple"), session_state








def create_chat_tab(constant: Dict[str, Any]) -> Dict[str, Any]:
    """Create the chat tab: left = main area (conversation + input at bottom); right = sidebar (examples, tips, log)."""
    session_state = gr.State(value=None)  # Lazy-init in send_message (State cannot deepcopy LLM/chains)
    gr.Markdown("""
    💡 *This is a academic demo version of VenusFactory (Max PI search 3, Max tools 40), please [local deployment](https://github.com/AI4Protein/VenusFactory) to unlock the full features or visit [project version](https://venusfactory.cn).*
    """)
    with gr.Row(equal_height=True, elem_id="vf-agent-page-row"):
        # Left (main): conversation area + input row at bottom
        with gr.Column(scale=3, min_width=200, elem_id="vf-agent-main-column"):
            with gr.Group(elem_id="vf-agent-main-group"):
                right_panel_display = gr.HTML(
                    value=_build_conversation_panel_html([]),
                    elem_id="vf-right-panel-root",
                )
                with gr.Row(elem_id="vf-agent-input-row"):
                    model_selector = gr.Dropdown(choices=["ChatGPT-4o", "Gemini-2.5-Pro", "Claude-3.7", "DeepSeek-R1"], value="Gemini-2.5-Pro", show_label=False, scale=2)
                    chat_input = gr.MultimodalTextbox(interactive=True, file_count="multiple", placeholder="💬 Ask me anything about AI protein.", show_label=False, file_types=[".fasta", ".fa", ".pdb", ".csv"], scale=8)
        # Sidebar: log, examples, tips (references chat_input)
        with gr.Column(scale=2, min_width=200, elem_id="vf-agent-sidebar-column"):
            with gr.Group():
                with gr.Accordion("📋 Backend Log", open=True, elem_id="vf-agent-backend-log"):
                    log_display = gr.HTML(value=_build_log_html({}), elem_id="vf-log-display")
                with gr.Accordion("💡 Example Research Questions", open=True):
                    gr.Examples(
                        examples=[
                            ["For UniProt P04040 (SOD1): search the literature for stability-related mutations and mechanisms, download the AlphaFold structure, then run in silico saturation mutagenesis to rank single-point mutations by predicted stability change and suggest top candidates for wet-lab validation."],
                            ["I have a FASTA file of protein sequences: extract UniProt IDs, look up InterPro domains for each, run protein function prediction, then suggest which sequences are best suited for a custom property prediction model and outline the training pipeline."],
                            ["For PDB 1A00: get chain sequences and run FoldSeek to find structurally similar proteins; download the AlphaFold structure of one similar UniProt entry, predict protein property (e.g. stability) for both, and compare the results with a brief literature context."]
                        ],
                        inputs=chat_input,
                        label=None
                    )
                with gr.Accordion("✨ Tips for Prompting VenusFactory", open=True):
                    gr.Markdown("""
**Capabilities**: Sequence/structure analysis, function prediction, mutation impact, DB queries (UniProt/PDB/NCBI).

**Input**: Paste FASTA, give UniProt/PDB ID, or upload .fasta/.pdb/.csv. State your goal clearly.

**Auto workflow**: Decompose tasks → run tools → integrate results.
                    """)

    model_selector.change(
        fn=update_llm_model,
        inputs=[model_selector, session_state],
        outputs=[session_state]
    )

    chat_input.submit(
        fn=send_message,
        inputs=[chat_input, session_state],
        outputs=[right_panel_display, log_display, chat_input, session_state],
        concurrency_limit=3,
        show_progress="full"
    )

    return {
        "chatbot": right_panel_display,
        "chat_input": chat_input,
        "session_state": session_state
    }

if __name__ == "__main__":
    components = create_chat_tab({})
    with gr.Blocks(theme=gr.themes.Soft(), css=AGENT_TAB_CSS or ".gradio-container {max-width: 95% !important;}") as demo:
        create_chat_tab({})

    demo.queue(
        concurrency_count=3,  
        max_size=20,
        api_open=False
    )

    demo.launch(
        share=True,
        max_threads=40,  
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )