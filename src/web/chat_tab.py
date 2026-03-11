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
    _build_conversation_panel_html, _build_log_html
)

from agent.chat_agent_utils import (
    AGENT_CHAT_MAX_MESSAGES, AGENT_CHAT_MAX_TOOL_CALLS, SEARCH_MAX_RESULTS,
    MLS_SELF_CHECK_MSG, MLS_POST_STEP_SELF_CHECK_MSG, MAX_STEP_RETRIES, TOOL_EXECUTION_TIMEOUT,
    extract_sequence_from_message, extract_uniprot_id_from_message,
    _mls_debug_step, _parse_mls_debug_output,
    _run_mls_debug_executor_sync, _run_mls_debug_with_tools, _parse_mls_post_step_output,
    _run_mls_post_step_verify, _output_looks_null_or_empty, _cb_post_step_check, _try_parse_json_array,
)
from agent.chat_graph import create_agent_graph


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

    # --- Phase: LangGraph Orchestration ---
    # Add initial feedback
    session_state['history'].append({"role": "assistant", "content": "🤔 Thinking... Initializing research plan...", "role_id": "principal_investigator"})
    yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state

    initial_state = {
        "messages": [HumanMessage(content=display_text)],
        "protein_context": protein_ctx,
        "session_id": session_state['session_id'],
        "agent_session_dir": agent_session_dir,
        "history": list(session_state['history']),
        "conversation_log": list(session_state.get("conversation_log", [])),
        "tool_executions": list(session_state.get("tool_executions", [])),
        "status": "started",
        "pi_report": "",
        "pi_suggest_steps": "",
        "plan": [],
        "current_step_index": 0,
        "step_results": {},
        "error": None,
        "research_sections": [],
        "research_idx": 0,
        "search_idx": 0,
        "current_search_results": [],
        "research_sub_reports": []
    }

    graph = create_agent_graph()
    config = {"configurable": {"chains": session_state, "session_id": session_state["session_id"]}}
    
    async for event in graph.astream(initial_state, config=config):
        for node, updates in event.items():
            # Update session state with the latest graph state
            for key, val in updates.items():
                if key in ("history", "conversation_log", "tool_executions", "status", "pi_report", "plan", "protein_context"):
                    session_state[key] = val
                
        yield _build_conversation_panel_html(session_state["history"]), _build_log_html(session_state), gr.MultimodalTextbox(value=None, interactive=False), session_state

    # Sync memory (simplified final step)
    final_content = session_state["history"][-1]["content"] if session_state["history"] else ""
    session_state['memory'].save_context({"input": display_text}, {"output": final_content})

    # Final yield to enable textbox
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