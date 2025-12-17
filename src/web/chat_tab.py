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

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from langchain_core.prompt_values import ChatPromptValue
from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks import CallbackManagerForLLMRun

from pathlib import Path
from dotenv import load_dotenv
from gradio_client import Client, handle_file
from web.chat_tools import *
# Import prompts from the new file
from web.prompts import PLANNER_PROMPT, WORKER_PROMPT, FINALIZER_PROMPT, CHAT_SYSTEM_PROMPT
from web.utils.chat_helpers import (
    handle_feedback_submit,
    export_chat_history_html,
    save_chat_history_to_server
)
import threading

load_dotenv()


class Chat_LLM(BaseChatModel):
    api_key: str = None
    base_url: str = "https://www.dmxapi.com/v1"
    model_name: str = "gemini-2.5-pro"
    temperature: float = 0.2
    max_tokens: int = 4096
    
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("OPENAI_API_KEY")

    def _generate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any,) -> ChatResult:
        if not self.api_key:
            raise ValueError("DeepSeek API key is not configured.")

        message_dicts = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, SystemMessage):
                role = "system"
            else: 
                role = "user" 
            message_dicts.append({"role": role, "content": msg.content})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": message_dicts,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            **kwargs,
        }
        
        import requests
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=120
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"API request failed: {response.status_code} - {response.text}")

        result = response.json()
        choice = result['choices'][0]
        message_data = choice['message']

        ai_message = AIMessage(
            content=message_data.get('content', ''),
            additional_kwargs=message_data,
        )
        
        generation = ChatGeneration(message=ai_message)
        return ChatResult(generations=[generation])

    async def _agenerate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any,) -> ChatResult:
        """Asynchronous generation for concurrent execution"""
        if not self.api_key:
            raise ValueError("DeepSeek API key is not configured.")

        message_dicts = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, SystemMessage):
                role = "system"
            else: 
                role = "user" 
            message_dicts.append({"role": role, "content": msg.content})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": message_dicts,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            **kwargs,
        }
        
        timeout = aiohttp.ClientTimeout(total=120)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    raise RuntimeError(f"API request failed: {response.status} - {text}")

                result = await response.json()
                choice = result['choices'][0]
                message_data = choice['message']

                ai_message = AIMessage(
                    content=message_data.get('content', ''),
                    additional_kwargs=message_data,
                )
                
                generation = ChatGeneration(message=ai_message)
                return ChatResult(generations=[generation])

    async def ainvoke(self, messages: List[BaseMessage], **kwargs) -> AIMessage:
        """Async invoke method"""
        result = await self._agenerate(messages, **kwargs)
        return result.generations[0].message

    @property
    def _llm_type(self) -> str:
        return "chat-llm"


class ProteinContextManager:
    def __init__(self, max_tool_history=10):
        self.sequences = {}  # {sequence_id: {'sequence': str, 'timestamp': datetime}}
        self.files = {}      # {file_id: {'path': str, 'type': str, 'timestamp': datetime}}
        self.uniprot_ids = {} # {uniprot_id: timestamp}
        self.structure_files = {} # {structure_id: {'path': str, 'source': str, 'uniprot_id': str, 'timestamp': datetime}}
        self.last_sequence = None
        self.last_file = None
        self.last_uniprot_id = None
        self.last_structure = None

        self.tool_history = []
        self.max_tool_history = max_tool_history  # Maximum number of tool calls to keep in history

    def add_tool_call(self, step: int, tool_name: str, inputs: dict, outputs: Any, cached: bool = False):
        merged_params = _merge_tool_parameters_with_context(self, inputs)
        cache_key = generate_cache_key(tool_name, merged_params)
        # Store structured tool record
        tool_record = {
            'step': step,
            'name': tool_name,
            'parameters': merged_params,
            'outputs': outputs,
            'cached': cached,
            'cache_key': cache_key,
            'timestamp': datetime.now().isoformat()
        }
        # Legacy fields kept for backward compatibility
        self.tool_history.append({
            'step': step,
            'tool_name': tool_name,
            'inputs': merged_params,
            'outputs': str(outputs),
            'cache_key': cache_key,
            'timestamp': datetime.now(),
            'cached': cached,
            'tool_record': tool_record,
        })
        
        # Limit history to max_tool_history entries (keep most recent)
        if len(self.tool_history) > self.max_tool_history:
            self.tool_history.pop(0)  # Remove oldest entry

    def get_tool_records(self, limit: int = None) -> List[Dict[str, Any]]:
        """Get tool call records, optionally limited to most recent N entries"""
        records = []
        # Apply limit if specified, otherwise return all
        history_slice = self.tool_history[-limit:] if limit else self.tool_history
        
        for call in history_slice:
            rec = call.get('tool_record')
            if rec:
                records.append(rec)
            else:
                # Fallback: construct a minimal record if missing
                records.append({
                    'step': call.get('step'),
                    'name': call.get('tool_name'),
                    'parameters': call.get('inputs'),
                    'outputs': call.get('outputs'),
                    'cached': call.get('cached', False),
                    'cache_key': call.get('cache_key'),
                    'timestamp': call.get('timestamp').isoformat() if call.get('timestamp') else None,
                })
        return records
    
    def get_tool_history_summary(self) -> str:
        if not self.tool_history:
            return "No tools called yet in this session."
        
        summary = f"Total tools called: {len(self.tool_history)}\n\n"
        for i, call in enumerate(self.tool_history, 1):
            cache_status = "✓ cached" if call['cached'] else "✗ executed"
            summary += f"{i}. [{cache_status}] Step {call['step']}: {call['tool_name']}\n"
            summary += f"   Inputs: {json.dumps(call['inputs'], indent=2)[:200]}...\n"
            summary += f"   Cache Key: {call.get('cache_key', 'N/A')}\n"
            summary += f"   Time: {call['timestamp'].strftime('%H:%M:%S')}\n\n"
        
        return summary
        
    def add_sequence(self, sequence: str) -> str:
        seq_id = f"seq_{len(self.sequences) + 1}"
        self.sequences[seq_id] = {
            'sequence': sequence,
            'timestamp': datetime.now(),
            'length': len(sequence)
        }
        self.last_sequence = sequence
        return seq_id
    
    def add_file(self, file_path: str) -> str:
        file_id = f"file_{len(self.files) + 1}"
        file_ext = os.path.splitext(file_path)[1].lower()
        file_type = self._determine_file_type(file_ext)
        
        self.files[file_id] = {
            'path': file_path,
            'type': file_type,
            'timestamp': datetime.now(),
            'name': os.path.basename(file_path)
        }
        self.last_file = file_path
        return file_id
    
    def add_uniprot_id(self, uniprot_id: str):
        self.uniprot_ids[uniprot_id] = datetime.now()
        self.last_uniprot_id = uniprot_id
    
    def add_structure_file(self, file_path: str, source: str, uniprot_id: str = None) -> str:
        """Add a structure file to context (AlphaFold, RCSB, etc.)"""
        struct_id = f"struct_{len(self.structure_files) + 1}"
        self.structure_files[struct_id] = {
            'path': file_path,
            'source': source,  # 'alphafold', 'rcsb', etc.
            'uniprot_id': uniprot_id,
            'timestamp': datetime.now(),
            'name': os.path.basename(file_path)
        }
        self.last_structure = file_path
        return struct_id
    
    def get_context_summary(self) -> str:
        summary_parts = []
        
        if self.last_sequence:
            summary_parts.append(f"Most recent sequence: {len(self.last_sequence)} amino acids")
        
        if self.last_file:
            file_name = self.last_file
            file_ext = os.path.splitext(file_name)[1]
            summary_parts.append(f"Most recent file: {file_name} ({file_ext})")
        
        if self.last_uniprot_id:
            summary_parts.append(f"Most recent UniProt ID: {self.last_uniprot_id}")
        
        if self.last_structure:
            struct_name = self.last_structure
            summary_parts.append(f"Most recent structure: {struct_name}")
        
        if len(self.sequences) > 1:
            summary_parts.append(f"Total sequences in memory: {len(self.sequences)}")
        if len(self.files) > 1:
            summary_parts.append(f"Total files in memory: {len(self.files)}")
        if len(self.structure_files) > 1:
            summary_parts.append(f"Total structures in memory: {len(self.structure_files)}")
        if len(self.uniprot_ids) > 1:
            summary_parts.append(f"Total UniProt IDs in memory: {len(self.uniprot_ids)}")
        
        return "; ".join(summary_parts) if summary_parts else "No protein data in memory"
    
    def _determine_file_type(self, file_ext: str) -> str:
        type_mapping = {
            '.fasta': 'sequence', '.fa': 'sequence',
            '.pdb': 'structure',
            '.csv': 'data'
        }
        return type_mapping.get(file_ext, 'unknown')
        
def generate_cache_key(tool_name: str, tool_input: dict) -> str:
    input_str = json.dumps(tool_input, sort_keys=True, ensure_ascii=False)
    params_hash = hashlib.md5(input_str.encode('utf-8')).hexdigest()[:12]
    cache_key = f"{tool_name}_{params_hash}"
    
    return cache_key


def _merge_tool_parameters_with_context(protein_ctx: "ProteinContextManager", base_params: Dict[str, Any]) -> Dict[str, Any]:
    """Merge tool input parameters with current protein context (files/sequence/UniProt).
    This ensures Planner receives unified, structured parameters for each tool call.
    """
    params = dict(base_params or {})
    try:
        # Add latest context snapshots
        params.setdefault("last_sequence", getattr(protein_ctx, "last_sequence", None))
        params.setdefault("last_uniprot_id", getattr(protein_ctx, "last_uniprot_id", None))
        params.setdefault("last_file", getattr(protein_ctx, "last_file", None))

        # Include all known files for reference
        all_files = []
        for _, f in getattr(protein_ctx, "files", {}).items():
            if f.get("path"):
                all_files.append(f["path"])
        params.setdefault("files", sorted(list(set(all_files))))
    except Exception:
        # Fail-safe: never crash tool recording due to context merge issues
        pass
    return params


def get_cached_tool_result(session_state: dict, tool_name: str, tool_input: dict) -> Optional[Dict[str, Any]]:
    cache_key = generate_cache_key(tool_name, tool_input)
    cache = session_state.get("tool_cache", {})
    cached_result = cache.get(cache_key)
    
    if cached_result:
        cached_inputs = cached_result.get("inputs", {})
        if cached_inputs == tool_input:
            print(f"✓ Cache HIT (exact match): {cache_key}")
            return cached_result
        else:
            print(f"⚠ Cache key collision detected for {cache_key}, inputs differ")
            return None
    
    print(f"✗ Cache MISS: {cache_key}")
    return None


def save_cached_tool_result(session_state: dict, tool_name: str, tool_input: dict, outputs: Any) -> bool:
    """Save tool result to cache only if execution was successful.
    
    Returns:
        bool: True if result was cached, False if result indicates failure and was not cached
    """
    # Check if the output indicates success
    is_success = True
    if isinstance(outputs, dict):
        # If output has 'success' field, use it to determine if we should cache
        if 'success' in outputs:
            is_success = outputs.get('success', False)
    elif isinstance(outputs, str):
        # Try to parse string output as JSON to check success field
        try:
            parsed = json.loads(outputs)
            if isinstance(parsed, dict) and 'success' in parsed:
                is_success = parsed.get('success', False)
        except (json.JSONDecodeError, ValueError):
            # If not JSON or no success field, assume success (for backward compatibility)
            pass
    
    # Only cache successful results
    if not is_success:
        print(f"⚠ Tool execution failed, not caching result for {tool_name}")
        return False
    
    cache_key = generate_cache_key(tool_name, tool_input)
    cache = session_state.setdefault("tool_cache", {})
    
    cache[cache_key] = {
        "tool_name": tool_name,
        "inputs": tool_input,
        "outputs": outputs,
        "timestamp": time.time(),
        "cache_key": cache_key
    }
    print(f"✓ Cached successful result for {tool_name}")
    return True
    


def save_inmemory_tool_result(session_state: dict, protein_id: str, tool_name: str, inputs: Dict[str, Any], outputs: Any, files: Optional[Dict[str, str]] = None, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cache = session_state.setdefault("tool_cache", {})
    p = cache.setdefault(protein_id, {"tools": [], "final_answers": {}, "created_at": time.time(), "last_updated": None})
    entry = {
        "name": tool_name,
        "invocation_id": f"inmem-{int(time.time()*1000)}",
        "inputs": inputs,
        "outputs": outputs,
        "files": files or {},
        "meta": meta or {},
        "timestamp": time.time()
    }
    p["tools"].append(entry)
    p["last_updated"] = time.time()
    return entry


def get_tools():
    """Returns a list of all available tool instances."""
    return [
        zero_shot_sequence_prediction_tool,
        zero_shot_structure_prediction_tool,
        protein_function_prediction_tool,
        functional_residue_prediction_tool,
        interpro_query_tool,
        uniprot_query_tool,
        pdb_structure_download_tool,
        protein_properties_generation_tool,
        generate_training_config_tool,
        train_protein_model_tool,
        predict_with_protein_model_tool,
        ai_code_execution_tool,
        ncbi_sequence_download_tool,
        alphafold_structure_download_tool,
        PDB_sequence_extraction_tool,
        literature_search_tool,
    ]

def create_planner_chain(llm: BaseChatModel, tools: List[BaseTool]):
    """Creates the Planner chain that generates a step-by-step plan."""
    tools_description = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
    planner_prompt_with_tools = PLANNER_PROMPT.partial(tools_description=tools_description)
    return planner_prompt_with_tools | llm | JsonOutputParser()

def create_worker_executor(llm: BaseChatModel, tools: List[BaseTool]):
    """Creates a Worker AgentExecutor for a given set of tools."""
    # Expecting tools to be a single-item list (one worker per tool)
    tool = tools[0] if isinstance(tools, list) and tools else None
    # Create a tool-specific prompt instance

    worker_prompt = WORKER_PROMPT.partial(tool_name=(tool.name if tool else "tool"), tool_description=(tool.description if tool else ""))

    agent = create_openai_tools_agent(llm, tools, worker_prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=2,
        max_execution_time=300,
        return_intermediate_steps=True,
    )
    return executor

def create_finalizer_chain(llm: BaseChatModel):
    """Creates the Finalizer chain to aggregate all analyses into a final report."""
    return FINALIZER_PROMPT | llm | StrOutputParser()

def initialize_session_state() -> Dict[str, Any]:
    """Initialize a new session state with all necessary components"""
    llm = Chat_LLM(temperature=0.1)
    all_tools = get_tools()

    planner_chain = create_planner_chain(llm, all_tools)
    finalizer_chain = create_finalizer_chain(llm)
    workers = {t.name: create_worker_executor(llm, [t]) for t in all_tools}
    
    return {
        'session_id': str(uuid.uuid4()),
        'planner': planner_chain,
        'workers': workers,
        'finalizer': finalizer_chain,
        'llm': llm,
        'memory': ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=10),
        'history': [],
        'protein_context': ProteinContextManager(),
        'temp_files': [],
        'tool_cache': {},  # in-memory per-session virtual cache for tool outputs
        'created_at': datetime.now()
    }

def update_llm_model(selected: str, state: Dict[str, Any]):
    mapping = {
        "ChatGPT-4o": "gpt-4o",
        "Gemini-2.5-Pro": "gemini-2.5-pro",
        "Claude-3.7": "claude-3-7-sonnet-20250219",
        "DeepSeek-R1": "deepseek-r1-0528"
    }
    state['llm'].model_name = mapping.get(selected, "gemini-2.5-pro")
    return state

def _dedupe_references(refs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for r in refs or []:
        if not isinstance(r, dict):
            continue
        title = (r.get('title') or '').strip().lower()
        doi = (r.get('doi') or '').strip().lower()
        url = (r.get('url') or '').strip().lower()
        key = (title, doi, url)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out

def extract_sequence_from_message(message: str) -> Optional[str]:
    """Extract protein sequence from user message"""
    sequence_pattern = r'[ACDEFGHIKLMNPQRSTVWY]{20,}'
    matches = re.findall(sequence_pattern, message.upper())
    return matches[0] if matches else None


def extract_uniprot_id_from_message(message: str) -> Optional[str]:
    """Extract UniProt ID from user message"""
    uniprot_pattern = r'\b[A-Z][A-Z0-9]{5}(?:[A-Z0-9]{4})?\b'
    matches = re.findall(uniprot_pattern, message.upper())
    return matches[0] if matches else None


async def send_message(history, message, session_state):
    """Async message handler with Planner-Worker-Finalizer workflow"""
    
    BASE_UPLOAD_DIR = "temp_outputs"
    current_time = time.localtime()
    time_stamped_subdir = os.path.join(
        str(current_time.tm_year),
        f"{current_time.tm_mon:02d}",
        f"{current_time.tm_mday:02d}",
        f"{current_time.tm_hour:02d}_{current_time.tm_min:02d}_{current_time.tm_sec:02d}"
    )
    UPLOAD_DIR = os.path.join(BASE_UPLOAD_DIR, time_stamped_subdir)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    if not message or not message.get("text"):
        yield history, gr.MultimodalTextbox(value=None)
        return

    text = message["text"]
    files = message.get("files", [])
    
    # Setup file paths
    file_paths = []
    if files:
        for file_obj in files:
            try:
                original_temp_path = file_obj
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
    
    history.append({"role": "user", "content": display_text})
    session_state['history'].append({"role": "user", "content": display_text})

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

    # Call Planner (with Advisor refinement)
    history.append({"role": "assistant", "content": "🤔 Thinking... Creating a plan..."})
    yield history, gr.MultimodalTextbox(value=None, interactive=False)

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
    
    if protein_ctx.structure_files:
        struct_info = []
        for struct_id, struct_data in protein_ctx.structure_files.items():
            struct_info.append(f"{struct_data['source']} structure: {struct_data['name']} (path: {struct_data['path']})")
        context_parts.append(f"Available structure files: {'; '.join(struct_info)}")

    context_parts.append(f"Protein context: {protein_ctx.get_context_summary()}")

    protein_context_summary = "; ".join(context_parts)

    memory = session_state['memory']
    chat_history = memory.chat_memory.messages

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

    tool_records = protein_ctx.get_tool_records(limit=10)  # Only send last 10 tool calls to Planner

    planner_inputs = {
        "input": text,
        "chat_history": chat_history,
        "protein_context_summary": protein_context_summary,
        "tool_outputs": json.dumps(tool_outputs_summary, ensure_ascii=False),
        "tools": json.dumps(tool_records, ensure_ascii=False)
    }

    try:
        # Async planner invocation
        plan = await asyncio.to_thread(session_state['planner'].invoke, planner_inputs)
    except Exception as e:
        plan = []
    
    # Ensure plan is a list; otherwise treat as empty (direct chat)
    if not isinstance(plan, list):
        plan = []

    # If plan is empty, just chat (unified handling for both planner failure and empty plan)
    if not plan:
        history[-1] = {"role": "assistant", "content": "🧭 Generating answers, please wait..."}
        yield history, gr.MultimodalTextbox(value=None, interactive=False)
        
        llm = session_state['llm']
        # Add system prompt for direct chat
        messages_with_system = [SystemMessage(content=CHAT_SYSTEM_PROMPT)] + session_state['memory'].chat_memory.messages + [HumanMessage(content=text)]
        response = await llm.ainvoke(messages_with_system)
        final_response = response.content
        
        # Update history and memory
        history[-1] = {"role": "assistant", "content": final_response}
        session_state['history'].append({"role": "assistant", "content": final_response})
        session_state['memory'].save_context({"input": display_text}, {"output": final_response})
        
        yield history, gr.MultimodalTextbox(value=None, interactive=True, file_count="multiple")
        return
    else:
        # Execute Plan
        plan_text = "📋 **Plan Created:**\n" + "\n".join([f"**Step {p['step']}**: {p['task_description']}" for p in plan])
        history[-1] = {"role": "assistant", "content": plan_text}
        yield history, gr.MultimodalTextbox(value=None, interactive=False)
        
        step_results = {}
        analysis_log = ""
        collected_references = []

        for step in plan:
            step_num = step['step']
            task_desc = step['task_description']
            tool_name = step['tool_name']
            tool_input = step['tool_input']

            history[-1] = {"role": "assistant", "content": f"{plan_text}\n\n---\n\n⏳ **Executing Step {step_num}:** {task_desc}"}
            yield history, gr.MultimodalTextbox(value=None, interactive=False)

            try:
                merged_tool_input = _merge_tool_parameters_with_context(protein_ctx, tool_input)
                
                # CRITICAL: Resolve dependencies BEFORE cache check and tool execution
                for key, value in merged_tool_input.items():
                    if isinstance(value, str) and value.startswith("dependency:"):
                        parts = value.split(':')
                        dep_step = int(parts[1].replace('step_', '').replace('step', ''))
                        dep_raw_output = step_results[dep_step]['raw_output']
                        if len(parts) > 2:
                            field_name = parts[2]
                            try:
                                parsed = json.loads(dep_raw_output) if isinstance(dep_raw_output, str) else dep_raw_output
                                merged_tool_input[key] = parsed.get(field_name, dep_raw_output)
                            except:
                                merged_tool_input[key] = dep_raw_output
                        else:
                            merged_tool_input[key] = dep_raw_output
                
                
                cached_entry = get_cached_tool_result(session_state, tool_name, merged_tool_input)
                
                if cached_entry:
                    raw_output = cached_entry.get("outputs", "")
                    step_results[step_num] = {'raw_output': raw_output, 'cached': True}
                    protein_ctx.add_tool_call(step_num, tool_name, merged_tool_input, raw_output, cached=True)

                    # Collect references from cached outputs if present
                    try:
                        cached_outputs = cached_entry.get("outputs", {})
                        if isinstance(cached_outputs, dict) and cached_outputs.get("references"):
                            for rr in cached_outputs.get("references", []):
                                if rr and (rr.get("title") or rr.get("url")):
                                    collected_references.append(rr)
                    except Exception:
                        pass

                    step_detail = f"**Step {step_num}:** {task_desc}\n\n"
                    step_detail += f"**Tool:** {tool_name} ⚡ (cached result)\n"
                    step_detail += f"**Input:** {json.dumps(merged_tool_input, indent=2)}\n\n"
                    step_detail += f"**Cache Key:** {cached_entry.get('cache_key', 'N/A')}\n\n"
                    step_detail += f"**Output:**\n```\n{str(raw_output)[:500]}{'...' if len(str(raw_output)) > 500 else ''}\n```"
                    
                    
                    analysis_log += f"--- Cached Analysis for Step {step_num}: {task_desc} ---\n\n"
                    analysis_log += f"Tool: {tool_name} (cached)\n"
                    analysis_log += f"Input: {json.dumps(merged_tool_input, indent=2)}\n"
                    analysis_log += f"Output: {raw_output}\n\n"

                    history[-1] = {"role": "assistant", "content": f"{plan_text}\n\n---\n\n✅ **Step {step_num} Complete (cached):** {task_desc}\n\n{step_detail}"}
                    yield history, gr.MultimodalTextbox(value=None, interactive=False)
                    continue

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

                executor_result = await asyncio.to_thread(
                    worker_executor.invoke, 
                    {
                        "input": agent_input_text,
                        **merged_tool_input
                    }
                )

                # Extract tool output from intermediate_steps (the actual tool return value)
                # intermediate_steps format: [(AgentAction, tool_output), ...]
                raw_output = ''
                if executor_result.get('intermediate_steps'):
                    # Get the last tool execution result (should only be 1 with max_iterations=3)
                    last_step = executor_result['intermediate_steps'][-1]
                    if len(last_step) >= 2:
                        tool_output = last_step[1]  # The tool's actual output
                        # Ensure output is a string (tools may return dict or str)
                        if isinstance(tool_output, str):
                            raw_output = tool_output
                        else:
                            # Convert dict/object to JSON string
                            raw_output = json.dumps(tool_output, ensure_ascii=False)
                    else:
                        raw_output = executor_result.get('output', '')
                else:
                    # Fallback to agent's text output if no intermediate steps
                    raw_output = executor_result.get('output', '')
                step_results[step_num] = {'raw_output': raw_output, 'cached': False}
                
                # Display training progress if this is a training tool
                if tool_name == 'train_protein_model_tool':
                    try:
                        parsed_output = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
                        if isinstance(parsed_output, dict) and 'training_progress' in parsed_output:
                            training_progress = parsed_output['training_progress']
                            if training_progress:
                                # Update the display with training progress
                                progress_display = f"**Training Progress:**\n```\n{training_progress}\n```"
                                history[-1] = {"role": "assistant", "content": f"{plan_text}\n\n---\n\n⏳ **Step {step_num}:** {task_desc}\n\n{progress_display}"}
                                yield history, gr.MultimodalTextbox(value=None, interactive=False)
                    except Exception as e:
                        print(f"Could not parse training progress: {e}")

                # Try to parse and cache the result (only if successful)
                try:
                    try:
                        parsed_output = json.loads(raw_output)
                    except Exception:
                        parsed_output = raw_output

                    # Only cache if the tool execution was successful
                    cached = save_cached_tool_result(session_state, tool_name, merged_tool_input, parsed_output)
                    if not cached:
                        print(f"⚠ Step {step_num} result not cached due to execution failure")

                    # Collect references from parsed output if provided by the tool
                    try:
                        if isinstance(parsed_output, dict) and parsed_output.get("references"):
                            for rr in parsed_output.get("references", []):
                                if rr and (rr.get("title") or rr.get("url")):
                                    collected_references.append(rr)
                    except Exception:
                        pass
                except Exception as e:
                    print(f"Failed to process result: {e}")
                
                # Always record tool call in history (for tracking purposes, regardless of success/failure)
                protein_ctx.add_tool_call(step_num, tool_name, merged_tool_input, raw_output, cached=False)
                
                # Parse tool output to update context
                try:
                    if tool_name in ['ncbi_sequence_download', 'alphafold_structure_download', 'uniprot_query', 'interpro_query',
                                     'protein_function_prediction', 'functional_residue_prediction',
                                     'protein_properties_generation', 'zero_shot_sequence_prediction', 'zero_shot_structure_prediction',
                                     'PDB_sequence_extraction', 'PDB_structure_download', 'literature_search']:
                        output_data = json.loads(raw_output)
                        # Structure downloads
                        if output_data.get('success') and 'file_path' in output_data:
                            file_path = output_data['file_path']
                            if tool_name == 'alphafold_structure_download':
                                uniprot_id = merged_tool_input.get('uniprot_id', 'unknown')
                                protein_ctx.add_structure_file(file_path, 'alphafold', uniprot_id)
                            elif tool_name == 'ncbi_sequence_download':
                                protein_ctx.add_file(file_path)
                        # Collect literature references if provided
                        if tool_name == 'literature_search' and isinstance(output_data, dict) and output_data.get('references'):
                            try:
                                for rr in output_data.get('references', []):
                                    if rr and (rr.get('title') or rr.get('url')):
                                        collected_references.append(rr)
                            except Exception:
                                pass
                        # Sequence from UniProt or NCBI when provided
                except (json.JSONDecodeError, KeyError):
                    pass
                
                # Create detailed step result display
                step_detail = f"**Step {step_num}:** {task_desc}\n\n"
                step_detail += f"**Tool:** {tool_name}\n"
                step_detail += f"**Input:** {json.dumps(merged_tool_input, indent=2)}\n\n"
                
                # For training tool, show training progress prominently
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
                        # Fallback to default display
                        output_str = str(raw_output) if not isinstance(raw_output, str) else raw_output
                        step_detail += f"**Output:**\n```\n{output_str[:500]}{'...' if len(output_str) > 500 else ''}\n```"
                else:
                    # Safely display output (ensure it's a string)
                    output_str = str(raw_output) if not isinstance(raw_output, str) else raw_output
                    step_detail += f"**Output:**\n```\n{output_str[:500]}{'...' if len(output_str) > 500 else ''}\n```"
                
                analysis_log += f"--- Analysis for Step {step_num}: {task_desc} ---\n\n"
                analysis_log += f"Tool: {tool_name}\n"
                analysis_log += f"Input: {json.dumps(merged_tool_input, indent=2)}\n"
                analysis_log += f"Output: {raw_output}\n\n"

                history[-1] = {"role": "assistant", "content": f"{plan_text}\n\n---\n\n✅ **Step {step_num} Complete:** {task_desc}\n\n{step_detail}"}
                yield history, gr.MultimodalTextbox(value=None, interactive=False)

            except Exception as e:
                error_message = f"❌ **Error in Step {step_num}:** {task_desc}\n`{str(e)}`"
                history[-1] = {"role": "assistant", "content": f"{plan_text}\n\n---\n\n{error_message}"}
                yield history, gr.MultimodalTextbox(value=None, interactive=True)
                return

        # Finalize Report
        history[-1] = {"role": "assistant", "content": f"{plan_text}\n\n---\n\n📄 **All steps complete. Generating final report...**"}
        yield history, gr.MultimodalTextbox(value=None, interactive=False)

        finalizer_inputs = {
            "original_input": text,
            "analysis_log": analysis_log,
            "references": json.dumps(_dedupe_references(collected_references), ensure_ascii=False)
        }
        final_response = await asyncio.to_thread(
            session_state['finalizer'].invoke,
            finalizer_inputs
        )

        history[-1] = {"role": "assistant", "content": final_response}
        session_state['history'].append({"role": "assistant", "content": final_response})
        
        # Update memory
        session_state['memory'].save_context({"input": display_text}, {"output": final_response})
        
        yield history, gr.MultimodalTextbox(value=None, interactive=True, file_count="multiple")


def create_chat_tab(constant: Dict[str, Any]) -> Dict[str, Any]:
    """Create the chat tab interface with concurrency support"""
    
    with gr.Blocks() as demo:
        session_state = gr.State(value=initialize_session_state())
        
        with gr.Column():
            chatbot = gr.Chatbot(
                label="VenusFactory AI Assistant",
                type="messages",
                height=900,
                show_label=False,
                avatar_images=(None, "https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/venus_logo.png"),
                bubble_full_width=False,
                show_copy_button=True
            )
            
            with gr.Row():
                model_selector = gr.Dropdown(
                    choices=["ChatGPT-4o", "Gemini-2.5-Pro", "Claude-3.7", "DeepSeek-R1"],
                    value="Gemini-2.5-Pro",
                    label="Chat Model"
                )
            chat_input = gr.MultimodalTextbox(
                interactive=True,
                file_count="multiple",
                placeholder="💬 Ask me about protein engineering, upload files (FASTA, PDB), or request analysis...",
                show_label=False,
                file_types=[".fasta", ".fa", ".pdb", ".csv"]
            )

            with gr.Accordion("✨ Tips for Prompting VenusFactory", open=True):
                gr.Markdown("""
                    **VenusFactory excels at**: Protein sequence/structure analysis, function prediction, stability/solubility assessment, mutation impact analysis, and database queries (UniProt/PDB/NCBI/InterPro).

                    **How to get the best results**:
                      Provide specific protein information (sequence, UniProt ID, PDB ID, etc.)
                      Clearly state your analysis goal (e.g., "predict stability", "compare two variants", "analyze mutation impact")
                      For complex tasks, describe the analysis steps and key parameters you expect
                    
                    **Supported input methods**:
                      Paste protein sequences directly (FASTA format)
                      Provide UniProt ID (e.g., P04040) or PDB ID
                      Upload files (.fasta, .pdb, .csv)
                      Descriptive questions (AI will automatically download and analyze required data)

                    **The system will automatically**:
                      Break down complex questions into multiple steps
                      Call appropriate analysis tools
                      Download necessary sequences or structures from databases
                      Integrate multiple analysis results into a comprehensive report

                    **Note**: This is a research demo with limited compute resources (16GB RAM, 4 vCPUs, no GPU) — please avoid submitting very large-scale computational tasks.
                        """)
            
            with gr.Accordion("💡 Example Research Questions", open=True):
                gr.Examples(
                    examples=[
                        ["Can you predict the stability of the protein Catalase (UniProt ID: P04040)?"],
                        ["Retrieve P05067 and determine its likely biological process using GO terms."],
                        ["What is the conservative mutation result of C113 site in P68871 protein?"]
                    ],
                    inputs=chat_input,
                    label=None
                )
            
            with gr.Accordion("📝 Provide Feedback", open=False):
                gr.Markdown("**Your Feedback**")
                feedback_input = gr.Textbox(
                    placeholder="Enter your feedback here...",
                    lines=4,
                    show_label=False
                )
                feedback_submit = gr.Button("Submit", variant="primary", size="sm")
                feedback_status = gr.Markdown(visible=False)
            
            with gr.Row():
                export_btn = gr.Button("Export as HTML", variant="primary", visible=True)
                save_server_btn = gr.DownloadButton("Save to PC", variant="primary", visible=True)
            export_saved_status = gr.Markdown("", visible=False)

        # Feedback submission handler
        feedback_submit.click(
            fn=handle_feedback_submit,
            inputs=[feedback_input],
            outputs=[feedback_input, feedback_status, feedback_status]
        )

        export_btn.click(
            fn=export_chat_history_html,
            inputs=[session_state],
            outputs=[export_saved_status, save_server_btn]
        )
        save_server_btn.click(
            fn=save_chat_history_to_server,
            inputs=[session_state],
            outputs=[save_server_btn]
        )

        # Event handler with concurrency limit
        model_selector.change(
            fn=update_llm_model,
            inputs=[model_selector, session_state],
            outputs=[session_state]
        )
        chat_input.submit(
            fn=send_message,
            inputs=[chatbot, chat_input, session_state],
            outputs=[chatbot, chat_input],
            concurrency_limit=3,
            show_progress="full"
        )

    return {
        "chatbot": chatbot,
        "chat_input": chat_input,
        "session_state": session_state
    }

if __name__ == "__main__":
    components = create_chat_tab({})
    
    with gr.Blocks(theme=gr.themes.Soft(), css=".gradio-container {max-width: 95% !important;}") as demo:
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