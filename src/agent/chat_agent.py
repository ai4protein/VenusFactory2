"""
Chat agent: LLM, planner/worker/finalizer chains, session state, and tool cache.
Used by web.chat_tab for the Agent chat UI.
"""
import hashlib
import json
import os
import time
import uuid
from collections.abc import Sequence
from copy import copy
from datetime import datetime
from typing import Any

import aiohttp
from dotenv import load_dotenv
from langchain.tools import BaseTool
from langchain_classic.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langgraph.prebuilt import create_react_agent

from agent.prompts import (
    CB_PLANNER_PROMPT,
    CB_STEP_PLANNING,
    MLS_DEBUG_PROMPT,
    MLS_POST_STEP_CHECK,
    MLS_PROMPT,
    PI_CHAT_PROMPT,
    PI_CLARIFICATION_PROMPT,
    PI_FINAL_REPORT_PROMPT,
    PI_PROMPT,
    PI_RESEARCH_PROMPT,
    PI_SECTIONS_PROMPT,
    PI_SUB_REPORT_PROMPT,
    PI_SUGGEST_STEPS_PROMPT,
    SC_PROMPT,
)
from agent.skills import get_skills_metadata_string
from config import get_config
from logger import get_logger
from tools.tools_agent_hub import get_pi_tools, get_tools
from web.utils.common_utils import get_web_v2_area_dir

load_dotenv()

_logger = get_logger("agent.chat")
_cfg = get_config()

_ONLINE_DISABLED_AGENT_TOOL_NAMES = {
    # Training-related tools
    "generate_training_config",
    "train_protein_model",
    "protein_model_predict",
    # Protein-discovery related tools
    "download_foldseek_results_by_pdb_file",
    "query_foldseek_search_by_pdb_file",
}


def _is_online_mode() -> bool:
    return _cfg.server.is_online


def _filter_agent_tools_for_runtime_mode(tools: list[BaseTool]) -> tuple[list[BaseTool], list[str]]:
    """Filter agent tools by runtime mode and return (enabled_tools, disabled_tool_names)."""
    if not _is_online_mode():
        return tools, []

    disabled: list[str] = []
    enabled: list[BaseTool] = []
    for tool in tools:
        name = getattr(tool, "name", "") or ""
        if name in _ONLINE_DISABLED_AGENT_TOOL_NAMES or "foldseek" in name.lower():
            disabled.append(name)
            continue
        enabled.append(tool)
    return enabled, sorted(set(disabled))


class _ChatBufferWindowMemory:
    """In-process chat history keeping last k message pairs (2k messages).
    Replaces deprecated ConversationBufferWindowMemory; same interface for chat_memory.messages and save_context."""
    __slots__ = ("_messages", "_k", "chat_memory")

    def __init__(self, k: int = 10):
        self._messages: list[BaseMessage] = []
        self._k = k
        self.chat_memory = _MessageListRef(self)

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
        user = inputs.get("input", "")
        assistant = outputs.get("output", "")
        self._messages.append(HumanMessage(content=user))
        self._messages.append(AIMessage(content=assistant))
        if self._k > 0:
            self._messages = self._messages[-self._k * 2 :]


class _MessageListRef:
    """Exposes trimmed messages for compatibility with memory.chat_memory.messages."""
    __slots__ = ("_parent",)

    def __init__(self, parent: _ChatBufferWindowMemory):
        self._parent = parent

    @property
    def messages(self) -> list[BaseMessage]:
        m = self._parent._messages
        k = self._parent._k
        return m[-k * 2 :] if k > 0 else list(m)


def _tools_to_openai_schema(tools: Sequence) -> list[dict[str, Any]]:
    """Convert LangChain tools to OpenAI tool schema for chat/completions."""
    out = []
    for t in tools:
        name = getattr(t, "name", None) or ""
        desc = getattr(t, "description", None) or ""
        schema = getattr(t, "args_schema", None)
        params = {"type": "object", "properties": {}, "required": []}
        if schema:
            fields = getattr(schema, "model_fields", None) or getattr(schema, "__fields__", None) or {}
            for fname, finfo in fields.items():
                params["properties"][fname] = {"description": getattr(finfo, "description", None) or ""}
                if getattr(finfo, "default", None) in (None, ...) or getattr(getattr(finfo, "field_info", None), "default", None) in (None, ...):
                    params["required"].append(fname)
        out.append({"type": "function", "function": {"name": name, "description": desc, "parameters": params}})
    return out


class Chat_LLM(BaseChatModel):
    api_key: str = None
    base_url: str = "https://www.dmxapi.cn/v1"
    model_name: str = "gemini-2.5-pro"
    temperature: float = 0.2
    max_tokens: int = 8192
    _bound_tools: list | None = None

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        llm_cfg = _cfg.llm
        self.api_key = self.api_key or llm_cfg.api_key
        self.base_url = kwargs.get("base_url") or llm_cfg.base_url
        self.model_name = kwargs.get("model_name") or llm_cfg.model_name
        if "max_tokens" not in kwargs and llm_cfg.max_tokens:
            self.max_tokens = llm_cfg.max_tokens

    def bind_tools(self, tools: Sequence, **kwargs: Any):
        """Return a copy of this model with tools bound so that _generate can send tools and parse tool_calls."""
        obj = copy(self)
        obj._bound_tools = list(tools) if tools else None
        return obj

    def _generate(
        self, messages: list[BaseMessage], stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any,) -> ChatResult:
        if not self.api_key:
            raise ValueError("OpenAI API key is not configured.")

        message_dicts = self._build_message_dicts(messages)
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
        if getattr(self, "_bound_tools", None):
            payload["tools"] = _tools_to_openai_schema(self._bound_tools)

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

        content = message_data.get('content', '') or ''
        tool_calls = message_data.get('tool_calls')
        if tool_calls:
            tc_list = [{"id": tc.get("id"), "name": tc.get("function", {}).get("name"), "args": json.loads(tc.get("function", {}).get("arguments", "{}") or "{}")} for tc in tool_calls]
            ai_message = AIMessage(content=content if content is not None else "", additional_kwargs={**message_data, "tool_calls": tc_list})
            if hasattr(ai_message, "tool_calls"):
                try:
                    ai_message.tool_calls = tc_list
                except Exception:
                    pass
        else:
            ai_message = AIMessage(content=content, additional_kwargs=message_data)

        generation = ChatGeneration(message=ai_message)
        return ChatResult(generations=[generation])

    def _stream(
        self, messages: list[BaseMessage], stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any,
    ):
        if not self.api_key:
            raise ValueError("OpenAI API key is not configured.")

        message_dicts = self._build_message_dicts(messages)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": message_dicts,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True,
            **kwargs,
        }
        if getattr(self, "_bound_tools", None):
            payload["tools"] = _tools_to_openai_schema(self._bound_tools)

        import requests as req_lib
        response = req_lib.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,
            stream=True,
        )
        if response.status_code != 200:
            raise RuntimeError(f"API request failed: {response.status_code} - {response.text}")

        for line in response.iter_lines():
            if not line:
                continue
            text = line.decode("utf-8").strip()
            if not text.startswith("data: "):
                continue
            data = text[6:]
            if data.strip() == "[DONE]":
                break
            try:
                chunk_data = json.loads(data)
                delta = chunk_data["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    chunk = ChatGenerationChunk(
                        message=AIMessageChunk(content=content)
                    )
                    if run_manager:
                        run_manager.on_llm_new_token(content, chunk=chunk)
                    yield chunk
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    def _build_message_dicts(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
        message_dicts = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                message_dicts.append({"role": "user", "content": msg.content or ""})
            elif isinstance(msg, SystemMessage):
                message_dicts.append({"role": "system", "content": msg.content or ""})
            elif isinstance(msg, AIMessage):
                entry = {"role": "assistant", "content": msg.content or ""}
                tool_calls = getattr(msg, "tool_calls", None) or msg.additional_kwargs.get("tool_calls") if hasattr(msg, "additional_kwargs") else None
                if tool_calls:
                    entry["tool_calls"] = [{"id": tc.get("id", ""), "type": "function", "function": {"name": tc.get("name", ""), "arguments": json.dumps(tc.get("args", {}))}} if isinstance(tc, dict) else {"id": getattr(tc, "id", ""), "type": "function", "function": {"name": getattr(tc, "name", ""), "arguments": json.dumps(getattr(tc, "args", {}) or {})}} for tc in tool_calls]
                message_dicts.append(entry)
            elif type(msg).__name__ == "ToolMessage":
                message_dicts.append({"role": "tool", "tool_call_id": getattr(msg, "tool_call_id", ""), "content": getattr(msg, "content", "") or ""})
            else:
                message_dicts.append({"role": "user", "content": str(getattr(msg, "content", msg))})
        return message_dicts

    async def _agenerate(
        self, messages: list[BaseMessage], stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any,) -> ChatResult:
        """Asynchronous generation for concurrent execution"""
        if not self.api_key:
            raise ValueError("OpenAI API key is not configured.")

        message_dicts = self._build_message_dicts(messages)
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
        if getattr(self, "_bound_tools", None):
            payload["tools"] = _tools_to_openai_schema(self._bound_tools)

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

                content = message_data.get('content', '') or ''
                tool_calls = message_data.get('tool_calls')
                if tool_calls:
                    tc_list = [{"id": tc.get("id"), "name": tc.get("function", {}).get("name"), "args": json.loads(tc.get("function", {}).get("arguments", "{}") or "{}")} for tc in tool_calls]
                    ai_message = AIMessage(content=content if content is not None else "", additional_kwargs={**message_data, "tool_calls": tc_list})
                    if hasattr(ai_message, "tool_calls"):
                        try:
                            ai_message.tool_calls = tc_list
                        except Exception:
                            pass
                else:
                    ai_message = AIMessage(content=content, additional_kwargs=message_data)

                generation = ChatGeneration(message=ai_message)
                return ChatResult(generations=[generation])

    async def ainvoke(self, messages: list[BaseMessage], **kwargs) -> AIMessage:
        """Async invoke method"""
        result = await self._agenerate(messages, **kwargs)
        return result.generations[0].message

    @property
    def _llm_type(self) -> str:
        return "chat-llm"


def generate_cache_key(tool_name: str, tool_input: dict) -> str:
    input_str = json.dumps(tool_input, sort_keys=True, ensure_ascii=False)
    params_hash = hashlib.md5(input_str.encode('utf-8')).hexdigest()[:12]
    return f"{tool_name}_{params_hash}"


def _merge_tool_parameters_with_context(protein_ctx: "ProteinContextManager", base_params: Any) -> dict[str, Any]:
    """Merge tool input parameters with current protein context (files/sequence/UniProt)."""
    if isinstance(base_params, dict):
        params = dict(base_params)
    elif isinstance(base_params, str):
        text = base_params.strip()
        parsed = None
        if text.startswith("{") and text.endswith("}"):
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
        if isinstance(parsed, dict):
            params = dict(parsed)
        else:
            params = {"input": base_params}
    elif isinstance(base_params, (list, tuple)):
        params = {"items": list(base_params)}
    elif base_params is None:
        params = {}
    else:
        params = {"input": base_params}
    try:
        params.setdefault("last_sequence", getattr(protein_ctx, "last_sequence", None))
        params.setdefault("last_uniprot_id", getattr(protein_ctx, "last_uniprot_id", None))
        params.setdefault("last_file", getattr(protein_ctx, "last_file", None))
        all_files = []
        for _, f in getattr(protein_ctx, "files", {}).items():
            if f.get("path"):
                all_files.append(f["path"])
        params.setdefault("files", sorted(list(set(all_files))))
    except Exception:
        pass
    return params


def _format_tool_input_summary(tool_input: dict, max_len: int = 400) -> str:
    """One-line summary of tool input for logging."""
    if not tool_input:
        return "{}"
    try:
        s = json.dumps(tool_input, ensure_ascii=False, sort_keys=True)
        return s if len(s) <= max_len else s[:max_len] + "..."
    except Exception:
        return str(tool_input)[:max_len]


def get_cached_tool_result(session_state: dict, tool_name: str, tool_input: dict) -> dict[str, Any] | None:
    cache_key = generate_cache_key(tool_name, tool_input)
    cache = session_state.get("tool_cache", {})
    cached_result = cache.get(cache_key)
    inputs_summary = _format_tool_input_summary(tool_input)
    if cached_result:
        cached_inputs = cached_result.get("inputs", {})
        if cached_inputs == tool_input:
            _logger.debug("Cache HIT: %s | key=%s | inputs=%s", tool_name, cache_key, inputs_summary)
            return cached_result
        _logger.debug("Cache key collision for %s, inputs differ", cache_key)
        return None
    _logger.debug("Cache MISS: %s | key=%s | inputs=%s", tool_name, cache_key, inputs_summary)
    return None


def save_cached_tool_result(session_state: dict, tool_name: str, tool_input: dict, outputs: Any) -> bool:
    from agent.chat_agent_utils import _tool_output_indicates_failure

    is_failure, reason = _tool_output_indicates_failure(outputs)
    if is_failure:
        _logger.debug("Tool execution failed, not caching result for %s: %s", tool_name, reason)
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
    _logger.debug("Cached successful result for %s", tool_name)
    return True


class ProteinContextManager:
    def __init__(self, max_tool_history=200):
        self.sequences = {}
        self.files = {}
        self.uniprot_ids = {}
        self.structure_files = {}
        self.last_sequence = None
        self.last_file = None
        self.last_uniprot_id = None
        self.last_structure = None
        self.tool_history = []
        self.max_tool_history = max_tool_history

    def add_tool_call(self, step: int, tool_name: str, inputs: dict, outputs: Any, cached: bool = False):
        merged_params = _merge_tool_parameters_with_context(self, inputs)
        cache_key = generate_cache_key(tool_name, merged_params)
        tool_record = {
            'step': step,
            'name': tool_name,
            'parameters': merged_params,
            'outputs': outputs,
            'cached': cached,
            'cache_key': cache_key,
            'timestamp': datetime.now().isoformat()
        }
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
        if len(self.tool_history) > self.max_tool_history:
            self.tool_history.pop(0)

    def get_tool_records(self, limit: int = None) -> list[dict[str, Any]]:
        history_slice = self.tool_history[-limit:] if limit else self.tool_history
        records = []
        for call in history_slice:
            rec = call.get('tool_record')
            if rec:
                records.append(rec)
            else:
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
        struct_id = f"struct_{len(self.structure_files) + 1}"
        self.structure_files[struct_id] = {
            'path': file_path,
            'source': source,
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
            summary_parts.append(f"Most recent structure: {self.last_structure}")
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


def _format_tool_with_params(tool: BaseTool) -> str:
    """Format a single tool as 'name: description. Parameters: param1 (description/default), ...' using args_schema if available."""
    base = f"- **{tool.name}**: {tool.description or '(no description)'}"
    schema = getattr(tool, "args_schema", None)
    if schema is None:
        return base
    fields = getattr(schema, "model_fields", None) or getattr(schema, "__fields__", None)
    if not fields:
        return base
    parts = []
    for fname, finfo in fields.items():
        # Pydantic v2: FieldInfo has .description, .default; v1: ModelField has .field_info
        desc = getattr(finfo, "description", None) or getattr(getattr(finfo, "field_info", None), "description", None)
        desc = desc or ""
        default = getattr(finfo, "default", None) or getattr(getattr(finfo, "field_info", None), "default", None)
        if default is not None and default != ...:
            default_str = repr(default) if not isinstance(default, str) else f'"{default}"'
            parts.append(f"  - **{fname}**: {desc or 'optional'} (default: {default_str})")
        else:
            parts.append(f"  - **{fname}**: {desc or 'required'}")
    if parts:
        return base + "\n  Parameters:\n" + "\n".join(parts)
    return base


def _build_tools_description(tools: list[BaseTool], with_params: bool = True) -> str:
    """Build tools_description for prompts; if with_params, include each tool's optional parameters from args_schema."""
    if not with_params:
        return "\n".join([f"- {t.name}: {t.description}" for t in tools])
    return "\n\n".join([_format_tool_with_params(t) for t in tools])


def create_planner_chain(llm: BaseChatModel, tools: list[BaseTool]):
    tools_description = _build_tools_description(tools)
    planner_prompt_with_tools = PI_PROMPT.partial(tools_description=tools_description)
    return planner_prompt_with_tools | llm | JsonOutputParser()


class LangGraphAgentExecutorWrapper:
    """Wrapper that adapts LangGraph's create_react_agent to match the legacy AgentExecutor interface."""
    def __init__(self, llm, tools, prompt):
        self.prompt_template = prompt
        self.graph = create_react_agent(llm, tools=tools)

    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        inputs_for_prompt = {**inputs, "agent_scratchpad": []}
        messages = self.prompt_template.invoke(inputs_for_prompt).to_messages()

        # Add recursion_limit to prevent runaway loops (e.g. agent repeatedly calling agent_generated_code)
        config = {"recursion_limit": 10}

        try:
            result = self.graph.invoke({"messages": messages}, config=config)
        except Exception as e:
            # If we hit GraphRecursionError, we catch it and use the messages we got so far
            if type(e).__name__ == "GraphRecursionError":
                _logger.warning("AgentExecutorWrapper: GraphRecursionError, loop limit reached")
                # Fallback: trying to get messages from exception, though usually it's raised before updating state.
                # If unavailable, we just return an error output
                return {"output": "Agent Execution stopped: reached maximum step limit. The tool might have succeeded, but the agent was looping.", "intermediate_steps": []}
            raise e

        intermediate_steps = []
        out_messages = result["messages"]
        new_messages = out_messages[len(messages):]

        class _AgentAction:
            def __init__(self, tool, tool_input):
                self.tool = tool
                self.tool_input = tool_input

        for i, msg in enumerate(new_messages):
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                for tool_call in msg.tool_calls:
                    action = _AgentAction(tool=tool_call["name"], tool_input=tool_call["args"])
                    obs_str = ""
                    for next_msg in new_messages[i+1:]:
                        if type(next_msg).__name__ == "ToolMessage" and getattr(next_msg, "tool_call_id", "") == tool_call["id"]:
                            obs_str = getattr(next_msg, "content", "")
                            break
                    intermediate_steps.append((action, obs_str))

        final_output = ""
        if new_messages and isinstance(new_messages[-1], AIMessage) and not getattr(new_messages[-1], "tool_calls", None):
            final_output = getattr(new_messages[-1], "content", "")

        return {"output": final_output, "intermediate_steps": intermediate_steps}


def create_pi_research_agent(llm: BaseChatModel, pi_tools: list[BaseTool]):
    """PI research phase: PI runs search tools (literature, web, dataset) itself in a multi-turn loop, then outputs the research report. Used before CB/MLS execution."""
    tools_description = _build_tools_description(pi_tools)
    prompt = PI_RESEARCH_PROMPT.partial(tools_description=tools_description)
    return LangGraphAgentExecutorWrapper(llm, pi_tools, prompt)


def create_worker_executor(llm: BaseChatModel, tools: list[BaseTool], all_tools: list[BaseTool] | None = None):
    """Worker executes tools: use MLS (Machine Learning Specialist), not CB. Pass all_tools so MLS sees full tool list and skills meta for self-check."""
    tool = tools[0] if isinstance(tools, list) and tools else None
    tool_desc = _format_tool_with_params(tool) if tool else "(no tool)"
    available_tools_list = ", ".join(t.name for t in (all_tools or tools)) if (all_tools or tools) else "(none)"
    available_skills_meta = get_skills_metadata_string()
    worker_prompt = MLS_PROMPT.partial(
        tool_name=(tool.name if tool else "tool"),
        tool_description=(tool_desc if tool else ""),
        available_tools_list=available_tools_list,
        available_skills_meta=available_skills_meta,
        machine_learning_specialist_post_step_check=MLS_POST_STEP_CHECK,
    )
    return LangGraphAgentExecutorWrapper(llm, tools, worker_prompt)


def create_mls_debug_executor(llm: BaseChatModel, debug_tools: list[BaseTool]):
    """MLS self-check with tools: may call read_skill, python_repl, agent_generated_code, etc. to diagnose/fix, then output retry_input or report_for_cb."""
    return LangGraphAgentExecutorWrapper(llm, debug_tools, MLS_DEBUG_PROMPT)


def create_finalizer_chain(llm: BaseChatModel):
    from copy import copy
    finalizer_llm = copy(llm)
    finalizer_llm.max_tokens = 16384
    return SC_PROMPT | finalizer_llm | StrOutputParser()


def create_pi_report_chain(llm: BaseChatModel, tools_description: str):
    """PI: fallback report from search results → report (Abstract, Introduction, Related Work, Tools, Methods, References)."""
    prompt = PI_PROMPT.partial(tools_description=tools_description)
    return prompt | llm | StrOutputParser()


def create_pi_sections_chain(llm: BaseChatModel):
    """PI: user question → JSON array of up to 5 sections (section_name, search_query, focus)."""
    return PI_SECTIONS_PROMPT | llm | StrOutputParser()


def create_pi_clarification_chain(llm: BaseChatModel):
    """PI: user question + sections → JSON array of 2-4 clarification questions."""
    return PI_CLARIFICATION_PROMPT | llm | StrOutputParser()


def create_pi_sub_report_chain(llm: BaseChatModel):
    """PI: one section name + focus + search results → one paragraph sub-report (with citations [1], [2])."""
    return PI_SUB_REPORT_PROMPT | llm | StrOutputParser()


def create_pi_final_report_chain(llm: BaseChatModel):
    """PI: all sub-reports + user question → research draft (Abstract, Introduction, Related Work, References)."""
    return PI_FINAL_REPORT_PROMPT | llm | StrOutputParser()


def create_pi_suggest_steps_chain(llm: BaseChatModel):
    """PI: draft report + user input → Suggest steps (Tools + Steps for CB/MLS)."""
    return PI_SUGGEST_STEPS_PROMPT | llm | StrOutputParser()


def create_cb_planner_chain(llm: BaseChatModel, tools_description: str, all_tools: list[BaseTool] | None = None):
    """CB: PI report → concrete JSON pipeline (same computational_biologist prompt, Mode A)."""
    available_tools_list = ", ".join(t.name for t in all_tools) if all_tools else "(none)"
    available_skills_meta = get_skills_metadata_string()
    prompt = CB_PLANNER_PROMPT.partial(
        tools_description=tools_description,
        tool_name="(planning mode)",
        tool_description="(N/A)",
        available_tools_list=available_tools_list,
        skills_metadata=available_skills_meta,
        computational_biologist_step_planning=CB_STEP_PLANNING,
        machine_learning_specialist_post_step_check=MLS_POST_STEP_CHECK,
    )
    return prompt | llm | JsonOutputParser()

def create_cb_planner_raw_chain(llm: BaseChatModel, tools_description: str, all_tools: list[BaseTool] | None = None):
    """CB: same as planner but returns raw LLM output (for fallback parsing in chat_tab)."""
    available_tools_list = ", ".join(t.name for t in all_tools) if all_tools else "(none)"
    available_skills_meta = get_skills_metadata_string()
    prompt = CB_PLANNER_PROMPT.partial(
        tools_description=tools_description,
        tool_name="(planning mode)",
        tool_description="(N/A)",
        available_tools_list=available_tools_list,
        skills_metadata=available_skills_meta,
        computational_biologist_step_planning=CB_STEP_PLANNING,
        machine_learning_specialist_post_step_check=MLS_POST_STEP_CHECK,
    )
    return prompt | llm


def run_pi_research_step(llm: BaseChatModel, pi_tools: list[BaseTool], messages: list) -> tuple:
    """
    Run one step of PI research: invoke LLM with tools; if it returns tool_calls, execute them and return
    (updated_messages, steps_this_round, is_final, final_content).
    steps_this_round: list of (tool_name, tool_input_dict, observation_str).
    """
    llm_with_tools = llm.bind_tools(pi_tools)
    response = llm_with_tools.invoke(messages)
    tool_calls = getattr(response, "tool_calls", None) or []
    if not tool_calls and hasattr(response, "additional_kwargs"):
        tool_calls = response.additional_kwargs.get("tool_calls") or []
    if not tool_calls:
        content = getattr(response, "content", "") or ""
        return (messages + [response], [], True, content)
    steps = []
    tool_messages = []
    for tc in tool_calls:
        name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
        args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {}) or {}
        tid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
        tool = next((t for t in pi_tools if t.name == name), None)
        if not tool:
            obs = json.dumps({"success": False, "error": f"Unknown tool: {name}"})
        else:
            try:
                _logger.info("PI research invoking: %s with %s", name, args)
                out = tool.invoke(args)
                obs = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
            except Exception as e:
                obs = json.dumps({"success": False, "error": str(e)})
        steps.append((name, args, obs))
        if tid:
            tool_messages.append(ToolMessage(content=obs, tool_call_id=tid))
    new_messages = messages + [response] + tool_messages
    return (new_messages, steps, False, None)


def create_pi_answer_chain(llm: BaseChatModel, tools_description: str = "(Answer mode — no tools.)"):
    """Chain for PI to answer with citations when no tool plan is needed; uses same PI prompt, StrOutputParser."""
    planner_prompt_partial = PI_PROMPT.partial(
        tools_description=tools_description,
        protein_context_summary="",
        tool_outputs="",
    )
    return planner_prompt_partial | llm | StrOutputParser()


def create_pi_chat_chain(llm: BaseChatModel):
    """Chain for PI chat mode: direct responses for greetings and simple conversations."""
    return PI_CHAT_PROMPT | llm | StrOutputParser()


def initialize_session_state() -> dict[str, Any]:
    llm = Chat_LLM(temperature=0.1)
    all_tools_raw = get_tools()
    all_tools, disabled_tool_names = _filter_agent_tools_for_runtime_mode(all_tools_raw)
    pi_tools = get_pi_tools()  # PI can execute only search tools (no download / train / etc.)
    if disabled_tool_names:
        pi_tools = [t for t in pi_tools if getattr(t, "name", "") not in set(disabled_tool_names)]
    tools_description = _build_tools_description(all_tools)
    pi_tools_description = _build_tools_description(pi_tools)
    planner_chain = create_planner_chain(llm, all_tools)
    finalizer_chain = create_finalizer_chain(llm)
    pi_research_agent = create_pi_research_agent(llm, pi_tools)
    pi_report_chain = create_pi_report_chain(llm, pi_tools_description)
    pi_sections_chain = create_pi_sections_chain(llm)
    pi_clarification_chain = create_pi_clarification_chain(llm)
    pi_sub_report_chain = create_pi_sub_report_chain(llm)
    pi_final_report_chain = create_pi_final_report_chain(llm)
    pi_suggest_steps_chain = create_pi_suggest_steps_chain(llm)
    cb_planner_chain = create_cb_planner_chain(llm, tools_description, all_tools=all_tools)
    cb_planner_raw_chain = create_cb_planner_raw_chain(llm, tools_description, all_tools=all_tools)
    pi_answer_chain = create_pi_answer_chain(llm, pi_tools_description)
    pi_chat_chain = create_pi_chat_chain(llm)
    workers = {t.name: create_worker_executor(llm, [t], all_tools=all_tools) for t in all_tools}
    # MLS self-check may use read_skill, python_repl, agent_generated_code, or any other tool to fix the error
    mls_debug_executor = create_mls_debug_executor(llm, all_tools)
    session_id = str(uuid.uuid4())
    agent_session_dir = str(get_web_v2_area_dir("sessions", tool="chat", session_id=session_id))
    os.makedirs(agent_session_dir, exist_ok=True)
    available_tools_list = ", ".join(t.name for t in all_tools) if all_tools else "(none)"
    skills_metadata = get_skills_metadata_string()
    return {
        'session_id': session_id,
        'agent_session_dir': agent_session_dir,
        'tools_description': tools_description,
        'skills_metadata': skills_metadata,
        'available_tools_list': available_tools_list,
        'all_tools': all_tools,
        'disabled_tool_names': disabled_tool_names,
        'planner': planner_chain,
        'pi_research_agent': pi_research_agent,
        'pi_report': pi_report_chain,
        'pi_sections': pi_sections_chain,
        'pi_clarification': pi_clarification_chain,
        'pi_sub_report': pi_sub_report_chain,
        'pi_final_report': pi_final_report_chain,
        'pi_suggest_steps': pi_suggest_steps_chain,
        'cb_planner': cb_planner_chain,
        'cb_planner_raw': cb_planner_raw_chain,
        'workers': workers,
        'mls_debug_executor': mls_debug_executor,
        'finalizer': finalizer_chain,
        'pi_answer': pi_answer_chain,
        'pi_chat': pi_chat_chain,
        'llm': llm,
        'memory': _ChatBufferWindowMemory(k=10),
        'dialogue_memory': [],
        'history': [],
        'conversation_log': [],   # LangChain-managed: every user/assistant message for backend display
        'tool_executions': [],    # Every tool call (including PI literature_search) for backend display
        'protein_context': ProteinContextManager(),
        'temp_files': [],
        'tool_cache': {},
        'execution_failed': False,
        'failed_step': None,
        'failed_reason': None,
        'latest_tool_output_file': None,
        'created_at': datetime.now()
    }


def update_llm_model(selected: str, state: dict[str, Any]) -> dict[str, Any]:
    mapping = {
        "ChatGPT-4o": "gpt-4o",
        "Gemini-2.5-Pro": "gemini-2.5-pro",
        "Claude-3.7": "claude-3-7-sonnet-20250219",
        "DeepSeek-R1": "deepseek-r1-0528"
    }
    if not state or state.get('llm') is None:
        return state
    state['llm'].model_name = mapping.get(selected, "gemini-2.5-pro")
    return state
