import gradio as gr
import json
import os
import re
import requests
import base64
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Mapping
import tempfile
import shutil
import time
from pathlib import Path
from dotenv import load_dotenv
from gradio_client import Client, handle_file
from web.chat_tools import *
import pandas as pd
import uuid
from datetime import datetime

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

# Import prompts from the new file
from web.prompts import PLANNER_PROMPT, WORKER_PROMPT, ANALYZER_PROMPT, FINALIZER_PROMPT

load_dotenv()


class DeepSeekLLM(BaseChatModel):
    api_key: str = None
    base_url: str = "https://api.deepseek.com/v1"
    model_name: str = "deepseek-chat"
    temperature: float = 0.7
    max_tokens: int = 4096
    
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("DEEPSEEK_API_KEY")

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

    @property
    def _llm_type(self) -> str:
        return "deepseek-chat"


class ProteinContextManager:
    def __init__(self):
        self.sequences = {}  # {sequence_id: {'sequence': str, 'timestamp': datetime}}
        self.files = {}      # {file_id: {'path': str, 'type': str, 'timestamp': datetime}}
        self.uniprot_ids = {} # {uniprot_id: timestamp}
        self.last_sequence = None
        self.last_file = None
        self.last_uniprot_id = None
    
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
    
    def get_context_summary(self) -> str:
        summary_parts = []
        
        if self.last_sequence:
            summary_parts.append(f"Most recent sequence: {len(self.last_sequence)} amino acids")
        
        if self.last_file:
            file_name = os.path.basename(self.last_file)
            file_ext = os.path.splitext(file_name)[1]
            summary_parts.append(f"Most recent file: {file_name} ({file_ext})")
        
        if self.last_uniprot_id:
            summary_parts.append(f"Most recent UniProt ID: {self.last_uniprot_id}")
        
        if len(self.sequences) > 1:
            summary_parts.append(f"Total sequences in memory: {len(self.sequences)}")
        if len(self.files) > 1:
            summary_parts.append(f"Total files in memory: {len(self.files)}")
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


def get_tools():
    """Returns a list of all available tool instances."""
    return [
        zero_shot_sequence_prediction_tool,
        zero_shot_structure_prediction_tool,
        protein_function_prediction_tool,
        functional_residue_prediction_tool,
        interpro_query_tool,
        uniprot_query_tool,
        pdb_query_tool,
        protein_properties_generation_tool,
        generate_training_config_tool,
        ai_code_execution_tool
    ]

def create_planner_chain(llm: BaseChatModel, tools: List[BaseTool]):
    """Creates the Planner chain that generates a step-by-step plan."""
    tools_description = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
    planner_prompt_with_tools = PLANNER_PROMPT.partial(tools_description=tools_description)
    return planner_prompt_with_tools | llm | JsonOutputParser()

def create_worker_executor(llm: BaseChatModel, tools: List[BaseTool]):
    """Creates a Worker AgentExecutor for a given set of tools."""
    agent = create_openai_tools_agent(llm, tools, WORKER_PROMPT)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    return executor

def create_finalizer_chain(llm: BaseChatModel):
    """Creates the Finalizer chain to aggregate all analyses into a final report."""
    return FINALIZER_PROMPT | llm | StrOutputParser()

class ConversationManager:
    def __init__(self):
        self.conversations = {}
        self.active_conversation = None

    def create_conversation(self, title: str = None) -> str:
        conv_id = str(uuid.uuid4())
        if not title:
            title = f"Chat {datetime.now().strftime('%H:%M')}"

        llm = DeepSeekLLM(temperature=0.1) # Use lower temp for predictable planning
        all_tools = get_tools()

        # Create the specialized agents for this conversation
        planner_chain = create_planner_chain(llm, all_tools)
        finalizer_chain = create_finalizer_chain(llm)

        # Create a dictionary of workers, each with their specific tool(s)
        # For simplicity, we create one worker per tool. In a real scenario, you might group them.
        workers = {tool.name: create_worker_executor(llm, [tool]) for tool in all_tools}
        
        self.conversations[conv_id] = {
            'id': conv_id,
            'title': title,
            'planner': planner_chain,
            'workers': workers,
            'finalizer': finalizer_chain,
            'memory': ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=10),
            'history': [],
            'created_at': datetime.now(),
            'protein_context': ProteinContextManager(),
        }
        return conv_id

    def get_conversation(self, conv_id: str) -> dict:
        return self.conversations.get(conv_id)

    def delete_conversation(self, conv_id: str):
        if conv_id in self.conversations:
            conv = self.conversations[conv_id]
            temp_files = conv.get('temp_files', [])
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except:
                    pass
            del self.conversations[conv_id]

    def cleanup_temp_files(self, conv_id: str):
        conv = self.get_conversation(conv_id)
        if conv and 'temp_files' in conv:
            for temp_file in conv['temp_files']:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except:
                    pass
            conv['temp_files'] = []

    def list_conversations(self) -> List[dict]:
        return [
            {'id': conv['id'], 'title': conv['title'], 'created_at': conv['created_at']}
            for conv in sorted(self.conversations.values(), key=lambda x: x['created_at'], reverse=True)
        ]

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

conv_manager = ConversationManager()

def create_chat_tab(constant: Dict[str, Any]) -> Dict[str, Any]:
    initial_conv_id = conv_manager.create_conversation("New Chat")
    conv_manager.active_conversation = initial_conv_id

    def create_new_conversation():
        conv_id = conv_manager.create_conversation()
        conv_manager.active_conversation = conv_id
        conversations = conv_manager.list_conversations()
        choices = [(f"{conv['title']} ({conv['created_at'].strftime('%H:%M')})", conv['id']) for conv in conversations]
        return gr.Dropdown(choices=choices, value=conv_id), [], gr.MultimodalTextbox(value=None)

    def switch_conversation(selected_conv_id):
        if not selected_conv_id:
            return [], gr.MultimodalTextbox(value=None)
        conv_manager.active_conversation = selected_conv_id
        conv = conv_manager.get_conversation(selected_conv_id)
        if conv:
            # Convert memory to chat history format
            history = []
            messages = conv['memory'].chat_memory.messages
            
            for i in range(0, len(messages), 2):
                if i + 1 < len(messages):
                    user_msg = messages[i]
                    ai_msg = messages[i + 1]
                    if isinstance(user_msg, HumanMessage) and isinstance(ai_msg, AIMessage):
                        history.append({"role": "user", "content": user_msg.content})
                        history.append({"role": "assistant", "content": ai_msg.content})
            
            return history, gr.MultimodalTextbox(value=None)

        return conv['history'] if conv else [], gr.MultimodalTextbox(value=None)

    def delete_current_conversation(selected_conv_id):
        """Delete the current conversation"""
        if selected_conv_id and len(conv_manager.conversations) > 1:
            conv_manager.delete_conversation(selected_conv_id)
            conversations = conv_manager.list_conversations()
            if conversations:
                new_conv_id = conversations[0]['id']
                conv_manager.active_conversation = new_conv_id
                choices = [(f"{conv['title']} ({conv['created_at'].strftime('%H:%M')})", conv['id']) for conv in conversations]
                
                return (
                    gr.Dropdown(choices=choices, value=new_conv_id),
                    [],
                    gr.MultimodalTextbox(value=None)
                )
        
        return gr.Dropdown(), [], gr.MultimodalTextbox(value=None)
    
    # This is the new COORDINATOR logic
    def send_message(history, message, selected_conv_id):
        BASE_UPLOAD_DIR = "temp_outputs/upload_data"
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
                    else:
                        print(f"Warning: Gradio temp file not found at {original_temp_path}")
                except Exception as e:
                    print(f"Error processing file: {e}")

        conv_id = selected_conv_id or conv_manager.active_conversation
        conv = conv_manager.get_conversation(conv_id)
        if not conv:
            history.append({"role": "user", "content": text})
            history.append({"role": "assistant", "content": "Error: Conversation not found."})
            yield history, gr.MultimodalTextbox(value=None)
            return

        display_text = text
        if file_paths:
            file_names = ", ".join([os.path.basename(f) for f in file_paths])
            display_text += f"\n📎 *Attached: {file_names}*"
        history.append({"role": "user", "content": display_text})
        conv['history'].append({"role": "user", "content": display_text}) # Save to our own history

        protein_ctx = conv['protein_context']
        sequence = extract_sequence_from_message(text)
        uniprot_id = extract_uniprot_id_from_message(text)
        if sequence: protein_ctx.add_sequence(sequence)
        if uniprot_id: protein_ctx.add_uniprot_id(uniprot_id)
        for fp in file_paths: protein_ctx.add_file(fp)

        # 2. Call Planner
        history.append({"role": "assistant", "content": "🤔 Thinking... Creating a plan..."})
        yield history, gr.MultimodalTextbox(value=None, interactive=False)
        
        planner_input = f"{text}\n\n[CONTEXT: User has uploaded files: {', '.join(file_paths)}]\n[CONTEXT: Protein context: {protein_ctx.get_context_summary()}]"
        
        try:
            plan = conv['planner'].invoke({"input": planner_input})
        except Exception as e:
            history[-1] = {"role": "assistant", "content": f"Sorry, I failed to create a plan. Error: {e}"}
            yield history, gr.MultimodalTextbox(value=None, interactive=True)
            return
        
        # If plan is empty, it means no tools are needed. Just chat.
        if not plan:
            history[-1] = {"role": "assistant", "content": "I can help with that."}
            yield history, gr.MultimodalTextbox(value=None, interactive=False)
            llm = DeepSeekLLM()
            response = llm.invoke(conv['memory'].chat_memory.messages + [HumanMessage(content=text)])
            final_response = response.content
        else:
            # 3. Execute Plan (Coordinator Loop)
            plan_text = "📝 **Plan Created:**\n" + "\n".join([f"**Step {p['step']}**: {p['task_description']}" for p in plan])
            history[-1] = {"role": "assistant", "content": plan_text}
            yield history, gr.MultimodalTextbox(value=None, interactive=False)
            
            step_results = {}
            analysis_log = ""

            for step in plan:
                step_num = step['step']
                task_desc = step['task_description']
                tool_name = step['tool_name']
                tool_input = step['tool_input']

                # Update UI for current step
                history[-1] = {"role": "assistant", "content": f"{plan_text}\n\n---\n\n⏳ **Executing Step {step_num}:** {task_desc}"}
                yield history, gr.MultimodalTextbox(value=None, interactive=False)

                try:
                    # Resolve dependencies
                    for key, value in tool_input.items():
                        if isinstance(value, str) and value.startswith("dependency:"):
                            parts = value.split(':')
                            dep_step, dep_key = int(parts[1].replace('step_','')), parts[2]
                            tool_input[key] = step_results[dep_step]['raw_output']

                    # Get worker and execute
                    worker = conv['workers'].get(tool_name)
                    if not worker: raise ValueError(f"Worker for tool '{tool_name}' not found.")
                    
                    # Worker input needs to be a single string for some agents
                    worker_input_str = json.dumps(tool_input)
                    worker_result = worker.invoke({"input": worker_input_str})
                    raw_output = str(worker_result)
                    step_results[step_num] = {'raw_output': raw_output}
                    analysis_log += f"--- Analysis for Step {step_num}: {task_desc} ---\n\n"

                    # Update UI with step completion
                    history[-1] = {"role": "assistant", "content": f"{plan_text}\n\n---\n\n✅ **Step {step_num} Complete:** {task_desc}"}
                    yield history, gr.MultimodalTextbox(value=None, interactive=False)

                except Exception as e:
                    error_message = f"❌ **Error in Step {step_num}:** {task_desc}\n`{str(e)}`"
                    history[-1] = {"role": "assistant", "content": f"{plan_text}\n\n---\n\n{error_message}"}
                    yield history, gr.MultimodalTextbox(value=None, interactive=True)
                    # Stop execution on error
                    return

            # 4. Finalize Report
            history[-1] = {"role": "assistant", "content": f"{plan_text}\n\n---\n\n📄 **All steps complete. Generating final report...**"}
            yield history, gr.MultimodalTextbox(value=None, interactive=False)

            final_response = conv['finalizer'].invoke({
                "original_input": text,
                "analysis_log": analysis_log
            })

        # --- COORDINATOR WORKFLOW END ---
        
        history[-1] = {"role": "assistant", "content": final_response}
        conv['history'].append({"role": "assistant", "content": final_response})
        # Update Langchain memory
        conv['memory'].save_context({"input": display_text}, {"output": final_response})
        
        yield history, gr.MultimodalTextbox(value=None, interactive=True, file_count="multiple")

    with gr.Blocks() as demo:
        custom_css = """
        <style>
        .claude-sidebar-actions {
            display: flex;
            flex-direction: column;
            gap: 8px;
            padding: 8px;
        }
        .claude-action-btn {
            padding: 10px 12px !important;
            border-radius: 8px !important;
            font-size: 14px !important;
            font-weight: 500;
            line-height: 1 !important;
            justify-content: center;
            transition: background-color 0.2s ease, border-color 0.2s ease;
        }
        .claude-action-btn.primary {
            background-color: #2563eb !important;
            color: white !important;
            border: none !important;
        }
        .claude-action-btn.primary:hover {
            background-color: #1d4ed8 !important;
        }
        .claude-action-btn.secondary {
            background-color: #f3f4f6 !important;
            color: #374151 !important;
            border: 1px solid #d1d5db !important;
        }
        .claude-action-btn.secondary:hover {
            background-color: #e5e7eb !important;
            border-color: #9ca3af !important;
        }
        </style>
        """
        with gr.Column():
            gr.HTML(custom_css)
            gr.Markdown("# 🧬 VenusFactory AI Assistant", elem_classes="chat-title")
            with gr.Row(elem_classes="chat-main-container"):
                with gr.Column(scale=1, elem_classes="claude-sidebar-container") as sidebar:
                    with gr.Column(elem_classes="claude-sidebar-actions"):
                        new_chat_btn = gr.Button("➕ New Chat", variant="primary", size="sm", elem_classes="claude-action-btn new-chat-btn")
                        delete_chat_btn = gr.Button("🗑️ Delete Chat", variant="secondary", size="sm", elem_classes="claude-action-btn delete-chat-btn")

                    with gr.Column(elem_classes="sidebar-content-area"):
                        conversations = conv_manager.list_conversations()
                        choices = [(f"{conv['title']} ({conv['created_at'].strftime('%H:%M')})", conv['id']) for conv in conversations]
                        conversation_dropdown = gr.Dropdown(choices=choices, value=initial_conv_id, label="Conversations", interactive=True)
                
                with gr.Column(scale=4, elem_classes="main-chat-container"):
                    chatbot = gr.Chatbot(
                        label="VenusFactory AI Assistant",
                        type="messages",
                        height=650,
                        show_label=False,
                        avatar_images=(None, "https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/venus_logo.png"),
                        bubble_full_width=False,
                        show_copy_button=True,
                        elem_classes="main-chatbot"
                    )
                    chat_input = gr.MultimodalTextbox(
                        interactive=True,
                        file_count="multiple",
                        placeholder="💬 Ask me about protein engineering, upload files (FASTA, PDB), or request analysis...",
                        show_label=False,
                        file_types=[".fasta", ".fa", ".pdb", ".csv"],
                        elem_classes="chat-input"
                    )

        # Event handlers
        new_chat_btn.click(
            fn=create_new_conversation,
            outputs=[conversation_dropdown, chatbot, chat_input]
        )
        delete_chat_btn.click(
            fn=delete_current_conversation,
            inputs=[conversation_dropdown],
            outputs=[conversation_dropdown, chatbot, chat_input]
        )
        conversation_dropdown.change(
            fn=switch_conversation,
            inputs=[conversation_dropdown],
            outputs=[chatbot, chat_input]
        )
        chat_input.submit(
            fn=send_message,
            inputs=[chatbot, chat_input, conversation_dropdown],
            outputs=[chatbot, chat_input]
        )

    return {
        "chatbot": chatbot, "chat_input": chat_input, "conversation_dropdown": conversation_dropdown,
        "new_chat_btn": new_chat_btn, "delete_chat_btn": delete_chat_btn
    }

# Example of how to run this if it's the main script
if __name__ == "__main__":
    # Create the Gradio UI from the function
    components = create_chat_tab({})
    # Launch the Gradio app
    with gr.Blocks(theme=gr.themes.Soft(), css=".gradio-container {max-width: 95% !important;}") as demo:
        # This assumes create_chat_tab populates a UI within the block context
        create_chat_tab({})
    
    demo.launch(share=True)