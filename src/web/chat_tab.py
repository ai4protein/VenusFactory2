import gradio as gr
import json
import os
import re
import aiohttp
import asyncio
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
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

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
    base_url: str = "https://www.dmxapi.com/v1"
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.4
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
        return "deepseek-chat"


class ProteinContextManager:
    def __init__(self):
        self.sequences = {}  # {sequence_id: {'sequence': str, 'timestamp': datetime}}
        self.files = {}      # {file_id: {'path': str, 'type': str, 'timestamp': datetime}}
        self.uniprot_ids = {} # {uniprot_id: timestamp}
        self.structure_files = {} # {structure_id: {'path': str, 'source': str, 'uniprot_id': str, 'timestamp': datetime}}
        self.last_sequence = None
        self.last_file = None
        self.last_uniprot_id = None
        self.last_structure = None
    
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
            file_name = os.path.basename(self.last_file)
            file_ext = os.path.splitext(file_name)[1]
            summary_parts.append(f"Most recent file: {file_name} ({file_ext})")
        
        if self.last_uniprot_id:
            summary_parts.append(f"Most recent UniProt ID: {self.last_uniprot_id}")
        
        if self.last_structure:
            struct_name = os.path.basename(self.last_structure)
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
        ai_code_execution_tool,
        ncbi_sequence_download_tool,
        alphafold_structure_download_tool
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
        handle_parsing_errors=True,
        max_iterations=3, 
        max_execution_time=300,
    )
    return executor

def create_finalizer_chain(llm: BaseChatModel):
    """Creates the Finalizer chain to aggregate all analyses into a final report."""
    return FINALIZER_PROMPT | llm | StrOutputParser()

def send_feedback_email(feedback_text: str) -> str:
    """Send feedback email via SMTP"""
    if not feedback_text.strip():
        return "❌ Please enter your feedback before submitting."
    
    sender_email = "zmm.zlr@qq.com"
    receiver_email = "zlr_zmm@163.com"
    subject = "VenusFactory User Feedback"
    password = os.getenv("EMAIL_PASSWORD")

    server = None
    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject

        body = f"""
VenusFactory User Feedback
--------------------------
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Feedback:
{feedback_text}
--------------------------
        """
        msg.attach(MIMEText(body, 'plain', 'utf-8'))

        smtp_server = "smtp.qq.com"
        smtp_port = 465
        
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        server.login(sender_email, password)
        server.send_message(msg)
        
        return "✅ Thank you for your feedback! Email sent successfully."

    except Exception as e:
        return f"❌ Failed to send feedback: {str(e)}"

    finally:
        if server:
            try:
                server.quit()
            except Exception:
                pass

def initialize_session_state() -> Dict[str, Any]:
    """Initialize a new session state with all necessary components"""
    llm = DeepSeekLLM(temperature=0.1)
    all_tools = get_tools()

    planner_chain = create_planner_chain(llm, all_tools)
    finalizer_chain = create_finalizer_chain(llm)
    workers = {tool.name: create_worker_executor(llm, [tool]) for tool in all_tools}
    
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
        'created_at': datetime.now()
    }


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

    # Call Planner
    history.append({"role": "assistant", "content": "🤔 Thinking... Creating a plan..."})
    yield history, gr.MultimodalTextbox(value=None, interactive=False)

    # Build comprehensive context
    context_parts = []
    if file_paths:
        context_parts.append(f"User has uploaded files: {', '.join(file_paths)}")
    
    if protein_ctx.structure_files:
        struct_info = []
        for struct_id, struct_data in protein_ctx.structure_files.items():
            struct_info.append(f"{struct_data['source']} structure: {struct_data['name']} (path: {struct_data['path']})")
        context_parts.append(f"Available structure files: {'; '.join(struct_info)}")
    
    context_parts.append(f"Protein context: {protein_ctx.get_context_summary()}")
    
    planner_input = f"{text}\n\n[CONTEXT: {'; '.join(context_parts)}]"
    
    try:
        # Async planner invocation
        plan = await asyncio.to_thread(session_state['planner'].invoke, {"input": planner_input})
    except Exception as e:
        history[-1] = {"role": "assistant", "content": f"❌ **Planning Failed:** Sorry, I failed to create a plan. Error: {e}"}
        yield history, gr.MultimodalTextbox(value=None, interactive=True)
        return
    
    # If plan is empty, just chat
    if not plan:
        history[-1] = {"role": "assistant", "content": "I can help with that! I'm generating answers, please be patient"}
        yield history, gr.MultimodalTextbox(value=None, interactive=False)
        
        llm = session_state['llm']
        response = await llm.ainvoke(session_state['memory'].chat_memory.messages + [HumanMessage(content=text)])
        final_response = response.content
    else:
        # Execute Plan
        plan_text = "📋 **Plan Created:**\n" + "\n".join([f"**Step {p['step']}**: {p['task_description']}" for p in plan])
        history[-1] = {"role": "assistant", "content": plan_text}
        yield history, gr.MultimodalTextbox(value=None, interactive=False)
        
        step_results = {}
        analysis_log = ""

        for step in plan:
            step_num = step['step']
            task_desc = step['task_description']
            tool_name = step['tool_name']
            tool_input = step['tool_input']

            history[-1] = {"role": "assistant", "content": f"{plan_text}\n\n---\n\n⏳ **Executing Step {step_num}:** {task_desc}"}
            yield history, gr.MultimodalTextbox(value=None, interactive=False)

            try:
                # Resolve dependencies
                for key, value in tool_input.items():
                    if isinstance(value, str) and value.startswith("dependency:"):
                        parts = value.split(':')
                        dep_step = int(parts[1].replace('step_', '').replace('step', ''))
                        
                        raw_output = step_results[dep_step]['raw_output']

                        if len(parts) > 2:
                            field_name = parts[2]
                            try:
                                parsed = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
                                tool_input[key] = parsed.get(field_name, raw_output)
                            except:
                                tool_input[key] = raw_output
                        else:
                            tool_input[key] = raw_output

                # Get worker and execute (run in thread to avoid blocking)
                worker = session_state['workers'].get(tool_name)
                if not worker: 
                    raise ValueError(f"Worker for tool '{tool_name}' not found.")
                
                worker_input_str = json.dumps(tool_input)
                worker_result = await asyncio.to_thread(worker.invoke, {"input": worker_input_str})
                
                raw_output = str(worker_result)
                step_results[step_num] = {'raw_output': raw_output}
                
                # Parse tool output to update context
                try:
                    if tool_name in ['ncbi_sequence_download', 'alphafold_structure_download']:
                        output_data = json.loads(raw_output)
                        if output_data.get('success') and 'file_path' in output_data:
                            file_path = output_data['file_path']
                            if tool_name == 'alphafold_structure_download':
                                uniprot_id = tool_input.get('uniprot_id', 'unknown')
                                protein_ctx.add_structure_file(file_path, 'alphafold', uniprot_id)
                            elif tool_name == 'ncbi_sequence_download':
                                protein_ctx.add_file(file_path)
                except (json.JSONDecodeError, KeyError):
                    pass
                
                # Create detailed step result display
                step_detail = f"**Step {step_num}:** {task_desc}\n\n"
                step_detail += f"**Tool:** {tool_name}\n"
                step_detail += f"**Input:** {json.dumps(tool_input, indent=2)}\n\n"
                step_detail += f"**Output:**\n```\n{raw_output[:500]}{'...' if len(raw_output) > 500 else ''}\n```"
                
                analysis_log += f"--- Analysis for Step {step_num}: {task_desc} ---\n\n"
                analysis_log += f"Tool: {tool_name}\n"
                analysis_log += f"Input: {json.dumps(tool_input, indent=2)}\n"
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

        final_response = await asyncio.to_thread(
            session_state['finalizer'].invoke,
            {"original_input": text, "analysis_log": analysis_log}
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
                height=600,
                show_label=False,
                avatar_images=(None, "https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/venus_logo.png"),
                bubble_full_width=False,
                show_copy_button=True
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

        # Feedback submission handler
        def handle_feedback_submit(feedback_text):
            result = send_feedback_email(feedback_text)
            return "", result, gr.update(visible=True)
        
        feedback_submit.click(
            fn=handle_feedback_submit,
            inputs=[feedback_input],
            outputs=[feedback_input, feedback_status, feedback_status]
        )

        # Event handler with concurrency limit
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