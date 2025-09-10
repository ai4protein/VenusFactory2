import gradio as gr
import json
import os
import re
import requests
import base64
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import tempfile
import shutil
import time
from pathlib import Path
from dotenv import load_dotenv
from gradio_client import Client, handle_file
from web.get_interpro_function import download_single_interpro, generate_interpro_ai_summary
import pandas as pd
import uuid
from datetime import datetime

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from langchain.tools.base import BaseTool
from pydantic import BaseModel, Field

load_dotenv()

# Custom DeepSeek LLM implementation
class DeepSeekLLM:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = "https://api.deepseek.com/v1"
    
    def invoke(self, messages: List[dict], **kwargs) -> str:
        if not self.api_key:
            return "Error: No API key configured"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 4000),
            "temperature": kwargs.get("temperature", 0.7)
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                return f"API request failed: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Request error: {str(e)}"

# Tool input schemas
class ZeroShotSequenceInput(BaseModel):
    """Input for zero-shot sequence mutation prediction"""
    sequence: Optional[str] = Field(None, description="Protein sequence in single letter amino acid code")
    fasta_file: Optional[str] = Field(None, description="Path to FASTA file")
    model_name: str = Field(default="ESM2-650M", description="Model name: ESM-1v, ESM2-650M, ESM-1b")

class ZeroShotStructureInput(BaseModel):
    """Input for zero-shot structure mutation prediction"""
    structure_file: str = Field(..., description="Path to PDB structure file")
    model_name: str = Field(default="ESM-IF1", description="Model name: SaProt, ProtSSN, ESM-IF1, MIF-ST, ProSST-2048")

class FunctionPredictionInput(BaseModel):
    """Input for protein function prediction"""
    sequence: Optional[str] = Field(None, description="Protein sequence in single letter amino acid code")
    fasta_file: Optional[str] = Field(None, description="Path to FASTA file")
    model_name: str = Field(default="ESM2-650M", description="Model name for function prediction")
    task: str = Field(default="Solubility", description="Task: Solubility, Localization, Metal ion binding, Stability, Sorting signal, Optimum temperature")

class ResidueFunctionPredictionInput(BaseModel):
    """Input for functional residue prediction"""
    sequence: Optional[str] = Field(None, description="Protein sequence in single letter amino acid code")
    fasta_file: Optional[str] = Field(None, description="Path to FASTA file")
    model_name: str = Field(default="ESM2-650M", description="Model name for function prediction")
    task: str = Field(default="Activate", description="Task: Activate, Binding, Evolutionary, Motif")

class InterProQueryInput(BaseModel):
    """Input for InterPro database query"""
    uniprot_id: str = Field(..., description="UniProt ID for protein function query")

class CSVTrainingConfigInput(BaseModel):
    """Input for CSV training config generation"""
    csv_file: str = Field(..., description="Path to CSV file with training data")
    test_csv_file: Optional[str] = Field(None, description="Optional path to test CSV file")
    output_name: str = Field(default="custom_training_config", description="Name for the generated config")

class ProteinPropertiesInput(BaseModel):
    """Input for protein properties generation"""
    sequence: Optional[str] = Field(None, description="Protein sequence in single letter amino acid code")
    fasta_file: Optional[str] = Field(None, description="Path to PDB structure file or fasta file")
    task_name: str = Field(default="Physical and chemical properties", description="Task name: Physical and chemical properties, Relative solvent accessible surface area (PDB only), SASA value (PDB only), Secondary structure (PDB only)")

# Langchain Tools
@tool("zero_shot_sequence_prediction", args_schema=ZeroShotSequenceInput)
def zero_shot_sequence_prediction_tool(sequence: Optional[str] = None, fasta_file: Optional[str] = None, model_name: str = "ESM2-650M") -> str:
    """Predict beneficial mutations using sequence-based zero-shot models. Use for mutation prediction with protein sequences."""
    try:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        
        if fasta_file:
            if not os.path.exists(fasta_file):
                return f"Error: FASTA file not found at path: {fasta_file}"
            return call_zero_shot_sequence_prediction_from_file(fasta_file, model_name, api_key)
        elif sequence:
            return call_zero_shot_sequence_prediction(sequence, model_name, api_key)
        else:
            return "Error: Either sequence or fasta_file must be provided"
    except Exception as e:
        return f"Zero-shot sequence prediction error: {str(e)}"

@tool("zero_shot_structure_prediction", args_schema=ZeroShotStructureInput)
def zero_shot_structure_prediction_tool(structure_file: str, model_name: str = "ESM-IF1") -> str:
    """Predict beneficial mutations using structure-based zero-shot models. Use for mutation prediction with PDB structure files."""
    try:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not os.path.exists(structure_file):
            return f"Error: Structure file not found at path: {structure_file}"
        return call_zero_shot_structure_prediction_from_file(structure_file, model_name, api_key)
    except Exception as e:
        return f"Zero-shot structure prediction error: {str(e)}"

@tool("protein_function_prediction", args_schema=FunctionPredictionInput)
def protein_function_prediction_tool(sequence: Optional[str] = None, fasta_file: Optional[str] = None, model_name: str = "ESM2-650M", task: str = "Solubility") -> str:
    """Predict protein functions like solubility, localization, metal ion binding, stability, sorting signal, and optimum temperature."""
    try:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if fasta_file and os.path.exists(fasta_file):
            return call_protein_function_prediction_from_file(fasta_file, model_name, task, api_key)
        elif sequence:
            return call_protein_function_prediction(sequence, model_name, task, api_key)
        else:
            return "Error: Either sequence or fasta_file must be provided"
    except Exception as e:
        return f"Function protein prediction error: {str(e)}"

@tool("functional_residue_prediction", args_schema=ResidueFunctionPredictionInput)
def functional_residue_prediction_tool(sequence: Optional[str] = None, fasta_file: Optional[str] = None, model_name: str = "ESM2-650M", task: str = "Activate") -> str:
    try:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if fasta_file and os.path.exists(fasta_file):
            return call_functional_residue_prediction_from_file(fasta_file, model_name, task, api_key)
        elif sequence:
            return call_functional_residue_prediction(sequence, model_name, task, api_key)
        else:
            return "Error: Either sequence or fasta_file must be procided"
    except Exception as e:
        return f"Functional residue prediction error: {str(e)}"

@tool("interpro_query", args_schema=InterProQueryInput)
def interpro_query_tool(uniprot_id: str) -> str:
    """Query InterPro database for protein function annotations and GO terms using UniProt ID."""
    try:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        return call_interpro_function_query(uniprot_id, api_key)
    except Exception as e:
        return f"InterPro query error: {str(e)}"

@tool("generate_training_config", args_schema=CSVTrainingConfigInput)
def generate_training_config_tool(csv_file: str, test_csv_file: Optional[str] = None, output_name: str = "custom_training_config") -> str:
    """Generate training JSON configuration from CSV files containing protein sequences and labels."""
    try:
        return process_csv_and_generate_config(csv_file, test_csv_file, output_name)
    except Exception as e:
        return f"Training config generation error: {str(e)}"

@tool("protein_properties_generation", args_schema=ProteinPropertiesInput)
def protein_properties_generation_tool(sequence: Optional[str] = None, fasta_file: Optional[str] = None, task_name = "Physical and chemical properties" ) -> str:
    """Predict the protein phyical, chemical, and structure properties."""
    try:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        
        if fasta_file:
            if not os.path.exists(fasta_file):
                return f"Error: FASTA file not found at path: {fasta_file}"
            return call_protein_properties_prediction_from_file(fasta_file, task_name, api_key)
        elif sequence:
            return call_protein_properties_prediction(sequence, task_name, api_key)
        else:
            return f"Error: Structure file not found at path: {fasta_file}"
        
    except Exception as e:
        return f"Zero-shot structure prediction error: {str(e)}"

# Conversation Manager
class ConversationManager:
    def __init__(self):
        self.conversations = {}
        self.active_conversation = None
    
    def create_conversation(self, title: str = None) -> str:
        """Create a new conversation and return its ID"""
        conv_id = str(uuid.uuid4())
        if not title:
            title = f"Chat {datetime.now().strftime('%H:%M')}"
        
        # Create agent and memory for this conversation
        llm = DeepSeekLLM()
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=10  # Keep last 10 exchanges
        )
        
        # Define tools
        tools = [
            zero_shot_sequence_prediction_tool,
            zero_shot_structure_prediction_tool,
            protein_function_prediction_tool,
            functional_residue_prediction_tool,
            interpro_query_tool,
            protein_properties_generation_tool,
            generate_training_config_tool
        ]
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are VenusFactory AI Assistant, a specialized protein engineering and bioinformatics expert. You can help users with:

            1. Zero-shot mutation prediction using sequence-based models (ESM-1v, ESM2-650M, ESM-1b) and structure-based models (SaProt, ProtSSN, ESM-IF1, MIF-ST, ProSST-2048)
            2. Protein function prediction including solubility, localization, metal ion binding, stability, sorting signal, and optimum temperature
            3. Functional residue prediction including activate, binding, evolutionary, and motif
            4. InterPro database queries for protein function annotations
            5. Protein porperties generation including Physical and chemical properties, Relative solvent accessible surface area (PDB only), SASA value (PDB only), Secondary structure (PDB only)
            6. Training configuration generation from CSV datasets

            When users ask for protein analysis, use the appropriate tools. Always respond in English and provide clear, actionable insights.

            For training config generation:
            - Accepts CSV files with 'aa_seq' and 'label' columns
            - Generates JSON configs compatible with ESM2-8M model
            - Validates data format and provides dataset statistics
            - Creates ready-to-use training configurations

            Available tools:
            - zero_shot_sequence_prediction: For mutation prediction with protein sequences or FASTA files
            - zero_shot_structure_prediction: For mutation prediction with PDB structure files  
            - protein_function_prediction: For predicting protein functions
            - functional_residue_prediction: For predicting residue functions
            - interpro_query: For querying protein annotations from InterPro database
            - protein_properties_generation_tool: For generat protein properties features
            - generate_training_config: For creating training configs from CSV files

            Guidelines for tool usage:
            - Use sequence-based models (ESM-1v, ESM2-650M, ESM-1b) with protein sequences
            - Use structure-based models (SaProt, ProtSSN, ESM-IF1, MIF-ST, ProSST-2048) with PDB files
            - For function prediction, available tasks are: Solubility, Localization, Metal ion binding, Stability, Sorting signal, Optimum temperature
            - For functional residue prediction, available tasks are Activate, Binding, Evolutionary, and Motif
            - For InterPro queries, extract UniProt IDs from user messages (format: P12345, Q9Y6K9, etc.)
            - For protein properties, available tasks are: Physical and chemical properties, Relative solvent accessible surface area (PDB only), SASA value (PDB only), Secondary structure (PDB only)
            - For generate training config, generate the training config for the uploaded datasets."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create agent executor
        agent_executor = create_agent_executor(llm, tools, prompt, memory)
        
        self.conversations[conv_id] = {
            'id': conv_id,
            'title': title,
            'agent': agent_executor,
            'memory': memory,
            'history': [],
            'created_at': datetime.now(),
            'files': []  # Store uploaded files for this conversation
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
    
    def delete_conversation(self, conv_id: str):
        if conv_id in self.conversations:
            del self.conversations[conv_id]
    
    def list_conversations(self) -> List[dict]:
        return [
            {
                'id': conv['id'],
                'title': conv['title'], 
                'created_at': conv['created_at']
            }
            for conv in sorted(self.conversations.values(), key=lambda x: x['created_at'], reverse=True)
        ]

def create_agent_executor(llm, tools, prompt, memory):
    """Create a simple agent executor"""
    class SimpleAgent:
        def __init__(self, llm, tools, prompt, memory):
            self.llm = llm
            self.tools = {tool.name: tool for tool in tools}
            self.prompt = prompt
            self.memory = memory
        
        def invoke(self, input_data: dict) -> dict:
            user_input = input_data["input"]
            print(user_input)
            # Get chat history
            history = self.memory.chat_memory.messages
            
            # Check if we need to use tools
            tool_call = self.should_use_tool(user_input, input_data.get("files", []))
            
            if tool_call:
                tool_name, tool_args = tool_call
                if tool_name in self.tools:
                    try:
                        result = self.tools[tool_name].invoke(tool_args)
                        response = f"Tool Result ({tool_name}):\n{result}"
                    except Exception as e:
                        response = f"Tool execution error: {str(e)}"
                else:
                    response = f"Tool {tool_name} not found"
            else:
                # Use LLM for general conversation
                messages = []
                for msg in history[-10:]:  # Last 10 messages
                    if isinstance(msg, HumanMessage):
                        messages.append({"role": "user", "content": msg.content})
                    elif isinstance(msg, AIMessage):
                        messages.append({"role": "assistant", "content": msg.content})
                
                messages.append({"role": "user", "content": user_input})
                response = self.llm.invoke(messages)
            
            # Update memory
            self.memory.chat_memory.add_user_message(user_input)
            self.memory.chat_memory.add_ai_message(response)
            
            return {"output": response}
        
        def should_use_tool(self, user_input: str, files: List[str] = None) -> Optional[Tuple[str, dict]]:
            """Determine if we should use a tool and which one"""
            user_lower = user_input.lower()
            # Check for uploaded files
            has_fasta = files and any(f.lower().endswith(('.fasta', '.fa')) for f in files if isinstance(f, str))
            has_structure = files and any(f.lower().endswith(('.pdb')) for f in files if isinstance(f, str))
            # Extract sequence from message
            sequence = extract_sequence_from_message(user_input)
            uniprot_id = extract_uniprot_id_from_message(user_input)
            
            # Check for mutation prediction keywords
            mutation_keywords = ['mutation', 'mutant', 'mutate', 'zero-shot', 'zero shot', 'design']
            is_mutation = any(word in user_lower for word in mutation_keywords)
            
            # Check for function prediction keywords
            function_tasks = {
                'solubility': 'Solubility',
                'localization': 'Localization', 
                'metal ion binding': 'Metal ion binding',
                'binding': 'Metal ion binding',
                'stability': 'Stability',
                'sorting signal': 'Sorting signal',
                'temperature': 'Optimum temperature'
            }
            
            detected_function = None
            for keyword, task in function_tasks.items():
                if keyword in user_lower:
                    detected_function = task
                    break
            
            functional_residue_tasks =  {
                "activate": "Activate",
                "binding residue": "Binding",
                "evolutionary": "Evolutionary",
                "motif": "Motif"
            }

            detected_functional_residue = None
            for keyword, task  in functional_residue_tasks.items():
                if keyword in user_lower:
                    detected_function = task
                    break
            
            protein_properties_tasks = {
                "physical properties": "Physical and chemical properties",
                "chemical properties": "Physical and chemical properties",
                "RSA": "Relative solvent accessible surface area (PDB only)",
                "relative solvent accessible surface area": "Relative solvent accessible surface area (PDB only)",
                "SASA": "SASA value (PDB only)",
                "secondary structure": "Secondary structure (PDB only)"
            }

            detected_properties = None
            for keyword, task in protein_properties_tasks.items():
                if keyword in user_lower:
                    detected_properties = task
                    break

            # Check for InterPro query
            interpro_keywords = ['function', 'interpro', 'annotation', 'go term']
            is_interpro = any(word in user_lower for word in interpro_keywords)
            
            # Decision logic
            if is_interpro and uniprot_id:
                return ("interpro_query", {"uniprot_id": uniprot_id})
            if is_mutation:
                if has_structure:
                    structure_file = next((f for f in files if f.lower().endswith(('.pdb'))), None)
                    return ("zero_shot_structure_prediction", {"structure_file": structure_file})
                elif has_fasta:
                    fasta_file = next((f for f in files if f.lower().endswith(('.fasta', '.fa'))), None)
                    return ("zero_shot_sequence_prediction", {"fasta_file": fasta_file})
                elif sequence:
                    return ("zero_shot_sequence_prediction", {"sequence": sequence})
            
            if detected_function:
                if has_fasta:
                    fasta_file = next((f for f in files if f.lower().endswith(('.fasta', '.fa'))), None)
                    return ("protein_function_prediction", {"fasta_file": fasta_file, "task": detected_function})
                elif sequence:
                    return ("protein_function_prediction", {"sequence": sequence, "task": detected_function})
            
            if detected_functional_residue:
                if has_fasta:
                    fasta_file = next((f for f in files if f.lower().endswith(('.fasta', '.fa'))), None)
                    return ("functional_residue_prediction", {"fasta_file": fasta_file, "task": detected_functional_residue})
                elif sequence:
                    return ("functional_residue_prediction", {"fasta_file": fasta_file, "task": detected_functional_residue})

            if detected_properties:
                if has_structure:
                    structure_file = next((f for f in files if f.lower().endswith(('.pdb'))), None)
                    return ("protein_properties_generation", {"fasta_file": structure_file})
                elif has_fasta:
                    fasta_file = next((f for f in files if f.lower().endswith(('.fasta', '.fa'))), None)
                    return ("protein_properties_generation", {"fasta_file": fasta_file})
                elif sequence:
                    return ("protein_properties_generation", {"sequence": sequence})

            return None
    
    return SimpleAgent(llm, tools, prompt, memory)

# Global conversation manager
conv_manager = ConversationManager()

def create_chat_tab(constant: Dict[str, Any]) -> Dict[str, Any]:
    """Creates the enhanced Chat tab with multi-conversation support"""
    
    # Create initial conversation
    initial_conv_id = conv_manager.create_conversation("New Chat")
    conv_manager.active_conversation = initial_conv_id
    
    def create_new_conversation():
        """Create a new conversation"""
        conv_id = conv_manager.create_conversation()
        conv_manager.active_conversation = conv_id
        
        # Update conversation list
        conversations = conv_manager.list_conversations()
        choices = [(f"{conv['title']} ({conv['created_at'].strftime('%H:%M')})", conv['id']) for conv in conversations]
        
        return (
            gr.Dropdown(choices=choices, value=conv_id),  # Updated dropdown
            [],  # Clear chatbot
            gr.MultimodalTextbox(value=None)  # Clear input
        )
    
    def switch_conversation(selected_conv_id):
        """Switch to a different conversation"""
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
        
        return [], gr.MultimodalTextbox(value=None)
    
    def delete_current_conversation(selected_conv_id):
        """Delete the current conversation"""
        if selected_conv_id and len(conv_manager.conversations) > 1:
            conv_manager.delete_conversation(selected_conv_id)
            
            # Get remaining conversations
            conversations = conv_manager.list_conversations()
            if conversations:
                new_conv_id = conversations[0]['id']
                conv_manager.active_conversation = new_conv_id
                choices = [(f"{conv['title']} ({conv['created_at'].strftime('%H:%M')})", conv['id']) for conv in conversations]
                
                return (
                    gr.Dropdown(choices=choices, value=new_conv_id),
                    [],  # Clear chatbot
                    gr.MultimodalTextbox(value=None)
                )
        
        return gr.Dropdown(), [], gr.MultimodalTextbox(value=None)
    
    def send_message(history, message, selected_conv_id, system_prompt):
        UPLOAD_DIR = "temp_outputs/upload_data"
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        if not message or not message.get("text"):
            yield history, gr.MultimodalTextbox(value=None)
            return

        text = message["text"]
        files = message.get("files", [])
        file_paths = []
        if files:
            for file_obj in files:
                try:
                    original_temp_path = file_obj
                    
                    if os.path.exists(original_temp_path):
                        original_filename = os.path.basename(original_temp_path)
                        unique_filename = f"{uuid.uuid4().hex}_{original_filename}"
                        destination_path = os.path.join(UPLOAD_DIR, unique_filename)
                        shutil.copy2(original_temp_path, destination_path)
                        normalized_path = destination_path.replace('\\', '/')
                        
                        file_paths.append(normalized_path)
                        print(f"File copied and normalized to: {normalized_path}")
                    else:
                        print(f"Warning: Gradio temp file not found at {original_temp_path}")
                except Exception as e:
                    print(f"Error processing file: {e}")

        conv_id = selected_conv_id or conv_manager.active_conversation
        conv = conv_manager.get_conversation(conv_id)
        if not conv:
            history.append({"role": "user", "content": text})
            history.append({"role": "assistant", "content": "Error: Could not find the active conversation. Please start a new chat."})
            yield history, gr.MultimodalTextbox(value=None)
            return

        display_text = text
        if file_paths:
            file_names = ", ".join([os.path.basename(f) for f in file_paths])
            display_text += f"📎 *Attached files: {file_names}*"
        history.append({"role": "user", "content": display_text})

        history.append({"role": "assistant", "content": "🤔 Analyzing your request..."})
        yield history, gr.MultimodalTextbox(value=None, interactive=False)

        agent_input = text
        if file_paths:
            agent_input += f"[CONTEXT: User has uploaded the following files. Use their full paths when a tool requires a file path: {', '.join(file_paths)}]"

        try:
            result = conv['agent'].invoke({
                "input": agent_input,
                "files": file_paths
            })
            response = result.get("output", "Sorry, I encountered an issue and could not generate a response.")
        except Exception as e:
            response = f"❌ An error occurred while processing your request: {str(e)}"


        history[-1] = {"role": "assistant", "content": response}
        yield history, gr.MultimodalTextbox(value=None, interactive=True)


    
    # Create UI
    with gr.Column(elem_classes="chat-container"):
        # Conversation management row
        with gr.Row():
            with gr.Column(scale=4):
                conversations = conv_manager.list_conversations()
                choices = [(f"{conv['title']} ({conv['created_at'].strftime('%H:%M')})", conv['id']) for conv in conversations]
                
                conversation_dropdown = gr.Dropdown(
                    choices=choices,
                    value=initial_conv_id,
                    label="Conversations",
                    interactive=True
                )
            
            with gr.Column(scale=1):
                new_chat_btn = gr.Button("➕ New Chat", variant="secondary", size="sm")
                delete_chat_btn = gr.Button("🗑️ Delete", variant="secondary", size="sm")
        
        # Settings accordion
        with gr.Accordion("⚙️ Settings", open=False):
            system_prompt_input = gr.Textbox(
                label="System Prompt",
                placeholder="Set AI assistant role and behavior...",
                lines=3,
                value="You are VenusFactory AI Assistant, a specialized protein engineering and bioinformatics expert."
            )
        
        # Main chat interface
        chatbot = gr.Chatbot(
            label="VenusFactory AI Assistant",
            type="messages",
            height=600,
            show_label=True,
            avatar_images=(
                None,
                "https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/venus_logo.png",
            ),
            bubble_full_width=False,
            show_copy_button=True
        )
        
        # Input area
        chat_input = gr.MultimodalTextbox(
            interactive=True,
            file_count="multiple",
            placeholder="💬 Enter your message or upload files (FASTA, PDB)...",
            show_label=False,
            file_types=[".fasta", ".fa", ".pdb", ".csv"]
        )
        
        # VenusFactory info
        with gr.Accordion("🔬 VenusFactory Tools", open=False):
            gr.Markdown("""
            ### Available Tools (Powered by Langchain)
            
            **🧬 Zero-shot Mutation Prediction:**
            - Sequence-based: ESM-1v, ESM2-650M, ESM-1b
            - Structure-based: SaProt, ProtSSN, ESM-IF1, MIF-ST, ProSST-2048
            
            **⚗️ Protein Function Prediction:**
            - Solubility, Localization, Metal ion binding, Stability, Sorting signal, Optimum temperature
            
            **📊 InterPro Database Query:**
            - Protein function annotations and GO terms
            
            The AI assistant will automatically select the appropriate tool based on your request!
            """)
    
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
        inputs=[chatbot, chat_input, conversation_dropdown, system_prompt_input],
        outputs=[chatbot, chat_input]
    )
    
    return {
        "chatbot": chatbot,
        "chat_input": chat_input,
        "conversation_dropdown": conversation_dropdown,
    }

def call_zero_shot_sequence_prediction(sequence: str, model_name: str = "ESM2-650M", api_key: str = None) -> str:
    """Call VenusFactory zero-shot sequence-based mutation prediction API"""
    try:
        temp_fasta = tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False)
        temp_fasta.write(f">temp_sequence\n{sequence}\n")
        temp_fasta.close()
        
        client = Client("http://localhost:7860/")
        result = client.predict(
            function_selection="Activity",
            file_obj=handle_file(temp_fasta.name),
            enable_ai=False,
            ai_model="DeepSeek",
            user_api_key=api_key,
            model_name=model_name,
            api_name="/handle_mutation_prediction_base"
        )
        os.unlink(temp_fasta.name)
        
        return parse_zero_shot_prediction_result(result, api_key)
    except Exception as e:
        return f"Zero-shot sequence prediction error: {str(e)}"

def call_zero_shot_sequence_prediction_from_file(fasta_file: str, model_name: str = "ESM2-650M", api_key: str = None) -> str:
    """Call VenusFactory zero-shot sequence prediction API using uploaded FASTA file"""
    try:
        client = Client("http://localhost:7860/")
        result = client.predict(
            function_selection="Activity",
            file_obj=handle_file(fasta_file),
            enable_ai=False,
            ai_model="DeepSeek",
            user_api_key=api_key,
            model_name=model_name,
            api_name="/handle_mutation_prediction_base"
        )
        return parse_zero_shot_prediction_result(result, api_key)
    except Exception as e:
        return f"Zero-shot sequence prediction error: {str(e)}"

def call_zero_shot_structure_prediction_from_file(structure_file: str, model_name: str = "ESM-IF1", api_key: str = None) -> str:
    """Call VenusFactory zero-shot structure-based mutation prediction API"""
    try:
        client = Client("http://localhost:7860/")
        result = client.predict(
            function_selection="Activity",
            file_obj=handle_file(structure_file),
            enable_ai=False,
            ai_model="DeepSeek",
            user_api_key=api_key,
            model_name=model_name,
            api_name="/handle_mutation_prediction_base"
        )
        return parse_zero_shot_prediction_result(result, api_key)
    except Exception as e:
        return f"Zero-shot structure prediction error: {str(e)}"

def generate_simple_ai_summary_mutatuon(df, api_key: str = None) -> str:
    """Generate simple AI summary for mutation prediction results"""
    try:
        if not api_key:
            api_key = os.getenv("DEEPSEEK_API_KEY")
        
        if not api_key:
            return "❌ No API key found. Please set DEEPSEEK_API_KEY environment variable or provide API key in the interface."

        score_col = next(
        (col for col in df.columns if 'Prediction Rank' in col.lower()), 
        df.columns[1] if len(df.columns) > 1 else None)
    
        if not score_col:
            return "Error: Score column not found."
        num_rows = len(df)
        top_count = max(20, int(num_rows * 0.05)) if num_rows >= 5 else num_rows
        top_mutations_str = df.head(top_count)[['Mutant', score_col]].to_string(index=False)
        prompt =  f"""
            Please act as an expert protein engineer and analyze the following mutation prediction results generated.
            A deep mutational scan was performed. The results are sorted from most beneficial to least beneficial based on the '{score_col}'. Below are the top 5% of mutations.
            ### Top 5% Predicted Mutations (Potentially Most Beneficial):
            ```
            {top_mutations_str}
            ```
            ### Your Analysis Task:
            Based on this data, provide a structured scientific analysis report that includes the following sections:
            1. Executive Summary: Briefly summarize the key findings. Are there clear hotspot regions?
            2. Analysis of Beneficial Mutations: Discuss the top mutations and their potential biochemical impact.
            3. Analysis of Detrimental Mutations: What do the most harmful mutations suggest about critical residues?
            4. Recommendations for Experimentation: Suggest 3-5 specific mutations for validation, with justifications.
            5. Do not output formatted content, just one paragraph is sufficient
            Provide a concise, clear, and insightful report in a professional scientific tone, summarize the above content into 1-2 paragraphs and output unformatted content.
            """

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a protein engineering expert. Provide concise analysis of mutation prediction results."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 1000
        }
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"❌ AI API call failed: {response.status_code}"
    except Exception as e:
        return f"❌ Error generating AI summary: {str(e)}"


def parse_zero_shot_prediction_result(result: tuple, api_key: str = None) -> str:
    """Parse zero-shot prediction result with AI summary"""
    try:
        data = result[2]
        if isinstance(data, dict) and 'headers' in data and 'data' in data:
            df = pd.DataFrame(data['data'], columns=data['headers'])
        elif isinstance(data, pd.DataFrame):
            df = data
        ai_summary = generate_simple_ai_summary_mutatuon(df, api_key)
        return ai_summary
        
    except Exception as e:
        return f"Error parsing prediction result: {str(e)}"

def call_protein_function_prediction_from_file(fasta_file: str, model_name: str = "ProtT5-xl-uniref50", task: str = "Solubility", api_key: str = None) -> str:
    """Call VenusFactory protein function prediction API using uploaded FASTA file"""
    try:
        dataset_mapping = {
            "Solubility": ["DeepSol", "DeepSoluE", "ProtSolM"],
            "Localization": ["DeepLocBinary", "DeepLocMulti"],
            "Metal ion binding": ["MetalIonBinding"],
            "Stability": ["Thermostability"],
            "Sorting signal": ["SortingSignal"],
            "Optimum temperature": ["DeepET_Topt"]
        }
        datasets = dataset_mapping.get(task, ["DeepSol"])
        
        client = Client("http://localhost:7860/")
        result = client.predict(
            task=task, 
            fasta_file=handle_file(fasta_file),
            model_name=model_name,
            datasets=datasets,
            enable_ai=True,
            ai_model="DeepSeek",
            user_api_key=api_key,
            api_name="/handle_protein_function_prediction_chat"
        )
        return parse_function_prediction_result(result, task, api_key)
    except Exception as e:
        return f"Function prediction error: {str(e)}"

def call_protein_function_prediction(sequence: str, model_name: str = "ProtT5-xl-uniref50", task: str = "Solubility", api_key: str = None) -> str:
    """Call VenusFactory protein function prediction API with sequence"""
    try:
        temp_fasta = tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False)
        temp_fasta.write(f">temp_sequence\n{sequence}\n")
        temp_fasta.close()

        dataset_mapping = {
            "Solubility": ["DeepSol", "DeepSoluE", "ProtSolM"],
            "Localization": ["DeepLocBinary", "DeepLocMulti"],
            "Metal ion binding": ["MetalIonBinding"],
            "Stability": ["Thermostability"],
            "Sorting signal": ["SortingSignal"],
            "Optimum temperature": ["DeepET_Topt"]
        }
        datasets = dataset_mapping.get(task, ["DeepSol"])
        
        client = Client("http://localhost:7860/")
        result = client.predict(
            task=task, 
            fasta_file=handle_file(temp_fasta.name),
            model_name=model_name,
            datasets=datasets,
            enable_ai=True,
            ai_model="DeepSeek",
            user_api_key=api_key,
            api_name="/handle_protein_function_prediction_chat"
        )
        os.unlink(temp_fasta.name)
        
        return parse_function_prediction_result(result, task, api_key)
    except Exception as e:
        return f"Function prediction error: {str(e)}"

def parse_function_prediction_result(result: tuple, task: str, api_key: str = None) -> str:
    """Parse function prediction result with AI summary"""
    try:
        data = result[1]
        
        if isinstance(data, dict) and 'headers' in data and 'data' in data:
            df = pd.DataFrame(data['data'], columns=data['headers'])
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            return f"Function prediction result: {str(data)}"
        
        if df is None or df.empty:
            return "Error: No function prediction data available"
        
        ai_summary = generate_simple_ai_summary_function(df, task, api_key)
        return ai_summary
        
    except Exception as e:
        return f"Error parsing function prediction result: {str(e)}"

def generate_simple_ai_summary_function(df, task: str, api_key: str = None) -> str:
    """Generate AI summary for function prediction results"""
    try:
        if not api_key:
            api_key = os.getenv("DEEPSEEK_API_KEY")
        
        if not api_key:
            return "❌ No API key found. Please set DEEPSEEK_API_KEY environment variable or provide API key in the interface."

        prompt = f"""
            You are a senior protein biochemist with extensive laboratory experience. A colleague has just shown you protein function prediction results for the task '{task}'. 
            Please analyze these results from a practical biologist's perspective:
            {df.to_string(index=False)}
            Provide a concise, practical analysis focusing ONLY on:
            0. The task '{task}' with "Stability" and "Optimum temperature" are regression task
            1. What the prediction results mean for each protein
            2. The biological significance of the confidence scores
            3. Practical experimental recommendations based on these predictions
            4. Any notable patterns or outliers in the results
            5. Do not output formatted content, just one paragraph is sufficient
            Use simple, clear language that a bench scientist would appreciate. Do NOT mention:
            - Training datasets or models
            - Technical implementation details  
            - Computational methods
            - Statistical concepts beyond confidence scores
            Keep your response under 200 words and speak as if you're having a conversation with a colleague in the lab.
            """
        # Call AI API
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a protein engineering expert. Provide concise analysis of mutation prediction results."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 1000
        }
        
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"❌ AI API call failed: {response.status_code}"
                
    except Exception as e:
        return f"❌ Error generating AI summary: {str(e)}"

def call_functional_residue_prediction_from_file(fasta_file: str, model_name: str = "ESM2-650M", task: str = "Activity", api_key: str = None) -> str:
    """Call VenusFactory functional residue prediction API using uploaded FASTA file"""
    try:

        client = Client("http://localhost:7860/")
        result = client.predict(
            task=task, 
            fasta_file=handle_file(fasta_file),
            enable_ai=True,
            ai_model="DeepSeek",
            user_api_key=api_key,
            model_name=model_name,
            api_name="/handle_protein_residue_function_prediction_chat"
        )
        return parse_functional_residue_prediction_result(result, task, api_key)
    except Exception as e:
        return f"Functional residue prediction error: {str(e)}"

def call_functional_residue_prediction(sequence: str, model_name: str = "ESM2-650M", task: str = "Activity", api_key: str = None) -> str:
    """Call VenusFactory functional residue prediction API using uploaded FASTA file"""
    try:
        temp_fasta = tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False)
        temp_fasta.write(f">temp_sequence\n{sequence}\n")
        temp_fasta.close()

        client = Client("http://localhost:7860/")
        result = client.predict(
            task=task, 
            fasta_file=handle_file(temp_fasta.name),
            enable_ai=True,
            ai_model="DeepSeek",
            user_api_key=api_key,
            model_name=model_name,
            api_name="/handle_protein_residue_function_prediction_chat"
        )
        os.unlink(temp_fasta.name)
        return parse_functional_residue_prediction_result(result, task, api_key)
    except Exception as e:
        return f"Functional residue prediction error: {str(e)}"

def parse_functional_residue_prediction_result(result: tuple, task: str, api_key: str=None) -> str:
    """Parse function prediction result with AI summary"""
    try:
        data = result[1]
        ai_summary = generate_simple_ai_summary_functional_residue(data, task, api_key)
        return ai_summary
    
    except Exception as e:
        return f"Error parsing functional resudue prediction result: {str(e)}"

def generate_simple_ai_summary_functional_residue(df, task: str, api_key: str = None) -> str:
    """Generate AI summary for functional residue prediction results"""
    try:
        if not api_key:
            api_key = os.getenv("DEEPSEEK_API_KEY")
        
        if not api_key:
            return "❌ No API key found. Please set DEEPSEEK_API_KEY environment variable or provide API key in the interface."

        prompt = f"""
        You are a senior protein biochemist and structural biologist with deep expertise in enzyme mechanisms and protein engineering. 
        A colleague has just brought you the following prediction results identifying key functional residues, specifically for the task '{task}'.
        Please analyze these residue-specific results from a practical, lab-focused perspective:
        {df}
        Provide a concise, practical analysis in a single paragraph, focusing ONLY on:
        1. The predicted functional role of the individual high-scoring residues (e.g., activity, binding, conserved, and motif).
        2. The significance of the confidence scores (i.e., how certain are we that a specific residue is critical for the function).
        3. Specific experimental validation steps, especially which residues are the top candidates for site-directed mutagenesis to confirm their roles.
        4. Any notable patterns or outliers (e.g., a cluster of high-scoring residues forming a potential active site, or a surprising critical residue far from the expected functional region).
        5. Do not output formatted content, just one paragraph is sufficient
        Use simple, direct language that a bench scientist would use. Do NOT mention:
        - The model used or its training data.
        - Any computational or implementation details.
        - Complex statistical concepts beyond confidence.

        Keep your response under 200 words and speak as if you are discussing next steps for an experiment with a colleague.
        """
        # Call AI API
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a protein engineering expert. Provide concise analysis of mutation prediction results."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 1000
        }
        
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"❌ AI API call failed: {response.status_code}"
                
    except Exception as e:
        return f"❌ Error generating AI summary: {str(e)}"

def call_interpro_function_query(uniprot_id: str, api_key: str = None) -> str:
    """Query InterPro database for protein function"""
    try:
        result = download_single_interpro(uniprot_id)
        if result and len(result) > 1:
            ai_summary = generate_interpro_ai_summary(result[1], api_key)
            return ai_summary
        else:
            return f"No InterPro data found for {uniprot_id}"
    except Exception as e:
        return f"InterPro query error: {str(e)}"

def call_protein_properties_prediction_from_file(fasta_file: str, task_name: str, api_key: str = None) -> str:
    try:
        client = Client("http://localhost:7860/")
        result = client.predict(
            task=task_name,
            file_obj=handle_file(fasta_file),
            api_name="/handle_protein_properties_generation"
        )
        return parse_protein_properties_prediction_result(result, task_name, api_key)
    except Exception as e:
        return f"Protein properties prediction error: {str(e)}"

def call_protein_properties_prediction(sequence: str, task_name, api_key: str = None) -> str:
    try:
        temp_fasta = tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False)
        temp_fasta.write(f">temp_sequence\n{sequence}\n")
        temp_fasta.close()

        client = Client("http://localhost:7860/")
        result = client.predict(
            task=task_name,
            file_obj=handle_file(temp_fasta.name),
            api_name="/handle_protein_properties_generation"
        )
        os.unlink(temp_fasta.name)
        return parse_protein_properties_prediction_result(result, task_name, api_key)
    except Exception as e:
        return f"Protein properties prediction error: {str(e)}"

def parse_protein_properties_prediction_result(result: tuple, task_name: str, api_key: str) -> str:
    try:
        data = result[1]
        ai_summary = generate_simple_ai_summary_protein_properties(data, task_name, api_key)
        return ai_summary
    
    except Exception as e:
        return f"Error parsing prediction result: {str(e)}"

def generate_simple_ai_summary_protein_properties(data, task_name, api_key: str = None) -> str:
    """Generate simple AI summary for protein properties results"""
    try:
        if not api_key:
            api_key = os.getenv("DEEPSEEK_API_KEY")
        
        if not api_key:
            return "❌ No API key found. Please set DEEPSEEK_API_KEY environment variable or provide API key in the interface."

        prompt =  f""" 
        You are a senior protein biochemist with extensive laboratory experience. A colleague has just shown you the calculated properties for a protein they plan to work with.
        The analysis performed was: '{task_name}'
        Here are the calculated results:
        {data}
        Please provide a concise, practical analysis of these features from a bench scientist's perspective. Focus ONLY on:
        1.  The practical implications of the most significant values (e.g., what does the pI mean for purification? What does the instability index suggest for expression?).
        2.  Any noteworthy or extreme values that might affect experimental work.
        3.  A brief recommendation for lab work based on these properties (e.g., buffer choice, solubility concerns).
        Keep your response to a single, unformatted paragraph under 200 words. Speak as if you're talking to a colleague in the lab.
        Do NOT mention:
        - The specific software or methods used for the calculation.
        - Any complex statistical concepts.
        - Technical implementation details.
        """
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a protein engineering expert. Provide concise analysis of mutation prediction results."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 1000
        }
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"❌ AI API call failed: {response.status_code}"
    except Exception as e:
        return f"❌ Error generating AI summary: {str(e)}"


def process_csv_and_generate_config(csv_file: str, test_csv_file: Optional[str] = None, output_name: str = "custom_training_config") -> str:
    try:
        df = pd.read_csv(csv_file)
        required_columns = ['aa_seq', 'label']
        
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            return f"Missing required columns: {missing}. Please ensure your CSV has 'aa_seq' and 'label' columns, then upload again."
        analysis = analyze_dataset_for_ai(df, test_csv_file)
        ai_config = generate_ai_training_config(analysis)
        config = create_comprehensive_config(csv_file, test_csv_file, ai_config, analysis)
        temp_dir = Path("temp_outputs")
        sequence_dir = temp_dir / "training_configs"
        os.makedirs(sequence_dir, exist_ok=True)
        timestamp = int(time.time())
        config_path = os.path.join(sequence_dir, f"{output_name}_{timestamp}.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        message = f"Training configuration generated successfully! Config file: {config_path} The configuration is ready for use in the training interface."
        return message

    except Exception as e:
        return f"Error processing CSV: {str(e)}"

def analyze_dataset_for_ai(df: pd.DataFrame, test_csv_file: Optional[str] = None) -> dict:
    """Analyze dataset to provide context for AI parameter selection"""
    
    def classify_task_heuristic(df: pd.DataFrame) -> str:
        """Classify task type based on label characteristics using heuristic rules"""
        
        label_data = df['label']
        sample_labels = label_data.head(50).tolist()  # Sample for analysis
        
        # Step 1: Determine if it's protein-level or residue-level
        # Check if labels are sequences (residue-level) or single values (protein-level)
        is_residue_level = False
        
        # Check label length vs sequence length for first few samples
        for i in range(min(10, len(df))):
            label_str = str(df.iloc[i]['label'])
            seq_len = len(df.iloc[i]['aa_seq'])
            
            # Remove common separators and spaces to get actual label length
            clean_label = label_str.replace(',', '').replace(' ', '').replace('[', '').replace(']', '')
            
            # If label length is close to sequence length, it's likely residue-level
            if len(clean_label) >= seq_len * 0.8:  # Allow some tolerance
                is_residue_level = True
                break
            
            # Also check if label looks like a sequence (e.g., "0,1,1,0,0,0" or "0.1,0.2,0.3")
            if ',' in label_str and len(label_str.split(',')) >= seq_len * 0.8:
                is_residue_level = True
                break
        
        # Step 2: Determine if it's classification or regression
        is_regression = False
        
        # Check if labels contain continuous values (regression)
        for label in sample_labels:
            label_str = str(label)
            
            if is_residue_level:
                # For residue-level, parse the sequence of values
                if ',' in label_str:
                    values = label_str.replace('[', '').replace(']', '').split(',')
                else:
                    values = list(label_str.replace('[', '').replace(']', ''))
                
                # Check if values are continuous (floats)
                try:
                    float_values = [float(v.strip()) for v in values if v.strip()]
                    # If we have decimal numbers, it's regression
                    if any('.' in str(v) for v in values if v.strip()):
                        is_regression = True
                        break
                    # If values have wide range, might be regression
                    if len(float_values) > 0 and (max(float_values) - min(float_values) > 10):
                        is_regression = True
                        break
                except ValueError:
                    # If can't convert to float, it's classification
                    continue
            else:
                # For protein-level, check the single label value
                try:
                    float_val = float(label_str)
                    # If it's a decimal number, it's regression
                    if '.' in label_str:
                        is_regression = True
                        break
                    # If integer range is large, might be regression
                    if abs(float_val) > 10:
                        is_regression = True
                        break
                except ValueError:
                    # If can't convert to float, it's classification
                    continue
        
        # Step 3: For classification, check if it's multi-label
        is_multi_label = False
        if not is_regression and not is_residue_level:
            # Check for multi-label indicators in protein-level classification
            for label in sample_labels:
                label_str = str(label)
                if any(sep in label_str for sep in [',', ';', '|', '&', '+']):
                    is_multi_label = True
                    break
                words = label_str.split()
                if len(words) > 1 and not any(char.isdigit() for char in label_str):
                    is_multi_label = True
                    break
        
        # Step 4: Return the classification
        if is_residue_level:
            if is_regression:
                return "residue_regression"
            else:
                return "residue_single_label_classification"
        else:
            if is_regression:
                return "regression"
            elif is_multi_label:
                return "multi_label_classification"
            else:
                return "single_label_classification"

    label_data = df['label']

    task_type = classify_task_heuristic(df)
    
    analysis = {
        "total_samples": int(len(df)),
        "unique_labels": int(df['label'].nunique()),
        "label_distribution": {str(k): int(v) for k, v in df['label'].value_counts().to_dict().items()},
        "sequence_stats": {
            "mean_length": float(df['aa_seq'].str.len().mean()),
            "min_length": int(df['aa_seq'].str.len().min()),
            "max_length": int(df['aa_seq'].str.len().max()),
            "std_length": float(df['aa_seq'].str.len().std())
        },
        "data_type": task_type,  # Heuristic-determined task type
        "class_balance": "balanced" if df['label'].value_counts().std() < df['label'].value_counts().mean() * 0.5 else "imbalanced"
    }
   
    if test_csv_file and os.path.exists(test_csv_file):
        test_df = pd.read_csv(test_csv_file)
        analysis["test_samples"] = int(len(test_df))
        analysis["has_test_set"] = True
    else:
        analysis["has_test_set"] = False
   
    return analysis

def convert_to_serializable(obj):
    """Convert pandas/numpy types to JSON serializable types"""
    import numpy as np
    
    if isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj

def generate_ai_training_config(analysis: dict) -> dict:
    """Use DeepSeek AI to generate optimal training configuration"""
    try:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            return get_default_config(analysis)
        
        prompt = f"""You are to act as VenusAgent, a premier expert in deep learning and protein engineering. Your core mission is to help users effortlessly configure their training tasks.
        Please analyze the user-provided dataset and, guided by the following principles, intelligently recommend an optimal set of training parameters:
        1、Efficiency-First: Prioritize configurations with low computational requirements. For example, recommend the Freeze training method to ensure smooth execution on standard hardware.
        2、Smart Inference: Automatically determine if the task is classification or regression, protein-level or residue-level based on the data's characteristics, and configure the appropriate metrics and monitoring strategies accordingly.
        3、Conventional & Robust: All parameters, such as learning rate and batch size, must be set to robust, widely-accepted default values within the field..

        Dataset Analysis:
        - Total samples: {analysis['total_samples']}
        - Problem type: {analysis['data_type']}
        - Number of labels: {analysis['unique_labels']}
        - Class balance: {analysis['class_balance']}
        - Sequence length (mean/min/max): {analysis['sequence_stats']['mean_length']:.1f}/{analysis['sequence_stats']['min_length']}/{analysis['sequence_stats']['max_length']}
        - Has test set: {analysis['has_test_set']}

        Respond with a JSON object containing optimal parameters:
        {{
            "training_method": "freeze|full|ses-adapter|plm-lora|plm-qlora",
            "problem_type": "single_label_classification|multi_label_classification|regression|residue_single_label_classification|residue_regression",
            "learning_rate": 0.00000001-0.01,
            "num_epochs": 1-200,
            "batch_size": 8-64,
            "max_seq_len": -1-2048,
            "patience": 1-50,
            "pooling_method": "mean|attention1d|light_attention",
            "scheduler": null|"linear"|"cosine"|"step",
            "monitored_metrics": "accuracy|recall|f1|mcc|auroc|f1_max|spearman_corr|mse",
            "monitored_strategy": "max|min",
            "gradient_accumulation_steps": 1-32,
            "warmup_steps": 0-1000,
            "max_grad_norm": 0.1-10.0,
            "num_workers": 0-16,
            "reasoning": "explain your choices"
        }}

        Consider:
        - Small datasets (<1000): lower learning rate, more epochs, early stopping
        - Large datasets (>10000): higher learning rate, fewer epochs
        - Long sequences (>500): smaller batch size, gradient accumulation
        - Imbalanced classes: appropriate metrics (f1, mcc)
        - Regression tasks: use spearman_corr with min strategy"""

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are a protein machine learning expert. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 800
        }
        
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        
        return get_default_config(analysis)
        
    except Exception as e:
        print(f"AI config generation failed: {e}")
        return get_default_config(analysis)

def get_default_config(analysis: dict) -> dict:
    """Fallback default configuration"""
    is_regression = analysis['data_type'] == 'regression'
    return {
        "training_method": "freeze",
        "learning_rate": 5e-4,
        "num_epochs": 20,
        "batch_size": 16,
        "max_seq_len": min(512, int(analysis['sequence_stats']['max_length'] * 1.2)),
        "patience": 10,
        "pooling_method": "mean",
        "scheduler": None,
        "monitored_metrics": "spearman_corr" if is_regression else "accuracy",
        "monitored_strategy": "max",
        "gradient_accumulation_steps": 1,
        "warmup_steps": 0,
        "max_grad_norm": 1.0,
        "num_workers": 1
    }

def create_comprehensive_config(csv_file: str, test_csv_file: Optional[str], ai_config: dict, analysis: dict) -> dict:
    """Create complete training configuration matching 1.py requirements"""
    is_regression = analysis['data_type'] == 'regression'
    dataset_directory = os.path.dirname(csv_file)
    config = {
        # Dataset configuration
        "dataset_selection": "Custom Dataset",
        "dataset_custom": dataset_directory,
        "problem_type": str(ai_config.get("problem_type", "regression")),
        "num_labels": 1 if is_regression else int(analysis['unique_labels']),
        "metrics": ["mse", "spearman_corr"] if is_regression else ["accuracy", "mcc", "f1", "precision", "recall", "auroc"],
        
        # Model configuration
        "plm_model": "ESM2-8M",
        "training_method": str(ai_config.get("training_method", "freeze")),
        "pooling_method": str(ai_config.get("pooling_method", "mean")),
        
        # Training parameters (ensure all are native Python types)
        "learning_rate": float(ai_config.get("learning_rate", 5e-4)),
        "num_epochs": int(ai_config.get("num_epochs", 20)),
        "max_seq_len": int(ai_config.get("max_seq_len", 512)),
        "patience": int(ai_config.get("patience", 10)),
        
        # Batch processing
        "batch_mode": "Batch Size Mode",
        "batch_size": int(ai_config.get("batch_size", 16)),
        
        # Advanced parameters
        "gradient_accumulation_steps": int(ai_config.get("gradient_accumulation_steps", 1)),
        "warmup_steps": int(ai_config.get("warmup_steps", 0)),
        "scheduler": ai_config.get("scheduler"),
        "max_grad_norm": float(ai_config.get("max_grad_norm", 1.0)),
        "num_workers": int(ai_config.get("num_workers", 1)),
        
        # Monitoring
        "monitored_metrics": str(ai_config.get("monitored_metrics", "accuracy")),
        "monitored_strategy": str(ai_config.get("monitored_strategy", "max")),
        
        # Output
        "output_model_name": f"model_{int(analysis['total_samples'])}samples",
        "output_dir": "./outputs",
        
        # Wandb (disabled by default)
        "wandb_enabled": False,
        
    }
    
    if test_csv_file:
        config["test_file"] = test_csv_file
    
    # Final conversion to ensure everything is serializable
    return convert_to_serializable(config)

def extract_sequence_from_message(message: str) -> Optional[str]:
    """Extract protein sequence from user message"""
    # Look for sequences (20+ consecutive amino acid letters)
    sequence_pattern = r'[ACDEFGHIKLMNPQRSTVWY]{20,}'
    matches = re.findall(sequence_pattern, message.upper())
    return matches[0] if matches else None

def extract_uniprot_id_from_message(message: str) -> Optional[str]:
    """Extract UniProt ID from user message"""
    # UniProt ID pattern: 6 or 10 alphanumeric characters starting with letter
    uniprot_pattern = r'\b[A-Z][A-Z0-9]{5}(?:[A-Z0-9]{4})?\b'
    matches = re.findall(uniprot_pattern, message.upper())
    return matches[0] if matches else None

class AIIntentRecognizer:
    """AI-based intent recognition for protein analysis tasks"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
    
    def analyze_intent(self, user_input: str, files: List[str] = None) -> dict:
        """Analyze user intent using AI"""
        try:
            # Quick heuristic checks first
            has_fasta = files and any(f.lower().endswith(('.fasta', '.fa')) for f in files if isinstance(f, str))
            has_structure = files and any(f.lower().endswith(('.pdb')) for f in files if isinstance(f, str))
            sequence = extract_sequence_from_message(user_input)
            uniprot_id = extract_uniprot_id_from_message(user_input)
            
            # If no API key, fall back to keyword matching
            if not self.api_key:
                return self._fallback_intent_analysis(user_input, files, has_fasta, has_structure, sequence, uniprot_id)
            
            context_info = self._build_context(user_input, files, has_fasta, has_structure, sequence, uniprot_id)
            
            prompt = f"""
                    You are an expert bioinformatics assistant. Analyze the user's request and determine the appropriate action.

                    Available tools:
                    1. zero_shot_sequence_prediction - For mutation prediction using protein sequences. Parameters: `sequence`, `fasta_file`.
                    2. zero_shot_structure_prediction - For mutation prediction using protein structures. Parameters: `structure_file`.
                    3. protein_function_prediction - For predicting protein properties (solubility, localization, etc.). Parameters: `sequence`, `fasta_file`, `task`.
                    4. functional_residue_prediction - For predicting functional residues (activate, binding, evolutionary, and motif ). Parameters: `sequence`, `fasta_file`, `task`.
                    5. interpro_query - For querying protein annotations from InterPro database. Parameters: `uniprot_id`.
                    6. protein_properties_generation - For generating protein properties. **Parameters: `fasta_file` (for PDB or FASTA), `sequence`, `task_name`.**
                    7. general_chat - For general questions not requiring specific tools.

                    Context: {context_info}
                    User request: "{user_input}"

                    Respond with a JSON object containing:
                    {{
                        "intent": "tool_name or general_chat",
                        "confidence": 0.0-1.0,
                        "parameters": {{}},
                        "reasoning": "brief explanation"
                    }}

                    For function prediction, available tasks are: Solubility, Localization, Metal ion binding, Stability, Sortingsignal, Optimum temperature.
                    For functional residue prediction, available tasks are: Activity Site, Binding Site, Conserved Site, Motif.
                    For protein properties, available tasks are: Physical and chemical properties, Relative solvent accessible surface area (PDB only), SASA value (PDB only), Secondary structure (PDB only).
                    """
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "You are a bioinformatics intent analyzer. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 300
            }
            
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=20
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # Extract JSON from response
                try:
                    # Find JSON in the response
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        intent_data = json.loads(json_match.group())
                        return self._validate_and_fix_intent(intent_data, context_info)
                except json.JSONDecodeError:
                    pass
            
            # Fallback to heuristic analysis
            return self._fallback_intent_analysis(user_input, files, has_fasta, has_structure, sequence, uniprot_id)
            
        except Exception as e:
            # Fallback to heuristic analysis on any error
            return self._fallback_intent_analysis(user_input, files, has_fasta, has_structure, sequence, uniprot_id)
    
    def _build_context(self, user_input: str, files: List[str], has_fasta: bool, has_structure: bool, sequence: str, uniprot_id: str) -> str:
        """Build context information for AI analysis"""
        context_parts = []
        
        if files:
            context_parts.append(f"Files uploaded: {', '.join([os.path.basename(f) for f in files])}")
        if has_fasta:
            context_parts.append("FASTA file detected")
        if has_structure:
            context_parts.append("Structure file (PDB) detected")
        if sequence:
            context_parts.append(f"Protein sequence found (length: {len(sequence)})")
        if uniprot_id:
            context_parts.append(f"UniProt ID detected: {uniprot_id}")
        
        return "; ".join(context_parts) if context_parts else "No specific context detected"
    
    def _validate_and_fix_intent(self, intent_data: dict, context_info: str) -> dict:
        """Validate and fix AI intent analysis results"""
        valid_intents = {
            'zero_shot_sequence_prediction', 'zero_shot_structure_prediction', 
            'protein_function_prediction', 'functional_residue_prediction', 'interpro_query', 
            'protein_properties_generation', 'general_chat'
        }
        
        if intent_data.get('intent') not in valid_intents:
            intent_data['intent'] = 'general_chat'
            intent_data['confidence'] = 0.1
        
        # Add missing required parameters based on context
        if 'parameters' not in intent_data:
            intent_data['parameters'] = {}
        
        if intent_data.get('intent') == 'protein_function_prediction':
            params = intent_data.get('parameters', {})
            if 'file_path' in params and 'fasta_file' not in params:
                params['fasta_file'] = params.pop('file_path')
        
        return intent_data
    
    def _fallback_intent_analysis(self, user_input: str, files: List[str], has_fasta: bool, has_structure: bool, sequence: str, uniprot_id: str) -> dict:
        """Fallback heuristic intent analysis"""
        user_lower = user_input.lower()
        has_csv = files and any(f.lower().endswith('.csv') for f in files if isinstance(f, str))
        
        # Check for InterPro query
        interpro_keywords = ['interpro', 'annotation', 'go term', 'domain']
        if any(re.search(r'\b' + re.escape(word) + r'\b', user_lower) for word in interpro_keywords) and uniprot_id:
            return {
                'intent': 'interpro_query',
                'confidence': 0.8,
                'parameters': {'uniprot_id': uniprot_id},
                'reasoning': 'UniProt ID and function keywords detected'
            }
        
        # Check for mutation prediction
        mutation_keywords = ['mutation', 'mutant', 'mutate', 'zero-shot', 'design', 'improve', 'optimize']
        if any(re.search(r'\b' + re.escape(word) + r'\b', user_lower) for word in mutation_keywords):
            if has_structure:
                structure_file = next((f for f in files if f.lower().endswith(('.pdb'))), None)
                return {
                    'intent': 'zero_shot_structure_prediction',
                    'confidence': 0.9,
                    'parameters': {'structure_file': structure_file, 'model_name': 'ESM-IF1'},
                    'reasoning': 'Mutation keywords and structure file detected'
                }
            elif has_fasta or sequence:
                fasta_file = next((f for f in files if f.lower().endswith(('.fasta', '.fa'))), None) if has_fasta else None
                params = {'model_name': 'ESM2-650M'}
                if fasta_file:
                    params['fasta_file'] = fasta_file
                elif sequence:
                    params['sequence'] = sequence
                
                return {
                    'intent': 'zero_shot_sequence_prediction',
                    'confidence': 0.8,
                    'parameters': params,
                    'reasoning': 'Mutation keywords and sequence data detected'
                }
        
        # Check for functional residue prediction
        functional_residue_tasks =  {
                "activity site": "Activity Site",
                "functional site": "Activity Site",
                "binding site": "Binding Site",
                "conservation site": "Conserved Site",
                "motif": "Motif"
            }
        
        for keyword, task in functional_residue_tasks.items():
            if re.search(r'\b' + re.escape(keyword) + r'\b', user_lower):
                params = {'task': task, 'model_name': 'ESM2-650M'}
                if has_fasta:
                    fasta_file = next((f for f in files if f.lower().endswith(('.fasta', '.fa'))), None)
                    params['fasta_file'] = fasta_file
                elif sequence:
                    params['sequence'] = sequence
                if 'fasta_file' in params or 'sequence' in params:
                    return {
                        'intent': 'functional_residue_prediction',
                        'confidence': 0.7,
                        'parameters': params,
                        'reasoning': f'Functional residue prediction keywords for {task} detected'
                    }


        # Check for function prediction
        function_tasks = {
            'solubility': 'Solubility',
            'localization': 'Localization', 
            'metal ion binding': 'Metal ion binding',
            'stability': 'Stability',
            'sorting signal': 'Sortingsignal',
            'temperature': 'Optimum temperature'
        }
        
        for keyword, task in function_tasks.items():
            if re.search(r'\b' + re.escape(keyword) + r'\b', user_lower):
                params = {'task': task, 'model_name': 'ESM2-650M'}
                if has_fasta:
                    fasta_file = next((f for f in files if f.lower().endswith(('.fasta', '.fa'))), None)
                    params['fasta_file'] = fasta_file
                elif sequence:
                    params['sequence'] = sequence
                if 'fasta_file' in params or 'sequence' in params:
                    return {
                        'intent': 'protein_function_prediction',
                        'confidence': 0.7,
                        'parameters': params,
                        'reasoning': f'Function prediction keywords for {task} detected'
                    }

        config_keywords = ['training', 'train', 'config', 'configuration', 'json', 'dataset preparation']
        if any(re.search(r'\b' + re.escape(word) + r'\b', user_lower) for word in config_keywords) and has_csv:
            csv_file = next((f for f in files if f.lower().endswith('.csv')), None)
            return {
                'intent': 'generate_training_config',
                'confidence': 0.8,
                'parameters': {'csv_file': csv_file, 'output_name': 'custom_training_config'},
                'reasoning': 'Training keywords and CSV file detected'
            }
        
        protein_properties_tasks = {
            "physical properties": "Physical and chemical properties",
            "chemical properties": "Physical and chemical properties",
            "properties": "Physical and chemical properties",
            "rsa": "Relative solvent accessible surface area (PDB only)",
            "relative solvent accessible surface area": "Relative solvent accessible surface area (PDB only)",
            "sasa": "SASA value (PDB only)",
            "secondary structure": "Secondary structure (PDB only)"
        }

        detected_properties = None
        for keyword, task in protein_properties_tasks.items():
            if re.search(r'\b' + re.escape(keyword) + r'\b', user_lower):
                detected_properties = task
                params = {'task_name': detected_properties}
                if has_structure:
                    structure_file = next((f for f in files if f.lower().endswith(('.pdb'))), None)
                    params['fasta_file'] = structure_file
                elif has_fasta:
                    fasta_file = next((f for f in files if f.lower().endswith(('.fasta', '.fa'))), None)
                    params['fasta_file'] = fasta_file
                elif sequence:
                    params['sequence'] = sequence
                
                return {
                    'intent': 'protein_properties_generation',
                    'confidence': 0.8,
                    'parameters': params,
                    'reasoning': f'Protein properties keywords for {detected_properties} detected'
                }

        return {
            'intent': 'general_chat',
            'confidence': 0.5,
            'parameters': {},
            'reasoning': 'No specific bioinformatics task detected'
        }

def create_agent_executor(llm, tools, prompt, memory):
    """Create an enhanced agent executor with AI intent recognition"""
    class EnhancedAgent:
        def __init__(self, llm, tools, prompt, memory):
            self.llm = llm
            self.tools = {tool.name: tool for tool in tools}
            self.prompt = prompt
            self.memory = memory
            self.intent_recognizer = AIIntentRecognizer()

        def _deterministic_file_check(self, user_input: str, files: List[str]) -> Optional[dict]:
            if not files:
                return None

            user_lower = user_input.lower()
            full_file_path = files[0]

            # Keywords for mutation prediction
            mutation_keywords = ['mutation', 'mutant', 'mutate', 'zero-shot', 'design', 'improve', 'optimize']
            if any(word in user_lower for word in mutation_keywords):
                if full_file_path.lower().endswith(('.pdb', '.cif')):
                    return {
                        'tool_name': 'zero_shot_structure_prediction',
                        'tool_args': {'structure_file': full_file_path}
                    }
                elif full_file_path.lower().endswith(('.fasta', '.fa')):
                    return {
                        'tool_name': 'zero_shot_sequence_prediction',
                        'tool_args': {'fasta_file': full_file_path}
                    }
            
            functional_residue_tasks =  {
                "activate": "Activate",
                "binding residue": "Binding",
                "evolutionary": "Evolutionary",
                "motif": "Motif"
            }

            for keyword, task in functional_residue_tasks.items():
                if re.search(r'\b' + re.escape(keyword) + r'\b', user_lower):
                    if full_file_path.lower().endswith(('.fasta', '.fa')):
                        return {
                            'tool_name': 'functional_residue_prediction',
                            'tool_args': {'fasta_file': full_file_path, 'task': task}
                        }

            # Keywords for function prediction
            function_tasks = {
                'solubility': 'Solubility', 'localization': 'Localization',
                'metal ion binding': 'Metal ion binding', 'binding': 'Metal ion binding',
                'stability': 'Stability', 'sorting signal': 'Sorting signal',
                'temperature': 'Optimum temperature'
            }
            for keyword, task in function_tasks.items():
                if re.search(r'\b' + re.escape(keyword) + r'\b', user_lower):
                    if full_file_path.lower().endswith(('.fasta', '.fa')):
                        return {
                            'tool_name': 'protein_function_prediction',
                            'tool_args': {'fasta_file': full_file_path, 'task': task}
                        }
            
            training_keywords = ['training', 'train', 'config', 'configuration', 'generate config', 'training config']
            if any(keyword in user_lower for keyword in training_keywords):
                if full_file_path.lower().endswith('.csv'):
                    return {
                        'tool_name': 'generate_training_config',
                        'tool_args': {'csv_file': full_file_path}
                    }
            
            return None


        def invoke(self, input_data: dict) -> dict:
            user_input = input_data["input"]
            files = input_data.get("files", [])

            response = ""
            direct_tool_call = self._deterministic_file_check(user_input, files)

            # Get chat history for context
            history = self.memory.chat_memory.messages
            
            if direct_tool_call:
                tool_name = direct_tool_call['tool_name']
                tool_args = direct_tool_call['tool_args']
                tool_name_for_display = tool_name.replace('_', ' ').title()
                
                print(f"Deterministic check successful. Calling tool '{tool_name}' with args: {tool_args}")
                
                if tool_name in self.tools:
                    try:
                        # Directly invoke the tool with the full, correct path.
                        result = self.tools[tool_name].invoke(tool_args)
                        response = f"🔬 **{tool_name_for_display}** \n{result}"
                    except Exception as e:
                        response = f"❌ Tool execution error ({tool_name}): {str(e)}"
                else:
                    response = f"❌ Tool {tool_name} not found"

            else:
                # If no file-based action is found, fall back to the general-purpose AI recognizer.
                print("No direct file-based tool found. Falling back to AI Intent Recognizer.")
                # This part of the logic handles sequence-in-text, UniProt IDs, and general chat.
                intent_analysis = self.intent_recognizer.analyze_intent(user_input, files)
                
                print(f"AI Intent Recognizer Result: {intent_analysis}")

                if intent_analysis['intent'] != 'general_chat' and intent_analysis['confidence'] > 0.5:
                    tool_name = intent_analysis['intent']
                    tool_args = intent_analysis['parameters']
                    
                    if tool_name in self.tools:
                        try:
                            result = self.tools[tool_name].invoke(tool_args)
                            response = f"🔬 **{tool_name.replace('_', ' ').title()}** \n{result}"
                        except Exception as e:
                            response = f"❌ Tool execution error ({tool_name}): {str(e)}"
                    else:
                        response = f"❌ Tool {tool_name} not found"
                else:
                    messages = self._prepare_llm_messages(self.memory.chat_memory.messages, user_input, files)
                    response = self.llm.invoke(messages)

            # Update memory with the final outcome.
            self.memory.chat_memory.add_user_message(user_input)
            self.memory.chat_memory.add_ai_message(response)
            
            return {"output": response}

        
        def _prepare_llm_messages(self, history: List[BaseMessage], user_input: str, files: List[str]) -> List[dict]:
            """Prepare messages for LLM with enhanced context"""
            messages = [
                {
                    "role": "system",
                    "content": """You are VenusFactory AI Assistant, a specialized protein engineering and bioinformatics expert.            
                                    You have access to powerful tools for protein analysis, but this message doesn't require tool usage. 
                                    Provide helpful, accurate information about protein engineering, bioinformatics, molecular biology, and related topics.
                                    Keep responses conversational yet informative. If the user seems to need tool-based analysis, gently suggest what they might want to do and mention the available tools."""
                }
            ]
            
            # Add conversation history (last 10 messages)
            for msg in history[-10:]:
                if hasattr(msg, 'content'): # Basic check for message structure
                    role = "user" if "Human" in str(type(msg)) else "assistant"
                    messages.append({"role": role, "content": msg.content})
            current_msg = user_input
            if files:
                current_msg += f" [Files attached: {', '.join([os.path.basename(f) for f in files])}]"
            messages.append({"role": "user", "content": current_msg})
            return messages
    
    return EnhancedAgent(llm, tools, prompt, memory)

# Update the send_message function in create_chat_tab
def create_chat_tab(constant: Dict[str, Any]) -> Dict[str, Any]:
    """Creates the enhanced Chat tab with AI-powered intent recognition"""
    
    # Create initial conversation
    initial_conv_id = conv_manager.create_conversation("New Chat")
    conv_manager.active_conversation = initial_conv_id
    
    def create_new_conversation():
        """Create a new conversation"""
        conv_id = conv_manager.create_conversation()
        conv_manager.active_conversation = conv_id
        
        # Update conversation list
        conversations = conv_manager.list_conversations()
        choices = [(f"{conv['title']} ({conv['created_at'].strftime('%H:%M')})", conv['id']) for conv in conversations]
        
        return (
            gr.Dropdown(choices=choices, value=conv_id),
            [],
            gr.MultimodalTextbox(value=None)
        )
    
    def switch_conversation(selected_conv_id):
        """Switch to a different conversation"""
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
        
        return [], gr.MultimodalTextbox(value=None)
    
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
    
    def send_message(history, message, selected_conv_id, system_prompt):
        """Enhanced message sending with AI intent recognition"""
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
        has_csv = files and any(f.lower().endswith('.csv') for f in files if isinstance(f, str))

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
            history.append({"role": "assistant", "content": "Error: Could not find the active conversation. Please start a new chat."})
            yield history, gr.MultimodalTextbox(value=None)
            return

        display_text = text
        if file_paths:
            file_names = ", ".join([os.path.basename(f) for f in file_paths])
            display_text += f"📎 *Attached files: {file_names}*"
        history.append({"role": "user", "content": display_text})

        history.append({"role": "assistant", "content": "🤔 Analyzing your request..."})
        yield history, gr.MultimodalTextbox(value=None, interactive=False)

        agent_input = text
        if file_paths:
            agent_input += f"[CONTEXT: User has uploaded the following files. Use their full paths when a tool requires a file path: {', '.join(file_paths)}]"

        try:
            result = conv['agent'].invoke({
                "input": agent_input,
                "files": file_paths
            })
            response = result.get("output", "Sorry, I encountered an issue and could not generate a response.")
        except Exception as e:
            response = f"❌ An error occurred while processing your request: {str(e)}"


        history[-1] = {"role": "assistant", "content": response}
        yield history, gr.MultimodalTextbox(value=None, interactive=True)

    
    # Create UI with enhanced styling
    with gr.Column(elem_classes="chat-container"):
        # Header with title and conversation management
        with gr.Row(elem_classes="header-row"):
            gr.Markdown("# 🧬 VenusFactory AI Assistant", elem_classes="main-title")
            
        with gr.Row():
            with gr.Column(scale=4):
                conversations = conv_manager.list_conversations()
                choices = [(f"{conv['title']} ({conv['created_at'].strftime('%H:%M')})", conv['id']) for conv in conversations]
                
                conversation_dropdown = gr.Dropdown(
                    choices=choices,
                    value=initial_conv_id,
                    label="💬 Conversations",
                    interactive=True,
                    elem_classes="conversation-dropdown"
                )
            
            with gr.Column(scale=1):
                with gr.Row():
                    new_chat_btn = gr.Button("➕ New Chat", variant="primary", size="sm")
                    delete_chat_btn = gr.Button("🗑️ Delete", variant="secondary", size="sm")
        
        # Settings accordion
        with gr.Accordion("⚙️ Settings", open=False):
            with gr.Row():
                with gr.Column():
                    system_prompt_input = gr.Textbox(
                        label="System Prompt",
                        placeholder="Customize AI assistant behavior...",
                        lines=3,
                        value="You are VenusFactory AI Assistant, a specialized protein engineering and bioinformatics expert."
                    )
                with gr.Column():
                    gr.Markdown("""
                    ### 🔧 Configuration
                    - **AI Intent Recognition:** Enabled
                    - **Tools:** 4 protein analysis tools available
                    - **Context:** Last 10 messages remembered
                    """)
        
        # Main chat interface
        chatbot = gr.Chatbot(
            label="VenusFactory AI Assistant",
            type="messages",
            height=650,
            show_label=True,
            avatar_images=(
                None,
                "https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/venus_logo.png",
            ),
            bubble_full_width=False,
            show_copy_button=True,
            elem_classes="main-chatbot"
        )
        
        # Enhanced input area
        chat_input = gr.MultimodalTextbox(
            interactive=True,
            file_count="multiple",
            placeholder="💬 Ask me about protein engineering, upload files (FASTA, PDB), or request analysis...",
            show_label=False,
            file_types=[".fasta", ".fa", ".pdb", ".csv"],
            elem_classes="chat-input"
        )
        gr.Markdown(
            "<p style='text-align: center;'>Responses from VenusAgent, including information related to people, may not be completely accurate. Please verify carefully.</p>",
            elem_classes="status-indicator"
        )        
        # Tool information panel
        with gr.Accordion("🔬 AI-Powered Analysis Tools", open=False):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    ### 🎯 Intelligent Tool Selection
                    The AI automatically selects the best tool based on your request:
                    
                    **🧬 Mutation Prediction**
                    - Sequence-based: VenusPLM, ESM-1v, ESM2-650M, ESM-1b, 
                    - Structure-based: ProSST-2048, ProtSSN, SaProt, ESM-IF1, MIF-ST, 
                    
                    **⚗️ Function Protein Prediction**
                    - Solubility, Localization, Metal ion binding
                    - Stability, Sorting signal, Optimum temperature

                    **🔬 Function Residue Prediction**
                    - Activity, Binding, Motif, Evolutionary
                    """)
                
                with gr.Column():
                    gr.Markdown("""
                    ### 📊 Database Integration
                    
                    **🔍 InterPro Queries**
                    - Protein function annotations
                    - GO terms and domains
                    - Automatic UniProt ID detection
                    
                    **💡 Smart Features**
                    - Physical and chemical properties
                    - Relative solvent accessible surface area
                    - SASA value
                    - Secondary structure
                    """)
        
        # Status indicator
        gr.Markdown("🟢 **Status:** AI Intent Recognition Active | DeepSeek API Connected", elem_classes="status-indicator")
    
    # Event handlers with enhanced functionality
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
        inputs=[chatbot, chat_input, conversation_dropdown, system_prompt_input],
        outputs=[chatbot, chat_input]
    )
    
    return {
        "chatbot": chatbot,
        "chat_input": chat_input,
        "conversation_dropdown": conversation_dropdown,
        "new_chat_btn": new_chat_btn,
        "delete_chat_btn": delete_chat_btn
    }