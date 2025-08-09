import gradio as gr
import json
import os
import re
import requests
import base64
import numpy as np
from typing import Dict, Any, List, Optional
import tempfile
import shutil
import time
from pathlib import Path
from dotenv import load_dotenv
from gradio_client import Client, handle_file
from web.get_interpro_function import download_single_interpro, generate_interpro_ai_summary
import pandas as pd
load_dotenv()

class DeepSeekChat:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = "https://api.deepseek.com/v1"
        self.conversation_history = []
        
    def chat(self, message: str, files: List[str] = None, system_prompt: str = None) -> str:
        """Send a message to DeepSeek API and return the response"""
        if not self.api_key:
            return "error: no api key"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Prepare messages
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Prepare content for current message
        content = [{"type": "text", "text": message}]
        
        # Add files if provided
        if files:
            for file_path in files:
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'rb') as f:
                            file_data = f.read()
                        
                        # Determine file type
                        file_ext = Path(file_path).suffix.lower()
                        if file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
                            mime_type = f"image/{file_ext[1:]}"
                        elif file_ext in ['.pdf']:
                            mime_type = "application/pdf"
                        elif file_ext in ['.txt', '.py', '.js', '.html', '.css', '.json', '.md', '.fasta', '.fa', '.pdb']:
                            mime_type = f"text/{file_ext[1:]}"
                        else:
                            mime_type = "application/octet-stream"
                        
                        # Encode file as base64
                        file_base64 = base64.b64encode(file_data).decode('utf-8')
                        
                        # For text files, also include the text content
                        if mime_type.startswith('text/'):
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    text_content = f.read()
                                content.append({
                                    "type": "text",
                                    "text": f"[File content: {text_content}]"
                                })
                            except:
                                pass
                        
                        # Add file attachment
                        content.append({
                            "type": "file_url",
                            "file_url": {
                                "url": f"data:{mime_type};base64,{file_base64}",
                                "detail": "high"
                            }
                        })
                    except Exception as e:
                        content.append({
                            "type": "text", 
                            "text": f"[file processing error: {file_path} - {str(e)}]"
                        })
        
        messages.append({"role": "user", "content": content})
        
        # Prepare request payload
        payload = {
            "model": "deepseek-chat",
            "messages": messages,
            "max_tokens": 4000,
            "temperature": 0.7
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
                assistant_message = result['choices'][0]['message']['content']
                
                # Update conversation history
                self.conversation_history.append({"role": "user", "content": message})
                self.conversation_history.append({"role": "assistant", "content": assistant_message})
                
                # Keep only last 10 exchanges to manage memory
                if len(self.conversation_history) > 20:
                    self.conversation_history = self.conversation_history[-20:]
                
                return assistant_message
            else:
                error_msg = f"API request failed: {response.status_code} - {response.text}"
                return error_msg
                
        except Exception as e:
            return f"request error: {str(e)}"
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []

def create_chat_tab(constant: Dict[str, Any]) -> Dict[str, Any]:
    """Creates the Chat tab with DeepSeek API integration"""
    
    # Initialize chat instance
    chat_instance = DeepSeekChat()
    
    def add_message(history, message):
        """Add user message to chat history"""
        if message and message.get("text"):
            # Extract text and files from the multimodal input
            text = message["text"]
            files = message.get("files", [])
            
            # Store files in a global variable - extract file paths properly
            global current_files
            current_files = []
            if files:
                for file in files:
                    if hasattr(file, 'name'):
                        current_files.append(file.name)
                    elif isinstance(file, str):
                        current_files.append(file)
                    else:
                        current_files.append(str(file))
            
            display_text = text
            if current_files:
                display_text += f" [Files: {', '.join([os.path.basename(f) for f in current_files])}]"
            
            history.append({"role": "user", "content": display_text})
        
        return history, gr.MultimodalTextbox(
            interactive=True,
            file_count="multiple", 
            placeholder="💬 Enter your message or upload files (FASTA, PDB, CIF)...",
            show_label=False,
            file_types=[".fasta", ".fa", ".pdb", ".cif"],
            value=None
        )
        
    def bot_response(history, api_key, system_prompt):
        """Generate bot response"""
        global current_files

        if not history:
            return history
        
        # Get the last user message
        last_message = history[-1]
        if last_message["role"] != "user":
            return history
        
        user_content = last_message["content"]
        message_text = user_content
        
        # Update API key if provided
        if api_key:
            chat_instance.api_key = api_key
        
        # Use the globally stored files
        files = getattr(globals(), 'current_files', [])
        
        try:
            # Detect user intent and handle VenusFactory functions
            intent = detect_user_intent(message_text, current_files)
            
            # Check if FASTA file is uploaded
            has_fasta_file = 'fasta_file' in intent
            has_structure_file = 'structure_file' in intent
            
            # Process the request based on intent
            if has_fasta_file and any(word in message_text.lower() for word in ['mutation', 'mutant', 'mutate', 'zero-shot', 'zero shot', '设计', 'design', 'function', 'solubility', 'localization', 'binding', 'stability', 'sorting', 'temperature']):
                # Use VenusFactory API for FASTA files
                structure_models = ['SaProt', 'ProtSSN', 'ESM-IF1', 'MIF-ST', 'ProSST-2048']
                if intent['action'] == 'predict_zero_shot_structure' and intent['model'] in structure_models:
                    response = f"❌ **Error**: You requested {intent['model']} which is a structure-based model, but you only uploaded a FASTA file.\n\n"
                    response += f"**Solution**: Please upload a PDB or CIF structure file instead of FASTA, or use a sequence-based model like ESM-1v or ESM2-650M.\n\n"
                    response += f"**Available options**:\n"
                    response += f"- **Structure models** (need PDB/CIF): SaProt, ProtSSN, ESM-IF1, MIF-ST, ProSST-2048\n"
                    response += f"- **Sequence models** (work with FASTA): ESM-1v, ESM2-650M, ESM-1b\n"
                elif intent['action'] == 'predict_zero_shot_sequence':
                    response = call_zero_shot_sequence_prediction_from_file(intent['fasta_file'], intent['model'], api_key)
                elif intent['action'] == 'predict_function':
                    response = call_protein_function_prediction_from_file(intent['fasta_file'], intent['model'], intent['task'], api_key)
                elif intent['action'] == "query_interpro":
                    response = call_interpro_function_query(intent['uniprot_id'], api_key)
                else:
                    response = call_zero_shot_sequence_prediction_from_file(intent['fasta_file'], intent['model'], api_key)
            elif has_structure_file and any(word in message_text.lower() for word in ['mutation', 'mutant', 'mutate', 'zero-shot', 'zero shot', '设计', 'design']):
                response = call_zero_shot_structure_prediction_from_file(intent['structure_file'], intent['model'], api_key)
            elif intent['action'] != 'chat':
                # Handle VenusFactory specific functions
                if intent['action'] == 'predict_zero_shot_sequence':
                    if 'fasta_file' in intent:
                        response = call_zero_shot_sequence_prediction_from_file(intent['fasta_file'], intent['model'], api_key)
                    elif intent['sequence']:
                        response = predict_zero_shot_sequence(intent['sequence'], intent['model'], api_key)
                    else:
                        response = "Error: No sequence or FASTA file provided for mutation prediction"
                elif intent['action'] == 'predict_zero_shot_structure':
                    if 'structure_file' in intent:
                        response = call_zero_shot_structure_prediction_from_file(intent['structure_file'], intent['model'], api_key)
                    else:
                        response = "Error: No structure file provided for structure-based mutation prediction"
                elif intent['action'] == 'predict_function':
                    if 'fasta_file' in intent:
                        response = call_protein_function_prediction_from_file(intent['fasta_file'], intent['model'], intent['task'], api_key)
                    elif intent['sequence']:
                        response = predict_protein_function(intent['sequence'], intent['model'], intent['task'], api_key)
                    else:
                        response = "Error: No sequence or FASTA file provided for function prediction"
                elif intent['action'] == "query_interpro":
                    response = call_interpro_function_query(intent['uniprot_id'], api_key)
                else:
                    response = chat_instance.chat(message_text, [], system_prompt)
            else:
                # Use DeepSeek API for general chat
                response = chat_instance.chat(message_text, [], system_prompt)
        except Exception as e:
            response = f"Error processing request: {str(e)}"
        
        # Add bot response to history
        history.append({"role": "assistant", "content": response})
        current_files = []
        return history
    
    def send_message_handler(history, message, api_key, system_prompt):
        """Handle sending message and getting response"""
        # Add user message
        history, cleared_input = add_message(history, message)
        # Get bot response  
        history = bot_response(history, api_key, system_prompt)
        return history, cleared_input
    
    def clear_chat():
        """Clear chat history"""
        chat_instance.clear_history()
        return []
    
    
    # Create UI with modern layout
    with gr.Column(elem_classes="chat-container"):
        # Collapsible settings section
        gr.Markdown("### 📍 The AI Memory feature is not included in the free trial version.")
        with gr.Accordion("⚙️ Settings", open=False):
            with gr.Row():
                api_key_input = gr.Textbox(
                    label="DeepSeek API Key",
                    placeholder="Enter your DeepSeek API Key (we have a free key for you)",
                    value=os.getenv("DEEPSEEK_API_KEY"),
                    type="password",
                    scale=3
                )
            
            system_prompt_input = gr.Textbox(
                label="System Prompt",
                placeholder="Set AI assistant role and behavior...",
                lines=3,
                value="You are VenusFactory AI Assistant, a specialized protein engineering and bioinformatics expert. You can help users with:\n\n1. Zero-shot mutation prediction using sequence-based models (ESM-1v, ESM2-650M, ESM-1b) and structure-based models (SaProt, ProtSSN, ESM-IF1, MIF-ST, ProSST-2048)\n2. Protein function prediction including solubility, localization, metal ion binding, stability, sorting signal, and optimum temperature.\n Always respond in English and provide clear, actionable insights."
            )
        
        # Main chat interface
        chatbot = gr.Chatbot(
            label="VenusFactory AI Assistant",
            type="messages",
            height=900,
            show_label=True,
            avatar_images=(
                None,
                "https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/venus_logo.png",
            ),
            bubble_full_width=False,
            show_copy_button=True,
            elem_classes="main-chatbot"
        )
        
        current_files = []
        # Input area with integrated file upload
        with gr.Row(elem_classes="input-row"):
            with gr.Column(scale=8):
                chat_input = gr.MultimodalTextbox(
                    interactive=True,
                    file_count="multiple",
                    placeholder="💬 Enter your message or upload files (FASTA, PDB, CIF)...",
                    show_label=False,
                    file_types=[".fasta", ".fa", ".pdb", ".cif"],
                    elem_classes="chat-input"
                )
                # Buttons row positioned at bottom of input area
                # with gr.Row():
                #     send_btn = gr.Button(
                #         "📤 Send", 
                #         variant="primary", 
                #         scale=1,
                #         min_width=80,
                #         elem_classes="send-button"
                #     )
                #     clear_btn = gr.Button(
                #         "🗑️ Clear", 
                #         variant="secondary", 
                #         scale=1,
                #         min_width=80,
                #         elem_classes="clear-button"
                #     )
        # VenusFactory API integration info (collapsible)
        with gr.Accordion("🔬 VenusFactory API Functions", open=False):
            gr.Markdown("""
            ### Available Functions
            
            **🧬 Zero-shot Mutation Prediction:**
            - **Sequence-based**: "Predict mutations using ESM-1v for this sequence: [sequence]"
            - **Structure-based**: "Predict mutations using SaProt for this structure" (upload PDB/CIF file)
            
            **⚗️ Protein Function Prediction:**
            - "Predict solubility for this protein: [sequence]"
            - "Predict localization using Ankh-large for this protein: [sequence]"
            - "Predict metal ion binding for this protein: [sequence]"
            
            **📊 InterPro Function Query:**
            - "Query function for UniProt ID P00734"
            - "Get InterPro annotation for P12345"
            - "What is the function of protein Q9Y6K9?"
            
            **📁 Supported File Types:**
            - **Sequence**: FASTA files (.fasta, .fa)
            - **Structure**: PDB files (.pdb), CIF files (.cif)
            
            **🤖 Supported Models:**
            - **Sequence-based**: ESM-1v, ESM2-650M, ESM-1b
            - **Structure-based**: SaProt, ProtSSN, ESM-IF1, MIF-ST, ProSST-2048
            - **Function Models**: ESM2-650M, Ankh-large, ProtBert-uniref50, ProtT5-xl-uniref50
            
            **🎯 Function Tasks:**
            - Solubility, Localization, Metal Ion Binding, Stability, Sorting signal, Optimum temperature
            """)
            
            # API status indicator
            api_status = gr.HTML(
                value='<div style="color: green; font-weight: bold;">✅ VenusFactory API Ready</div>',
                elem_classes="api-status"
            )
    
    chat_input.submit(
        fn=send_message_handler,
        inputs=[chatbot, chat_input, api_key_input, system_prompt_input],
        outputs=[chatbot, chat_input]
    )
    
    
    return {
        "chatbot": chatbot,
        "chat_input": chat_input,
        "api_key_input": api_key_input,
    }

def call_zero_shot_sequence_prediction(sequence: str, model_name: str = "ESM2-650M", api_key: str = None) -> str:
    """Call VenusFactory zero-shot sequence-based mutation prediction API"""
    try:
        # Create a temporary FASTA file
        temp_fasta = tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False)
        temp_fasta.write(f">temp_sequence\n{sequence}\n")
        temp_fasta.close()
        
        # Call the Gradio API for sequence-based prediction
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
        # Clean up temporary file
        os.unlink(temp_fasta.name)
        
        return parse_zero_shot_prediction_result(result, api_key)
    except Exception as e:
        return f"Zero-shot sequence prediction error: {str(e)}"

def call_zero_shot_sequence_prediction_from_file(fasta_file: str, model_name: str = "ESM2-650M", api_key: str = None) -> str:
    """Call VenusFactory zero-shot sequence prediction API using uploaded FASTA file"""
    try:
        # Call the Gradio API for sequence-based prediction
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
        # Call the Gradio API for structure-based prediction
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
        # Parse the result - handle_mutation_prediction_advance returns multiple values
        return parse_zero_shot_prediction_result(result, api_key)
    except Exception as e:
        return f"Zero-shot structure prediction error: {str(e)}"

def parse_zero_shot_prediction_result(result: tuple, api_key: str = None) -> str:
    """Parse zero-shot prediction result with AI summary"""
    try:
        import pandas as pd
        
        # Extract DataFrame from result[2]
        data = result[2]
        
        # Handle different data formats
        if isinstance(data, dict) and 'headers' in data and 'data' in data:
            # Convert dict format to DataFrame
            df = pd.DataFrame(data['data'], columns=data['headers'])
        elif isinstance(data, pd.DataFrame):
            # Already a DataFrame
            df = data
        else:
            return f"Error: Unexpected data format: {type(data)}"
        
        if df is None or df.empty:
            return "Error: No prediction data available"
        
        # Generate AI summary with provided api_key
        ai_summary = generate_simple_ai_summary_mutatuon(df, api_key)
        
        # Return the AI summary
        return ai_summary
        
    except Exception as e:
        return f"Error parsing prediction result: {str(e)}"

def generate_simple_ai_summary_function(df, task, api_key: str = None) -> str:
    try:
        import requests
        import os
        
        # Use provided API key or fallback to environment variable
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


def generate_simple_ai_summary_mutatuon(df, api_key: str = None) -> str:
    """Generate simple AI summary for mutation prediction results"""
    try:
        import requests
        import os
        
        # Use provided API key or fallback to environment variable
        if not api_key:
            api_key = os.getenv("DEEPSEEK_API_KEY")
        
        if not api_key:
            return "❌ No API key found. Please set DEEPSEEK_API_KEY environment variable or provide API key in the interface."
        
        # Prepare data for AI analysis
        score_col = next(
        (col for col in df.columns if 'Prediction Rank' in col.lower()), 
        df.columns[1] if len(df.columns) > 1 else None
        )
    
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

def call_protein_function_prediction_from_file(fasta_file: str, model_name: str = "ProtT5-xl-uniref50", task: str = "Solubility", api_key: str = None) -> str:
    """Call VenusFactory protein function prediction API using uploaded FASTA file"""

    try:
        # Get datasets for the task
        dataset_mapping = {
            "Solubility": ["DeepSol", "DeepSoluE", "ProtSolM"],
            "Localization": ["DeepLocBinary", "DeepLocMulti"],
            "Metal ion binding": ["MetalIonBinding"],
            "Stability": ["Thermostability"],
            "Sorting signal": ["SortingSignal"],
            "Optimum temperature": ["DeepET_Topt"]
        }
        datasets = dataset_mapping.get(task, ["DeepSol"])
        
        # Call the Gradio API
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
        
        # Parse the result
        return parse_function_prediction_result(result, model_name, task, api_key)
    except Exception as e:
        return f"Protein function prediction error: {str(e)}"

def parse_function_prediction_result(result, model_name: str, task: str, api_key: str = None) -> str:
    try:
        import pandas as pd
        
        # Extract DataFrame from result[1]
        data = result[1]
        
        # Handle different data formats
        if isinstance(data, dict) and 'headers' in data and 'data' in data:
            # Convert dict format to DataFrame
            df = pd.DataFrame(data['data'], columns=data['headers'])
        elif isinstance(data, pd.DataFrame):
            # Already a DataFrame
            df = data
        else:
            return f"Error: Unexpected data format: {type(data)}"
        
        if df is None or df.empty:
            return "Error: No prediction data available"
        
        ai_summary = generate_simple_ai_summary_function(df, task, api_key)

        # Return the AI summary
        return ai_summary
        
    except Exception as e:
        return f"Error parsing prediction result: {str(e)}"

def call_protein_function_prediction(sequence: str, model_name: str = "ProtT5-xl-uniref50", task: str = "Solubility", api_key: str = None) -> str:
    """Call VenusFactory protein function prediction API"""
    try:
        # Create a temporary FASTA file
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
        # Call the Gradio API
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

        # Clean up temporary file
        os.unlink(temp_fasta.name)
        
        return parse_function_prediction_result(result, model_name, task)
    except Exception as e:
        return f"Protein function prediction error: {str(e)}"

def predict_zero_shot_sequence(sequence: str, model_name: str = "ESM2-650M", api_key: str = None) -> str:
    """Predict mutations using sequence-based zero-shot models"""
    try:
        sequence = sequence.strip().upper()
        if not sequence or not all(aa in 'ACDEFGHIKLMNPQRSTVWY' for aa in sequence):
            return "Error: Please enter a valid protein sequence (only standard amino acids)"
        
        # Call VenusFactory zero-shot sequence prediction API
        result = call_zero_shot_sequence_prediction(sequence, model_name, api_key)
        
        return f"{result}"
    except Exception as e:
        return f"Sequence prediction error: {str(e)}"

def predict_zero_shot_structure(structure_file: str, model_name: str = "ESM-IF1", api_key: str = None) -> str:
    """Predict mutations using structure-based zero-shot models"""
    try:

        # Call VenusFactory zero-shot structure prediction API
        result = call_zero_shot_structure_prediction_from_file(structure_file, model_name, api_key)
        
        return f"Zero-shot Structure-based Mutation Prediction Results:\n\n{result}"
    except Exception as e:
        return f"Structure prediction error: {str(e)}"

def predict_protein_function(sequence: str, model_name: str = "ProtT5-xl-uniref50", task: str = "Solubility", api_key: str = None) -> str:
    """Predict protein function using VenusFactory models"""
    try:
        sequence = sequence.strip().upper()
        if not sequence or not all(aa in 'ACDEFGHIKLMNPQRSTVWY' for aa in sequence):
            return "Error: Please enter a valid protein sequence (only standard amino acids)"
        
        # Call VenusFactory protein function prediction API
        result = call_protein_function_prediction(sequence, model_name, task)
        
        return f"{result}"
    except Exception as e:
        return f"Function prediction error: {str(e)}"

def extract_sequence_from_message(message: str) -> str:
    """Extract protein sequence from user message"""
    # Common patterns for protein sequences
    patterns = [
        r'[ACDEFGHIKLMNPQRSTVWY]{10,}',  # Pure amino acid sequence
        r'序列[：:]\s*([ACDEFGHIKLMNPQRSTVWY]+)',  # Chinese pattern
        r'sequence[：:]\s*([ACDEFGHIKLMNPQRSTVWY]+)',  # English pattern
        r'蛋白质[：:]\s*([ACDEFGHIKLMNPQRSTVWY]+)',  # Chinese protein pattern
    ]
    
    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            if len(match.groups()) > 0:
                return match.group(1)
            else:
                return match.group(0)
    
    return ""

def extract_uniprot_id_from_message(message: str) -> str:
    """Extract UniProt ID from user message"""
    # UniProt ID patterns: P12345, Q9Y6K9, etc.
    patterns = [
        r'\b[OPQ][0-9][A-Z0-9]{3}[0-9]\b',  # Standard UniProt format
        r'\b[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}\b',  # Alternative formats
        r'uniprot[：:\s]+([A-Z0-9]+)',  # "uniprot: P12345"
        r'id[：:\s]+([A-Z0-9]+)',  # "id: P12345"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            if len(match.groups()) > 0:
                return match.group(1).upper()
            else:
                return match.group(0).upper()
    
    return ""

def call_interpro_function_query(uniprot_id: str, api_key: str = None) -> str:
    try:
        if not uniprot_id:
            return "Error: No UniProt ID provided for InterPro query"

        result_json = download_single_interpro(uniprot_id)
        result_data = json.loads(result_json)
        
        if not result_data.get('success', False):
            return f"Error: {result_data.get('error_message', 'Unknown error occurred')}"
        ai_summary = generate_interpro_ai_summary(result_data, api_key)
        
        return ai_summary
        
    except Exception as e:
        return f"InterPro query error: {str(e)}"

def detect_user_intent(message: str, files: List[str] = None) -> dict:
    """Detect user intent using AI-based analysis"""
    
    # Default intent
    intent = {
        'type': 'general',
        'sequence': '',
        'action': 'chat', 
        'model': 'ESM2-650M',
        'task': 'Solubility',
        'prediction_type': 'sequence',
        'uniprot_id': ''
    }
    
    # Extract sequence if present
    sequence = extract_sequence_from_message(message)
    if sequence:
        intent['sequence'] = sequence
    
    uniprot_id = extract_uniprot_id_from_message(message)
    if uniprot_id:
        intent['uniprot_id'] = uniprot_id

    # Check uploaded files - fix file path handling
    has_structure_file = False
    has_fasta_file = False

    if files:
        for file in files:
            # Handle both file objects and file paths
            file_path = file if isinstance(file, str) else getattr(file, 'name', str(file))

            if file_path and os.path.exists(file_path):
                if file_path.lower().endswith(('.pdb', '.cif')):
                    has_structure_file = True
                    intent['structure_file'] = file_path
                    intent['model'] = 'ESM-IF1'
                    intent['prediction_type'] = 'structure'
                elif file_path.lower().endswith(('.fasta', '.fa')):
                    has_fasta_file = True
                    intent['fasta_file'] = file_path
                    intent['model'] = 'ESM2-650M'
                    intent['prediction_type'] = 'sequence'
    # Use AI to analyze intent if we have an API key

    try:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if api_key:
            intent = analyze_intent_with_ai(message, has_fasta_file, has_structure_file, intent)
        else:
            # If no API key, use fallback
            intent = fallback_intent_detection(message, has_fasta_file, has_structure_file, intent)
    except Exception as e:
        # Fallback to simple keyword detection
        intent = fallback_intent_detection(message, has_fasta_file, has_structure_file, intent)
    return intent

def analyze_intent_with_ai(message: str, has_fasta_file: bool, has_structure_file: bool, default_intent: dict) -> dict:
    """Use AI to analyze user intent"""
    
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return fallback_intent_detection(message, has_fasta_file, has_structure_file, default_intent)
    
    # Create context for AI analysis
    context = f"""
User message: "{message}"
Uploaded files: FASTA={has_fasta_file}, Structure={has_structure_file}

Available VenusFactory functions:
1. Zero-shot mutation prediction (sequence-based): ESM-1v, ESM2-650M, ESM-1b
2. Zero-shot mutation prediction (structure-based): ESM-IF1, SaProt, ProtSSN, MIF-ST, ProSST-2048
3. Protein function prediction: Solubility, Localization, Metal ion binding, Stability, Sorting signal, Optimum temperature
4. InterPro protein function query: Query protein annotations and GO terms from InterPro database

TASK: Analyze the user's intent and return ONLY a JSON response. Be very precise about the action type.

ACTION RULES:
- "predict_zero_shot_sequence": When user wants mutation/mutant prediction with sequence data or FASTA files
- "predict_zero_shot_structure": When user wants mutation/mutant prediction with structure files (PDB/CIF) OR specifically mentions structure-based models
- "predict_function": When user wants to predict protein properties like solubility, localization, metal ion binding, stability, etc.
- "query_interpro": When user wants to query InterPro database for protein function/annotation OR mentions "function", "interpro", "go annotation", "annotation"
- "chat": For general questions, greetings, or unclear requests

MODEL RULES:
- For zero-shot sequence: ESM-1v, ESM2-650M, ESM-1b
- For zero-shot structure: ESM-IF1, SaProt, ProtSSN, MIF-ST, ProSST-2048  
- For function prediction: always use "ESM2-650M"
- For InterPro query: not applicable

IMPORTANT: If Structure=True, the action should be "predict_zero_shot_structure" and model should be from structure models.

Return JSON format:
{{
    "action": "chat|predict_zero_shot_sequence|predict_zero_shot_structure|predict_function|query_interpro",
    "model_name": "model_name_or_null",
    "task": "task_name_or_null",
    "reasoning": "brief explanation"
}}
"""
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system", 
                    "content": "You are an intent classifier. Analyze the user's message and return ONLY a JSON response. Be precise about categorizing intentions."
                },
                {
                    "role": "user", 
                    "content": context
                }
            ],
            "temperature": 0.0,
            "max_tokens": 300
        }
        
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            ai_response = result['choices'][0]['message']['content']
            
            # Parse AI response
            try:
                # Clean up the response - remove markdown code blocks if present
                cleaned_response = ai_response.strip()
                if cleaned_response.startswith('```json'):
                    cleaned_response = cleaned_response[7:]
                if cleaned_response.startswith('```'):
                    cleaned_response = cleaned_response[3:]
                if cleaned_response.endswith('```'):
                    cleaned_response = cleaned_response[:-3]
                cleaned_response = cleaned_response.strip()
                
                ai_intent = json.loads(cleaned_response)
                
                valid_actions = ["chat", "predict_zero_shot_sequence", "predict_zero_shot_structure", "predict_function", "query_interpro"]
                if ai_intent.get('action') not in valid_actions:
                    return fallback_intent_detection(message, has_fasta_file, has_structure_file, default_intent)
                
                default_intent['action'] = ai_intent.get('action', 'chat')
                ai_model = ai_intent.get('model_name')
                
                if ai_intent.get('action') == 'predict_function':
                    default_intent['task'] = ai_intent.get('task', 'Solubility')
                    default_intent['model'] = 'ESM2-650M'
                elif ai_intent.get('action') == 'predict_zero_shot_structure':
                    structure_models = ['SaProt', 'ProtSSN', 'ESM-IF1', 'MIF-ST', 'ProSST-2048']
                    if ai_model and ai_model in structure_models:
                        default_intent['model'] = ai_model
                    elif default_intent['model'] not in structure_models:
                        default_intent['model'] = 'ESM-IF1'
                    default_intent['task'] = None
                elif ai_intent.get('action') == 'predict_zero_shot_sequence':
                    sequence_models = ['ESM-1v', 'ESM2-650M', 'ESM-1b']
                    if ai_model and ai_model in sequence_models:
                        default_intent['model'] = ai_model
                    else:
                        default_intent['model'] = 'ESM2-650M'
                    default_intent['task'] = None
                else:
                    default_intent['task'] = None
                
                if 'zero_shot' in ai_intent.get('action', ''):
                    default_intent['type'] = 'zero_shot_prediction'
                elif 'function' in ai_intent.get('action', ''):
                    default_intent['type'] = 'function_prediction'
                elif 'interpro' in ai_intent.get('action', ''):
                    default_intent['type'] = 'interpro_query'
                else:
                    default_intent['type'] = 'general'
                
                return default_intent
                
            except json.JSONDecodeError as e:
                return fallback_intent_detection(message, has_fasta_file, has_structure_file, default_intent)
        else:
            return fallback_intent_detection(message, has_fasta_file, has_structure_file, default_intent)
            
    except Exception as e:
        return fallback_intent_detection(message, has_fasta_file, has_structure_file, default_intent)

def fallback_intent_detection(message: str, has_fasta_file: bool, has_structure_file: bool, intent: dict) -> dict:
    """Fallback intent detection with exact matching to Gradio UI choices"""
    message_lower = message.lower()

    # InterPro query detection
    interpro_keywords = ['function', 'interpro', 'annotation', 'go term', 'go annotation', '功能', '注释']
    is_interpro_request = any(word in message_lower for word in interpro_keywords)
    has_uniprot_id = bool(intent.get('uniprot_id'))

    if is_interpro_request and has_uniprot_id:
        intent['action'] = 'query_interpro'
        intent['type'] = 'interpro_query'
        return intent

    mutation_keywords = ['mutation', 'mutant', 'mutate', '突变', '设计', 'design', 'zero-shot', 'zero shot']
    is_mutation_request = any(word in message_lower for word in mutation_keywords)

    function_keywords = {
        'Solubility': ['solubility', '溶解度', 'soluble'],
        'Localization': ['localization', '定位', 'location', 'subcellular'],
        'Metal ion binding': ['binding', '结合', 'bind', 'interaction', 'metal ion', 'metal'],
        'Stability': ['stability', '稳定性', 'stable', 'thermal'],
        'Sorting signal': ['sorting', 'signal', '信号', '分选'],
        'Optimum temperature': ['temperature', '温度', 'thermal', 'optimum', 'optimum temperature']
    }
    
    detected_function = None
    for func_name, keywords in function_keywords.items():
        if any(keyword in message_lower for keyword in keywords):
            detected_function = func_name 
            break
    
    sequence_models = {
        'ESM-1v': ['esm-1v', 'esm1v', 'esm_1v'],
        'ESM2-650M': ['esm2', 'esm-2', 'esm2-650m', 'esm2_650m', 'esm-2-650m'],
        'ESM-1b': ['esm1b', 'esm-1b']
    }
    
    structure_models = {
        'SaProt': ['saprot'],
        'ProtSSN': ['protssn'],
        'ESM-IF1': ['esm-if1', 'esmif1'],
        'MIF-ST': ['mif-st', 'mifst'],
        'ProSST-2048': ['prosst', 'prosst-2048']
    }
    
    # Default models
    detected_seq_model = 'ESM2-650M'
    detected_struct_model = 'ESM-IF1' 
    
    for model_name, keywords in sequence_models.items():
        if any(keyword in message_lower for keyword in keywords):
            detected_seq_model = model_name
            break
    
    for model_name, keywords in structure_models.items():
        if any(keyword in message_lower for keyword in keywords):
            detected_struct_model = model_name
            break

    if is_mutation_request:
        structure_indicators = ['structure', 'pdb', 'cif'] + [name.lower() for name in structure_models.keys()]
        wants_structure_based = (any(indicator in message_lower for indicator in structure_indicators) 
                               or has_structure_file)
        
        if wants_structure_based:
            intent['action'] = 'predict_zero_shot_structure'
            intent['type'] = 'zero_shot_prediction'
            intent['model'] = detected_struct_model
        else:
            intent['action'] = 'predict_zero_shot_sequence'
            intent['type'] = 'zero_shot_prediction'
            intent['model'] = detected_seq_model

    elif detected_function:
        intent['action'] = 'predict_function'
        intent['type'] = 'function_prediction'
        intent['model'] = 'ESM2-650M'
        intent['task'] = detected_function

    else:
        intent['action'] = 'chat'
        intent['type'] = 'general'

    return intent