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
    
    # Load CSS styles
    css_path = os.path.join(os.path.dirname(__file__), "assets", "chat_ui.css")
    if os.path.exists(css_path):
        try:
            custom_css = open(css_path, "r").read()
            gr.HTML(f"<style>{custom_css}</style>", visible=False)
        except Exception as e:
            print(f"Warning: Could not load chat CSS file: {e}")
    
    # Initialize chat instance
    chat_instance = DeepSeekChat()
    
    def send_message(message: str, api_key: str, files: List[str], system_prompt: str, history: List[List[str]]) -> tuple:
        try:
            import requests
            requests.post("/api/stats/track", json={"module": "agent_usage"})
        except Exception:
            pass
        if not message.strip():
            return "", history
        
        # Update API key if provided
        if api_key:
            chat_instance.api_key = api_key
        
        # Process uploaded files
        file_paths = []
        if files:
            for file in files:
                if hasattr(file, 'name') and file.name:
                    file_paths.append(file.name)
        
        # Detect user intent and handle VenusFactory functions
        intent = detect_user_intent(message, file_paths)
        print(f"DEBUG: Detected intent: {intent}")
        
        # Check if FASTA file is uploaded
        has_fasta_file = 'fasta_file' in intent
        
        try:
            # If FASTA file is uploaded, prioritize VenusFactory API
            if has_fasta_file and any(word in message.lower() for word in ['mutation', 'mutant', 'mutate', 'zero-shot', 'zero shot', '设计', 'design', 'function', 'solubility', 'localization', 'binding', 'stability', 'sorting', 'temperature']):
                # Use VenusFactory API for FASTA files
                print(f"DEBUG: Processing FASTA file with intent: {intent}")
                
                # Check if user wants structure-based model but only has FASTA file
                structure_models = ['SaProt', 'ProtSSN', 'ESM-IF1', 'MIF-ST', 'ProSST-2048']
                if intent['action'] == 'predict_zero_shot_structure' and intent['model'] in structure_models:
                    # User wants structure model but only has FASTA file - provide helpful error
                    response = f"❌ **Error**: You requested {intent['model']} which is a structure-based model, but you only uploaded a FASTA file.\n\n"
                    response += f"**Solution**: Please upload a PDB or CIF structure file instead of FASTA, or use a sequence-based model like ESM-1v or ESM2-650M.\n\n"
                    response += f"**Available options**:\n"
                    response += f"- **Structure models** (need PDB/CIF): SaProt, ProtSSN, ESM-IF1, MIF-ST, ProSST-2048\n"
                    response += f"- **Sequence models** (work with FASTA): ESM-1v, ESM2-650M, ESM-1b\n"
                elif intent['action'] == 'predict_zero_shot_sequence':
                    print(f"DEBUG: Calling sequence prediction with model: {intent['model']}")
                    response = call_zero_shot_sequence_prediction_from_file(intent['fasta_file'], intent['model'], api_key)
                elif intent['action'] == 'predict_function':
                    print(f"DEBUG: Calling function prediction with model: {intent['model']}, task: {intent['task']}")
                    response = call_protein_function_prediction_from_file(intent['fasta_file'], intent['model'], intent['task'], api_key)
                else:
                    # Default to sequence prediction for FASTA files
                    print(f"DEBUG: Defaulting to sequence prediction with model: {intent['model']}")
                    response = call_zero_shot_sequence_prediction_from_file(intent['fasta_file'], intent['model'], api_key)
            elif intent['action'] != 'chat':
                # Handle VenusFactory specific functions
                print(f"DEBUG: Processing without FASTA file, intent: {intent}")
                if intent['action'] == 'predict_zero_shot_sequence':
                    print(f"DEBUG: Processing sequence prediction")
                    if 'fasta_file' in intent:
                        # Use uploaded FASTA file
                        response = call_zero_shot_sequence_prediction_from_file(intent['fasta_file'], intent['model'], api_key)
                    elif intent['sequence']:
                        # Use extracted sequence
                        response = predict_zero_shot_sequence(intent['sequence'], intent['model'], api_key)
                    else:
                        response = "Error: No sequence or FASTA file provided for mutation prediction"
                elif intent['action'] == 'predict_zero_shot_structure':
                    print(f"DEBUG: Processing structure prediction, structure_file in intent: {'structure_file' in intent}")
                    if 'structure_file' in intent:
                        print(f"DEBUG: Calling structure prediction with model: {intent['model']}")
                        response = call_zero_shot_structure_prediction(intent['structure_file'], intent['model'], api_key)
                    else:
                        response = "Error: No structure file provided for structure-based mutation prediction"
                elif intent['action'] == 'predict_function':
                    print(f"DEBUG: Processing function prediction")
                    if 'fasta_file' in intent:
                        # Use uploaded FASTA file
                        response = call_protein_function_prediction_from_file(intent['fasta_file'], intent['model'], intent['task'], api_key)
                    elif intent['sequence']:
                        # Use extracted sequence
                        response = predict_protein_function(intent['sequence'], intent['model'], intent['task'], api_key)
                    else:
                        response = "Error: No sequence or FASTA file provided for function prediction"
                else:
                    print(f"DEBUG: Falling back to DeepSeek API")
                    # Fall back to DeepSeek API (without files to avoid JSON errors)
                    response = chat_instance.chat(message, [], system_prompt)
            else:
                # Use DeepSeek API for general chat (without files to avoid JSON errors)
                response = chat_instance.chat(message, [], system_prompt)
        except Exception as e:
            response = f"Error processing request: {str(e)}"
        
        # Update history
        history.append([message, response])
        
        return "", history

    
    def clear_chat(history: List[List[str]]) -> List[List[str]]:
        """Clear chat history"""
        chat_instancea.clear_history()
        return []
    
    def upload_files(files):
        """Handle file uploads"""
        if files:
            return f"Uploaded {len(files)} files: {', '.join([os.path.basename(f.name) for f in files])}"
        return "No files uploaded"
    
    # Create UI components
    with gr.Column():
        # API Key input
        api_key_input = gr.Textbox(
            label="DeepSeek API Key",
            placeholder="Enter your DeepSeek API Key, we have a free key for you",
            value=os.getenv("DEEPSEEK_API_KEY"),
            type="password",
            lines=1
        )
        
        # System prompt
        system_prompt_input = gr.Textbox(
            label="System Prompt (Optional)",
            placeholder="Set AI assistant role and behavior...",
            lines=3,
            value="You are VenusFactory AI Assistant, a specialized protein engineering and bioinformatics expert. You can help users with:\n\n1. Zero-shot mutation prediction using sequence-based models (ESM-1v, ESM2-650M, ESM-1b) and structure-based models (SaProt, ProtSSN, ESM-IF1, MIF-ST, ProSST-2048)\n2. Protein function prediction including solubility, localization, binding, stability, sorting signal, and optimum temperature.\n Always respond in English and provide clear, actionable insights."
        )
        
        # File upload
        file_upload = gr.File(
            label="Upload Files",
            file_count="multiple",
            file_types=[".fasta", ".cif", ".pdb", ".fa"]
        )
        
        file_status = gr.Textbox(
            label="File Status",
            interactive=False,
            lines=1
        )
        
        # Chat interface
        chatbot = gr.Chatbot(
            label="VenusFactory AI Assistant",
            height=500,
            show_label=True
        )
        
        # Message input
        msg_input = gr.Textbox(
            label="Input Message",
            placeholder="Enter your question...",
            lines=3
        )
        
        # Buttons
        with gr.Row():
            send_btn = gr.Button("Send", variant="primary")
            clear_btn = gr.Button("Clear Chat")
        
        # VenusFactory API integration section
        with gr.Accordion("VenusFactory API Integration", open=False):
            gr.Markdown("""
            ### Available VenusFactory Functions
            
            You can call VenusFactory functions using the following patterns:
            
            **1. Zero-shot Mutation Prediction:**
            - **Sequence-based**: "Predict mutations using ESM-1v for this sequence: [sequence]"
            - **Structure-based**: "Predict mutations using SaProt for this structure" (upload PDB/CIF file)
            
            **2. Protein Function Prediction:**
            - "Predict solubility for this protein: [sequence]"
            - "Predict localization using Ankh-large for this protein: [sequence]"
            - "Predict binding for this protein: [sequence]"
            
            The AI assistant will automatically detect your intent and call the appropriate function.
            
            **Supported Zero-shot Models:**
            - **Sequence-based**: ESM-1v, ESM2-650M
            - **Structure-based**: SaProt, ProtSSN, ESM-IF1, MIF-ST, ProSST-2048
            
            **Supported Function Models:**
            - ESM2-650M, Ankh-large, ProtBert-uniref50, ProtT5-xl-uniref50
            
            **Supported Function Tasks:**
            - Solubility, Localization, Binding, Stability, Sorting signal, Optimum temperature
            
            **File Types:**
            - Sequence: FASTA files (.fasta, .fa)
            - Structure: PDB files (.pdb), CIF files (.cif)
            """)
            
            # API status
            api_status = gr.Textbox(
                label="API Status",
                value="VenusFactory API Ready",
                interactive=False
            )
    
    # Event handlers
    file_upload.change(
        fn=upload_files,
        inputs=[file_upload],
        outputs=[file_status]
    )
    
    send_btn.click(
        fn=send_message,
        inputs=[msg_input, api_key_input, file_upload, system_prompt_input, chatbot],
        outputs=[msg_input, chatbot]
    )
    
    msg_input.submit(
        fn=send_message,
        inputs=[msg_input, api_key_input, file_upload, system_prompt_input, chatbot],
        outputs=[msg_input, chatbot]
    )
    
    clear_btn.click(
        fn=clear_chat,
        inputs=[chatbot],
        outputs=[chatbot]
    )
    
    return {
        "chatbot": chatbot,
        "msg_input": msg_input,
        "api_key_input": api_key_input,
        "file_upload": file_upload
    }

# VenusFactory API integration functions
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
            api_name="/handle_mutation_prediction_advance"
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
            api_name="/handle_mutation_prediction_advance"
        )
        
        # Parse the result - handle_mutation_prediction_advance returns multiple values
        return parse_zero_shot_prediction_result(result, api_key)
    except Exception as e:
        return f"Zero-shot sequence prediction error: {str(e)}"

def call_zero_shot_structure_prediction(structure_file: str, model_name: str = "SaProt", api_key: str = None) -> str:
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
            api_name="/handle_mutation_prediction_advance"
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
        ai_summary = generate_simple_ai_summary(df, api_key)
        
        # Return the AI summary
        return ai_summary
        
    except Exception as e:
        return f"Error parsing prediction result: {str(e)}"

def generate_simple_ai_summary(df, api_key: str = None) -> str:
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
        score_col = next((col for col in df.columns if isinstance(col, str) and 'score' in col.lower()), None)
        if not score_col:
            score_col = next((col for col in df.columns if isinstance(col, str) and any(word in col.lower() for word in ['prediction', 'value', 'rank', 'effect'])), None)
        
        if not score_col:
            return "❌ Could not identify score column for analysis."
        
        # Sort by score and get top mutations
        df_sorted = df.sort_values(score_col, ascending=False)
        top_mutations = df_sorted.head(10)
        
        # Create simple prompt for AI
        prompt = f"""
        Please analyze these protein mutation prediction results and provide a concise summary:

        Top 10 predicted mutations:
        {top_mutations.to_string(index=False)}

        Summary statistics:
        - Total mutations: {len(df)}
        - Best score: {df_sorted.iloc[0].get(score_col, 'N/A'):.4f}
        - Average score: {df[score_col].mean():.4f}

        Please provide a brief analysis of the key findings and suggest the most promising mutations for experimental validation.
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
            "Binding": ["MetalIonBinding"],
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
            enable_ai=False,
            ai_model="DeepSeek",
            user_api_key=api_key,
            model_name=model_name,
            datasets=datasets,
            api_name="/handle_protein_function_prediction_advance"
        )
        # Parse the result
        return parse_function_prediction_result(result, model_name, task, api_key)
    except Exception as e:
        return f"Protein function prediction error: {str(e)}"

def parse_function_prediction_result(result, model_name: str, task: str, api_key: str = None) -> str:
    try:
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
        
        # Generate AI summary with provided api_key
        ai_summary = generate_simple_ai_summary(df, api_key)
        
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
            "Binding": ["MetalIonBinding"],
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
            enable_ai=False,
            ai_model="DeepSeek",
            user_api_key=api_key,
            model_name=model_name,
            datasets=datasets,
            api_name="/handle_protein_function_prediction_advance"
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
        
        return f"Zero-shot Sequence-based Mutation Prediction Results:\n\n{result}"
    except Exception as e:
        return f"Sequence prediction error: {str(e)}"

def predict_zero_shot_structure(structure_file: str, model_name: str = "SaProt", api_key: str = None) -> str:
    """Predict mutations using structure-based zero-shot models"""
    try:
        if not structure_file or not os.path.exists(structure_file):
            return "Error: Please provide a valid structure file (PDB/CIF)"
        
        # Call VenusFactory zero-shot structure prediction API
        result = call_zero_shot_structure_prediction(structure_file, model_name)
        
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
        
        return f"Protein Function Prediction Results:\n\n{result}"
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

def detect_user_intent(message: str, files: List[str] = None) -> dict:
    """Detect user intent using AI-based analysis"""
    
    # Default intent
    intent = {
        'type': 'general',
        'sequence': '',
        'action': 'chat', 
        'model': 'ESM2-650M',
        'task': 'Solubility',
        'prediction_type': 'sequence'
    }
    
    # Extract sequence if present
    sequence = extract_sequence_from_message(message)
    if sequence:
        intent['sequence'] = sequence
    
    # Check uploaded files
    has_structure_file = False
    has_fasta_file = False
    if files:
        for file in files:
            if file and os.path.exists(file):
                if file.lower().endswith(('.pdb', '.cif')):
                    has_structure_file = True
                    intent['structure_file'] = file
                elif file.lower().endswith(('.fasta', '.fa')):
                    has_fasta_file = True
                    intent['fasta_file'] = file
    
    # Use AI to analyze intent if we have an API key
    try:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if api_key:
            intent = analyze_intent_with_ai(message, has_fasta_file, has_structure_file, intent)
        else:
            # If no API key, use fallback
            intent = fallback_intent_detection(message, has_fasta_file, has_structure_file, intent)
    except Exception as e:
        print(f"AI intent analysis failed, using fallback: {e}")
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
2. Zero-shot mutation prediction (structure-based): SaProt, ProtSSN, ESM-IF1, MIF-ST, ProSST-2048
3. Protein function prediction: Solubility, Localization, Binding, Stability, Sorting signal, Optimum temperature

TASK: Analyze the user's intent and return ONLY a JSON response. Be very precise about the action type.

ACTION RULES:
- "predict_zero_shot_sequence": When user wants mutation/mutant prediction with sequence data or FASTA files
- "predict_zero_shot_structure": When user wants mutation/mutant prediction with structure files (PDB/CIF) OR specifically mentions structure-based models
- "predict_function": When user wants to predict protein properties like solubility, localization, binding, stability, etc.
- "chat": For general questions, greetings, or unclear requests

MODEL RULES:
- For zero-shot sequence: ESM-1v, ESM2-650M, ESM-1b
- For zero-shot structure: SaProt, ProtSSN, ESM-IF1, MIF-ST, ProSST-2048  
- For function prediction: always use "ESM2-650M"

TASK RULES (only for function prediction):
- Solubility, Localization, Binding, Stability, "Sorting signal", "Optimum temperature"

Keywords to look for:
- Mutation/mutant/design → zero-shot prediction
- Solubility/localization/binding/stability → function prediction
- Structure-based models → zero-shot structure
- General questions → chat

Return JSON format:
{{
    "action": "chat|predict_zero_shot_sequence|predict_zero_shot_structure|predict_function",
    "model": "model_name",
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
            "temperature": 0.0,  # 降低温度确保一致性
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
                
                # 验证和修正AI的响应
                valid_actions = ["chat", "predict_zero_shot_sequence", "predict_zero_shot_structure", "predict_function"]
                if ai_intent.get('action') not in valid_actions:
                    print(f"Invalid action from AI: {ai_intent.get('action')}, using fallback")
                    return fallback_intent_detection(message, has_fasta_file, has_structure_file, default_intent)
                
                # Update intent based on AI analysis
                default_intent['action'] = ai_intent.get('action', 'chat')
                default_intent['model'] = ai_intent.get('model', 'ESM2-650M')
                
                # 任务名称处理
                if ai_intent.get('action') == 'predict_function':
                    default_intent['task'] = ai_intent.get('task', 'Solubility')
                    default_intent['model'] = 'ESM2-650M'  # 功能预测固定使用ESM2-650M
                else:
                    default_intent['task'] = None
                
                # Set type based on action
                if 'zero_shot' in ai_intent.get('action', ''):
                    default_intent['type'] = 'zero_shot_prediction'
                elif 'function' in ai_intent.get('action', ''):
                    default_intent['type'] = 'function_prediction'
                else:
                    default_intent['type'] = 'general'
                
                print(f"DEBUG: AI intent parsed successfully: {default_intent}")
                print(f"DEBUG: AI reasoning: {ai_intent.get('reasoning', 'No reasoning provided')}")
                return default_intent
                
            except json.JSONDecodeError as e:
                print(f"Failed to parse AI response: {ai_response}")
                print(f"JSON decode error: {e}")
                return fallback_intent_detection(message, has_fasta_file, has_structure_file, default_intent)
        else:
            print(f"AI API call failed: {response.status_code}")
            return fallback_intent_detection(message, has_fasta_file, has_structure_file, default_intent)
            
    except Exception as e:
        print(f"AI intent analysis error: {e}")
        return fallback_intent_detection(message, has_fasta_file, has_structure_file, default_intent)


def fallback_intent_detection(message: str, has_fasta_file: bool, has_structure_file: bool, intent: dict) -> dict:
    """Fallback intent detection using simple keyword matching"""
    message_lower = message.lower()
    
    print(f"DEBUG: Fallback intent detection for message: '{message_lower}'")
    print(f"DEBUG: Files - FASTA: {has_fasta_file}, Structure: {has_structure_file}")
    
    # 重新设计优先级逻辑
    
    # 1. 首先检查是否是突变/设计相关（最高优先级）
    mutation_keywords = ['mutation', 'mutant', 'mutate', '突变', '设计', 'design', 'zero-shot', 'zero shot']
    is_mutation_request = any(word in message_lower for word in mutation_keywords)
    
    # 2. 检查是否是功能预测相关
    function_keywords = {
        'solubility': ['solubility', '溶解度', 'soluble'],
        'localization': ['localization', '定位', 'location', 'subcellular'],
        'binding': ['binding', '结合', 'bind', 'interaction'],
        'stability': ['stability', '稳定性', 'stable', 'thermal'],
        'sorting signal': ['sorting', 'signal', '信号', '分选'],
        'optimum temperature': ['temperature', '温度', 'thermal', 'optimum']
    }
    
    detected_function = None
    for func_name, keywords in function_keywords.items():
        if any(keyword in message_lower for keyword in keywords):
            detected_function = func_name
            break
    
    # 3. 检查模型偏好
    sequence_models = {
        'ESM-1v': ['esm-1v', 'esm1v'],
        'ESM2-650M': ['esm2', 'esm-2', 'esm2-650m'],
        'ESM-1b': ['esm-1b', 'esm1b']
    }
    
    structure_models = {
        'SaProt': ['saprot'],
        'ProtSSN': ['protssn'],
        'ESM-IF1': ['esm-if1', 'esmif1'],
        'MIF-ST': ['mif-st', 'mifst'],
        'ProSST-2048': ['prosst', 'prosst-2048']
    }
    
    detected_seq_model = 'ESM2-650M'  # 默认
    detected_struct_model = 'SaProt'  # 默认
    
    for model_name, keywords in sequence_models.items():
        if any(keyword in message_lower for keyword in keywords):
            detected_seq_model = model_name
            break
    
    for model_name, keywords in structure_models.items():
        if any(keyword in message_lower for keyword in keywords):
            detected_struct_model = model_name
            break
    
    # 4. 决策逻辑
    if is_mutation_request:
        # 突变预测请求
        structure_indicators = ['structure', 'pdb', 'cif', '结构'] + list(structure_models.keys())
        wants_structure_based = (any(indicator.lower() in message_lower for indicator in structure_indicators) 
                               or has_structure_file)
        
        if wants_structure_based:
            intent['action'] = 'predict_zero_shot_structure'
            intent['type'] = 'zero_shot_prediction'
            intent['model'] = detected_struct_model
            print(f"DEBUG: Detected structure-based mutation prediction with model {detected_struct_model}")
        else:
            intent['action'] = 'predict_zero_shot_sequence'
            intent['type'] = 'zero_shot_prediction'
            intent['model'] = detected_seq_model
            print(f"DEBUG: Detected sequence-based mutation prediction with model {detected_seq_model}")
    
    elif detected_function:
        intent['action'] = 'predict_function'
        intent['type'] = 'function_prediction'
        intent['model'] = 'ESM2-650M'
        intent['task'] = detected_function.title()
        print(f"DEBUG: Detected function prediction for {detected_function}")
    
    else:
        intent['action'] = 'chat'
        intent['type'] = 'general'
        print(f"DEBUG: Detected general chat")
    
    return intent
