"""AI-related helper functions for API calls and prompt generation."""

import os
import requests
from typing import Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import gradio as gr

from .constants import AI_MODELS


@dataclass
class AIConfig:
    """Configuration for AI API calls."""
    api_key: str
    ai_model_name: str
    api_base: str
    model: str


def get_api_key(ai_provider: str, user_input_key: str = "") -> Optional[str]:
    """Get API key based on provider and user input."""
    model_config = AI_MODELS.get(ai_provider, {})
    env_key = model_config.get("env_key")
    
    if env_key:
        env_api_key = os.getenv(env_key)
        if env_api_key and env_api_key.strip():
            return env_api_key.strip()
        if user_input_key and user_input_key.strip():
            return user_input_key.strip()
        return None
    
    if user_input_key and user_input_key.strip():
        return user_input_key.strip()
    return None


def call_ai_api(config: AIConfig, prompt: str) -> str:
    """Make API call to AI service."""
    if config.ai_model_name == "ChatGPT":
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert protein scientist. Provide clear, structured, and insightful analysis based on the data provided. Do not ask interactive questions."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 2000
        }
        endpoint = f"{config.api_base}/chat/completions"
        
    elif config.ai_model_name == "Gemini":
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "contents": [{
                "parts": [{
                    "text": f"You are an expert protein scientist. {prompt}"
                }]
            }],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 2000
            }
        }
        endpoint = f"{config.api_base}/models/{config.model}:generateContent?key={config.api_key}"
        
    else:
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert protein scientist. Provide clear, structured, and insightful analysis based on the data provided. Do not ask interactive questions."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 2000
        }
        endpoint = f"{config.api_base}/chat/completions"
    
    try:
        response = requests.post(
            endpoint,
            headers=headers,
            json=data,
            timeout=60
        )
        response.raise_for_status()
        response_json = response.json()
        
        if config.ai_model_name == "Gemini":
            if "candidates" in response_json and len(response_json["candidates"]) > 0:
                return response_json["candidates"][0]["content"]["parts"][0]["text"]
            else:
                return "❌ Gemini API returned empty response"
        else:
            return response_json["choices"][0]["message"]["content"]
            
    except requests.exceptions.RequestException as e:
        return f"❌ Network error: {str(e)}"
    except KeyError as e:
        return f"❌ API response format error: {str(e)}"
    except Exception as e:
        return f"❌ API call failed: {str(e)}"


def check_ai_config_status(ai_provider: str, user_api_key: str) -> Tuple[bool, str]:
    """Check AI configuration status and return validity and message."""
    model_config = AI_MODELS.get(ai_provider, {})
    env_key = model_config.get("env_key")
    
    if env_key:
        env_api_key = os.getenv(env_key)
        if env_api_key and env_api_key.strip():
            return True, "✓ Using the Provided API Key"
        elif user_api_key and user_api_key.strip():
            return True, "✓ The server will not save your API Key"
        else:
            return False, "⚠ No API Key found in .env file"
    else:
        if user_api_key and user_api_key.strip():
            return True, "✓ Manual API Key provided"
        else:
            return False, "⚠ Manual API Key required"


def on_ai_model_change(ai_provider: str) -> Tuple:
    """Handle AI model selection change."""
    model_config = AI_MODELS.get(ai_provider, {})
    env_key = model_config.get("env_key")
    
    if env_key:
        env_api_key = os.getenv(env_key)
        if env_api_key and env_api_key.strip():
            status_msg = "✓ Using API Key from .env file"
            show_input = False
            placeholder = ""
        else:
            status_msg = "⚠ No API Key available. Please enter manually below:"
            show_input = True
            placeholder = "Enter your DeepSeek API Key"
    else:
        status_msg = f"⚠ Manual API Key required for {ai_provider}"
        show_input = True
        if ai_provider == "ChatGPT":
            placeholder = "Enter your OpenAI API Key (sk-...)"
        elif ai_provider == "Gemini":
            placeholder = "Enter your Google AI API Key"
        else:
            placeholder = f"Enter your {ai_provider} API Key"
    
    return (
        gr.update(visible=show_input, placeholder=placeholder, value=""),
        gr.update(value=status_msg)
    )


def generate_expert_analysis_prompt_residue(results_df: pd.DataFrame, task: str) -> str:
    """Generate prompt for LLM to analyze residue-level prediction results."""
    prompt = f"""
        You are a senior protein biochemist and structural biologist with deep expertise in enzyme mechanisms and protein engineering. 
        A colleague has just brought you the following prediction results identifying key functional residues, specifically for the task '{task}'.
        Please analyze these residue-specific results from a practical, lab-focused perspective:

        {results_df.to_string(index=False)}

        Provide a concise, practical analysis in a single paragraph, focusing ONLY on:
        1. The predicted functional role of the individual high-scoring residues (e.g., activity site, binding site, conserved site, and motif).
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
    return prompt


def generate_expert_analysis_prompt(results_df: pd.DataFrame, task: str) -> str:
    """Generate prompt for function prediction analysis."""
    protein_count = len(results_df)

    prompt = f"""
        You are a senior protein biochemist with extensive laboratory experience. A colleague has just shown you protein function prediction results for the task '{task}'. 
        Please analyze these results from a practical biologist's perspective:
        {results_df.to_string(index=False)}
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
    return prompt


def format_expert_response(ai_response: str) -> str:
    """Format AI response as HTML with expert avatar and speech bubble."""
    escaped_response = ai_response.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')
    
    html = f"""
    <div style="min-height: 150px; padding: 10px; display: flex; align-items: flex-end; font-family: Arial, sans-serif;">
        <div style="margin-right: 15px; text-align: center;">
            <div style="width: 60px; height: 60px; background-color: #4A90E2; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 24px; color: white; border: 3px solid #357ABD;">
                👨‍🔬
            </div>
            <div style="font-size: 12px; color: #666; margin-top: 5px; font-weight: bold;">
                Expert
            </div>
        </div>
        <div style="flex: 1; position: relative;">
            <div style="background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 15px; padding: 15px; position: relative; max-width: 90%; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="position: absolute; left: -10px; bottom: 20px; width: 0; height: 0; border-top: 10px solid transparent; border-bottom: 10px solid transparent; border-right: 10px solid #f8f9fa;"></div>
                <div style="position: absolute; left: -11px; bottom: 20px; width: 0; height: 0; border-top: 10px solid transparent; border-bottom: 10px solid transparent; border-right: 10px solid #dee2e6;"></div>
                <div style="font-size: 15px; line-height: 1.4; color: #333;">
                    {escaped_response}
                </div>
            </div>
        </div>
    </div>
    """
    return html


def generate_mutation_ai_prompt(results_df: pd.DataFrame, model_name: str, function_selection: str) -> str:
    """Generate AI prompt for mutation analysis."""
    if 'Mutant' not in results_df.columns:
        return "Error: 'Mutant' column not found."
    
    score_col = next(
        (col for col in results_df.columns if 'Prediction Rank' in col.lower()), 
        results_df.columns[1] if len(results_df.columns) > 1 else None
    )
    
    if not score_col:
        return "Error: Score column not found."

    num_rows = len(results_df)
    top_count = max(20, int(num_rows * 0.05)) if num_rows >= 5 else num_rows
    top_mutations_str = results_df.head(top_count)[['Mutant', score_col]].to_string(index=False)
    
    return f"""
        Please act as an expert protein engineer and analyze the following mutation prediction results generated by the '{model_name}' model for the function '{function_selection}'.
        A deep mutational scan was performed. The results are sorted from most beneficial to least beneficial based on the '{score_col}'. Below are the top 5% of mutations.

        ### Top 5% Predicted Mutations (Potentially Most Beneficial):
        ```
        {top_mutations_str}
        ```

        ### Your Analysis Task:
        Based on this data, provide a structured scientific analysis report that includes the following sections:
        1. Executive Summary: Briefly summarize the key findings. Are there clear hotspot regions?
        2. Analysis of Beneficial Mutations: Discuss the top mutations and their potential biochemical impact on '{function_selection}'.
        3. Analysis of Detrimental Mutations: What do the most harmful mutations suggest about critical residues?
        4. Recommendations for Experimentation: Suggest 3-5 specific mutations for validation, with justifications.
        5. Do not output formatted content, just one paragraph is sufficient
        Provide a concise, clear, and insightful report in a professional scientific tone, summarize the above content into 1-2 paragraphs and output unformatted content.
        """


def generate_ai_summary_prompt(results_df: pd.DataFrame, task: str, model: str) -> str:
    """Generate AI prompt for function prediction summary."""
    prompt = f"""
        You are a senior protein biochemist with extensive laboratory experience. A colleague has just shown you protein function prediction results for the task '{task}' on model '{model}'. 
        Please analyze these results from a practical biologist's perspective:
        {results_df.to_string(index=False)}
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
    return prompt

