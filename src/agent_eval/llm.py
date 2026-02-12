import os
import asyncio
import aiohttp
import requests
from typing import Any, List, Optional, Dict
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks import CallbackManagerForLLMRun

class LLM_Model(BaseChatModel):
    api_key: Optional[str] = None
    base_url: str = "https://www.dmxapi.com/v1"
    model_name: str = "gpt-4o-mini" 
    temperature: float = 0.2
    max_tokens: int = 4096
    
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("OPENAI_API_KEY")

    def _generate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> ChatResult:
    
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


    async def _agenerate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> ChatResult:
        """Asynchronous generation for concurrent execution"""
        if not self.api_key:
            raise ValueError("API key is not configured.")

        message_dicts = []
        for msg in messages:
            role = "user"
            if isinstance(msg, HumanMessage): role = "user"
            elif isinstance(msg, AIMessage): role = "assistant"
            elif isinstance(msg, SystemMessage): role = "system"
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
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status != 200:
                        text = await response.text()
                        raise RuntimeError(f"Async API request failed: {response.status} - {text}")

                    result = await response.json()
                    choice = result['choices'][0]
                    message_data = choice['message']

                    ai_message = AIMessage(
                        content=message_data.get('content', ''),
                        additional_kwargs=message_data,
                    )
                    
                    generation = ChatGeneration(message=ai_message)
                    return ChatResult(generations=[generation])
        except Exception as e:
             raise RuntimeError(f"Async API exception: {str(e)}")

    @property
    def _llm_type(self) -> str:
        return "custom-llm-model"
