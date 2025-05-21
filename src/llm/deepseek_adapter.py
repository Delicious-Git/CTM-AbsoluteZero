"""
DeepSeek LLM adapter for CTM-AbsoluteZero.
This module provides an interface to interact with DeepSeek AI models.
"""

import os
import time
import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple, Union
import requests
from transformers import AutoTokenizer

class DeepSeekAdapter:
    """
    Adapter for interacting with DeepSeek models.
    """
    
    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        debug: bool = False
    ):
        """
        Initialize the DeepSeek adapter.
        
        Args:
            model: DeepSeek model to use
            api_key: DeepSeek API key (defaults to DEEPSEEK_API_KEY env var)
            api_base: DeepSeek API base URL (defaults to DEEPSEEK_API_BASE env var)
            max_tokens: Maximum number of tokens in the response
            temperature: Temperature parameter for generation
            top_p: Top-p parameter for generation
            debug: Whether to print debug information
        """
        self.model = model
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.api_base = api_base or os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.debug = debug
        
        if not self.api_key:
            raise ValueError("API key is required. Set the DEEPSEEK_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize tokenizer for token counting
        self.tokenizer = None
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-33b-instruct")
        except Exception as e:
            if self.debug:
                print(f"Warning: Failed to load tokenizer: {e}")
                print("Token counting will be approximate.")
        
        # Track conversation state
        self.conversation_id = None
        self.conversation_history = []
    
    async def generate(self, prompt: str) -> str:
        """
        Generate a response from DeepSeek.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated response text
        """
        try:
            if self.debug:
                print(f"Generating response from DeepSeek model {self.model}...")
                print(f"Prompt: {prompt[:100]}...")
            
            # Call DeepSeek API
            response = await self._call_api(prompt)
            
            if self.debug:
                print(f"Response: {response[:100]}...")
            
            return response
        except Exception as e:
            if self.debug:
                print(f"Error generating response: {e}")
            raise
    
    async def generate_with_metrics(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a response from DeepSeek with usage metrics.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Tuple of (generated response text, metrics dictionary)
        """
        try:
            if self.debug:
                print(f"Generating response with metrics from DeepSeek model {self.model}...")
            
            # Count prompt tokens
            prompt_tokens = self.count_tokens(prompt)
            
            # Call DeepSeek API
            start_time = time.time()
            response = await self._call_api(prompt)
            end_time = time.time()
            
            # Count response tokens
            completion_tokens = self.count_tokens(response)
            
            # Calculate metrics
            metrics = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "execution_time": end_time - start_time,
                "model": self.model
            }
            
            if self.debug:
                print(f"Metrics: {metrics}")
            
            return response, metrics
        except Exception as e:
            if self.debug:
                print(f"Error generating response with metrics: {e}")
            raise
    
    async def _call_api(self, prompt: str) -> str:
        """
        Call the DeepSeek API.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated response text
        """
        # Format the messages based on the conversation history
        if not self.conversation_history:
            messages = [{"role": "user", "content": prompt}]
        else:
            # Append the new message to the conversation history
            messages = self.conversation_history + [{"role": "user", "content": prompt}]
        
        # Prepare the API request
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p
        }
        
        # Add conversation ID if available
        if self.conversation_id:
            payload["conversation_id"] = self.conversation_id
        
        # Convert the request to a coroutine for asyncio
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.post(url, headers=headers, json=payload)
        )
        
        # Process the response
        if response.status_code != 200:
            error_message = f"API request failed with status code {response.status_code}: {response.text}"
            if self.debug:
                print(error_message)
            raise Exception(error_message)
        
        response_data = response.json()
        
        # Extract conversation ID if available
        if "conversation_id" in response_data:
            self.conversation_id = response_data["conversation_id"]
        
        # Extract the response text
        if "choices" in response_data and len(response_data["choices"]) > 0:
            message = response_data["choices"][0]["message"]
            content = message.get("content", "")
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": content})
            
            # Limit conversation history length to prevent token explosion
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            return content
        else:
            raise Exception("No response content found in API response")
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the text.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        if self.tokenizer:
            # Use the actual tokenizer
            return len(self.tokenizer.encode(text))
        else:
            # Approximate token count (about 0.75 tokens per word for English text)
            words = text.split()
            return int(len(words) * 1.25)  # Slightly higher ratio to be conservative
    
    def reset_conversation(self):
        """Reset the conversation history and ID."""
        self.conversation_id = None
        self.conversation_history = []
    
    async def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model": self.model,
            "provider": "deepseek",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "conversation_id": self.conversation_id,
            "conversation_history_length": len(self.conversation_history)
        }