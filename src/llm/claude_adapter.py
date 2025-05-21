"""
Claude LLM adapter for CTM-AbsoluteZero.
This module provides an interface to interact with Claude AI models.
"""

import os
import time
import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple, Union
import anthropic

class ClaudeAdapter:
    """
    Adapter for interacting with Claude models.
    """
    
    def __init__(
        self,
        model: str = "claude-3-sonnet-20240229",
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        debug: bool = False
    ):
        """
        Initialize the Claude adapter.
        
        Args:
            model: Claude model to use
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            max_tokens: Maximum number of tokens in the response
            temperature: Temperature parameter for generation
            top_p: Top-p parameter for generation
            debug: Whether to print debug information
        """
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.debug = debug
        
        if not self.api_key:
            raise ValueError("API key is required. Set the ANTHROPIC_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize client
        self.client = anthropic.Anthropic(api_key=self.api_key)
    
    async def generate(self, prompt: str) -> str:
        """
        Generate a response from Claude.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated response text
        """
        try:
            if self.debug:
                print(f"Generating response from Claude model {self.model}...")
                print(f"Prompt: {prompt[:100]}...")
            
            # Call Claude API
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract response text
            response = message.content[0].text
            
            if self.debug:
                print(f"Response: {response[:100]}...")
            
            return response
        except Exception as e:
            if self.debug:
                print(f"Error generating response: {e}")
            raise
    
    async def generate_with_metrics(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a response from Claude with usage metrics.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Tuple of (generated response text, metrics dictionary)
        """
        try:
            if self.debug:
                print(f"Generating response with metrics from Claude model {self.model}...")
            
            # Call Claude API
            start_time = time.time()
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            end_time = time.time()
            
            # Extract response text
            response = message.content[0].text
            
            # Extract usage metrics
            metrics = {
                "prompt_tokens": message.usage.input_tokens,
                "completion_tokens": message.usage.output_tokens,
                "total_tokens": message.usage.input_tokens + message.usage.output_tokens,
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
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the text.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        # Use the client's tokenizer
        return self.client.count_tokens(text)
    
    async def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        # Currently, Anthropic doesn't provide a direct API for model info
        # Return basic information that we know
        return {
            "model": self.model,
            "provider": "anthropic",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p
        }