"""
Claude Integration for CTM-AbsoluteZero.
This module provides integration with Anthropic Claude models.
"""
import os
import sys
import logging
import time
import json
import asyncio
import requests
from typing import Dict, List, Any, Optional, Union, Callable, Generator

# Setup logger
logger = logging.getLogger("ctm-az.agentic.claude")

class ClaudeConfig:
    """Configuration for Claude API integration."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-opus-20240229",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 0.95,
        api_base: str = "https://api.anthropic.com/v1",
        timeout: int = 60,
    ):
        """
        Initialize Claude configuration.
        
        Args:
            api_key: Claude API key (if None, will try to use environment variable)
            model: Model name to use
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            api_base: Base URL for API
            timeout: Timeout for API calls in seconds
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.warning("No Claude API key provided. Please set ANTHROPIC_API_KEY environment variable")
            
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.api_base = api_base
        self.timeout = timeout


class ClaudeClient:
    """Client for Claude API."""
    
    def __init__(self, config: Optional[ClaudeConfig] = None):
        """
        Initialize Claude client.
        
        Args:
            config: Claude configuration
        """
        self.config = config or ClaudeConfig()
        self.session = requests.Session()
        self.session.headers.update({
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        })
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stream: bool = False,
    ) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """
        Generate a completion.
        
        Args:
            messages: List of message objects with role and content
            system: System prompt
            temperature: Temperature for generation (overrides config)
            max_tokens: Maximum tokens to generate (overrides config)
            top_p: Top-p sampling parameter (overrides config)
            stream: Whether to stream the response
            
        Returns:
            API response
        """
        url = f"{self.config.api_base}/messages"
        
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature or self.config.temperature,
            "top_p": top_p or self.config.top_p,
            "stream": stream
        }
        
        if system:
            payload["system"] = system
        
        logger.debug(f"Sending request to Claude API: {url}")
        
        try:
            # Run in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.session.post(
                    url, 
                    json=payload,
                    timeout=self.config.timeout
                )
            )
            
            if response.status_code != 200:
                logger.error(f"Claude API error: {response.status_code} - {response.text}")
                return {"error": response.text}
            
            if stream:
                return self._handle_stream(response)
            else:
                return response.json()
                
        except Exception as e:
            logger.error(f"Claude API request failed: {e}")
            return {"error": str(e)}
    
    def _handle_stream(self, response) -> Generator[Dict[str, Any], None, None]:
        """Handle streaming response."""
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    if line[6:].strip() == "[DONE]":
                        break
                    data = json.loads(line[6:])
                    yield data


class ClaudeAgentic:
    """
    Agentic wrapper for Claude integration.
    This class implements a simple agentic workflow using Claude models.
    """
    
    def __init__(
        self,
        config: Optional[ClaudeConfig] = None,
        ctm_interface: Any = None,
        reward_system: Any = None,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize Claude agentic wrapper.
        
        Args:
            config: Claude configuration
            ctm_interface: Interface to CTM system
            reward_system: Reward system to use
            system_prompt: System prompt to use
        """
        self.client = ClaudeClient(config)
        self.ctm_interface = ctm_interface
        self.reward_system = reward_system
        self.system_prompt = system_prompt or self._default_system_prompt()
        
        # Conversation history
        self.conversation: List[Dict[str, str]] = []
        
        # Task history
        self.task_history: List[Dict[str, Any]] = []
        self.solution_history: List[Dict[str, Any]] = []
        
        logger.info("Claude Agentic wrapper initialized")
    
    def _default_system_prompt(self) -> str:
        """Get default system prompt."""
        return """You are an AI assistant integrated with the CTM-AbsoluteZero framework, 
        a self-learning system that combines Continuous Thought Machine and Absolute Zero Reasoner 
        paradigms. You help generate tasks, analyze solutions, and provide insights 
        across various domains including quantum computing, maze solving, and sorting algorithms.
        
        When asked to generate a task, make sure it's appropriately challenging based on the 
        current skill level and includes all necessary parameters for execution.
        
        Always provide your responses in the requested JSON format when asked to.
        """
    
    async def generate_task(
        self, 
        domain: str,
        difficulty: str = "medium",
        constraints: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a task using Claude.
        
        Args:
            domain: Task domain (e.g., "quantum", "maze", "sorting")
            difficulty: Task difficulty level
            constraints: Optional constraints for the task
            context: Optional context for generation
            
        Returns:
            Generated task
        """
        # Create the prompt
        prompt = f"""Generate a {difficulty} difficulty task in the {domain} domain.
        
        The task should be clear, specific, and include all necessary parameters for execution.
        
        {f'Constraints: {json.dumps(constraints)}' if constraints else ''}
        
        Return ONLY a valid JSON object with the following structure:
        {{
            "domain": "{domain}",
            "description": "Clear task description",
            "parameters": {{
                // Domain-specific parameters
            }}
        }}
        """
        
        # Add user message to conversation
        user_message = {"role": "user", "content": prompt}
        self.conversation.append(user_message)
        
        # Generate response
        response = await self.client.generate(
            messages=self.conversation,
            system=self.system_prompt
        )
        
        if "error" in response:
            logger.error(f"Task generation failed: {response['error']}")
            return {"error": response["error"]}
        
        # Extract the response content
        assistant_message = {
            "role": "assistant",
            "content": response["content"][0]["text"]
        }
        self.conversation.append(assistant_message)
        
        assistant_response = assistant_message["content"]
        
        # Parse the JSON task
        try:
            # Extract JSON if it's wrapped in markdown code blocks
            if "```json" in assistant_response:
                json_str = assistant_response.split("```json")[1].split("```")[0].strip()
            elif "```" in assistant_response:
                json_str = assistant_response.split("```")[1].strip()
            else:
                json_str = assistant_response
                
            task = json.loads(json_str)
            
            # Ensure required fields
            if "domain" not in task or "description" not in task or "parameters" not in task:
                raise ValueError("Generated task is missing required fields")
                
            return task
            
        except Exception as e:
            logger.error(f"Failed to parse generated task: {e}")
            return {
                "error": f"Failed to parse generated task: {e}",
                "raw_response": assistant_response
            }
    
    async def analyze_solution(
        self,
        task: Dict[str, Any],
        solution: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze a task solution using Claude.
        
        Args:
            task: The original task
            solution: The solution to analyze
            metrics: Performance metrics for the solution
            
        Returns:
            Analysis of the solution
        """
        # Create the prompt
        prompt = f"""Analyze the following solution to a {task['domain']} task:
        
        TASK:
        {json.dumps(task, indent=2)}
        
        SOLUTION:
        {json.dumps(solution, indent=2)}
        
        METRICS:
        {json.dumps(metrics, indent=2)}
        
        Provide an analysis of the solution's effectiveness, efficiency, and correctness.
        Suggest potential improvements to the solution.
        
        Return your analysis as a valid JSON object with the following structure:
        {{
            "effectiveness": 0.0-1.0,
            "efficiency": 0.0-1.0,
            "correctness": 0.0-1.0,
            "strengths": ["strength1", "strength2", ...],
            "weaknesses": ["weakness1", "weakness2", ...],
            "improvement_suggestions": ["suggestion1", "suggestion2", ...],
            "overall_rating": 0.0-1.0
        }}
        """
        
        # Add user message to conversation
        user_message = {"role": "user", "content": prompt}
        self.conversation.append(user_message)
        
        # Generate response
        response = await self.client.generate(
            messages=self.conversation,
            system=self.system_prompt
        )
        
        if "error" in response:
            logger.error(f"Solution analysis failed: {response['error']}")
            return {"error": response["error"]}
        
        # Extract the response content
        assistant_message = {
            "role": "assistant",
            "content": response["content"][0]["text"]
        }
        self.conversation.append(assistant_message)
        
        assistant_response = assistant_message["content"]
        
        # Parse the JSON analysis
        try:
            # Extract JSON if it's wrapped in markdown code blocks
            if "```json" in assistant_response:
                json_str = assistant_response.split("```json")[1].split("```")[0].strip()
            elif "```" in assistant_response:
                json_str = assistant_response.split("```")[1].strip()
            else:
                json_str = assistant_response
                
            analysis = json.loads(json_str)
            
            # Ensure required fields
            required_fields = ["effectiveness", "efficiency", "correctness", 
                               "strengths", "weaknesses", "improvement_suggestions", 
                               "overall_rating"]
            
            for field in required_fields:
                if field not in analysis:
                    raise ValueError(f"Generated analysis is missing required field: {field}")
                    
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to parse solution analysis: {e}")
            return {
                "error": f"Failed to parse solution analysis: {e}",
                "raw_response": assistant_response
            }
    
    async def run_cycle(
        self,
        domain: str,
        difficulty: str = "medium",
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run a complete agentic cycle (generate task, solve, analyze).
        
        Args:
            domain: Task domain
            difficulty: Task difficulty
            constraints: Optional constraints
            
        Returns:
            Results of the cycle
        """
        logger.info(f"Starting Claude agentic cycle for domain: {domain}")
        
        # Step 1: Generate task
        task = await self.generate_task(
            domain=domain,
            difficulty=difficulty,
            constraints=constraints
        )
        
        if "error" in task:
            logger.error(f"Task generation failed: {task['error']}")
            return {"error": task["error"], "stage": "task_generation"}
        
        logger.info(f"Generated task: {task['description']}")
        self.task_history.append(task)
        
        # Step 2: Execute task
        if self.ctm_interface:
            logger.info("Executing task with CTM")
            try:
                solution = await self.ctm_interface.execute_task(task)
            except Exception as e:
                logger.error(f"Task execution failed: {e}")
                return {"error": str(e), "stage": "task_execution", "task": task}
        else:
            logger.warning("No CTM interface provided, generating mock solution")
            solution = self._generate_mock_solution(task)
        
        logger.info(f"Task executed with result status: {solution.get('status', 'unknown')}")
        self.solution_history.append(solution)
        
        # Step 3: Analyze solution
        analysis = await self.analyze_solution(
            task=task,
            solution=solution,
            metrics=solution.get("metrics", {})
        )
        
        if "error" in analysis:
            logger.error(f"Solution analysis failed: {analysis['error']}")
            return {
                "error": analysis["error"], 
                "stage": "solution_analysis",
                "task": task,
                "solution": solution
            }
        
        logger.info(f"Solution analysis complete with overall rating: {analysis.get('overall_rating', 0)}")
        
        # Step 4: Calculate reward
        reward = 0.0
        if self.reward_system:
            logger.info("Calculating reward")
            reward_context = {
                "task": task,
                "solution": solution,
                "analysis": analysis
            }
            try:
                reward = self.reward_system.calculate_reward(reward_context)
            except Exception as e:
                logger.error(f"Reward calculation failed: {e}")
                reward = analysis.get("overall_rating", 0) * 10  # Fallback
        else:
            logger.warning("No reward system provided, using analysis rating as reward")
            reward = analysis.get("overall_rating", 0) * 10
        
        logger.info(f"Cycle completed with reward: {reward}")
        
        # Return complete results
        return {
            "task": task,
            "solution": solution,
            "analysis": analysis,
            "reward": reward
        }
    
    def _generate_mock_solution(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a mock solution for testing."""
        domain = task.get("domain", "unknown")
        
        if domain == "quantum":
            return {
                "status": "success",
                "result": {
                    "circuit_depth": task.get("parameters", {}).get("circuit_depth", 5),
                    "measurement_results": {"0": 0.7, "1": 0.3},
                    "fidelity": 0.85
                },
                "metrics": {
                    "execution_time": 0.5,
                    "success_probability": 0.85,
                    "circuit_complexity": task.get("parameters", {}).get("circuit_depth", 5) * 0.2
                }
            }
        elif domain == "maze":
            size = task.get("parameters", {}).get("size", [10, 10])
            return {
                "status": "success",
                "result": {
                    "path_length": sum(size) * 0.8,
                    "path": [[0, 0], [size[0]-1, size[1]-1]],
                    "optimal": True
                },
                "metrics": {
                    "execution_time": size[0] * size[1] * 0.001,
                    "efficiency": 0.9,
                    "path_optimality": 0.95
                }
            }
        elif domain == "sorting":
            array_size = task.get("parameters", {}).get("array_size", 100)
            return {
                "status": "success",
                "result": {
                    "sorted": True,
                    "comparisons": array_size * (array_size - 1) // 2,
                    "swaps": array_size * 0.8
                },
                "metrics": {
                    "execution_time": array_size * 0.001,
                    "efficiency": 0.85,
                    "stability": 1.0
                }
            }
        else:
            return {
                "status": "success",
                "result": {"completed": True},
                "metrics": {
                    "execution_time": 1.0,
                    "efficiency": 0.8
                }
            }
    
    async def clear_conversation(self) -> None:
        """Clear the current conversation."""
        self.conversation = []


# Usage example
async def main():
    """Example usage of Claude integration."""
    config = ClaudeConfig(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        model="claude-3-opus-20240229",
        temperature=0.7
    )
    
    agentic = ClaudeAgentic(config)
    
    # Run a cycle
    result = await agentic.run_cycle(
        domain="quantum",
        difficulty="medium",
        constraints={"max_qubits": 8, "noise_level": 0.01}
    )
    
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())