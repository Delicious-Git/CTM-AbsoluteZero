"""
DeepSeek Brain Integration for CTM-AbsoluteZero.
This module provides integration with DeepSeek Brain models as a cost-effective
alternative to Anthropic Claude.
"""
import os
import sys
import logging
import time
import json
import asyncio
import requests
from typing import Dict, List, Any, Optional, Union, Callable, Tuple

from .token_optimizer import DeepSeekOptimizer, TokenAnalyzer
from src.utils.logging import get_logger

# Setup logger
logger = get_logger("ctm-az.agentic.deepseek")

class DeepSeekConfig:
    """Configuration for DeepSeek API integration."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-chat-32k",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 0.95,
        api_base: str = "https://api.deepseek.com/v1",
        timeout: int = 60,
        token_optimization: bool = True,  # Enable token optimization by default
        optimization_level: str = "balanced",  # balanced, minimal, or aggressive
        token_log_dir: Optional[str] = None  # Directory to store token usage logs
    ):
        """
        Initialize DeepSeek configuration.
        
        Args:
            api_key: DeepSeek API key (if None, will try to use environment variable)
            model: Model name to use
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            api_base: Base URL for API
            timeout: Timeout for API calls in seconds
            token_optimization: Whether to enable token optimization
            optimization_level: Token optimization aggressiveness
            token_log_dir: Directory to store token usage logs
        """
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            logger.warning("No DeepSeek API key provided. Please set DEEPSEEK_API_KEY environment variable")
            
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.api_base = api_base
        self.timeout = timeout
        self.token_optimization = token_optimization
        self.optimization_level = optimization_level
        self.token_log_dir = token_log_dir
        
        if token_log_dir:
            os.makedirs(token_log_dir, exist_ok=True)


class DeepSeekClient:
    """Client for DeepSeek API."""
    
    def __init__(
        self, 
        config: Optional[DeepSeekConfig] = None, 
        token_analyzer: Optional[TokenAnalyzer] = None
    ):
        """
        Initialize DeepSeek client.
        
        Args:
            config: DeepSeek configuration
            token_analyzer: Token analyzer for tracking token usage
        """
        self.config = config or DeepSeekConfig()
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        })
        
        # Initialize token analyzer
        self.token_analyzer = token_analyzer or TokenAnalyzer(log_dir=self.config.token_log_dir)
        
        # Initialize token optimizer if enabled
        self.token_optimizer = DeepSeekOptimizer() if self.config.token_optimization else None
        
        # Track usage
        self.usage = {
            "total_tokens": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_cost": 0.0,
            "requests": 0
        }
    
    async def chat_complete(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stream: bool = False,
        optimize_tokens: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Generate a chat completion.
        
        Args:
            messages: List of message objects with role and content
            temperature: Temperature for generation (overrides config)
            max_tokens: Maximum tokens to generate (overrides config)
            top_p: Top-p sampling parameter (overrides config)
            stream: Whether to stream the response
            optimize_tokens: Whether to optimize token usage (overrides config)
            
        Returns:
            API response
        """
        # Determine if we should optimize tokens
        should_optimize = optimize_tokens if optimize_tokens is not None else self.config.token_optimization
        
        # Apply token optimization if enabled
        if should_optimize and self.token_optimizer:
            optimized_messages = []
            for message in messages:
                if message["role"] == "user" or message["role"] == "system":
                    # Only optimize user and system messages
                    optimized_content, _ = self.token_optimizer.optimize_prompt(
                        message["content"], 
                        max_tokens=max_tokens or self.config.max_tokens
                    )
                    optimized_messages.append({
                        "role": message["role"],
                        "content": optimized_content
                    })
                else:
                    optimized_messages.append(message)
            
            # Use optimized messages for the request
            messages_to_use = optimized_messages
        else:
            # Use original messages
            messages_to_use = messages
        
        url = f"{self.config.api_base}/chat/completions"
        
        # Create a copy of the config parameters to potentially optimize
        params = {
            "model": self.config.model,
            "messages": messages_to_use,
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            "top_p": top_p or self.config.top_p,
            "stream": stream
        }
        
        # Optimize parameters if enabled
        if should_optimize and self.token_optimizer:
            params = self.token_optimizer.optimize_system_config(params)
        
        logger.debug(f"Sending request to DeepSeek API: {url}")
        
        # Estimate prompt tokens
        prompt_tokens_estimate = self._estimate_tokens(messages_to_use)
        
        start_time = time.time()
        
        try:
            # Run in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.session.post(
                    url, 
                    json=params,
                    timeout=self.config.timeout
                )
            )
            
            # Calculate response time
            response_time = time.time() - start_time
            
            if response.status_code != 200:
                logger.error(f"DeepSeek API error: {response.status_code} - {response.text}")
                return {"error": response.text}
            
            if stream:
                return self._handle_stream(response)
            else:
                # Parse JSON response
                response_data = response.json()
                
                # Extract completion text
                completion = response_data["choices"][0]["message"]["content"]
                
                # Track token usage
                if "usage" in response_data:
                    usage = response_data["usage"]
                    self.usage["total_tokens"] += usage.get("total_tokens", 0)
                    self.usage["total_prompt_tokens"] += usage.get("prompt_tokens", 0)
                    self.usage["total_completion_tokens"] += usage.get("completion_tokens", 0)
                    
                    # Calculate cost (very approximate)
                    cost_per_token = 0.0001 / 1000  # $0.0001 per 1K tokens
                    request_cost = usage.get("total_tokens", 0) * cost_per_token
                    self.usage["total_cost"] += request_cost
                    
                    # Log token usage
                    original_prompt = "\n".join(m["content"] for m in messages if m["role"] in ["user", "system"])
                    optimized_prompt = "\n".join(m["content"] for m in messages_to_use if m["role"] in ["user", "system"])
                    
                    # Log to token analyzer
                    self.token_analyzer.log_request(
                        model=self.config.model,
                        prompt=optimized_prompt,
                        response=completion,
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0),
                        metadata={
                            "temperature": params["temperature"],
                            "max_tokens": params["max_tokens"],
                            "optimization": should_optimize,
                            "response_time": response_time,
                            "original_prompt_size": len(original_prompt),
                            "optimized_prompt_size": len(optimized_prompt),
                            "optimization_ratio": len(optimized_prompt) / len(original_prompt) if len(original_prompt) > 0 else 1.0
                        }
                    )
                
                # Increment request counter
                self.usage["requests"] += 1
                
                # Add token optimization info to response
                response_data["token_optimization"] = {
                    "enabled": should_optimize,
                    "messages_optimized": should_optimize,
                    "original_prompt_tokens": prompt_tokens_estimate,
                    "optimized_prompt_tokens": usage.get("prompt_tokens", prompt_tokens_estimate) if should_optimize else prompt_tokens_estimate
                }
                
                return response_data
                
        except Exception as e:
            logger.error(f"DeepSeek API request failed: {e}")
            return {"error": str(e)}
    
    def _handle_stream(self, response):
        """Handle streaming response."""
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    yield data
    
    async def embeddings(self, texts: List[str]) -> Dict[str, Any]:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            API response with embeddings
        """
        url = f"{self.config.api_base}/embeddings"
        
        payload = {
            "model": "deepseek-embeddings",
            "input": texts
        }
        
        logger.debug(f"Sending embeddings request to DeepSeek API: {url}")
        
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
                logger.error(f"DeepSeek API error: {response.status_code} - {response.text}")
                return {"error": response.text}
            
            return response.json()
                
        except Exception as e:
            logger.error(f"DeepSeek API request failed: {e}")
            return {"error": str(e)}
    
    def _estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Estimate token count for messages.
        
        Args:
            messages: List of messages
            
        Returns:
            Estimated token count
        """
        # Simple estimation based on character count
        # This is a very rough approximation
        total_chars = sum(len(message["content"]) for message in messages)
        
        # Approximate tokens (roughly 4 chars per token)
        return total_chars // 4
    
    def get_token_usage(self) -> Dict[str, Any]:
        """
        Get token usage statistics.
        
        Returns:
            Token usage statistics
        """
        usage_stats = self.usage.copy()
        
        # Add token analyzer statistics
        analyzer_stats = self.token_analyzer.get_token_usage_summary()
        
        usage_stats.update({
            "avg_tokens_per_request": analyzer_stats.get("avg_tokens_per_request", 0),
            "token_efficiency": analyzer_stats.get("efficiency_score", 0),
            "cost_per_request": usage_stats["total_cost"] / usage_stats["requests"] if usage_stats["requests"] > 0 else 0
        })
        
        return usage_stats


class DeepSeekBrainAdapter:
    """
    Adapter to use DeepSeek Brain models within the CTM-AbsoluteZero framework.
    This adapter allows using DeepSeek models as a cost-effective alternative
    to Anthropic Claude.
    """
    
    def __init__(
        self,
        config: Optional[DeepSeekConfig] = None,
        system_prompt: Optional[str] = None,
        token_analyzer: Optional[TokenAnalyzer] = None
    ):
        """
        Initialize DeepSeek Brain adapter.
        
        Args:
            config: DeepSeek configuration
            system_prompt: System prompt to use for all requests
            token_analyzer: Token analyzer for tracking usage
        """
        # Create config if not provided
        if not config:
            config = DeepSeekConfig()
        
        # Initialize token analyzer
        self.token_analyzer = token_analyzer or TokenAnalyzer(log_dir=config.token_log_dir)
        
        # Initialize client
        self.client = DeepSeekClient(config, token_analyzer=self.token_analyzer)
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.current_conversation: List[Dict[str, str]] = []
        
        # Initialize conversation with system prompt
        if self.system_prompt:
            self.current_conversation.append({
                "role": "system",
                "content": self.system_prompt
            })
    
    def _default_system_prompt(self) -> str:
        """Get default system prompt."""
        return """You are an AI assistant integrated with the CTM-AbsoluteZero framework, 
        a self-learning system that combines Continuous Thought Machine and Absolute Zero Reasoner 
        paradigms. You help generate tasks, analyze solutions, and provide insights 
        across various domains including quantum computing, maze solving, and sorting algorithms.
        When asked to generate a task, make sure it's appropriately challenging based on the 
        current skill level and includes all necessary parameters for execution.
        """
    
    async def generate_task(
        self, 
        domain: str,
        difficulty: str = "medium",
        constraints: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a task using DeepSeek Brain.
        
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
        
        # Add message to conversation
        conversation = self.current_conversation.copy()
        conversation.append({
            "role": "user",
            "content": prompt
        })
        
        # Generate response with token optimization
        response = await self.client.chat_complete(
            messages=conversation,
            temperature=0.7,
            max_tokens=1024,
            optimize_tokens=True
        )
        
        if "error" in response:
            logger.error(f"Task generation failed: {response['error']}")
            return {"error": response["error"]}
        
        # Extract the response content
        assistant_response = response["choices"][0]["message"]["content"]
        self.current_conversation.append({
            "role": "user",
            "content": prompt
        })
        self.current_conversation.append({
            "role": "assistant",
            "content": assistant_response
        })
        
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
                
            # Add token optimization info to task
            task["token_optimization"] = response.get("token_optimization", {})
            
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
        metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze a task solution using DeepSeek Brain.
        
        Args:
            task: The original task
            solution: The solution to analyze
            metrics: Performance metrics for the solution
            
        Returns:
            Analysis of the solution
        """
        # Use empty metrics if not provided
        metrics = metrics or {}
        
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
        
        # Add message to conversation
        conversation = self.current_conversation.copy()
        conversation.append({
            "role": "user",
            "content": prompt
        })
        
        # Generate response with token optimization
        response = await self.client.chat_complete(
            messages=conversation,
            temperature=0.4,  # Lower for more precise analysis
            max_tokens=1024,
            optimize_tokens=True
        )
        
        if "error" in response:
            logger.error(f"Solution analysis failed: {response['error']}")
            return {"error": response["error"]}
        
        # Extract the response content
        assistant_response = response["choices"][0]["message"]["content"]
        self.current_conversation.append({
            "role": "user",
            "content": prompt
        })
        self.current_conversation.append({
            "role": "assistant",
            "content": assistant_response
        })
        
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
            
            # Add token optimization info
            analysis["token_optimization"] = response.get("token_optimization", {})
                    
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to parse solution analysis: {e}")
            return {
                "error": f"Failed to parse solution analysis: {e}",
                "raw_response": assistant_response
            }
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        response = await self.client.embeddings(texts)
        
        if "error" in response:
            logger.error(f"Embedding generation failed: {response['error']}")
            return []
        
        return [item["embedding"] for item in response["data"]]
    
    async def clear_conversation(self) -> None:
        """Clear the current conversation."""
        self.current_conversation = []
        
        # Add system prompt back
        if self.system_prompt:
            self.current_conversation.append({
                "role": "system",
                "content": self.system_prompt
            })
    
    def get_token_efficiency_report(self) -> Dict[str, Any]:
        """
        Get token efficiency report.
        
        Returns:
            Token efficiency report
        """
        return self.client.get_token_usage()


class DeepSeekAgentic:
    """
    Agentic wrapper for DeepSeek Brain integration.
    This class implements a simple agentic workflow using DeepSeek Brain models.
    """
    
    def __init__(
        self,
        config: Optional[DeepSeekConfig] = None,
        ctm_interface: Any = None,
        reward_system: Any = None
    ):
        """
        Initialize DeepSeek agentic wrapper.
        
        Args:
            config: DeepSeek configuration
            ctm_interface: Interface to CTM system
            reward_system: Reward system to use
        """
        # Create config with token optimization enabled
        if not config:
            config = DeepSeekConfig(token_optimization=True)
        elif not hasattr(config, 'token_optimization'):
            config.token_optimization = True
            
        # Create token log directory
        log_dir = config.token_log_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "logs", "tokens"
        )
        os.makedirs(log_dir, exist_ok=True)
        config.token_log_dir = log_dir
        
        # Initialize token analyzer
        self.token_analyzer = TokenAnalyzer(log_dir=log_dir)
        
        # Initialize adapter
        self.adapter = DeepSeekBrainAdapter(
            config=config, 
            token_analyzer=self.token_analyzer
        )
        
        self.ctm_interface = ctm_interface
        self.reward_system = reward_system
        
        # Task history
        self.task_history: List[Dict[str, Any]] = []
        self.solution_history: List[Dict[str, Any]] = []
        
        logger.info("DeepSeek Agentic wrapper initialized with token optimization")
    
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
        logger.info(f"Starting agentic cycle for domain: {domain}")
        
        # Start timing
        cycle_start = time.time()
        
        # Step 1: Generate task
        task = await self.adapter.generate_task(
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
        analysis = await self.adapter.analyze_solution(
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
        
        # Calculate total duration
        cycle_duration = time.time() - cycle_start
        logger.info(f"Cycle completed in {cycle_duration:.2f}s with reward: {reward}")
        
        # Get token efficiency stats
        token_stats = self.adapter.get_token_efficiency_report()
        
        # Return complete results
        return {
            "task": task,
            "solution": solution,
            "analysis": analysis,
            "reward": reward,
            "duration": cycle_duration,
            "token_stats": token_stats
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
    
    def get_token_efficiency_report(self) -> Dict[str, Any]:
        """
        Get token efficiency report.
        
        Returns:
            Token efficiency report with cost comparison
        """
        usage_stats = self.adapter.get_token_efficiency_report()
        
        # Calculate cost comparison with Claude
        token_count = usage_stats.get("total_tokens", 0)
        deepseek_cost = token_count * (0.0001 / 1000)  # $0.0001 per 1K tokens
        claude_cost = token_count * (0.008 / 1000)     # $0.008 per 1K tokens
        
        return {
            "usage": usage_stats,
            "cost_comparison": {
                "deepseek_cost": deepseek_cost,
                "claude_cost": claude_cost,
                "cost_difference": claude_cost - deepseek_cost,
                "cost_ratio": claude_cost / deepseek_cost if deepseek_cost > 0 else 0,
                "cost_savings_percentage": (1 - (deepseek_cost / claude_cost)) * 100 if claude_cost > 0 else 0
            }
        }
    
    async def benchmark_token_efficiency(
        self, 
        prompts: List[str], 
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Benchmark token efficiency for a list of prompts.
        
        Args:
            prompts: List of prompts to benchmark
            system_message: Optional system message
            
        Returns:
            Benchmark results
        """
        results = {
            "prompts_tested": len(prompts),
            "with_optimization": {
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost": 0.0
            },
            "without_optimization": {
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost": 0.0
            },
            "comparison": {
                "token_savings": 0,
                "cost_savings": 0.0,
                "savings_percentage": 0.0
            },
            "prompt_results": []
        }
        
        # Create conversation with system message
        base_messages = []
        if system_message:
            base_messages.append({
                "role": "system",
                "content": system_message
            })
        
        # Test each prompt with and without optimization
        for i, prompt in enumerate(prompts):
            prompt_result = {
                "prompt_index": i,
                "prompt_length": len(prompt),
                "with_optimization": {},
                "without_optimization": {}
            }
            
            # Create messages
            messages = base_messages.copy()
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            # Test with optimization
            optimized_response = await self.adapter.client.chat_complete(
                messages=messages,
                optimize_tokens=True
            )
            
            # Test without optimization
            unoptimized_response = await self.adapter.client.chat_complete(
                messages=messages,
                optimize_tokens=False
            )
            
            # Record results
            if "usage" in optimized_response:
                usage = optimized_response["usage"]
                prompt_result["with_optimization"] = {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "cost": usage.get("total_tokens", 0) * (0.0001 / 1000)
                }
                
                results["with_optimization"]["prompt_tokens"] += usage.get("prompt_tokens", 0)
                results["with_optimization"]["completion_tokens"] += usage.get("completion_tokens", 0)
                results["with_optimization"]["total_tokens"] += usage.get("total_tokens", 0)
                results["with_optimization"]["cost"] += usage.get("total_tokens", 0) * (0.0001 / 1000)
            
            if "usage" in unoptimized_response:
                usage = unoptimized_response["usage"]
                prompt_result["without_optimization"] = {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "cost": usage.get("total_tokens", 0) * (0.0001 / 1000)
                }
                
                results["without_optimization"]["prompt_tokens"] += usage.get("prompt_tokens", 0)
                results["without_optimization"]["completion_tokens"] += usage.get("completion_tokens", 0)
                results["without_optimization"]["total_tokens"] += usage.get("total_tokens", 0)
                results["without_optimization"]["cost"] += usage.get("total_tokens", 0) * (0.0001 / 1000)
            
            # Calculate savings for this prompt
            if "total_tokens" in prompt_result["with_optimization"] and "total_tokens" in prompt_result["without_optimization"]:
                opt_tokens = prompt_result["with_optimization"]["total_tokens"]
                unopt_tokens = prompt_result["without_optimization"]["total_tokens"]
                
                if unopt_tokens > 0:
                    token_savings = unopt_tokens - opt_tokens
                    savings_percentage = (token_savings / unopt_tokens) * 100
                    
                    prompt_result["token_savings"] = token_savings
                    prompt_result["savings_percentage"] = savings_percentage
                    
                    # Add to total savings calculations
                    results["comparison"]["token_savings"] += token_savings
            
            # Add to result list
            results["prompt_results"].append(prompt_result)
        
        # Calculate overall comparison
        if results["without_optimization"]["total_tokens"] > 0:
            savings_percentage = (results["comparison"]["token_savings"] / results["without_optimization"]["total_tokens"]) * 100
            results["comparison"]["savings_percentage"] = savings_percentage
            
            # Calculate cost savings
            cost_optimized = results["with_optimization"]["cost"]
            cost_unoptimized = results["without_optimization"]["cost"]
            results["comparison"]["cost_savings"] = cost_unoptimized - cost_optimized
        
        return results


# Usage example
async def main():
    """Example usage of DeepSeek Brain integration with token optimization."""
    config = DeepSeekConfig(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        model="deepseek-chat-32k",
        temperature=0.7,
        token_optimization=True,
        optimization_level="balanced",
        token_log_dir="./logs/tokens"
    )
    
    agentic = DeepSeekAgentic(config)
    
    # Run a cycle
    result = await agentic.run_cycle(
        domain="quantum",
        difficulty="medium",
        constraints={"max_qubits": 8, "noise_level": 0.01}
    )
    
    # Get token efficiency report
    efficiency_report = agentic.get_token_efficiency_report()
    
    print("Token Efficiency Report:")
    print(f"Total tokens: {efficiency_report['usage']['total_tokens']}")
    print(f"DeepSeek cost: ${efficiency_report['cost_comparison']['deepseek_cost']:.6f}")
    print(f"Claude cost: ${efficiency_report['cost_comparison']['claude_cost']:.6f}")
    print(f"Cost savings: ${efficiency_report['cost_comparison']['cost_difference']:.6f}")
    print(f"DeepSeek is {efficiency_report['cost_comparison']['cost_ratio']:.1f}x cheaper than Claude")
    
    print("\nTask result:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())