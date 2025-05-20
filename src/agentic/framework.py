"""
Agentic Framework for CTM-AbsoluteZero.
This module provides a unified framework for agentic AI with CTM-AbsoluteZero.
"""
import os
import sys
import logging
import time
import json
import asyncio
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Callable, Type

# Import integration modules
from .claude_integration import ClaudeAgentic, ClaudeConfig
from .deepseek_integration import DeepSeekAgentic, DeepSeekConfig

# Import CTM components
from ..utils.logging import get_logger
from ..utils.config import ConfigManager

# Setup logger
logger = get_logger("ctm-az.agentic.framework")

class AgentType(Enum):
    """Agent type enum."""
    CLAUDE = "claude"
    DEEPSEEK = "deepseek"
    CUSTOM = "custom"
    

class AgenController:
    """
    Controller for agentic AI with CTM-AbsoluteZero.
    This class provides a unified interface for working with different
    agentic AI implementations.
    """
    
    def __init__(
        self,
        agent_type: Union[str, AgentType] = AgentType.CLAUDE,
        config: Optional[Dict[str, Any]] = None,
        ctm_interface: Optional[Any] = None,
        reward_system: Optional[Any] = None,
    ):
        """
        Initialize the agent controller.
        
        Args:
            agent_type: Type of agent to use
            config: Configuration for the agent
            ctm_interface: Interface to CTM system
            reward_system: Reward system to use
        """
        self.config = config or {}
        self.ctm_interface = ctm_interface
        self.reward_system = reward_system
        
        # Convert string to enum if needed
        if isinstance(agent_type, str):
            try:
                self.agent_type = AgentType(agent_type.lower())
            except ValueError:
                logger.warning(f"Unknown agent type: {agent_type}, defaulting to CLAUDE")
                self.agent_type = AgentType.CLAUDE
        else:
            self.agent_type = agent_type
            
        # Initialize the agent
        self.agent = self._create_agent()
        logger.info(f"Initialized agent controller with agent type: {self.agent_type.value}")
    
    def _create_agent(self) -> Any:
        """Create the appropriate agent based on agent_type."""
        if self.agent_type == AgentType.CLAUDE:
            return self._create_claude_agent()
        elif self.agent_type == AgentType.DEEPSEEK:
            return self._create_deepseek_agent()
        elif self.agent_type == AgentType.CUSTOM:
            return self._create_custom_agent()
        else:
            raise ValueError(f"Unsupported agent type: {self.agent_type}")
    
    def _create_claude_agent(self) -> ClaudeAgentic:
        """Create a Claude agent."""
        # Create Claude-specific config
        claude_config = ClaudeConfig(
            api_key=self.config.get("api_key"),
            model=self.config.get("model", "claude-3-opus-20240229"),
            temperature=self.config.get("temperature", 0.7),
            max_tokens=self.config.get("max_tokens", 4096),
            top_p=self.config.get("top_p", 0.95),
            api_base=self.config.get("api_base", "https://api.anthropic.com/v1"),
            timeout=self.config.get("timeout", 60)
        )
        
        # Create and return the agent
        return ClaudeAgentic(
            config=claude_config,
            ctm_interface=self.ctm_interface,
            reward_system=self.reward_system,
            system_prompt=self.config.get("system_prompt")
        )
    
    def _create_deepseek_agent(self) -> DeepSeekAgentic:
        """Create a DeepSeek agent."""
        # Create DeepSeek-specific config
        deepseek_config = DeepSeekConfig(
            api_key=self.config.get("api_key"),
            model=self.config.get("model", "deepseek-chat-32k"),
            temperature=self.config.get("temperature", 0.7),
            max_tokens=self.config.get("max_tokens", 4096),
            top_p=self.config.get("top_p", 0.95),
            api_base=self.config.get("api_base", "https://api.deepseek.com/v1"),
            timeout=self.config.get("timeout", 60)
        )
        
        # Create and return the agent
        return DeepSeekAgentic(
            config=deepseek_config,
            ctm_interface=self.ctm_interface,
            reward_system=self.reward_system
        )
    
    def _create_custom_agent(self) -> Any:
        """Create a custom agent."""
        custom_agent_class = self.config.get("agent_class")
        if not custom_agent_class:
            raise ValueError("Custom agent_class must be provided for CUSTOM agent type")
            
        # Create and return the custom agent
        return custom_agent_class(
            config=self.config,
            ctm_interface=self.ctm_interface,
            reward_system=self.reward_system
        )
    
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
        logger.info(f"Running agentic cycle with agent type {self.agent_type.value}")
        return await self.agent.run_cycle(
            domain=domain,
            difficulty=difficulty,
            constraints=constraints
        )
    
    async def generate_task(
        self, 
        domain: str,
        difficulty: str = "medium",
        constraints: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a task using the agent.
        
        Args:
            domain: Task domain
            difficulty: Task difficulty level
            constraints: Optional constraints for the task
            context: Optional context for generation
            
        Returns:
            Generated task
        """
        logger.info(f"Generating task with agent type {self.agent_type.value}")
        return await self.agent.generate_task(
            domain=domain,
            difficulty=difficulty,
            constraints=constraints,
            context=context
        )
    
    async def analyze_solution(
        self,
        task: Dict[str, Any],
        solution: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze a task solution using the agent.
        
        Args:
            task: The original task
            solution: The solution to analyze
            metrics: Performance metrics for the solution
            
        Returns:
            Analysis of the solution
        """
        logger.info(f"Analyzing solution with agent type {self.agent_type.value}")
        return await self.agent.analyze_solution(
            task=task,
            solution=solution,
            metrics=metrics
        )
    
    async def clear_conversation(self) -> None:
        """Clear the current conversation."""
        if hasattr(self.agent, "clear_conversation"):
            await self.agent.clear_conversation()
    
    @property
    def task_history(self) -> List[Dict[str, Any]]:
        """Get task history."""
        return getattr(self.agent, "task_history", [])
    
    @property
    def solution_history(self) -> List[Dict[str, Any]]:
        """Get solution history."""
        return getattr(self.agent, "solution_history", [])


class BrainFramework:
    """
    Unified Brain Framework for CTM-AbsoluteZero with DFZ integration.
    This framework combines agentic AI capabilities with CTM-AbsoluteZero and DFZ.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        ctm_interface: Optional[Any] = None,
        reward_system: Optional[Any] = None,
        dfz_adapter: Optional[Any] = None
    ):
        """
        Initialize the Brain Framework.
        
        Args:
            config_path: Path to configuration file
            ctm_interface: Interface to CTM system
            reward_system: Reward system to use
            dfz_adapter: DFZ adapter for integration
        """
        # Load configuration
        self.config_manager = ConfigManager(config_path) if config_path else ConfigManager()
        self.config = self.config_manager.to_dict()
        
        # Set up components
        self.ctm_interface = ctm_interface
        self.reward_system = reward_system
        self.dfz_adapter = dfz_adapter
        
        # Initialize agents
        self.agent_controllers: Dict[str, AgenController] = {}
        self._init_agents()
        
        # Set default agent
        default_agent = self.config.get("default_agent", "claude")
        if default_agent in self.agent_controllers:
            self.default_agent = default_agent
        else:
            self.default_agent = next(iter(self.agent_controllers.keys()), None)
        
        logger.info(f"Brain Framework initialized with {len(self.agent_controllers)} agents")
        if self.default_agent:
            logger.info(f"Default agent: {self.default_agent}")
    
    def _init_agents(self) -> None:
        """Initialize agent controllers from configuration."""
        agents_config = self.config.get("agents", {})
        
        for agent_name, agent_config in agents_config.items():
            try:
                agent_type = agent_config.get("type", "claude")
                logger.info(f"Initializing agent '{agent_name}' with type '{agent_type}'")
                
                controller = AgenController(
                    agent_type=agent_type,
                    config=agent_config,
                    ctm_interface=self.ctm_interface,
                    reward_system=self.reward_system
                )
                
                self.agent_controllers[agent_name] = controller
                
            except Exception as e:
                logger.error(f"Failed to initialize agent '{agent_name}': {e}")
    
    def get_agent(self, agent_name: Optional[str] = None) -> Optional[AgenController]:
        """
        Get an agent controller by name.
        
        Args:
            agent_name: Name of the agent (if None, uses default)
            
        Returns:
            Agent controller or None if not found
        """
        if not agent_name:
            agent_name = self.default_agent
            
        return self.agent_controllers.get(agent_name)
    
    async def run_cycle(
        self,
        domain: str,
        difficulty: str = "medium",
        constraints: Optional[Dict[str, Any]] = None,
        agent_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run a complete agentic cycle (generate task, solve, analyze).
        
        Args:
            domain: Task domain
            difficulty: Task difficulty
            constraints: Optional constraints
            agent_name: Name of the agent to use (if None, uses default)
            
        Returns:
            Results of the cycle
        """
        agent = self.get_agent(agent_name)
        if not agent:
            logger.error(f"Agent not found: {agent_name or self.default_agent}")
            return {"error": f"Agent not found: {agent_name or self.default_agent}"}
            
        return await agent.run_cycle(
            domain=domain,
            difficulty=difficulty,
            constraints=constraints
        )
    
    async def generate_task(
        self, 
        domain: str,
        difficulty: str = "medium",
        constraints: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        agent_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a task using an agent.
        
        Args:
            domain: Task domain
            difficulty: Task difficulty level
            constraints: Optional constraints for the task
            context: Optional context for generation
            agent_name: Name of the agent to use (if None, uses default)
            
        Returns:
            Generated task
        """
        agent = self.get_agent(agent_name)
        if not agent:
            logger.error(f"Agent not found: {agent_name or self.default_agent}")
            return {"error": f"Agent not found: {agent_name or self.default_agent}"}
            
        return await agent.generate_task(
            domain=domain,
            difficulty=difficulty,
            constraints=constraints,
            context=context
        )
    
    async def analyze_solution(
        self,
        task: Dict[str, Any],
        solution: Dict[str, Any],
        metrics: Dict[str, Any],
        agent_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a task solution using an agent.
        
        Args:
            task: The original task
            solution: The solution to analyze
            metrics: Performance metrics for the solution
            agent_name: Name of the agent to use (if None, uses default)
            
        Returns:
            Analysis of the solution
        """
        agent = self.get_agent(agent_name)
        if not agent:
            logger.error(f"Agent not found: {agent_name or self.default_agent}")
            return {"error": f"Agent not found: {agent_name or self.default_agent}"}
            
        return await agent.analyze_solution(
            task=task,
            solution=solution,
            metrics=metrics
        )
    
    async def send_to_dfz(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send a message to DFZ.
        
        Args:
            message: Message to send
            context: Optional context for the message
            
        Returns:
            Response from DFZ
        """
        if not self.dfz_adapter:
            logger.error("DFZ adapter not initialized")
            return {"error": "DFZ adapter not initialized"}
            
        logger.info(f"Sending message to DFZ: {message[:50]}...")
        
        try:
            response = await self.dfz_adapter.send_message(message, context)
            return response
        except Exception as e:
            logger.error(f"Failed to send message to DFZ: {e}")
            return {"error": str(e)}
    
    async def generate_dfz_task(
        self,
        domain: str = "general",
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate tasks using DFZ.
        
        Args:
            domain: Task domain
            context: Optional context for task generation
            
        Returns:
            List of generated tasks
        """
        if not self.dfz_adapter:
            logger.error("DFZ adapter not initialized")
            return [{"error": "DFZ adapter not initialized"}]
            
        logger.info(f"Generating tasks from DFZ for domain: {domain}")
        
        try:
            tasks = await self.dfz_adapter.generate_task(domain, context)
            return tasks
        except Exception as e:
            logger.error(f"Failed to generate tasks from DFZ: {e}")
            return [{"error": str(e)}]
    
    async def execute_dfz_task(
        self,
        task: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a task using DFZ.
        
        Args:
            task: Task to execute
            context: Optional execution context
            
        Returns:
            Task execution results
        """
        if not self.dfz_adapter:
            logger.error("DFZ adapter not initialized")
            return {"error": "DFZ adapter not initialized"}
            
        logger.info(f"Executing task through DFZ: {task.get('id', 'unknown')}")
        
        try:
            result = await self.dfz_adapter.execute_task(task, context)
            return result
        except Exception as e:
            logger.error(f"Failed to execute task through DFZ: {e}")
            return {"error": str(e)}


# Usage example
async def main():
    """Example usage of Brain Framework."""
    # Create a framework with default configuration
    framework = BrainFramework()
    
    # Run a cycle with the default agent
    result = await framework.run_cycle(
        domain="quantum",
        difficulty="medium",
        constraints={"max_qubits": 8, "noise_level": 0.01}
    )
    
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())