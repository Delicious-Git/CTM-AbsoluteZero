"""
DFZ integration module for CTM-AbsoluteZero.
This module provides classes and functions to integrate CTM-AbsoluteZero with the DFZ
conversational intelligence system.
"""
import logging
import json
import os
import sys
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
import time
import numpy as np

# Add paths for both CTM-AbsoluteZero and DFZ
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'evolution')))

# Import CTM components
from src.ctm.interface import RealCTMInterface
from src.ctm_az_agent import AbsoluteZeroAgent
from src.rewards.composite import CompositeRewardSystem
from src.transfer.adapter import NeuralTransferAdapter
from src.transfer.phase import PhaseController
from src.utils.logging import get_logger, CTMLogger
from src.utils.config import ConfigManager

# Try to import DFZ components (may not be available in standalone mode)
try:
    from evolution.plugins import Plugin, ManagerExtension
    from evolution.utils import ConfigManager as DFZConfigManager
    DFZ_AVAILABLE = True
except ImportError:
    # Create stub classes for standalone operation
    class Plugin:
        def __init__(self, *args, **kwargs):
            pass
            
    class ManagerExtension:
        def __init__(self, *args, **kwargs):
            pass
            
    class DFZConfigManager:
        def __init__(self, *args, **kwargs):
            pass
            
    DFZ_AVAILABLE = False

logger = get_logger("ctm-az.integration.dfz")

class CTMAbsoluteZeroPlugin(Plugin, ManagerExtension):
    """
    Plugin for integrating CTM-AbsoluteZero with DFZ.
    This plugin allows DFZ to use CTM-AbsoluteZero for task generation and execution.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        standalone: bool = False
    ):
        """
        Initialize the CTM-AbsoluteZero plugin.
        
        Args:
            config_path: Path to configuration file
            standalone: Whether to operate in standalone mode (without DFZ)
        """
        self.standalone = standalone
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        
        # Initialize CTM components
        self.agent = None
        self.ctm_interface = None
        self.reward_system = None
        self.transfer_adapter = None
        self.phase_controller = None
        
        # Integration state
        self.conv_history = []
        self.task_registry = {}
        self.embedding_cache = {}
        
        # Performance tracking
        self.performance_metrics = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "task_durations": [],
        }
    
    @property
    def name(self) -> str:
        """Get the plugin name."""
        return "ctm_az"
    
    @property
    def version(self) -> str:
        """Get the plugin version."""
        return "1.0.0"
    
    @property
    def description(self) -> str:
        """Get the plugin description."""
        return "CTM-AbsoluteZero plugin for DFZ"
    
    async def initialize(self, manager=None) -> bool:
        """
        Initialize the plugin.
        
        Args:
            manager: DFZ manager (None in standalone mode)
            
        Returns:
            True if initialization was successful
        """
        try:
            logger.info("Initializing CTM-AbsoluteZero plugin")
            
            # Initialize phase controller
            self.phase_controller = PhaseController(
                phase_duration=self.config.get("phase_duration", 600),
                initial_phase=self.config.get("initial_phase", "exploration")
            )
            
            # Initialize task tracking components
            reward_config = self.config.get("rewards", {})
            
            # Initialize transfer adapter
            domains = self.config.get("domains", ["general"])
            self.transfer_adapter = NeuralTransferAdapter(domains)
            
            # Create composite reward system
            self.reward_system = self._setup_reward_system(reward_config)
            
            # Create CTM interface
            ctm_config = self.config.get("ctm", {})
            self.ctm_interface = RealCTMInterface(ctm_config)
            
            # Create agent
            agent_config = self.config.get("agent", {})
            self.agent = self._setup_agent(agent_config)
            
            logger.info("CTM-AbsoluteZero plugin initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize CTM-AbsoluteZero plugin: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Clean up resources when shutting down."""
        logger.info("Shutting down CTM-AbsoluteZero plugin")
        
        # Save any state if needed
        if self.agent:
            try:
                # Save agent state
                state_path = os.path.join(
                    self.config.get("state_dir", "./state"),
                    "agent_state.json"
                )
                os.makedirs(os.path.dirname(state_path), exist_ok=True)
                self.agent.save_state(state_path)
                logger.info(f"Agent state saved to {state_path}")
            except Exception as e:
                logger.error(f"Failed to save agent state: {e}")
    
    def register_hooks(self, manager) -> None:
        """
        Register hooks with the DFZ manager.
        
        Args:
            manager: DFZ manager
        """
        if self.standalone:
            logger.info("Running in standalone mode, no hooks registered")
            return
            
        logger.info("Registering hooks with DFZ manager")
        
        # Register hooks for task handling
        manager.register_hook(
            "pre_message_process",
            self.pre_message_process_hook
        )
        
        manager.register_hook(
            "post_message_process",
            self.post_message_process_hook
        )
        
        manager.register_hook(
            "task_generation",
            self.task_generation_hook
        )
        
        manager.register_hook(
            "intent_detection",
            self.intent_detection_hook
        )
    
    async def pre_message_process_hook(
        self,
        message: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Hook called before processing a message in DFZ.
        
        Args:
            message: User message
            context: Conversation context
            
        Returns:
            Modified message
        """
        logger.debug(f"Pre-message process hook: {message}")
        
        # Add to conversation history
        if "text" in message:
            self.conv_history.append({
                "role": "user",
                "content": message["text"],
                "timestamp": time.time()
            })
            
            # Check if this is a task-related message
            keywords = ["task", "solve", "problem", "challenge", "learn", "teach"]
            if any(kw in message["text"].lower() for kw in keywords):
                # Add task intent metadata
                message["detected_intents"] = message.get("detected_intents", []) + ["task_request"]
                
        return message
    
    async def post_message_process_hook(
        self,
        response: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Hook called after processing a message in DFZ.
        
        Args:
            response: System response
            context: Conversation context
            
        Returns:
            Modified response
        """
        logger.debug(f"Post-message process hook: {response}")
        
        # Add to conversation history
        if "text" in response:
            self.conv_history.append({
                "role": "assistant",
                "content": response["text"],
                "timestamp": time.time()
            })
            
        return response
    
    async def task_generation_hook(
        self,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Hook for generating tasks in DFZ.
        
        Args:
            context: Conversation context
            
        Returns:
            List of generated tasks
        """
        logger.info("Generating tasks")
        
        try:
            # Extract domain information from context
            domain = context.get("domain", "general")
            
            # Generate tasks using the agent
            tasks = self.agent.generate_tasks(
                domain=domain,
                count=3,
                conversation_history=self.conv_history[-10:] if self.conv_history else []
            )
            
            logger.info(f"Generated {len(tasks)} tasks")
            
            # Register tasks in the task registry
            for task in tasks:
                task_id = task.get("id", f"task_{len(self.task_registry)}")
                task["id"] = task_id
                self.task_registry[task_id] = task
                
            return tasks
        except Exception as e:
            logger.error(f"Task generation failed: {e}")
            return []
    
    async def intent_detection_hook(
        self,
        message: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Hook for detecting intents in DFZ.
        
        Args:
            message: User message
            context: Conversation context
            
        Returns:
            Intent detection results
        """
        if "text" not in message:
            return {"intents": []}
            
        text = message["text"]
        
        # Simple keyword-based intent detection
        intents = []
        
        task_keywords = ["task", "problem", "solve", "challenge"]
        if any(kw in text.lower() for kw in task_keywords):
            intents.append({"name": "task_request", "confidence": 0.8})
            
        learn_keywords = ["learn", "teach", "explain", "how to"]
        if any(kw in text.lower() for kw in learn_keywords):
            intents.append({"name": "learning_request", "confidence": 0.7})
        
        return {"intents": intents}
    
    def execute_task(
        self,
        task: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a task using CTM-AbsoluteZero.
        
        Args:
            task: Task to execute
            context: Execution context
            
        Returns:
            Task execution results
        """
        start_time = time.time()
        
        try:
            logger.info(f"Executing task: {task.get('id', 'unknown')}")
            
            task_config = {
                "domain": task.get("domain", "general"),
                "description": task.get("description", ""),
                "parameters": task.get("parameters", {}),
                "context": context or {}
            }
            
            # Execute task
            result = self.agent.solve_task(task_config)
            
            duration = time.time() - start_time
            
            # Update performance metrics
            self.performance_metrics["total_tasks"] += 1
            self.performance_metrics["successful_tasks"] += 1
            self.performance_metrics["task_durations"].append(duration)
            
            logger.info(f"Task executed successfully in {duration:.2f}s")
            
            return {
                "task_id": task.get("id", "unknown"),
                "status": "success",
                "result": result,
                "duration": duration
            }
        except Exception as e:
            duration = time.time() - start_time
            
            # Update performance metrics
            self.performance_metrics["total_tasks"] += 1
            self.performance_metrics["failed_tasks"] += 1
            self.performance_metrics["task_durations"].append(duration)
            
            logger.error(f"Task execution failed: {e}")
            
            return {
                "task_id": task.get("id", "unknown"),
                "status": "error",
                "error": str(e),
                "duration": duration
            }
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a task from the task registry.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task or None if not found
        """
        return self.task_registry.get(task_id)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Returns:
            Performance metrics
        """
        metrics = self.performance_metrics.copy()
        
        # Calculate additional metrics
        total_tasks = metrics["total_tasks"]
        if total_tasks > 0:
            metrics["success_rate"] = metrics["successful_tasks"] / total_tasks
            metrics["failure_rate"] = metrics["failed_tasks"] / total_tasks
            
        durations = metrics["task_durations"]
        if durations:
            metrics["avg_duration"] = sum(durations) / len(durations)
            metrics["min_duration"] = min(durations)
            metrics["max_duration"] = max(durations)
            
        return metrics
    
    def _setup_reward_system(
        self,
        reward_config: Dict[str, Any]
    ) -> CompositeRewardSystem:
        """
        Set up the reward system.
        
        Args:
            reward_config: Reward system configuration
            
        Returns:
            Configured reward system
        """
        from src.rewards.novelty import SemanticNoveltyTracker
        from src.rewards.progress import SkillPyramid
        
        novelty_tracker = SemanticNoveltyTracker(
            embedding_dim=reward_config.get("embedding_dim", 768),
            novelty_threshold=reward_config.get("novelty_threshold", 0.2)
        )
        
        skill_pyramid = SkillPyramid(
            domains=self.config.get("domains", ["general"]),
            levels=reward_config.get("skill_levels", 5)
        )
        
        return CompositeRewardSystem(
            novelty_tracker=novelty_tracker,
            skill_pyramid=skill_pyramid,
            phase_controller=self.phase_controller,
            hyperparams=reward_config.get("hyperparams")
        )
    
    def _setup_agent(self, agent_config: Dict[str, Any]) -> AbsoluteZeroAgent:
        """
        Set up the AbsoluteZero agent.
        
        Args:
            agent_config: Agent configuration
            
        Returns:
            Configured agent
        """
        # Get model paths
        proposer_model_path = agent_config.get("proposer_model_path", "")
        solver_model_path = agent_config.get("solver_model_path", "")
        
        # Create agent
        agent = AbsoluteZeroAgent(
            proposer_model_path=proposer_model_path,
            proposer_tokenizer_path=agent_config.get("proposer_tokenizer_path", proposer_model_path),
            solver_model_path=solver_model_path,
            solver_tokenizer_path=agent_config.get("solver_tokenizer_path", solver_model_path),
            reward_system=self.reward_system,
            transfer_adapter=self.transfer_adapter,
            ctm_interface=self.ctm_interface,
            config=agent_config
        )
        
        return agent


class DFZAdapter:
    """
    Adapter for connecting CTM-AbsoluteZero to DFZ.
    This adapter provides methods to interact with DFZ from CTM-AbsoluteZero.
    """
    
    def __init__(
        self,
        dfz_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the DFZ adapter.
        
        Args:
            dfz_path: Path to DFZ installation
            config: Adapter configuration
        """
        self.dfz_path = dfz_path
        self.config = config or {}
        self.logger = get_logger("ctm-az.integration.dfz_adapter")
        
        # Check if DFZ is available
        self.dfz_available = False
        if dfz_path and os.path.exists(dfz_path):
            sys.path.append(os.path.abspath(dfz_path))
            
            try:
                self.logger.info("Attempting to import DFZ components")
                # Try to import DFZ components
                from evolution.manager import SystemEvolutionManager
                self.dfz_available = True
                self.manager = SystemEvolutionManager()
                self.logger.info("DFZ components imported successfully")
            except ImportError as e:
                self.logger.warning(f"Failed to import DFZ components: {e}")
                self.dfz_available = False
        else:
            self.logger.warning(f"DFZ path not found: {dfz_path}")
            
        # Create plugin instance
        self.plugin = CTMAbsoluteZeroPlugin(
            config_path=self.config.get("config_path"),
            standalone=not self.dfz_available
        )
    
    async def initialize(self) -> bool:
        """
        Initialize the adapter.
        
        Returns:
            True if initialization was successful
        """
        # Initialize the plugin
        success = await self.plugin.initialize()
        
        if self.dfz_available:
            # Register the plugin with DFZ
            self.logger.info("Registering plugin with DFZ")
            # TODO: Implement proper DFZ plugin registration when API is available
            
        return success
    
    async def send_message(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send a message to DFZ.
        
        Args:
            message: Message to send
            context: Message context
            
        Returns:
            Response from DFZ
        """
        if not self.dfz_available:
            self.logger.warning("DFZ not available, message not sent")
            return {"error": "DFZ not available"}
            
        try:
            self.logger.info(f"Sending message to DFZ: {message}")
            
            # Create message object
            message_obj = {
                "text": message,
                "timestamp": time.time(),
                "sender": "ctm-az",
            }
            
            if context:
                message_obj["context"] = context
                
            # Process message through DFZ
            response = await self.manager.process_input(
                message_obj,
                user_id="ctm-az-agent",
                channel="api"
            )
            
            self.logger.info("Message processed by DFZ")
            return response
        except Exception as e:
            self.logger.error(f"Failed to send message to DFZ: {e}")
            return {"error": str(e)}
    
    async def generate_task(
        self,
        domain: str = "general",
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate tasks using DFZ.
        
        Args:
            domain: Task domain
            context: Task generation context
            
        Returns:
            List of generated tasks
        """
        if not self.dfz_available:
            self.logger.warning("DFZ not available, falling back to plugin for task generation")
            ctx = context or {}
            ctx["domain"] = domain
            return await self.plugin.task_generation_hook(ctx)
            
        try:
            self.logger.info(f"Generating tasks through DFZ for domain: {domain}")
            
            # Create context
            task_context = context or {}
            task_context["domain"] = domain
            
            # Call DFZ to generate tasks
            tasks = await self.manager.generate_tasks(task_context)
            
            self.logger.info(f"Generated {len(tasks)} tasks through DFZ")
            return tasks
        except Exception as e:
            self.logger.error(f"Failed to generate tasks through DFZ: {e}")
            return []
    
    async def execute_task(
        self,
        task: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a task using the plugin.
        
        Args:
            task: Task to execute
            context: Execution context
            
        Returns:
            Task execution results
        """
        return self.plugin.execute_task(task, context)
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get conversation history.
        
        Returns:
            Conversation history
        """
        return self.plugin.conv_history
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a task from the task registry.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task or None if not found
        """
        return self.plugin.get_task(task_id)


def create_dfz_plugin(config_path: Optional[str] = None) -> CTMAbsoluteZeroPlugin:
    """
    Create a CTM-AbsoluteZero plugin for DFZ.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        CTM-AbsoluteZero plugin
    """
    return CTMAbsoluteZeroPlugin(config_path=config_path)