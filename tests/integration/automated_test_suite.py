#!/usr/bin/env python3
"""
Automated test suite for CTM-AbsoluteZero.

This script runs a comprehensive set of tests to ensure the CTM-AbsoluteZero
framework is working correctly and can be used for monitoring during standby.
"""
import os
import sys
import argparse
import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.utils.logging import get_logger, configure_logging
from src.utils.config import ConfigManager
from src.ctm.interface import RealCTMInterface
from src.ctm_az_agent import AbsoluteZeroAgent
from src.rewards.composite import CompositeRewardSystem
from src.rewards.novelty import SemanticNoveltyTracker
from src.rewards.progress import SkillPyramid
from src.transfer.adapter import NeuralTransferAdapter
from src.transfer.phase import PhaseController
from src.router.universal_router import UniversalRouter, Task, SolverInterface
from src.agentic.framework import BrainFramework
from src.integration.dfz import DFZAdapter

# Setup logger
logger = get_logger("ctm-az.test_suite")

class TestSuite:
    """Automated test suite for CTM-AbsoluteZero."""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        output_dir: str = "./test_results",
        dfz_path: Optional[str] = None,
        skip_slow_tests: bool = False
    ):
        """
        Initialize the test suite.
        
        Args:
            config_path: Path to configuration file
            output_dir: Directory to save results
            dfz_path: Path to DFZ installation
            skip_slow_tests: Whether to skip slow tests
        """
        self.config_path = config_path
        self.output_dir = output_dir
        self.dfz_path = dfz_path
        self.skip_slow_tests = skip_slow_tests
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load configuration
        self.config_manager = ConfigManager(config_path) if config_path else ConfigManager()
        self.config = self.config_manager.to_dict()
        
        # Create test context
        self.context = {
            "start_time": time.time(),
            "results": {},
            "errors": [],
            "warnings": [],
            "components": {},
            "success_rate": 0.0
        }
        
        # Initialize components
        self.ctm_interface = None
        self.router = None
        self.brain_framework = None
        self.dfz_adapter = None
        
        logger.info("Test suite initialized")
    
    async def run(self) -> Dict[str, Any]:
        """
        Run the test suite.
        
        Returns:
            Test results
        """
        logger.info("Starting test suite")
        
        # Test CTM Interface
        await self._test_ctm_interface()
        
        # Test Universal Router
        await self._test_universal_router()
        
        # Test Agentic Brain
        await self._test_agentic_brain()
        
        # Test DFZ Integration
        if self.dfz_path:
            await self._test_dfz_integration()
        else:
            logger.warning("Skipping DFZ integration tests (dfz_path not provided)")
            self.context["warnings"].append("DFZ integration tests skipped (dfz_path not provided)")
        
        # Calculate success rate
        total_tests = 0
        successful_tests = 0
        
        for component, component_results in self.context["results"].items():
            for test_name, test_result in component_results.items():
                total_tests += 1
                if test_result.get("status") == "pass":
                    successful_tests += 1
        
        if total_tests > 0:
            self.context["success_rate"] = successful_tests / total_tests
        
        # Calculate duration
        self.context["duration"] = time.time() - self.context["start_time"]
        
        # Save results
        self._save_results()
        
        logger.info(f"Test suite completed in {self.context['duration']:.2f}s")
        logger.info(f"Success rate: {self.context['success_rate']:.1%}")
        
        return self.context
    
    async def _test_ctm_interface(self) -> None:
        """Test the CTM interface."""
        logger.info("Testing CTM interface")
        
        # Initialize results
        self.context["results"]["ctm_interface"] = {}
        
        # Create CTM interface
        try:
            self.ctm_interface = RealCTMInterface({
                "components": ["maze_solver", "image_classifier", "quantum_sim", "sorter"],
                "metrics_enabled": True
            })
            
            self.context["components"]["ctm_interface"] = self.ctm_interface
            
            # Test basic connectivity
            result = await self._test_ctm_connectivity()
            self.context["results"]["ctm_interface"]["connectivity"] = result
            
            # Test maze solver
            result = await self._test_ctm_maze_solver()
            self.context["results"]["ctm_interface"]["maze_solver"] = result
            
            # Test quantum simulator
            if not self.skip_slow_tests:
                result = await self._test_ctm_quantum_sim()
                self.context["results"]["ctm_interface"]["quantum_sim"] = result
            
            # Test sorter
            result = await self._test_ctm_sorter()
            self.context["results"]["ctm_interface"]["sorter"] = result
            
        except Exception as e:
            logger.error(f"Failed to initialize CTM interface: {e}")
            self.context["errors"].append(f"CTM interface initialization failed: {e}")
            
            # Create failure results for all tests
            for test in ["connectivity", "maze_solver", "quantum_sim", "sorter"]:
                self.context["results"]["ctm_interface"][test] = {
                    "status": "fail",
                    "error": str(e),
                    "duration": 0.0
                }
    
    async def _test_ctm_connectivity(self) -> Dict[str, Any]:
        """
        Test CTM interface connectivity.
        
        Returns:
            Test result
        """
        logger.info("Testing CTM interface connectivity")
        
        start_time = time.time()
        
        try:
            # Check components
            components = self.ctm_interface.list_components()
            
            if not components:
                return {
                    "status": "fail",
                    "error": "No components found",
                    "duration": time.time() - start_time
                }
            
            # Check health
            health = self.ctm_interface.health_check()
            
            if not all(status == "ok" for _, status in health.items()):
                return {
                    "status": "fail",
                    "error": f"Component health check failed: {health}",
                    "duration": time.time() - start_time
                }
            
            return {
                "status": "pass",
                "components": components,
                "health": health,
                "duration": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"CTM interface connectivity test failed: {e}")
            
            return {
                "status": "fail",
                "error": str(e),
                "duration": time.time() - start_time
            }
    
    async def _test_ctm_maze_solver(self) -> Dict[str, Any]:
        """
        Test CTM maze solver.
        
        Returns:
            Test result
        """
        logger.info("Testing CTM maze solver")
        
        start_time = time.time()
        
        try:
            # Create maze task
            task = {
                "domain": "maze",
                "description": "Solve a simple maze",
                "parameters": {
                    "size": [5, 5],
                    "complexity": 0.3
                }
            }
            
            # Execute task
            result = await self.ctm_interface.execute_task(task)
            
            # Check result
            if result.get("status") != "success":
                return {
                    "status": "fail",
                    "error": f"Maze solver execution failed: {result.get('error', 'Unknown error')}",
                    "duration": time.time() - start_time
                }
            
            return {
                "status": "pass",
                "result": result,
                "duration": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"CTM maze solver test failed: {e}")
            
            return {
                "status": "fail",
                "error": str(e),
                "duration": time.time() - start_time
            }
    
    async def _test_ctm_quantum_sim(self) -> Dict[str, Any]:
        """
        Test CTM quantum simulator.
        
        Returns:
            Test result
        """
        logger.info("Testing CTM quantum simulator")
        
        start_time = time.time()
        
        try:
            # Create quantum task
            task = {
                "domain": "quantum",
                "description": "Run a simple quantum circuit",
                "parameters": {
                    "num_qubits": 4,
                    "algorithm": "grover",
                    "circuit_depth": 3,
                    "noise_level": 0.01
                }
            }
            
            # Execute task
            result = await self.ctm_interface.execute_task(task)
            
            # Check result
            if result.get("status") != "success":
                return {
                    "status": "fail",
                    "error": f"Quantum simulator execution failed: {result.get('error', 'Unknown error')}",
                    "duration": time.time() - start_time
                }
            
            return {
                "status": "pass",
                "result": result,
                "duration": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"CTM quantum simulator test failed: {e}")
            
            return {
                "status": "fail",
                "error": str(e),
                "duration": time.time() - start_time
            }
    
    async def _test_ctm_sorter(self) -> Dict[str, Any]:
        """
        Test CTM sorter.
        
        Returns:
            Test result
        """
        logger.info("Testing CTM sorter")
        
        start_time = time.time()
        
        try:
            # Create sorting task
            task = {
                "domain": "sorting",
                "description": "Sort a small array",
                "parameters": {
                    "array_size": 100,
                    "algorithm": "quick"
                }
            }
            
            # Execute task
            result = await self.ctm_interface.execute_task(task)
            
            # Check result
            if result.get("status") != "success":
                return {
                    "status": "fail",
                    "error": f"Sorter execution failed: {result.get('error', 'Unknown error')}",
                    "duration": time.time() - start_time
                }
            
            return {
                "status": "pass",
                "result": result,
                "duration": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"CTM sorter test failed: {e}")
            
            return {
                "status": "fail",
                "error": str(e),
                "duration": time.time() - start_time
            }
    
    async def _test_universal_router(self) -> None:
        """Test the Universal Router."""
        logger.info("Testing Universal Router")
        
        # Initialize results
        self.context["results"]["universal_router"] = {}
        
        # Create router
        try:
            self.router = UniversalRouter()
            
            self.context["components"]["universal_router"] = self.router
            
            # Test router initialization
            result = self._test_router_initialization()
            self.context["results"]["universal_router"]["initialization"] = result
            
            # Define mock solver
            class MockSolver(SolverInterface):
                """Mock solver for testing."""
                
                async def solve(self, task: Task) -> Dict[str, Any]:
                    """Solve a task."""
                    return {
                        "status": "success",
                        "result": {"completed": True},
                        "metrics": {
                            "execution_time": 0.1,
                            "efficiency": 0.9
                        }
                    }
                
                def can_solve(self, task: Task) -> bool:
                    """Check if this solver can solve the given task."""
                    return task.domain in ["test"]
                
                def get_capabilities(self) -> Dict[str, Any]:
                    """Get solver capabilities."""
                    return {
                        "domains": ["test"],
                        "max_complexity": 10
                    }
            
            # Register solver
            mock_solver = MockSolver("mock_solver")
            self.router.register_solver(mock_solver, ["test"])
            
            # Test task execution
            result = await self._test_router_task_execution()
            self.context["results"]["universal_router"]["task_execution"] = result
            
            # Test task routing
            result = await self._test_router_task_routing()
            self.context["results"]["universal_router"]["task_routing"] = result
            
            # Test router stats
            result = self._test_router_stats()
            self.context["results"]["universal_router"]["stats"] = result
            
        except Exception as e:
            logger.error(f"Failed to initialize Universal Router: {e}")
            self.context["errors"].append(f"Universal Router initialization failed: {e}")
            
            # Create failure results for all tests
            for test in ["initialization", "task_execution", "task_routing", "stats"]:
                self.context["results"]["universal_router"][test] = {
                    "status": "fail",
                    "error": str(e),
                    "duration": 0.0
                }
    
    def _test_router_initialization(self) -> Dict[str, Any]:
        """
        Test router initialization.
        
        Returns:
            Test result
        """
        logger.info("Testing router initialization")
        
        start_time = time.time()
        
        try:
            # Check router
            if not isinstance(self.router, UniversalRouter):
                return {
                    "status": "fail",
                    "error": "Router is not a UniversalRouter instance",
                    "duration": time.time() - start_time
                }
            
            # Check resource manager
            if not hasattr(self.router, "resource_manager"):
                return {
                    "status": "fail",
                    "error": "Router does not have a resource manager",
                    "duration": time.time() - start_time
                }
            
            return {
                "status": "pass",
                "duration": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Router initialization test failed: {e}")
            
            return {
                "status": "fail",
                "error": str(e),
                "duration": time.time() - start_time
            }
    
    async def _test_router_task_execution(self) -> Dict[str, Any]:
        """
        Test router task execution.
        
        Returns:
            Test result
        """
        logger.info("Testing router task execution")
        
        start_time = time.time()
        
        try:
            # Create task
            task = Task(
                task_id="test_1",
                domain="test",
                description="Test task",
                parameters={}
            )
            
            # Execute task
            result = await self.router.execute(task)
            
            # Check result
            if result.get("status") != "success":
                return {
                    "status": "fail",
                    "error": f"Task execution failed: {result.get('error', 'Unknown error')}",
                    "duration": time.time() - start_time
                }
            
            return {
                "status": "pass",
                "result": result,
                "duration": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Router task execution test failed: {e}")
            
            return {
                "status": "fail",
                "error": str(e),
                "duration": time.time() - start_time
            }
    
    async def _test_router_task_routing(self) -> Dict[str, Any]:
        """
        Test router task routing.
        
        Returns:
            Test result
        """
        logger.info("Testing router task routing")
        
        start_time = time.time()
        
        try:
            # Create tasks with different domains
            tasks = [
                Task(task_id="test_2", domain="test", description="Test task", parameters={}),
                Task(task_id="unknown_1", domain="unknown", description="Unknown task", parameters={})
            ]
            
            # Execute tasks
            results = await asyncio.gather(
                self.router.execute(tasks[0]),
                self.router.execute(tasks[1]),
                return_exceptions=True
            )
            
            # Check results
            if not isinstance(results[0], dict) or results[0].get("status") != "success":
                return {
                    "status": "fail",
                    "error": f"Test task execution failed: {results[0]}",
                    "duration": time.time() - start_time
                }
            
            if not isinstance(results[1], dict) or results[1].get("status") != "error":
                return {
                    "status": "fail",
                    "error": f"Unknown task should have failed: {results[1]}",
                    "duration": time.time() - start_time
                }
            
            return {
                "status": "pass",
                "results": results,
                "duration": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Router task routing test failed: {e}")
            
            return {
                "status": "fail",
                "error": str(e),
                "duration": time.time() - start_time
            }
    
    def _test_router_stats(self) -> Dict[str, Any]:
        """
        Test router statistics.
        
        Returns:
            Test result
        """
        logger.info("Testing router statistics")
        
        start_time = time.time()
        
        try:
            # Get stats
            stats = self.router.get_stats()
            
            # Check stats
            if not isinstance(stats, dict):
                return {
                    "status": "fail",
                    "error": "Stats is not a dictionary",
                    "duration": time.time() - start_time
                }
            
            # Check stats contents
            required_keys = ["total_tasks", "successful_tasks", "failed_tasks", "domains", "solvers"]
            missing_keys = [key for key in required_keys if key not in stats]
            
            if missing_keys:
                return {
                    "status": "fail",
                    "error": f"Stats is missing required keys: {missing_keys}",
                    "duration": time.time() - start_time
                }
            
            return {
                "status": "pass",
                "stats": stats,
                "duration": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Router stats test failed: {e}")
            
            return {
                "status": "fail",
                "error": str(e),
                "duration": time.time() - start_time
            }
    
    async def _test_agentic_brain(self) -> None:
        """Test the Agentic Brain Framework."""
        logger.info("Testing Agentic Brain Framework")
        
        # Initialize results
        self.context["results"]["agentic_brain"] = {}
        
        # Create brain framework
        try:
            # Look for config path
            brain_config_path = self.config_path
            
            if not brain_config_path or not os.path.exists(brain_config_path):
                # Try to find agentic_brain.yaml
                default_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    "configs", "agentic_brain.yaml"
                )
                
                if os.path.exists(default_path):
                    brain_config_path = default_path
            
            self.brain_framework = BrainFramework(
                config_path=brain_config_path,
                ctm_interface=self.ctm_interface,
                reward_system=None,
                dfz_adapter=None
            )
            
            self.context["components"]["brain_framework"] = self.brain_framework
            
            # Test brain initialization
            result = self._test_brain_initialization()
            self.context["results"]["agentic_brain"]["initialization"] = result
            
            # Test agent availability
            result = self._test_brain_agents()
            self.context["results"]["agentic_brain"]["agents"] = result
            
            # Test claude agent (if available)
            if "claude" in self.brain_framework.agent_controllers:
                result = await self._test_claude_agent()
                self.context["results"]["agentic_brain"]["claude_agent"] = result
            else:
                self.context["warnings"].append("Claude agent not available for testing")
            
            # Test deepseek agent (if available)
            if "deepseek" in self.brain_framework.agent_controllers:
                result = await self._test_deepseek_agent()
                self.context["results"]["agentic_brain"]["deepseek_agent"] = result
            else:
                self.context["warnings"].append("DeepSeek agent not available for testing")
            
        except Exception as e:
            logger.error(f"Failed to initialize Agentic Brain Framework: {e}")
            self.context["errors"].append(f"Agentic Brain Framework initialization failed: {e}")
            
            # Create failure results for all tests
            for test in ["initialization", "agents", "claude_agent", "deepseek_agent"]:
                self.context["results"]["agentic_brain"][test] = {
                    "status": "fail",
                    "error": str(e),
                    "duration": 0.0
                }
    
    def _test_brain_initialization(self) -> Dict[str, Any]:
        """
        Test brain framework initialization.
        
        Returns:
            Test result
        """
        logger.info("Testing brain framework initialization")
        
        start_time = time.time()
        
        try:
            # Check brain framework
            if not isinstance(self.brain_framework, BrainFramework):
                return {
                    "status": "fail",
                    "error": "Brain framework is not a BrainFramework instance",
                    "duration": time.time() - start_time
                }
            
            # Check agent controllers
            if not hasattr(self.brain_framework, "agent_controllers") or not self.brain_framework.agent_controllers:
                return {
                    "status": "fail",
                    "error": "Brain framework does not have agent controllers",
                    "duration": time.time() - start_time
                }
            
            return {
                "status": "pass",
                "controllers": list(self.brain_framework.agent_controllers.keys()),
                "duration": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Brain framework initialization test failed: {e}")
            
            return {
                "status": "fail",
                "error": str(e),
                "duration": time.time() - start_time
            }
    
    def _test_brain_agents(self) -> Dict[str, Any]:
        """
        Test brain agents.
        
        Returns:
            Test result
        """
        logger.info("Testing brain agents")
        
        start_time = time.time()
        
        try:
            # Check agent controllers
            agents = self.brain_framework.agent_controllers
            
            if not agents:
                return {
                    "status": "fail",
                    "error": "No agent controllers found",
                    "duration": time.time() - start_time
                }
            
            # Check default agent
            if not self.brain_framework.default_agent:
                return {
                    "status": "fail",
                    "error": "No default agent set",
                    "duration": time.time() - start_time
                }
            
            # Check if default agent exists
            if self.brain_framework.default_agent not in agents:
                return {
                    "status": "fail",
                    "error": f"Default agent '{self.brain_framework.default_agent}' not found in controllers",
                    "duration": time.time() - start_time
                }
            
            return {
                "status": "pass",
                "agents": list(agents.keys()),
                "default_agent": self.brain_framework.default_agent,
                "duration": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Brain agents test failed: {e}")
            
            return {
                "status": "fail",
                "error": str(e),
                "duration": time.time() - start_time
            }
    
    async def _test_claude_agent(self) -> Dict[str, Any]:
        """
        Test Claude agent.
        
        Returns:
            Test result
        """
        logger.info("Testing Claude agent")
        
        start_time = time.time()
        
        try:
            # Skip if we don't have an API key set
            if not os.environ.get("ANTHROPIC_API_KEY"):
                return {
                    "status": "skip",
                    "reason": "ANTHROPIC_API_KEY not set",
                    "duration": time.time() - start_time
                }
            
            # Skip if in a slow test exclusion mode
            if self.skip_slow_tests:
                return {
                    "status": "skip",
                    "reason": "Slow tests disabled",
                    "duration": time.time() - start_time
                }
            
            # Get agent controller
            agent = self.brain_framework.get_agent("claude")
            
            if not agent:
                return {
                    "status": "fail",
                    "error": "Claude agent not found",
                    "duration": time.time() - start_time
                }
            
            # Try to generate a simple task
            try:
                # Use a short timeout to avoid hanging
                task = await asyncio.wait_for(
                    agent.generate_task(
                        domain="test",
                        difficulty="easy",
                        constraints={"max_tokens": 100},
                        context={"test_mode": True}
                    ),
                    timeout=5.0
                )
                
                if "error" in task:
                    return {
                        "status": "fail",
                        "error": f"Task generation failed: {task['error']}",
                        "duration": time.time() - start_time
                    }
                
                return {
                    "status": "pass",
                    "task": task,
                    "duration": time.time() - start_time
                }
                
            except asyncio.TimeoutError:
                logger.warning("Claude API request timed out")
                
                return {
                    "status": "skip",
                    "reason": "API request timed out",
                    "duration": time.time() - start_time
                }
                
            except Exception as e:
                # If API error, mark as skip instead of fail
                if "API" in str(e):
                    logger.warning(f"Claude API error: {e}")
                    
                    return {
                        "status": "skip",
                        "reason": f"API error: {e}",
                        "duration": time.time() - start_time
                    }
                
                raise
            
        except Exception as e:
            logger.error(f"Claude agent test failed: {e}")
            
            return {
                "status": "fail",
                "error": str(e),
                "duration": time.time() - start_time
            }
    
    async def _test_deepseek_agent(self) -> Dict[str, Any]:
        """
        Test DeepSeek agent.
        
        Returns:
            Test result
        """
        logger.info("Testing DeepSeek agent")
        
        start_time = time.time()
        
        try:
            # Skip if we don't have an API key set
            if not os.environ.get("DEEPSEEK_API_KEY"):
                return {
                    "status": "skip",
                    "reason": "DEEPSEEK_API_KEY not set",
                    "duration": time.time() - start_time
                }
            
            # Skip if in a slow test exclusion mode
            if self.skip_slow_tests:
                return {
                    "status": "skip",
                    "reason": "Slow tests disabled",
                    "duration": time.time() - start_time
                }
            
            # Get agent controller
            agent = self.brain_framework.get_agent("deepseek")
            
            if not agent:
                return {
                    "status": "fail",
                    "error": "DeepSeek agent not found",
                    "duration": time.time() - start_time
                }
            
            # Try to generate a simple task
            try:
                # Use a short timeout to avoid hanging
                task = await asyncio.wait_for(
                    agent.generate_task(
                        domain="test",
                        difficulty="easy",
                        constraints={"max_tokens": 100},
                        context={"test_mode": True}
                    ),
                    timeout=5.0
                )
                
                if "error" in task:
                    return {
                        "status": "fail",
                        "error": f"Task generation failed: {task['error']}",
                        "duration": time.time() - start_time
                    }
                
                return {
                    "status": "pass",
                    "task": task,
                    "duration": time.time() - start_time
                }
                
            except asyncio.TimeoutError:
                logger.warning("DeepSeek API request timed out")
                
                return {
                    "status": "skip",
                    "reason": "API request timed out",
                    "duration": time.time() - start_time
                }
                
            except Exception as e:
                # If API error, mark as skip instead of fail
                if "API" in str(e):
                    logger.warning(f"DeepSeek API error: {e}")
                    
                    return {
                        "status": "skip",
                        "reason": f"API error: {e}",
                        "duration": time.time() - start_time
                    }
                
                raise
            
        except Exception as e:
            logger.error(f"DeepSeek agent test failed: {e}")
            
            return {
                "status": "fail",
                "error": str(e),
                "duration": time.time() - start_time
            }
    
    async def _test_dfz_integration(self) -> None:
        """Test the DFZ integration."""
        logger.info("Testing DFZ integration")
        
        # Initialize results
        self.context["results"]["dfz_integration"] = {}
        
        # Create DFZ adapter
        try:
            self.dfz_adapter = DFZAdapter(
                dfz_path=self.dfz_path,
                config={}
            )
            
            await self.dfz_adapter.initialize()
            
            self.context["components"]["dfz_adapter"] = self.dfz_adapter
            
            # Test adapter initialization
            result = self._test_dfz_adapter_initialization()
            self.context["results"]["dfz_integration"]["initialization"] = result
            
            # Test message sending
            if not self.skip_slow_tests:
                result = await self._test_dfz_message_sending()
                self.context["results"]["dfz_integration"]["message_sending"] = result
            
            # Test task generation
            if not self.skip_slow_tests:
                result = await self._test_dfz_task_generation()
                self.context["results"]["dfz_integration"]["task_generation"] = result
            
            # Add brain framework integration
            if self.brain_framework:
                self.brain_framework.dfz_adapter = self.dfz_adapter
                
                # Test brain framework integration
                result = await self._test_dfz_brain_integration()
                self.context["results"]["dfz_integration"]["brain_integration"] = result
            
        except Exception as e:
            logger.error(f"Failed to initialize DFZ adapter: {e}")
            self.context["errors"].append(f"DFZ adapter initialization failed: {e}")
            
            # Create failure results for all tests
            for test in ["initialization", "message_sending", "task_generation", "brain_integration"]:
                self.context["results"]["dfz_integration"][test] = {
                    "status": "fail",
                    "error": str(e),
                    "duration": 0.0
                }
    
    def _test_dfz_adapter_initialization(self) -> Dict[str, Any]:
        """
        Test DFZ adapter initialization.
        
        Returns:
            Test result
        """
        logger.info("Testing DFZ adapter initialization")
        
        start_time = time.time()
        
        try:
            # Check adapter
            if not isinstance(self.dfz_adapter, DFZAdapter):
                return {
                    "status": "fail",
                    "error": "DFZ adapter is not a DFZAdapter instance",
                    "duration": time.time() - start_time
                }
            
            # Check if DFZ is available
            if not self.dfz_adapter.dfz_available:
                return {
                    "status": "skip",
                    "reason": "DFZ is not available",
                    "duration": time.time() - start_time
                }
            
            return {
                "status": "pass",
                "dfz_path": self.dfz_adapter.dfz_path,
                "duration": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"DFZ adapter initialization test failed: {e}")
            
            return {
                "status": "fail",
                "error": str(e),
                "duration": time.time() - start_time
            }
    
    async def _test_dfz_message_sending(self) -> Dict[str, Any]:
        """
        Test DFZ message sending.
        
        Returns:
            Test result
        """
        logger.info("Testing DFZ message sending")
        
        start_time = time.time()
        
        try:
            # Skip if DFZ is not available
            if not self.dfz_adapter.dfz_available:
                return {
                    "status": "skip",
                    "reason": "DFZ is not available",
                    "duration": time.time() - start_time
                }
            
            # Try to send a message
            try:
                response = await asyncio.wait_for(
                    self.dfz_adapter.send_message(
                        "This is a test message from CTM-AbsoluteZero",
                        context={"test_mode": True}
                    ),
                    timeout=5.0
                )
                
                if "error" in response:
                    return {
                        "status": "fail",
                        "error": f"Message sending failed: {response['error']}",
                        "duration": time.time() - start_time
                    }
                
                return {
                    "status": "pass",
                    "response": response,
                    "duration": time.time() - start_time
                }
                
            except asyncio.TimeoutError:
                logger.warning("DFZ message sending timed out")
                
                return {
                    "status": "skip",
                    "reason": "Message sending timed out",
                    "duration": time.time() - start_time
                }
            
        except Exception as e:
            logger.error(f"DFZ message sending test failed: {e}")
            
            return {
                "status": "fail",
                "error": str(e),
                "duration": time.time() - start_time
            }
    
    async def _test_dfz_task_generation(self) -> Dict[str, Any]:
        """
        Test DFZ task generation.
        
        Returns:
            Test result
        """
        logger.info("Testing DFZ task generation")
        
        start_time = time.time()
        
        try:
            # Skip if DFZ is not available
            if not self.dfz_adapter.dfz_available:
                return {
                    "status": "skip",
                    "reason": "DFZ is not available",
                    "duration": time.time() - start_time
                }
            
            # Try to generate tasks
            try:
                tasks = await asyncio.wait_for(
                    self.dfz_adapter.generate_task(
                        domain="test",
                        context={"test_mode": True}
                    ),
                    timeout=5.0
                )
                
                if not tasks:
                    return {
                        "status": "fail",
                        "error": "No tasks generated",
                        "duration": time.time() - start_time
                    }
                
                if isinstance(tasks, list) and len(tasks) > 0 and "error" in tasks[0]:
                    return {
                        "status": "fail",
                        "error": f"Task generation failed: {tasks[0]['error']}",
                        "duration": time.time() - start_time
                    }
                
                return {
                    "status": "pass",
                    "tasks": tasks,
                    "duration": time.time() - start_time
                }
                
            except asyncio.TimeoutError:
                logger.warning("DFZ task generation timed out")
                
                return {
                    "status": "skip",
                    "reason": "Task generation timed out",
                    "duration": time.time() - start_time
                }
            
        except Exception as e:
            logger.error(f"DFZ task generation test failed: {e}")
            
            return {
                "status": "fail",
                "error": str(e),
                "duration": time.time() - start_time
            }
    
    async def _test_dfz_brain_integration(self) -> Dict[str, Any]:
        """
        Test DFZ brain integration.
        
        Returns:
            Test result
        """
        logger.info("Testing DFZ brain integration")
        
        start_time = time.time()
        
        try:
            # Skip if DFZ is not available
            if not self.dfz_adapter.dfz_available:
                return {
                    "status": "skip",
                    "reason": "DFZ is not available",
                    "duration": time.time() - start_time
                }
            
            # Check if brain framework has DFZ adapter
            if not self.brain_framework.dfz_adapter:
                return {
                    "status": "fail",
                    "error": "Brain framework does not have DFZ adapter",
                    "duration": time.time() - start_time
                }
            
            # Try to send a message through the brain framework
            try:
                response = await asyncio.wait_for(
                    self.brain_framework.send_to_dfz(
                        "This is a test message from CTM-AbsoluteZero brain framework",
                        context={"test_mode": True}
                    ),
                    timeout=5.0
                )
                
                if "error" in response:
                    return {
                        "status": "fail",
                        "error": f"Message sending failed: {response['error']}",
                        "duration": time.time() - start_time
                    }
                
                return {
                    "status": "pass",
                    "response": response,
                    "duration": time.time() - start_time
                }
                
            except asyncio.TimeoutError:
                logger.warning("DFZ brain integration timed out")
                
                return {
                    "status": "skip",
                    "reason": "Brain integration timed out",
                    "duration": time.time() - start_time
                }
            
        except Exception as e:
            logger.error(f"DFZ brain integration test failed: {e}")
            
            return {
                "status": "fail",
                "error": str(e),
                "duration": time.time() - start_time
            }
    
    def _save_results(self) -> None:
        """Save test results to file."""
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results object
        results = {
            "timestamp": timestamp,
            "duration": self.context["duration"],
            "success_rate": self.context["success_rate"],
            "errors": self.context["errors"],
            "warnings": self.context["warnings"],
            "results": self.context["results"]
        }
        
        # Save JSON results
        json_path = os.path.join(self.output_dir, f"test_results_{timestamp}.json")
        
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved test results to {json_path}")
        
        # Create readable summary
        summary_path = os.path.join(self.output_dir, f"test_summary_{timestamp}.txt")
        
        with open(summary_path, 'w') as f:
            f.write("=== CTM-AbsoluteZero Test Suite Summary ===\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {self.context['duration']:.2f} seconds\n")
            f.write(f"Success Rate: {self.context['success_rate']:.1%}\n\n")
            
            if self.context["errors"]:
                f.write("Errors:\n")
                for error in self.context["errors"]:
                    f.write(f"- {error}\n")
                f.write("\n")
            
            if self.context["warnings"]:
                f.write("Warnings:\n")
                for warning in self.context["warnings"]:
                    f.write(f"- {warning}\n")
                f.write("\n")
            
            f.write("=== Component Results ===\n\n")
            
            for component, component_results in self.context["results"].items():
                f.write(f"{component}:\n")
                
                for test_name, test_result in component_results.items():
                    status = test_result.get("status", "unknown")
                    status_str = "✅ PASS" if status == "pass" else "❌ FAIL" if status == "fail" else "⚠️ SKIP"
                    
                    f.write(f"  {test_name}: {status_str}\n")
                    
                    if status == "fail":
                        f.write(f"    Error: {test_result.get('error', 'Unknown error')}\n")
                    elif status == "skip":
                        f.write(f"    Reason: {test_result.get('reason', 'Unknown reason')}\n")
                
                f.write("\n")
            
            f.write("=== Detailed Results ===\n\n")
            
            for component, component_results in self.context["results"].items():
                f.write(f"{component}:\n")
                
                for test_name, test_result in component_results.items():
                    status = test_result.get("status", "unknown")
                    status_str = "✅ PASS" if status == "pass" else "❌ FAIL" if status == "fail" else "⚠️ SKIP"
                    duration = test_result.get("duration", 0)
                    
                    f.write(f"  {test_name}: {status_str} ({duration:.2f}s)\n")
                    
                    if status == "fail":
                        f.write(f"    Error: {test_result.get('error', 'Unknown error')}\n")
                    elif status == "skip":
                        f.write(f"    Reason: {test_result.get('reason', 'Unknown reason')}\n")
                    
                    # Add additional details for passing tests
                    if status == "pass":
                        if "components" in test_result:
                            f.write(f"    Components: {', '.join(test_result['components'])}\n")
                        
                        if "controllers" in test_result:
                            f.write(f"    Controllers: {', '.join(test_result['controllers'])}\n")
                        
                        if "agents" in test_result:
                            f.write(f"    Agents: {', '.join(test_result['agents'])}\n")
                            f.write(f"    Default: {test_result.get('default_agent', 'None')}\n")
                    
                f.write("\n")
            
            f.write("=== Conclusion ===\n\n")
            
            if self.context["success_rate"] >= 0.9:
                f.write("✅ The system is functioning correctly and ready for standby.\n")
            elif self.context["success_rate"] >= 0.7:
                f.write("⚠️ The system is functioning with minor issues. Review warnings before standby.\n")
            else:
                f.write("❌ The system has critical issues that need to be addressed before standby.\n")
        
        logger.info(f"Saved test summary to {summary_path}")
        
        # Save latest.txt for quick reference
        latest_path = os.path.join(self.output_dir, "latest.txt")
        
        with open(latest_path, 'w') as f:
            f.write(f"Latest test: {timestamp}\n")
            f.write(f"Success rate: {self.context['success_rate']:.1%}\n")
            f.write(f"Duration: {self.context['duration']:.2f}s\n")
            f.write(f"Full results: {json_path}\n")
            f.write(f"Summary: {summary_path}\n")
        
        logger.info(f"Updated latest test reference at {latest_path}")

def main():
    """Run test suite from command line."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="CTM-AbsoluteZero Test Suite")
    
    # Add arguments
    parser.add_argument(
        "--config", "-c",
        help="Path to configuration file",
        type=str
    )
    parser.add_argument(
        "--output-dir", "-o",
        help="Directory to save results",
        type=str,
        default="./test_results"
    )
    parser.add_argument(
        "--dfz-path", "-d",
        help="Path to DFZ installation",
        type=str
    )
    parser.add_argument(
        "--skip-slow",
        help="Skip slow tests (API calls, etc.)",
        action="store_true"
    )
    parser.add_argument(
        "--log-level",
        help="Logging level",
        choices=["debug", "info", "warning", "error"],
        default="info"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Configure logging
    log_level = getattr(logging, args.log_level.upper())
    configure_logging(log_level=log_level)
    
    # Create test suite
    test_suite = TestSuite(
        config_path=args.config,
        output_dir=args.output_dir,
        dfz_path=args.dfz_path,
        skip_slow_tests=args.skip_slow
    )
    
    # Run test suite
    asyncio.run(test_suite.run())

if __name__ == "__main__":
    main()