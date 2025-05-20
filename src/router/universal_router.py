"""
Universal Router implementation for CTM-AbsoluteZero.

This module provides a unified routing mechanism for directing tasks to appropriate
solvers based on domain, requirements, and resource availability.
"""
import logging
import time
import asyncio
import json
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
import numpy as np

from ..utils.logging import get_logger

# Setup logger
logger = get_logger("ctm-az.router")

@dataclass
class Task:
    """Task representation for the router."""
    
    task_id: str
    domain: str
    description: str
    parameters: Dict[str, Any]
    priority: int = 0
    deadline: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "task_id": self.task_id,
            "domain": self.domain,
            "description": self.description,
            "parameters": self.parameters,
            "priority": self.priority,
            "deadline": self.deadline,
            "metadata": self.metadata,
            "context": self.context
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create task from dictionary."""
        return cls(
            task_id=data["task_id"],
            domain=data["domain"],
            description=data["description"],
            parameters=data["parameters"],
            priority=data.get("priority", 0),
            deadline=data.get("deadline"),
            metadata=data.get("metadata", {}),
            context=data.get("context", {})
        )


class SolverInterface:
    """Interface for solver implementations."""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        Initialize the solver interface.
        
        Args:
            name: Solver name
            config: Solver configuration
        """
        self.name = name
        self.config = config or {}
        self.logger = get_logger(f"ctm-az.router.solver.{name}")
    
    async def solve(self, task: Task) -> Dict[str, Any]:
        """
        Solve a task.
        
        Args:
            task: Task to solve
            
        Returns:
            Solution result
        """
        raise NotImplementedError("Solver must implement solve method")
    
    def can_solve(self, task: Task) -> bool:
        """
        Check if this solver can solve the given task.
        
        Args:
            task: Task to check
            
        Returns:
            True if this solver can solve the task, False otherwise
        """
        raise NotImplementedError("Solver must implement can_solve method")
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get solver capabilities.
        
        Returns:
            Dictionary of capabilities
        """
        raise NotImplementedError("Solver must implement get_capabilities method")


class ResourceManager:
    """Resource manager for the Universal Router."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the resource manager.
        
        Args:
            config: Resource manager configuration
        """
        self.config = config or {}
        self.logger = get_logger("ctm-az.router.resources")
        
        # Resource limits
        self.max_concurrent_tasks = self.config.get("max_concurrent_tasks", 8)
        self.max_memory = self.config.get("max_memory", 48 * 1024 * 1024 * 1024)  # 48GB
        self.max_storage = self.config.get("max_storage", 200 * 1024 * 1024)  # 200MB
        self.max_api_calls = self.config.get("max_api_calls", 50)
        self.max_cost = self.config.get("max_cost", 2.0)  # $2.00
        
        # Resource usage tracking
        self.active_tasks = 0
        self.memory_usage = 0
        self.storage_usage = 0
        self.api_calls = 0
        self.cost = 0.0
        
        # Locks for resource allocation
        self._lock = asyncio.Lock()
    
    async def allocate(self, resources: Dict[str, Any]) -> bool:
        """
        Allocate resources for a task.
        
        Args:
            resources: Resource requirements
            
        Returns:
            True if resources were allocated, False otherwise
        """
        async with self._lock:
            # Check if resources are available
            if not self._check_availability(resources):
                return False
            
            # Allocate resources
            self.active_tasks += 1
            self.memory_usage += resources.get("memory", 0)
            self.storage_usage += resources.get("storage", 0)
            self.api_calls += resources.get("api_calls", 0)
            self.cost += resources.get("cost", 0.0)
            
            return True
    
    async def release(self, resources: Dict[str, Any]) -> None:
        """
        Release allocated resources.
        
        Args:
            resources: Resources to release
        """
        async with self._lock:
            # Release resources
            self.active_tasks = max(0, self.active_tasks - 1)
            self.memory_usage = max(0, self.memory_usage - resources.get("memory", 0))
            self.storage_usage = max(0, self.storage_usage - resources.get("storage", 0))
            self.api_calls = max(0, self.api_calls - resources.get("api_calls", 0))
            self.cost = max(0.0, self.cost - resources.get("cost", 0.0))
    
    def _check_availability(self, resources: Dict[str, Any]) -> bool:
        """
        Check if resources are available.
        
        Args:
            resources: Resource requirements
            
        Returns:
            True if resources are available, False otherwise
        """
        if self.active_tasks >= self.max_concurrent_tasks:
            self.logger.warning("Max concurrent tasks limit reached")
            return False
        
        if self.memory_usage + resources.get("memory", 0) > self.max_memory:
            self.logger.warning("Memory limit would be exceeded")
            return False
        
        if self.storage_usage + resources.get("storage", 0) > self.max_storage:
            self.logger.warning("Storage limit would be exceeded")
            return False
        
        if self.api_calls + resources.get("api_calls", 0) > self.max_api_calls:
            self.logger.warning("API calls limit would be exceeded")
            return False
        
        if self.cost + resources.get("cost", 0.0) > self.max_cost:
            self.logger.warning("Cost limit would be exceeded")
            return False
        
        return True
    
    def get_usage(self) -> Dict[str, Any]:
        """
        Get current resource usage.
        
        Returns:
            Dictionary of resource usage
        """
        return {
            "active_tasks": self.active_tasks,
            "memory_usage": self.memory_usage,
            "storage_usage": self.storage_usage,
            "api_calls": self.api_calls,
            "cost": self.cost,
            "limits": {
                "max_concurrent_tasks": self.max_concurrent_tasks,
                "max_memory": self.max_memory,
                "max_storage": self.max_storage,
                "max_api_calls": self.max_api_calls,
                "max_cost": self.max_cost
            }
        }


class UniversalRouter:
    """
    Universal Router for CTM-AbsoluteZero.
    
    This router directs tasks to appropriate solvers based on domain,
    requirements, and resource availability.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Universal Router.
        
        Args:
            config: Router configuration
        """
        self.config = config or {}
        self.logger = get_logger("ctm-az.router")
        
        # Initialize resource manager
        resource_config = self.config.get("resources", {})
        self.resource_manager = ResourceManager(resource_config)
        
        # Task queue
        self.task_queue: asyncio.Queue = asyncio.Queue()
        
        # Registered solvers
        self.solvers: Dict[str, SolverInterface] = {}
        
        # Domain to solver mapping
        self.domain_mappings: Dict[str, List[str]] = {}
        
        # Statistics
        self.stats = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "task_durations": [],
            "avg_duration": 0.0,
            "domains": {}
        }
        
        # State
        self.running = False
        self._worker_tasks = []
        
        self.logger.info("Universal Router initialized")
    
    def register_solver(self, solver: SolverInterface, domains: List[str] = None) -> None:
        """
        Register a solver with the router.
        
        Args:
            solver: Solver to register
            domains: Optional list of domains this solver can handle
        """
        # Register solver
        self.solvers[solver.name] = solver
        
        # Update domain mappings
        if domains:
            for domain in domains:
                if domain not in self.domain_mappings:
                    self.domain_mappings[domain] = []
                self.domain_mappings[domain].append(solver.name)
        
        self.logger.info(f"Registered solver: {solver.name}")
    
    def add_task(self, task: Union[Task, Dict[str, Any]]) -> None:
        """
        Add a task to the queue.
        
        Args:
            task: Task to add
        """
        # Convert dictionary to Task if needed
        if isinstance(task, dict):
            task = Task.from_dict(task)
        
        # Add to queue
        self.task_queue.put_nowait(task)
        
        self.logger.info(f"Added task to queue: {task.task_id}")
    
    async def execute(
        self,
        task: Union[Task, Dict[str, Any]],
        wait: bool = True,
        timeout: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a task immediately (bypassing the queue).
        
        Args:
            task: Task to execute
            wait: Whether to wait for the result
            timeout: Optional timeout for waiting
            
        Returns:
            Execution result if wait is True, None otherwise
        """
        # Convert dictionary to Task if needed
        if isinstance(task, dict):
            task = Task.from_dict(task)
        
        self.logger.info(f"Executing task: {task.task_id}")
        
        if wait:
            # Execute directly and return result
            return await self._process_task(task)
        else:
            # Add to queue and return immediately
            self.add_task(task)
            return None
    
    async def start(self, num_workers: int = 4) -> None:
        """
        Start the router workers.
        
        Args:
            num_workers: Number of worker tasks to start
        """
        if self.running:
            self.logger.warning("Router is already running")
            return
        
        self.running = True
        
        # Start worker tasks
        for i in range(num_workers):
            worker = asyncio.create_task(self._worker())
            self._worker_tasks.append(worker)
        
        self.logger.info(f"Started {num_workers} worker tasks")
    
    async def stop(self) -> None:
        """Stop the router workers."""
        if not self.running:
            self.logger.warning("Router is not running")
            return
        
        self.running = False
        
        # Cancel all worker tasks
        for worker in self._worker_tasks:
            worker.cancel()
        
        # Wait for workers to finish
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        
        self._worker_tasks = []
        
        self.logger.info("Router stopped")
    
    async def _worker(self) -> None:
        """Worker task that processes tasks from the queue."""
        while self.running:
            try:
                # Get task from queue
                task = await self.task_queue.get()
                
                # Process task
                await self._process_task(task)
                
                # Mark task as done
                self.task_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Worker error: {e}")
    
    async def _process_task(self, task: Task) -> Dict[str, Any]:
        """
        Process a single task.
        
        Args:
            task: Task to process
            
        Returns:
            Task result
        """
        self.logger.info(f"Processing task: {task.task_id}")
        
        # Update statistics
        self.stats["total_tasks"] += 1
        if task.domain not in self.stats["domains"]:
            self.stats["domains"][task.domain] = {
                "total_tasks": 0,
                "successful_tasks": 0,
                "failed_tasks": 0
            }
        self.stats["domains"][task.domain]["total_tasks"] += 1
        
        # Find appropriate solver
        solver_name = await self._find_solver(task)
        
        if not solver_name:
            error_msg = f"No suitable solver found for task: {task.task_id}"
            self.logger.error(error_msg)
            
            # Update statistics
            self.stats["failed_tasks"] += 1
            self.stats["domains"][task.domain]["failed_tasks"] += 1
            
            return {
                "task_id": task.task_id,
                "status": "error",
                "error": error_msg
            }
        
        # Get solver
        solver = self.solvers[solver_name]
        
        # Estimate resource requirements
        resources = self._estimate_resources(task, solver_name)
        
        # Allocate resources
        if not await self.resource_manager.allocate(resources):
            error_msg = f"Failed to allocate resources for task: {task.task_id}"
            self.logger.error(error_msg)
            
            # Update statistics
            self.stats["failed_tasks"] += 1
            self.stats["domains"][task.domain]["failed_tasks"] += 1
            
            return {
                "task_id": task.task_id,
                "status": "error",
                "error": error_msg
            }
        
        # Execute task
        start_time = time.time()
        
        try:
            # Solve task
            result = await solver.solve(task)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Update statistics
            self.stats["successful_tasks"] += 1
            self.stats["domains"][task.domain]["successful_tasks"] += 1
            self.stats["task_durations"].append(duration)
            self.stats["avg_duration"] = sum(self.stats["task_durations"]) / len(self.stats["task_durations"])
            
            # Add metadata to result
            result["task_id"] = task.task_id
            result["solver"] = solver_name
            result["duration"] = duration
            
            self.logger.info(f"Task {task.task_id} completed in {duration:.2f}s")
            
            return result
            
        except Exception as e:
            # Calculate duration
            duration = time.time() - start_time
            
            self.logger.error(f"Task {task.task_id} failed: {e}")
            
            # Update statistics
            self.stats["failed_tasks"] += 1
            self.stats["domains"][task.domain]["failed_tasks"] += 1
            
            return {
                "task_id": task.task_id,
                "status": "error",
                "error": str(e),
                "solver": solver_name,
                "duration": duration
            }
            
        finally:
            # Release resources
            await self.resource_manager.release(resources)
    
    async def _find_solver(self, task: Task) -> Optional[str]:
        """
        Find a suitable solver for a task.
        
        Args:
            task: Task to find solver for
            
        Returns:
            Solver name or None if no suitable solver found
        """
        # Check domain mappings first
        if task.domain in self.domain_mappings:
            for solver_name in self.domain_mappings[task.domain]:
                solver = self.solvers.get(solver_name)
                if solver and solver.can_solve(task):
                    return solver_name
        
        # Check all solvers if no domain mapping or no suitable solver found
        for solver_name, solver in self.solvers.items():
            if solver.can_solve(task):
                return solver_name
        
        return None
    
    def _estimate_resources(self, task: Task, solver_name: str) -> Dict[str, Any]:
        """
        Estimate resource requirements for a task.
        
        Args:
            task: Task to estimate resources for
            solver_name: Name of the solver that will process the task
            
        Returns:
            Dictionary of resource requirements
        """
        domain = task.domain
        parameters = task.parameters
        
        # Default values
        resources = {
            "memory": 100 * 1024 * 1024,  # 100MB
            "storage": 1 * 1024 * 1024,   # 1MB
            "api_calls": 0,
            "cost": 0.0
        }
        
        # Adjust based on domain
        if domain == "quantum":
            num_qubits = parameters.get("num_qubits", 4)
            resources["memory"] = 100 * 1024 * 1024 * (2 ** min(num_qubits, 24))  # Exponential with number of qubits
            resources["api_calls"] = 0
            resources["cost"] = 0.0
            
        elif domain == "maze":
            size = parameters.get("size", [10, 10])
            area = size[0] * size[1]
            resources["memory"] = 100 * 1024 * 1024 * (area / 100)  # Linear with maze area
            resources["api_calls"] = 0
            resources["cost"] = 0.0
            
        elif domain == "sorting":
            array_size = parameters.get("array_size", 1000)
            resources["memory"] = 100 * 1024 * 1024 * (array_size / 1000)  # Linear with array size
            resources["api_calls"] = 0
            resources["cost"] = 0.0
            
        # Adjust based on solver
        if "claude" in solver_name.lower():
            resources["api_calls"] += 1
            resources["cost"] += 0.01  # $0.01 per API call (approximate)
            
        elif "deepseek" in solver_name.lower():
            resources["api_calls"] += 1
            resources["cost"] += 0.001  # $0.001 per API call (approximate)
        
        return resources
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get router statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            **self.stats,
            "solvers": list(self.solvers.keys()),
            "domains": list(self.domain_mappings.keys()),
            "resources": self.resource_manager.get_usage()
        }