"""
DFZ integration module for CTM-AbsoluteZero with concurrent task execution support.
This module extends the base DFZ integration with support for concurrent task execution.
"""
import logging
import json
import os
import sys
import time
import asyncio
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import numpy as np

# Add paths for both CTM-AbsoluteZero and DFZ
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'evolution')))

# Import base DFZ integration components
from src.integration.dfz import CTMAbsoluteZeroPlugin, DFZAdapter, create_dfz_plugin
from src.utils.logging import get_logger, CTMLogger

logger = get_logger("ctm-az.integration.dfz_concurrent")

class ConcurrentTaskManager:
    """
    Manager for concurrent task execution in CTM-AbsoluteZero.
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        use_processes: bool = False
    ):
        """
        Initialize the concurrent task manager.
        
        Args:
            max_workers: Maximum number of concurrent workers
            use_processes: Use process-based parallelism instead of threads
        """
        self.max_workers = max_workers
        self.use_processes = use_processes
        
        # Create executor based on preference
        self._create_executor()
        
        # Task tracking
        self.active_tasks = {}
        self.completed_tasks = {}
        self.task_lock = threading.RLock()
        
        # Performance metrics
        self.metrics = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "task_durations": [],
            "concurrent_executions": [],
            "max_concurrency": 0
        }
    
    def _create_executor(self):
        """Create the appropriate executor"""
        if self.use_processes:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
    
    def submit_task(
        self,
        task_func: Callable,
        task: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Submit a task for execution.
        
        Args:
            task_func: Function to execute the task
            task: Task to execute
            context: Execution context
            
        Returns:
            Future representing the pending task
        """
        with self.task_lock:
            # Update metrics
            self.metrics["total_tasks"] += 1
            
            # Record concurrency level
            current_concurrency = len(self.active_tasks) + 1
            self.metrics["concurrent_executions"].append(current_concurrency)
            self.metrics["max_concurrency"] = max(
                self.metrics["max_concurrency"], 
                current_concurrency
            )
            
            # Submit the task with start time tracking
            task_id = task.get("id", f"task_{len(self.active_tasks)}")
            start_time = time.time()
            
            future = self.executor.submit(
                self._execute_task_with_timing,
                task_func, task, context, start_time
            )
            
            # Register the active task
            self.active_tasks[task_id] = {
                "future": future,
                "task": task,
                "start_time": start_time
            }
            
            return future
    
    def _execute_task_with_timing(
        self,
        task_func: Callable,
        task: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        start_time: float
    ):
        """
        Execute a task and track timing information.
        
        Args:
            task_func: Function to execute the task
            task: Task to execute
            context: Execution context
            start_time: Task start time
            
        Returns:
            Task result with timing information
        """
        try:
            # Execute the task
            result = task_func(task, context)
            
            # Add timing information
            end_time = time.time()
            duration = end_time - start_time
            
            if isinstance(result, dict):
                result["start_time"] = start_time
                result["end_time"] = end_time
                result["duration"] = duration
            else:
                # Handle unexpected return type
                result = {
                    "task_id": task.get("id", "unknown"),
                    "status": "unknown",
                    "result": result,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": duration
                }
            
            with self.task_lock:
                # Update metrics
                self.metrics["task_durations"].append(duration)
                if result.get("status") == "success":
                    self.metrics["successful_tasks"] += 1
                else:
                    self.metrics["failed_tasks"] += 1
            
            return result
        except Exception as e:
            # Handle exceptions
            end_time = time.time()
            duration = end_time - start_time
            
            error_result = {
                "task_id": task.get("id", "unknown"),
                "status": "error",
                "error": str(e),
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration
            }
            
            with self.task_lock:
                self.metrics["task_durations"].append(duration)
                self.metrics["failed_tasks"] += 1
            
            return error_result
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the status of a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task status information
        """
        # Check if the task is active
        if task_id in self.active_tasks:
            task_info = self.active_tasks[task_id]
            future = task_info["future"]
            
            if future.done():
                # Task completed but not yet moved
                try:
                    result = future.result()
                    status = "completed"
                except Exception as e:
                    result = {"error": str(e)}
                    status = "failed"
            else:
                # Task still running
                result = None
                status = "running"
            
            return {
                "task_id": task_id,
                "status": status,
                "start_time": task_info["start_time"],
                "elapsed": time.time() - task_info["start_time"],
                "result": result
            }
        
        # Check if the task is completed
        elif task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        
        # Task not found
        return {
            "task_id": task_id,
            "status": "unknown",
            "error": "Task not found"
        }
    
    def collect_completed_tasks(self):
        """
        Collect results from completed tasks and update internal state.
        
        Returns:
            Dictionary of completed task results
        """
        completed = {}
        
        with self.task_lock:
            # Check for completed tasks
            completed_ids = []
            
            for task_id, task_info in self.active_tasks.items():
                future = task_info["future"]
                
                if future.done():
                    completed_ids.append(task_id)
                    
                    try:
                        result = future.result()
                        self.completed_tasks[task_id] = {
                            "task_id": task_id,
                            "status": "completed",
                            "start_time": task_info["start_time"],
                            "end_time": time.time(),
                            "result": result
                        }
                        completed[task_id] = self.completed_tasks[task_id]
                    except Exception as e:
                        self.completed_tasks[task_id] = {
                            "task_id": task_id,
                            "status": "failed",
                            "start_time": task_info["start_time"],
                            "end_time": time.time(),
                            "error": str(e)
                        }
                        completed[task_id] = self.completed_tasks[task_id]
            
            # Remove completed tasks from active tasks
            for task_id in completed_ids:
                del self.active_tasks[task_id]
        
        return completed
    
    def wait_all(self, timeout=None):
        """
        Wait for all active tasks to complete.
        
        Args:
            timeout: Maximum time to wait (None = wait indefinitely)
            
        Returns:
            Dictionary of completed task results
        """
        with self.task_lock:
            futures = [info["future"] for info in self.active_tasks.values()]
        
        # Wait for all futures to complete
        if futures:
            if timeout is not None:
                done, not_done = asyncio.wait(
                    [asyncio.wrap_future(f) for f in futures],
                    timeout=timeout
                )
            else:
                for future in futures:
                    # Handle synchronous wait
                    try:
                        future.result()
                    except Exception:
                        # Exceptions are already captured in task results
                        pass
        
        # Collect all completed tasks
        return self.collect_completed_tasks()
    
    def get_metrics(self):
        """
        Get performance metrics.
        
        Returns:
            Performance metrics
        """
        with self.task_lock:
            metrics = self.metrics.copy()
            
            # Calculate derived metrics
            if metrics["total_tasks"] > 0:
                metrics["success_rate"] = metrics["successful_tasks"] / metrics["total_tasks"]
                metrics["failure_rate"] = metrics["failed_tasks"] / metrics["total_tasks"]
            else:
                metrics["success_rate"] = 0.0
                metrics["failure_rate"] = 0.0
                
            if metrics["task_durations"]:
                metrics["avg_duration"] = sum(metrics["task_durations"]) / len(metrics["task_durations"])
                metrics["min_duration"] = min(metrics["task_durations"])
                metrics["max_duration"] = max(metrics["task_durations"])
            else:
                metrics["avg_duration"] = 0.0
                metrics["min_duration"] = 0.0
                metrics["max_duration"] = 0.0
                
            if metrics["concurrent_executions"]:
                metrics["avg_concurrency"] = sum(metrics["concurrent_executions"]) / len(metrics["concurrent_executions"])
            else:
                metrics["avg_concurrency"] = 0.0
            
            return metrics
    
    def shutdown(self):
        """Shut down the executor"""
        self.executor.shutdown(wait=True)


class ConcurrentCTMAbsoluteZeroPlugin(CTMAbsoluteZeroPlugin):
    """
    Extended plugin for CTM-AbsoluteZero with concurrent task execution support.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        standalone: bool = False,
        max_workers: int = 4,
        use_processes: bool = False
    ):
        """
        Initialize the concurrent CTM-AbsoluteZero plugin.
        
        Args:
            config_path: Path to configuration file
            standalone: Whether to operate in standalone mode (without DFZ)
            max_workers: Maximum number of concurrent workers
            use_processes: Use process-based parallelism instead of threads
        """
        # Initialize base plugin
        super().__init__(config_path, standalone)
        
        # Create concurrent task manager
        self.task_manager = ConcurrentTaskManager(
            max_workers=max_workers,
            use_processes=use_processes
        )
    
    async def initialize(self, manager=None) -> bool:
        """
        Initialize the plugin.
        
        Args:
            manager: DFZ manager (None in standalone mode)
            
        Returns:
            True if initialization was successful
        """
        # Initialize base plugin
        success = await super().initialize(manager)
        
        if success:
            logger.info(f"Concurrent CTM-AbsoluteZero plugin initialized with {self.task_manager.max_workers} workers")
        
        return success
    
    async def shutdown(self) -> None:
        """Clean up resources when shutting down."""
        # Shut down task manager
        self.task_manager.shutdown()
        
        # Call base shutdown
        await super().shutdown()
    
    def execute_task_concurrent(
        self,
        task: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Submit a task for concurrent execution.
        
        Args:
            task: Task to execute
            context: Execution context
            
        Returns:
            Future representing the pending task
        """
        return self.task_manager.submit_task(
            super().execute_task,
            task,
            context
        )
    
    def execute_tasks_batch(
        self,
        tasks: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
        wait: bool = True,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Execute a batch of tasks concurrently.
        
        Args:
            tasks: List of tasks to execute
            context: Execution context
            wait: Whether to wait for tasks to complete
            timeout: Maximum time to wait if wait=True
            
        Returns:
            Dictionary with task futures and/or results
        """
        # Submit all tasks
        futures = {}
        for task in tasks:
            task_id = task.get("id", f"task_{len(futures)}")
            future = self.execute_task_concurrent(task, context)
            futures[task_id] = future
        
        if wait:
            # Wait for all tasks to complete
            completed = self.task_manager.wait_all(timeout)
            return {
                "futures": futures,
                "results": completed,
                "metrics": self.task_manager.get_metrics()
            }
        else:
            # Return futures without waiting
            return {
                "futures": futures,
                "metrics": self.task_manager.get_metrics()
            }
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the status of a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task status information
        """
        return self.task_manager.get_task_status(task_id)
    
    def collect_completed_tasks(self) -> Dict[str, Any]:
        """
        Collect results from completed tasks.
        
        Returns:
            Dictionary of completed task results
        """
        return self.task_manager.collect_completed_tasks()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Returns:
            Performance metrics
        """
        # Get base metrics
        base_metrics = super().get_performance_metrics()
        
        # Get concurrent metrics
        concurrent_metrics = self.task_manager.get_metrics()
        
        # Combine metrics
        return {
            **base_metrics,
            "concurrent": concurrent_metrics
        }


class ConcurrentDFZAdapter(DFZAdapter):
    """
    Extended adapter for DFZ with concurrent task execution support.
    """
    
    def __init__(
        self,
        dfz_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        max_workers: int = 4,
        use_processes: bool = False
    ):
        """
        Initialize the concurrent DFZ adapter.
        
        Args:
            dfz_path: Path to DFZ installation
            config: Adapter configuration
            max_workers: Maximum number of concurrent workers
            use_processes: Use process-based parallelism instead of threads
        """
        # Initialize base fields
        self.dfz_path = dfz_path
        self.config = config or {}
        self.logger = get_logger("ctm-az.integration.dfz_concurrent_adapter")
        
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
            
        # Create concurrent plugin instance
        self.plugin = ConcurrentCTMAbsoluteZeroPlugin(
            config_path=self.config.get("config_path"),
            standalone=not self.dfz_available,
            max_workers=max_workers,
            use_processes=use_processes
        )
    
    async def execute_tasks_batch(
        self,
        tasks: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
        wait: bool = True,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Execute a batch of tasks concurrently.
        
        Args:
            tasks: List of tasks to execute
            context: Execution context
            wait: Whether to wait for tasks to complete
            timeout: Maximum time to wait if wait=True
            
        Returns:
            Dictionary with task futures and/or results
        """
        return self.plugin.execute_tasks_batch(tasks, context, wait, timeout)
    
    async def execute_task_concurrent(
        self,
        task: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Submit a task for concurrent execution.
        
        Args:
            task: Task to execute
            context: Execution context
            
        Returns:
            Future representing the pending task
        """
        return self.plugin.execute_task_concurrent(task, context)
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the status of a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task status information
        """
        return self.plugin.get_task_status(task_id)
    
    def collect_completed_tasks(self) -> Dict[str, Any]:
        """
        Collect results from completed tasks.
        
        Returns:
            Dictionary of completed task results
        """
        return self.plugin.collect_completed_tasks()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Returns:
            Performance metrics
        """
        return self.plugin.get_performance_metrics()


def create_concurrent_dfz_plugin(
    config_path: Optional[str] = None,
    max_workers: int = 4,
    use_processes: bool = False
) -> ConcurrentCTMAbsoluteZeroPlugin:
    """
    Create a concurrent CTM-AbsoluteZero plugin for DFZ.
    
    Args:
        config_path: Path to configuration file
        max_workers: Maximum number of concurrent workers
        use_processes: Use process-based parallelism instead of threads
        
    Returns:
        Concurrent CTM-AbsoluteZero plugin
    """
    return ConcurrentCTMAbsoluteZeroPlugin(
        config_path=config_path,
        max_workers=max_workers,
        use_processes=use_processes
    )