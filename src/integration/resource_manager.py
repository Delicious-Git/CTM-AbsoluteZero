"""
Resource manager for CTM-AbsoluteZero.
This module provides dynamic resource allocation for concurrent task execution.
"""

import os
import time
import asyncio
import threading
import json
import logging
import psutil
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from datetime import datetime, timedelta
import numpy as np
from collections import deque
import heapq

# Configure logging
logger = logging.getLogger("ctm-az.integration.resource_manager")

class ResourceProfile:
    """
    Resource profile for a task domain.
    This tracks the resource usage patterns of different task types.
    """
    
    def __init__(
        self,
        domain: str,
        window_size: int = 50
    ):
        """
        Initialize the resource profile.
        
        Args:
            domain: Task domain
            window_size: Size of the sliding window for metrics
        """
        self.domain = domain
        self.window_size = window_size
        
        # Metrics storage with sliding window
        self.cpu_usage = deque(maxlen=window_size)
        self.memory_usage = deque(maxlen=window_size)
        self.execution_times = deque(maxlen=window_size)
        self.success_rates = deque(maxlen=window_size)
        
        # Aggregated statistics
        self._avg_cpu_usage = 0.0
        self._avg_memory_usage = 0.0
        self._avg_execution_time = 0.0
        self._success_rate = 1.0
        
        # Task type specific metrics
        self.task_type_metrics = {}
    
    def add_metrics(
        self,
        cpu_usage: float,
        memory_usage: float,
        execution_time: float,
        success: bool,
        task_type: Optional[str] = None
    ):
        """
        Add metrics from a task execution.
        
        Args:
            cpu_usage: CPU usage percentage (0-100)
            memory_usage: Memory usage in MB
            execution_time: Execution time in seconds
            success: Whether the task succeeded
            task_type: Optional task type for more granular tracking
        """
        # Add to sliding windows
        self.cpu_usage.append(cpu_usage)
        self.memory_usage.append(memory_usage)
        self.execution_times.append(execution_time)
        self.success_rates.append(1.0 if success else 0.0)
        
        # Update aggregated statistics
        self._update_statistics()
        
        # Update task type specific metrics
        if task_type:
            if task_type not in self.task_type_metrics:
                self.task_type_metrics[task_type] = {
                    "cpu_usage": deque(maxlen=self.window_size),
                    "memory_usage": deque(maxlen=self.window_size),
                    "execution_times": deque(maxlen=self.window_size),
                    "success_rates": deque(maxlen=self.window_size),
                    "avg_cpu_usage": 0.0,
                    "avg_memory_usage": 0.0,
                    "avg_execution_time": 0.0,
                    "success_rate": 1.0
                }
            
            metrics = self.task_type_metrics[task_type]
            metrics["cpu_usage"].append(cpu_usage)
            metrics["memory_usage"].append(memory_usage)
            metrics["execution_times"].append(execution_time)
            metrics["success_rates"].append(1.0 if success else 0.0)
            
            # Update task type statistics
            metrics["avg_cpu_usage"] = sum(metrics["cpu_usage"]) / len(metrics["cpu_usage"]) if metrics["cpu_usage"] else 0.0
            metrics["avg_memory_usage"] = sum(metrics["memory_usage"]) / len(metrics["memory_usage"]) if metrics["memory_usage"] else 0.0
            metrics["avg_execution_time"] = sum(metrics["execution_times"]) / len(metrics["execution_times"]) if metrics["execution_times"] else 0.0
            metrics["success_rate"] = sum(metrics["success_rates"]) / len(metrics["success_rates"]) if metrics["success_rates"] else 1.0
    
    def _update_statistics(self):
        """Update aggregated statistics based on sliding windows."""
        self._avg_cpu_usage = sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0.0
        self._avg_memory_usage = sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0.0
        self._avg_execution_time = sum(self.execution_times) / len(self.execution_times) if self.execution_times else 0.0
        self._success_rate = sum(self.success_rates) / len(self.success_rates) if self.success_rates else 1.0
    
    @property
    def avg_cpu_usage(self) -> float:
        """Get the average CPU usage."""
        return self._avg_cpu_usage
    
    @property
    def avg_memory_usage(self) -> float:
        """Get the average memory usage."""
        return self._avg_memory_usage
    
    @property
    def avg_execution_time(self) -> float:
        """Get the average execution time."""
        return self._avg_execution_time
    
    @property
    def success_rate(self) -> float:
        """Get the success rate."""
        return self._success_rate
    
    def estimate_resources(
        self,
        task_type: Optional[str] = None,
        complexity_factor: float = 1.0
    ) -> Tuple[float, float, float]:
        """
        Estimate the resources needed for a task.
        
        Args:
            task_type: Optional task type for more accurate estimation
            complexity_factor: Factor to adjust for task complexity
            
        Returns:
            Tuple of (estimated CPU usage, estimated memory usage, estimated execution time)
        """
        if task_type and task_type in self.task_type_metrics:
            # Use task type specific metrics
            metrics = self.task_type_metrics[task_type]
            cpu = metrics["avg_cpu_usage"] * complexity_factor
            memory = metrics["avg_memory_usage"] * complexity_factor
            time = metrics["avg_execution_time"] * complexity_factor
        else:
            # Use domain-wide metrics
            cpu = self._avg_cpu_usage * complexity_factor
            memory = self._avg_memory_usage * complexity_factor
            time = self._avg_execution_time * complexity_factor
        
        return cpu, memory, time
    
    def get_metrics(self, include_task_types: bool = False) -> Dict[str, Any]:
        """
        Get the current resource profile metrics.
        
        Args:
            include_task_types: Whether to include task type specific metrics
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            "domain": self.domain,
            "avg_cpu_usage": self._avg_cpu_usage,
            "avg_memory_usage": self._avg_memory_usage,
            "avg_execution_time": self._avg_execution_time,
            "success_rate": self._success_rate,
            "sample_count": len(self.execution_times)
        }
        
        if include_task_types:
            task_types = {}
            for task_type, type_metrics in self.task_type_metrics.items():
                task_types[task_type] = {
                    "avg_cpu_usage": type_metrics["avg_cpu_usage"],
                    "avg_memory_usage": type_metrics["avg_memory_usage"],
                    "avg_execution_time": type_metrics["avg_execution_time"],
                    "success_rate": type_metrics["success_rate"],
                    "sample_count": len(type_metrics["execution_times"])
                }
            metrics["task_types"] = task_types
        
        return metrics

class PriorityTask:
    """
    Task with priority for the priority queue.
    """
    
    def __init__(
        self,
        task_id: str,
        priority: int,
        task: Dict[str, Any],
        timestamp: float
    ):
        """
        Initialize the priority task.
        
        Args:
            task_id: Task ID
            priority: Task priority (lower is higher priority)
            task: Task data
            timestamp: Task creation timestamp
        """
        self.task_id = task_id
        self.priority = priority
        self.task = task
        self.timestamp = timestamp
    
    def __lt__(self, other):
        """Compare tasks based on priority and then timestamp."""
        if self.priority == other.priority:
            return self.timestamp < other.timestamp
        return self.priority < other.priority

class ResourceManager:
    """
    Dynamic resource manager for concurrent task execution.
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        min_workers: int = 1,
        cpu_threshold: float = 80.0,
        memory_threshold: float = 80.0,
        adjustment_interval: float = 5.0,
        enable_priority_queue: bool = True,
        domain_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the resource manager.
        
        Args:
            max_workers: Maximum number of workers
            min_workers: Minimum number of workers
            cpu_threshold: CPU usage threshold percentage
            memory_threshold: Memory usage threshold percentage
            adjustment_interval: Worker count adjustment interval in seconds
            enable_priority_queue: Whether to enable the priority queue
            domain_weights: Domain weight dictionary for resource allocation
        """
        self.max_workers = max_workers
        self.min_workers = min_workers
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.adjustment_interval = adjustment_interval
        self.enable_priority_queue = enable_priority_queue
        
        # Current worker count
        self.current_workers = max(min_workers, max_workers // 2)
        
        # Resource profiles
        self.resource_profiles = {}
        
        # Domain weights
        self.domain_weights = domain_weights or {}
        
        # System resource usage
        self.system_cpu_usage = deque(maxlen=10)
        self.system_memory_usage = deque(maxlen=10)
        
        # Priority queue
        self.priority_queue = []
        self.queue_lock = threading.Lock()
        
        # Resource monitor thread
        self.monitor_thread = None
        self.stop_event = threading.Event()
    
    def start_monitoring(self):
        """Start the resource monitoring thread."""
        if self.monitor_thread is not None:
            return
        
        self.stop_event.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop the resource monitoring thread."""
        if self.monitor_thread is None:
            return
        
        self.stop_event.set()
        self.monitor_thread.join(timeout=1.0)
        self.monitor_thread = None
        
        logger.info("Resource monitoring stopped")
    
    def _monitor_resources(self):
        """Monitor system resources and adjust worker count."""
        last_adjustment_time = 0.0
        
        while not self.stop_event.is_set():
            try:
                # Get current system resource usage
                cpu_percent = psutil.cpu_percent(interval=None)
                memory_percent = psutil.virtual_memory().percent
                
                # Add to sliding windows
                self.system_cpu_usage.append(cpu_percent)
                self.system_memory_usage.append(memory_percent)
                
                # Calculate averages
                avg_cpu = sum(self.system_cpu_usage) / len(self.system_cpu_usage) if self.system_cpu_usage else 0.0
                avg_memory = sum(self.system_memory_usage) / len(self.system_memory_usage) if self.system_memory_usage else 0.0
                
                # Check if it's time to adjust worker count
                current_time = time.time()
                if current_time - last_adjustment_time >= self.adjustment_interval:
                    self._adjust_worker_count(avg_cpu, avg_memory)
                    last_adjustment_time = current_time
                
                # Sleep for a short time
                time.sleep(1.0)
            except Exception as e:
                logger.error(f"Error in resource monitoring thread: {e}")
                time.sleep(5.0)
    
    def _adjust_worker_count(self, cpu_percent: float, memory_percent: float):
        """
        Adjust the worker count based on system resource usage.
        
        Args:
            cpu_percent: CPU usage percentage
            memory_percent: Memory usage percentage
        """
        old_workers = self.current_workers
        
        # Check if we need to reduce workers
        if cpu_percent > self.cpu_threshold or memory_percent > self.memory_threshold:
            # System is overloaded, reduce workers
            self.current_workers = max(self.min_workers, self.current_workers - 1)
        elif cpu_percent < self.cpu_threshold * 0.7 and memory_percent < self.memory_threshold * 0.7:
            # System has plenty of resources, increase workers
            self.current_workers = min(self.max_workers, self.current_workers + 1)
        
        if old_workers != self.current_workers:
            logger.info(f"Adjusted worker count: {old_workers} -> {self.current_workers} "
                       f"(CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%)")
    
    def add_task(
        self,
        task_id: str,
        task: Dict[str, Any],
        priority: int = 5,
        domain: Optional[str] = None
    ):
        """
        Add a task to the queue.
        
        Args:
            task_id: Task ID
            task: Task data
            priority: Task priority (1-10, lower is higher priority)
            domain: Task domain (optional)
        """
        # Use domain from task if not provided
        if domain is None and isinstance(task, dict):
            domain = task.get("domain")
        
        # Adjust priority based on domain weight
        if domain and domain in self.domain_weights:
            weight = self.domain_weights[domain]
            priority = max(1, min(10, int(priority * (1.0 / weight))))
        
        # Create priority task
        priority_task = PriorityTask(
            task_id=task_id,
            priority=priority,
            task=task,
            timestamp=time.time()
        )
        
        # Add to priority queue
        with self.queue_lock:
            heapq.heappush(self.priority_queue, priority_task)
        
        logger.debug(f"Task {task_id} added to queue with priority {priority}")
    
    def get_next_task(self) -> Optional[Dict[str, Any]]:
        """
        Get the next task from the queue.
        
        Returns:
            Task data or None if queue is empty
        """
        if not self.enable_priority_queue or not self.priority_queue:
            return None
        
        with self.queue_lock:
            if not self.priority_queue:
                return None
            
            priority_task = heapq.heappop(self.priority_queue)
            
            logger.debug(f"Task {priority_task.task_id} retrieved from queue "
                        f"(priority: {priority_task.priority})")
            
            return priority_task.task
    
    def get_queue_length(self) -> int:
        """
        Get the current queue length.
        
        Returns:
            Number of tasks in the queue
        """
        with self.queue_lock:
            return len(self.priority_queue)
    
    def get_recommended_workers(self) -> int:
        """
        Get the recommended number of workers based on current system load.
        
        Returns:
            Recommended number of workers
        """
        return self.current_workers
    
    def update_resource_profile(
        self,
        domain: str,
        cpu_usage: float,
        memory_usage: float,
        execution_time: float,
        success: bool,
        task_type: Optional[str] = None
    ):
        """
        Update the resource profile for a domain.
        
        Args:
            domain: Task domain
            cpu_usage: CPU usage percentage
            memory_usage: Memory usage in MB
            execution_time: Execution time in seconds
            success: Whether the task succeeded
            task_type: Optional task type
        """
        if domain not in self.resource_profiles:
            self.resource_profiles[domain] = ResourceProfile(domain)
        
        self.resource_profiles[domain].add_metrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            execution_time=execution_time,
            success=success,
            task_type=task_type
        )
        
        logger.debug(f"Updated resource profile for domain '{domain}': "
                    f"CPU={cpu_usage:.1f}%, Memory={memory_usage:.1f}MB, "
                    f"Time={execution_time:.3f}s, Success={success}")
    
    def estimate_resources(
        self,
        domain: str,
        task_type: Optional[str] = None,
        complexity_factor: float = 1.0
    ) -> Tuple[float, float, float]:
        """
        Estimate the resources needed for a task.
        
        Args:
            domain: Task domain
            task_type: Optional task type
            complexity_factor: Factor to adjust for task complexity
            
        Returns:
            Tuple of (estimated CPU usage, estimated memory usage, estimated execution time)
        """
        if domain in self.resource_profiles:
            return self.resource_profiles[domain].estimate_resources(
                task_type=task_type,
                complexity_factor=complexity_factor
            )
        else:
            # No profile available, use default estimates
            return 20.0, 100.0, 1.0
    
    def get_domain_metrics(self, domain: str, include_task_types: bool = False) -> Dict[str, Any]:
        """
        Get metrics for a specific domain.
        
        Args:
            domain: Task domain
            include_task_types: Whether to include task type specific metrics
            
        Returns:
            Dictionary of metrics or empty dict if domain not found
        """
        if domain in self.resource_profiles:
            return self.resource_profiles[domain].get_metrics(include_task_types)
        return {}
    
    def get_all_metrics(self, include_task_types: bool = False) -> Dict[str, Any]:
        """
        Get metrics for all domains.
        
        Args:
            include_task_types: Whether to include task type specific metrics
            
        Returns:
            Dictionary of metrics by domain
        """
        metrics = {}
        for domain, profile in self.resource_profiles.items():
            metrics[domain] = profile.get_metrics(include_task_types)
        
        # Add system metrics
        metrics["system"] = {
            "cpu_usage": sum(self.system_cpu_usage) / len(self.system_cpu_usage) if self.system_cpu_usage else 0.0,
            "memory_usage": sum(self.system_memory_usage) / len(self.system_memory_usage) if self.system_memory_usage else 0.0,
            "current_workers": self.current_workers,
            "max_workers": self.max_workers,
            "queue_length": self.get_queue_length()
        }
        
        return metrics
    
    def set_domain_weight(self, domain: str, weight: float):
        """
        Set the weight for a domain.
        
        Args:
            domain: Task domain
            weight: Weight factor (higher means more resources)
        """
        self.domain_weights[domain] = weight
        logger.debug(f"Set weight for domain '{domain}': {weight}")
    
    def get_domain_weight(self, domain: str) -> float:
        """
        Get the weight for a domain.
        
        Args:
            domain: Task domain
            
        Returns:
            Weight factor
        """
        return self.domain_weights.get(domain, 1.0)
    
    def save_profiles(self, filepath: str):
        """
        Save resource profiles to a file.
        
        Args:
            filepath: Path to save the profiles
        """
        data = {
            "profiles": {},
            "domain_weights": self.domain_weights,
            "timestamp": datetime.now().isoformat()
        }
        
        # Convert profiles to serializable format
        for domain, profile in self.resource_profiles.items():
            data["profiles"][domain] = profile.get_metrics(include_task_types=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Resource profiles saved to {filepath}")
    
    def load_profiles(self, filepath: str) -> bool:
        """
        Load resource profiles from a file.
        
        Args:
            filepath: Path to load the profiles from
            
        Returns:
            True if loaded successfully
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Load domain weights
            self.domain_weights = data.get("domain_weights", {})
            
            # Load profiles
            for domain, metrics in data.get("profiles", {}).items():
                if domain not in self.resource_profiles:
                    self.resource_profiles[domain] = ResourceProfile(domain)
                
                profile = self.resource_profiles[domain]
                
                # Add some sample data points based on the averages
                sample_count = min(10, metrics.get("sample_count", 0))
                if sample_count > 0:
                    avg_cpu = metrics.get("avg_cpu_usage", 0.0)
                    avg_memory = metrics.get("avg_memory_usage", 0.0)
                    avg_time = metrics.get("avg_execution_time", 0.0)
                    success_rate = metrics.get("success_rate", 1.0)
                    
                    for _ in range(sample_count):
                        # Add some random variation
                        cpu = max(0.1, avg_cpu * (0.9 + 0.2 * np.random.random()))
                        memory = max(0.1, avg_memory * (0.9 + 0.2 * np.random.random()))
                        time = max(0.1, avg_time * (0.9 + 0.2 * np.random.random()))
                        success = np.random.random() < success_rate
                        
                        profile.add_metrics(cpu, memory, time, success)
                
                # Load task type metrics
                for task_type, type_metrics in metrics.get("task_types", {}).items():
                    sample_count = min(5, type_metrics.get("sample_count", 0))
                    if sample_count > 0:
                        avg_cpu = type_metrics.get("avg_cpu_usage", 0.0)
                        avg_memory = type_metrics.get("avg_memory_usage", 0.0)
                        avg_time = type_metrics.get("avg_execution_time", 0.0)
                        success_rate = type_metrics.get("success_rate", 1.0)
                        
                        for _ in range(sample_count):
                            # Add some random variation
                            cpu = max(0.1, avg_cpu * (0.9 + 0.2 * np.random.random()))
                            memory = max(0.1, avg_memory * (0.9 + 0.2 * np.random.random()))
                            time = max(0.1, avg_time * (0.9 + 0.2 * np.random.random()))
                            success = np.random.random() < success_rate
                            
                            profile.add_metrics(cpu, memory, time, success, task_type)
            
            logger.info(f"Resource profiles loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load resource profiles: {e}")
            return False
    
    def clear_queue(self):
        """Clear the task queue."""
        with self.queue_lock:
            self.priority_queue = []
        
        logger.info("Task queue cleared")
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the task queue.
        
        Returns:
            Dictionary of queue statistics
        """
        with self.queue_lock:
            if not self.priority_queue:
                return {
                    "length": 0,
                    "priorities": {},
                    "domains": {},
                    "oldest_task_age": 0,
                    "newest_task_age": 0
                }
            
            # Calculate statistics
            priorities = {}
            domains = {}
            timestamps = []
            
            for task in self.priority_queue:
                # Count priorities
                priority = task.priority
                priorities[priority] = priorities.get(priority, 0) + 1
                
                # Count domains
                domain = task.task.get("domain", "unknown")
                domains[domain] = domains.get(domain, 0) + 1
                
                # Add timestamp
                timestamps.append(task.timestamp)
            
            current_time = time.time()
            oldest_task_age = current_time - min(timestamps) if timestamps else 0
            newest_task_age = current_time - max(timestamps) if timestamps else 0
            
            return {
                "length": len(self.priority_queue),
                "priorities": priorities,
                "domains": domains,
                "oldest_task_age": oldest_task_age,
                "newest_task_age": newest_task_age
            }

def create_resource_manager(
    max_workers: int = 4,
    enable_priority_queue: bool = True,
    config_path: Optional[str] = None
) -> ResourceManager:
    """
    Create a resource manager.
    
    Args:
        max_workers: Maximum number of workers
        enable_priority_queue: Whether to enable the priority queue
        config_path: Path to configuration file
        
    Returns:
        Resource manager instance
    """
    # Load configuration if provided
    config = {}
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load resource manager configuration: {e}")
    
    # Create resource manager with configuration
    manager = ResourceManager(
        max_workers=config.get("max_workers", max_workers),
        min_workers=config.get("min_workers", 1),
        cpu_threshold=config.get("cpu_threshold", 80.0),
        memory_threshold=config.get("memory_threshold", 80.0),
        adjustment_interval=config.get("adjustment_interval", 5.0),
        enable_priority_queue=config.get("enable_priority_queue", enable_priority_queue),
        domain_weights=config.get("domain_weights", {})
    )
    
    # Load profiles if available
    profiles_path = config.get("profiles_path")
    if profiles_path and os.path.exists(profiles_path):
        manager.load_profiles(profiles_path)
    
    # Start monitoring
    manager.start_monitoring()
    
    return manager