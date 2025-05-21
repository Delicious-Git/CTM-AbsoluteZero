#!/usr/bin/env python3
"""
Demo for the resource manager with concurrent task execution.
This script demonstrates how to use the resource manager to optimize concurrent task execution.
"""

import os
import sys
import asyncio
import time
import json
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components
from src.integration.dfz_concurrent import ConcurrentDFZAdapter
from src.integration.resource_manager import ResourceManager, create_resource_manager
from src.ctm.quantum_sim import QuantumSimulator
from src.utils.config import ConfigManager

async def generate_mixed_tasks(count_per_type=5):
    """
    Generate a mix of tasks with different priorities.
    
    Args:
        count_per_type: Number of tasks per type
        
    Returns:
        List of tasks with priorities
    """
    task_data = []
    
    # Generate quantum tasks (high priority)
    for i in range(count_per_type):
        algorithm = np.random.choice(["vqe", "grover", "qft"])
        num_qubits = np.random.randint(3, 8)
        
        task = {
            "id": f"quantum_task_{i+1}",
            "domain": "quantum",
            "description": f"Quantum task {i+1} ({algorithm})",
            "parameters": {
                "algorithm": algorithm,
                "num_qubits": num_qubits,
                "noise_level": np.random.uniform(0.01, 0.1),
                "circuit_depth": np.random.randint(2, 8)
            }
        }
        priority = np.random.randint(1, 4)  # High priority (1-3)
        task_data.append((task, priority))
    
    # Generate sorting tasks (medium priority)
    for i in range(count_per_type):
        list_length = np.random.randint(10, 100)
        list_type = np.random.choice(["random", "nearly_sorted", "reverse_sorted"])
        
        task = {
            "id": f"sorting_task_{i+1}",
            "domain": "sorting",
            "description": f"Sorting task {i+1} ({list_type})",
            "parameters": {
                "list_length": list_length,
                "list_type": list_type,
                "value_range": [0, 1000],
                "algorithm": np.random.choice(["quicksort", "mergesort", "heapsort"])
            }
        }
        priority = np.random.randint(4, 7)  # Medium priority (4-6)
        task_data.append((task, priority))
    
    # Generate image analysis tasks (low priority)
    for i in range(count_per_type):
        image_size = np.random.randint(32, 256)
        categories = np.random.randint(2, 10)
        
        task = {
            "id": f"image_task_{i+1}",
            "domain": "image_classification",
            "description": f"Image analysis task {i+1}",
            "parameters": {
                "image_size": image_size,
                "num_categories": categories,
                "augmentation": random.choice([True, False]),
                "model_type": np.random.choice(["cnn", "resnet", "vit"])
            }
        }
        priority = np.random.randint(7, 10)  # Low priority (7-9)
        task_data.append((task, priority))
    
    # Shuffle tasks
    random.shuffle(task_data)
    
    return task_data

class ResourceManagerDemo:
    """
    Demo for the resource manager with concurrent task execution.
    """
    
    def __init__(
        self,
        config_path=None,
        max_workers=4,
        use_processes=False,
        enable_priority_queue=True
    ):
        """
        Initialize the demo.
        
        Args:
            config_path: Path to configuration file
            max_workers: Maximum number of concurrent workers
            use_processes: Use processes instead of threads
            enable_priority_queue: Whether to enable the priority queue
        """
        self.config_path = config_path
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.enable_priority_queue = enable_priority_queue
        
        # Initialize components
        self.dfz_adapter = None
        self.resource_manager = None
        
        # Results
        self.results_dir = Path("results/resource_manager")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create resource profiles
        self._create_sample_profiles()
    
    def _create_sample_profiles(self):
        """Create sample resource profiles for demonstration."""
        profiles_path = self.results_dir / "sample_profiles.json"
        
        if profiles_path.exists():
            # Profiles already exist
            return
        
        # Create profile data
        data = {
            "profiles": {
                "quantum": {
                    "domain": "quantum",
                    "avg_cpu_usage": 70.0,
                    "avg_memory_usage": 250.0,
                    "avg_execution_time": 2.5,
                    "success_rate": 0.95,
                    "sample_count": 20,
                    "task_types": {
                        "vqe": {
                            "avg_cpu_usage": 80.0,
                            "avg_memory_usage": 300.0,
                            "avg_execution_time": 3.0,
                            "success_rate": 0.93,
                            "sample_count": 10
                        },
                        "grover": {
                            "avg_cpu_usage": 65.0,
                            "avg_memory_usage": 220.0,
                            "avg_execution_time": 2.0,
                            "success_rate": 0.97,
                            "sample_count": 8
                        },
                        "qft": {
                            "avg_cpu_usage": 60.0,
                            "avg_memory_usage": 200.0,
                            "avg_execution_time": 1.8,
                            "success_rate": 0.98,
                            "sample_count": 7
                        }
                    }
                },
                "sorting": {
                    "domain": "sorting",
                    "avg_cpu_usage": 50.0,
                    "avg_memory_usage": 150.0,
                    "avg_execution_time": 1.0,
                    "success_rate": 0.98,
                    "sample_count": 15,
                    "task_types": {
                        "quicksort": {
                            "avg_cpu_usage": 45.0,
                            "avg_memory_usage": 120.0,
                            "avg_execution_time": 0.8,
                            "success_rate": 0.99,
                            "sample_count": 5
                        },
                        "mergesort": {
                            "avg_cpu_usage": 55.0,
                            "avg_memory_usage": 180.0,
                            "avg_execution_time": 1.2,
                            "success_rate": 0.97,
                            "sample_count": 5
                        },
                        "heapsort": {
                            "avg_cpu_usage": 60.0,
                            "avg_memory_usage": 160.0,
                            "avg_execution_time": 1.1,
                            "success_rate": 0.98,
                            "sample_count": 5
                        }
                    }
                },
                "image_classification": {
                    "domain": "image_classification",
                    "avg_cpu_usage": 85.0,
                    "avg_memory_usage": 400.0,
                    "avg_execution_time": 4.0,
                    "success_rate": 0.92,
                    "sample_count": 12,
                    "task_types": {
                        "cnn": {
                            "avg_cpu_usage": 75.0,
                            "avg_memory_usage": 350.0,
                            "avg_execution_time": 3.5,
                            "success_rate": 0.93,
                            "sample_count": 4
                        },
                        "resnet": {
                            "avg_cpu_usage": 90.0,
                            "avg_memory_usage": 450.0,
                            "avg_execution_time": 4.5,
                            "success_rate": 0.90,
                            "sample_count": 4
                        },
                        "vit": {
                            "avg_cpu_usage": 95.0,
                            "avg_memory_usage": 500.0,
                            "avg_execution_time": 5.0,
                            "success_rate": 0.89,
                            "sample_count": 4
                        }
                    }
                }
            },
            "domain_weights": {
                "quantum": 1.5,
                "sorting": 1.0,
                "image_classification": 0.8
            },
            "timestamp": time.time()
        }
        
        # Save profile data
        with open(profiles_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Sample resource profiles created at {profiles_path}")
    
    async def initialize(self):
        """Initialize components."""
        print("Initializing components...")
        
        # Initialize DFZ adapter
        self.dfz_adapter = ConcurrentDFZAdapter(
            config={"config_path": self.config_path},
            max_workers=self.max_workers,
            use_processes=self.use_processes
        )
        
        # Initialize the adapter
        await self.dfz_adapter.initialize()
        print(f"DFZ adapter initialized with {self.max_workers} workers")
        
        # Initialize resource manager
        self.resource_manager = create_resource_manager(
            max_workers=self.max_workers,
            enable_priority_queue=self.enable_priority_queue
        )
        
        # Load sample profiles
        profiles_path = self.results_dir / "sample_profiles.json"
        if profiles_path.exists():
            self.resource_manager.load_profiles(str(profiles_path))
            print("Resource profiles loaded")
        
        print("Components initialized")
        return True
    
    async def run_standard_concurrent(self, tasks):
        """
        Run tasks using standard concurrent execution.
        
        Args:
            tasks: List of tasks
            
        Returns:
            Execution results
        """
        print(f"Running {len(tasks)} tasks with standard concurrent execution...")
        
        start_time = time.time()
        
        # Extract tasks without priorities
        task_list = [task for task, _ in tasks]
        
        # Use standard batch execution
        batch_result = await self.dfz_adapter.execute_tasks_batch(
            task_list, wait=True
        )
        
        total_duration = time.time() - start_time
        
        # Process results
        results = list(batch_result["results"].values())
        
        # Extract task results from the "result" field
        for r in results:
            if "result" in r and isinstance(r["result"], dict):
                r.update(r["result"])
        
        # Collect metrics
        metrics = {
            "total_duration": total_duration,
            "avg_task_duration": sum(r.get("duration", 0) for r in results) / len(results),
            "task_durations": [r.get("duration", 0) for r in results],
            "success_rate": sum(1 for r in results if r.get("status") == "success") / len(results),
            "concurrent_workers": self.max_workers,
        }
        
        # Add concurrent metrics from the adapter
        adapter_metrics = self.dfz_adapter.get_performance_metrics()
        if "concurrent" in adapter_metrics:
            metrics.update(adapter_metrics["concurrent"])
        
        print(f"Standard concurrent execution completed in {total_duration:.3f}s")
        print(f"Average task duration: {metrics['avg_task_duration']:.3f}s")
        print(f"Success rate: {metrics['success_rate'] * 100:.1f}%")
        
        return {
            "results": results,
            "metrics": metrics
        }
    
    async def run_resource_managed(self, tasks):
        """
        Run tasks using resource-managed concurrent execution.
        
        Args:
            tasks: List of tasks
            
        Returns:
            Execution results
        """
        print(f"Running {len(tasks)} tasks with resource-managed concurrent execution...")
        
        # Add tasks to the resource manager queue
        for task, priority in tasks:
            self.resource_manager.add_task(
                task_id=task["id"],
                task=task,
                priority=priority,
                domain=task.get("domain")
            )
        
        # Get the recommended worker count
        recommended_workers = self.resource_manager.get_recommended_workers()
        print(f"Resource manager recommends {recommended_workers} workers")
        
        # Adjust the DFZ adapter's worker count if needed
        if recommended_workers != self.dfz_adapter.plugin.task_manager.max_workers:
            print(f"Adjusting worker count: {self.dfz_adapter.plugin.task_manager.max_workers} -> {recommended_workers}")
            # Note: In a real implementation, we would modify the worker pool
            
        start_time = time.time()
        results = []
        
        # Process tasks from the queue
        while self.resource_manager.get_queue_length() > 0:
            # Get the next task from the queue
            task = self.resource_manager.get_next_task()
            if not task:
                break
            
            domain = task.get("domain", "unknown")
            task_type = None
            if domain == "quantum":
                task_type = task.get("parameters", {}).get("algorithm")
            elif domain == "sorting":
                task_type = task.get("parameters", {}).get("algorithm")
            elif domain == "image_classification":
                task_type = task.get("parameters", {}).get("model_type")
            
            # Get resource estimates
            cpu_estimate, memory_estimate, time_estimate = self.resource_manager.estimate_resources(
                domain=domain,
                task_type=task_type
            )
            
            print(f"Processing task {task['id']} (priority: {task.get('_priority', 'unknown')})")
            print(f"  Resource estimates: CPU={cpu_estimate:.1f}%, Memory={memory_estimate:.1f}MB, Time={time_estimate:.2f}s")
            
            # Execute the task
            task_start = time.time()
            result = await self.dfz_adapter.execute_task(task)
            task_duration = time.time() - task_start
            
            # Add execution time to result
            result["execution_time"] = task_duration
            results.append(result)
            
            # Update resource profile with actual usage
            cpu_usage = np.random.uniform(0.8, 1.2) * cpu_estimate  # Simulate actual usage
            memory_usage = np.random.uniform(0.8, 1.2) * memory_estimate
            success = result.get("status") == "success"
            
            self.resource_manager.update_resource_profile(
                domain=domain,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                execution_time=task_duration,
                success=success,
                task_type=task_type
            )
            
            print(f"  Task completed in {task_duration:.3f}s (status: {result.get('status', 'unknown')})")
            print(f"  Updated resource profile for domain '{domain}'")
        
        total_duration = time.time() - start_time
        
        # Collect metrics
        metrics = {
            "total_duration": total_duration,
            "avg_task_duration": sum(r["execution_time"] for r in results) / len(results),
            "task_durations": [r["execution_time"] for r in results],
            "success_rate": sum(1 for r in results if r.get("status") == "success") / len(results),
            "recommended_workers": recommended_workers
        }
        
        print(f"Resource-managed execution completed in {total_duration:.3f}s")
        print(f"Average task duration: {metrics['avg_task_duration']:.3f}s")
        print(f"Success rate: {metrics['success_rate'] * 100:.1f}%")
        
        # Save updated profiles
        profiles_path = self.results_dir / "updated_profiles.json"
        self.resource_manager.save_profiles(str(profiles_path))
        print(f"Updated resource profiles saved to {profiles_path}")
        
        return {
            "results": results,
            "metrics": metrics
        }
    
    async def run_profile_comparison(self):
        """
        Run a comparison of resource profiles before and after execution.
        
        Returns:
            Comparison results
        """
        print("Comparing resource profiles...")
        
        # Get all metrics
        all_metrics = self.resource_manager.get_all_metrics(include_task_types=True)
        
        # Plot domain metrics
        self._plot_domain_metrics(all_metrics)
        
        # Plot task type metrics
        for domain in all_metrics:
            if domain == "system":
                continue
            
            if "task_types" in all_metrics[domain]:
                self._plot_task_type_metrics(domain, all_metrics[domain]["task_types"])
        
        return all_metrics
    
    def _plot_domain_metrics(self, all_metrics):
        """
        Plot domain metrics.
        
        Args:
            all_metrics: Dictionary of all metrics
        """
        # Exclude system metrics
        domains = [d for d in all_metrics.keys() if d != "system"]
        
        if not domains:
            return
        
        # Extract metrics
        cpu_usage = [all_metrics[d]["avg_cpu_usage"] for d in domains]
        memory_usage = [all_metrics[d]["avg_memory_usage"] for d in domains]
        execution_times = [all_metrics[d]["avg_execution_time"] for d in domains]
        success_rates = [all_metrics[d]["success_rate"] * 100 for d in domains]
        
        # Normalize for bar chart
        cpu_norm = [u / max(cpu_usage) for u in cpu_usage]
        memory_norm = [u / max(memory_usage) for u in memory_usage]
        time_norm = [t / max(execution_times) for t in execution_times]
        
        # Create figure with 4 subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # CPU usage
        ax1.bar(domains, cpu_usage, color='skyblue')
        ax1.set_ylabel('CPU Usage (%)')
        ax1.set_title('Average CPU Usage by Domain')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Memory usage
        ax2.bar(domains, memory_usage, color='lightgreen')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('Average Memory Usage by Domain')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Execution time
        ax3.bar(domains, execution_times, color='salmon')
        ax3.set_ylabel('Execution Time (s)')
        ax3.set_title('Average Execution Time by Domain')
        ax3.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Success rate
        ax4.bar(domains, success_rates, color='gold')
        ax4.set_ylabel('Success Rate (%)')
        ax4.set_title('Success Rate by Domain')
        ax4.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(self.results_dir / "domain_metrics.png")
        plt.close()
        
        print(f"Domain metrics plot saved to {self.results_dir / 'domain_metrics.png'}")
        
        # Create normalized comparison chart
        plt.figure(figsize=(10, 6))
        x = np.arange(len(domains))
        width = 0.25
        
        plt.bar(x - width, cpu_norm, width, label='CPU', color='skyblue')
        plt.bar(x, memory_norm, width, label='Memory', color='lightgreen')
        plt.bar(x + width, time_norm, width, label='Time', color='salmon')
        
        plt.xlabel('Domain')
        plt.ylabel('Normalized Value')
        plt.title('Resource Usage by Domain (Normalized)')
        plt.xticks(x, domains)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "domain_comparison.png")
        plt.close()
        
        print(f"Domain comparison plot saved to {self.results_dir / 'domain_comparison.png'}")
    
    def _plot_task_type_metrics(self, domain, task_types):
        """
        Plot task type metrics for a domain.
        
        Args:
            domain: Domain name
            task_types: Dictionary of task type metrics
        """
        if not task_types:
            return
        
        # Extract metrics
        types = list(task_types.keys())
        cpu_usage = [task_types[t]["avg_cpu_usage"] for t in types]
        memory_usage = [task_types[t]["avg_memory_usage"] for t in types]
        execution_times = [task_types[t]["avg_execution_time"] for t in types]
        success_rates = [task_types[t]["success_rate"] * 100 for t in types]
        
        # Create figure with 4 subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # CPU usage
        ax1.bar(types, cpu_usage, color='skyblue')
        ax1.set_ylabel('CPU Usage (%)')
        ax1.set_title(f'CPU Usage by {domain.capitalize()} Task Type')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Memory usage
        ax2.bar(types, memory_usage, color='lightgreen')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title(f'Memory Usage by {domain.capitalize()} Task Type')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Execution time
        ax3.bar(types, execution_times, color='salmon')
        ax3.set_ylabel('Execution Time (s)')
        ax3.set_title(f'Execution Time by {domain.capitalize()} Task Type')
        ax3.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Success rate
        ax4.bar(types, success_rates, color='gold')
        ax4.set_ylabel('Success Rate (%)')
        ax4.set_title(f'Success Rate by {domain.capitalize()} Task Type')
        ax4.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(self.results_dir / f"{domain}_task_types.png")
        plt.close()
        
        print(f"{domain.capitalize()} task type metrics plot saved to {self.results_dir / f'{domain}_task_types.png'}")

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Resource manager demo")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--workers", type=int, default=4, help="Maximum number of concurrent workers")
    parser.add_argument("--tasks", type=int, default=5, help="Number of tasks per type")
    parser.add_argument("--use-processes", action="store_true", help="Use processes instead of threads")
    parser.add_argument("--no-priority-queue", action="store_true", help="Disable priority queue")
    
    args = parser.parse_args()
    
    # Create demo
    demo = ResourceManagerDemo(
        config_path=args.config,
        max_workers=args.workers,
        use_processes=args.use_processes,
        enable_priority_queue=not args.no_priority_queue
    )
    
    # Initialize
    await demo.initialize()
    
    # Generate tasks
    tasks = await generate_mixed_tasks(count_per_type=args.tasks)
    print(f"Generated {len(tasks)} tasks")
    
    # Run standard concurrent execution
    standard_results = await demo.run_standard_concurrent(tasks)
    
    # Run resource-managed execution
    managed_results = await demo.run_resource_managed(tasks)
    
    # Run profile comparison
    profile_metrics = await demo.run_profile_comparison()
    
    # Compare results
    standard_duration = standard_results["metrics"]["total_duration"]
    managed_duration = managed_results["metrics"]["total_duration"]
    standard_success = standard_results["metrics"]["success_rate"]
    managed_success = managed_results["metrics"]["success_rate"]
    
    print("\nComparison results:")
    print(f"  Standard concurrent execution: {standard_duration:.3f}s, {standard_success*100:.1f}% success")
    print(f"  Resource-managed execution: {managed_duration:.3f}s, {managed_success*100:.1f}% success")
    
    if managed_duration < standard_duration:
        improvement = (standard_duration - managed_duration) / standard_duration * 100
        print(f"  Resource management improved performance by {improvement:.1f}%")
    else:
        degradation = (managed_duration - standard_duration) / standard_duration * 100
        print(f"  Resource management degraded performance by {degradation:.1f}%")
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())