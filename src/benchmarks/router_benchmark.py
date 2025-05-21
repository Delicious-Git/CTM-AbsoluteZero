#!/usr/bin/env python3
"""
Benchmark for the UniversalRouter in CTM-AbsoluteZero.

This script provides comprehensive benchmarking of the UniversalRouter's
performance, scalability, and reliability under various load scenarios.
"""
import os
import sys
import asyncio
import time
import json
import argparse
import logging
from typing import Dict, List, Any, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.router.universal_router import UniversalRouter, Task, SolverInterface
from src.utils.logging import get_logger, configure_logging
from src.utils.config import ConfigManager

# Setup logger
logger = get_logger("ctm-az.benchmarks.router")

# Domain test sets
DOMAINS = ["quantum", "maze", "sorting", "vision", "general"]

# Benchmark configurations
class BenchmarkConfig:
    """Configuration for router benchmarking."""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        num_tasks: int = 100,
        concurrent_tasks: int = 10,
        domains: Optional[List[str]] = None,
        task_complexity: str = "medium",
        output_dir: str = "./benchmark_results",
        timeout: float = 60.0,
        skip_plots: bool = False
    ):
        """
        Initialize benchmark configuration.
        
        Args:
            config_path: Path to router configuration file
            num_tasks: Total number of tasks to process
            concurrent_tasks: Number of concurrent tasks
            domains: List of domains to benchmark
            task_complexity: Task complexity (easy, medium, hard)
            output_dir: Directory to save results
            timeout: Timeout for each task in seconds
            skip_plots: Whether to skip generating plots
        """
        self.config_path = config_path
        self.num_tasks = num_tasks
        self.concurrent_tasks = concurrent_tasks
        self.domains = domains or DOMAINS
        self.task_complexity = task_complexity
        self.output_dir = output_dir
        self.timeout = timeout
        self.skip_plots = skip_plots

class MockSolver(SolverInterface):
    """Mock solver for benchmarking."""
    
    def __init__(
        self, 
        name: str, 
        domains: List[str],
        success_rate: float = 0.95,
        avg_duration: float = 0.2,
        duration_variance: float = 0.1,
        fail_fast: bool = False
    ):
        """
        Initialize mock solver.
        
        Args:
            name: Solver name
            domains: Domains this solver can handle
            success_rate: Success rate (0-1)
            avg_duration: Average task duration in seconds
            duration_variance: Variance in task duration
            fail_fast: Whether to fail immediately or after duration
        """
        super().__init__(name, {})
        self.domains = domains
        self.success_rate = success_rate
        self.avg_duration = avg_duration
        self.duration_variance = duration_variance
        self.fail_fast = fail_fast
    
    async def solve(self, task: Task) -> Dict[str, Any]:
        """Solve a task."""
        start_time = time.time()
        
        # Determine success/failure based on success rate
        success = np.random.random() < self.success_rate
        
        if not success and self.fail_fast:
            raise Exception(f"Task {task.task_id} failed (simulated failure)")
        
        # Calculate random duration based on config
        duration = max(0.01, np.random.normal(
            self.avg_duration, self.duration_variance
        ))
        
        # Simulate task execution time
        await asyncio.sleep(duration)
        
        if not success:
            raise Exception(f"Task {task.task_id} failed after {duration:.2f}s (simulated failure)")
        
        return {
            "status": "success",
            "result": {"completed": True, "domain": task.domain},
            "metrics": {
                "execution_time": duration,
                "efficiency": 0.9
            }
        }
    
    def can_solve(self, task: Task) -> bool:
        """Check if this solver can solve the given task."""
        return task.domain in self.domains
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get solver capabilities."""
        return {
            "domains": self.domains,
            "max_complexity": 10,
            "success_rate": self.success_rate
        }

class RouterBenchmark:
    """Benchmark for the UniversalRouter."""
    
    def __init__(self, config: BenchmarkConfig):
        """
        Initialize the benchmark.
        
        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.logger = get_logger("ctm-az.benchmarks.router")
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Load configuration
        config_manager = ConfigManager(config_path)
        router_config = config_manager.to_dict() if config_path else {}
        
        # Create router
        self.router = UniversalRouter(router_config)
        
        # Register solvers
        self._register_solvers()
        
        # Results
        self.results = {
            "config": {
                "num_tasks": config.num_tasks,
                "concurrent_tasks": config.concurrent_tasks,
                "domains": config.domains,
                "task_complexity": config.task_complexity
            },
            "performance": {
                "total_time": 0.0,
                "throughput": 0.0,
                "success_rate": 0.0,
                "avg_task_duration": 0.0,
                "avg_queue_time": 0.0,
                "domains": {}
            },
            "scalability": {
                "max_concurrent": 0,
                "avg_concurrent": 0.0,
                "resource_usage": {}
            },
            "tasks": []
        }
        
        for domain in config.domains:
            self.results["performance"]["domains"][domain] = {
                "success_rate": 0.0,
                "avg_duration": 0.0,
                "throughput": 0.0,
                "count": 0
            }
    
    def _register_solvers(self) -> None:
        """Register solvers for the benchmark."""
        # Create mock solvers for each domain
        for domain in self.config.domains:
            # High success rate solver
            solver_high = MockSolver(
                f"{domain}_high",
                [domain],
                success_rate=0.95,
                avg_duration=0.2,
                duration_variance=0.05
            )
            
            # Medium success rate solver
            solver_med = MockSolver(
                f"{domain}_med",
                [domain],
                success_rate=0.8,
                avg_duration=0.1,
                duration_variance=0.03
            )
            
            # Register solvers
            self.router.register_solver(solver_high, [domain])
            self.router.register_solver(solver_med, [domain])
        
        # General solver with lower success rate
        general_solver = MockSolver(
            "general",
            self.config.domains,
            success_rate=0.7,
            avg_duration=0.3,
            duration_variance=0.1
        )
        
        self.router.register_solver(general_solver, self.config.domains)
        
        self.logger.info(f"Registered {2 * len(self.config.domains) + 1} solvers")
    
    async def run(self) -> Dict[str, Any]:
        """
        Run the benchmark.
        
        Returns:
            Benchmark results
        """
        self.logger.info(f"Starting benchmark with {self.config.num_tasks} tasks")
        
        start_time = time.time()
        
        # Start router
        await self.router.start(num_workers=self.config.concurrent_tasks)
        
        # Create tasks
        tasks = self._create_tasks()
        
        # Submit all tasks
        self.logger.info(f"Submitting {len(tasks)} tasks")
        submission_start = time.time()
        
        for task in tasks:
            self.router.add_task(task)
        
        submission_time = time.time() - submission_start
        self.logger.info(f"Submitted {len(tasks)} tasks in {submission_time:.2f}s")
        
        # Wait for all tasks to complete
        await self.router.task_queue.join()
        
        # Stop router
        await self.router.stop()
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Update results
        self.results["performance"]["total_time"] = total_time
        self.results["performance"]["throughput"] = self.config.num_tasks / total_time
        
        # Get router stats
        stats = self.router.get_stats()
        
        # Update results with stats
        self.results["performance"]["success_rate"] = stats["successful_tasks"] / stats["total_tasks"] if stats["total_tasks"] > 0 else 0
        self.results["performance"]["avg_task_duration"] = stats["avg_duration"]
        
        # Update domain results
        for domain, domain_stats in stats["domains"].items():
            if domain in self.results["performance"]["domains"]:
                domain_count = domain_stats.get("total_tasks", 0)
                
                if domain_count > 0:
                    self.results["performance"]["domains"][domain]["count"] = domain_count
                    self.results["performance"]["domains"][domain]["success_rate"] = domain_stats.get("successful_tasks", 0) / domain_count
        
        # Update scalability results
        self.results["scalability"]["resource_usage"] = stats["resources"]
        
        # Generate plots
        if not self.config.skip_plots:
            self._generate_plots()
        
        # Save results
        self._save_results()
        
        self.logger.info(f"Benchmark completed in {total_time:.2f}s")
        self.logger.info(f"Throughput: {self.results['performance']['throughput']:.2f} tasks/s")
        self.logger.info(f"Success rate: {self.results['performance']['success_rate']:.1%}")
        
        return self.results
    
    def _create_tasks(self) -> List[Task]:
        """
        Create tasks for the benchmark.
        
        Returns:
            List of tasks
        """
        tasks = []
        
        # Distribute tasks across domains
        domain_distribution = {}
        
        # Calculate roughly equal distribution
        base_count = self.config.num_tasks // len(self.config.domains)
        remainder = self.config.num_tasks % len(self.config.domains)
        
        for i, domain in enumerate(self.config.domains):
            count = base_count + (1 if i < remainder else 0)
            domain_distribution[domain] = count
        
        # Create tasks for each domain
        for domain, count in domain_distribution.items():
            for i in range(count):
                task_id = f"{domain}_{i}"
                task = Task(
                    task_id=task_id,
                    domain=domain,
                    description=f"Benchmark task for {domain}",
                    parameters=self._get_parameters(domain),
                    priority=np.random.randint(0, 100),
                    deadline=time.time() + 60
                )
                tasks.append(task)
        
        # Shuffle tasks for realistic mixed workload
        np.random.shuffle(tasks)
        
        return tasks
    
    def _get_parameters(self, domain: str) -> Dict[str, Any]:
        """
        Get domain-specific parameters.
        
        Args:
            domain: Task domain
            
        Returns:
            Task parameters
        """
        if domain == "quantum":
            return {
                "num_qubits": np.random.randint(2, 8),
                "algorithm": np.random.choice(["grover", "shor", "vqe"]),
                "circuit_depth": np.random.randint(2, 10)
            }
        elif domain == "maze":
            size = np.random.randint(5, 20)
            return {
                "size": [size, size],
                "complexity": np.random.uniform(0.1, 0.9)
            }
        elif domain == "sorting":
            return {
                "array_size": np.random.randint(100, 10000),
                "algorithm": np.random.choice(["quick", "merge", "heap", "bubble"])
            }
        elif domain == "vision":
            return {
                "image_size": [np.random.randint(100, 1000), np.random.randint(100, 1000)],
                "channels": np.random.choice([1, 3]),
                "model_type": np.random.choice(["cnn", "transformer", "hybrid"])
            }
        else:  # general
            return {
                "complexity": np.random.uniform(0.1, 0.9),
                "input_size": np.random.randint(10, 1000)
            }
    
    def _generate_plots(self) -> None:
        """Generate benchmark plots."""
        plots_dir = os.path.join(self.config.output_dir, "router_plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot throughput
        self._plot_throughput(plots_dir)
        
        # Plot success rate by domain
        self._plot_success_rate(plots_dir)
        
        # Plot resource usage
        self._plot_resource_usage(plots_dir)
    
    def _plot_throughput(self, plots_dir: str) -> None:
        """
        Plot throughput results.
        
        Args:
            plots_dir: Directory to save plots
        """
        plt.figure(figsize=(10, 6))
        
        # Get domain names and throughput values
        domains = list(self.results["performance"]["domains"].keys())
        throughputs = [
            self.results["performance"]["domains"][domain]["count"] / self.results["performance"]["total_time"]
            for domain in domains
        ]
        
        # Create bar chart
        plt.bar(domains, throughputs)
        plt.axhline(
            y=self.results["performance"]["throughput"],
            color='r',
            linestyle='-',
            label=f'Overall: {self.results["performance"]["throughput"]:.2f} tasks/s'
        )
        
        plt.xlabel('Domain')
        plt.ylabel('Throughput (tasks/s)')
        plt.title('Router Throughput by Domain')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Save plot
        plt.savefig(os.path.join(plots_dir, "throughput.png"))
        plt.close()
    
    def _plot_success_rate(self, plots_dir: str) -> None:
        """
        Plot success rate by domain.
        
        Args:
            plots_dir: Directory to save plots
        """
        plt.figure(figsize=(10, 6))
        
        # Get domain names and success rates
        domains = list(self.results["performance"]["domains"].keys())
        success_rates = [
            self.results["performance"]["domains"][domain]["success_rate"]
            for domain in domains
        ]
        
        # Create bar chart
        plt.bar(domains, success_rates)
        plt.axhline(
            y=self.results["performance"]["success_rate"],
            color='r',
            linestyle='-',
            label=f'Overall: {self.results["performance"]["success_rate"]:.1%}'
        )
        
        plt.xlabel('Domain')
        plt.ylabel('Success Rate')
        plt.title('Router Success Rate by Domain')
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Save plot
        plt.savefig(os.path.join(plots_dir, "success_rate.png"))
        plt.close()
    
    def _plot_resource_usage(self, plots_dir: str) -> None:
        """
        Plot resource usage.
        
        Args:
            plots_dir: Directory to save plots
        """
        plt.figure(figsize=(10, 6))
        
        resources = self.results["scalability"]["resource_usage"]
        
        # Get resource types
        resource_types = ["active_tasks", "memory_usage", "storage_usage", "api_calls"]
        
        # Calculate percentages
        percentages = []
        labels = []
        
        for resource in resource_types:
            if resource in resources and f"max_{resource}" in resources["limits"]:
                value = resources[resource]
                max_value = resources["limits"][f"max_{resource}"]
                
                if max_value > 0:
                    percentage = value / max_value
                    percentages.append(percentage)
                    labels.append(resource.replace("_", " ").title())
        
        # Create bar chart
        plt.bar(labels, percentages)
        
        plt.xlabel('Resource Type')
        plt.ylabel('Usage (% of maximum)')
        plt.title('Resource Usage')
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        
        # Add values on top of bars
        for i, v in enumerate(percentages):
            plt.text(i, v + 0.02, f"{v:.1%}", ha='center')
        
        # Save plot
        plt.savefig(os.path.join(plots_dir, "resource_usage.png"))
        plt.close()
    
    def _save_results(self) -> None:
        """Save benchmark results to file."""
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results file path
        results_file = os.path.join(
            self.config.output_dir,
            f"router_benchmark_{timestamp}.json"
        )
        
        # Save as JSON
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.logger.info(f"Results saved to {results_file}")
        
        # Create summary report
        summary_file = os.path.join(
            self.config.output_dir,
            f"router_benchmark_summary_{timestamp}.txt"
        )
        
        with open(summary_file, 'w') as f:
            f.write("=== CTM-AbsoluteZero Router Benchmark ===\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Tasks: {self.config.num_tasks}\n")
            f.write(f"Concurrent tasks: {self.config.concurrent_tasks}\n")
            f.write(f"Domains: {', '.join(self.config.domains)}\n\n")
            
            f.write("=== PERFORMANCE ===\n\n")
            f.write(f"Total time: {self.results['performance']['total_time']:.2f}s\n")
            f.write(f"Throughput: {self.results['performance']['throughput']:.2f} tasks/s\n")
            f.write(f"Success rate: {self.results['performance']['success_rate']:.1%}\n")
            f.write(f"Average task duration: {self.results['performance']['avg_task_duration']:.3f}s\n\n")
            
            f.write("Domain performance:\n")
            for domain, stats in self.results["performance"]["domains"].items():
                count = stats.get("count", 0)
                success_rate = stats.get("success_rate", 0)
                f.write(f"  {domain}: {count} tasks, {success_rate:.1%} success rate\n")
            
            f.write("\n=== RESOURCE USAGE ===\n\n")
            f.write(f"Active tasks: {self.results['scalability']['resource_usage'].get('active_tasks', 0)}\n")
            
            # Add resource usage if available
            resources = self.results["scalability"]["resource_usage"]
            for resource, value in resources.items():
                if resource != "limits" and resource != "active_tasks":
                    f.write(f"{resource.replace('_', ' ').title()}: {value}\n")
            
            f.write("\n=== CONCLUSION ===\n\n")
            
            # Add conclusion based on throughput
            throughput = self.results["performance"]["throughput"]
            if throughput > 100:
                f.write("The router has excellent throughput capability, handling more than 100 tasks per second.\n")
            elif throughput > 50:
                f.write("The router has good throughput capability, handling more than 50 tasks per second.\n")
            elif throughput > 10:
                f.write("The router has moderate throughput capability, handling more than 10 tasks per second.\n")
            else:
                f.write("The router has limited throughput capability, handling fewer than 10 tasks per second.\n")
            
            # Add conclusion based on success rate
            success_rate = self.results["performance"]["success_rate"]
            if success_rate > 0.95:
                f.write("The router has excellent reliability, with a success rate above 95%.\n")
            elif success_rate > 0.9:
                f.write("The router has good reliability, with a success rate above 90%.\n")
            elif success_rate > 0.8:
                f.write("The router has moderate reliability, with a success rate above 80%.\n")
            else:
                f.write("The router has limited reliability, with a success rate below 80%.\n")
            
            # Add recommendation
            f.write("\nRecommendation: ")
            if success_rate > 0.9 and throughput > 50:
                f.write("The router is performing excellently and is suitable for production use.\n")
            elif success_rate > 0.8 and throughput > 10:
                f.write("The router is performing well but could benefit from optimization of the task execution pipeline.\n")
            else:
                f.write("The router requires further optimization before production use.\n")
        
        self.logger.info(f"Summary saved to {summary_file}")
        
        # Create latest link
        latest_link = os.path.join(self.config.output_dir, "latest_router_benchmark.json")
        try:
            if os.path.exists(latest_link):
                os.remove(latest_link)
            os.symlink(results_file, latest_link)
        except Exception as e:
            self.logger.warning(f"Failed to create latest link: {e}")

async def main():
    """Run benchmark from command line."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="UniversalRouter Benchmark")
    
    # Add arguments
    parser.add_argument(
        "--config", "-c",
        help="Path to router configuration file",
        type=str
    )
    parser.add_argument(
        "--num-tasks", "-n",
        help="Number of tasks to process",
        type=int,
        default=100
    )
    parser.add_argument(
        "--concurrent", "-p",
        help="Number of concurrent tasks",
        type=int,
        default=10
    )
    parser.add_argument(
        "--domains", "-d",
        help="Comma-separated list of domains to benchmark",
        type=str
    )
    parser.add_argument(
        "--complexity",
        help="Task complexity (easy, medium, hard)",
        choices=["easy", "medium", "hard"],
        default="medium"
    )
    parser.add_argument(
        "--output-dir", "-o",
        help="Directory to save results",
        type=str,
        default="./benchmark_results"
    )
    parser.add_argument(
        "--timeout", "-t",
        help="Timeout for each task in seconds",
        type=float,
        default=60.0
    )
    parser.add_argument(
        "--no-plots",
        help="Don't generate plots",
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
    
    # Parse domains
    domains = args.domains.split(",") if args.domains else None
    
    # Create benchmark config
    config = BenchmarkConfig(
        config_path=args.config,
        num_tasks=args.num_tasks,
        concurrent_tasks=args.concurrent,
        domains=domains,
        task_complexity=args.complexity,
        output_dir=args.output_dir,
        timeout=args.timeout,
        skip_plots=args.no_plots
    )
    
    # Run benchmark
    benchmark = RouterBenchmark(config)
    await benchmark.run()

if __name__ == "__main__":
    asyncio.run(main())