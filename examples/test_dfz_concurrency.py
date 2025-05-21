"""
Test script for DFZ integration with concurrent tasks in CTM-AbsoluteZero
"""
import sys
import os
import asyncio
import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import CTM and DFZ components
from src.integration.dfz_concurrent import ConcurrentDFZAdapter, ConcurrentCTMAbsoluteZeroPlugin
from src.ctm.quantum_sim import QuantumSimulator
from src.ctm_az_agent import CTM_AbsoluteZero_Agent
from src.utils.config import ConfigManager

class ConcurrentTaskTester:
    """
    Test harness for DFZ integration with concurrent task execution.
    """
    def __init__(self, config_path=None, standalone=True, max_workers=4, use_processes=False):
        """
        Initialize the concurrent task tester.
        
        Args:
            config_path: Path to configuration file
            standalone: Whether to operate in standalone mode (without DFZ)
            max_workers: Maximum number of concurrent workers
            use_processes: Use process-based parallelism instead of threads
        """
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "default.yaml"
        )
        self.standalone = standalone
        self.max_workers = max_workers
        self.use_processes = use_processes
        
        # Load configuration
        self.config_manager = ConfigManager(self.config_path)
        self.config = self.config_manager.config
        
        # Initialize DFZ adapter
        self.dfz_adapter = None
        
        # Results storage
        self.results = {
            "sequential": {},
            "concurrent": {},
            "comparison": {}
        }
        
        # Create results directory
        Path("results").mkdir(exist_ok=True)
    
    async def initialize(self):
        """
        Initialize the DFZ adapter and other components.
        
        Returns:
            True if initialization was successful
        """
        print("Initializing Concurrent DFZ adapter...")
        
        # Define DFZ path - in standalone mode, this is None
        dfz_path = None if self.standalone else os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "evolution"
        )
        
        # Create Concurrent DFZ adapter
        self.dfz_adapter = ConcurrentDFZAdapter(
            dfz_path=dfz_path,
            config={"config_path": self.config_path},
            max_workers=self.max_workers,
            use_processes=self.use_processes
        )
        
        # Initialize the adapter
        success = await self.dfz_adapter.initialize()
        if success:
            print(f"Concurrent DFZ adapter initialized successfully with {self.max_workers} workers")
        else:
            print("Failed to initialize Concurrent DFZ adapter")
            
        return success
    
    async def generate_test_tasks(self, domain="quantum", count=10):
        """
        Generate test tasks for benchmarking.
        
        Args:
            domain: Task domain
            count: Number of tasks to generate
            
        Returns:
            List of generated tasks
        """
        print(f"Generating {count} test tasks for domain: {domain}...")
        
        tasks = []
        for i in range(count):
            # Generate task parameters based on domain
            if domain == "quantum":
                task = {
                    "id": f"task_{i+1}",
                    "domain": "quantum",
                    "description": f"Quantum task {i+1}",
                    "parameters": {
                        "algorithm": np.random.choice(["vqe", "grover", "qft"]),
                        "num_qubits": np.random.randint(2, 8),
                        "noise_level": np.random.uniform(0.01, 0.1),
                        "circuit_depth": np.random.randint(2, 10)
                    }
                }
            elif domain == "maze":
                task = {
                    "id": f"task_{i+1}",
                    "domain": "maze",
                    "description": f"Maze task {i+1}",
                    "parameters": {
                        "size_x": np.random.randint(5, 15),
                        "size_y": np.random.randint(5, 15),
                        "complexity": np.random.uniform(0.3, 0.8),
                        "seed": np.random.randint(1, 1000)
                    }
                }
            else:
                # Default generic task
                task = {
                    "id": f"task_{i+1}",
                    "domain": domain,
                    "description": f"{domain.capitalize()} task {i+1}",
                    "parameters": {
                        "difficulty": np.random.choice(["easy", "medium", "hard"]),
                        "seed": np.random.randint(1, 1000)
                    }
                }
            
            tasks.append(task)
            
        print(f"Generated {len(tasks)} tasks")
        return tasks
    
    async def execute_sequential(self, tasks, context=None):
        """
        Execute tasks sequentially.
        
        Args:
            tasks: List of tasks to execute
            context: Execution context
            
        Returns:
            Execution results
        """
        print("Executing tasks sequentially...")
        
        results = []
        start_time = time.time()
        
        for task in tasks:
            task_start = time.time()
            result = await self.dfz_adapter.execute_task(task, context)
            task_duration = time.time() - task_start
            
            result["execution_time"] = task_duration
            results.append(result)
            
            print(f"  Task {task['id']} executed in {task_duration:.3f}s")
        
        total_duration = time.time() - start_time
        
        # Collect metrics
        metrics = {
            "total_duration": total_duration,
            "avg_task_duration": total_duration / len(tasks),
            "task_durations": [r["execution_time"] for r in results],
            "success_rate": sum(1 for r in results if r["status"] == "success") / len(results)
        }
        
        print(f"Sequential execution completed in {total_duration:.3f}s")
        print(f"Average task duration: {metrics['avg_task_duration']:.3f}s")
        print(f"Success rate: {metrics['success_rate'] * 100:.1f}%")
        
        return {
            "results": results,
            "metrics": metrics
        }
    
    async def execute_concurrent(self, tasks, context=None):
        """
        Execute tasks concurrently using the ConcurrentDFZAdapter.
        
        Args:
            tasks: List of tasks to execute
            context: Execution context
            
        Returns:
            Execution results
        """
        print(f"Executing tasks concurrently with {self.max_workers} workers...")
        
        start_time = time.time()
        
        # Use the batch execution method from ConcurrentDFZAdapter
        batch_result = await self.dfz_adapter.execute_tasks_batch(
            tasks, context, wait=True
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
        
        print(f"Concurrent execution completed in {total_duration:.3f}s")
        print(f"Average task duration: {metrics['avg_task_duration']:.3f}s")
        print(f"Success rate: {metrics['success_rate'] * 100:.1f}%")
        
        return {
            "results": results,
            "metrics": metrics
        }
    
    async def run_benchmark(self, domain="quantum", task_count=10):
        """
        Run a complete benchmark comparing sequential vs concurrent execution.
        
        Args:
            domain: Task domain
            task_count: Number of tasks to generate
            
        Returns:
            Benchmark results
        """
        print(f"Running benchmark for {domain} domain with {task_count} tasks...")
        
        # Generate test tasks
        tasks = await self.generate_test_tasks(domain, task_count)
        
        # Run sequential execution
        sequential_results = await self.execute_sequential(tasks)
        self.results["sequential"][domain] = sequential_results
        
        # Run concurrent execution
        concurrent_results = await self.execute_concurrent(tasks)
        self.results["concurrent"][domain] = concurrent_results
        
        # Compare results
        speedup = sequential_results["metrics"]["total_duration"] / concurrent_results["metrics"]["total_duration"]
        
        comparison = {
            "speedup": speedup,
            "sequential_duration": sequential_results["metrics"]["total_duration"],
            "concurrent_duration": concurrent_results["metrics"]["total_duration"],
            "sequential_success_rate": sequential_results["metrics"]["success_rate"],
            "concurrent_success_rate": concurrent_results["metrics"]["success_rate"]
        }
        
        self.results["comparison"][domain] = comparison
        
        print(f"Benchmark completed with {speedup:.2f}x speedup")
        return self.results
    
    async def run_concurrency_scaling_test(self, domain="quantum", task_count=20, max_workers_list=None):
        """
        Run a test to measure performance scaling with different numbers of concurrent workers.
        
        Args:
            domain: Task domain
            task_count: Number of tasks to generate
            max_workers_list: List of worker counts to test
            
        Returns:
            Scaling test results
        """
        if max_workers_list is None:
            max_workers_list = [1, 2, 4, 8, 16]
        
        print(f"Running concurrency scaling test for {domain} domain...")
        print(f"Testing with worker counts: {max_workers_list}")
        
        # Generate test tasks once
        tasks = await self.generate_test_tasks(domain, task_count)
        
        # Results for different worker counts
        scaling_results = {}
        
        # Run sequential test first (equivalent to 1 worker)
        sequential_results = await self.execute_sequential(tasks)
        scaling_results[1] = {
            "total_duration": sequential_results["metrics"]["total_duration"],
            "success_rate": sequential_results["metrics"]["success_rate"],
            "method": "sequential"
        }
        
        # Skip 1 in max_workers_list since we already tested it
        for worker_count in [w for w in max_workers_list if w > 1]:
            print(f"\nTesting with {worker_count} workers...")
            
            # Create a new tester with the specified worker count
            tester = ConcurrentTaskTester(
                config_path=self.config_path,
                standalone=self.standalone,
                max_workers=worker_count,
                use_processes=self.use_processes
            )
            await tester.initialize()
            
            # Run concurrent test
            concurrent_results = await tester.execute_concurrent(tasks)
            
            scaling_results[worker_count] = {
                "total_duration": concurrent_results["metrics"]["total_duration"],
                "success_rate": concurrent_results["metrics"]["success_rate"],
                "method": "concurrent"
            }
        
        # Plot results
        self._plot_scaling_results(scaling_results, domain)
        
        return scaling_results
    
    def _plot_scaling_results(self, scaling_results, domain):
        """
        Plot the results of the concurrency scaling test.
        
        Args:
            scaling_results: Scaling test results
            domain: Task domain
        """
        worker_counts = sorted(list(scaling_results.keys()))
        durations = [scaling_results[w]["total_duration"] for w in worker_counts]
        success_rates = [scaling_results[w]["success_rate"] * 100 for w in worker_counts]
        
        # Calculate speedup compared to single worker
        single_worker_time = scaling_results[min(worker_counts)]["total_duration"]
        speedups = [single_worker_time / duration for duration in durations]
        
        # Calculate theoretical speedup (Amdahl's Law with 90% parallelizable)
        theoretical_speedups = [1 / (0.1 + 0.9/w) for w in worker_counts]
        
        # Plot execution time and speedup
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Execution time plot
        ax1.plot(worker_counts, durations, marker='o', linewidth=2, label='Actual')
        
        # Add ideal scaling line (T = T1/n)
        ideal_durations = [durations[0] / w for w in worker_counts]
        ax1.plot(worker_counts, ideal_durations, 'k--', linewidth=1, label='Ideal')
        
        ax1.set_xlabel('Number of Workers')
        ax1.set_ylabel('Total Execution Time (s)')
        ax1.set_title(f'Execution Time vs Workers ({domain.capitalize()} Tasks)')
        ax1.grid(True)
        ax1.legend()
        
        # Speedup plot
        ax2.plot(worker_counts, speedups, marker='s', color='green', linewidth=2, label='Actual')
        ax2.plot(worker_counts, theoretical_speedups, 'k--', linewidth=1, label='Theoretical (90% parallel)')
        ax2.plot(worker_counts, worker_counts, 'r:', linewidth=1, label='Linear')
        
        ax2.set_xlabel('Number of Workers')
        ax2.set_ylabel('Speedup Factor')
        ax2.set_title(f'Speedup vs Workers ({domain.capitalize()} Tasks)')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f"results/scaling_{domain}_tasks.png")
        
        # Plot success rate
        plt.figure(figsize=(10, 6))
        plt.plot(worker_counts, success_rates, marker='^', color='orange', linewidth=2)
        plt.xlabel('Number of Workers')
        plt.ylabel('Success Rate (%)')
        plt.title(f'Success Rate vs Workers ({domain.capitalize()} Tasks)')
        plt.grid(True)
        plt.savefig(f"results/success_rate_{domain}_tasks.png")
    
    def save_results(self, filename="dfz_concurrent_results.json"):
        """
        Save benchmark results to a JSON file.
        
        Args:
            filename: Output filename
        """
        output_path = os.path.join("results", filename)
        
        # Convert any non-serializable objects
        def clean_for_json(obj):
            if isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
                return float(obj)
            return obj
        
        # Clean results
        clean_results = json.loads(json.dumps(self.results, default=clean_for_json))
        
        with open(output_path, 'w') as f:
            json.dump(clean_results, f, indent=2)
            
        print(f"Results saved to {output_path}")

async def main():
    """Main function to run the DFZ concurrent task tests"""
    parser = argparse.ArgumentParser(description="Test DFZ integration with concurrent tasks")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--domain", type=str, default="quantum", help="Task domain (quantum, maze, etc.)")
    parser.add_argument("--count", type=int, default=10, help="Number of tasks to generate")
    parser.add_argument("--workers", type=int, default=4, help="Maximum number of concurrent workers")
    parser.add_argument("--use-processes", action="store_true", help="Use processes instead of threads")
    parser.add_argument("--scaling-test", action="store_true", help="Run concurrency scaling test")
    parser.add_argument("--max-workers", type=str, default="1,2,4,8,16", help="Comma-separated list of worker counts for scaling test")
    
    args = parser.parse_args()
    
    # Create tester
    tester = ConcurrentTaskTester(
        config_path=args.config,
        max_workers=args.workers,
        use_processes=args.use_processes
    )
    
    # Initialize
    await tester.initialize()
    
    if args.scaling_test:
        # Parse worker counts for scaling test
        max_workers_list = [int(w) for w in args.max_workers.split(',')]
        
        # Run scaling test
        scaling_results = await tester.run_concurrency_scaling_test(
            domain=args.domain,
            task_count=args.count,
            max_workers_list=max_workers_list
        )
    else:
        # Run benchmark
        benchmark_results = await tester.run_benchmark(
            domain=args.domain,
            task_count=args.count
        )
    
    # Save results
    tester.save_results()

if __name__ == "__main__":
    asyncio.run(main())