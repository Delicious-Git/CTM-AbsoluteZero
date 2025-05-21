"""
Example script for testing DFZ integration with mixed concurrent tasks.
This script demonstrates executing a mix of quantum, sorting, and image analysis tasks concurrently.
"""
import sys
import os
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

# Import CTM and DFZ components
from src.integration.dfz_concurrent import ConcurrentDFZAdapter
from src.ctm.quantum_sim import QuantumSimulator
from src.utils.config import ConfigManager

def generate_mixed_tasks(count_per_type=5):
    """
    Generate a mix of quantum, sorting, and image analysis tasks.
    
    Args:
        count_per_type: Number of tasks per type
        
    Returns:
        List of mixed tasks
    """
    tasks = []
    
    # Generate quantum tasks
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
        tasks.append(task)
    
    # Generate sorting tasks
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
        tasks.append(task)
    
    # Generate image analysis tasks
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
        tasks.append(task)
    
    # Shuffle tasks to mix them
    random.shuffle(tasks)
    
    return tasks

async def test_sequential_execution(tasks, adapter):
    """
    Execute tasks sequentially.
    
    Args:
        tasks: List of tasks to execute
        adapter: DFZ adapter
        
    Returns:
        Dictionary with execution results and metrics
    """
    print(f"Executing {len(tasks)} tasks sequentially...")
    
    results = []
    start_time = time.time()
    
    for task in tasks:
        task_start = time.time()
        result = await adapter.execute_task(task)
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

async def test_concurrent_execution(tasks, adapter):
    """
    Execute tasks concurrently.
    
    Args:
        tasks: List of tasks to execute
        adapter: DFZ adapter
        
    Returns:
        Dictionary with execution results and metrics
    """
    print(f"Executing {len(tasks)} tasks concurrently...")
    
    start_time = time.time()
    
    # Use the batch execution method
    batch_result = await adapter.execute_tasks_batch(
        tasks, wait=True
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
        "concurrent_workers": adapter.plugin.task_manager.max_workers,
    }
    
    # Add concurrent metrics from the adapter
    adapter_metrics = adapter.get_performance_metrics()
    if "concurrent" in adapter_metrics:
        metrics.update(adapter_metrics["concurrent"])
    
    print(f"Concurrent execution completed in {total_duration:.3f}s")
    print(f"Average task duration: {metrics['avg_task_duration']:.3f}s")
    print(f"Success rate: {metrics['success_rate'] * 100:.1f}%")
    
    return {
        "results": results,
        "metrics": metrics
    }

async def run_mixed_tasks_test(config_path=None, count_per_type=5, max_workers=4, use_processes=False):
    """
    Run test with mixed concurrent tasks.
    
    Args:
        config_path: Path to configuration file
        count_per_type: Number of tasks per type
        max_workers: Maximum number of concurrent workers
        use_processes: Use processes instead of threads
        
    Returns:
        Test results
    """
    print(f"Running mixed tasks test with {count_per_type} tasks per type...")
    
    # Initialize adapter
    adapter = ConcurrentDFZAdapter(
        config={"config_path": config_path},
        max_workers=max_workers,
        use_processes=use_processes
    )
    
    # Initialize the adapter
    await adapter.initialize()
    print(f"Concurrent DFZ adapter initialized with {max_workers} workers")
    
    # Generate mixed tasks
    tasks = generate_mixed_tasks(count_per_type)
    print(f"Generated {len(tasks)} mixed tasks")
    
    # Run sequential execution
    sequential_results = await test_sequential_execution(tasks, adapter)
    
    # Run concurrent execution
    concurrent_results = await test_concurrent_execution(tasks, adapter)
    
    # Calculate speedup
    speedup = sequential_results["metrics"]["total_duration"] / concurrent_results["metrics"]["total_duration"]
    
    # Compare results
    comparison = {
        "speedup": speedup,
        "sequential_duration": sequential_results["metrics"]["total_duration"],
        "concurrent_duration": concurrent_results["metrics"]["total_duration"],
        "sequential_success_rate": sequential_results["metrics"]["success_rate"],
        "concurrent_success_rate": concurrent_results["metrics"]["success_rate"]
    }
    
    print(f"\nComparison results:")
    print(f"- Speedup: {speedup:.2f}x")
    print(f"- Sequential duration: {comparison['sequential_duration']:.3f}s")
    print(f"- Concurrent duration: {comparison['concurrent_duration']:.3f}s")
    print(f"- Sequential success rate: {comparison['sequential_success_rate']*100:.1f}%")
    print(f"- Concurrent success rate: {comparison['concurrent_success_rate']*100:.1f}%")
    
    # Group results by domain for analysis
    domain_results = {}
    
    # Process sequential results
    for result in sequential_results["results"]:
        task_id = result["task_id"]
        task = next((t for t in tasks if t["id"] == task_id), None)
        if task:
            domain = task["domain"]
            if domain not in domain_results:
                domain_results[domain] = {"sequential": [], "concurrent": []}
            domain_results[domain]["sequential"].append(result)
    
    # Process concurrent results
    for result in concurrent_results["results"]:
        task_id = result["task_id"]
        task = next((t for t in tasks if t["id"] == task_id), None)
        if task:
            domain = task["domain"]
            if domain not in domain_results:
                domain_results[domain] = {"sequential": [], "concurrent": []}
            domain_results[domain]["concurrent"].append(result)
    
    # Calculate domain-specific metrics
    domain_metrics = {}
    for domain, results in domain_results.items():
        seq_durations = [r.get("execution_time", 0) for r in results["sequential"]]
        conc_durations = [r.get("duration", 0) for r in results["concurrent"]]
        
        domain_metrics[domain] = {
            "sequential_avg_duration": sum(seq_durations) / len(seq_durations),
            "concurrent_avg_duration": sum(conc_durations) / len(conc_durations),
            "speedup": sum(seq_durations) / sum(conc_durations) if sum(conc_durations) > 0 else 0
        }
    
    print("\nDomain-specific results:")
    for domain, metrics in domain_metrics.items():
        print(f"- {domain}:")
        print(f"  - Sequential avg duration: {metrics['sequential_avg_duration']:.3f}s")
        print(f"  - Concurrent avg duration: {metrics['concurrent_avg_duration']:.3f}s")
        print(f"  - Speedup: {metrics['speedup']:.2f}x")
    
    # Plot results
    plot_results(
        sequential_results,
        concurrent_results,
        domain_metrics,
        max_workers
    )
    
    return {
        "sequential": sequential_results,
        "concurrent": concurrent_results,
        "comparison": comparison,
        "domain_metrics": domain_metrics
    }

def plot_results(sequential_results, concurrent_results, domain_metrics, max_workers):
    """
    Plot test results.
    
    Args:
        sequential_results: Sequential execution results
        concurrent_results: Concurrent execution results
        domain_metrics: Domain-specific metrics
        max_workers: Maximum number of concurrent workers
    """
    # Create output directory
    output_dir = Path("results/mixed_tasks")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Overall execution time comparison
    plt.figure(figsize=(10, 6))
    labels = ["Sequential", f"Concurrent ({max_workers} workers)"]
    durations = [
        sequential_results["metrics"]["total_duration"],
        concurrent_results["metrics"]["total_duration"]
    ]
    
    plt.bar(labels, durations, color=["blue", "green"])
    plt.ylabel("Execution Time (s)")
    plt.title("Mixed Tasks: Sequential vs Concurrent Execution Time")
    
    # Add text labels
    for i, duration in enumerate(durations):
        plt.text(i, duration + 0.1, f"{duration:.2f}s", ha="center")
    
    # Add speedup annotation
    speedup = sequential_results["metrics"]["total_duration"] / concurrent_results["metrics"]["total_duration"]
    plt.annotate(
        f"Speedup: {speedup:.2f}x",
        xy=(0.5, max(durations) * 0.5),
        xytext=(0.5, max(durations) * 0.7),
        textcoords="data",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round", fc="yellow", alpha=0.7),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3")
    )
    
    plt.tight_layout()
    plt.savefig(output_dir / "overall_execution_time.png")
    plt.close()
    
    # Plot 2: Domain-specific speedup
    plt.figure(figsize=(12, 6))
    domains = list(domain_metrics.keys())
    speedups = [domain_metrics[d]["speedup"] for d in domains]
    
    plt.bar(domains, speedups, color="green")
    plt.ylabel("Speedup Factor")
    plt.title(f"Domain-Specific Speedup with {max_workers} Workers")
    
    # Add text labels
    for i, speedup in enumerate(speedups):
        plt.text(i, speedup + 0.1, f"{speedup:.2f}x", ha="center")
    
    # Add horizontal line at y=1 (no speedup)
    plt.axhline(y=1, color="red", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / "domain_speedup.png")
    plt.close()
    
    # Plot 3: Domain-specific execution times
    plt.figure(figsize=(14, 7))
    domain_names = list(domain_metrics.keys())
    seq_times = [domain_metrics[d]["sequential_avg_duration"] for d in domain_names]
    conc_times = [domain_metrics[d]["concurrent_avg_duration"] for d in domain_names]
    
    x = range(len(domain_names))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], seq_times, width, label="Sequential", color="blue")
    plt.bar([i + width/2 for i in x], conc_times, width, label="Concurrent", color="green")
    
    plt.ylabel("Average Task Duration (s)")
    plt.title("Domain-Specific Average Task Duration")
    plt.xticks(x, domain_names)
    plt.legend()
    
    # Add text labels
    for i, (seq, conc) in enumerate(zip(seq_times, conc_times)):
        plt.text(i - width/2, seq + 0.05, f"{seq:.2f}s", ha="center", va="bottom")
        plt.text(i + width/2, conc + 0.05, f"{conc:.2f}s", ha="center", va="bottom")
    
    plt.tight_layout()
    plt.savefig(output_dir / "domain_execution_times.png")
    plt.close()
    
    # Plot 4: Task duration distribution
    plt.figure(figsize=(12, 6))
    
    # Convert to arrays for easier manipulation
    seq_durations = np.array(sequential_results["metrics"]["task_durations"])
    conc_durations = np.array(concurrent_results["metrics"]["task_durations"])
    
    # Create histogram
    bins = np.linspace(0, max(np.max(seq_durations), np.max(conc_durations)) * 1.1, 20)
    plt.hist(seq_durations, bins=bins, alpha=0.5, label="Sequential", color="blue")
    plt.hist(conc_durations, bins=bins, alpha=0.5, label="Concurrent", color="green")
    
    plt.xlabel("Task Duration (s)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Task Durations")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "task_duration_distribution.png")
    plt.close()
    
    print(f"Plots saved to {output_dir}/")

async def run_concurrency_scaling_test(config_path=None, count_per_type=5, max_workers_list=None):
    """
    Run test to measure performance scaling with different numbers of workers.
    
    Args:
        config_path: Path to configuration file
        count_per_type: Number of tasks per type
        max_workers_list: List of worker counts to test
        
    Returns:
        Scaling test results
    """
    if max_workers_list is None:
        max_workers_list = [1, 2, 4, 8]
    
    print(f"Running concurrency scaling test with {count_per_type} tasks per type...")
    print(f"Testing with worker counts: {max_workers_list}")
    
    # Generate mixed tasks once
    tasks = generate_mixed_tasks(count_per_type)
    print(f"Generated {len(tasks)} mixed tasks")
    
    # Run sequential test first
    adapter = ConcurrentDFZAdapter(
        config={"config_path": config_path},
        max_workers=1
    )
    await adapter.initialize()
    sequential_results = await test_sequential_execution(tasks, adapter)
    
    # Results for different worker counts
    scaling_results = {
        1: {
            "total_duration": sequential_results["metrics"]["total_duration"],
            "success_rate": sequential_results["metrics"]["success_rate"],
            "method": "sequential"
        }
    }
    
    # Test with different worker counts
    for worker_count in [w for w in max_workers_list if w > 1]:
        print(f"\nTesting with {worker_count} workers...")
        
        # Create a new adapter with the specified worker count
        adapter = ConcurrentDFZAdapter(
            config={"config_path": config_path},
            max_workers=worker_count
        )
        await adapter.initialize()
        
        # Run concurrent test
        concurrent_results = await test_concurrent_execution(tasks, adapter)
        
        scaling_results[worker_count] = {
            "total_duration": concurrent_results["metrics"]["total_duration"],
            "success_rate": concurrent_results["metrics"]["success_rate"],
            "method": "concurrent"
        }
    
    # Plot scaling results
    plot_scaling_results(scaling_results, tasks)
    
    return scaling_results

def plot_scaling_results(scaling_results, tasks):
    """
    Plot the results of the concurrency scaling test.
    
    Args:
        scaling_results: Scaling test results
        tasks: List of tasks
    """
    # Create output directory
    output_dir = Path("results/scaling_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    worker_counts = sorted(list(scaling_results.keys()))
    durations = [scaling_results[w]["total_duration"] for w in worker_counts]
    success_rates = [scaling_results[w]["success_rate"] * 100 for w in worker_counts]
    
    # Calculate speedup compared to single worker
    single_worker_time = scaling_results[min(worker_counts)]["total_duration"]
    speedups = [single_worker_time / duration for duration in durations]
    
    # Calculate theoretical speedup (Amdahl's Law with 90% parallelizable)
    theoretical_speedups = [1 / (0.1 + 0.9/w) for w in worker_counts]
    
    # Calculate ideal speedup (linear)
    ideal_speedups = [float(w) for w in worker_counts]
    
    # Plot 1: Execution time vs Workers
    plt.figure(figsize=(12, 6))
    plt.plot(worker_counts, durations, marker='o', linewidth=2, label='Actual')
    
    # Add ideal scaling line (T = T1/n)
    ideal_durations = [durations[0] / w for w in worker_counts]
    plt.plot(worker_counts, ideal_durations, 'k--', linewidth=1, label='Ideal')
    
    plt.xlabel('Number of Workers')
    plt.ylabel('Total Execution Time (s)')
    plt.title('Mixed Tasks: Execution Time vs Workers')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "execution_time_vs_workers.png")
    plt.close()
    
    # Plot 2: Speedup vs Workers
    plt.figure(figsize=(12, 6))
    plt.plot(worker_counts, speedups, marker='s', color='green', linewidth=2, label='Actual')
    plt.plot(worker_counts, theoretical_speedups, 'k--', linewidth=1, label='Theoretical (90% parallel)')
    plt.plot(worker_counts, ideal_speedups, 'r:', linewidth=1, label='Linear')
    
    plt.xlabel('Number of Workers')
    plt.ylabel('Speedup Factor')
    plt.title('Mixed Tasks: Speedup vs Workers')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "speedup_vs_workers.png")
    plt.close()
    
    # Plot 3: Success rate vs Workers
    plt.figure(figsize=(10, 6))
    plt.plot(worker_counts, success_rates, marker='^', color='orange', linewidth=2)
    plt.xlabel('Number of Workers')
    plt.ylabel('Success Rate (%)')
    plt.title('Mixed Tasks: Success Rate vs Workers')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / "success_rate_vs_workers.png")
    plt.close()
    
    # Group tasks by domain
    domain_tasks = {}
    for task in tasks:
        domain = task["domain"]
        if domain not in domain_tasks:
            domain_tasks[domain] = []
        domain_tasks[domain].append(task)
    
    # Plot 4: Domain distribution pie chart
    plt.figure(figsize=(8, 8))
    domains = list(domain_tasks.keys())
    counts = [len(domain_tasks[d]) for d in domains]
    
    plt.pie(counts, labels=domains, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Test Tasks Distribution by Domain')
    
    plt.tight_layout()
    plt.savefig(output_dir / "domain_distribution.png")
    plt.close()
    
    print(f"Plots saved to {output_dir}/")

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test DFZ integration with mixed concurrent tasks")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--count", type=int, default=5, help="Number of tasks per type")
    parser.add_argument("--workers", type=int, default=4, help="Maximum number of concurrent workers")
    parser.add_argument("--use-processes", action="store_true", help="Use processes instead of threads")
    parser.add_argument("--scaling-test", action="store_true", help="Run concurrency scaling test")
    parser.add_argument("--max-workers", type=str, default="1,2,4,8", help="Comma-separated list of worker counts for scaling test")
    
    args = parser.parse_args()
    
    if args.scaling_test:
        # Parse worker counts for scaling test
        max_workers_list = [int(w) for w in args.max_workers.split(',')]
        
        # Run scaling test
        await run_concurrency_scaling_test(
            config_path=args.config,
            count_per_type=args.count,
            max_workers_list=max_workers_list
        )
    else:
        # Run mixed tasks test
        await run_mixed_tasks_test(
            config_path=args.config,
            count_per_type=args.count,
            max_workers=args.workers,
            use_processes=args.use_processes
        )

if __name__ == "__main__":
    asyncio.run(main())