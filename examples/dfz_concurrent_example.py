"""
Example of DFZ integration with concurrent tasks in CTM-AbsoluteZero
"""
import sys
import os
import asyncio
import time
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import concurrent DFZ integration components
from src.integration.dfz_concurrent import ConcurrentDFZAdapter
from src.utils.config import ConfigManager

async def run_single_task_example():
    """
    Example of executing a single task with concurrent DFZ adapter.
    """
    print("\n===== Single Task Example =====")
    
    # Create adapter
    adapter = ConcurrentDFZAdapter(
        config={"config_path": "configs/default.yaml"},
        max_workers=4
    )
    
    # Initialize
    await adapter.initialize()
    
    # Create a quantum task
    task = {
        "id": "single_quantum_task",
        "domain": "quantum",
        "description": "Example quantum task",
        "parameters": {
            "algorithm": "vqe",
            "num_qubits": 5,
            "noise_level": 0.01,
            "circuit_depth": 4
        }
    }
    
    print(f"Executing single task: {task['description']}")
    start_time = time.time()
    
    # Execute task using regular execution
    result = await adapter.execute_task(task)
    
    duration = time.time() - start_time
    print(f"Task executed in {duration:.3f}s with status: {result['status']}")
    
    # Print task result details
    if result['status'] == 'success':
        print("Task result metrics:")
        for key, value in result.items():
            if key not in ['task_id', 'status', 'duration']:
                print(f"  {key}: {value}")

async def run_batch_task_example():
    """
    Example of executing a batch of tasks concurrently with DFZ adapter.
    """
    print("\n===== Batch Task Example =====")
    
    # Create adapter
    adapter = ConcurrentDFZAdapter(
        config={"config_path": "configs/default.yaml"},
        max_workers=4
    )
    
    # Initialize
    await adapter.initialize()
    
    # Create a batch of tasks
    tasks = []
    for i in range(10):
        # Mix of different quantum algorithms
        algorithm = np.random.choice(["vqe", "grover", "qft"])
        num_qubits = np.random.randint(3, 8)
        
        task = {
            "id": f"batch_task_{i+1}",
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
    
    print(f"Executing batch of {len(tasks)} tasks concurrently...")
    start_time = time.time()
    
    # Execute tasks concurrently
    batch_result = await adapter.execute_tasks_batch(tasks, wait=True)
    
    duration = time.time() - start_time
    print(f"Batch executed in {duration:.3f}s")
    
    # Process results
    completed_tasks = batch_result["results"]
    success_count = sum(1 for r in completed_tasks.values() if r.get("status") == "success")
    
    print(f"Completed {len(completed_tasks)}/{len(tasks)} tasks")
    print(f"Success rate: {success_count/len(tasks):.1%}")
    
    # Get performance metrics
    metrics = adapter.get_performance_metrics()
    if "concurrent" in metrics:
        concurrent_metrics = metrics["concurrent"]
        print("\nConcurrent execution metrics:")
        print(f"  Average concurrency: {concurrent_metrics.get('avg_concurrency', 0):.2f}")
        print(f"  Maximum concurrency: {concurrent_metrics.get('max_concurrency', 0)}")
        print(f"  Average task duration: {concurrent_metrics.get('avg_duration', 0):.3f}s")

async def run_sequential_vs_concurrent_comparison():
    """
    Compare sequential vs concurrent execution of tasks.
    """
    print("\n===== Sequential vs Concurrent Comparison =====")
    
    # Create adapter
    adapter = ConcurrentDFZAdapter(
        config={"config_path": "configs/default.yaml"},
        max_workers=4
    )
    
    # Initialize
    await adapter.initialize()
    
    # Create tasks (same set for both sequential and concurrent)
    num_tasks = 8
    tasks = []
    for i in range(num_tasks):
        # Mix of different quantum algorithms
        algorithm = np.random.choice(["vqe", "grover", "qft"])
        num_qubits = np.random.randint(3, 7)
        
        task = {
            "id": f"compare_task_{i+1}",
            "domain": "quantum",
            "description": f"Quantum task {i+1} ({algorithm})",
            "parameters": {
                "algorithm": algorithm,
                "num_qubits": num_qubits,
                "noise_level": 0.02,
                "circuit_depth": 4
            }
        }
        tasks.append(task)
    
    # Execute sequentially
    print(f"Executing {num_tasks} tasks sequentially...")
    sequential_start = time.time()
    
    sequential_results = []
    for task in tasks:
        result = await adapter.execute_task(task)
        sequential_results.append(result)
    
    sequential_duration = time.time() - sequential_start
    print(f"Sequential execution completed in {sequential_duration:.3f}s")
    
    # Execute concurrently
    print(f"\nExecuting {num_tasks} tasks concurrently...")
    concurrent_start = time.time()
    
    batch_result = await adapter.execute_tasks_batch(tasks, wait=True)
    
    concurrent_duration = time.time() - concurrent_start
    print(f"Concurrent execution completed in {concurrent_duration:.3f}s")
    
    # Calculate speedup
    speedup = sequential_duration / concurrent_duration
    print(f"\nSpeedup with concurrent execution: {speedup:.2f}x")
    
    # Create a bar chart to visualize the comparison
    plt.figure(figsize=(10, 6))
    plt.bar(['Sequential', 'Concurrent'], [sequential_duration, concurrent_duration], color=['blue', 'green'])
    plt.ylabel('Execution Time (s)')
    plt.title('Sequential vs Concurrent Execution Time Comparison')
    
    # Add text labels for durations and speedup
    plt.text(0, sequential_duration + 0.1, f"{sequential_duration:.2f}s", ha='center')
    plt.text(1, concurrent_duration + 0.1, f"{concurrent_duration:.2f}s", ha='center')
    plt.text(0.5, max(sequential_duration, concurrent_duration) * 0.5, 
             f"Speedup: {speedup:.2f}x", 
             ha='center', 
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Save the chart
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/seq_vs_concurrent.png")
    plt.close()
    print("Comparison chart saved to 'results/seq_vs_concurrent.png'")

async def main():
    """Main function to run the examples"""
    print("CTM-AbsoluteZero - DFZ Concurrent Task Integration Examples")
    print("=" * 60)
    
    # Run examples
    await run_single_task_example()
    await run_batch_task_example()
    await run_sequential_vs_concurrent_comparison()
    
    print("\nAll examples completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())