#!/usr/bin/env python3
"""
Script to run the UniversalRouter benchmark with different configurations.
This script tests the router's performance under various load scenarios.
"""
import os
import sys
import asyncio
import argparse
import logging
from typing import List, Dict, Any

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.benchmarks.router_benchmark import RouterBenchmark, BenchmarkConfig
from src.utils.logging import configure_logging

# Define benchmark suite configurations
BENCHMARK_SUITES = {
    "basic": {
        "num_tasks": 100,
        "concurrent_tasks": 5,
        "domains": ["quantum", "maze", "sorting"],
        "task_complexity": "medium"
    },
    "high_concurrency": {
        "num_tasks": 1000,
        "concurrent_tasks": 50,
        "domains": ["quantum", "maze", "sorting", "vision", "general"],
        "task_complexity": "medium"
    },
    "high_volume": {
        "num_tasks": 5000,
        "concurrent_tasks": 20,
        "domains": ["quantum", "maze", "sorting", "vision", "general"],
        "task_complexity": "medium"
    },
    "quantum_focused": {
        "num_tasks": 1000,
        "concurrent_tasks": 10,
        "domains": ["quantum"],
        "task_complexity": "hard"
    },
    "mixed_load": {
        "num_tasks": 2000,
        "concurrent_tasks": 30,
        "domains": ["quantum", "maze", "sorting", "vision", "general"],
        "task_complexity": "medium"
    }
}

async def run_benchmark_suite(
    suite_name: str,
    config_path: str,
    output_dir: str,
    skip_plots: bool = False
) -> Dict[str, Any]:
    """
    Run a benchmark suite.
    
    Args:
        suite_name: Name of the benchmark suite
        config_path: Path to the router configuration
        output_dir: Directory to save results
        skip_plots: Whether to skip generating plots
        
    Returns:
        Benchmark results
    """
    print(f"Running benchmark suite: {suite_name}")
    
    # Get suite configuration
    suite_config = BENCHMARK_SUITES.get(suite_name, BENCHMARK_SUITES["basic"])
    
    # Create benchmark config
    config = BenchmarkConfig(
        config_path=config_path,
        num_tasks=suite_config["num_tasks"],
        concurrent_tasks=suite_config["concurrent_tasks"],
        domains=suite_config["domains"],
        task_complexity=suite_config["task_complexity"],
        output_dir=os.path.join(output_dir, suite_name),
        skip_plots=skip_plots
    )
    
    # Run benchmark
    benchmark = RouterBenchmark(config)
    results = await benchmark.run()
    
    return results

async def run_all_suites(
    config_path: str,
    output_dir: str,
    suites: List[str] = None,
    skip_plots: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    Run multiple benchmark suites.
    
    Args:
        config_path: Path to the router configuration
        output_dir: Directory to save results
        suites: List of suite names to run (runs all if None)
        skip_plots: Whether to skip generating plots
        
    Returns:
        Dictionary of benchmark results by suite name
    """
    # If no suites specified, run all
    if not suites:
        suites = list(BENCHMARK_SUITES.keys())
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run suites
    results = {}
    
    for suite_name in suites:
        if suite_name in BENCHMARK_SUITES:
            suite_results = await run_benchmark_suite(
                suite_name,
                config_path,
                output_dir,
                skip_plots
            )
            
            results[suite_name] = suite_results
        else:
            print(f"Unknown benchmark suite: {suite_name}")
    
    return results

async def main():
    """Run benchmark suites from command line."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="UniversalRouter Benchmark Suite")
    
    # Add arguments
    parser.add_argument(
        "--config", "-c",
        help="Path to router configuration file",
        type=str
    )
    parser.add_argument(
        "--output-dir", "-o",
        help="Directory to save results",
        type=str,
        default="./router_benchmark_results"
    )
    parser.add_argument(
        "--suites", "-s",
        help="Comma-separated list of benchmark suites to run",
        type=str
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
    
    # Parse suites
    suites = args.suites.split(",") if args.suites else None
    
    # Run benchmark suites
    await run_all_suites(
        args.config,
        args.output_dir,
        suites,
        args.no_plots
    )

if __name__ == "__main__":
    asyncio.run(main())