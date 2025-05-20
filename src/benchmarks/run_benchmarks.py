#!/usr/bin/env python3
"""
Run benchmarks for CTM-AbsoluteZero.

This script runs benchmarks for the CTM-AbsoluteZero framework, including
agentic model comparison and performance analysis.
"""
import os
import sys
import argparse
import asyncio
import logging
from typing import List, Dict, Any, Optional

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.utils.logging import configure_logging
from src.benchmarks.agentic_comparison import AgenticBenchmark, BenchmarkConfig

async def run_agentic_comparison(args: argparse.Namespace) -> None:
    """
    Run agentic comparison benchmark.
    
    Args:
        args: Command-line arguments
    """
    # Parse domains and difficulties
    domains = args.domains.split(",") if args.domains else None
    difficulties = args.difficulties.split(",") if args.difficulties else None
    
    # Create benchmark config
    config = BenchmarkConfig(
        config_path=args.config,
        domains=domains,
        difficulties=difficulties,
        num_iterations=args.iterations,
        output_dir=args.output_dir,
        timeout=args.timeout,
        estimate_tokens=not args.no_tokens,
        skip_plots=args.no_plots
    )
    
    # Run benchmark
    benchmark = AgenticBenchmark(config)
    await benchmark.run()

def main():
    """Run benchmarks from command line."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="CTM-AbsoluteZero Benchmarks")
    
    # Add subparsers
    subparsers = parser.add_subparsers(dest="command", help="Benchmark to run")
    
    # Agentic comparison benchmark
    agentic_parser = subparsers.add_parser("agentic", help="Run agentic model comparison benchmark")
    agentic_parser.add_argument(
        "--config", "-c",
        help="Path to agent configuration file",
        type=str
    )
    agentic_parser.add_argument(
        "--domains", "-d",
        help="Comma-separated list of domains to benchmark (default: quantum,maze,sorting)",
        type=str,
        default="quantum,maze,sorting"
    )
    agentic_parser.add_argument(
        "--difficulties", "-l",
        help="Comma-separated list of difficulties to benchmark (default: easy,medium,hard)",
        type=str,
        default="easy,medium,hard"
    )
    agentic_parser.add_argument(
        "--iterations", "-i",
        help="Number of iterations per configuration",
        type=int,
        default=5
    )
    agentic_parser.add_argument(
        "--output-dir", "-o",
        help="Directory to save results",
        type=str,
        default="./benchmark_results"
    )
    agentic_parser.add_argument(
        "--timeout", "-t",
        help="Timeout for each task in seconds",
        type=float,
        default=60.0
    )
    agentic_parser.add_argument(
        "--no-tokens",
        help="Disable token usage estimation",
        action="store_true"
    )
    agentic_parser.add_argument(
        "--no-plots",
        help="Disable plot generation",
        action="store_true"
    )
    
    # Other benchmarks can be added here
    
    # Common arguments
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
    
    # Run benchmark
    if args.command == "agentic":
        asyncio.run(run_agentic_comparison(args))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()