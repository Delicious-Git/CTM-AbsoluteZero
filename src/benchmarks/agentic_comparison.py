"""
Benchmark comparing Claude and DeepSeek agentic performance.

This module provides tools for comparing the performance, cost, and efficiency
of Claude and DeepSeek models in the CTM-AbsoluteZero framework.
"""
import os
import sys
import time
import json
import asyncio
import logging
import argparse
from typing import Dict, List, Any, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.agentic.framework import BrainFramework
from src.utils.logging import get_logger, configure_logging
from src.utils.config import ConfigManager

# Setup logger
logger = get_logger("ctm-az.benchmarks.agentic")

# Constants
TOKEN_COST_CLAUDE = 0.008 / 1000  # $0.008 per 1K tokens for Claude-3-Opus
TOKEN_COST_DEEPSEEK = 0.0001 / 1000  # $0.0001 per 1K tokens for DeepSeek-Chat

class BenchmarkConfig:
    """Configuration for agentic comparison benchmark."""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        domains: Optional[List[str]] = None,
        difficulties: Optional[List[str]] = None,
        num_iterations: int = 5,
        output_dir: str = "./benchmark_results",
        timeout: float = 60.0,
        estimate_tokens: bool = True,
        skip_plots: bool = False
    ):
        """
        Initialize benchmark configuration.
        
        Args:
            config_path: Path to agent configuration file
            domains: List of domains to benchmark
            difficulties: List of difficulties to benchmark
            num_iterations: Number of iterations per domain/difficulty combination
            output_dir: Directory to save results
            timeout: Timeout for each task in seconds
            estimate_tokens: Whether to estimate token usage
            skip_plots: Whether to skip generating plots
        """
        self.config_path = config_path
        self.domains = domains or ["quantum", "maze", "sorting"]
        self.difficulties = difficulties or ["easy", "medium", "hard"]
        self.num_iterations = num_iterations
        self.output_dir = output_dir
        self.timeout = timeout
        self.estimate_tokens = estimate_tokens
        self.skip_plots = skip_plots


class AgenticBenchmark:
    """Benchmark for comparing Claude and DeepSeek agentic performance."""
    
    def __init__(self, config: BenchmarkConfig):
        """
        Initialize the benchmark.
        
        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.logger = get_logger("ctm-az.benchmarks.agentic")
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Load brain framework
        self.framework = self._load_framework()
        
        # Results
        self.results = {
            "claude": {},
            "deepseek": {},
            "summary": {}
        }
    
    def _load_framework(self) -> BrainFramework:
        """
        Load brain framework from configuration.
        
        Returns:
            Configured brain framework
        """
        config_path = self.config.config_path
        
        if not config_path or not os.path.exists(config_path):
            # Look for default config
            default_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "configs", "agentic_brain.yaml"
            )
            
            if os.path.exists(default_path):
                config_path = default_path
            else:
                self.logger.warning("No config file found, using default configuration")
                return BrainFramework()
        
        self.logger.info(f"Loading brain framework from {config_path}")
        return BrainFramework(config_path=config_path)
    
    async def run(self) -> Dict[str, Any]:
        """
        Run the benchmark.
        
        Returns:
            Benchmark results
        """
        self.logger.info(f"Starting benchmark with {self.config.num_iterations} iterations per configuration")
        
        start_time = time.time()
        
        # Initialize empty results
        for agent in ["claude", "deepseek"]:
            for domain in self.config.domains:
                for difficulty in self.config.difficulties:
                    key = f"{domain}_{difficulty}"
                    if domain not in self.results[agent]:
                        self.results[agent][domain] = {}
                    self.results[agent][domain][difficulty] = {
                        "iterations": [],
                        "avg_time": 0.0,
                        "avg_rating": 0.0,
                        "avg_reward": 0.0,
                        "success_rate": 0.0,
                        "tokens_generated": 0
                    }
        
        # Run benchmarks
        for domain in self.config.domains:
            for difficulty in self.config.difficulties:
                await self._run_domain_difficulty(domain, difficulty)
        
        # Calculate summary statistics
        self._calculate_summary()
        
        # Generate plots
        if not self.config.skip_plots:
            self._generate_plots()
        
        # Save results
        self._save_results()
        
        duration = time.time() - start_time
        self.logger.info(f"Benchmark completed in {duration:.2f}s")
        
        return self.results
    
    async def _run_domain_difficulty(self, domain: str, difficulty: str) -> None:
        """
        Run benchmark for a specific domain and difficulty.
        
        Args:
            domain: Domain to benchmark
            difficulty: Difficulty level
        """
        self.logger.info(f"Running benchmark for domain={domain}, difficulty={difficulty}")
        
        # Run Claude
        claude_results = await self._run_agent_iterations(
            "claude", domain, difficulty, self.config.num_iterations
        )
        
        # Run DeepSeek
        deepseek_results = await self._run_agent_iterations(
            "deepseek", domain, difficulty, self.config.num_iterations
        )
        
        # Store results
        self.results["claude"][domain][difficulty]["iterations"] = claude_results
        self.results["deepseek"][domain][difficulty]["iterations"] = deepseek_results
        
        # Calculate statistics
        for agent, results_list in [("claude", claude_results), ("deepseek", deepseek_results)]:
            if not results_list:
                continue
                
            # Filter out errors
            successful_results = [r for r in results_list if "error" not in r]
            
            if successful_results:
                self.results[agent][domain][difficulty]["avg_time"] = sum(
                    r.get("duration", 0) for r in successful_results
                ) / len(successful_results)
                
                self.results[agent][domain][difficulty]["avg_rating"] = sum(
                    r.get("analysis", {}).get("overall_rating", 0) for r in successful_results
                ) / len(successful_results)
                
                self.results[agent][domain][difficulty]["avg_reward"] = sum(
                    r.get("reward", 0) for r in successful_results
                ) / len(successful_results)
            
            self.results[agent][domain][difficulty]["success_rate"] = len(successful_results) / len(results_list) if results_list else 0
            
            # Estimate tokens
            if self.config.estimate_tokens:
                self.results[agent][domain][difficulty]["tokens_generated"] = self._estimate_tokens(agent, successful_results)
            
        # Log comparison
        self._log_comparison(domain, difficulty)
    
    async def _run_agent_iterations(
        self,
        agent: str,
        domain: str,
        difficulty: str,
        num_iterations: int
    ) -> List[Dict[str, Any]]:
        """
        Run benchmark iterations for a specific agent, domain and difficulty.
        
        Args:
            agent: Agent name
            domain: Domain to benchmark
            difficulty: Difficulty level
            num_iterations: Number of iterations
            
        Returns:
            List of iteration results
        """
        results = []
        
        for i in range(num_iterations):
            self.logger.info(f"Running {agent} iteration {i+1}/{num_iterations} for {domain}/{difficulty}")
            
            try:
                # Run with timeout
                result = await asyncio.wait_for(
                    self.framework.run_cycle(
                        domain=domain,
                        difficulty=difficulty,
                        constraints=None,
                        agent_name=agent
                    ),
                    timeout=self.config.timeout
                )
                
                results.append(result)
                
                # Log result
                status = "success" if "error" not in result else "error"
                self.logger.info(f"{agent} iteration {i+1} {status}")
                
            except asyncio.TimeoutError:
                self.logger.error(f"{agent} iteration {i+1} timed out after {self.config.timeout}s")
                results.append({"error": "timeout"})
            except Exception as e:
                self.logger.error(f"{agent} iteration {i+1} failed: {e}")
                results.append({"error": str(e)})
        
        return results
    
    def _estimate_tokens(self, agent: str, results: List[Dict[str, Any]]) -> int:
        """
        Estimate token usage for the agent.
        
        Args:
            agent: Agent name
            results: List of successful results
            
        Returns:
            Estimated token count
        """
        if not results:
            return 0
        
        # Rough estimation based on text length
        total_tokens = 0
        
        for result in results:
            # Task description tokens
            if "task" in result:
                task_text = json.dumps(result["task"])
                total_tokens += len(task_text) // 4  # Very rough estimation
            
            # Analysis tokens
            if "analysis" in result:
                analysis_text = json.dumps(result["analysis"])
                total_tokens += len(analysis_text) // 4
        
        # Add fixed overhead per task for prompt and internal processing
        overhead_per_task = 1000  # Rough estimate
        total_tokens += len(results) * overhead_per_task
        
        return total_tokens
    
    def _log_comparison(self, domain: str, difficulty: str) -> None:
        """
        Log comparison between Claude and DeepSeek for a domain/difficulty.
        
        Args:
            domain: Domain
            difficulty: Difficulty level
        """
        claude = self.results["claude"][domain][difficulty]
        deepseek = self.results["deepseek"][domain][difficulty]
        
        self.logger.info(f"\n--- {domain.upper()} / {difficulty.upper()} COMPARISON ---")
        
        self.logger.info(f"Success Rate: Claude {claude['success_rate']:.1%} vs DeepSeek {deepseek['success_rate']:.1%}")
        
        if claude["success_rate"] > 0 and deepseek["success_rate"] > 0:
            self.logger.info(f"Avg Time: Claude {claude['avg_time']:.2f}s vs DeepSeek {deepseek['avg_time']:.2f}s")
            self.logger.info(f"Avg Rating: Claude {claude['avg_rating']:.2f} vs DeepSeek {deepseek['avg_rating']:.2f}")
            self.logger.info(f"Avg Reward: Claude {claude['avg_reward']:.2f} vs DeepSeek {deepseek['avg_reward']:.2f}")
            
            # Estimate cost
            if self.config.estimate_tokens:
                claude_cost = claude["tokens_generated"] * TOKEN_COST_CLAUDE
                deepseek_cost = deepseek["tokens_generated"] * TOKEN_COST_DEEPSEEK
                
                self.logger.info(f"Est. Tokens: Claude {claude['tokens_generated']} vs DeepSeek {deepseek['tokens_generated']}")
                self.logger.info(f"Est. Cost: Claude ${claude_cost:.4f} vs DeepSeek ${deepseek_cost:.4f}")
                
                if deepseek_cost > 0:
                    cost_ratio = claude_cost / deepseek_cost
                    self.logger.info(f"Cost Ratio: Claude is {cost_ratio:.1f}x more expensive")
                
                # Performance per dollar
                if claude_cost > 0 and deepseek_cost > 0:
                    claude_perf_per_dollar = claude["avg_rating"] / claude_cost
                    deepseek_perf_per_dollar = deepseek["avg_rating"] / deepseek_cost
                    
                    self.logger.info(f"Rating per Dollar: Claude {claude_perf_per_dollar:.2f} vs DeepSeek {deepseek_perf_per_dollar:.2f}")
                    
                    if deepseek_perf_per_dollar > claude_perf_per_dollar:
                        self.logger.info(f"DeepSeek has {deepseek_perf_per_dollar/claude_perf_per_dollar:.1f}x better performance per dollar")
                    else:
                        self.logger.info(f"Claude has {claude_perf_per_dollar/deepseek_perf_per_dollar:.1f}x better performance per dollar")
    
    def _calculate_summary(self) -> None:
        """Calculate summary statistics across all benchmarks."""
        # Initialize summary
        summary = {
            "claude": {
                "avg_time": 0.0,
                "avg_rating": 0.0,
                "avg_reward": 0.0,
                "success_rate": 0.0,
                "total_tokens": 0,
                "estimated_cost": 0.0,
                "task_count": 0,
                "domains": {}
            },
            "deepseek": {
                "avg_time": 0.0,
                "avg_rating": 0.0,
                "avg_reward": 0.0,
                "success_rate": 0.0,
                "total_tokens": 0,
                "estimated_cost": 0.0,
                "task_count": 0,
                "domains": {}
            },
            "comparison": {
                "time_ratio": 0.0,
                "rating_ratio": 0.0,
                "reward_ratio": 0.0,
                "success_ratio": 0.0,
                "cost_ratio": 0.0,
                "performance_per_dollar_ratio": 0.0
            }
        }
        
        # Calculate statistics for each agent
        for agent in ["claude", "deepseek"]:
            total_time = 0.0
            total_rating = 0.0
            total_reward = 0.0
            total_success_count = 0
            total_task_count = 0
            
            # Total per domain
            for domain in self.config.domains:
                domain_time = 0.0
                domain_rating = 0.0
                domain_reward = 0.0
                domain_success_count = 0
                domain_task_count = 0
                
                for difficulty in self.config.difficulties:
                    results = self.results[agent][domain][difficulty]
                    
                    # Count tasks
                    task_count = len(results["iterations"])
                    success_count = int(results["success_rate"] * task_count)
                    
                    # Add to totals
                    if success_count > 0:
                        total_time += results["avg_time"] * success_count
                        total_rating += results["avg_rating"] * success_count
                        total_reward += results["avg_reward"] * success_count
                    
                    total_success_count += success_count
                    total_task_count += task_count
                    
                    # Add to domain totals
                    if success_count > 0:
                        domain_time += results["avg_time"] * success_count
                        domain_rating += results["avg_rating"] * success_count
                        domain_reward += results["avg_reward"] * success_count
                    
                    domain_success_count += success_count
                    domain_task_count += task_count
                    
                    # Add tokens to total
                    if self.config.estimate_tokens:
                        summary[agent]["total_tokens"] += results["tokens_generated"]
                
                # Calculate domain averages
                if domain_success_count > 0:
                    summary[agent]["domains"][domain] = {
                        "avg_time": domain_time / domain_success_count,
                        "avg_rating": domain_rating / domain_success_count,
                        "avg_reward": domain_reward / domain_success_count,
                        "success_rate": domain_success_count / domain_task_count if domain_task_count > 0 else 0
                    }
            
            # Calculate overall averages
            if total_success_count > 0:
                summary[agent]["avg_time"] = total_time / total_success_count
                summary[agent]["avg_rating"] = total_rating / total_success_count
                summary[agent]["avg_reward"] = total_reward / total_success_count
            
            summary[agent]["success_rate"] = total_success_count / total_task_count if total_task_count > 0 else 0
            summary[agent]["task_count"] = total_task_count
            
            # Calculate cost
            if self.config.estimate_tokens:
                token_cost = TOKEN_COST_CLAUDE if agent == "claude" else TOKEN_COST_DEEPSEEK
                summary[agent]["estimated_cost"] = summary[agent]["total_tokens"] * token_cost
        
        # Calculate comparison ratios
        if summary["deepseek"]["avg_time"] > 0:
            summary["comparison"]["time_ratio"] = summary["claude"]["avg_time"] / summary["deepseek"]["avg_time"]
        
        if summary["deepseek"]["avg_rating"] > 0:
            summary["comparison"]["rating_ratio"] = summary["claude"]["avg_rating"] / summary["deepseek"]["avg_rating"]
        
        if summary["deepseek"]["avg_reward"] > 0:
            summary["comparison"]["reward_ratio"] = summary["claude"]["avg_reward"] / summary["deepseek"]["avg_reward"]
        
        if summary["deepseek"]["success_rate"] > 0:
            summary["comparison"]["success_ratio"] = summary["claude"]["success_rate"] / summary["deepseek"]["success_rate"]
        
        if summary["deepseek"]["estimated_cost"] > 0:
            summary["comparison"]["cost_ratio"] = summary["claude"]["estimated_cost"] / summary["deepseek"]["estimated_cost"]
        
        # Performance per dollar
        if summary["claude"]["estimated_cost"] > 0 and summary["deepseek"]["estimated_cost"] > 0:
            claude_perf_per_dollar = summary["claude"]["avg_rating"] / summary["claude"]["estimated_cost"]
            deepseek_perf_per_dollar = summary["deepseek"]["avg_rating"] / summary["deepseek"]["estimated_cost"]
            
            if deepseek_perf_per_dollar > 0:
                summary["comparison"]["performance_per_dollar_ratio"] = claude_perf_per_dollar / deepseek_perf_per_dollar
        
        # Store summary
        self.results["summary"] = summary
        
        # Log summary
        self._log_summary()
    
    def _log_summary(self) -> None:
        """Log benchmark summary."""
        summary = self.results["summary"]
        
        self.logger.info("\n===== BENCHMARK SUMMARY =====")
        
        self.logger.info("\nClaude:")
        self.logger.info(f"Success Rate: {summary['claude']['success_rate']:.1%}")
        self.logger.info(f"Avg Rating: {summary['claude']['avg_rating']:.2f}")
        self.logger.info(f"Avg Time: {summary['claude']['avg_time']:.2f}s")
        self.logger.info(f"Total Tokens: {summary['claude']['total_tokens']}")
        self.logger.info(f"Est. Cost: ${summary['claude']['estimated_cost']:.4f}")
        
        self.logger.info("\nDeepSeek:")
        self.logger.info(f"Success Rate: {summary['deepseek']['success_rate']:.1%}")
        self.logger.info(f"Avg Rating: {summary['deepseek']['avg_rating']:.2f}")
        self.logger.info(f"Avg Time: {summary['deepseek']['avg_time']:.2f}s")
        self.logger.info(f"Total Tokens: {summary['deepseek']['total_tokens']}")
        self.logger.info(f"Est. Cost: ${summary['deepseek']['estimated_cost']:.4f}")
        
        self.logger.info("\nComparison (Claude/DeepSeek ratio):")
        self.logger.info(f"Rating Ratio: {summary['comparison']['rating_ratio']:.2f}x")
        self.logger.info(f"Success Rate Ratio: {summary['comparison']['success_ratio']:.2f}x")
        self.logger.info(f"Time Ratio: {summary['comparison']['time_ratio']:.2f}x")
        self.logger.info(f"Cost Ratio: {summary['comparison']['cost_ratio']:.2f}x")
        
        # Performance per dollar
        claude_ppd = summary['claude']['avg_rating'] / summary['claude']['estimated_cost'] if summary['claude']['estimated_cost'] > 0 else 0
        deepseek_ppd = summary['deepseek']['avg_rating'] / summary['deepseek']['estimated_cost'] if summary['deepseek']['estimated_cost'] > 0 else 0
        
        self.logger.info(f"\nPerformance per Dollar:")
        self.logger.info(f"Claude: {claude_ppd:.2f}")
        self.logger.info(f"DeepSeek: {deepseek_ppd:.2f}")
        
        if claude_ppd > 0 and deepseek_ppd > 0:
            ratio = deepseek_ppd / claude_ppd
            better = "DeepSeek" if ratio > 1 else "Claude"
            self.logger.info(f"{better} has {max(ratio, 1/ratio):.1f}x better performance per dollar")
    
    def _generate_plots(self) -> None:
        """Generate benchmark plots."""
        self.logger.info("Generating benchmark plots")
        
        # Create output directory
        plots_dir = os.path.join(self.config.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot success rates
        self._plot_success_rates(plots_dir)
        
        # Plot ratings
        self._plot_ratings(plots_dir)
        
        # Plot times
        self._plot_times(plots_dir)
        
        # Plot costs
        if self.config.estimate_tokens:
            self._plot_costs(plots_dir)
            
            # Plot performance per dollar
            self._plot_performance_per_dollar(plots_dir)
    
    def _plot_success_rates(self, plots_dir: str) -> None:
        """
        Plot success rates.
        
        Args:
            plots_dir: Directory to save plot
        """
        plt.figure(figsize=(12, 8))
        
        # Data
        domains = self.config.domains
        width = 0.35
        x = np.arange(len(domains))
        
        claude_rates = []
        deepseek_rates = []
        
        for domain in domains:
            claude_domain_success = 0
            claude_domain_total = 0
            deepseek_domain_success = 0
            deepseek_domain_total = 0
            
            for difficulty in self.config.difficulties:
                claude_results = self.results["claude"][domain][difficulty]
                deepseek_results = self.results["deepseek"][domain][difficulty]
                
                claude_domain_success += int(claude_results["success_rate"] * len(claude_results["iterations"]))
                claude_domain_total += len(claude_results["iterations"])
                
                deepseek_domain_success += int(deepseek_results["success_rate"] * len(deepseek_results["iterations"]))
                deepseek_domain_total += len(deepseek_results["iterations"])
            
            claude_rate = claude_domain_success / claude_domain_total if claude_domain_total > 0 else 0
            deepseek_rate = deepseek_domain_success / deepseek_domain_total if deepseek_domain_total > 0 else 0
            
            claude_rates.append(claude_rate)
            deepseek_rates.append(deepseek_rate)
        
        # Plot
        plt.bar(x - width/2, claude_rates, width, label='Claude')
        plt.bar(x + width/2, deepseek_rates, width, label='DeepSeek')
        
        plt.xlabel('Domain')
        plt.ylabel('Success Rate')
        plt.title('Success Rate by Domain')
        plt.xticks(x, domains)
        plt.ylim(0, 1.1)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Add values above bars
        for i, v in enumerate(claude_rates):
            plt.text(i - width/2, v + 0.05, f"{v:.1%}", ha='center')
        
        for i, v in enumerate(deepseek_rates):
            plt.text(i + width/2, v + 0.05, f"{v:.1%}", ha='center')
        
        # Save
        plt.savefig(os.path.join(plots_dir, "success_rates.png"))
        plt.close()
    
    def _plot_ratings(self, plots_dir: str) -> None:
        """
        Plot ratings.
        
        Args:
            plots_dir: Directory to save plot
        """
        plt.figure(figsize=(12, 8))
        
        # Data
        domains = self.config.domains
        difficulties = self.config.difficulties
        
        # Set up plot
        num_groups = len(domains)
        num_bars = len(difficulties) * 2  # 2 agents per difficulty
        group_width = 0.8
        bar_width = group_width / num_bars
        
        # Colors for difficulties
        colors = ['#5DA5DA', '#FAA43A', '#60BD68']
        
        for i, domain in enumerate(domains):
            for j, difficulty in enumerate(difficulties):
                claude_rating = self.results["claude"][domain][difficulty]["avg_rating"]
                deepseek_rating = self.results["deepseek"][domain][difficulty]["avg_rating"]
                
                # Calculate x positions
                claude_x = i - group_width/2 + (j*2 + 0.5) * bar_width
                deepseek_x = i - group_width/2 + (j*2 + 1.5) * bar_width
                
                # Plot bars
                plt.bar(claude_x, claude_rating, width=bar_width, color=colors[j], alpha=0.7, 
                       label=f'Claude {difficulty}' if i == 0 else "")
                plt.bar(deepseek_x, deepseek_rating, width=bar_width, color=colors[j], alpha=0.4,
                       label=f'DeepSeek {difficulty}' if i == 0 else "")
                
                # Add values above bars
                if claude_rating > 0:
                    plt.text(claude_x, claude_rating + 0.05, f"{claude_rating:.2f}", ha='center', va='bottom', fontsize=8)
                if deepseek_rating > 0:
                    plt.text(deepseek_x, deepseek_rating + 0.05, f"{deepseek_rating:.2f}", ha='center', va='bottom', fontsize=8)
        
        plt.xlabel('Domain')
        plt.ylabel('Average Rating (0-1)')
        plt.title('Average Rating by Domain and Difficulty')
        plt.xticks(range(len(domains)), domains)
        plt.ylim(0, 1.1)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Save
        plt.savefig(os.path.join(plots_dir, "ratings.png"))
        plt.close()
    
    def _plot_times(self, plots_dir: str) -> None:
        """
        Plot execution times.
        
        Args:
            plots_dir: Directory to save plot
        """
        plt.figure(figsize=(12, 8))
        
        # Data
        domains = self.config.domains
        width = 0.35
        x = np.arange(len(domains))
        
        claude_times = []
        deepseek_times = []
        
        for domain in domains:
            claude_domain_time = 0
            claude_domain_count = 0
            deepseek_domain_time = 0
            deepseek_domain_count = 0
            
            for difficulty in self.config.difficulties:
                claude_results = self.results["claude"][domain][difficulty]
                deepseek_results = self.results["deepseek"][domain][difficulty]
                
                if claude_results["avg_time"] > 0:
                    claude_domain_time += claude_results["avg_time"] * int(claude_results["success_rate"] * len(claude_results["iterations"]))
                    claude_domain_count += int(claude_results["success_rate"] * len(claude_results["iterations"]))
                
                if deepseek_results["avg_time"] > 0:
                    deepseek_domain_time += deepseek_results["avg_time"] * int(deepseek_results["success_rate"] * len(deepseek_results["iterations"]))
                    deepseek_domain_count += int(deepseek_results["success_rate"] * len(deepseek_results["iterations"]))
            
            claude_avg = claude_domain_time / claude_domain_count if claude_domain_count > 0 else 0
            deepseek_avg = deepseek_domain_time / deepseek_domain_count if deepseek_domain_count > 0 else 0
            
            claude_times.append(claude_avg)
            deepseek_times.append(deepseek_avg)
        
        # Plot
        plt.bar(x - width/2, claude_times, width, label='Claude')
        plt.bar(x + width/2, deepseek_times, width, label='DeepSeek')
        
        plt.xlabel('Domain')
        plt.ylabel('Average Time (seconds)')
        plt.title('Average Execution Time by Domain')
        plt.xticks(x, domains)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Add values above bars
        for i, v in enumerate(claude_times):
            if v > 0:
                plt.text(i - width/2, v + 0.2, f"{v:.2f}s", ha='center')
        
        for i, v in enumerate(deepseek_times):
            if v > 0:
                plt.text(i + width/2, v + 0.2, f"{v:.2f}s", ha='center')
        
        # Save
        plt.savefig(os.path.join(plots_dir, "times.png"))
        plt.close()
    
    def _plot_costs(self, plots_dir: str) -> None:
        """
        Plot estimated costs.
        
        Args:
            plots_dir: Directory to save plot
        """
        plt.figure(figsize=(12, 8))
        
        # Data
        domains = self.config.domains
        width = 0.35
        x = np.arange(len(domains))
        
        claude_costs = []
        deepseek_costs = []
        
        for domain in domains:
            claude_domain_tokens = 0
            deepseek_domain_tokens = 0
            
            for difficulty in self.config.difficulties:
                claude_results = self.results["claude"][domain][difficulty]
                deepseek_results = self.results["deepseek"][domain][difficulty]
                
                claude_domain_tokens += claude_results["tokens_generated"]
                deepseek_domain_tokens += deepseek_results["tokens_generated"]
            
            claude_cost = claude_domain_tokens * TOKEN_COST_CLAUDE
            deepseek_cost = deepseek_domain_tokens * TOKEN_COST_DEEPSEEK
            
            claude_costs.append(claude_cost)
            deepseek_costs.append(deepseek_cost)
        
        # Plot
        plt.bar(x - width/2, claude_costs, width, label='Claude')
        plt.bar(x + width/2, deepseek_costs, width, label='DeepSeek')
        
        plt.xlabel('Domain')
        plt.ylabel('Estimated Cost ($)')
        plt.title('Estimated Cost by Domain')
        plt.xticks(x, domains)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Add values above bars
        for i, v in enumerate(claude_costs):
            if v > 0:
                plt.text(i - width/2, v + max(claude_costs) * 0.05, f"${v:.4f}", ha='center')
        
        for i, v in enumerate(deepseek_costs):
            if v > 0:
                plt.text(i + width/2, v + max(claude_costs) * 0.05, f"${v:.4f}", ha='center')
        
        # Save
        plt.savefig(os.path.join(plots_dir, "costs.png"))
        plt.close()
    
    def _plot_performance_per_dollar(self, plots_dir: str) -> None:
        """
        Plot performance per dollar.
        
        Args:
            plots_dir: Directory to save plot
        """
        plt.figure(figsize=(12, 8))
        
        # Data
        domains = self.config.domains
        width = 0.35
        x = np.arange(len(domains))
        
        claude_ppd = []
        deepseek_ppd = []
        
        for domain in domains:
            claude_domain_rating = 0
            claude_domain_count = 0
            deepseek_domain_rating = 0
            deepseek_domain_count = 0
            claude_domain_tokens = 0
            deepseek_domain_tokens = 0
            
            for difficulty in self.config.difficulties:
                claude_results = self.results["claude"][domain][difficulty]
                deepseek_results = self.results["deepseek"][domain][difficulty]
                
                if claude_results["avg_rating"] > 0:
                    claude_domain_rating += claude_results["avg_rating"] * int(claude_results["success_rate"] * len(claude_results["iterations"]))
                    claude_domain_count += int(claude_results["success_rate"] * len(claude_results["iterations"]))
                
                if deepseek_results["avg_rating"] > 0:
                    deepseek_domain_rating += deepseek_results["avg_rating"] * int(deepseek_results["success_rate"] * len(deepseek_results["iterations"]))
                    deepseek_domain_count += int(deepseek_results["success_rate"] * len(deepseek_results["iterations"]))
                
                claude_domain_tokens += claude_results["tokens_generated"]
                deepseek_domain_tokens += deepseek_results["tokens_generated"]
            
            claude_avg_rating = claude_domain_rating / claude_domain_count if claude_domain_count > 0 else 0
            deepseek_avg_rating = deepseek_domain_rating / deepseek_domain_count if deepseek_domain_count > 0 else 0
            
            claude_cost = claude_domain_tokens * TOKEN_COST_CLAUDE
            deepseek_cost = deepseek_domain_tokens * TOKEN_COST_DEEPSEEK
            
            claude_domain_ppd = claude_avg_rating / claude_cost if claude_cost > 0 else 0
            deepseek_domain_ppd = deepseek_avg_rating / deepseek_cost if deepseek_cost > 0 else 0
            
            claude_ppd.append(claude_domain_ppd)
            deepseek_ppd.append(deepseek_domain_ppd)
        
        # Plot
        plt.bar(x - width/2, claude_ppd, width, label='Claude')
        plt.bar(x + width/2, deepseek_ppd, width, label='DeepSeek')
        
        plt.xlabel('Domain')
        plt.ylabel('Performance per Dollar (Rating / $)')
        plt.title('Performance per Dollar by Domain')
        plt.xticks(x, domains)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Add values above bars
        for i, v in enumerate(claude_ppd):
            if v > 0:
                plt.text(i - width/2, v + max(deepseek_ppd) * 0.05, f"{v:.2f}", ha='center')
        
        for i, v in enumerate(deepseek_ppd):
            if v > 0:
                plt.text(i + width/2, v + max(deepseek_ppd) * 0.05, f"{v:.2f}", ha='center')
        
        # Save
        plt.savefig(os.path.join(plots_dir, "performance_per_dollar.png"))
        plt.close()
    
    def _save_results(self) -> None:
        """Save benchmark results to file."""
        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp}.json"
        filepath = os.path.join(self.config.output_dir, filename)
        
        # Add benchmark configuration to results
        output = {
            "config": {
                "domains": self.config.domains,
                "difficulties": self.config.difficulties,
                "num_iterations": self.config.num_iterations,
                "timeout": self.config.timeout,
                "estimate_tokens": self.config.estimate_tokens,
                "timestamp": timestamp
            },
            "results": self.results
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        self.logger.info(f"Saved benchmark results to {filepath}")
        
        # Also save summary report
        summary_path = os.path.join(self.config.output_dir, f"benchmark_summary_{timestamp}.txt")
        
        with open(summary_path, 'w') as f:
            f.write("=== CTM-AbsoluteZero Agentic Model Comparison ===\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Domains: {', '.join(self.config.domains)}\n")
            f.write(f"Difficulties: {', '.join(self.config.difficulties)}\n")
            f.write(f"Iterations: {self.config.num_iterations} per configuration\n\n")
            
            summary = self.results["summary"]
            
            f.write("=== SUMMARY ===\n\n")
            
            f.write("Claude:\n")
            f.write(f"Success Rate: {summary['claude']['success_rate']:.1%}\n")
            f.write(f"Avg Rating: {summary['claude']['avg_rating']:.2f}\n")
            f.write(f"Avg Time: {summary['claude']['avg_time']:.2f}s\n")
            if self.config.estimate_tokens:
                f.write(f"Total Tokens: {summary['claude']['total_tokens']}\n")
                f.write(f"Est. Cost: ${summary['claude']['estimated_cost']:.4f}\n")
            
            f.write("\nDeepSeek:\n")
            f.write(f"Success Rate: {summary['deepseek']['success_rate']:.1%}\n")
            f.write(f"Avg Rating: {summary['deepseek']['avg_rating']:.2f}\n")
            f.write(f"Avg Time: {summary['deepseek']['avg_time']:.2f}s\n")
            if self.config.estimate_tokens:
                f.write(f"Total Tokens: {summary['deepseek']['total_tokens']}\n")
                f.write(f"Est. Cost: ${summary['deepseek']['estimated_cost']:.4f}\n")
            
            f.write("\nComparison (Claude/DeepSeek ratio):\n")
            f.write(f"Rating Ratio: {summary['comparison']['rating_ratio']:.2f}x\n")
            f.write(f"Success Rate Ratio: {summary['comparison']['success_ratio']:.2f}x\n")
            f.write(f"Time Ratio: {summary['comparison']['time_ratio']:.2f}x\n")
            if self.config.estimate_tokens:
                f.write(f"Cost Ratio: {summary['comparison']['cost_ratio']:.2f}x\n")
                
                # Performance per dollar
                claude_ppd = summary['claude']['avg_rating'] / summary['claude']['estimated_cost'] if summary['claude']['estimated_cost'] > 0 else 0
                deepseek_ppd = summary['deepseek']['avg_rating'] / summary['deepseek']['estimated_cost'] if summary['deepseek']['estimated_cost'] > 0 else 0
                
                f.write(f"\nPerformance per Dollar:\n")
                f.write(f"Claude: {claude_ppd:.2f}\n")
                f.write(f"DeepSeek: {deepseek_ppd:.2f}\n")
                
                if claude_ppd > 0 and deepseek_ppd > 0:
                    ratio = deepseek_ppd / claude_ppd
                    better = "DeepSeek" if ratio > 1 else "Claude"
                    f.write(f"{better} has {max(ratio, 1/ratio):.1f}x better performance per dollar\n")
            
            f.write("\n=== DOMAIN DETAILS ===\n\n")
            
            for domain in self.config.domains:
                f.write(f"Domain: {domain}\n")
                
                claude_domain = summary["claude"]["domains"].get(domain, {})
                deepseek_domain = summary["deepseek"]["domains"].get(domain, {})
                
                if claude_domain:
                    f.write(f"  Claude Success Rate: {claude_domain.get('success_rate', 0):.1%}\n")
                    f.write(f"  Claude Avg Rating: {claude_domain.get('avg_rating', 0):.2f}\n")
                    f.write(f"  Claude Avg Time: {claude_domain.get('avg_time', 0):.2f}s\n")
                
                if deepseek_domain:
                    f.write(f"  DeepSeek Success Rate: {deepseek_domain.get('success_rate', 0):.1%}\n")
                    f.write(f"  DeepSeek Avg Rating: {deepseek_domain.get('avg_rating', 0):.2f}\n")
                    f.write(f"  DeepSeek Avg Time: {deepseek_domain.get('avg_time', 0):.2f}s\n")
                
                f.write("\n")
            
            f.write("\n=== CONCLUSION ===\n\n")
            
            # Performance analysis
            if summary["comparison"]["rating_ratio"] > 1.1:
                f.write(f"Claude outperforms DeepSeek by {(summary['comparison']['rating_ratio'] - 1) * 100:.1f}% in terms of solution quality.\n")
            elif summary["comparison"]["rating_ratio"] < 0.9:
                f.write(f"DeepSeek outperforms Claude by {(1 - summary['comparison']['rating_ratio']) * 100:.1f}% in terms of solution quality.\n")
            else:
                f.write("Claude and DeepSeek perform similarly in terms of solution quality.\n")
            
            # Cost analysis
            if self.config.estimate_tokens and summary["comparison"]["cost_ratio"] > 0:
                f.write(f"\nClaude is approximately {summary['comparison']['cost_ratio']:.1f}x more expensive than DeepSeek.\n")
                
                # Value recommendation
                if summary["comparison"]["rating_ratio"] < 0.9 or (summary["comparison"]["rating_ratio"] < 1.1 and summary["comparison"]["cost_ratio"] > 5):
                    f.write("\nRecommendation: Consider using DeepSeek for most tasks due to its favorable cost-performance ratio.\n")
                elif summary["comparison"]["rating_ratio"] > 1.5 and summary["comparison"]["cost_ratio"] < 10:
                    f.write("\nRecommendation: Consider using Claude for high-value tasks where quality is critical.\n")
                else:
                    f.write("\nRecommendation: Use a mix of both models - DeepSeek for routine tasks and Claude for critical tasks.\n")
        
        self.logger.info(f"Saved benchmark summary to {summary_path}")

async def main():
    """Run benchmark from command line."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Agentic Model Comparison Benchmark")
    
    # Add arguments
    parser.add_argument(
        "--config", "-c",
        help="Path to agent configuration file",
        type=str
    )
    parser.add_argument(
        "--domains", "-d",
        help="Comma-separated list of domains to benchmark",
        type=str
    )
    parser.add_argument(
        "--difficulties", "-l",
        help="Comma-separated list of difficulties to benchmark",
        type=str
    )
    parser.add_argument(
        "--iterations", "-i",
        help="Number of iterations per configuration",
        type=int,
        default=5
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
        "--no-tokens",
        help="Disable token usage estimation",
        action="store_true"
    )
    parser.add_argument(
        "--no-plots",
        help="Disable plot generation",
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

if __name__ == "__main__":
    asyncio.run(main())