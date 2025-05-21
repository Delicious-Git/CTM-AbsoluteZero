#!/usr/bin/env python3
"""
Benchmark utility for comparing Claude and DeepSeek language models.
This script performs comprehensive benchmarking between Claude and DeepSeek
across various metrics including token efficiency, response quality, and performance.
"""

import os
import sys
import time
import json
import argparse
import asyncio
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

# Import CTM-AZ components - adapter implementation will need to be created
from src.llm.claude_adapter import ClaudeAdapter
from src.llm.deepseek_adapter import DeepSeekAdapter

# Constants
BENCHMARK_DOMAINS = ["general", "quantum", "maze", "sorting", "image_classification"]
TASK_TYPES = ["generation", "solving", "analysis", "optimization"]
COMPLEXITY_LEVELS = ["simple", "medium", "complex"]


class LLMBenchmark:
    """
    Benchmark utility for comparing language models performance.
    """
    
    def __init__(
        self,
        claude_config: Optional[Dict[str, Any]] = None,
        deepseek_config: Optional[Dict[str, Any]] = None,
        output_dir: str = "./benchmark_results"
    ):
        """
        Initialize the benchmark utility.
        
        Args:
            claude_config: Configuration for Claude adapter
            deepseek_config: Configuration for DeepSeek adapter
            output_dir: Directory for saving benchmark results
        """
        self.claude_config = claude_config or {}
        self.deepseek_config = deepseek_config or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize adapters
        self.claude_adapter = ClaudeAdapter(**self.claude_config)
        self.deepseek_adapter = DeepSeekAdapter(**self.deepseek_config)
        
        # Results storage
        self.results = {
            "claude": {},
            "deepseek": {},
            "comparison": {}
        }
        
        # Prompt templates - will be expanded for different domains
        self._load_prompt_templates()
    
    def _load_prompt_templates(self):
        """Load prompt templates for different tasks and domains."""
        # Load from templates directory or define inline
        templates_path = Path(project_root) / "templates" / "benchmark_prompts.json"
        
        if templates_path.exists():
            with open(templates_path, "r") as f:
                self.prompt_templates = json.load(f)
        else:
            # Default templates for each domain and task type
            self.prompt_templates = {
                "general": {
                    "generation": "Generate a detailed plan for {topic} with at least 5 main points and 3 sub-points each.",
                    "solving": "Solve the following problem step-by-step: {problem}",
                    "analysis": "Analyze the following text and provide key insights: {text}",
                    "optimization": "Optimize the following process for efficiency: {process}"
                },
                "quantum": {
                    "generation": "Design a quantum circuit that can {objective} using {num_qubits} qubits.",
                    "solving": "Solve this quantum algorithm problem: {problem}",
                    "analysis": "Analyze the performance of this quantum circuit: {circuit}",
                    "optimization": "Optimize the following quantum algorithm to reduce circuit depth: {algorithm}"
                },
                "maze": {
                    "generation": "Design a maze-solving algorithm for a {size_x}x{size_y} maze with {complexity} complexity.",
                    "solving": "Find the shortest path through this maze: {maze}",
                    "analysis": "Analyze the effectiveness of the following maze-solving approach: {approach}",
                    "optimization": "Optimize this maze-solving algorithm for better performance: {algorithm}"
                },
                "sorting": {
                    "generation": "Create a sorting algorithm for {dataset_type} data with {constraints}.",
                    "solving": "Sort the following array and explain your approach: {array}",
                    "analysis": "Compare the efficiency of these sorting algorithms: {algorithms}",
                    "optimization": "Optimize this sorting implementation for better performance: {implementation}"
                },
                "image_classification": {
                    "generation": "Design a neural network architecture for classifying {categories} with {constraints}.",
                    "solving": "How would you classify this image? {image_description}",
                    "analysis": "Analyze the performance of this image classification model: {model}",
                    "optimization": "Optimize this CNN architecture for better accuracy: {architecture}"
                }
            }
    
    def generate_benchmark_tasks(
        self,
        domains: Optional[List[str]] = None,
        task_types: Optional[List[str]] = None,
        complexity_levels: Optional[List[str]] = None,
        count_per_combination: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate benchmark tasks for specified domains and task types.
        
        Args:
            domains: List of domains to include
            task_types: List of task types to include
            complexity_levels: List of complexity levels to include
            count_per_combination: Number of tasks per domain/type/complexity combination
            
        Returns:
            List of benchmark tasks
        """
        domains = domains or BENCHMARK_DOMAINS
        task_types = task_types or TASK_TYPES
        complexity_levels = complexity_levels or COMPLEXITY_LEVELS
        
        tasks = []
        
        for domain in domains:
            for task_type in task_types:
                for complexity in complexity_levels:
                    for i in range(count_per_combination):
                        # Generate parameter values based on domain and task type
                        params = self._generate_task_parameters(domain, task_type, complexity)
                        
                        # Generate prompt using template and parameters
                        template = self.prompt_templates.get(domain, {}).get(task_type, "")
                        prompt = template.format(**params)
                        
                        task = {
                            "id": f"{domain}_{task_type}_{complexity}_{i+1}",
                            "domain": domain,
                            "task_type": task_type,
                            "complexity": complexity,
                            "prompt": prompt,
                            "parameters": params
                        }
                        
                        tasks.append(task)
        
        return tasks
    
    def _generate_task_parameters(
        self,
        domain: str,
        task_type: str,
        complexity: str
    ) -> Dict[str, Any]:
        """
        Generate parameters for a specific task.
        
        Args:
            domain: Task domain
            task_type: Task type
            complexity: Task complexity
            
        Returns:
            Dictionary of parameters
        """
        # Parameter generation based on domain, task type and complexity
        if domain == "general":
            return self._generate_general_params(task_type, complexity)
        elif domain == "quantum":
            return self._generate_quantum_params(task_type, complexity)
        elif domain == "maze":
            return self._generate_maze_params(task_type, complexity)
        elif domain == "sorting":
            return self._generate_sorting_params(task_type, complexity)
        elif domain == "image_classification":
            return self._generate_image_params(task_type, complexity)
        else:
            return {}
    
    def _generate_general_params(self, task_type: str, complexity: str) -> Dict[str, Any]:
        """Generate parameters for general domain tasks."""
        topics = [
            "project management", "software development", "data analysis",
            "marketing strategy", "research methodology", "product design",
            "user experience optimization", "knowledge management",
            "artificial intelligence ethics", "business strategy"
        ]
        
        problems = [
            "Optimize a website's performance for better user engagement",
            "Design a data pipeline that can handle 1 million events per second",
            "Create an algorithm for recommending products to users based on past purchases",
            "Develop a strategy for reducing customer churn by 20%",
            "Design a system for detecting fraud in financial transactions"
        ]
        
        complexity_factor = {"simple": 1, "medium": 2, "complex": 3}
        factor = complexity_factor.get(complexity, 1)
        
        if task_type == "generation":
            return {"topic": np.random.choice(topics)}
        elif task_type == "solving":
            return {"problem": np.random.choice(problems)}
        elif task_type == "analysis":
            text_length = 100 * factor
            return {"text": f"Sample text for analysis ({text_length} words)..."}
        elif task_type == "optimization":
            return {"process": f"Sample process for optimization (complexity: {complexity})..."}
        else:
            return {}
    
    def _generate_quantum_params(self, task_type: str, complexity: str) -> Dict[str, Any]:
        """Generate parameters for quantum domain tasks."""
        objectives = [
            "implement Grover's search algorithm",
            "perform quantum Fourier transform",
            "create a variational quantum eigensolver",
            "simulate a quantum random walk",
            "implement quantum phase estimation"
        ]
        
        problems = [
            "Calculate the ground state energy of a H2 molecule using VQE",
            "Implement Shor's algorithm for factoring 15",
            "Create a quantum circuit for solving the Deutsch-Jozsa problem",
            "Design a quantum teleportation protocol",
            "Optimize a QAOA circuit for the Max-Cut problem"
        ]
        
        complexity_factor = {"simple": 1, "medium": 2, "complex": 3}
        factor = complexity_factor.get(complexity, 1)
        num_qubits = 2 + factor * 2
        
        if task_type == "generation":
            return {
                "objective": np.random.choice(objectives),
                "num_qubits": num_qubits
            }
        elif task_type == "solving":
            return {"problem": np.random.choice(problems)}
        elif task_type == "analysis":
            circuit = f"Sample quantum circuit with {num_qubits} qubits and depth {factor * 3}..."
            return {"circuit": circuit}
        elif task_type == "optimization":
            algorithm = f"Sample quantum algorithm with {num_qubits} qubits..."
            return {"algorithm": algorithm}
        else:
            return {}
    
    def _generate_maze_params(self, task_type: str, complexity: str) -> Dict[str, Any]:
        """Generate parameters for maze domain tasks."""
        complexity_factor = {"simple": 1, "medium": 2, "complex": 3}
        factor = complexity_factor.get(complexity, 1)
        
        size_x = 5 * factor
        size_y = 5 * factor
        maze_complexity = 0.3 * factor
        
        approaches = [
            "Depth-First Search", "Breadth-First Search", "A* algorithm",
            "Dijkstra's algorithm", "Wall follower", "Random mouse algorithm"
        ]
        
        if task_type == "generation":
            return {
                "size_x": size_x,
                "size_y": size_y,
                "complexity": f"{maze_complexity:.1f}"
            }
        elif task_type == "solving":
            maze = f"Sample {size_x}x{size_y} maze representation..."
            return {"maze": maze}
        elif task_type == "analysis":
            approach = np.random.choice(approaches)
            return {"approach": approach}
        elif task_type == "optimization":
            algorithm = f"Sample maze-solving algorithm using {np.random.choice(approaches)}..."
            return {"algorithm": algorithm}
        else:
            return {}
    
    def _generate_sorting_params(self, task_type: str, complexity: str) -> Dict[str, Any]:
        """Generate parameters for sorting domain tasks."""
        complexity_factor = {"simple": 1, "medium": 2, "complex": 3}
        factor = complexity_factor.get(complexity, 1)
        
        dataset_types = [
            "nearly sorted", "reverse sorted", "random integers",
            "strings with varying lengths", "objects with multiple keys"
        ]
        
        constraints = [
            "minimal memory usage", "optimal time complexity",
            "stable sorting requirements", "parallel execution",
            "large datasets (millions of elements)"
        ]
        
        algorithms = [
            "Quick Sort", "Merge Sort", "Heap Sort", 
            "Insertion Sort", "Bubble Sort", "Radix Sort"
        ]
        
        array_size = 10 * factor
        array = f"[sample array with {array_size} elements...]"
        
        if task_type == "generation":
            return {
                "dataset_type": np.random.choice(dataset_types),
                "constraints": np.random.choice(constraints)
            }
        elif task_type == "solving":
            return {"array": array}
        elif task_type == "analysis":
            alg_selection = np.random.sample(algorithms, size=min(factor+1, len(algorithms)))
            return {"algorithms": ", ".join(alg_selection)}
        elif task_type == "optimization":
            implementation = f"Sample implementation of {np.random.choice(algorithms)}..."
            return {"implementation": implementation}
        else:
            return {}
    
    def _generate_image_params(self, task_type: str, complexity: str) -> Dict[str, Any]:
        """Generate parameters for image classification domain tasks."""
        complexity_factor = {"simple": 1, "medium": 2, "complex": 3}
        factor = complexity_factor.get(complexity, 1)
        
        categories = [
            "cats and dogs", "handwritten digits", "facial expressions",
            "medical images", "satellite imagery", "traffic signs"
        ]
        
        constraints = [
            "limited training data", "real-time inference requirements",
            "deployment on mobile devices", "high accuracy requirements",
            "interpretability requirements"
        ]
        
        if task_type == "generation":
            return {
                "categories": np.random.choice(categories),
                "constraints": np.random.choice(constraints)
            }
        elif task_type == "solving":
            return {"image_description": f"Sample image description (complexity: {complexity})..."}
        elif task_type == "analysis":
            return {"model": f"Sample CNN model for {np.random.choice(categories)} classification..."}
        elif task_type == "optimization":
            return {"architecture": f"Sample CNN architecture with {3*factor} layers..."}
        else:
            return {}
    
    async def run_benchmark(
        self,
        tasks: List[Dict[str, Any]],
        max_concurrent: int = 5
    ) -> Dict[str, Any]:
        """
        Run the benchmark for all specified tasks.
        
        Args:
            tasks: List of benchmark tasks
            max_concurrent: Maximum number of concurrent requests
            
        Returns:
            Benchmark results
        """
        print(f"Running benchmark with {len(tasks)} tasks...")
        
        # Create semaphore for limiting concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Create tasks for Claude
        claude_tasks = [
            self._run_single_benchmark(task, "claude", semaphore)
            for task in tasks
        ]
        
        # Create tasks for DeepSeek
        deepseek_tasks = [
            self._run_single_benchmark(task, "deepseek", semaphore)
            for task in tasks
        ]
        
        # Run all tasks concurrently (with concurrency limit)
        print("Running Claude benchmarks...")
        claude_results = await asyncio.gather(*claude_tasks)
        
        print("Running DeepSeek benchmarks...")
        deepseek_results = await asyncio.gather(*deepseek_tasks)
        
        # Process results
        for result in claude_results:
            task_id = result["task_id"]
            self.results["claude"][task_id] = result
        
        for result in deepseek_results:
            task_id = result["task_id"]
            self.results["deepseek"][task_id] = result
        
        # Generate comparison metrics
        self._generate_comparison_metrics()
        
        return self.results
    
    async def _run_single_benchmark(
        self,
        task: Dict[str, Any],
        model: str,
        semaphore: asyncio.Semaphore
    ) -> Dict[str, Any]:
        """
        Run a single benchmark task.
        
        Args:
            task: Benchmark task
            model: Model to use ("claude" or "deepseek")
            semaphore: Semaphore for limiting concurrent requests
            
        Returns:
            Benchmark result
        """
        async with semaphore:
            task_id = task["id"]
            prompt = task["prompt"]
            
            print(f"Running {model} benchmark for task {task_id}...")
            
            try:
                # Select the appropriate adapter
                adapter = self.claude_adapter if model == "claude" else self.deepseek_adapter
                
                # Measure performance
                start_time = time.time()
                response, metrics = await adapter.generate_with_metrics(prompt)
                end_time = time.time()
                
                # Calculate metrics
                execution_time = end_time - start_time
                
                result = {
                    "task_id": task_id,
                    "model": model,
                    "domain": task["domain"],
                    "task_type": task["task_type"],
                    "complexity": task["complexity"],
                    "prompt": prompt,
                    "response": response,
                    "execution_time": execution_time,
                    "prompt_tokens": metrics.get("prompt_tokens", 0),
                    "completion_tokens": metrics.get("completion_tokens", 0),
                    "total_tokens": metrics.get("total_tokens", 0),
                    "tokens_per_second": metrics.get("total_tokens", 0) / execution_time,
                    "status": "success"
                }
                
                return result
            except Exception as e:
                # Handle errors
                return {
                    "task_id": task_id,
                    "model": model,
                    "domain": task["domain"],
                    "task_type": task["task_type"],
                    "complexity": task["complexity"],
                    "prompt": prompt,
                    "error": str(e),
                    "status": "error"
                }
    
    def _generate_comparison_metrics(self):
        """Generate comparison metrics between models."""
        # Initialize comparison metrics
        comparisons = {}
        
        # Get all unique task IDs
        task_ids = set(self.results["claude"].keys()) | set(self.results["deepseek"].keys())
        
        for task_id in task_ids:
            # Skip if either result is missing
            if task_id not in self.results["claude"] or task_id not in self.results["deepseek"]:
                continue
            
            claude_result = self.results["claude"][task_id]
            deepseek_result = self.results["deepseek"][task_id]
            
            # Skip error results
            if claude_result.get("status") != "success" or deepseek_result.get("status") != "success":
                continue
            
            # Calculate efficiency metrics
            token_reduction = 1.0 - (deepseek_result["total_tokens"] / claude_result["total_tokens"])
            time_reduction = 1.0 - (deepseek_result["execution_time"] / claude_result["execution_time"])
            
            comparison = {
                "task_id": task_id,
                "domain": claude_result["domain"],
                "task_type": claude_result["task_type"],
                "complexity": claude_result["complexity"],
                "claude_tokens": claude_result["total_tokens"],
                "deepseek_tokens": deepseek_result["total_tokens"],
                "token_reduction": token_reduction,
                "claude_time": claude_result["execution_time"],
                "deepseek_time": deepseek_result["execution_time"],
                "time_reduction": time_reduction,
                "claude_tokens_per_second": claude_result["tokens_per_second"],
                "deepseek_tokens_per_second": deepseek_result["tokens_per_second"],
            }
            
            comparisons[task_id] = comparison
        
        self.results["comparison"] = comparisons
        
        # Calculate aggregate metrics
        self._calculate_aggregate_metrics()
    
    def _calculate_aggregate_metrics(self):
        """Calculate aggregate metrics across all tasks."""
        comparisons = list(self.results["comparison"].values())
        
        if not comparisons:
            return
        
        # Initialize aggregate metrics
        aggregates = {
            "overall": {
                "count": len(comparisons),
                "avg_token_reduction": np.mean([c["token_reduction"] for c in comparisons]),
                "avg_time_reduction": np.mean([c["time_reduction"] for c in comparisons]),
                "median_token_reduction": np.median([c["token_reduction"] for c in comparisons]),
                "median_time_reduction": np.median([c["time_reduction"] for c in comparisons]),
            }
        }
        
        # Calculate metrics by domain
        domains = set(c["domain"] for c in comparisons)
        for domain in domains:
            domain_comps = [c for c in comparisons if c["domain"] == domain]
            aggregates[f"domain_{domain}"] = {
                "count": len(domain_comps),
                "avg_token_reduction": np.mean([c["token_reduction"] for c in domain_comps]),
                "avg_time_reduction": np.mean([c["time_reduction"] for c in domain_comps]),
                "median_token_reduction": np.median([c["token_reduction"] for c in domain_comps]),
                "median_time_reduction": np.median([c["time_reduction"] for c in domain_comps]),
            }
        
        # Calculate metrics by task type
        task_types = set(c["task_type"] for c in comparisons)
        for task_type in task_types:
            type_comps = [c for c in comparisons if c["task_type"] == task_type]
            aggregates[f"type_{task_type}"] = {
                "count": len(type_comps),
                "avg_token_reduction": np.mean([c["token_reduction"] for c in type_comps]),
                "avg_time_reduction": np.mean([c["time_reduction"] for c in type_comps]),
                "median_token_reduction": np.median([c["token_reduction"] for c in type_comps]),
                "median_time_reduction": np.median([c["time_reduction"] for c in type_comps]),
            }
        
        # Calculate metrics by complexity
        complexities = set(c["complexity"] for c in comparisons)
        for complexity in complexities:
            complexity_comps = [c for c in comparisons if c["complexity"] == complexity]
            aggregates[f"complexity_{complexity}"] = {
                "count": len(complexity_comps),
                "avg_token_reduction": np.mean([c["token_reduction"] for c in complexity_comps]),
                "avg_time_reduction": np.mean([c["time_reduction"] for c in complexity_comps]),
                "median_token_reduction": np.median([c["token_reduction"] for c in complexity_comps]),
                "median_time_reduction": np.median([c["time_reduction"] for c in complexity_comps]),
            }
        
        self.results["aggregates"] = aggregates
    
    def plot_results(self):
        """Generate visualization plots for benchmark results."""
        # Skip if no comparison data
        if not self.results.get("comparison"):
            print("No comparison data available for plotting.")
            return
        
        # Create output directory for plots
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Create comparison DataFrame
        comparisons = list(self.results["comparison"].values())
        df = pd.DataFrame(comparisons)
        
        # Plot 1: Token reduction by domain
        self._plot_by_category(
            df, 
            "domain", 
            "token_reduction", 
            "Token Reduction by Domain", 
            plots_dir / "token_reduction_by_domain.png"
        )
        
        # Plot 2: Token reduction by task type
        self._plot_by_category(
            df, 
            "task_type", 
            "token_reduction", 
            "Token Reduction by Task Type", 
            plots_dir / "token_reduction_by_task_type.png"
        )
        
        # Plot 3: Token reduction by complexity
        self._plot_by_category(
            df, 
            "complexity", 
            "token_reduction", 
            "Token Reduction by Complexity", 
            plots_dir / "token_reduction_by_complexity.png"
        )
        
        # Plot 4: Scatter plot of token counts
        self._plot_token_scatter(df, plots_dir / "token_scatter.png")
        
        # Plot 5: ROI analysis
        self._plot_roi_analysis(df, plots_dir / "roi_analysis.png")
    
    def _plot_by_category(
        self,
        df: pd.DataFrame,
        category: str,
        metric: str,
        title: str,
        output_path: Path
    ):
        """
        Generate a bar plot for a metric grouped by category.
        
        Args:
            df: DataFrame with comparison data
            category: Category column for grouping
            metric: Metric to plot
            title: Plot title
            output_path: Output file path
        """
        plt.figure(figsize=(10, 6))
        
        # Calculate mean metric by category
        grouped = df.groupby(category)[metric].mean().sort_values(ascending=False)
        
        # Create bar plot
        ax = grouped.plot(kind='bar', color='skyblue')
        
        # Add value labels on top of bars
        for i, v in enumerate(grouped):
            ax.text(i, v + 0.01, f"{v:.2f}", ha='center')
        
        # Set labels and title
        ax.set_xlabel(category.capitalize())
        ax.set_ylabel(f"{metric.replace('_', ' ').title()} %")
        ax.set_title(title)
        
        # Add grid
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add a horizontal line at y=0
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def _plot_token_scatter(self, df: pd.DataFrame, output_path: Path):
        """
        Generate a scatter plot comparing token counts between models.
        
        Args:
            df: DataFrame with comparison data
            output_path: Output file path
        """
        plt.figure(figsize=(10, 8))
        
        # Create scatter plot
        plt.scatter(df['claude_tokens'], df['deepseek_tokens'], alpha=0.7, 
                    c=df['token_reduction'], cmap='RdYlGn')
        
        # Add color bar
        cbar = plt.colorbar()
        cbar.set_label('Token Reduction %')
        
        # Add diagonal line (y = x)
        max_val = max(df['claude_tokens'].max(), df['deepseek_tokens'].max())
        plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
        
        # Set labels and title
        plt.xlabel('Claude Tokens')
        plt.ylabel('DeepSeek Tokens')
        plt.title('Token Usage Comparison: Claude vs DeepSeek')
        
        # Add grid
        plt.grid(linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def _plot_roi_analysis(self, df: pd.DataFrame, output_path: Path):
        """
        Generate ROI analysis plot.
        
        Args:
            df: DataFrame with comparison data
            output_path: Output file path
        """
        plt.figure(figsize=(12, 8))
        
        # Calculate token cost estimates (fictional values for illustration)
        claude_token_cost = 0.00003  # $0.03 per 1000 tokens
        deepseek_token_cost = 0.00001  # $0.01 per 1000 tokens
        
        df['claude_cost'] = df['claude_tokens'] * claude_token_cost
        df['deepseek_cost'] = df['deepseek_tokens'] * deepseek_token_cost
        df['cost_savings'] = df['claude_cost'] - df['deepseek_cost']
        df['cost_reduction_percent'] = (df['cost_savings'] / df['claude_cost']) * 100
        
        # Group by domain for analysis
        grouped = df.groupby('domain').agg({
            'claude_tokens': 'sum',
            'deepseek_tokens': 'sum',
            'claude_cost': 'sum',
            'deepseek_cost': 'sum',
            'cost_savings': 'sum',
            'cost_reduction_percent': 'mean'
        }).sort_values('cost_reduction_percent', ascending=False)
        
        # Create a subplot with 2 axes
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Cost comparison by domain
        grouped[['claude_cost', 'deepseek_cost']].plot(kind='bar', ax=ax1)
        ax1.set_ylabel('Cost ($)')
        ax1.set_title('Cost Comparison by Domain')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot 2: Cost savings percentage by domain
        grouped['cost_reduction_percent'].plot(kind='bar', color='green', ax=ax2)
        ax2.set_ylabel('Cost Reduction (%)')
        ax2.set_title('Cost Reduction by Domain')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels
        for i, v in enumerate(grouped['cost_reduction_percent']):
            ax2.text(i, v + 1, f"{v:.1f}%", ha='center')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def save_results(self, filename: Optional[str] = None):
        """
        Save benchmark results to a file.
        
        Args:
            filename: Output filename (default: benchmark_results_<timestamp>.json)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        # Clean up results for saving (remove large response texts)
        clean_results = self._clean_results_for_saving()
        
        with open(output_path, 'w') as f:
            json.dump(clean_results, f, indent=2)
        
        print(f"Results saved to {output_path}")
        
        # Also save as CSV for easier analysis
        self._save_as_csv(clean_results)
    
    def _clean_results_for_saving(self) -> Dict[str, Any]:
        """
        Clean up results for saving by removing large texts.
        
        Returns:
            Cleaned results
        """
        clean_results = {
            "claude": {},
            "deepseek": {},
            "comparison": self.results.get("comparison", {}),
            "aggregates": self.results.get("aggregates", {}),
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "claude_config": self.claude_config,
                "deepseek_config": self.deepseek_config,
                "task_count": len(self.results.get("claude", {}))
            }
        }
        
        # Clean up individual results
        for model in ["claude", "deepseek"]:
            for task_id, result in self.results.get(model, {}).items():
                clean_result = result.copy()
                
                # Truncate large text fields
                if "prompt" in clean_result:
                    clean_result["prompt_length"] = len(clean_result["prompt"])
                    if len(clean_result["prompt"]) > 100:
                        clean_result["prompt"] = clean_result["prompt"][:100] + "..."
                
                if "response" in clean_result:
                    clean_result["response_length"] = len(clean_result["response"])
                    if len(clean_result["response"]) > 100:
                        clean_result["response"] = clean_result["response"][:100] + "..."
                
                clean_results[model][task_id] = clean_result
        
        return clean_results
    
    def _save_as_csv(self, results: Dict[str, Any]):
        """
        Save results as CSV files for easier analysis.
        
        Args:
            results: Benchmark results
        """
        # Save comparison data
        if results.get("comparison"):
            comparisons = list(results["comparison"].values())
            df_comp = pd.DataFrame(comparisons)
            df_comp.to_csv(self.output_dir / "comparison_results.csv", index=False)
        
        # Save aggregate data
        if results.get("aggregates"):
            # Convert nested dict to dataframe
            agg_data = []
            for category, metrics in results["aggregates"].items():
                row = {"category": category, **metrics}
                agg_data.append(row)
            
            df_agg = pd.DataFrame(agg_data)
            df_agg.to_csv(self.output_dir / "aggregate_results.csv", index=False)
        
        # Save model-specific metrics
        for model in ["claude", "deepseek"]:
            if results.get(model):
                model_data = []
                for task_id, result in results[model].items():
                    # Extract relevant metrics
                    metrics = {
                        "task_id": task_id,
                        "domain": result.get("domain"),
                        "task_type": result.get("task_type"),
                        "complexity": result.get("complexity"),
                        "prompt_tokens": result.get("prompt_tokens"),
                        "completion_tokens": result.get("completion_tokens"),
                        "total_tokens": result.get("total_tokens"),
                        "execution_time": result.get("execution_time"),
                        "tokens_per_second": result.get("tokens_per_second"),
                        "status": result.get("status")
                    }
                    model_data.append(metrics)
                
                df_model = pd.DataFrame(model_data)
                df_model.to_csv(self.output_dir / f"{model}_results.csv", index=False)


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Benchmark Claude vs DeepSeek LLMs")
    parser.add_argument("--domains", type=str, default="general,quantum,maze",
                      help="Comma-separated list of domains to benchmark")
    parser.add_argument("--task-types", type=str, default="generation,solving",
                      help="Comma-separated list of task types to benchmark")
    parser.add_argument("--complexities", type=str, default="simple,medium,complex",
                      help="Comma-separated list of complexity levels to benchmark")
    parser.add_argument("--count", type=int, default=2,
                      help="Number of tasks per domain/type/complexity combination")
    parser.add_argument("--output-dir", type=str, default="./benchmark_results",
                      help="Output directory for benchmark results")
    parser.add_argument("--concurrent", type=int, default=3,
                      help="Maximum number of concurrent requests")
    
    args = parser.parse_args()
    
    # Parse comma-separated lists
    domains = args.domains.split(",")
    task_types = args.task_types.split(",")
    complexities = args.complexities.split(",")
    
    # Configure the benchmark
    claude_config = {
        # Configure Claude adapter
    }
    
    deepseek_config = {
        # Configure DeepSeek adapter
    }
    
    # Create benchmark instance
    benchmark = LLMBenchmark(
        claude_config=claude_config,
        deepseek_config=deepseek_config,
        output_dir=args.output_dir
    )
    
    # Generate benchmark tasks
    tasks = benchmark.generate_benchmark_tasks(
        domains=domains,
        task_types=task_types,
        complexity_levels=complexities,
        count_per_combination=args.count
    )
    
    print(f"Generated {len(tasks)} benchmark tasks")
    
    # Run the benchmark
    results = await benchmark.run_benchmark(
        tasks=tasks,
        max_concurrent=args.concurrent
    )
    
    # Generate visualization plots
    benchmark.plot_results()
    
    # Save results
    benchmark.save_results()
    
    print("Benchmark completed successfully")


if __name__ == "__main__":
    asyncio.run(main())