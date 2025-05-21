"""
Token Optimizer for Agentic Brain Framework.

This module provides tools for analyzing and optimizing token usage across
different LLM implementations, with a focus on efficiency for DeepSeek models.
"""
import json
import re
import os
import time
from typing import Dict, List, Any, Optional, Union, Tuple, Set
import numpy as np

class TokenAnalyzer:
    """Analyzes token usage patterns in LLM requests and responses."""
    
    def __init__(self, log_dir: str = None):
        """
        Initialize token analyzer.
        
        Args:
            log_dir: Directory to save logs (defaults to None, no logging)
        """
        self.log_dir = log_dir
        self.token_logs = []
        
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        # Token counters
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.request_count = 0
        
        # Performance metrics
        self.avg_tokens_per_request = 0
        self.token_efficiency_score = 1.0  # Higher is better
    
    def log_request(
        self,
        model: str,
        prompt: str,
        response: str,
        prompt_tokens: int,
        completion_tokens: int,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Log a request and its token usage.
        
        Args:
            model: Model name
            prompt: Prompt text
            response: Response text
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
            metadata: Additional metadata
            
        Returns:
            Analysis results for this request
        """
        # Update counters
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.request_count += 1
        
        # Calculate metrics
        token_ratio = completion_tokens / prompt_tokens if prompt_tokens > 0 else 0
        total_tokens = prompt_tokens + completion_tokens
        
        # Create log entry
        entry = {
            "timestamp": time.time(),
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "token_ratio": token_ratio,
            "metadata": metadata or {}
        }
        
        # Perform analysis
        analysis = self._analyze_request(entry)
        entry["analysis"] = analysis
        
        # Add to logs
        self.token_logs.append(entry)
        
        # Save logs if directory provided
        if self.log_dir:
            self._save_logs()
        
        # Update aggregate metrics
        self._update_metrics()
        
        return analysis
    
    def _analyze_request(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a request for optimization opportunities.
        
        Args:
            entry: Request log entry
            
        Returns:
            Analysis results
        """
        analysis = {
            "efficiency_score": 0.0,
            "optimization_potential": 0.0,
            "observations": [],
            "recommendations": []
        }
        
        # Calculate efficiency score
        # Higher token_ratio is better (more output for input)
        token_ratio = entry["token_ratio"]
        total_tokens = entry["total_tokens"]
        
        # Base efficiency score
        if token_ratio > 2.0:
            # Very efficient (more than 2:1 output to input)
            base_score = 0.9
        elif token_ratio > 1.0:
            # Good efficiency (more output than input)
            base_score = 0.7
        elif token_ratio > 0.5:
            # Moderate efficiency
            base_score = 0.5
        else:
            # Poor efficiency (much more input than output)
            base_score = 0.3
        
        # Adjust for total token count
        if total_tokens < 1000:
            # Very small request, good
            size_factor = 1.1
        elif total_tokens < 5000:
            # Moderate size, neutral
            size_factor = 1.0
        else:
            # Large request, penalize
            size_factor = 0.9 - min(0.4, (total_tokens - 5000) / 20000)
        
        # Efficiency score
        efficiency_score = min(1.0, base_score * size_factor)
        analysis["efficiency_score"] = efficiency_score
        
        # Optimization potential (inverse of efficiency)
        # Higher means more room for improvement
        analysis["optimization_potential"] = 1.0 - efficiency_score
        
        # Add observations
        if token_ratio < 0.3:
            analysis["observations"].append("Very low output-to-input token ratio")
        
        if total_tokens > 10000:
            analysis["observations"].append("Extremely large token usage")
        
        if entry["prompt_tokens"] > 3000:
            analysis["observations"].append("Large prompt size")
        
        # Add recommendations
        if entry["prompt_tokens"] > 3000:
            analysis["recommendations"].append("Consider reducing prompt size through summarization or chunking")
        
        if token_ratio < 0.3:
            analysis["recommendations"].append("Request more specific, concise responses")
        
        if total_tokens > 10000:
            analysis["recommendations"].append("Consider breaking down into multiple smaller requests")
        
        return analysis
    
    def _update_metrics(self) -> None:
        """Update aggregate metrics."""
        if self.request_count > 0:
            self.avg_tokens_per_request = (
                self.total_prompt_tokens + self.total_completion_tokens
            ) / self.request_count
            
            # Calculate weighted efficiency score
            efficiency_scores = [
                entry["analysis"]["efficiency_score"] * entry["total_tokens"]
                for entry in self.token_logs
            ]
            
            total_tokens = sum(entry["total_tokens"] for entry in self.token_logs)
            
            if total_tokens > 0:
                self.token_efficiency_score = sum(efficiency_scores) / total_tokens
    
    def _save_logs(self) -> None:
        """Save logs to file."""
        if not self.log_dir:
            return
            
        log_path = os.path.join(self.log_dir, "token_logs.json")
        
        with open(log_path, "w") as f:
            json.dump(self.token_logs, f, indent=2)
    
    def get_token_usage_summary(self) -> Dict[str, Any]:
        """
        Get summary of token usage.
        
        Returns:
            Token usage summary
        """
        summary = {
            "total_requests": self.request_count,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "avg_tokens_per_request": self.avg_tokens_per_request,
            "efficiency_score": self.token_efficiency_score,
            "models": {}
        }
        
        # Group by model
        model_data = {}
        
        for entry in self.token_logs:
            model = entry["model"]
            
            if model not in model_data:
                model_data[model] = {
                    "request_count": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "efficiency_scores": []
                }
            
            model_data[model]["request_count"] += 1
            model_data[model]["prompt_tokens"] += entry["prompt_tokens"]
            model_data[model]["completion_tokens"] += entry["completion_tokens"]
            model_data[model]["efficiency_scores"].append(entry["analysis"]["efficiency_score"])
        
        # Calculate model-specific metrics
        for model, data in model_data.items():
            avg_efficiency = sum(data["efficiency_scores"]) / len(data["efficiency_scores"])
            total_tokens = data["prompt_tokens"] + data["completion_tokens"]
            
            summary["models"][model] = {
                "request_count": data["request_count"],
                "prompt_tokens": data["prompt_tokens"],
                "completion_tokens": data["completion_tokens"],
                "total_tokens": total_tokens,
                "avg_tokens_per_request": total_tokens / data["request_count"],
                "efficiency_score": avg_efficiency
            }
        
        return summary

class TokenOptimizer:
    """Optimizes prompts and configurations for token efficiency."""
    
    def __init__(
        self, 
        analyzer: TokenAnalyzer = None,
        optimization_level: str = "balanced"
    ):
        """
        Initialize token optimizer.
        
        Args:
            analyzer: Token analyzer to use
            optimization_level: Optimization aggressiveness
                ("minimal", "balanced", "aggressive")
        """
        self.analyzer = analyzer or TokenAnalyzer()
        self.optimization_level = optimization_level
        
        # Configure optimization levels
        self.optimization_params = {
            "minimal": {
                "max_prompt_tokens": 4000,
                "compression_ratio": 0.9,
                "context_retention": 0.95
            },
            "balanced": {
                "max_prompt_tokens": 3000,
                "compression_ratio": 0.7,
                "context_retention": 0.9
            },
            "aggressive": {
                "max_prompt_tokens": 2000,
                "compression_ratio": 0.5,
                "context_retention": 0.8
            }
        }
    
    def optimize_prompt(self, prompt: str, model: str = "deepseek") -> str:
        """
        Optimize a prompt for token efficiency.
        
        Args:
            prompt: Original prompt
            model: Target model
            
        Returns:
            Optimized prompt
        """
        # Get optimization parameters
        params = self.optimization_params.get(
            self.optimization_level,
            self.optimization_params["balanced"]
        )
        
        # Estimate token count (rough approximation)
        token_count = self._estimate_token_count(prompt)
        
        # If under threshold, no optimization needed
        if token_count <= params["max_prompt_tokens"]:
            return prompt
        
        # Apply optimizations based on token count
        if token_count > params["max_prompt_tokens"] * 2:
            # Very long prompt, apply aggressive optimization
            return self._optimize_long_prompt(prompt, params)
        else:
            # Moderately long prompt
            return self._optimize_medium_prompt(prompt, params)
    
    def _estimate_token_count(self, text: str) -> int:
        """
        Estimate token count in text.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        # Rough approximation based on whitespace and punctuation
        # In practice, you would use a proper tokenizer
        words = re.findall(r'\b\w+\b', text)
        punctuation = re.findall(r'[.,!?;:]', text)
        
        # Each word is approximately 1.3 tokens on average
        # Punctuation is roughly 1 token each
        word_tokens = len(words) * 1.3
        punct_tokens = len(punctuation)
        
        return int(word_tokens + punct_tokens)
    
    def _optimize_long_prompt(self, prompt: str, params: Dict[str, Any]) -> str:
        """
        Optimize a very long prompt.
        
        Args:
            prompt: Original prompt
            params: Optimization parameters
            
        Returns:
            Optimized prompt
        """
        # Split into lines
        lines = prompt.split('\n')
        
        # Identify potential sections (chunks separated by blank lines)
        sections = []
        current_section = []
        
        for line in lines:
            if line.strip() == "":
                if current_section:
                    sections.append('\n'.join(current_section))
                    current_section = []
            else:
                current_section.append(line)
        
        if current_section:
            sections.append('\n'.join(current_section))
        
        # Prioritize sections
        prioritized_sections = self._prioritize_sections(sections)
        
        # Keep only essential sections
        essential_count = max(1, int(len(prioritized_sections) * params["context_retention"]))
        essential_sections = prioritized_sections[:essential_count]
        
        # For aggressive optimization, summarize sections
        summarized_sections = []
        for section in essential_sections:
            # Check if section needs summarization
            if self._estimate_token_count(section) > 500:
                summarized_sections.append(self._summarize_section(section, params))
            else:
                summarized_sections.append(section)
        
        # Recombine into optimized prompt
        return '\n\n'.join(summarized_sections)
    
    def _optimize_medium_prompt(self, prompt: str, params: Dict[str, Any]) -> str:
        """
        Optimize a medium-length prompt.
        
        Args:
            prompt: Original prompt
            params: Optimization parameters
            
        Returns:
            Optimized prompt
        """
        # Split into lines
        lines = prompt.split('\n')
        
        # Remove unnecessary blank lines (collapse multiple blank lines)
        filtered_lines = []
        last_was_blank = False
        
        for line in lines:
            is_blank = line.strip() == ""
            
            if is_blank and last_was_blank:
                continue
            
            filtered_lines.append(line)
            last_was_blank = is_blank
        
        # Compress verbose sections
        compressed_lines = self._compress_verbose_sections(filtered_lines, params)
        
        # Recombine into optimized prompt
        return '\n'.join(compressed_lines)
    
    def _prioritize_sections(self, sections: List[str]) -> List[str]:
        """
        Prioritize sections by importance.
        
        Args:
            sections: List of sections
            
        Returns:
            Prioritized list of sections
        """
        # Score sections based on heuristics
        scored_sections = []
        
        for section in sections:
            score = 0
            
            # Key indicators of importance
            if "important" in section.lower() or "note" in section.lower():
                score += 3
            
            if "task" in section.lower() or "problem" in section.lower():
                score += 2
            
            if "example" in section.lower() or "context" in section.lower():
                score += 1
            
            # Length is a weak signal of importance
            score += min(5, len(section) / 500)
            
            # Position - earlier sections usually more important
            score += 0  # Will be added based on index
            
            scored_sections.append((section, score))
        
        # Add position bonus
        for i, (section, score) in enumerate(scored_sections):
            position_bonus = max(0, 10 - i) / 2  # Earlier sections get higher bonus
            scored_sections[i] = (section, score + position_bonus)
        
        # Sort by score descending
        scored_sections.sort(key=lambda x: x[1], reverse=True)
        
        # Return prioritized sections
        return [section for section, _ in scored_sections]
    
    def _summarize_section(self, section: str, params: Dict[str, Any]) -> str:
        """
        Summarize a section to reduce tokens.
        
        Args:
            section: Section text
            params: Optimization parameters
            
        Returns:
            Summarized section
        """
        # Simple heuristic summarization (extract key sentences)
        sentences = re.split(r'(?<=[.!?])\s+', section)
        
        # Calculate target count
        target_count = max(3, int(len(sentences) * params["compression_ratio"]))
        
        # If few sentences, don't summarize
        if len(sentences) <= target_count:
            return section
        
        # Score sentences
        scored_sentences = []
        
        for i, sentence in enumerate(sentences):
            score = 0
            
            # Position matters
            if i == 0:  # First sentence
                score += 3
            elif i == len(sentences) - 1:  # Last sentence
                score += 2
            
            # Length (prefer medium-length sentences)
            words = len(sentence.split())
            if 8 <= words <= 20:
                score += 1
            
            # Content indicators
            if re.search(r'\b(important|key|critical|essential|must|should)\b', sentence, re.I):
                score += 2
            
            # Transition phrases reduce importance
            if re.search(r'\b(however|moreover|furthermore|additionally)\b', sentence, re.I):
                score -= 1
            
            scored_sentences.append((sentence, score))
        
        # Sort by score
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Take top sentences
        top_sentences = [s[0] for s in scored_sentences[:target_count]]
        
        # Sort by original position
        ordered_sentences = []
        for sentence in sentences:
            if sentence in top_sentences:
                ordered_sentences.append(sentence)
        
        # Return summarized section
        return ' '.join(ordered_sentences)
    
    def _compress_verbose_sections(self, lines: List[str], params: Dict[str, Any]) -> List[str]:
        """
        Compress verbose sections by removing redundant content.
        
        Args:
            lines: List of lines
            params: Optimization parameters
            
        Returns:
            Compressed lines
        """
        # Identify verbose sections (consecutive non-blank lines)
        sections = []
        current_section = []
        
        for line in lines:
            if line.strip() == "":
                if current_section:
                    sections.append(current_section)
                    current_section = []
                sections.append([line])  # Keep blank line as separator
            else:
                current_section.append(line)
        
        if current_section:
            sections.append(current_section)
        
        # Compress each section
        compressed_sections = []
        
        for section in sections:
            # Skip blank line separators
            if len(section) == 1 and section[0].strip() == "":
                compressed_sections.append(section)
                continue
            
            # Skip short sections
            if len(section) <= 3:
                compressed_sections.append(section)
                continue
            
            # Compress verbose sections
            if len(section) > 10:
                # Calculate target length
                target_length = max(3, int(len(section) * params["compression_ratio"]))
                
                # Find most important lines
                scored_lines = self._score_lines(section)
                
                # Keep top lines
                top_scores = sorted(scored_lines, key=lambda x: x[1], reverse=True)[:target_length]
                top_indices = [i for i, _ in top_scores]
                top_indices.sort()  # Maintain original order
                
                compressed_section = [section[i] for i in top_indices]
                compressed_sections.append(compressed_section)
            else:
                compressed_sections.append(section)
        
        # Flatten sections back to lines
        compressed_lines = []
        for section in compressed_sections:
            compressed_lines.extend(section)
        
        return compressed_lines
    
    def _score_lines(self, lines: List[str]) -> List[Tuple[int, float]]:
        """
        Score lines by importance.
        
        Args:
            lines: List of lines
            
        Returns:
            List of (index, score) tuples
        """
        scored_lines = []
        
        for i, line in enumerate(lines):
            score = 0
            
            # Position
            if i == 0:  # First line
                score += 3
            elif i == len(lines) - 1:  # Last line
                score += 2
            
            # Content
            if re.search(r'\b(important|key|critical|essential|must|should)\b', line, re.I):
                score += 2
            
            # Headings or lists
            if re.match(r'^\s*[#*-]+', line):
                score += 1.5
            
            # Line length (prefer non-empty, non-verbose lines)
            if line.strip():
                words = len(line.split())
                if words <= 3:
                    score += 0.5
                elif words >= 25:
                    score -= 1
            
            scored_lines.append((i, score))
        
        return scored_lines

class DeepSeekOptimizer:
    """Token optimization specifically for DeepSeek models."""
    
    def __init__(self, base_optimizer: TokenOptimizer = None):
        """
        Initialize DeepSeek optimizer.
        
        Args:
            base_optimizer: Base token optimizer to extend
        """
        self.base_optimizer = base_optimizer or TokenOptimizer(optimization_level="balanced")
        
        # DeepSeek-specific patterns to optimize
        self.redundant_patterns = [
            # Verbose token generators
            (r'please think step-by-step', 'step-by-step'),
            (r'I would like you to', ''),
            (r'You are going to', 'Please'),
            (r'I want you to', 'Please'),
            
            # Hedging language
            (r'I believe that', ''),
            (r'In my opinion,', ''),
            (r'It seems like', ''),
            
            # Self-references
            (r'As an AI assistant,', ''),
            (r'As an AI language model,', ''),
            
            # Wordiness
            (r'in order to', 'to'),
            (r'due to the fact that', 'because'),
            (r'at this point in time', 'now'),
            (r'a large number of', 'many'),
            (r'the vast majority of', 'most'),
            
            # Redundant formality
            (r'Please be advised that', 'Note:'),
            (r'Thank you for your understanding', '')
        ]
    
    def optimize_prompt(self, prompt: str, max_tokens: int = 2048) -> Tuple[str, Dict[str, Any]]:
        """
        Optimize a prompt for DeepSeek models.
        
        Args:
            prompt: Original prompt
            max_tokens: Maximum desired tokens
            
        Returns:
            Tuple of (optimized prompt, optimization info)
        """
        # Apply base optimizations
        optimized = self.base_optimizer.optimize_prompt(prompt, model="deepseek")
        
        # If already small enough, apply only pattern replacements
        if self.base_optimizer._estimate_token_count(optimized) <= max_tokens:
            pattern_optimized = self._apply_pattern_optimizations(optimized)
            return pattern_optimized, {"optimization": "pattern_only"}
        
        # Otherwise apply DeepSeek-specific prompt format
        structured_prompt = self._create_structured_prompt(optimized)
        
        # Get final prompt from structured representation
        final_prompt = self._format_structured_prompt(structured_prompt)
        
        # Apply patterns to final prompt
        final_optimized = self._apply_pattern_optimizations(final_prompt)
        
        # Return
        return final_optimized, {
            "optimization": "full",
            "structure": structured_prompt
        }
    
    def _apply_pattern_optimizations(self, prompt: str) -> str:
        """
        Apply pattern-based optimizations.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Optimized prompt
        """
        optimized = prompt
        
        for pattern, replacement in self.redundant_patterns:
            optimized = re.sub(pattern, replacement, optimized, flags=re.IGNORECASE)
        
        return optimized
    
    def _create_structured_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Create a structured representation of the prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Structured prompt representation
        """
        # Extract components
        task_match = re.search(r'task:(.+?)(?=context:|examples:|constraints:|$)', prompt, re.DOTALL | re.IGNORECASE)
        context_match = re.search(r'context:(.+?)(?=task:|examples:|constraints:|$)', prompt, re.DOTALL | re.IGNORECASE)
        examples_match = re.search(r'examples?:(.+?)(?=task:|context:|constraints:|$)', prompt, re.DOTALL | re.IGNORECASE)
        constraints_match = re.search(r'constraints:(.+?)(?=task:|context:|examples:|$)', prompt, re.DOTALL | re.IGNORECASE)
        
        # Create structure
        structure = {
            "task": task_match.group(1).strip() if task_match else prompt,
            "context": context_match.group(1).strip() if context_match else None,
            "examples": examples_match.group(1).strip() if examples_match else None,
            "constraints": constraints_match.group(1).strip() if constraints_match else None
        }
        
        return structure
    
    def _format_structured_prompt(self, structure: Dict[str, Any]) -> str:
        """
        Format a structured prompt for DeepSeek.
        
        Args:
            structure: Structured prompt
            
        Returns:
            Formatted prompt
        """
        parts = []
        
        # Task (required)
        parts.append(f"TASK: {structure['task']}")
        
        # Context (optional)
        if structure["context"]:
            # Summarize if too long
            if self.base_optimizer._estimate_token_count(structure["context"]) > 500:
                context = self.base_optimizer._summarize_section(structure["context"], 
                                              self.base_optimizer.optimization_params["aggressive"])
            else:
                context = structure["context"]
                
            parts.append(f"CONTEXT: {context}")
        
        # Constraints (optional but important)
        if structure["constraints"]:
            parts.append(f"CONSTRAINTS: {structure['constraints']}")
        
        # Examples (optional)
        if structure["examples"]:
            # Take only one example if multiple
            examples = structure["examples"]
            example_parts = re.split(r'Example \d+:', examples)
            
            if len(example_parts) > 2:  # More than one example
                # Take first example
                example = example_parts[1]
                parts.append(f"EXAMPLE: {example.strip()}")
            else:
                parts.append(f"EXAMPLE: {examples}")
        
        return "\n\n".join(parts)
    
    def optimize_system_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize system configuration for DeepSeek.
        
        Args:
            config: Original configuration
            
        Returns:
            Optimized configuration
        """
        # Clone config to avoid modifying original
        optimized = config.copy()
        
        # Adjust temperature for efficiency
        if "temperature" in optimized:
            optimized["temperature"] = min(optimized["temperature"], 0.7)
        
        # Limit max_tokens for efficiency
        if "max_tokens" in optimized:
            optimized["max_tokens"] = min(optimized["max_tokens"], 2048)
        
        # Set top_p for better token efficiency
        optimized["top_p"] = 0.9
        
        # Add frequency and presence penalties
        optimized["frequency_penalty"] = 0.1
        optimized["presence_penalty"] = 0.1
        
        return optimized