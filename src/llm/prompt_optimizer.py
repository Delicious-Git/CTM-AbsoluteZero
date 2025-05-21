"""
Prompt optimizer module for CTM-AbsoluteZero.
This module provides functions to optimize prompts by reducing redundancy and token usage.
"""

import re
import difflib
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import numpy as np
from collections import Counter

class PromptOptimizer:
    """
    Optimizer for reducing token usage in prompts by detecting and removing redundancy.
    """
    
    def __init__(
        self,
        mode: str = "balanced",
        min_chunk_size: int = 10,
        similarity_threshold: float = 0.8,
        preserve_keywords: Optional[List[str]] = None,
        debug: bool = False
    ):
        """
        Initialize the prompt optimizer.
        
        Args:
            mode: Optimization mode ("aggressive", "balanced", or "minimal")
            min_chunk_size: Minimum chunk size for redundancy detection
            similarity_threshold: Threshold for considering text chunks similar
            preserve_keywords: List of keywords to preserve in the prompt
            debug: Whether to print debug information
        """
        self.mode = mode
        self.min_chunk_size = min_chunk_size
        self.similarity_threshold = similarity_threshold
        self.preserve_keywords = preserve_keywords or []
        self.debug = debug
        
        # Set mode-specific parameters
        self._configure_mode()
    
    def _configure_mode(self):
        """Configure optimizer based on the selected mode."""
        if self.mode == "aggressive":
            self.similarity_threshold = 0.7
            self.remove_repeated_instructions = True
            self.compress_examples = True
            self.simplify_formatting = True
            self.remove_redundant_context = True
        elif self.mode == "balanced":
            self.similarity_threshold = 0.8
            self.remove_repeated_instructions = True
            self.compress_examples = True
            self.simplify_formatting = False
            self.remove_redundant_context = True
        elif self.mode == "minimal":
            self.similarity_threshold = 0.9
            self.remove_repeated_instructions = True
            self.compress_examples = False
            self.simplify_formatting = False
            self.remove_redundant_context = False
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def optimize(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """
        Optimize a prompt by detecting and removing redundancy.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Tuple of (optimized prompt, optimization metrics)
        """
        if self.debug:
            print(f"Optimizing prompt in {self.mode} mode...")
            print(f"Original prompt length: {len(prompt)}")
        
        # Store the original prompt
        original_prompt = prompt
        original_length = len(prompt)
        
        # Apply various optimization techniques
        optimizations = {}
        
        # 1. Remove repeated instructions
        if self.remove_repeated_instructions:
            prompt, metrics = self._remove_repeated_instructions(prompt)
            optimizations["repeated_instructions"] = metrics
        
        # 2. Compress examples
        if self.compress_examples:
            prompt, metrics = self._compress_examples(prompt)
            optimizations["compressed_examples"] = metrics
        
        # 3. Simplify formatting
        if self.simplify_formatting:
            prompt, metrics = self._simplify_formatting(prompt)
            optimizations["simplified_formatting"] = metrics
        
        # 4. Remove redundant context
        if self.remove_redundant_context:
            prompt, metrics = self._remove_redundant_context(prompt)
            optimizations["redundant_context"] = metrics
        
        # Calculate overall optimization metrics
        optimized_length = len(prompt)
        reduction = original_length - optimized_length
        reduction_percent = (reduction / original_length) * 100 if original_length > 0 else 0
        
        metrics = {
            "original_length": original_length,
            "optimized_length": optimized_length,
            "reduction": reduction,
            "reduction_percent": reduction_percent,
            "optimizations": optimizations,
            "mode": self.mode
        }
        
        if self.debug:
            print(f"Optimized prompt length: {optimized_length}")
            print(f"Reduction: {reduction} characters ({reduction_percent:.2f}%)")
        
        return prompt, metrics
    
    def _remove_repeated_instructions(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """
        Remove repeated instructions from the prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Tuple of (optimized prompt, optimization metrics)
        """
        original_length = len(prompt)
        
        # Find common instruction patterns
        instruction_patterns = [
            r"(Please|Kindly)?\s*(remember|note|be\s+sure|make\s+sure).*?(\.|$)",
            r"(Your\s+task\s+is\s+to|You\s+should|You\s+need\s+to|You\s+must).*?(\.|$)",
            r"(Important|Note|Remember):\s*.*?(\.|$)",
            r"(Follow\s+these\s+steps|Here\s+are\s+the\s+steps).*?(\.|$)"
        ]
        
        # Find all instruction matches
        instructions = []
        for pattern in instruction_patterns:
            matches = re.finditer(pattern, prompt, re.IGNORECASE | re.DOTALL)
            for match in matches:
                instructions.append((match.start(), match.end(), match.group(0)))
        
        # Sort instructions by start position
        instructions.sort()
        
        # Identify similar instructions
        to_remove = set()
        for i in range(len(instructions)):
            if i in to_remove:
                continue
                
            for j in range(i + 1, len(instructions)):
                if j in to_remove:
                    continue
                    
                # Get instruction texts
                _, _, text_i = instructions[i]
                _, _, text_j = instructions[j]
                
                # Check if they are similar
                similarity = difflib.SequenceMatcher(None, text_i.lower(), text_j.lower()).ratio()
                if similarity > self.similarity_threshold:
                    # Keep the first occurrence, mark the second for removal
                    to_remove.add(j)
        
        # Remove the redundant instructions (from end to start to maintain indices)
        removed_texts = []
        for idx in sorted(to_remove, reverse=True):
            start, end, text = instructions[idx]
            removed_texts.append(text)
            prompt = prompt[:start] + prompt[end:]
        
        # Calculate metrics
        optimized_length = len(prompt)
        reduction = original_length - optimized_length
        reduction_percent = (reduction / original_length) * 100 if original_length > 0 else 0
        
        metrics = {
            "removed_count": len(removed_texts),
            "removed_texts": removed_texts[:3],  # Just store the first few for debugging
            "reduction": reduction,
            "reduction_percent": reduction_percent
        }
        
        if self.debug and removed_texts:
            print(f"Removed {len(removed_texts)} repeated instructions")
            print(f"Example: {removed_texts[0]}")
        
        return prompt, metrics
    
    def _compress_examples(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """
        Compress examples in the prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Tuple of (optimized prompt, optimization metrics)
        """
        original_length = len(prompt)
        
        # Find example blocks
        example_patterns = [
            r"(Example|For example|Here's an example)[\s\:]+(.*?)(?=Example|\n\n|$)",
            r"```.*?```",
            r"<example>.*?</example>"
        ]
        
        examples = []
        for pattern in example_patterns:
            matches = re.finditer(pattern, prompt, re.IGNORECASE | re.DOTALL)
            for match in matches:
                examples.append((match.start(), match.end(), match.group(0)))
        
        # Sort examples by start position
        examples.sort()
        
        # Find similar examples
        to_compress = []
        for i in range(len(examples)):
            for j in range(i + 1, len(examples)):
                # Get example texts
                _, _, text_i = examples[i]
                _, _, text_j = examples[j]
                
                # Check if they are similar
                similarity = difflib.SequenceMatcher(None, text_i.lower(), text_j.lower()).ratio()
                if similarity > self.similarity_threshold:
                    # Keep both but mark for potential compression
                    to_compress.append((i, j, similarity))
        
        # Group similar examples
        example_groups = {}
        for i, j, similarity in to_compress:
            if i not in example_groups:
                example_groups[i] = [i]
            example_groups[i].append(j)
        
        # Compress similar examples
        for group_key, group_indices in example_groups.items():
            if len(group_indices) <= 1:
                continue
                
            # Get the first example as a reference
            start_ref, end_ref, text_ref = examples[group_key]
            
            # Replace the similar examples with a compressed version
            for idx in group_indices[1:]:
                start, end, text = examples[idx]
                
                # Create a compressed version that references the first example
                compressed_text = f"[Similar to previous example {group_key + 1}]"
                
                # Replace in the prompt
                prompt = prompt[:start] + compressed_text + prompt[end:]
                
                # Update the indices of subsequent examples
                length_diff = len(compressed_text) - (end - start)
                for k in range(idx + 1, len(examples)):
                    examples[k] = (examples[k][0] + length_diff, examples[k][1] + length_diff, examples[k][2])
        
        # Calculate metrics
        optimized_length = len(prompt)
        reduction = original_length - optimized_length
        reduction_percent = (reduction / original_length) * 100 if original_length > 0 else 0
        
        metrics = {
            "compressed_count": sum(len(group) - 1 for group in example_groups.values()),
            "groups_count": len(example_groups),
            "reduction": reduction,
            "reduction_percent": reduction_percent
        }
        
        if self.debug and example_groups:
            print(f"Compressed {metrics['compressed_count']} similar examples in {len(example_groups)} groups")
        
        return prompt, metrics
    
    def _simplify_formatting(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """
        Simplify formatting in the prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Tuple of (optimized prompt, optimization metrics)
        """
        original_length = len(prompt)
        
        # Replace multiple newlines with a single newline
        simplified = re.sub(r'\n{3,}', '\n\n', prompt)
        
        # Replace multiple spaces with a single space
        simplified = re.sub(r' {2,}', ' ', simplified)
        
        # Remove excessive punctuation
        simplified = re.sub(r'[.!?]{2,}', '.', simplified)
        
        # Calculate metrics
        optimized_length = len(simplified)
        reduction = original_length - optimized_length
        reduction_percent = (reduction / original_length) * 100 if original_length > 0 else 0
        
        metrics = {
            "reduction": reduction,
            "reduction_percent": reduction_percent
        }
        
        if self.debug:
            print(f"Simplified formatting: {reduction} characters ({reduction_percent:.2f}%)")
        
        return simplified, metrics
    
    def _remove_redundant_context(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """
        Remove redundant context from the prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Tuple of (optimized prompt, optimization metrics)
        """
        original_length = len(prompt)
        
        # Split prompt into chunks
        chunk_size = max(self.min_chunk_size, 20)
        chunks = self._split_into_chunks(prompt, chunk_size)
        
        # Calculate similarity between chunks
        chunk_pairs = []
        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                # Calculate similarity
                similarity = difflib.SequenceMatcher(None, chunks[i].lower(), chunks[j].lower()).ratio()
                if similarity > self.similarity_threshold:
                    chunk_pairs.append((i, j, similarity))
        
        # Sort by similarity (descending)
        chunk_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Remove similar chunks (keep the first occurrence)
        to_remove = set()
        for i, j, similarity in chunk_pairs:
            # Only remove if the chunk hasn't been marked for removal
            if j not in to_remove:
                # Check if it contains any preserve keywords
                if not any(keyword in chunks[j] for keyword in self.preserve_keywords):
                    to_remove.add(j)
        
        # Create a new prompt without the redundant chunks
        optimized_chunks = [chunks[i] for i in range(len(chunks)) if i not in to_remove]
        optimized = ''.join(optimized_chunks)
        
        # Calculate metrics
        optimized_length = len(optimized)
        reduction = original_length - optimized_length
        reduction_percent = (reduction / original_length) * 100 if original_length > 0 else 0
        
        metrics = {
            "removed_chunks": len(to_remove),
            "total_chunks": len(chunks),
            "reduction": reduction,
            "reduction_percent": reduction_percent
        }
        
        if self.debug:
            print(f"Removed {len(to_remove)} redundant chunks out of {len(chunks)}")
            print(f"Reduction from context removal: {reduction} characters ({reduction_percent:.2f}%)")
        
        return optimized, metrics
    
    def _split_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """
        Split text into chunks of approximately equal size.
        
        Args:
            text: Input text
            chunk_size: Approximate size of each chunk (in words)
            
        Returns:
            List of text chunks
        """
        # Split by natural boundaries if possible
        boundaries = []
        
        # Find paragraph boundaries
        for match in re.finditer(r'\n\s*\n', text):
            boundaries.append(match.start())
        
        # Find sentence boundaries if not enough paragraph boundaries
        if len(boundaries) < len(text) // (chunk_size * 5):  # Approximate word to char ratio
            for match in re.finditer(r'[.!?]\s+', text):
                if match.start() not in boundaries:
                    boundaries.append(match.start())
        
        # Sort boundaries
        boundaries.sort()
        
        # Add the end of the text
        boundaries.append(len(text))
        
        # Create chunks using the boundaries
        chunks = []
        start = 0
        current_size = 0
        for boundary in boundaries:
            boundary_text = text[start:boundary]
            boundary_words = len(boundary_text.split())
            
            if current_size + boundary_words <= chunk_size or current_size == 0:
                # Add to the current chunk
                current_size += boundary_words
            else:
                # Start a new chunk
                chunks.append(text[start:boundary])
                start = boundary
                current_size = boundary_words
        
        # Add the last chunk if not added
        if start < len(text):
            chunks.append(text[start:])
        
        return chunks


class TokenReducer:
    """
    Utility class for reducing token usage in text by applying various optimization techniques.
    """
    
    @staticmethod
    def reduce_lists(text: str) -> str:
        """
        Optimize lists to use less tokens.
        
        Args:
            text: Input text with lists
            
        Returns:
            Optimized text
        """
        # Replace numbered lists with more compact versions
        text = re.sub(r'(\d+)\.\s+', r'\1) ', text)
        
        # Replace bullet lists with more compact versions
        text = re.sub(r'•\s+', '- ', text)
        
        return text
    
    @staticmethod
    def reduce_code_examples(text: str, max_examples: int = 2) -> str:
        """
        Reduce the number of code examples.
        
        Args:
            text: Input text with code examples
            max_examples: Maximum number of examples to keep
            
        Returns:
            Optimized text
        """
        # Find all code blocks
        code_blocks = re.findall(r'```.*?```', text, re.DOTALL)
        
        # If we have more than max_examples, keep only the first max_examples
        if len(code_blocks) > max_examples:
            for i in range(max_examples, len(code_blocks)):
                text = text.replace(code_blocks[i], "[Code example omitted for brevity]")
        
        return text
    
    @staticmethod
    def simplify_repetitive_phrases(text: str) -> str:
        """
        Simplify repetitive phrases.
        
        Args:
            text: Input text
            
        Returns:
            Optimized text
        """
        # Common phrases that can be simplified
        replacements = {
            r'please note that ': '',
            r'keep in mind that ': '',
            r'it is important to remember that ': '',
            r'as mentioned earlier,? ': '',
            r'as discussed above,? ': '',
            r'as noted,? ': '',
            r'as previously mentioned,? ': '',
            r'in order to ': 'to ',
            r'for the purpose of ': 'for ',
            r'in the event that ': 'if ',
            r'in the case that ': 'if ',
            r'due to the fact that ': 'because ',
            r'for the reason that ': 'because ',
            r'on the grounds that ': 'because ',
            r'based on the fact that ': 'because ',
            r'in light of the fact that ': 'because ',
            r'in view of the fact that ': 'because ',
            r'in spite of the fact that ': 'although ',
            r'despite the fact that ': 'although ',
            r'with reference to ': 'about ',
            r'with regard to ': 'about ',
            r'with respect to ': 'about ',
            r'concerning the matter of ': 'about ',
            r'in the matter of ': 'about ',
            r'in relation to ': 'about ',
            r'in connection with ': 'about ',
            r'by means of ': 'by ',
            r'in the process of ': 'while ',
            r'in the course of ': 'during ',
            r'at this point in time ': 'now ',
            r'at the present time ': 'now ',
            r'in the near future ': 'soon ',
            r'on the occasion of ': 'when ',
        }
        
        # Apply replacements
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    @staticmethod
    def reduce_html_tags(text: str) -> str:
        """
        Simplify HTML tags to reduce token usage.
        
        Args:
            text: Input text with HTML tags
            
        Returns:
            Optimized text
        """
        # Replace div with span (shorter)
        text = re.sub(r'<div([^>]*)>', r'<span\1>', text)
        text = re.sub(r'</div>', '</span>', text)
        
        # Remove empty classes and ids
        text = re.sub(r' class=["\']\s*["\']', '', text)
        text = re.sub(r' id=["\']\s*["\']', '', text)
        
        # Remove style attributes
        text = re.sub(r' style=["\'][^"\']*["\']', '', text)
        
        return text

def optimize_prompt(
    prompt: str,
    mode: str = "balanced",
    preserve_keywords: Optional[List[str]] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Optimize a prompt by detecting and removing redundancy.
    
    Args:
        prompt: Input prompt
        mode: Optimization mode ("aggressive", "balanced", or "minimal")
        preserve_keywords: List of keywords to preserve in the prompt
        
    Returns:
        Tuple of (optimized prompt, optimization metrics)
    """
    optimizer = PromptOptimizer(mode=mode, preserve_keywords=preserve_keywords)
    return optimizer.optimize(prompt)

def optimize_message_history(
    messages: List[Dict[str, str]],
    mode: str = "balanced",
    max_history: int = 10
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """
    Optimize a message history by detecting and removing redundancy.
    
    Args:
        messages: List of message dictionaries with "role" and "content" keys
        mode: Optimization mode ("aggressive", "balanced", or "minimal")
        max_history: Maximum number of messages to keep
        
    Returns:
        Tuple of (optimized messages, optimization metrics)
    """
    # If history is too long, truncate it
    if len(messages) > max_history:
        messages = messages[-max_history:]
    
    # Create optimizer
    optimizer = PromptOptimizer(mode=mode)
    
    # Optimize each message
    optimized_messages = []
    metrics = {
        "original_total": 0,
        "optimized_total": 0,
        "message_metrics": []
    }
    
    for message in messages:
        # Skip if not a text message
        if "content" not in message or not isinstance(message["content"], str):
            optimized_messages.append(message)
            continue
        
        # Optimize the content
        original_content = message["content"]
        optimized_content, message_metrics = optimizer.optimize(original_content)
        
        # Create optimized message
        optimized_message = message.copy()
        optimized_message["content"] = optimized_content
        optimized_messages.append(optimized_message)
        
        # Update metrics
        metrics["original_total"] += len(original_content)
        metrics["optimized_total"] += len(optimized_content)
        metrics["message_metrics"].append(message_metrics)
    
    # Calculate overall metrics
    metrics["reduction"] = metrics["original_total"] - metrics["optimized_total"]
    metrics["reduction_percent"] = (metrics["reduction"] / metrics["original_total"]) * 100 if metrics["original_total"] > 0 else 0
    metrics["mode"] = mode
    
    return optimized_messages, metrics