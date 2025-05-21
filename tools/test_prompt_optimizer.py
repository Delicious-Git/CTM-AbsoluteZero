#!/usr/bin/env python3
"""
Test script for prompt optimizer.
This script demonstrates the token reduction capabilities of the prompt optimizer.
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Any

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from src.llm.prompt_optimizer import PromptOptimizer, optimize_prompt, optimize_message_history

# Sample prompts for testing
SAMPLE_PROMPTS = {
    "general": """
Please analyze the given text and provide insights on the main themes, arguments, and rhetorical devices used. 
Please be thorough in your analysis and make sure to cover all important aspects. Please note that your analysis should be detailed and comprehensive.
Remember to pay special attention to the use of language, tone, and structure.
Please note that it's important to consider the historical and cultural context of the text.
Please make sure to include specific examples from the text to support your analysis.
Remember that a good analysis goes beyond surface-level observations.
Please ensure that your response is well-structured, with clear sections for different aspects of your analysis.

The text is as follows:
[Long sample text would go here...]

Remember to be thorough in your analysis. Make sure to cover all important aspects of the text. 
Note that it's essential to back up your observations with specific examples from the text.
Please be sure to consider both the explicit and implicit messages in the text.
Remember to analyze the text in its historical and cultural context.
Please make sure your analysis is detailed and comprehensive.
""",

    "code": """
Write a Python function that implements a binary search algorithm. 
The function should be efficient and follow best practices. 
Make sure to include proper documentation and error handling.
Remember that binary search only works on sorted arrays.
Please ensure that your implementation handles edge cases correctly.
Please add detailed comments explaining how the algorithm works.
Please make sure to include type hints for parameters and return values.
Please ensure that your function includes proper error checking for invalid inputs.
Remember to handle empty arrays and arrays with a single element.
Please make sure your algorithm correctly handles the case where the target element is not found.

Here's an example of how your function might be called:
```python
result = binary_search([1, 2, 3, 4, 5], 3)
print(result)  # Should print the index of the element (2)
```

Please ensure that your implementation is efficient with a time complexity of O(log n).
Remember to check if the array is already sorted before performing the search.
Please make sure your function returns -1 or None if the element is not found.
Please note that the array should be sorted in ascending order.
Please remember to handle the case where the input array is None.
""",

    "conversation": [
        {"role": "system", "content": "You are a helpful AI assistant that helps users with coding questions. Please be thorough and detailed in your explanations. Remember to be clear and concise, but also comprehensive. Please ensure that your code examples are properly formatted and easy to understand. Please make sure to address all parts of the user's question. Remember that good explanations include examples."},
        {"role": "user", "content": "Can you explain how a binary search algorithm works? Please include a simple example."},
        {"role": "assistant", "content": "Binary search is an efficient algorithm for finding an item from a sorted list of items. It works by repeatedly dividing in half the portion of the list that could contain the item, until you've narrowed down the possible locations to just one. Here's how it works:\n\n1. Start with the middle element of the entire array.\n2. If the search key is equal to the middle element, the search is complete.\n3. If the search key is less than the middle element, narrow the search to the lower half of the array.\n4. If the search key is greater than the middle element, narrow the search to the upper half of the array.\n5. Repeat steps 1-4 until the search key is found or the subarray has size 0.\n\nHere's a simple example:\n\nSuppose we want to find the number 7 in the sorted array [1, 3, 5, 7, 9, 11, 13].\n\n1. The middle element is 7 (at index 3).\n2. 7 == 7, so we've found the element! Return index 3.\n\nIf we were looking for 9 instead:\n1. The middle element is 7 (at index 3).\n2. 9 > 7, so we search the upper half: [9, 11, 13].\n3. The middle element of this subarray is 11 (at index 1 in the subarray, which is index 5 in the original array).\n4. 9 < 11, so we search the lower half: [9].\n5. The middle (and only) element is 9, which equals our search key. Return index 4 (the index in the original array).\n\nBinary search has a time complexity of O(log n), making it much more efficient than linear search for large arrays."},
        {"role": "user", "content": "Thanks! Can you now show me how to implement this in Python? What are some edge cases I should consider?"},
        {"role": "assistant", "content": "Here's a Python implementation of binary search:\n\n```python\ndef binary_search(arr, target):\n    \"\"\"\n    Perform binary search on a sorted array.\n    \n    Args:\n        arr (list): A sorted list of elements\n        target: The element to find\n        \n    Returns:\n        int: The index of the target element if found, -1 otherwise\n    \"\"\"\n    if not arr:\n        return -1\n        \n    left, right = 0, len(arr) - 1\n    \n    while left <= right:\n        mid = (left + right) // 2\n        \n        # Check if target is present at mid\n        if arr[mid] == target:\n            return mid\n        \n        # If target is greater, ignore left half\n        elif arr[mid] < target:\n            left = mid + 1\n        \n        # If target is smaller, ignore right half\n        else:\n            right = mid - 1\n    \n    # Element is not present in the array\n    return -1\n```\n\nEdge cases to consider:\n\n1. Empty array: The function should return -1.\n2. Target not in array: The function should return -1.\n3. Duplicate elements: This implementation returns the index of one occurrence, not necessarily the first or last.\n4. Array with one element: The algorithm still works correctly.\n5. Target smaller than all elements or larger than all elements: The function will correctly return -1.\n\nImportant considerations:\n\n1. The array must be sorted for binary search to work correctly.\n2. Binary search requires random access to elements (constant-time indexing), so it works well with arrays but not with linked lists.\n3. For very large arrays, you might need to use `left + (right - left) // 2` instead of `(left + right) // 2` to avoid integer overflow in some languages (though this is not a concern in Python).\n4. If you need to find the insertion point for a target (even if it's not in the array), you can return `left` instead of `-1` at the end."},
        {"role": "user", "content": "What's the difference between binary search and linear search in terms of efficiency?"}
    ],

    "redundant": """
Welcome to our advanced system for processing your request. This system is designed to handle complex tasks efficiently.
Our system uses advanced algorithms to process your request. These algorithms are designed to be fast and accurate.
The system will now process your request. Please be patient while the system processes your request.
The processing may take a few moments. Please wait while the system completes the processing of your request.
Once the processing is complete, the system will display the results. The results will be shown on the screen.
If you have any questions about the results, please contact our support team. Our support team is available 24/7.
Thank you for using our advanced system for processing your request. We appreciate your patience and understanding.
Our system is constantly being improved to better serve your needs. We value your feedback on how we can improve the system.
Please remember that our system is designed to be user-friendly and efficient. We hope you find the system easy to use.
The system is capable of handling a wide variety of tasks. We are constantly adding new features to the system.
Thank you for choosing our system for your processing needs. We hope you find the results satisfactory.
"""
}

def test_optimizer(prompt_key: str, modes: List[str]):
    """
    Test the prompt optimizer on a sample prompt.
    
    Args:
        prompt_key: Key of the prompt to test
        modes: List of optimization modes to test
    """
    print(f"\n=== Testing prompt optimizer on '{prompt_key}' prompt ===\n")
    
    # Get the prompt
    prompt = SAMPLE_PROMPTS.get(prompt_key)
    if not prompt:
        print(f"Error: Prompt '{prompt_key}' not found.")
        return
    
    # Handle conversation separately
    if prompt_key == "conversation":
        for mode in modes:
            print(f"\n--- Mode: {mode} ---")
            
            # Optimize the conversation
            optimized_messages, metrics = optimize_message_history(prompt, mode=mode)
            
            # Print metrics
            print(f"Original length: {metrics['original_total']} characters")
            print(f"Optimized length: {metrics['optimized_total']} characters")
            print(f"Reduction: {metrics['reduction']} characters ({metrics['reduction_percent']:.2f}%)")
            
            # Print first few optimized messages
            print("\nSample of optimized messages:")
            for i, message in enumerate(optimized_messages[:2]):
                print(f"\nMessage {i+1} ({message['role']}):")
                print(message['content'][:200] + "..." if len(message['content']) > 200 else message['content'])
    else:
        # Print original prompt (truncated if too long)
        print(f"Original prompt ({len(prompt)} characters):")
        print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
        print()
        
        # Test each mode
        for mode in modes:
            print(f"\n--- Mode: {mode} ---")
            
            # Create optimizer with the specified mode
            optimizer = PromptOptimizer(mode=mode, debug=True)
            
            # Optimize the prompt
            optimized, metrics = optimizer.optimize(prompt)
            
            # Print metrics
            print(f"\nOptimization metrics:")
            print(f"Original length: {metrics['original_length']} characters")
            print(f"Optimized length: {metrics['optimized_length']} characters")
            print(f"Reduction: {metrics['reduction']} characters ({metrics['reduction_percent']:.2f}%)")
            
            # Print optimized prompt (truncated if too long)
            print(f"\nOptimized prompt ({len(optimized)} characters):")
            print(optimized[:200] + "..." if len(optimized) > 200 else optimized)
            
            # Print detailed metrics for each optimization technique
            print("\nDetailed metrics:")
            for technique, technique_metrics in metrics.get("optimizations", {}).items():
                if isinstance(technique_metrics, dict) and "reduction" in technique_metrics:
                    print(f"  {technique}: {technique_metrics['reduction']} characters ({technique_metrics.get('reduction_percent', 0):.2f}%)")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test prompt optimizer")
    parser.add_argument("--prompt", type=str, default="general", choices=SAMPLE_PROMPTS.keys(),
                        help="Prompt to test (general, code, conversation, redundant)")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "aggressive", "balanced", "minimal"],
                        help="Optimization mode to test")
    
    args = parser.parse_args()
    
    # Determine modes to test
    if args.mode == "all":
        modes = ["aggressive", "balanced", "minimal"]
    else:
        modes = [args.mode]
    
    # Test the optimizer
    test_optimizer(args.prompt, modes)

if __name__ == "__main__":
    main()