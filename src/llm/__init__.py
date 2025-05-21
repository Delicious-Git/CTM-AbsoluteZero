"""
Language model adapters for CTM-AbsoluteZero.
"""

from .claude_adapter import ClaudeAdapter
from .deepseek_adapter import DeepSeekAdapter

__all__ = ["ClaudeAdapter", "DeepSeekAdapter"]