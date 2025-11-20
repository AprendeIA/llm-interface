"""
AutoGen framework adapter for llm_interface.

This module provides integration with Microsoft's AutoGen framework,
enabling multi-agent conversations with provider abstraction.
"""

from .adapter import AutoGenAdapter
from .config_loader import AutoGenConfigLoader

__all__ = ["AutoGenAdapter", "AutoGenConfigLoader"]
