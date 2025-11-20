"""
Semantic Kernel framework adapter for llm_interface.

This module provides integration with Microsoft's Semantic Kernel framework,
enabling AI orchestration with plugins, skills, and planners.
"""

from .adapter import SemanticKernelAdapter
from .config_loader import SemanticKernelConfigLoader

__all__ = ["SemanticKernelAdapter", "SemanticKernelConfigLoader"]
