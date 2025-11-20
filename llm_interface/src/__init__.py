"""LLM Interface - A flexible library for handling multiple LLM providers with LangGraph"""

from .core.config import LLMConfig, ProviderType
from .manager import LLMManager
from .factory import LLMProviderFactory
from .framework.langchain.graph import LLMGraph
from .config_loader import ConfigLoader

__version__ = "1.0.0"
__author__ = "Your Name"

__all__ = [
    "LLMConfig",
    "ProviderType",
    "LLMManager",
    "LLMProviderFactory",
    "LLMGraph",
    "ConfigLoader",
]
