"""Core interfaces, configuration, and exceptions.

This module contains the fundamental building blocks of the LLM Interface:
- Configuration classes (LLMConfig, ProviderType)
- Core interfaces (LLMProvider)
- Exception hierarchy
"""

from .config import LLMConfig, ProviderType
from .interfaces import LLMProvider
from .exceptions import (
    LLMInterfaceError,
    ConfigurationError,
    ProviderNotFoundError,
    ProviderAlreadyExistsError,
    UnsupportedProviderError,
    ProviderValidationError,
    APIKeyError,
    ModelNotFoundError,
    EmbeddingsNotSupportedError,
    InvalidInputError,
    GraphExecutionError,
)

__all__ = [
    # Configuration
    "LLMConfig",
    "ProviderType",
    # Interfaces
    "LLMProvider",
    # Exceptions
    "LLMInterfaceError",
    "ConfigurationError",
    "ProviderNotFoundError",
    "ProviderAlreadyExistsError",
    "UnsupportedProviderError",
    "ProviderValidationError",
    "APIKeyError",
    "ModelNotFoundError",
    "EmbeddingsNotSupportedError",
    "InvalidInputError",
    "GraphExecutionError",
]
